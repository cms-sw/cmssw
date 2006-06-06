/*
 * \file EBTemperatureDb.cc
 * 
 * $Date: 2005/11/24 10:55:51 $
 * $Revision: 1.29 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorDbModule/interface/EBTemperatureDb.h>

EBTemperatureDb::EBTemperatureDb(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile.open("EBTemperatureDb.log");

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTemperatureDb");

    meTemp_ = dbe->book2D("TEMP", "TEMP", 17, -0.5, 16.5, 10, -0.5, 9.5);

  }

}

EBTemperatureDb::~EBTemperatureDb(){

//  logFile.close();

}

void EBTemperatureDb::beginJob(const edm::EventSetup& c){

  ievt_ = 0;
    
}

void EBTemperatureDb::endJob(){

  cout << "EBTemperatureDb: analyzed " << ievt_ << " events" << endl;

}

void EBTemperatureDb::analyze(const edm::Event& e, const edm::EventSetup& c, DaqMonitorBEInterface* dbe, ISessionProxy* isp){

  ievt_++;

//  char* temp_sql = "select CHANNELVIEW.ID1, CHANNELVIEW.ID2, cast(MON_TR_CAPS_DAT.CAPS_TEMP as number) from CHANNELVIEW, MON_TR_CAPS_DAT where MON_TR_CAPS_DAT.IOV_ID = (SELECT MAX(IOV_ID) from MON_TR_CAPS_DAT) and CHANNELVIEW.LOGIC_ID=MON_TR_CAPS_DAT.LOGIC_ID order by ID1, ID2";

  if ( isp )  {

//    TSQLResult* res = db->Query(temp_sql);

    // Query stuff
    isp->transaction().start();

    ITable& table = isp->nominalSchema().tableHandle("MON_TR_CAPS_DAT");
    IQuery* query = table.newQuery();
    query->addToOutputList("CHANNELVIEW.ID1");
    query->addToOutputList("CHANNELVIEW.ID2");

    AttributeList bindVariableList;
    bindVariableList.extend ("idvalue", typeid (int));
    bindVariableList["idvalue"].data<int>() = 1;

    query->setCondition ("ID == :idvalue", bindVariableList);

    ICursor& cursor = query->execute();

    while (cursor.next()) {

      const AttributeList& row = cursor.currentRow();

      cerr << "Name:" << row["Name"].data <string>() << endl;

    }

    delete query;

    isp->transaction().commit();

    float temp = 0;
    int chan = 0;
    int j = 0;

    // pause the shipping of monitoring elements
    if ( dbe ) dbe->lock();

    do {

//      TSQLRow* row = res->Next();

//      for ( int i = 0; i < res->GetFieldCount(); i++ ) {

//        printf(" %*.*s \n", row->GetFieldLength(i), row->GetFieldLength(i), row->GetField(i));

//        if( i==1 ) chan = atoi( row->GetField(i) );
//        if( i==2 ) temp = atof( row->GetField(i) );

//      }
//      cout << endl;

      meTemp_->Fill(((chan-1)/10), ((chan-1)%10), temp);

//      cout << chan << " " << ((chan-1)/10) << " " << ((chan-1)%10) << " " << temp << endl;

//      delete row;

      j++;

    } while ( j < 170 );

    // resume the shipping of monitoring elements
    if ( dbe ) dbe->unlock();

//    delete res;

  }

}

void EBTemperatureDb::htmlOutput(string htmlDir){

  gStyle->SetOptStat(0);
  gStyle->SetOptFit();
  gStyle->SetPalette(1,0);
  TCanvas* c1 = new TCanvas("c1","The Temperatures",200,10,600,400);
  c1->SetGrid();

  c1->cd();
  MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (meTemp_);
  if ( ob ) {
    TH2F* h2d = dynamic_cast<TH2F*> (ob->operator->());
    if ( h2d ) h2d->Draw("colz");
  }
  c1->Update();

  c1->SaveAs((htmlDir + "/" + "meTemp.jpg").c_str());

  delete c1;

}

