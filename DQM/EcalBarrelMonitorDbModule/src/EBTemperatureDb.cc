/*
 * \file EBTemperatureDb.cc
 * 
 * $Date: 2006/06/06 14:51:19 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorDbModule/interface/EBTemperatureDb.h>

EBTemperatureDb::EBTemperatureDb(const ParameterSet& ps, DaqMonitorBEInterface* dbe){

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBTemperatureDb");

    meTemp_ = dbe->book2D("TEMP", "TEMP", 17, -0.5, 16.5, 10, -0.5, 9.5);

  }

}

EBTemperatureDb::~EBTemperatureDb(){

}

void EBTemperatureDb::beginJob(const EventSetup& c){

  ievt_ = 0;
    
}

void EBTemperatureDb::endJob(){

  cout << "EBTemperatureDb: analyzed " << ievt_ << " events" << endl;

}

void EBTemperatureDb::analyze(const Event& e, const EventSetup& c, DaqMonitorBEInterface* dbe, ISessionProxy* session){

  ievt_++;

//  char* temp_sql = "select CHANNELVIEW.ID1, CHANNELVIEW.ID2, cast(MON_TR_CAPS_DAT.CAPS_TEMP as number) from CHANNELVIEW, MON_TR_CAPS_DAT where MON_TR_CAPS_DAT.IOV_ID = (SELECT MAX(IOV_ID) from MON_TR_CAPS_DAT) and CHANNELVIEW.LOGIC_ID=MON_TR_CAPS_DAT.LOGIC_ID order by ID1, ID2";

  if ( session )  {

    // Query stuff
    session->transaction().start(true);

    ISchema& schema = session->nominalSchema();

    IQuery* query = schema.newQuery();

    query->addToOutputList("CHANNELVIEW.ID1", "X");
    query->addToOutputList("CHANNELVIEW.ID2", "Y");
    query->addToOutputList("cast(MON_TR_CAPS_DAT.CAPS_TEMP as number)", "Z");

    query->addToTableList("CHANNELVIEW");
    query->addToTableList("MON_TR_CAPS_DAT");

    AttributeList bindVariableList;

    query->setCondition("MON_TR_CAPS_DAT.IOV_ID = (SELECT max(IOV_ID) from MON_TR_CAPS_DAT) and CHANNELVIEW.LOGIC_ID = MON_TR_CAPS_DAT.LOGIC_ID", bindVariableList);

    query->addToOrderList("ID1");
    query->addToOrderList("ID2");

    ICursor& cursor = query->execute();

    // pause the shipping of monitoring elements
    if ( dbe ) dbe->lock();

    int j = 0;

    while ( cursor.next() && j < 170 ) {

      const AttributeList& row = cursor.currentRow();

//      cout << row["X"].data<int>() << " "
//           << row["Y"].data<int>() << " "
//           << row["Z"].data<float>() << " " << endl;

      int chan = row["Y"].data<int>();
      float temp = row["Z"].data<float>();

      meTemp_->Fill(((chan-1)/10), ((chan-1)%10), temp);

//      cout << chan << " " << ((chan-1)/10)
//                   << " " << ((chan-1)%10)
//                   << " " << temp << endl;

      j++;

    }

    // resume the shipping of monitoring elements
    if ( dbe ) dbe->unlock();

    delete query;

    session->transaction().commit();

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

  c1->SaveAs((htmlDir + "/" + "meTemp.png").c_str());

  delete c1;

}

