/*
 * \file EBTemperatureDb.cc
 * 
 * $Date: 2006/06/07 07:34:31 $
 * $Revision: 1.4 $
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

  if ( session )  {

    // Query stuff
    session->transaction().start(true);

    ISchema& schema = session->nominalSchema();

    IQuery* query = schema.newQuery();

    query->addToTableList("CHANNELVIEW");
    query->addToTableList("MON_TR_CAPS_DAT");

    query->addToOutputList("cast( floor((CHANNELVIEW.ID2-1) / 10) as float )", "X");
    query->addToOutputList("cast( mod((CHANNELVIEW.ID2-1) , 10) as float )", "Y");
    query->addToOutputList("cast( MON_TR_CAPS_DAT.CAPS_TEMP as float )", "Z");

    query->setCondition("MON_TR_CAPS_DAT.IOV_ID = (select max(IOV_ID) from MON_TR_CAPS_DAT) and CHANNELVIEW.LOGIC_ID = MON_TR_CAPS_DAT.LOGIC_ID", AttributeList());

    query->addToOrderList("ID2");

    ICursor& cursor = query->execute();

    // pause the shipping of monitoring elements
    if ( dbe ) dbe->lock();

    int j = 0;

    while ( cursor.next() && j < 170 ) {

      const AttributeList& row = cursor.currentRow();

      float xchan = row["X"].data<float>();
      float ychan = row["Y"].data<float>();
      float temp = row["Z"].data<float>();

      meTemp_->Fill(xchan, ychan, temp);

//      cout << xchan << " " << ychan << " " << temp << endl;

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

