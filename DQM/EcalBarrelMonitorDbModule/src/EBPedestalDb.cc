/*
 * \file EBPedestalDb.cc
 * 
 * $Date: 2006/06/07 10:42:06 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorDbModule/interface/EBPedestalDb.h>

EBPedestalDb::EBPedestalDb(const ParameterSet& ps, DaqMonitorBEInterface* dbe){

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPedestalDb");

    mePed01_  = dbe->book1D("ped_mon01", "PED MON G01", 100, 0., 300.);
    mePed06_  = dbe->book1D("ped_mon06", "PED MON G06", 100, 0., 300.);
    mePed12_  = dbe->book1D("ped_mon12", "PED MON G12", 100, 0., 300.);
    meRms01_  = dbe->book1D("rms_mon01", "RMS MON G01", 100, 0., 1.);
    meRms06_  = dbe->book1D("rms_mon06", "RMS MON G06", 100, 0., 1.);
    meRms12_  = dbe->book1D("rms_mon12", "RMS MON G12", 100, 0., 1.);

  }

}

EBPedestalDb::~EBPedestalDb(){

}

void EBPedestalDb::beginJob(const EventSetup& c){

  ievt_ = 0;
    
}

void EBPedestalDb::endJob(){

  cout << "EBPedestalDb: analyzed " << ievt_ << " events" << endl;

}

void EBPedestalDb::analyze(const Event& e, const EventSetup& c, DaqMonitorBEInterface* dbe, ISessionProxy* session){

  ievt_++;

  if ( session )  {

    // Query stuff
    session->transaction().start(true);

    ISchema& schema = session->nominalSchema();

    IQuery* query = schema.newQuery();

    query->addToTableList("MON_PEDESTALS_DAT");

    query->addToOutputList("cast( MON_PEDESTALS_DAT.PED_MEAN_G1 as float )", "X");

    query->setCondition("MON_PEDESTALS_DAT.LOGIC_ID != 0", AttributeList());

    query->addToOrderList("LOGIC_ID");

    ICursor& cursor = query->execute();

    // pause the shipping of monitoring elements
    if ( dbe ) dbe->lock();

    while ( cursor.next() ) {

      const AttributeList& row = cursor.currentRow();

      float xmean01 = row["X"].data<float>();

      mePed01_->Fill(xmean01);

      cout << xmean01 << endl;

    }

    // resume the shipping of monitoring elements
    if ( dbe ) dbe->unlock();

    delete query;

    session->transaction().commit();

  }

}

void EBPedestalDb::htmlOutput(string htmlDir){

  gStyle->SetOptStat(0);
  gStyle->SetOptFit();
  gStyle->SetPalette(1,0);
  TCanvas* c1 = new TCanvas("c1","The Pedestals",200,10,600,400);
  c1->SetGrid();

  c1->cd();
  MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (mePed01_);
  if ( ob ) {
    TH2F* h2d = dynamic_cast<TH2F*> (ob->operator->());
    if ( h2d ) h2d->Draw("colz");
  }
  c1->Update();

  c1->SaveAs((htmlDir + "/" + "mePed01.png").c_str());

  delete c1;

}

