/*
 * \file EBIntegrityClient.cc
 * 
 * $Date: 2005/11/14 13:33:33 $
 * $Revision: 1.11 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>

EBIntegrityClient::EBIntegrityClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

  }

}

EBIntegrityClient::~EBIntegrityClient(){

  this->unsubscribe();

}

void EBIntegrityClient::beginJob(const edm::EventSetup& c){

  cout << "EBIntegrityClient: beginJob" << endl;

  ievt_ = 0;

}

void EBIntegrityClient::beginRun(const edm::EventSetup& c){

  cout << "EBIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  for ( int ism = 1; ism <= 36; ism++ ) {

    h00 = 0;
    h01[ism-1] = 0;
    h02[ism-1] = 0;
    h03[ism-1] = 0;
    h04[ism-1] = 0;

  }

}

void EBIntegrityClient::endJob(void) {

  cout << "EBIntegrityClient: endJob, ievt = " << ievt_ << endl;

}

void EBIntegrityClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBIntegrityClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  RunConsistencyDat cons;
  map<EcalLogicID, RunConsistencyDat> dataset;

  cout << "Writing RunConsistencyDatObjects to database ..." << endl;

  float num00;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num00 = -1.;

        if ( h00 ) {
          num00  = h00->GetBinContent(h00->GetBin(ie, ip));
        }

        num01 = num02 = num03 = num04 = -1.;

        bool update_channel = false;

        if ( h01[ism-1] ) {
          num01  = h01[ism-1]->GetBinContent(h01[ism-1]->GetBin(ie, ip));
          update_channel = true;
        }

        if ( h02[ism-1] ) {
          num02  = h02[ism-1]->GetBinContent(h02[ism-1]->GetBin(ie, ip));
          update_channel = true;
        }

        if ( h03[ism-1] ) {
          num03  = h03[ism-1]->GetBinContent(h03[ism-1]->GetBin(ie, ip));
          update_channel = true;
        }

        if ( h04[ism-1] ) {
          num04  = h04[ism-1]->GetBinContent(h04[ism-1]->GetBin(ie, ip));
          update_channel = true;
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "(" << ie << "," << ip << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << " " << num04 << endl;

          }

          cons.setExpectedEvents(0);
          cons.setProblemsInGain(int(num01));
          cons.setProblemsInId(int(num02));
          cons.setProblemsInSample(int(-999));
          cons.setProblemsInADC(int(-999));

          try {
            if ( econn ) ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
            dataset[ecid] = cons;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }

        }

      }
    }
  
  }

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
      econn->insertDataSet(&dataset, runiov, runtag );
      cout << "done." << endl; 
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBIntegrityClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/DCC size error");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/Gain/EI gain SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/ChId/EI ChId SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/TTId/EI TTId SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");

}

void EBIntegrityClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/DCC size error");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/Gain/EI gain SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/ChId/EI ChId SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/TTId/EI TTId SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");

}

void EBIntegrityClient::unsubscribe(void){
  
  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/DCC size error");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/Gain/EI gain SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/ChId/EI ChId SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/TTId/EI TTId SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");

}

void EBIntegrityClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )
  cout << "EBIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];
  
  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  h00 = 0;
  sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/DCC size error");
  me = mui_->get(histo);
  if ( me ) {
    cout << "Found '" << histo << "'" << endl;
    ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    if ( ob ) h00 = dynamic_cast<TH2D*> (ob->operator->());
  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    h01[ism-1] = 0;
    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/Gain/EI gain SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) h01[ism-1] = dynamic_cast<TH2D*> (ob->operator->());
    }

    h02[ism-1] = 0;
    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/ChId/EI ChId SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) h02[ism-1] = dynamic_cast<TH2D*> (ob->operator->());
    }

    h03[ism-1] = 0;
    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/TTId/EI TTId SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) h03[ism-1] = dynamic_cast<TH2D*> (ob->operator->());
    }

    h04[ism-1] = 0;
    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) h04[ism-1] = dynamic_cast<TH2D*> (ob->operator->());
    }

  }

}

void EBIntegrityClient::htmlOutput(int run, string htmlDir){

  cout << "Preparing EBIntegrityClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + "EBIntegrityClient.html").c_str());


  htmlFile.close();

}

