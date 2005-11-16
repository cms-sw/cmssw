/*
 * \file EBTestPulseClient.cc
 * 
 * $Date: 2005/11/16 08:36:44 $
 * $Revision: 1.13 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

EBTestPulseClient::EBTestPulseClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    h01_[i] = 0;
    h02_[i] = 0;

  }

}

EBTestPulseClient::~EBTestPulseClient(){

  this->unsubscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];

  }

}

void EBTestPulseClient::beginJob(const edm::EventSetup& c){

  cout << "EBTestPulseClient: beginJob" << endl;

  ievt_ = 0;

}

void EBTestPulseClient::beginRun(const edm::EventSetup& c){

  cout << "EBTestPulseClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    h01_[i] = 0;
    h02_[i] = 0;

  }

}

void EBTestPulseClient::endJob(void) {

  cout << "EBTestPulseClient: endJob, ievt = " << ievt_ << endl;

}

void EBTestPulseClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBTestPulseClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  MonTestPulseDat adc;
  map<EcalLogicID, MonTestPulseDat> dataset1;
  MonPulseShapeDat shape;
  map<EcalLogicID, MonPulseShapeDat> dataset2;

  cout << "Writing MonTestPulseDatObjects to database ..." << endl;

  float n_min_tot = 1000.;
  float n_min_bin = 30.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01;
    float mean01;
    float rms01;

    vector<int> sample;

    for ( int ie = 1; ie <= 85; ie++ ) { 
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = -1.;
        mean01 = -1.;
        rms01  = -1.;

        sample.clear();

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "G01 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;

          }

          adc.setADCMean(mean01);
          adc.setADCRMS(rms01);

          adc.setTaskStatus(1);

          if ( ie == 1 && ip == 1 ) {


            if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample.push_back(int(h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(1, i))));
              }
            }

            cout << "sample= " << flush;
            for ( unsigned int i = 0; i < sample.size(); i++ ) {
              cout << sample[i] << " " << flush;
            }
            cout << endl;

            shape.setSamples(sample);

          }

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
              dataset1[ecid] = adc;
              if ( ie == 1 && ip == 1 ) dataset2[ecid] = shape;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
      econn->insertDataSet(&dataset1, runiov, runtag );
      econn->insertDataSet(&dataset2, runiov, runtag );
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBTestPulseClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM*");

}

void EBTestPulseClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM*");

}

void EBTestPulseClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM*");

}

void EBTestPulseClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )  
    cout << "EBTestPulseClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM%02d G01", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM%02d G01", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        h02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

  }

}

void EBTestPulseClient::htmlOutput(int run, string htmlDir){

  cout << "Preparing EBTestPulseClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + "EBTestPulseClient.html").c_str());


  htmlFile.close();

}

