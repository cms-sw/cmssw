/*
 * \file EBTestPulseClient.cc
 * 
 * $Date: 2005/12/28 21:10:50 $
 * $Revision: 1.49 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

EBTestPulseClient::EBTestPulseClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  for ( int ism = 1; ism <= 36; ism++ ) {

    ha01_[ism-1] = 0;
    ha02_[ism-1] = 0;
    ha03_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;

    he01_[ism-1] = 0;
    he02_[ism-1] = 0;
    he03_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g01_[ism-1] = 0;
    g02_[ism-1] = 0;
    g03_[ism-1] = 0;

    a01_[ism-1] = 0;
    a02_[ism-1] = 0;
    a03_[ism-1] = 0;

  }

  amplitudeThreshold_ = 400.0;
  RMSThreshold_ = 300.0;
  threshold_on_AmplitudeErrorsNumber_ = 0.02;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBTestPulseClient::~EBTestPulseClient(){

  this->cleanup(); 

}

void EBTestPulseClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBTestPulseClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTestPulseClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBTestPulseClient: beginRun" << endl;

  jevt_ = 0;

  this->setup(); 

  this->subscribe();

}

void EBTestPulseClient::endJob(void) {

  if ( verbose_ ) cout << "EBTestPulseClient: endJob, ievt = " << ievt_ << endl;

}

void EBTestPulseClient::endRun(void) {

  if ( verbose_ ) cout << "EBTestPulseClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup(); 

}

void EBTestPulseClient::setup(void) {

  Char_t histo[50];

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    sprintf(histo, "EBPT test pulse quality G01 SM%02d", ism);
    g01_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g02_[ism-1] ) delete g02_[ism-1];
    sprintf(histo, "EBPT test pulse quality G06 SM%02d", ism);
    g02_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g03_[ism-1] ) delete g03_[ism-1];
    sprintf(histo, "EBPT test pulse quality G12 SM%02d", ism);
    g03_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( a01_[ism-1] ) delete a01_[ism-1];
    sprintf(histo, "EBPT test pulse amplitude G01 SM%02d", ism);
    a01_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( a02_[ism-1] ) delete a02_[ism-1];
    sprintf(histo, "EBPT test pulse amplitude G06 SM%02d", ism);
    a02_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( a03_[ism-1] ) delete a03_[ism-1];
    sprintf(histo, "EBPT test pulse amplitude G12 SM%02d", ism);
    a03_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g01_[ism-1]->Reset();
    g02_[ism-1]->Reset();
    g03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);
        g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), 2.);
        g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

      }
    }

    a01_[ism-1]->Reset();
    a02_[ism-1]->Reset();
    a03_[ism-1]->Reset();

  }

}

void EBTestPulseClient::cleanup(void) {

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( ha01_[ism-1] ) delete ha01_[ism-1];
    ha01_[ism-1] = 0;
    if ( ha02_[ism-1] ) delete ha02_[ism-1];
    ha02_[ism-1] = 0;
    if ( ha03_[ism-1] ) delete ha03_[ism-1];
    ha03_[ism-1] = 0;

    if ( hs01_[ism-1] ) delete hs01_[ism-1];
    hs01_[ism-1] = 0;
    if ( hs02_[ism-1] ) delete hs02_[ism-1];
    hs02_[ism-1] = 0;
    if ( hs03_[ism-1] ) delete hs03_[ism-1];
    hs03_[ism-1] = 0;

    if ( he01_[ism-1] ) delete he01_[ism-1];
    he01_[ism-1] = 0;
    if ( he02_[ism-1] ) delete he02_[ism-1];
    he02_[ism-1] = 0;
    if ( he03_[ism-1] ) delete he03_[ism-1];
    he03_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    g01_[ism-1] = 0;
    if ( g02_[ism-1] ) delete g02_[ism-1];
    g02_[ism-1] = 0;
    if ( g03_[ism-1] ) delete g03_[ism-1];
    g03_[ism-1] = 0;

    if ( a01_[ism-1] ) delete a01_[ism-1];
    a01_[ism-1] = 0;
    if ( a02_[ism-1] ) delete a02_[ism-1];
    a02_[ism-1] = 0;
    if ( a03_[ism-1] ) delete a03_[ism-1];
    a03_[ism-1] = 0;

  }

}

void EBTestPulseClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  EcalLogicID ecid;
  MonTestPulseDat adc;
  map<EcalLogicID, MonTestPulseDat> dataset1;
  MonPulseShapeDat shape;
  map<EcalLogicID, MonPulseShapeDat> dataset2;

  cout << "Writing MonTestPulseDatObjects to database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 10.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03;
    float mean01, mean02, mean03;
    float rms01, rms02, rms03;

    vector<int> sample01, sample02, sample03;

    for ( int ie = 1; ie <= 85; ie++ ) { 
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = -1.;
        mean01 = mean02 = mean03 = -1.;
        rms01  = rms02  = rms03  = -1.;

        sample01.clear();
        sample02.clear();
        sample03.clear();

        bool update_channel = false;

        float numEventsinCry[3] = {0., 0., 0.};

        if ( ha01_[ism-1] ) numEventsinCry[0] = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
        if ( ha02_[ism-1] ) numEventsinCry[1] = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
        if ( ha03_[ism-1] ) numEventsinCry[2] = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));

        if ( ha01_[ism-1] && ha01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = ha01_[ism-1]->GetBinContent(ha01_[ism-1]->GetBin(ie, ip));
            rms01  = ha01_[ism-1]->GetBinError(ha01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( ha02_[ism-1] && ha02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = ha02_[ism-1]->GetBinContent(ha02_[ism-1]->GetBin(ie, ip));
            rms02  = ha02_[ism-1]->GetBinError(ha02_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( ha03_[ism-1] && ha03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = ha03_[ism-1]->GetBinContent(ha03_[ism-1]->GetBin(ie, ip));
            rms03  = ha03_[ism-1]->GetBinError(ha03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;
            cout << "G01 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;
            cout << "G06 (" << ie << "," << ip << ") " << num02 << " " << mean02 << " " << rms02 << endl;
            cout << "G12 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;

          }

          adc.setADCMean(mean01);
          adc.setADCRMS(rms01);

          // adc.setADCMean(mean02);
          // adc.setADCRMS(rms02);

          // adc.setADCMean(mean03);
          // adc.setADCRMS(rms03);

          if ( g01_[ism-1]->GetBinContent(g01_[ism-1]->GetBin(ie, ip)) == 1. &&
               g02_[ism-1]->GetBinContent(g02_[ism-1]->GetBin(ie, ip)) == 1. &&
               g03_[ism-1]->GetBinContent(g03_[ism-1]->GetBin(ie, ip)) == 1. ) {
            adc.setTaskStatus(true);
          } else {
            adc.setTaskStatus(false);
          }

          if ( ie == 1 && ip == 1 ) {

            if ( hs01_[ism-1] && hs01_[ism-1]->GetEntries() >= n_min_tot ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample01.push_back(int(hs01_[ism-1]->GetBinContent(hs01_[ism-1]->GetBin(1, i))));
              }
            }

            if ( hs02_[ism-1] && hs02_[ism-1]->GetEntries() >= n_min_tot ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample02.push_back(int(hs02_[ism-1]->GetBinContent(hs02_[ism-1]->GetBin(1, i))));
              }
            }

            if ( hs03_[ism-1] && hs03_[ism-1]->GetEntries() >= n_min_tot ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample03.push_back(int(hs03_[ism-1]->GetBinContent(hs03_[ism-1]->GetBin(1, i))));
              }
            }

            cout << "sample01 = " << flush;
            for ( unsigned int i = 0; i < sample01.size(); i++ ) {
              cout << sample01[i] << " " << flush;
            }
            cout << endl;

            cout << "sample02 = " << flush;
            for ( unsigned int i = 0; i < sample02.size(); i++ ) {
              cout << sample02[i] << " " << flush;
            }
            cout << endl;

            cout << "sample03 = " << flush;
            for ( unsigned int i = 0; i < sample03.size(); i++ ) {
              cout << sample03[i] << " " << flush;
            }
            cout << endl;

            shape.setSamples(sample01);
//            shape.setSamples(sample02);
//            shape.setSamples(sample03);

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

  }

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
      econn->insertDataSet(&dataset1, runiov, runtag);
      econn->insertDataSet(&dataset2, runiov, runtag);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBTestPulseClient::subscribe(void){

  if ( verbose_ ) cout << "EBTestPulseClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude error SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude error SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude error SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBTestPulseClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBTT amplitude SM%02d G01", ism);
      me_ha01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM%02d G01", ism);
      mui_->add(me_ha01_[ism-1], histo);

      sprintf(histo, "EBTT amplitude SM%02d G06", ism);
      me_ha02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude SM%02d G06", ism);
      mui_->add(me_ha02_[ism-1], histo);

      sprintf(histo, "EBTT amplitude SM%02d G12", ism);
      me_ha03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude SM%02d G12", ism);
      mui_->add(me_ha03_[ism-1], histo);

      sprintf(histo, "EBTT shape SM%02d G01", ism);
      me_hs01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM%02d G01", ism);
      mui_->add(me_hs01_[ism-1], histo);

      sprintf(histo, "EBTT shape SM%02d G06", ism);
      me_hs02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain06/EBTT shape SM%02d G06", ism);
      mui_->add(me_hs02_[ism-1], histo);

      sprintf(histo, "EBTT shape SM%02d G12", ism);
      me_hs03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain12/EBTT shape SM%02d G12", ism);
      mui_->add(me_hs03_[ism-1], histo);

      sprintf(histo, "EBTT amplitude error SM%02d G01", ism);
      me_he01_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude error SM%02d G01", ism);
      mui_->add(me_he01_[ism-1], histo);

      sprintf(histo, "EBTT amplitude error SM%02d G06", ism);
      me_he02_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude error SM%02d G06", ism);
      mui_->add(me_he02_[ism-1], histo);

      sprintf(histo, "EBTT amplitude error SM%02d G12", ism);
      me_he03_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude error SM%02d G12", ism);
      mui_->add(me_he03_[ism-1], histo);

    }

  }

}

void EBTestPulseClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude error SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT shape SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude error SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT shape SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude error SM*");

}

void EBTestPulseClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBTestPulseClient: unsubscribe" << endl;

  if ( collateSources_ ) {
 
    if ( verbose_ ) cout << "EBTestPulseClient: uncollate" << endl;

    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( bei ) {
      Char_t histo[80];

      for ( int ism = 1; ism <= 36; ism++ ) {

        sprintf(histo, "EBTT amplitude SM%02d G01", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain01");
        bei->removeElement(histo);
  
        sprintf(histo, "EBTT amplitude SM%02d G06", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain06");
        bei->removeElement(histo);

        sprintf(histo, "EBTT amplitude SM%02d G12", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain12");
        bei->removeElement(histo);

        sprintf(histo, "EBTT shape SM%02d G01", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain01");
        bei->removeElement(histo);

        sprintf(histo, "EBTT shape SM%02d G06", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain06");
        bei->removeElement(histo);

        sprintf(histo, "EBTT shape SM%02d G12", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain12");
        bei->removeElement(histo);

        sprintf(histo, "EBTT amplitude error SM%02d G01", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain01");
        bei->removeElement(histo);

        sprintf(histo, "EBTT amplitude error SM%02d G06", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain06");
        bei->removeElement(histo);

        sprintf(histo, "EBTT amplitude error SM%02d G12", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBTestPulseTask/Gain12");
        bei->removeElement(histo);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude error SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT shape SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude error SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT shape SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude error SM*");

}

void EBTestPulseClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) { 
    if ( verbose_ ) cout << "EBTestPulseClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01/EBTT amplitude SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( ha01_[ism-1] ) delete ha01_[ism-1];
        sprintf(histo, "ME EBTT amplitude SM%02d G01", ism);
        ha01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        ha01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06/EBTT amplitude SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( ha02_[ism-1] ) delete ha02_[ism-1];
        sprintf(histo, "ME EBTT amplitude SM%02d G06", ism);
        ha02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        ha02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12/EBTT amplitude SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( ha03_[ism-1] ) delete ha03_[ism-1];
        sprintf(histo, "ME EBTT amplitude SM%02d G12", ism);
        ha03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        ha03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01/EBTT shape SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( hs01_[ism-1] ) delete hs01_[ism-1];
        sprintf(histo, "ME EBTT shape SM%02d G01", ism);
        hs01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        hs01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06/EBTT shape SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTT shape SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( hs02_[ism-1] ) delete hs02_[ism-1];
        sprintf(histo, "ME EBTT shape SM%02d G06", ism);
        hs02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        hs02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12/EBTT shape SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTT shape SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( hs03_[ism-1] ) delete hs03_[ism-1];
        sprintf(histo, "ME EBTT shape SM%02d G12", ism);
        hs03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        hs03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01/EBTT amplitude error SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude error SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( he01_[ism-1] ) delete he01_[ism-1];
        sprintf(histo, "ME EBTT amplitude error SM%02d G01", ism);
        he01_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        he01_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06/EBTT amplitude error SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude error SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( he02_[ism-1] ) delete he02_[ism-1];
        sprintf(histo, "ME EBTT amplitude error SM%02d G06", ism);
        he02_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        he02_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12/EBTT amplitude error SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude error SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( he03_[ism-1] ) delete he03_[ism-1];
        sprintf(histo, "ME EBTT amplitude error SM%02d G12", ism);
        he03_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        he03_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02, num03;
    float mean01, mean02, mean03;
    float rms01, rms02, rms03;

    g01_[ism-1]->Reset();
    g02_[ism-1]->Reset();
    g03_[ism-1]->Reset();

    a01_[ism-1]->Reset();
    a02_[ism-1]->Reset();
    a03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = -1.;
        mean01 = mean02 = mean03 = -1.;
        rms01  = rms02  = rms03  = -1.;

        g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);
        g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), 2.);
        g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

        bool update_channel = false;

        float numEventsinCry[3] = {0., 0., 0.};

        if ( ha01_[ism-1] ) numEventsinCry[0] = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
        if ( ha02_[ism-1] ) numEventsinCry[1] = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
        if ( ha03_[ism-1] ) numEventsinCry[2] = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));

        if ( ha01_[ism-1] && ha01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = ha01_[ism-1]->GetBinContent(ha01_[ism-1]->GetBin(ie, ip));
            rms01  = ha01_[ism-1]->GetBinError(ha01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( ha02_[ism-1] && ha02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = ha02_[ism-1]->GetBinContent(ha02_[ism-1]->GetBin(ie, ip));
            rms02  = ha02_[ism-1]->GetBinError(ha02_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( ha03_[ism-1] && ha03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = ha03_[ism-1]->GetBinContent(ha03_[ism-1]->GetBin(ie, ip));
            rms03  = ha03_[ism-1]->GetBinError(ha03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          float val;

          val = 1.;
          if ( mean01 < amplitudeThreshold_ )
            val = 0.;
          if ( rms01 > RMSThreshold_ )
            val = 0.;
          if ( he01_[ism-1] ) {
            float errorRate = he01_[ism-1]->GetBinContent(he01_[ism-1]->GetBin(ie, ip)) / numEventsinCry[0];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);

          a01_[ism-1]->SetBinContent(ip+20*(ie-1), mean01);
          a01_[ism-1]->SetBinError(ip+20*(ie-1), rms01);

          val = 1.;
          if ( mean02 < amplitudeThreshold_ )
            val = 0.;
          if ( rms02 > RMSThreshold_ )
            val = 0.;
          if ( he02_[ism-1] ) {
            float errorRate = he02_[ism-1]->GetBinContent(he02_[ism-1]->GetBin(ie, ip)) / numEventsinCry[1];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), val);

          a02_[ism-1]->SetBinContent(ip+20*(ie-1), mean02);
          a02_[ism-1]->SetBinError(ip+20*(ie-1), rms02);

          val = 1.;
          if ( mean03 < amplitudeThreshold_ )
            val = 0.;
          if ( rms03 > RMSThreshold_ )
            val = 0.;
          if ( he03_[ism-1] ) {
            float errorRate = he03_[ism-1]->GetBinContent(he03_[ism-1]->GetBin(ie, ip)) / numEventsinCry[2];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);

          a03_[ism-1]->SetBinContent(ip+20*(ie-1), mean03);
          a03_[ism-1]->SetBinError(ip+20*(ie-1), rms03);

        }

      }
    }

  }

}

void EBTestPulseClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBTestPulseClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:TestPulseTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">TEST PULSE</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  int csize = 250;

  double histMax = 1.e15;

  int pCol3[3] = { 2, 3, 5 };

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);

  string imgNameQual[3] , imgNameAmp[3] , imgNameShape[3] , imgName , meName;

  TCanvas* cQual = new TCanvas("cQual" , "Temp", 2*csize , csize );
  TCanvas* cAmp = new TCanvas("cAmp" , "Temp", csize , csize );
  TCanvas* cShape = new TCanvas("cShape" , "Temp", csize , csize );

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {
 
    if (  g01_[ism-1] &&  g02_[ism-1] &&  g03_[ism-1] &&
          a01_[ism-1] &&  a02_[ism-1] &&  a03_[ism-1] &&
         hs01_[ism-1] && hs02_[ism-1] && hs03_[ism-1] ) {

      // Loop on gains

      for ( int iCanvas=1 ; iCanvas <= 3 ; iCanvas++ ) {

        // Quality plots

        TH2F* obj2f = 0; 
        switch ( iCanvas ) {
          case 1:
            obj2f = g01_[ism-1];
            break;
          case 2:
            obj2f = g02_[ism-1];
            break;
          case 3:
            obj2f = g03_[ism-1];
            break;
          default:
            break;
        }
        meName = obj2f->GetName();

        for ( unsigned int iQual = 0 ; iQual < meName.size(); iQual++ ) {
          if ( meName.substr(iQual, 1) == " " )  {
            meName.replace(iQual, 1, "_");
          }
        }
        imgNameQual[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameQual[iCanvas-1];

        cQual->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(3, pCol3);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cQual->SetGridx();
        cQual->SetGridy();
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum(2.0);
        obj2f->Draw("col");
        dummy.Draw("text,same");
        cQual->Update();
//        cQual->SaveAs(imgName.c_str());
        gErrorIgnoreLevel = kWarning;
        cQual->Print(imgName.c_str(), "eps");
        gErrorIgnoreLevel = kInfo;
        system(("/usr/bin/convert -geometry 500x250 eps:" + imgName + " png:" + imgName).c_str());

        // Amplitude distributions
        
        TH1F* obj1f = 0; 
        switch ( iCanvas ) {
          case 1:
            obj1f = a01_[ism-1];
            break;
          case 2:
            obj1f = a02_[ism-1];
            break;
          case 3:
            obj1f = a03_[ism-1];
            break;
          default:
            break;
        }
        meName = obj1f->GetName();

        for ( unsigned int iAmp=0 ; iAmp < meName.size(); iAmp++ ) {
          if ( meName.substr(iAmp,1) == " " )  {
            meName.replace(iAmp, 1 ,"_" );
          }
        }
        imgNameAmp[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameAmp[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cAmp->Update();
        gPad->SetLogy(0);
        TPaveStats* stAmp = dynamic_cast<TPaveStats*>(obj1f->FindObject("stats"));
        if ( stAmp ) {
          stAmp->SetX1NDC(0.6);
          stAmp->SetY1NDC(0.75);
        }
//        cAmp->SaveAs(imgName.c_str());
        gErrorIgnoreLevel = kWarning;
        cAmp->Print(imgName.c_str(), "eps");
        gErrorIgnoreLevel = kInfo;
        system(("/usr/bin/convert -geometry 250x250 eps:" + imgName + " png:" + imgName).c_str());
        gPad->SetLogy(0);

        // Shape distributions

        TH1D* obj1d = 0;
        switch ( iCanvas ) {
          case 1:
            obj1d = hs01_[ism-1]->ProjectionY("_py", 1, 10, "e");
            break;
          case 2:
            obj1d = hs02_[ism-1]->ProjectionY("_py", 1, 10, "e");
            break;
          case 3:
            obj1d = hs03_[ism-1]->ProjectionY("_py", 1, 10, "e");
            break;
          default:
            break;
        }
        meName = obj1d->GetName();

        for ( unsigned int iShape=0 ; iShape < meName.size(); iShape++ ) {
          if ( meName.substr(iShape,1) == " " )  {
            meName.replace(iShape, 1, "_");
          }
        }
        imgNameShape[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameShape[iCanvas-1];

        cShape->cd();
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->Draw();
        cShape->Update();
        TPaveStats* stShape = dynamic_cast<TPaveStats*>(obj1d->FindObject("stats"));
        if ( stShape ) {
          stShape->SetX1NDC(0.6);
          stShape->SetY1NDC(0.75);
        }
//        cShape->SaveAs(imgName.c_str());
        gErrorIgnoreLevel = kWarning;
        cShape->Print(imgName.c_str(), "eps");
        gErrorIgnoreLevel = kInfo;
        system(("/usr/bin/convert -geometry 250x250 eps:" + imgName + " png:" + imgName).c_str());
        gPad->SetLogy(0);

        delete obj1d; 

      }

      htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
      htmlFile << "<tr align=\"center\">" << endl;

      for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgNameQual[iCanvas-1].size() != 0 ) 
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<img src=\"" << " " << "\"></td>" << endl;

      }
      htmlFile << "</tr>" << endl;
      htmlFile << "<tr>" << endl;

      for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

        if ( imgNameAmp[iCanvas-1].size() != 0 ) 
          htmlFile << "<td><img src=\"" << imgNameAmp[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<img src=\"" << " " << "\"></td>" << endl;
        
        if ( imgNameShape[iCanvas-1].size() != 0 ) 
          htmlFile << "<td><img src=\"" << imgNameShape[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<img src=\"" << " " << "\"></td>" << endl;

      }

      htmlFile << "</tr>" << endl;

      htmlFile << "<tr align=\"center\"><td colspan=\"2\">Gain 1</td><td colspan=\"2\">Gain 6</td><td colspan=\"2\">Gain 12</td></tr>" << endl;
      htmlFile << "</table>" << endl;
      htmlFile << "<br>" << endl;

    }

  }

  delete cQual;
  delete cAmp;
  delete cShape;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

