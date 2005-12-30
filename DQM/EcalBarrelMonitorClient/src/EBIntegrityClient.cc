/*
 * \file EBIntegrityClient.cc
 *
 * $Date: 2005/12/29 19:41:37 $
 * $Revision: 1.60 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>

EBIntegrityClient::EBIntegrityClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  h00_ = 0;

  for ( int ism = 1; ism <= 36; ism++ ) {

    h_[ism-1] = 0;

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g01_[ism-1] = 0;

  }

  threshCry_ = 0.;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBIntegrityClient::~EBIntegrityClient(){

  this->cleanup();

}

void EBIntegrityClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBIntegrityClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBIntegrityClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBIntegrityClient::endJob(void) {

  if ( verbose_ ) cout << "EBIntegrityClient: endJob, ievt = " << ievt_ << endl;

}

void EBIntegrityClient::endRun(void) {

  if ( verbose_ ) cout << "EBIntegrityClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBIntegrityClient::setup(void) {

  Char_t histo[50];

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    sprintf(histo, "EBPT data integrity quality SM%02d", ism);
    g01_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g01_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);

      }
    }

  }

}

void EBIntegrityClient::cleanup(void) {

  if ( h00_ ) delete h00_;
  h00_ = 0;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( h_[ism-1] ) delete h_[ism-1];
    h_[ism-1] = 0;

    if ( h01_[ism-1] ) delete h01_[ism-1];
    h01_[ism-1] = 0;
    if ( h02_[ism-1] ) delete h02_[ism-1];
    h02_[ism-1] = 0;
    if ( h03_[ism-1] ) delete h03_[ism-1];
    h03_[ism-1] = 0;
    if ( h04_[ism-1] ) delete h04_[ism-1];
    h04_[ism-1] = 0;
    if ( h05_[ism-1] ) delete h05_[ism-1];
    h05_[ism-1] = 0;
    if ( h06_[ism-1] ) delete h06_[ism-1];
    h06_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    g01_[ism-1] = 0;

  }

}

void EBIntegrityClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  EcalLogicID ecid;
  RunConsistencyDat c;
  map<EcalLogicID, RunConsistencyDat> dataset;

  cout << "Writing RunConsistencyDatObjects to database ..." << endl;

  const float n_min_tot = 0.;
  const float n_min_bin = 0.;

  float num00;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04, num05, num06;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        float numEventsinCry = 0.;
        if ( h_[ism-1] ) numEventsinCry = h_[ism-1]->GetBinEntries(h_[ism-1]->GetBin(ie, ip)) / 3.;

        // cout << "Number of events per crystal (" << ie << "," << ip << ") SM " << ism << " " << numEventsinCry << endl;

        if ( numEventsinCry > n_min_tot ) {

          num00 = -1.;

          num01 = num02 = num03 = num04 = num05 = num06 = -1.;

          bool update_channel = false;

          if ( h00_ ) {
            num00  = h00_->GetBinContent(h00_->GetBin(ism));
//            if ( num00 > n_min_bin ) update_channel = true;
          }

          if ( h01_[ism-1] ) {
            num01  = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            if ( num01 > n_min_bin ) update_channel = true;
          }

          if ( h02_[ism-1] ) {
            num02  = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            if ( num02 > n_min_bin ) update_channel = true;
          }

          if ( h03_[ism-1] ) {
            num03  = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            if ( num03 > n_min_bin ) update_channel = true;
          }

          if ( h04_[ism-1] ) {
            num04  = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(ie, ip));
            if ( num04 > n_min_bin ) update_channel = true;
          }

          int iet = 1 + ((ie-1)/5);
          int ipt = 1 + ((ip-1)/5);

          if ( h05_[ism-1] ) {
            num05  = h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(iet, ipt));
            if ( num05 > n_min_bin ) update_channel = true;
          }

          if ( h06_[ism-1] ) {
            num06  = h06_[ism-1]->GetBinContent(h06_[ism-1]->GetBin(iet, ipt));
            if ( num06 > n_min_bin ) update_channel = true;
          }

          if ( update_channel ) {

            if ( ie == 1 && ip == 1 ) {

              cout << "Inserting dataset for SM=" << ism << endl;

              cout << "(" << ie << "," << ip << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << " " << num04 << " " << num05 << " " << num06 << endl;

            }

            c.setExpectedEvents(0);
            c.setProblemsInGain(int(num01));
            c.setProblemsInId(int(num02));
            c.setProblemsInSample(int(-999));
            c.setProblemsInADC(int(-999));

//            if ( g01_[ism-1]->GetBinContent(g01_[ism-1]->GetBin(ie, ip)) == 1. ) {
//               c.setTaskStatus(true);
//            } else {
//               c.setTaskStatus(false);
//            }

            if ( econn ) {
              try {
                ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
                dataset[ecid] = c;
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
      econn->insertDataSet(&dataset, runiov, runtag);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBIntegrityClient::subscribe(void){

  if ( verbose_ ) cout << "EBIntegrityClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  //mui_->subscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPPST pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/EBIT DCC size error");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/Gain/EBIT gain SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/ChId/EBIT ChId SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/GainSwitch/EBIT gain switch SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/GainSwitchStay/EBIT gain switch stay SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/TTId/EBIT TTId SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/TTBlockSize/EBIT TTBlockSize SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBIntegrityClient: collate" << endl;

    Char_t histo[80];

    sprintf(histo, "EBIT DCC size error");
    me_h00_ = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EcalIntegrity");
    sprintf(histo, "*/EcalBarrel/EcalIntegrity/EBIT DCC size error");
    mui_->add(me_h00_, histo);

    for ( int ism = 1; ism <= 36; ism++ ) {

//   not needed: the same CollateMonitorElements are built in EBPedPreSample.cc

//      sprintf(histo, "EBPPST pedestal SM%02d G12", ism);
//      me_h_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedPreSampleTask/Gain12");
//      sprintf(histo, "*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPPST pedestal SM%02d G12", ism);
//      mui_->add(me_h_[ism-1], histo);

      sprintf(histo, "EBIT gain SM%02d", ism);
      me_h01_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EcalIntegrity/Gain");
      sprintf(histo, "*/EcalBarrel/EcalIntegrity/Gain/EBIT gain SM%02d", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBIT ChId SM%02d", ism);
      me_h02_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EcalIntegrity/ChId");
      sprintf(histo, "*/EcalBarrel/EcalIntegrity/ChId/EBIT ChId SM%02d", ism);
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EBIT gain switch SM%02d", ism);
      me_h03_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EcalIntegrity/GainSwitch");
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EBIT gain switch stay SM%02d", ism);
      me_h04_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EcalIntegrity/GainSwitchStay");
      sprintf(histo, "*/EcalBarrel/EcalIntegrity/TTBlockSize/EBIT TTBlockSize SM%02d", ism);
      mui_->add(me_h04_[ism-1], histo);

      sprintf(histo, "EBIT TTId SM%02d", ism);
      me_h05_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EcalIntegrity/TTId");
      sprintf(histo, "*/EcalBarrel/EcalIntegrity/TTId/EBIT TTId SM%02d", ism);
      mui_->add(me_h05_[ism-1], histo);

      sprintf(histo, "EBIT TTBlockSize SM%02d", ism);
      me_h06_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EcalIntegrity/TTBlockSize");
      sprintf(histo, "*/EcalBarrel/EcalIntegrity/TTBlockSize/EBIT TTBlockSize SM%02d", ism);
      mui_->add(me_h06_[ism-1], histo);

    }

  }

}

void EBIntegrityClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  //mui_->subscribeNew("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPPST pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/EBIT DCC size error");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/Gain/EBIT gain SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/ChId/EBIT ChId SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/GainSwitch/EBIT gain switch SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/GainSwitchStay/EBIT gain switch stay SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/TTId/EBIT TTId SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/TTBlockSize/EBIT TTBlockSize SM*");

}

void EBIntegrityClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBIntegrityClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBIntegrityClient: uncollate" << endl;

    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( bei ) {

      Char_t histo[80];

      sprintf(histo, "EBIT DCC size error");
      bei->setCurrentFolder("EcalBarrel/Sums/EcalIntegrity");
      bei->removeElement(histo);

      for ( int ism = 1; ism <= 36; ism++ ) {

//     not needed: the same CollateMonitorElements are built in EBPedPreSample.cc

//        sprintf(histo, "EBPPST pedestal SM%02d G12", ism);
//        bei->setCurrentFolder("EcalBarrel/Sums/EBPedPreSampleTask/Gain12");
//        bei->removeElement(histo);

        sprintf(histo, "EBIT gain SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EcalIntegrity/Gain");
        bei->removeElement(histo);

        sprintf(histo, "EBIT ChId SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EcalIntegrity/ChId");
        bei->removeElement(histo);

        sprintf(histo, "EBIT gain switch SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EcalIntegrity/GainSwitch");
        bei->removeElement(histo);

        sprintf(histo, "EBIT gain switch stay SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EcalIntegrity/GainSwitchStay");
        bei->removeElement(histo);

        sprintf(histo, "EBIT TTId SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EcalIntegrity/TTId");
        bei->removeElement(histo);

        sprintf(histo, "EBIT TTBlockSize SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EcalIntegrity/TTBlockSize");
        bei->removeElement(histo);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  //mui_->unsubscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPPST pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/EBIT DCC size error");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/Gain/EBIT gain SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/ChId/EBIT ChId SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/GainSwitch/EBIT gain switch SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/GainSwitchStay/EBIT gain switch stay SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/TTId/EBIT TTId SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/TTBlockSize/EBIT TTBlockSize SM*");

}

void EBIntegrityClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  if ( collateSources_ ) {
    sprintf(histo, "EcalBarrel/Sums/EcalIntegrity/EBIT DCC size error");
  } else {
    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/EBIT DCC size error");
  }
  me = mui_->get(histo);
  if ( me ) {
    if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
    ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    if ( ob ) {
      if ( h00_ ) delete h00_;
      sprintf(histo, "ME EBIT DCC size error");
      h00_ = dynamic_cast<TH1F*> ((ob->operator->())->Clone(histo));
//      h00_ = dynamic_cast<TH1F*> (ob->operator->());
    }
  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedPreSampleTask/Gain12/EBPPST pedestal SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedPreSampleTask/Gain12/EBPPST pedestal SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h_[ism-1] ) delete h_[ism-1];
        sprintf(histo, "ME EBPPST pedestal SM%02d G12", ism);
        h_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EcalIntegrity/Gain/EBIT gain SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/Gain/EBIT gain SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        sprintf(histo, "ME EBIT gain SM%02d", ism);
        h01_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        h01_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EcalIntegrity/ChId/EBIT ChId SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/ChId/EBIT ChId SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        sprintf(histo, "ME EBIT ChId SM%02d", ism);
        h02_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        h02_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EcalIntegrity/GainSwitch/EBIT gain switch SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/GainSwitch/EBIT gain switch SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        sprintf(histo, "ME EBIT gain switch SM%02d", ism);
        h03_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        h03_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EcalIntegrity/GainSwitchStay/EBIT gain switch stay SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/GainSwitchStay/EBIT gain switch stay SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h04_[ism-1] ) delete h04_[ism-1];
        sprintf(histo, "ME EBIT gain switch stay SM%02d", ism);
        h04_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        h04_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EcalIntegrity/TTId/EBIT TTId SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/TTId/EBIT TTId SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h05_[ism-1] ) delete h05_[ism-1];
        sprintf(histo, "ME EBIT TTId SM%02d", ism);
        h05_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        h05_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EcalIntegrity/TTBlockSize/EBIT TTBlockSize SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/TTBlockSize/EBIT TTBlockSize SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h06_[ism-1] ) delete h06_[ism-1];
        sprintf(histo, "ME EBIT TTBlockSize SM%02d", ism);
        h06_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
//        h06_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
      }
    }

    const float n_min_tot = 0.;

    float num00;

    float num01, num02, num03, num04, num05, num06;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        float numEventsinCry = 0.;
        if ( h_[ism-1] ) numEventsinCry = h_[ism-1]->GetBinEntries(h_[ism-1]->GetBin(ie, ip)) / 3.;

        // cout << "Number of events per crystal (" << ie << "," << ip << ") SM " << ism << " " << numEventsinCry << endl;

        if ( numEventsinCry > n_min_tot ) {

          num00 = 0.;

          if ( h00_ ) {
            num00  = h00_->GetBinContent(h00_->GetBin(ie, ip));
          }

          num01 = num02 = num03 = num04 = num05 = num06 = 0.;

          bool update_channel = false;

          g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);

          if ( h01_[ism-1] ) {
            num01  = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }

          if ( h02_[ism-1] ) {
            num02  = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }

          if ( h03_[ism-1] ) {
            num03  = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }

          if ( h04_[ism-1] ) {
            num04  = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }

          int iet = 1 + ((ie-1)/5);
          int ipt = 1 + ((ip-1)/5);

          if ( h05_[ism-1] ) {
            num05  = h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(iet, ipt));
            update_channel = true;
          }

          if ( h06_[ism-1] ) {
            num06  = h06_[ism-1]->GetBinContent(h06_[ism-1]->GetBin(iet, ipt));
            update_channel = true;
          }

          if ( update_channel ) {

            float val;

            val = 1.;
            if ( (( num01 + num02 + num03 + num04 + num05 + num06) / numEventsinCry / 6. ) > threshCry_ )
              val = 0.;
            g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);

          }

        }

      }
    }

  }

}

void EBIntegrityClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBIntegrityClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:IntegrityTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">INTEGRITY</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  int csize = 250;

  int pCol3[3] = { 2, 3, 5 };
  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 30+i;

  TH2C dummy1( "dummy1", "dummy1 for sm", 85, 0, 85, 20, 0, 20 );
  for( short i=0; i<68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy1.Fill( a, b, i+1 );
  }
  dummy1.SetMarkerSize(2);

  TH2C dummy2( "dummy2", "dummy2 for sm", 17, 0, 17, 4, 0, 4 );
  for( short i=0; i<68; i++ ) {
    int a = ( i/4 );
    int b = ( i%4 );
    dummy2.Fill( a, b, i+1 );
  }
  dummy2.SetMarkerSize(2);

  string imgNameDCC, imgNameQual, imgNameME[6], imgName , meName;

  TCanvas* cDCC = new TCanvas("cDCC", "Temp", 2*csize, csize);
  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cMe = new TCanvas("cMe", "Temp", 2*csize, csize);

  TH1F* obj1f;
  TH2F* obj2f;

  // DCC size error

  imgNameDCC = "";

  obj1f = h00_;

  if ( obj1f ) {

    meName = obj1f->GetName();

    for ( unsigned int i = 0; i < meName.size(); i++ ) {
      if ( meName.substr(i, 1) == " " )  {
        meName.replace(i, 1, "_");
      }
    }
    imgNameDCC = meName + ".png";
    imgName = htmlDir + imgNameDCC;

    cDCC->cd();
    gStyle->SetOptStat(" ");
    obj1f->Draw();
    cDCC->Update();
    cDCC->SaveAs(imgName.c_str());

  }

  htmlFile << "<h3><strong>DCC size error</strong></h3>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr align=\"left\">" << endl;

  if ( imgNameDCC.size() != 0 )
    htmlFile << "<td><img src=\"" << imgNameDCC << "\"></td>" << endl;
  else
    htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    // Quality plots

    imgNameQual = "";

    obj2f = g01_[ism-1];

    if ( obj2f ) {

      meName = obj2f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameQual = meName + ".png";
      imgName = htmlDir + imgNameQual;

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
      dummy1.Draw("text,same");
      cQual->Update();
      cQual->SaveAs(imgName.c_str());

    }

    // Monitoring elements plots

    for ( int iCanvas = 1; iCanvas <= 6; iCanvas++ ) {

      imgNameME[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = h01_[ism-1];
          break;
        case 2:
          obj2f = h02_[ism-1];
          break;
        case 3:
          obj2f = h03_[ism-1];
          break;
        case 4:
          obj2f = h04_[ism-1];
          break;
        case 5:
          obj2f = h05_[ism-1];
          break;
        case 6:
          obj2f = h06_[ism-1];
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int iMe = 0; iMe < meName.size(); iMe++ ) {
          if ( meName.substr(iMe, 1) == " " )  {
            meName.replace(iMe, 1, "_");
          }
        }
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMe->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cMe->SetGridx();
        cMe->SetGridy();
        obj2f->SetMaximum();
        obj2f->Draw("colz");
        if ( iCanvas < 5 )
          dummy1.Draw("text,same");
        else
          dummy2.Draw("text,same");
        cMe->Update();
        cMe->SaveAs(imgName.c_str());

      }

    }

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\"> " << endl;
    htmlFile << "<tr align=\"left\">" << endl;

    if ( imgNameQual.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameQual << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 3 ; iCanvas <= 4 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 5 ; iCanvas <= 6 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cDCC;
  delete cQual;
  delete cMe;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

