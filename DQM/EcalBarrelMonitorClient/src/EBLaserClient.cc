/*
 * \file EBLaserClient.cc
 * 
 * $Date: 2005/12/15 14:20:30 $
 * $Revision: 1.37 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>

EBLaserClient::EBLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    h01_[i] = 0;
    h02_[i] = 0;
    h03_[i] = 0;
    h04_[i] = 0;
    h05_[i] = 0;
    h06_[i] = 0;
    h07_[i] = 0;
    h08_[i] = 0;

    sprintf(histo, "EBLT laser quality L1 SM%02d", i+1);
    g01_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBLT laser quality L2 SM%02d", i+1);
    g02_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBLT laser quality L3 SM%02d", i+1);
    g03_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBLT laser quality L4 SM%02d", i+1);
    g04_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBLT laser amplitude L1 SM%02d", i+1);
    a01_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBLT laser amplitude L2 SM%02d", i+1);
    a02_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBLT laser amplitude L3 SM%02d", i+1);
    a03_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBLT laser amplitude L4 SM%02d", i+1);
    a04_[i] = new TH1F(histo, histo, 1700, 0., 1700.);

    sprintf(histo, "EBLT laser amplitude over PN L1 SM%02d", i+1);
    aopn01_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBLT laser amplitude over PN L2 SM%02d", i+1);
    aopn02_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBLT laser amplitude over PN L3 SM%02d", i+1);
    aopn03_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBLT laser amplitude over PN L4 SM%02d", i+1);
    aopn04_[i] = new TH1F(histo, histo, 1700, 0., 1700.);

  }

  percentVariation_ = 0.4; 

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBLaserClient::~EBLaserClient(){

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    if ( h03_[i] ) delete h03_[i];
    if ( h04_[i] ) delete h04_[i];
    if ( h05_[i] ) delete h05_[i];
    if ( h06_[i] ) delete h06_[i];
    if ( h07_[i] ) delete h07_[i];
    if ( h08_[i] ) delete h08_[i];

    delete g01_[i];
    delete g02_[i];
    delete g03_[i];
    delete g04_[i];

    delete a01_[i];
    delete a02_[i];
    delete a03_[i];
    delete a04_[i];

    delete aopn01_[i];
    delete aopn02_[i];
    delete aopn03_[i];
    delete aopn04_[i];

  }

}

void EBLaserClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBLaserClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBLaserClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBLaserClient: beginRun" << endl;

  jevt_ = 0;

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    if ( h03_[i] ) delete h03_[i];
    if ( h04_[i] ) delete h04_[i];
    if ( h05_[i] ) delete h05_[i];
    if ( h06_[i] ) delete h06_[i];
    if ( h07_[i] ) delete h07_[i];
    if ( h08_[i] ) delete h08_[i];
    h01_[i] = 0;
    h02_[i] = 0;
    h03_[i] = 0;
    h04_[i] = 0;
    h05_[i] = 0;
    h06_[i] = 0;
    h07_[i] = 0;
    h08_[i] = 0;

    g01_[i]->Reset();
    g02_[i]->Reset();
    g03_[i]->Reset();
    g04_[i]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g01_[i]->SetBinContent(g01_[i]->GetBin(ie, ip), 2.);
        g02_[i]->SetBinContent(g02_[i]->GetBin(ie, ip), 2.);
        g03_[i]->SetBinContent(g03_[i]->GetBin(ie, ip), 2.);
        g04_[i]->SetBinContent(g04_[i]->GetBin(ie, ip), 2.);

      }
    }

    a01_[i]->Reset();
    a02_[i]->Reset();
    a03_[i]->Reset();
    a04_[i]->Reset();

    aopn01_[i]->Reset();
    aopn02_[i]->Reset();
    aopn03_[i]->Reset();
    aopn04_[i]->Reset();

  }

  this->subscribe();

}

void EBLaserClient::endJob(void) {

  if ( verbose_ ) cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

}

void EBLaserClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  if ( verbose_ ) cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  MonLaserBlueDat apd_bl;
  map<EcalLogicID, MonLaserBlueDat> dataset_bl;
  MonLaserGreenDat apd_gr;
  map<EcalLogicID, MonLaserGreenDat> dataset_gr;
  MonLaserInfraredDat apd_ir;
  map<EcalLogicID, MonLaserInfraredDat> dataset_ir;
  MonLaserRedDat apd_rd;
  map<EcalLogicID, MonLaserRedDat> dataset_rd;

  cout << "Writing MonLaserDatObjects to database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 50.;
  
  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04;
    float mean01, mean02, mean03, mean04;
    float rms01, rms02, rms03, rms04;

    float meanAmplL1, meanAmplL2;
    int nCryL1, nCryL2;
    meanAmplL1 = meanAmplL2 = -1.;
    nCryL1 = nCryL2 = 0;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num03 = -1;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            meanAmplL1 += h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            nCryL1++;
          }
        }

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            meanAmplL2 += h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            nCryL2++;
          }
        }
        
      }
    }
    
    if ( nCryL1 > 0 ) meanAmplL1 /= float (nCryL1);
    if ( nCryL2 > 0 ) meanAmplL2 /= float (nCryL2);


    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = num04  = -1.;
        mean01 = mean02 = mean03 = mean04 = -1.;
        rms01  = rms02  = rms03  = rms04  = -1.;

        bool update_channel1 = false;
        bool update_channel2 = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            rms02  = h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( h04_[ism-1] && h04_[ism-1]->GetEntries() >= n_min_tot ) {
          num04 = h04_[ism-1]->GetBinEntries(h04_[ism-1]->GetBin(ie, ip));
          if ( num04 >= n_min_bin ) {
            mean04 = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(ie, ip));
            rms04  = h04_[ism-1]->GetBinError(h04_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( update_channel1 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "L1 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;
            cout << "L2 (" << ie << "," << ip << ") " << num02 << " " << mean02 << " " << rms02 << endl;
            cout << "L3 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;
            cout << "L4 (" << ie << "," << ip << ") " << num04 << " " << mean04 << " " << rms04 << endl;

          }

          apd_bl.setAPDMean(mean01);
          apd_bl.setAPDRMS(rms01);
          
          apd_bl.setAPDOverPNMean(mean02);
          apd_bl.setAPDOverPNRMS(rms02);

          apd_bl.setTaskStatus(1);

          float val;

          if ( g01_[ism-1] ) {
            val = 1.;
            if ( abs(mean01 - meanAmplL1) > abs(percentVariation_ * meanAmplL1) )
              val = 0.;
            g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);
          }
          
          if ( a01_[ism-1] ) {
            a01_[ism-1]->SetBinContent(ip+20*(ie-1), mean01);
            a01_[ism-1]->SetBinError(ip+20*(ie-1), rms01);
          }

          if ( aopn01_[ism-1] ) {
            aopn01_[ism-1]->SetBinContent(ip+20*(ie-1), mean02);
            aopn01_[ism-1]->SetBinError(ip+20*(ie-1), rms02);
          }

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
              dataset_bl[ecid] = apd_bl;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

        if ( update_channel2 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "L2 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;

          }

          apd_ir.setAPDMean(mean03);
          apd_ir.setAPDRMS(rms03);

          apd_ir.setAPDOverPNMean(mean04);
          apd_ir.setAPDOverPNRMS(rms04);

          apd_ir.setTaskStatus(1);

          float val;

          if ( g02_[ism-1] ) {
            val = 1.;
            if ( abs(mean03 - meanAmplL2) > abs(percentVariation_ * meanAmplL2) )
              val = 0.;
            g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), val);
          }
          
          if ( a02_[ism-1] ) {
            a02_[ism-1]->SetBinContent(ip+20*(ie-1), mean03);
            a02_[ism-1]->SetBinError(ip+20*(ie-1), rms03);
          }

          if ( aopn02_[ism-1] ) {
            aopn02_[ism-1]->SetBinContent(ip+20*(ie-1), mean04);
            aopn02_[ism-1]->SetBinError(ip+20*(ie-1), rms04);
          }

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
              dataset_ir[ecid] = apd_ir;
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
      econn->insertDataSet(&dataset_bl, runiov, runtag );
      econn->insertDataSet(&dataset_ir, runiov, runtag );
      cout << "done." << endl; 
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  this->unsubscribe();

}

void EBLaserClient::subscribe(void){

  if ( verbose_ ) cout << "EBLaserClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBLaserClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBLT amplitude SM%02d L1", ism);
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L1", ism);
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1", ism);
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L2", ism);
      me_h03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2", ism);
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L2", ism);
      me_h04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2", ism);
      mui_->add(me_h04_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L3", ism);
      me_h05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3", ism);
      mui_->add(me_h05_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L3", ism);
      me_h06_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3", ism);
      mui_->add(me_h06_[ism-1], histo);

      sprintf(histo, "EBLT amplitude SM%02d L4", ism);
      me_h07_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4", ism);
      mui_->add(me_h07_[ism-1], histo);

      sprintf(histo, "EBLT amplitude over PN SM%02d L4", ism);
      me_h08_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBLaserTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4", ism);
      mui_->add(me_h08_[ism-1], histo);

    }

  }

}

void EBLaserClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM*");

}

void EBLaserClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBLaserClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBLaserClient: uncollate" << endl;

    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( bei ) {

      Char_t histo[80];

      for ( int ism = 1; ism <= 36; ism++ ) {

        sprintf(histo, "EBLT amplitude SM%02d L1", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser1");
        bei->removeElement(histo);

        sprintf(histo, "EBLT amplitude over PN SM%02d L1", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser1");
        bei->removeElement(histo);

        sprintf(histo, "EBLT amplitude SM%02d L2", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser2");
        bei->removeElement(histo);

        sprintf(histo, "EBLT amplitude over PN SM%02d L2", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser2");
        bei->removeElement(histo);

        sprintf(histo, "EBLT amplitude SM%02d L3", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser3");
        bei->removeElement(histo);

        sprintf(histo, "EBLT amplitude over PN SM%02d L3", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser3");
        bei->removeElement(histo);

        sprintf(histo, "EBLT amplitude SM%02d L4", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser4");
        bei->removeElement(histo);

        sprintf(histo, "EBLT amplitude over PN SM%02d L4", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBLaserTask/Laser4");
        bei->removeElement(histo);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM*");

}

void EBLaserClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) { 
    if ( verbose_ ) cout << "EBLaserClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];
  
  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude SM%02d L1", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude SM%02d L1", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        sprintf(histo, "ME EBLT amplitude SM%02d L1", ism);
        h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser1/EBLT amplitude over PN SM%02d L1", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        sprintf(histo, "ME EBLT amplitude over PN SM%02d L1", ism);
        h02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      } 
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude SM%02d L2", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude SM%02d L2", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        sprintf(histo, "ME EBLT amplitude SM%02d L2", ism);
        h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      } 
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser2/EBLT amplitude over PN SM%02d L2", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h04_[ism-1] ) delete h04_[ism-1];
        sprintf(histo, "ME EBLT amplitude over PN SM%02d L2", ism);
        h04_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h04_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      } 
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude SM%02d L3", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude SM%02d L3", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h05_[ism-1] ) delete h05_[ism-1];
        sprintf(histo, "ME EBLT amplitude SM%02d L3", ism);
        h05_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h05_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser3/EBLT amplitude over PN SM%02d L3", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h06_[ism-1] ) delete h06_[ism-1];
        sprintf(histo, "ME EBLT amplitude over PN SM%02d L3", ism);
        h06_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h06_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude SM%02d L4", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude SM%02d L4", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h07_[ism-1] ) delete h07_[ism-1];
        sprintf(histo, "ME EBLT amplitude SM%02d L4", ism);
        h07_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h07_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBLaserTask/Laser4/EBLT amplitude over PN SM%02d L4", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h08_[ism-1] ) delete h08_[ism-1];
        sprintf(histo, "ME EBLT amplitude over PN SM%02d L4", ism);
        h08_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h08_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

  }

}

void EBLaserClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBLaserClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:LaserTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">LASER</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=white>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .jpg files from existing histograms

  int csize = 250;

  double histMax = 1.e15;

  int pCol3[3] = { 2, 3, 10 };

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);

  string imgNameQual[2] , imgNameAmp[2] , imgNameAmpoPN[2] , imgName , meName;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {
    
    if ( g01_[ism-1] && g02_[ism-1] &&
         a01_[ism-1] && a02_[ism-1] &&
         aopn01_[ism-1] && aopn02_[ism-1] ) {

      // Loop on wavelenght

      for ( int iCanvas=1 ; iCanvas <= 2 ; iCanvas++ ) {

        // Quality plots

        TH2F* obj2f = 0; 

        switch ( iCanvas ) {
        case 1:
          meName = g01_[ism-1]->GetName();
          obj2f = g01_[ism-1];
          break;
        case 2:
          meName = g02_[ism-1]->GetName();
          obj2f = g02_[ism-1];
          break;
        default:
          break;
        }

        TCanvas *cQual = new TCanvas("cQual" , "Temp", 2*csize , csize );
        for ( unsigned int iQual = 0 ; iQual < meName.size(); iQual++ ) {
          if ( meName.substr(iQual, 1) == " " )  {
            meName.replace(iQual, 1, "_");
          }
        }
        imgNameQual[iCanvas-1] = meName + ".jpg";
        imgName = htmlDir + imgNameQual[iCanvas-1];
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
        cQual->SaveAs(imgName.c_str());
        delete cQual;

        // Amplitude distributions
        
        TH1F* obj1f = 0; 
        
        switch ( iCanvas ) {
        case 1:
          meName = a01_[ism-1]->GetName();
          obj1f = a01_[ism-1];
          break;
        case 2:
          meName = a02_[ism-1]->GetName();
          obj1f = a02_[ism-1];
          break;
        default:
          break;
        }
        
        TCanvas *cAmp = new TCanvas("cAmp" , "Temp", csize , csize );
        for ( unsigned int iAmp=0 ; iAmp < meName.size(); iAmp++ ) {
          if ( meName.substr(iAmp,1) == " " )  {
            meName.replace(iAmp, 1 ,"_" );
          }
        }
        imgNameAmp[iCanvas-1] = meName + ".jpg";
        imgName = htmlDir + imgNameAmp[iCanvas-1];
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
        cAmp->SaveAs(imgName.c_str());
        delete cAmp;
        
        // Amplitude over PN distributions
        
        switch ( iCanvas ) {
        case 1:
          meName = aopn01_[ism-1]->GetName();
          obj1f = aopn01_[ism-1];
          break;
        case 2:
          meName = aopn02_[ism-1]->GetName();
          obj1f = aopn02_[ism-1];
          break;
        default:
            break;
          }
        
        TCanvas *cAmpoPN = new TCanvas("cAmpoPN" , "Temp", csize , csize );
        for ( unsigned int iAmpoPN=0 ; iAmpoPN < meName.size(); iAmpoPN++ ) {
          if ( meName.substr(iAmpoPN,1) == " " )  {
            meName.replace(iAmpoPN, 1, "_");
          }
        }
        imgNameAmpoPN[iCanvas-1] = meName + ".jpg";
        imgName = htmlDir + imgNameAmpoPN[iCanvas-1];
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->Draw();
        cAmpoPN->Update();
        TPaveStats* stAmpoPN = dynamic_cast<TPaveStats*>(obj1f->FindObject("stats"));
        if ( stAmpoPN ) {
          stAmpoPN->SetX1NDC(0.6);
          stAmpoPN->SetY1NDC(0.75);
        }
        cAmpoPN->SaveAs(imgName.c_str());
        gPad->SetLogy(0);
        delete cAmpoPN;
        
      }

      htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
      htmlFile << "<tr align=\"center\">" << endl;

      for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

        if ( imgNameQual[iCanvas-1].size() != 0 ) 
          htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<img src=\"" << " " << "\"></td>" << endl;

      }
      htmlFile << "</tr>" << endl;
      htmlFile << "<tr>" << endl;

      for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

        if ( imgNameAmp[iCanvas-1].size() != 0 ) 
          htmlFile << "<td><img src=\"" << imgNameAmp[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<img src=\"" << " " << "\"></td>" << endl;
        
        if ( imgNameAmpoPN[iCanvas-1].size() != 0 ) 
          htmlFile << "<td><img src=\"" << imgNameAmpoPN[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<img src=\"" << " " << "\"></td>" << endl;

      }

      htmlFile << "</tr>" << endl;

      htmlFile << "<tr align=\"center\"><td colspan=\"2\">Laser 1</td><td colspan=\"2\">Laser 2</td></tr>" << endl;
      htmlFile << "</table>" << endl;
      htmlFile << "<br>" << endl;
    
    }

  }

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

