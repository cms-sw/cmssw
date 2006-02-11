/*
 * \file EBPedestalClient.cc
 *
 * $Date: 2006/02/09 13:37:56 $
 * $Revision: 1.61 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>

EBPedestalClient::EBPedestalClient(const ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  for ( int ism = 1; ism <= 36; ism++ ) {

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;

    k01_[ism-1] = 0;
    k02_[ism-1] = 0;
    k03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g01_[ism-1] = 0;
    g02_[ism-1] = 0;
    g03_[ism-1] = 0;

    p01_[ism-1] = 0;
    p02_[ism-1] = 0;
    p03_[ism-1] = 0;

    r01_[ism-1] = 0;
    r02_[ism-1] = 0;
    r03_[ism-1] = 0;

    s01_[ism-1] = 0;
    s02_[ism-1] = 0;
    s03_[ism-1] = 0;

    t01_[ism-1] = 0;
    t02_[ism-1] = 0;
    t03_[ism-1] = 0;

  }

  expectedMean_[0] = 200.0;
  expectedMean_[1] = 200.0;
  expectedMean_[2] = 200.0;
  discrepancyMean_[0] = 25.0;
  discrepancyMean_[1] = 25.0;
  discrepancyMean_[2] = 25.0;
  RMSThreshold_[0] = 1.0;
  RMSThreshold_[1] = 1.2;
  RMSThreshold_[2] = 2.0;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBPedestalClient::~EBPedestalClient(){

  this->cleanup();

}

void EBPedestalClient::beginJob(void){

  if ( verbose_ ) cout << "EBPedestalClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBPedestalClient::beginRun(void){

  if ( verbose_ ) cout << "EBPedestalClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBPedestalClient::endJob(void) {

  if ( verbose_ ) cout << "EBPedestalClient: endJob, ievt = " << ievt_ << endl;

}

void EBPedestalClient::endRun(void) {

  if ( verbose_ ) cout << "EBPedestalClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBPedestalClient::setup(void) {

  Char_t histo[50];

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    sprintf(histo, "EBPT pedestal quality G01 SM%02d", ism);
    g01_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g02_[ism-1] ) delete g02_[ism-1];
    sprintf(histo, "EBPT pedestal quality G06 SM%02d", ism);
    g02_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g03_[ism-1] ) delete g03_[ism-1];
    sprintf(histo, "EBPT pedestal quality G12 SM%02d", ism);
    g03_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( p01_[ism-1] ) delete p01_[ism-1];
    sprintf(histo, "EBPT pedestal mean G01 SM%02d", ism);
    p01_[ism-1] = new TH1F(histo, histo, 100, 150., 250.);
    if ( p02_[ism-1] ) delete p02_[ism-1];
    sprintf(histo, "EBPT pedestal mean G06 SM%02d", ism);
    p02_[ism-1] = new TH1F(histo, histo, 100, 150., 250.);
    if ( p03_[ism-1] ) delete p03_[ism-1];
    sprintf(histo, "EBPT pedestal mean G12 SM%02d", ism);
    p03_[ism-1] = new TH1F(histo, histo, 100, 150., 250.);

    if ( r01_[ism-1] ) delete r01_[ism-1];
    sprintf(histo, "EBPT pedestal rms G01 SM%02d", ism);
    r01_[ism-1] = new TH1F(histo, histo, 100, 0., 10.);
    if ( r02_[ism-1] ) delete r02_[ism-1];
    sprintf(histo, "EBPT pedestal rms G06 SM%02d", ism);
    r02_[ism-1] = new TH1F(histo, histo, 100, 0., 10.);
    if ( r03_[ism-1] ) delete r03_[ism-1];
    sprintf(histo, "EBPT pedestal rms G12 SM%02d", ism);
    r03_[ism-1] = new TH1F(histo, histo, 100, 0., 10.);

    if ( s01_[ism-1] ) delete s01_[ism-1];
    sprintf(histo, "EBPT 3sum G01 SM%02d", ism);
    s01_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( s02_[ism-1] ) delete s02_[ism-1];
    sprintf(histo, "EBPT 3sum G06 SM%02d", ism);
    s02_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( s03_[ism-1] ) delete s03_[ism-1];
    sprintf(histo, "EBPT 3sum G12 SM%02d", ism);
    s03_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( t01_[ism-1] ) delete t01_[ism-1];
    sprintf(histo, "EBPT 5sum G01 SM%02d", ism);
    t01_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( t02_[ism-1] ) delete t02_[ism-1];
    sprintf(histo, "EBPT 5sum G06 SM%02d", ism);
    t02_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( t03_[ism-1] ) delete t03_[ism-1];
    sprintf(histo, "EBPT 5sum G12 SM%02d", ism);
    t03_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

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

    p01_[ism-1]->Reset();
    p02_[ism-1]->Reset();
    p03_[ism-1]->Reset();

    r01_[ism-1]->Reset();
    r02_[ism-1]->Reset();
    r03_[ism-1]->Reset();

    s01_[ism-1]->Reset();
    s02_[ism-1]->Reset();
    s03_[ism-1]->Reset();

    t01_[ism-1]->Reset();
    t02_[ism-1]->Reset();
    t03_[ism-1]->Reset();

  }

}

void EBPedestalClient::cleanup(void) {

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];

      if ( j01_[ism-1] ) delete j01_[ism-1];
      if ( j02_[ism-1] ) delete j02_[ism-1];
      if ( j03_[ism-1] ) delete j03_[ism-1];

      if ( k01_[ism-1] ) delete k01_[ism-1];
      if ( k02_[ism-1] ) delete k02_[ism-1];
      if ( k03_[ism-1] ) delete k03_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;

    k01_[ism-1] = 0;
    k02_[ism-1] = 0;
    k03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    g01_[ism-1] = 0;
    if ( g02_[ism-1] ) delete g02_[ism-1];
    g02_[ism-1] = 0;
    if ( g03_[ism-1] ) delete g03_[ism-1];
    g03_[ism-1] = 0;

    if ( p01_[ism-1] ) delete p01_[ism-1];
    p01_[ism-1] = 0;
    if ( p02_[ism-1] ) delete p02_[ism-1];
    p02_[ism-1] = 0;
    if ( p03_[ism-1] ) delete p03_[ism-1];
    p03_[ism-1] = 0;

    if ( r01_[ism-1] ) delete r01_[ism-1];
    r01_[ism-1] = 0;
    if ( r02_[ism-1] ) delete r02_[ism-1];
    r02_[ism-1] = 0;
    if ( r03_[ism-1] ) delete r03_[ism-1];
    r03_[ism-1] = 0;

    if ( s01_[ism-1] ) delete s01_[ism-1];
    s01_[ism-1] = 0;
    if ( s02_[ism-1] ) delete s02_[ism-1];
    s02_[ism-1] = 0;
    if ( s03_[ism-1] ) delete s03_[ism-1];
    s03_[ism-1] = 0;

    if ( t01_[ism-1] ) delete t01_[ism-1];
    t01_[ism-1] = 0;
    if ( t02_[ism-1] ) delete t02_[ism-1];
    t02_[ism-1] = 0;
    if ( t03_[ism-1] ) delete t03_[ism-1];
    t03_[ism-1] = 0;

  }

}

void EBPedestalClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov) {

  EcalLogicID ecid;
  MonPedestalsDat p;
  map<EcalLogicID, MonPedestalsDat> dataset1;

  cout << "Creating MonPedestalsDatObjects for the database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03;
    float mean01, mean02, mean03;
    float rms01, rms02, rms03;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = -1.;
        mean01 = mean02 = mean03 = -1.;
        rms01  = rms02  = rms03  = -1.;

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            rms02  = h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "G01 (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "G06 (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << endl;
            cout << "G12 (" << ie << "," << ip << ") " << num03  << " " << mean03 << " " << rms03  << endl;

          }

          p.setPedMeanG1(mean01);
          p.setPedRMSG1(rms01);

          p.setPedMeanG6(mean02);
          p.setPedRMSG6(rms02);

          p.setPedMeanG12(mean03);
          p.setPedRMSG12(rms03);

          if ( g01_[ism-1] && g01_[ism-1]->GetBinContent(g01_[ism-1]->GetBin(ie, ip)) == 1. &&
               g02_[ism-1] && g02_[ism-1]->GetBinContent(g02_[ism-1]->GetBin(ie, ip)) == 1. &&
               g03_[ism-1] && g03_[ism-1]->GetBinContent(g03_[ism-1]->GetBin(ie, ip)) == 1. ) {
            p.setTaskStatus(true);
          } else {
            p.setTaskStatus(false);
          }

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
              dataset1[ecid] = p;
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
      if ( dataset1.size() != 0 ) econn->insertDataSet(&dataset1, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  MonPNPedDat pn;
  map<EcalLogicID, MonPNPedDat> dataset2;

  cout << "Creating MonPnDatObjects for the database ..." << endl;

  const float m_min_tot = 1000.;
  const float m_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int i = 1; i <= 10; i++ ) {

      num01  = num02  = -1.;
      mean01 = mean02 = -1.;
      rms01  = rms02  = -1.;

      bool update_channel = false;

      if ( i01_[ism-1] && i01_[ism-1]->GetEntries() >= m_min_tot ) {
        num01 = i01_[ism-1]->GetBinEntries(i01_[ism-1]->GetBin(1, i));
        if ( num01 >= m_min_bin ) {
          mean01 = i01_[ism-1]->GetBinContent(i01_[ism-1]->GetBin(1, i));
          rms01  = i01_[ism-1]->GetBinError(i01_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( i02_[ism-1] && i02_[ism-1]->GetEntries() >= m_min_tot ) {
        num02 = i02_[ism-1]->GetBinEntries(i02_[ism-1]->GetBin(1, i));
        if ( num02 >= m_min_bin ) {
          mean02 = i02_[ism-1]->GetBinContent(i02_[ism-1]->GetBin(1, i));
          rms02  = i02_[ism-1]->GetBinError(i02_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( update_channel ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "PNs (" << i << ") G16 " << num01  << " " << mean01 << " " << rms01  << endl;

        }

        pn.setPedMeanG1(mean01);
        pn.setPedRMSG1(rms01);

        pn.setPedMeanG16(mean02);
        pn.setPedRMSG16(rms02);

        if ( mean01 > 200. ) {
          pn.setTaskStatus(true);
        } else {
          pn.setTaskStatus(false);
        }

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset2[ecid] = pn;
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
      if ( dataset2.size() != 0 ) econn->insertDataSet(&dataset2, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBPedestalClient::subscribe(void){

  if ( verbose_ ) cout << "EBPedestalClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT 3sum SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT 3sum SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT 3sum SM*");

  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT 5sum SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT 5sum SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT 5sum SM*");

  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPedestalClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBPT pedestal SM%02d G01", ism);
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM%02d G01", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBPT pedestal SM%02d G06", ism);
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM%02d G06", ism);
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EBPT pedestal SM%02d G12", ism);
      me_h03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM%02d G12", ism);
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EBPT 3sum SM%02d G01", ism);
      me_j01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain01/EBPT 3sum SM%02d G01", ism);
      mui_->add(me_j01_[ism-1], histo);

      sprintf(histo, "EBPT 3sum SM%02d G06", ism);
      me_j02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain06/EBPT 3sum SM%02d G06", ism);
      mui_->add(me_j02_[ism-1], histo);

      sprintf(histo, "EBPT 3sum SM%02d G12", ism);
      me_j03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain12/EBPT 3sum SM%02d G12", ism);
      mui_->add(me_j03_[ism-1], histo);

      sprintf(histo, "EBPT 5sum SM%02d G01", ism);
      me_k01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain01/EBPT 5sum SM%02d G01", ism);
      mui_->add(me_k01_[ism-1], histo);

      sprintf(histo, "EBPT 5sum SM%02d G06", ism);
      me_k02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain06/EBPT 5sum SM%02d G06", ism);
      mui_->add(me_k02_[ism-1], histo);

      sprintf(histo, "EBPT 5sum SM%02d G12", ism);
      me_k03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBPedestalTask/Gain12/EBPT 5sum SM%02d G12", ism);
      mui_->add(me_k03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01", ism);
      me_i01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM%02d G01", ism);
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16", ism);
      me_i02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM%02d G16", ism);
      mui_->add(me_i02_[ism-1], histo);

    }

  }

}

void EBPedestalClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain01/EBPT 3sum SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain06/EBPT 3sum SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain12/EBPT 3sum SM*");

  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain01/EBPT 5sum SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain06/EBPT 5sum SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain12/EBPT 5sum SM*");

  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM*");

}

void EBPedestalClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBPedestalClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPedestalClient: uncollate" << endl;

    if ( mui_ ) {

      for ( int ism = 1; ism <= 36; ism++ ) {

        mui_->removeCollate(me_h01_[ism-1]);
        mui_->removeCollate(me_h02_[ism-1]);
        mui_->removeCollate(me_h03_[ism-1]);

        mui_->removeCollate(me_j01_[ism-1]);
        mui_->removeCollate(me_j02_[ism-1]);
        mui_->removeCollate(me_j03_[ism-1]);

        mui_->removeCollate(me_k01_[ism-1]);
        mui_->removeCollate(me_k02_[ism-1]);
        mui_->removeCollate(me_k03_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);
        mui_->removeCollate(me_i02_[ism-1]);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT 3sum SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT 3sum SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT 3sum SM*");

  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT 5sum SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT 5sum SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT 5sum SM*");

  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM*");

}

void EBPedestalClient::analyze(void){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBPedestalClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain01/EBPT pedestal SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( h01_[ism-1] ) delete h01_[ism-1];
          sprintf(histo, "ME EBPT pedestal SM%02d G01", ism);
          h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          h01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain06/EBPT pedestal SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( h02_[ism-1] ) delete h02_[ism-1];
          sprintf(histo, "ME EBPT pedestal SM%02d G06", ism);
          h02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          h02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain12/EBPT pedestal SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( h03_[ism-1] ) delete h03_[ism-1];
          sprintf(histo, "ME EBPT pedestal SM%02d G12", ism);
          h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          h03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain01/EBPT 3sum SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain01/EBPT 3sum SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( j01_[ism-1] ) delete j01_[ism-1];
          sprintf(histo, "ME EBPT 3sum SM%02d G01", ism);
          j01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          j01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) { 
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain06/EBPT 3sum SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain06/EBPT 3sum SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( j02_[ism-1] ) delete j02_[ism-1];
          sprintf(histo, "ME EBPT 3sum SM%02d G06", ism);
          j02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          j02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) { 
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain12/EBPT 3sum SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain12/EBPT 3sum SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( j03_[ism-1] ) delete j03_[ism-1];
          sprintf(histo, "ME EBPT 3sum SM%02d G12", ism);
          j03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          j03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain01/EBPT 5sum SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain01/EBPT 5sum SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( k01_[ism-1] ) delete k01_[ism-1];
          sprintf(histo, "ME EBPT 5sum SM%02d G01", ism);
          k01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          k01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain06/EBPT 5sum SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain06/EBPT 5sum SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( k02_[ism-1] ) delete k02_[ism-1];
          sprintf(histo, "ME EBPT 5sum SM%02d G06", ism);
          k02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          k02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain12/EBPT 5sum SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain12/EBPT 5sum SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( k03_[ism-1] ) delete k03_[ism-1];
          sprintf(histo, "ME EBPT 5sum SM%02d G12", ism);
          k03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          k03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( i01_[ism-1] ) delete i01_[ism-1];
          sprintf(histo, "ME EBPDT PNs pedestal SM%02d G01", ism);
          i01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          i01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM%02d G16", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM%02d G16", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( i02_[ism-1] ) delete i02_[ism-1];
          sprintf(histo, "ME EBPDT PNs pedestal SM%02d G16", ism);
          i02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          i02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 50.;

    float num01, num02, num03;
    float mean01, mean02, mean03;
    float rms01, rms02, rms03;

    if ( g01_[ism-1] ) g01_[ism-1]->Reset();
    if ( g02_[ism-1] ) g02_[ism-1]->Reset();
    if ( g03_[ism-1] ) g03_[ism-1]->Reset();

    if ( p01_[ism-1] ) p01_[ism-1]->Reset();
    if ( p02_[ism-1] ) p02_[ism-1]->Reset();
    if ( p03_[ism-1] ) p03_[ism-1]->Reset();

    if ( r01_[ism-1] ) r01_[ism-1]->Reset();
    if ( r02_[ism-1] ) r02_[ism-1]->Reset();
    if ( r03_[ism-1] ) r03_[ism-1]->Reset();

    if ( s01_[ism-1] ) s01_[ism-1]->Reset();
    if ( s02_[ism-1] ) s02_[ism-1]->Reset();
    if ( s03_[ism-1] ) s03_[ism-1]->Reset();

    if ( t01_[ism-1] ) t01_[ism-1]->Reset();
    if ( t02_[ism-1] ) t02_[ism-1]->Reset();
    if ( t03_[ism-1] ) t03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = -1.;
        mean01 = mean02 = mean03 = -1.;
        rms01  = rms02  = rms03  = -1.;

        if ( g01_[ism-1] ) g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);
        if ( g02_[ism-1] ) g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), 2.);
        if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            rms02  = h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          float val;

          val = 1.;
          if ( abs(mean01 - expectedMean_[0]) > discrepancyMean_[0] )
            val = 0.;
          if ( rms01 > RMSThreshold_[0] )
            val = 0.;
          if ( g01_[ism-1] ) g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);

          if ( p01_[ism-1] ) p01_[ism-1]->Fill(mean01);
          if ( r01_[ism-1] ) r01_[ism-1]->Fill(rms01);

          val = 1.;
          if ( abs(mean02 - expectedMean_[1]) > discrepancyMean_[1] )
            val = 0.;
          if ( rms02 > RMSThreshold_[1] )
            val = 0.;
          if ( g02_[ism-1] ) g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), val);

          if ( p02_[ism-1] ) p02_[ism-1]->Fill(mean02);
          if ( r02_[ism-1] ) r02_[ism-1]->Fill(rms02);

          val = 1.;
          if ( abs(mean03 - expectedMean_[2]) > discrepancyMean_[2] )
            val = 0.;
          if ( rms03 > RMSThreshold_[2] )
            val = 0.;
          if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);

          if ( p03_[ism-1] ) p03_[ism-1]->Fill(mean03);
          if ( r03_[ism-1] ) r03_[ism-1]->Fill(rms03);

        }

        float xval;
        float yval;
        float zval;

        if ( ie >= 2 && ie <= 84 && ip >= 2 && ip <= 19 ) {

          xval = 0.;
          if ( j01_[ism-1] ) xval = j01_[ism-1]->GetBinError(j01_[ism-1]->GetBin(ie, ip)) *
                                    j01_[ism-1]->GetBinError(j01_[ism-1]->GetBin(ie, ip));

          yval = 0.;
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {

              if ( h01_[ism-1] ) yval = yval + h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie+i, ip+j)) *
                                               h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie+i, ip+j));

            }
          }

          zval = 0.;
          if ( xval != 0 && yval != 0 ) zval = (xval - yval) / yval;

          if ( s01_[ism-1] ) s01_[ism-1]->SetBinContent(s01_[ism-1]->GetBin(ie, ip), zval);

          xval = 0.;
          if ( j02_[ism-1] ) xval = j02_[ism-1]->GetBinError(j02_[ism-1]->GetBin(ie, ip)) *
                                    j02_[ism-1]->GetBinError(j02_[ism-1]->GetBin(ie, ip));

          yval = 0.; 
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {
 
              if ( h02_[ism-1] ) yval = yval + h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie+i, ip+j)) *
                                               h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie+i, ip+j));
 
            } 
          } 

          zval = 0.;
          if ( xval != 0 && yval != 0 ) zval = (xval - yval) / yval;
          
          if ( s02_[ism-1] ) s02_[ism-1]->SetBinContent(s02_[ism-1]->GetBin(ie, ip), zval);

          xval = 0.;
          if ( j03_[ism-1] ) xval = j03_[ism-1]->GetBinError(j03_[ism-1]->GetBin(ie, ip)) *
                                    j03_[ism-1]->GetBinError(j03_[ism-1]->GetBin(ie, ip));

          yval = 0.; 
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {
 
              if ( h03_[ism-1] ) yval = yval + h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie+i, ip+j)) *
                                               h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie+i, ip+j));
 
            } 
          } 

          zval = 0.;
          if ( xval != 0 && yval != 0 ) zval = (xval - yval) / yval;
          
          if ( s03_[ism-1] ) s03_[ism-1]->SetBinContent(s03_[ism-1]->GetBin(ie, ip), zval);

        }

        if ( ie >= 3 && ie <= 83 && ip >= 3 && ip <= 18 ) {

          xval = 0.;
          if ( k01_[ism-1] ) xval = k01_[ism-1]->GetBinError(k01_[ism-1]->GetBin(ie, ip)) *
                                    k01_[ism-1]->GetBinError(k01_[ism-1]->GetBin(ie, ip));

          yval = 0.;
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {

              if ( h01_[ism-1] ) yval = yval + h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie+i, ip+j)) *
                                               h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie+i, ip+j));

            }
          }

          zval = 0.;
          if ( xval != 0 && yval != 0 ) zval = (xval - yval) / yval;

          if ( t01_[ism-1] ) t01_[ism-1]->SetBinContent(t01_[ism-1]->GetBin(ie, ip), zval);

          xval = 0.;
          if ( k02_[ism-1] ) xval = k02_[ism-1]->GetBinError(k02_[ism-1]->GetBin(ie, ip)) *
                                    k02_[ism-1]->GetBinError(k02_[ism-1]->GetBin(ie, ip));

          yval = 0.; 
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {
 
              if ( h02_[ism-1] ) yval = yval + h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie+i, ip+j)) *
                                               h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(ie+i, ip+j));
 
            } 
          } 

          zval = 0.;
          if ( xval != 0 && yval != 0 ) zval = (xval - yval) / yval;
          
          if ( t02_[ism-1] ) t02_[ism-1]->SetBinContent(t02_[ism-1]->GetBin(ie, ip), zval);

          xval = 0.;
          if ( k03_[ism-1] ) xval = k03_[ism-1]->GetBinError(k03_[ism-1]->GetBin(ie, ip)) *
                                    k03_[ism-1]->GetBinError(k03_[ism-1]->GetBin(ie, ip));

          yval = 0.; 
          for ( int i = -1; i <= +1; i++ ) {
            for ( int j = -1; j <= +1; j++ ) {
 
              if ( h03_[ism-1] ) yval = yval + h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie+i, ip+j)) *
                                               h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(ie+i, ip+j));
 
            } 
          } 

          zval = 0.;
          if ( xval != 0 && yval != 0 ) zval = (xval - yval) / yval;
          
          if ( t03_[ism-1] ) t03_[ism-1]->SetBinContent(t03_[ism-1]->GetBin(ie, ip), zval);

        }

      }
    }

  }

}

void EBPedestalClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBPedestalClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PedestalTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PEDESTAL</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol3[3] = { 2, 3, 5 };

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 30+i;

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);

  string imgNameQual[3], imgNameMean[3], imgNameRMS[3], imgName3Sum[3], imgName5Sum[3], imgNameMEPnPed[2], imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cMean = new TCanvas("cMean", "Temp", csize, csize);
  TCanvas* cRMS = new TCanvas("cRMS", "Temp", csize, csize);
  TCanvas* c3Sum = new TCanvas("c3Sum", "Temp", 2*csize, csize);
  TCanvas* c5Sum = new TCanvas("c5Sum", "Temp", 2*csize, csize);
  TCanvas* cPed = new TCanvas("cPed", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TH1D* obj1d;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    // Loop on gains

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      // Quality plots

      imgNameQual[iCanvas-1] = "";

      obj2f = 0;
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

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
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
        cQual->SaveAs(imgName.c_str());

      }

      // Mean distributions

      imgNameMean[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = p01_[ism-1];
          break;
        case 2:
          obj1f = p02_[ism-1];
          break;
        case 3:
          obj1f = p03_[ism-1];
          break;
        default:
            break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMean[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMean[iCanvas-1];

        cMean->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cMean->Update();
        gPad->SetLogy(0);
        cMean->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // RMS distributions

      imgNameRMS[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = r01_[ism-1];
          break;
        case 2:
          obj1f = r02_[ism-1];
          break;
        case 3:
          obj1f = r03_[ism-1];
          break;
        default:
          break;
      }

      if ( obj1f ) {

        meName = obj1f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameRMS[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameRMS[iCanvas-1];

        cRMS->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
        if ( obj1f->GetMaximum(histMax) > 0. ) {
          gPad->SetLogy(1);
        } else {
          gPad->SetLogy(0);
        }
        obj1f->Draw();
        cRMS->Update();
        cRMS->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // 3Sum distributions

      imgName3Sum[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = s01_[ism-1];
          break;
        case 2:
          obj2f = s02_[ism-1];
          break;
        case 3:
          obj2f = s03_[ism-1];
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgName3Sum[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgName3Sum[iCanvas-1];

        c3Sum->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        c3Sum->SetGridx();
        c3Sum->SetGridy();
        obj2f->Draw("colz");
        dummy.Draw("text,same");
        c3Sum->Update();
        c3Sum->SaveAs(imgName.c_str());

      }

      // 5Sum distributions

      imgName5Sum[iCanvas-1] = "";

      obj2f = 0;
      switch ( iCanvas ) {
        case 1:
          obj2f = t01_[ism-1];
          break;
        case 2:
          obj2f = t02_[ism-1];
          break;
        case 3:
          obj2f = t03_[ism-1];
          break;
        default:
          break;
      }

      if ( obj2f ) {

        meName = obj2f->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgName5Sum[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgName5Sum[iCanvas-1];

        c5Sum->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        c5Sum->SetGridx();
        c5Sum->SetGridy();
        obj2f->Draw("colz");
        dummy.Draw("text,same");
        c5Sum->Update();
        c5Sum->SaveAs(imgName.c_str());

      }

    }

    // Loop on gains

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      // Monitoring elements plots

      imgNameMEPnPed[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i01_[ism-1] ) obj1d = i01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i02_[ism-1] ) obj1d = i02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameMEPnPed[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPed[iCanvas-1];

        cPed->cd();
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.);
        obj1d->Draw();
        cPed->Update();
        cPed->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

    }

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgNameQual[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "<tr>" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgNameMean[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameMean[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameRMS[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameRMS[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td colspan=\"2\">Gain 1</td><td colspan=\"2\">Gain 6</td><td colspan=\"2\">Gain 12</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgName3Sum[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgName3Sum[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 3 ; iCanvas++ ) {

      if ( imgName5Sum[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgName5Sum[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td>Gain 1</td><td>Gain 6</td><td>Gain 12</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameMEPnPed[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPed[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td colspan=\"2\">Gain 1</td><td colspan=\"2\">Gain 16</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cMean;
  delete cRMS;
  delete c3Sum;
  delete c5Sum;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

