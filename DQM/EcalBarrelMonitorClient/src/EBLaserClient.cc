/*
 * \file EBLaserClient.cc
 *
 * $Date: 2006/01/11 09:37:03 $
 * $Revision: 1.54 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>

EBLaserClient::EBLaserClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  for ( int ism = 1; ism <= 36; ism++ ) {

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;
    h07_[ism-1] = 0;
    h08_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;
    i05_[ism-1] = 0;
    i06_[ism-1] = 0;
    i07_[ism-1] = 0;
    i08_[ism-1] = 0;

    j01_[ism-1] = 0;
    j02_[ism-1] = 0;
    j03_[ism-1] = 0;
    j04_[ism-1] = 0;
    j05_[ism-1] = 0;
    j06_[ism-1] = 0;
    j07_[ism-1] = 0;
    j08_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g01_[ism-1] = 0;
    g02_[ism-1] = 0;
    g03_[ism-1] = 0;
    g04_[ism-1] = 0;

    a01_[ism-1] = 0;
    a02_[ism-1] = 0;
    a03_[ism-1] = 0;
    a04_[ism-1] = 0;

    aopn01_[ism-1] = 0;
    aopn02_[ism-1] = 0;
    aopn03_[ism-1] = 0;
    aopn04_[ism-1] = 0;

  }

  percentVariation_ = 0.4;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBLaserClient::~EBLaserClient(){

  this->cleanup();

}

void EBLaserClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBLaserClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBLaserClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBLaserClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBLaserClient::endJob(void) {

  if ( verbose_ ) cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

}

void EBLaserClient::endRun(void) {

  if ( verbose_ ) cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBLaserClient::setup(void) {

  Char_t histo[50];

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    sprintf(histo, "EBLT laser quality L1 SM%02d", ism);
    g01_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g02_[ism-1] ) delete g02_[ism-1];
    sprintf(histo, "EBLT laser quality L2 SM%02d", ism);
    g02_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g03_[ism-1] ) delete g03_[ism-1];
    sprintf(histo, "EBLT laser quality L3 SM%02d", ism);
    g03_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g04_[ism-1] ) delete g04_[ism-1];
    sprintf(histo, "EBLT laser quality L4 SM%02d", ism);
    g04_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( a01_[ism-1] ) delete a01_[ism-1];
    sprintf(histo, "EBLT laser amplitude L1 SM%02d", ism);
    a01_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( a02_[ism-1] ) delete a02_[ism-1];
    sprintf(histo, "EBLT laser amplitude L2 SM%02d", ism);
    a02_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( a03_[ism-1] ) delete a03_[ism-1];
    sprintf(histo, "EBLT laser amplitude L3 SM%02d", ism);
    a03_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( a04_[ism-1] ) delete a04_[ism-1];
    sprintf(histo, "EBLT laser amplitude L4 SM%02d", ism);
    a04_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);

    if ( aopn01_[ism-1] ) delete aopn01_[ism-1];
    sprintf(histo, "EBLT laser amplitude over PN L1 SM%02d", ism);
    aopn01_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( aopn02_[ism-1] ) delete aopn02_[ism-1];
    sprintf(histo, "EBLT laser amplitude over PN L2 SM%02d", ism);
    aopn02_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( aopn03_[ism-1] ) delete aopn03_[ism-1];
    sprintf(histo, "EBLT laser amplitude over PN L3 SM%02d", ism);
    aopn03_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( aopn04_[ism-1] ) delete aopn04_[ism-1];
    sprintf(histo, "EBLT laser amplitude over PN L4 SM%02d", ism);
    aopn04_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g01_[ism-1]->Reset();
    g02_[ism-1]->Reset();
    g03_[ism-1]->Reset();
    g04_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);
        g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), 2.);
        g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);
        g04_[ism-1]->SetBinContent(g04_[ism-1]->GetBin(ie, ip), 2.);

      }
    }

    a01_[ism-1]->Reset();
    a02_[ism-1]->Reset();
    a03_[ism-1]->Reset();
    a04_[ism-1]->Reset();

    aopn01_[ism-1]->Reset();
    aopn02_[ism-1]->Reset();
    aopn03_[ism-1]->Reset();
    aopn04_[ism-1]->Reset();

  }

}

void EBLaserClient::cleanup(void) {

  for ( int ism = 1; ism <= 36; ism++ ) {

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
    if ( h07_[ism-1] ) delete h07_[ism-1];
    h07_[ism-1] = 0;
    if ( h08_[ism-1] ) delete h08_[ism-1];
    h08_[ism-1] = 0;

    if ( i01_[ism-1] ) delete i01_[ism-1];
    i01_[ism-1] = 0;
    if ( i02_[ism-1] ) delete i02_[ism-1];
    i02_[ism-1] = 0;
    if ( i03_[ism-1] ) delete i03_[ism-1];
    i03_[ism-1] = 0;
    if ( i04_[ism-1] ) delete i04_[ism-1];
    i04_[ism-1] = 0;
    if ( i05_[ism-1] ) delete i05_[ism-1];
    i05_[ism-1] = 0;
    if ( i06_[ism-1] ) delete i06_[ism-1];
    i06_[ism-1] = 0;
    if ( i07_[ism-1] ) delete i07_[ism-1];
    i07_[ism-1] = 0;
    if ( i08_[ism-1] ) delete i08_[ism-1];
    i08_[ism-1] = 0;

    if ( j01_[ism-1] ) delete j01_[ism-1];
    j01_[ism-1] = 0;
    if ( j02_[ism-1] ) delete j02_[ism-1];
    j02_[ism-1] = 0;
    if ( j03_[ism-1] ) delete j03_[ism-1];
    j03_[ism-1] = 0;
    if ( j04_[ism-1] ) delete j04_[ism-1];
    j04_[ism-1] = 0;
    if ( j05_[ism-1] ) delete j05_[ism-1];
    j05_[ism-1] = 0;
    if ( j06_[ism-1] ) delete j06_[ism-1];
    j06_[ism-1] = 0;
    if ( j07_[ism-1] ) delete j07_[ism-1];
    j07_[ism-1] = 0;
    if ( j08_[ism-1] ) delete j08_[ism-1];
    j08_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g01_[ism-1] ) delete g01_[ism-1];
    g01_[ism-1] = 0;
    if ( g02_[ism-1] ) delete g02_[ism-1];
    g02_[ism-1] = 0;
    if ( g03_[ism-1] ) delete g03_[ism-1];
    g03_[ism-1] = 0;
    if ( g04_[ism-1] ) delete g04_[ism-1];
    g04_[ism-1] = 0;

    if ( a01_[ism-1] ) delete a01_[ism-1];
    a01_[ism-1] = 0;
    if ( a02_[ism-1] ) delete a02_[ism-1];
    a02_[ism-1] = 0;
    if ( a03_[ism-1] ) delete a03_[ism-1];
    a03_[ism-1] = 0;
    if ( a04_[ism-1] ) delete a04_[ism-1];
    a04_[ism-1] = 0;

    if ( aopn01_[ism-1] ) delete aopn01_[ism-1];
    aopn01_[ism-1] = 0;
    if ( aopn02_[ism-1] ) delete aopn02_[ism-1];
    aopn02_[ism-1] = 0;
    if ( aopn03_[ism-1] ) delete aopn03_[ism-1];
    aopn03_[ism-1] = 0;
    if ( aopn04_[ism-1] ) delete aopn04_[ism-1];
    aopn04_[ism-1] = 0;

  }

}

void EBLaserClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov) {

  EcalLogicID ecid;
  MonLaserBlueDat apd_bl;
  map<EcalLogicID, MonLaserBlueDat> dataset1_bl;
  MonLaserGreenDat apd_gr;
  map<EcalLogicID, MonLaserGreenDat> dataset1_gr;
  MonLaserIRedDat apd_ir;
  map<EcalLogicID, MonLaserIRedDat> dataset1_ir;
  MonLaserRedDat apd_rd;
  map<EcalLogicID, MonLaserRedDat> dataset1_rd;

  cout << "Creating MonLaserDatObjects for the database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04, num05, num06, num07, num08;
    float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
    float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = num04  = num05  = num06  = num07  = num08  = -1.;
        mean01 = mean02 = mean03 = mean04 = mean05 = mean06 = mean07 = mean08 = -1.;
        rms01  = rms02  = rms03  = rms04  = rms05  = rms06  = rms07  = rms08  = -1.;

        bool update_channel1 = false;
        bool update_channel2 = false;
        bool update_channel3 = false;
        bool update_channel4 = false;

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

        if ( h05_[ism-1] && h05_[ism-1]->GetEntries() >= n_min_tot ) {
          num05 = h05_[ism-1]->GetBinEntries(h05_[ism-1]->GetBin(ie, ip));
          if ( num05 >= n_min_bin ) {
            mean05 = h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(ie, ip));
            rms05  = h05_[ism-1]->GetBinError(h05_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h06_[ism-1] && h06_[ism-1]->GetEntries() >= n_min_tot ) {
          num06 = h06_[ism-1]->GetBinEntries(h06_[ism-1]->GetBin(ie, ip));
          if ( num06 >= n_min_bin ) {
            mean06 = h06_[ism-1]->GetBinContent(h06_[ism-1]->GetBin(ie, ip));
            rms06  = h06_[ism-1]->GetBinError(h06_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h07_[ism-1] && h07_[ism-1]->GetEntries() >= n_min_tot ) {
          num07 = h07_[ism-1]->GetBinEntries(h07_[ism-1]->GetBin(ie, ip));
          if ( num07 >= n_min_bin ) {
            mean07 = h07_[ism-1]->GetBinContent(h07_[ism-1]->GetBin(ie, ip));
            rms07  = h07_[ism-1]->GetBinError(h07_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( h08_[ism-1] && h08_[ism-1]->GetEntries() >= n_min_tot ) {
          num08 = h08_[ism-1]->GetBinEntries(h08_[ism-1]->GetBin(ie, ip));
          if ( num08 >= n_min_bin ) {
            mean08 = h08_[ism-1]->GetBinContent(h08_[ism-1]->GetBin(ie, ip));
            rms08  = h08_[ism-1]->GetBinError(h08_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( update_channel1 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "L1 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;

          }

          apd_bl.setAPDMean(mean01);
          apd_bl.setAPDRMS(rms01);

          apd_bl.setAPDOverPNMean(mean02);
          apd_bl.setAPDOverPNRMS(rms02);

          if ( g01_[ism-1]->GetBinContent(g01_[ism-1]->GetBin(ie, ip)) == 1. ) {
            apd_bl.setTaskStatus(true);
          } else {
            apd_bl.setTaskStatus(false);
          }

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
              dataset1_bl[ecid] = apd_bl;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

        if ( update_channel2 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "L2 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;

          }

          apd_ir.setAPDMean(mean03);
          apd_ir.setAPDRMS(rms03);

          apd_ir.setAPDOverPNMean(mean04);
          apd_ir.setAPDOverPNRMS(rms04);

          if ( g02_[ism-1]->GetBinContent(g02_[ism-1]->GetBin(ie, ip)) == 1. ) {
            apd_ir.setTaskStatus(true);
          } else {
            apd_ir.setTaskStatus(false);
          }

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
              dataset1_ir[ecid] = apd_ir;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

        if ( update_channel3 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "L3 (" << ie << "," << ip << ") " << num05 << " " << mean05 << " " << rms05 << endl;

          }

          apd_gr.setAPDMean(mean05);
          apd_gr.setAPDRMS(rms05);

          apd_gr.setAPDOverPNMean(mean06);
          apd_gr.setAPDOverPNRMS(rms06);

          if ( g03_[ism-1]->GetBinContent(g03_[ism-1]->GetBin(ie, ip)) == 1. ) {
            apd_gr.setTaskStatus(true);
          } else {
            apd_gr.setTaskStatus(false);
          }

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
              dataset1_gr[ecid] = apd_gr;
            } catch (runtime_error &e) {
              cerr << e.what() << endl;
            }
          }

        }

        if ( update_channel4 ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "L4 (" << ie << "," << ip << ") " << num07 << " " << mean07 << " " << rms07 << endl;

          }

          apd_rd.setAPDMean(mean07);
          apd_rd.setAPDRMS(rms07);

          apd_rd.setAPDOverPNMean(mean08);
          apd_rd.setAPDOverPNRMS(rms08);

          if ( g04_[ism-1]->GetBinContent(g04_[ism-1]->GetBin(ie, ip)) == 1. ) {
            apd_rd.setTaskStatus(true);
          } else {
            apd_rd.setTaskStatus(false);
          }

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
              dataset1_rd[ecid] = apd_rd;
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
      if ( dataset1_bl.size() != 0 ) econn->insertDataSet(&dataset1_bl, moniov);
      if ( dataset1_ir.size() != 0 ) econn->insertDataSet(&dataset1_ir, moniov);
      if ( dataset1_gr.size() != 0 ) econn->insertDataSet(&dataset1_gr, moniov);
      if ( dataset1_rd.size() != 0 ) econn->insertDataSet(&dataset1_rd, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  MonPNBlueDat pn_bl;
  map<EcalLogicID, MonPNBlueDat> dataset2_bl;
  MonPNGreenDat pn_gr;
  map<EcalLogicID, MonPNGreenDat> dataset2_gr;
  MonPNIRedDat pn_ir;
  map<EcalLogicID, MonPNIRedDat> dataset2_ir;
  MonPNRedDat pn_rd;
  map<EcalLogicID, MonPNRedDat> dataset2_rd;

  cout << "Creating MonPnDatObjects for the database ..." << endl;

  const float m_min_tot = 1000.;
  const float m_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04, num05, num06, num07, num08;
    float num09, num10, num11, num12, num13, num14, num15, num16;
    float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
    float mean09, mean10, mean11, mean12, mean13, mean14, mean15, mean16;
    float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
    float rms09, rms10, rms11, rms12, rms13, rms14, rms15, rms16;

    for ( int i = 1; i <= 10; i++ ) {

      num01  = num02  = num03  = num04  = num05  = num06  = num07  = num08  = -1.;
      num09  = num10  = num11  = num12  = num13  = num14  = num15  = num16  = -1.;
      mean01 = mean02 = mean03 = mean04 = mean05 = mean06 = mean07 = mean08 = -1.;
      mean09 = mean10 = mean11 = mean12 = mean13 = mean14 = mean15 = mean16 = -1.;
      rms01  = rms02  = rms03  = rms04  = rms05  = rms06  = rms07  = rms08  = -1.;
      rms09  = rms10  = rms11  = rms12  = rms13  = rms14  = rms15  = rms16  = -1.;

      bool update_channel1 = false;
      bool update_channel2 = false;
      bool update_channel3 = false;
      bool update_channel4 = false;
      bool update_channel5 = false;
      bool update_channel6 = false;
      bool update_channel7 = false;
      bool update_channel8 = false;

      if ( i01_[ism-1] && i01_[ism-1]->GetEntries() >= m_min_tot ) {
        num01 = i01_[ism-1]->GetBinEntries(i01_[ism-1]->GetBin(1, i));
        if ( num01 >= m_min_bin ) {
          mean01 = i01_[ism-1]->GetBinContent(i01_[ism-1]->GetBin(1, i));
          rms01  = i01_[ism-1]->GetBinError(i01_[ism-1]->GetBin(1, i));
          update_channel1 = true;
        }
      }

      if ( i02_[ism-1] && i02_[ism-1]->GetEntries() >= m_min_tot ) {
        num02 = i02_[ism-1]->GetBinEntries(i02_[ism-1]->GetBin(1, i));
        if ( num02 >= m_min_bin ) {
          mean02 = i02_[ism-1]->GetBinContent(i02_[ism-1]->GetBin(1, i));
          rms02  = i02_[ism-1]->GetBinError(i02_[ism-1]->GetBin(1, i));
          update_channel2 = true;
        }
      }

      if ( i03_[ism-1] && i03_[ism-1]->GetEntries() >= m_min_tot ) {
        num03 = i03_[ism-1]->GetBinEntries(i03_[ism-1]->GetBin(i));
        if ( num03 >= m_min_bin ) {
          mean03 = i03_[ism-1]->GetBinContent(i03_[ism-1]->GetBin(1, i));
          rms03  = i03_[ism-1]->GetBinError(i03_[ism-1]->GetBin(1, i));
          update_channel3 = true;
        }
      }

      if ( i04_[ism-1] && i04_[ism-1]->GetEntries() >= m_min_tot ) {
        num04 = i04_[ism-1]->GetBinEntries(i04_[ism-1]->GetBin(1, i));
        if ( num04 >= m_min_bin ) {
          mean04 = i04_[ism-1]->GetBinContent(i04_[ism-1]->GetBin(1, i));
          rms04  = i04_[ism-1]->GetBinError(i04_[ism-1]->GetBin(1, i));
          update_channel4 = true;
        }
      }

      if ( i05_[ism-1] && i05_[ism-1]->GetEntries() >= m_min_tot ) {
        num05 = i05_[ism-1]->GetBinEntries(i05_[ism-1]->GetBin(1, i));
        if ( num05 >= m_min_bin ) {
          mean05 = i05_[ism-1]->GetBinContent(i05_[ism-1]->GetBin(1, i));
          rms05  = i05_[ism-1]->GetBinError(i05_[ism-1]->GetBin(1, i));
          update_channel5 = true;
        }
      }
      if ( i06_[ism-1] && i06_[ism-1]->GetEntries() >= m_min_tot ) {
        num06 = i06_[ism-1]->GetBinEntries(i06_[ism-1]->GetBin(1, i));
        if ( num06 >= m_min_bin ) {
          mean06 = i06_[ism-1]->GetBinContent(i06_[ism-1]->GetBin(1, i));
          rms06  = i06_[ism-1]->GetBinError(i06_[ism-1]->GetBin(1, i));
          update_channel6 = true;
        }
      }

      if ( i07_[ism-1] && i07_[ism-1]->GetEntries() >= m_min_tot ) {
        num07 = i07_[ism-1]->GetBinEntries(i07_[ism-1]->GetBin(1, i));
        if ( num07 >= m_min_bin ) {
          mean07 = i07_[ism-1]->GetBinContent(i07_[ism-1]->GetBin(1, i));
          rms07  = i07_[ism-1]->GetBinError(i07_[ism-1]->GetBin(1, i));
          update_channel7 = true;
        }
      }

      if ( i08_[ism-1] && i08_[ism-1]->GetEntries() >= m_min_tot ) {
        num08 = i08_[ism-1]->GetBinEntries(i08_[ism-1]->GetBin(1, i));
        if ( num08 >= m_min_bin ) {
          mean08 = i08_[ism-1]->GetBinContent(i08_[ism-1]->GetBin(1, i));
          rms08  = i08_[ism-1]->GetBinError(i08_[ism-1]->GetBin(1, i));
          update_channel8 = true;
        }
      }

      if ( j01_[ism-1] && j01_[ism-1]->GetEntries() >= m_min_tot ) {
        num09 = j01_[ism-1]->GetBinEntries(j01_[ism-1]->GetBin(1, i));
        if ( num09 >= m_min_bin ) {
          mean09 = j01_[ism-1]->GetBinContent(j01_[ism-1]->GetBin(1, i));
          rms09  = j01_[ism-1]->GetBinError(j01_[ism-1]->GetBin(1, i));
          update_channel1 = true;
        }
      }

      if ( j02_[ism-1] && j02_[ism-1]->GetEntries() >= m_min_tot ) {
        num10 = j02_[ism-1]->GetBinEntries(j02_[ism-1]->GetBin(1, i));
        if ( num10 >= m_min_bin ) {
          mean10 = j02_[ism-1]->GetBinContent(j02_[ism-1]->GetBin(1, i));
          rms10  = j02_[ism-1]->GetBinError(j02_[ism-1]->GetBin(1, i));
          update_channel2 = true;
        }
      }

      if ( j03_[ism-1] && j03_[ism-1]->GetEntries() >= m_min_tot ) {
        num11 = j03_[ism-1]->GetBinEntries(j03_[ism-1]->GetBin(i));
        if ( num11 >= m_min_bin ) {
          mean11 = j03_[ism-1]->GetBinContent(j03_[ism-1]->GetBin(1, i));
          rms11  = j03_[ism-1]->GetBinError(j03_[ism-1]->GetBin(1, i));
          update_channel3 = true;
        }
      }

      if ( j04_[ism-1] && j04_[ism-1]->GetEntries() >= m_min_tot ) {
        num12 = j04_[ism-1]->GetBinEntries(j04_[ism-1]->GetBin(1, i));
        if ( num12 >= m_min_bin ) {
          mean12 = j04_[ism-1]->GetBinContent(j04_[ism-1]->GetBin(1, i));
          rms12  = j04_[ism-1]->GetBinError(j04_[ism-1]->GetBin(1, i));
          update_channel4 = true;
        }
      }

      if ( j05_[ism-1] && j05_[ism-1]->GetEntries() >= m_min_tot ) {
        num13 = j05_[ism-1]->GetBinEntries(j05_[ism-1]->GetBin(1, i));
        if ( num13 >= m_min_bin ) {
          mean13 = j05_[ism-1]->GetBinContent(j05_[ism-1]->GetBin(1, i));
          rms13  = j05_[ism-1]->GetBinError(j05_[ism-1]->GetBin(1, i));
          update_channel5 = true;
        }
      }
      if ( j06_[ism-1] && j06_[ism-1]->GetEntries() >= m_min_tot ) {
        num14 = j06_[ism-1]->GetBinEntries(j06_[ism-1]->GetBin(1, i));
        if ( num14 >= m_min_bin ) {
          mean14 = j06_[ism-1]->GetBinContent(j06_[ism-1]->GetBin(1, i));
          rms14  = j06_[ism-1]->GetBinError(j06_[ism-1]->GetBin(1, i));
          update_channel6 = true;
        }
      }

      if ( j07_[ism-1] && j07_[ism-1]->GetEntries() >= m_min_tot ) {
        num15 = j07_[ism-1]->GetBinEntries(j07_[ism-1]->GetBin(1, i));
        if ( num15 >= m_min_bin ) {
          mean15 = j07_[ism-1]->GetBinContent(j07_[ism-1]->GetBin(1, i));
          rms15  = j07_[ism-1]->GetBinError(j07_[ism-1]->GetBin(1, i));
          update_channel7 = true;
        }
      }

      if ( j08_[ism-1] && j08_[ism-1]->GetEntries() >= m_min_tot ) {
        num16 = j08_[ism-1]->GetBinEntries(j08_[ism-1]->GetBin(1, i));
        if ( num16 >= m_min_bin ) {
          mean16 = j08_[ism-1]->GetBinContent(j08_[ism-1]->GetBin(1, i));
          rms16  = j08_[ism-1]->GetBinError(j08_[ism-1]->GetBin(1, i));
          update_channel8 = true;
        }
      }

      if ( update_channel1 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") L1 G01 " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "PNs (" << i << ") L1 G16 " << num09  << " " << mean09 << " " << rms09  << endl;

        }

        pn_bl.setADCMeanG1(mean01);
        pn_bl.setADCRMSG1(rms01);

        pn_bl.setPedMeanG1(mean05);
        pn_bl.setPedRMSG1(rms05);

        pn_bl.setADCMeanG16(mean09);
        pn_bl.setADCRMSG16(rms09);

        pn_bl.setPedMeanG16(mean13);
        pn_bl.setPedRMSG16(rms13);

        pn_bl.setTaskStatus(true);

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset2_bl[ecid] = pn_bl;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

      if ( update_channel2 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") L2 G01 " << num02  << " " << mean02 << " " << rms02  << endl;
          cout << "PNs (" << i << ") L2 G16 " << num10  << " " << mean10 << " " << rms10  << endl;

        }

        pn_ir.setADCMeanG1(mean02);
        pn_ir.setADCRMSG1(rms02);

        pn_ir.setPedMeanG1(mean06);
        pn_ir.setPedRMSG1(rms06);

        pn_ir.setADCMeanG16(mean10);
        pn_ir.setADCRMSG16(rms10);

        pn_ir.setPedMeanG16(mean14);
        pn_ir.setPedRMSG16(rms14);

        pn_ir.setTaskStatus(true);

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset2_ir[ecid] = pn_ir;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

      if ( update_channel3 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") L3 G01 " << num03  << " " << mean03 << " " << rms03  << endl;
          cout << "PNs (" << i << ") L3 G16 " << num11  << " " << mean11 << " " << rms11  << endl;

        }

        pn_gr.setADCMeanG1(mean03);
        pn_gr.setADCRMSG1(rms03);

        pn_gr.setPedMeanG1(mean07);
        pn_gr.setPedRMSG1(rms07);

        pn_gr.setADCMeanG16(mean11);
        pn_gr.setADCRMSG16(rms11);

        pn_gr.setPedMeanG16(mean15);
        pn_gr.setPedRMSG16(rms15);

        pn_gr.setTaskStatus(true);

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset2_gr[ecid] = pn_gr;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

      if ( update_channel4 ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") L4 G01 " << num04  << " " << mean04 << " " << rms04  << endl;
          cout << "PNs (" << i << ") L4 G16 " << num12  << " " << mean12 << " " << rms12  << endl;

        }

        pn_rd.setADCMeanG1(mean04);
        pn_rd.setADCRMSG1(rms04);

        pn_rd.setPedMeanG1(mean08);
        pn_rd.setPedRMSG1(mean08);

        pn_rd.setADCMeanG16(mean12);
        pn_rd.setADCRMSG16(rms12);

        pn_rd.setPedMeanG16(mean16);
        pn_rd.setPedRMSG16(rms16);

        pn_rd.setTaskStatus(true);

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset2_rd[ecid] = pn_rd;
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
      if ( dataset2_bl.size() != 0 ) econn->insertDataSet(&dataset2_bl, moniov);
      if ( dataset2_ir.size() != 0 ) econn->insertDataSet(&dataset2_ir, moniov);
      if ( dataset2_gr.size() != 0 ) econn->insertDataSet(&dataset2_gr, moniov);
      if ( dataset2_rd.size() != 0 ) econn->insertDataSet(&dataset2_rd, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

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

  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM*");

  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM*");

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

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L1", ism);
      me_i01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L2", ism);
      me_i02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
      mui_->add(me_i02_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L3", ism);
      me_i03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
      mui_->add(me_i03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L4", ism);
      me_i04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
      mui_->add(me_i04_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L1", ism);
      me_i05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
      mui_->add(me_i05_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L2", ism);
      me_i06_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
      mui_->add(me_i06_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L3", ism);
      me_i07_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
      mui_->add(me_i07_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L4", ism);
      me_i08_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
      mui_->add(me_i08_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L1", ism);
      me_j01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
      mui_->add(me_j01_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L2", ism);
      me_j02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
      mui_->add(me_j02_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L3", ism);
      me_j03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
      mui_->add(me_j03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L4", ism);
      me_j04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
      mui_->add(me_j04_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L1", ism);
      me_j05_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
      mui_->add(me_j05_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L2", ism);
      me_j06_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
      mui_->add(me_j06_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L3", ism);
      me_j07_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
      mui_->add(me_j07_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L4", ism);
      me_j08_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
      mui_->add(me_j08_[ism-1], histo);

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

  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM*");

  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM*");

}

void EBLaserClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBLaserClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBLaserClient: uncollate" << endl;

    if ( mui_ ) {

      for ( int ism = 1; ism <= 36; ism++ ) {

        mui_->removeCollate(me_h01_[ism-1]);
        mui_->removeCollate(me_h02_[ism-1]);
        mui_->removeCollate(me_h03_[ism-1]);
        mui_->removeCollate(me_h04_[ism-1]);
        mui_->removeCollate(me_h05_[ism-1]);
        mui_->removeCollate(me_h06_[ism-1]);
        mui_->removeCollate(me_h07_[ism-1]);
        mui_->removeCollate(me_h08_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);
        mui_->removeCollate(me_i02_[ism-1]);
        mui_->removeCollate(me_i03_[ism-1]);
        mui_->removeCollate(me_i04_[ism-1]);
        mui_->removeCollate(me_i05_[ism-1]);
        mui_->removeCollate(me_i06_[ism-1]);
        mui_->removeCollate(me_i07_[ism-1]);
        mui_->removeCollate(me_i08_[ism-1]);

        mui_->removeCollate(me_j01_[ism-1]);
        mui_->removeCollate(me_j02_[ism-1]);
        mui_->removeCollate(me_j03_[ism-1]);
        mui_->removeCollate(me_j04_[ism-1]);
        mui_->removeCollate(me_j05_[ism-1]);
        mui_->removeCollate(me_j06_[ism-1]);
        mui_->removeCollate(me_j07_[ism-1]);
        mui_->removeCollate(me_j08_[ism-1]);

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

  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM*");

  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM*");

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

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs amplitude SM%02d G01 L1", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i01_[ism-1] ) delete i01_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G01 L1", ism);
        i01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs amplitude SM%02d G01 L2", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i02_[ism-1] ) delete i02_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G01 L2", ism);
        i02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs amplitude SM%02d G01 L3", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i03_[ism-1] ) delete i03_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G01 L3", ism);
        i03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs amplitude SM%02d G01 L4", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i04_[ism-1] ) delete i04_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G01 L4", ism);
        i04_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i04_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser1/Gain01/EBPDT PNs pedestal SM%02d G01 L1", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i05_[ism-1] ) delete i05_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G01 L1", ism);
        i05_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i05_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser2/Gain01/EBPDT PNs pedestal SM%02d G01 L2", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i06_[ism-1] ) delete i06_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G01 L2", ism);
        i06_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i06_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser3/Gain01/EBPDT PNs pedestal SM%02d G01 L3", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i07_[ism-1] ) delete i07_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G01 L3", ism);
        i07_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i07_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser4/Gain01/EBPDT PNs pedestal SM%02d G01 L4", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( i08_[ism-1] ) delete i08_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G01 L4", ism);
        i08_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        i08_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs amplitude SM%02d G16 L1", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j01_[ism-1] ) delete j01_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G16 L1", ism);
        j01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs amplitude SM%02d G16 L2", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j02_[ism-1] ) delete j02_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G16 L2", ism);
        j02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs amplitude SM%02d G16 L3", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j03_[ism-1] ) delete j03_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G16 L3", ism);
        j03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs amplitude SM%02d G16 L4", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j04_[ism-1] ) delete j04_[ism-1];
        sprintf(histo, "ME EBPDT PNs amplitude SM%02d G16 L4", ism);
        j04_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j04_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser1/Gain16/EBPDT PNs pedestal SM%02d G16 L1", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j05_[ism-1] ) delete j05_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G16 L1", ism);
        j05_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j05_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser2/Gain16/EBPDT PNs pedestal SM%02d G16 L2", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j06_[ism-1] ) delete j06_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G16 L2", ism);
        j06_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j06_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser3/Gain16/EBPDT PNs pedestal SM%02d G16 L3", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j07_[ism-1] ) delete j07_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G16 L3", ism);
        j07_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j07_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser4/Gain16/EBPDT PNs pedestal SM%02d G16 L4", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( j08_[ism-1] ) delete j08_[ism-1];
        sprintf(histo, "ME EBPDT PNs pedestal SM%02d G16 L4", ism);
        j08_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        j08_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 50.;

    float num01, num02, num03, num04, num05, num06, num07, num08;
    float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
    float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;

    if ( g01_[ism-1] ) g01_[ism-1]->Reset();
    if ( g02_[ism-1] ) g02_[ism-1]->Reset();
    if ( g03_[ism-1] ) g03_[ism-1]->Reset();
    if ( g04_[ism-1] ) g04_[ism-1]->Reset();

    if ( g01_[ism-1] ) a01_[ism-1]->Reset();
    if ( a02_[ism-1] ) a02_[ism-1]->Reset();
    if ( a03_[ism-1] ) a03_[ism-1]->Reset();
    if ( a04_[ism-1] ) a04_[ism-1]->Reset();

    if ( aopn01_[ism-1] ) aopn01_[ism-1]->Reset();
    if ( aopn01_[ism-1] ) aopn02_[ism-1]->Reset();
    if ( aopn01_[ism-1] ) aopn03_[ism-1]->Reset();
    if ( aopn01_[ism-1] ) aopn04_[ism-1]->Reset();

    float meanAmplL1, meanAmplL2, meanAmplL3, meanAmplL4;
    int nCryL1, nCryL2, nCryL3, nCryL4;
    meanAmplL1 = meanAmplL2 = meanAmplL3 = meanAmplL4 = -1.;
    nCryL1 = nCryL2 = nCryL3 = nCryL4 = 0;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01 = num03 = num05 = num07 = -1;

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

        if ( h05_[ism-1] && h05_[ism-1]->GetEntries() >= n_min_tot ) {
          num05 = h05_[ism-1]->GetBinEntries(h05_[ism-1]->GetBin(ie, ip));
          if ( num05 >= n_min_bin ) {
            meanAmplL3 += h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(ie, ip));
            nCryL3++;
          }
        }

        if ( h07_[ism-1] && h07_[ism-1]->GetEntries() >= n_min_tot ) {
          num07 = h07_[ism-1]->GetBinEntries(h07_[ism-1]->GetBin(ie, ip));
          if ( num07 >= n_min_bin ) {
            meanAmplL4 += h07_[ism-1]->GetBinContent(h07_[ism-1]->GetBin(ie, ip));
            nCryL4++;
          }
        }

      }
    }

    if ( nCryL1 > 0 ) meanAmplL1 /= float (nCryL1);
    if ( nCryL2 > 0 ) meanAmplL2 /= float (nCryL2);
    if ( nCryL3 > 0 ) meanAmplL3 /= float (nCryL3);
    if ( nCryL4 > 0 ) meanAmplL4 /= float (nCryL4);

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = num04  = num05  = num06  = num07  = num08  = -1.;
        mean01 = mean02 = mean03 = mean04 = mean05 = mean06 = mean07 = mean08 = -1.;
        rms01  = rms02  = rms03  = rms04  = rms05  = rms06  = rms07  = rms08  = -1.;

        if ( g01_[ism-1] ) g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);
        if ( g02_[ism-1] ) g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), 2.);
        if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);
        if ( g04_[ism-1] ) g04_[ism-1]->SetBinContent(g04_[ism-1]->GetBin(ie, ip), 2.);

        bool update_channel1 = false;
        bool update_channel2 = false;
        bool update_channel3 = false;
        bool update_channel4 = false;

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

        if ( h05_[ism-1] && h05_[ism-1]->GetEntries() >= n_min_tot ) {
          num05 = h05_[ism-1]->GetBinEntries(h05_[ism-1]->GetBin(ie, ip));
          if ( num05 >= n_min_bin ) {
            mean05 = h05_[ism-1]->GetBinContent(h05_[ism-1]->GetBin(ie, ip));
            rms05  = h05_[ism-1]->GetBinError(h05_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h06_[ism-1] && h06_[ism-1]->GetEntries() >= n_min_tot ) {
          num06 = h06_[ism-1]->GetBinEntries(h06_[ism-1]->GetBin(ie, ip));
          if ( num06 >= n_min_bin ) {
            mean06 = h06_[ism-1]->GetBinContent(h06_[ism-1]->GetBin(ie, ip));
            rms06  = h06_[ism-1]->GetBinError(h06_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( h07_[ism-1] && h07_[ism-1]->GetEntries() >= n_min_tot ) {
          num07 = h07_[ism-1]->GetBinEntries(h07_[ism-1]->GetBin(ie, ip));
          if ( num07 >= n_min_bin ) {
            mean07 = h07_[ism-1]->GetBinContent(h07_[ism-1]->GetBin(ie, ip));
            rms07  = h07_[ism-1]->GetBinError(h07_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( h08_[ism-1] && h08_[ism-1]->GetEntries() >= n_min_tot ) {
          num08 = h08_[ism-1]->GetBinEntries(h08_[ism-1]->GetBin(ie, ip));
          if ( num08 >= n_min_bin ) {
            mean08 = h08_[ism-1]->GetBinContent(h08_[ism-1]->GetBin(ie, ip));
            rms08  = h08_[ism-1]->GetBinError(h08_[ism-1]->GetBin(ie, ip));
            update_channel4 = true;
          }
        }

        if ( update_channel1 ) {

          float val;

          val = 1.;
          if ( abs(mean01 - meanAmplL1) > abs(percentVariation_ * meanAmplL1) )
            val = 0.;
          if ( g01_[ism-1] ) g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);

          if ( a01_[ism-1] ) a01_[ism-1]->SetBinContent(ip+20*(ie-1), mean01);
          if ( a01_[ism-1] ) a01_[ism-1]->SetBinError(ip+20*(ie-1), rms01);

          if ( aopn01_[ism-1] ) aopn01_[ism-1]->SetBinContent(ip+20*(ie-1), mean02);
          if ( aopn01_[ism-1] ) aopn01_[ism-1]->SetBinError(ip+20*(ie-1), rms02);

        }

        if ( update_channel2 ) {

          float val;

          val = 1.;
          if ( abs(mean03 - meanAmplL2) > abs(percentVariation_ * meanAmplL2) )
            val = 0.;
          if ( g02_[ism-1] ) g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), val);

          if ( a02_[ism-1] ) a02_[ism-1]->SetBinContent(ip+20*(ie-1), mean03);
          if ( a02_[ism-1] ) a02_[ism-1]->SetBinError(ip+20*(ie-1), rms03);

          if ( aopn02_[ism-1] ) aopn02_[ism-1]->SetBinContent(ip+20*(ie-1), mean04);
          if ( aopn02_[ism-1] ) aopn02_[ism-1]->SetBinError(ip+20*(ie-1), rms04);

        }

        if ( update_channel3 ) {

          float val;

          val = 1.;
          if ( abs(mean05 - meanAmplL3) > abs(percentVariation_ * meanAmplL3) )
            val = 0.;
          if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);

          if ( a03_[ism-1] ) a03_[ism-1]->SetBinContent(ip+20*(ie-1), mean05);
          if ( a03_[ism-1] ) a03_[ism-1]->SetBinError(ip+20*(ie-1), rms05);

          if ( aopn03_[ism-1] ) aopn03_[ism-1]->SetBinContent(ip+20*(ie-1), mean06);
          if ( aopn03_[ism-1] ) aopn03_[ism-1]->SetBinError(ip+20*(ie-1), rms06);

        }

        if ( update_channel4 ) {

          float val;

          val = 1.;
          if ( abs(mean07 - meanAmplL4) > abs(percentVariation_ * meanAmplL4) )
            val = 0.;
          if ( g04_[ism-1] ) g04_[ism-1]->SetBinContent(g04_[ism-1]->GetBin(ie, ip), val);

          if ( a04_[ism-1] ) a04_[ism-1]->SetBinContent(ip+20*(ie-1), mean07);
          if ( a04_[ism-1] ) a04_[ism-1]->SetBinError(ip+20*(ie-1), rms07);

          if ( aopn04_[ism-1] ) aopn04_[ism-1]->SetBinContent(ip+20*(ie-1), mean08);
          if ( aopn04_[ism-1] ) aopn04_[ism-1]->SetBinError(ip+20*(ie-1), rms08);

        }

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
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  const int csize = 250;

  const double histMax = 1.e15;

  int pCol3[3] = { 2, 3, 5 };

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);

  string imgNameQual[4], imgNameAmp[4], imgNameAmpoPN[4], imgNameMEPnG01[4], imgNameMEPnPedG01[4], imgNameMEPnG16[4], imgNameMEPnPedG16[4], imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);
  TCanvas* cAmpoPN = new TCanvas("cAmpoPN", "Temp", csize, csize);
  TCanvas* cPed = new TCanvas("cPed", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;
  TH1D* obj1d;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    // Loop on wavelength

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

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

      // Amplitude distributions

      imgNameAmp[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = a01_[ism-1];
          break;
        case 2:
          obj1f = a02_[ism-1];
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
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Amplitude over PN distributions

      imgNameAmpoPN[iCanvas-1] = "";

      obj1f = 0;
      switch ( iCanvas ) {
        case 1:
          obj1f = aopn01_[ism-1];
          break;
        case 2:
          obj1f = aopn02_[ism-1];
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
        imgNameAmpoPN[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameAmpoPN[iCanvas-1];

        cAmpoPN->cd();
        gStyle->SetOptStat("euomr");
        obj1f->SetStats(kTRUE);
//        if ( obj1f->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1f->Draw();
        cAmpoPN->Update();
        cAmpoPN->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

      }

      // Monitoring elements plots

      imgNameMEPnG01[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i01_[ism-1] ) obj1d = i01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i02_[ism-1] ) obj1d = i02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( i03_[ism-1] ) obj1d = i03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( i04_[ism-1] ) obj1d = i04_[ism-1]->ProjectionY("_py", 1, 1, "e");
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
        imgNameMEPnG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG01[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.);
        obj1d->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      imgNameMEPnG16[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( j01_[ism-1] ) obj1d = j01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( j02_[ism-1] ) obj1d = j02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( j03_[ism-1] ) obj1d = j03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( j04_[ism-1] ) obj1d = j04_[ism-1]->ProjectionY("_py", 1, 1, "e");
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
        imgNameMEPnG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnG16[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->SetMinimum(0.);
        obj1d->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

      // Monitoring elements plots

      imgNameMEPnPedG01[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i05_[ism-1] ) obj1d = i05_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( i06_[ism-1] ) obj1d = i06_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( i07_[ism-1] ) obj1d = i07_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( i08_[ism-1] ) obj1d = i08_[ism-1]->ProjectionY("_py", 1, 1, "e");
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
        imgNameMEPnPedG01[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG01[iCanvas-1];

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

      imgNameMEPnPedG16[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( j05_[ism-1] ) obj1d = j05_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( j06_[ism-1] ) obj1d = j06_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( j07_[ism-1] ) obj1d = j07_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 4:
          if ( j08_[ism-1] ) obj1d = j08_[ism-1]->ProjectionY("_py", 1, 1, "e");
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
        imgNameMEPnPedG16[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPnPedG16[iCanvas-1];

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

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
      htmlFile << "<tr align=\"center\">" << endl;

      if ( imgNameMEPnPedG01[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPedG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnG01[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnG01[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnPedG16[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPedG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPnG16[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnG16[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      htmlFile << "</tr>" << endl;

      htmlFile << "<tr align=\"center\"><td colspan=\"4\">Gain 1</td><td colspan=\"4\">Gain 16</td></tr>" << endl;
      htmlFile << "</table>" << endl;

    }

    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cAmp;
  delete cAmpoPN;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

