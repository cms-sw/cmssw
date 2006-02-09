/*
 * \file EBTestPulseClient.cc
 *
 * $Date: 2006/02/08 18:31:55 $
 * $Revision: 1.63 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

EBTestPulseClient::EBTestPulseClient(const ParameterSet& ps, MonitorUserInterface* mui){

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

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;

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

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBTestPulseClient::~EBTestPulseClient(){

  this->cleanup();

}

void EBTestPulseClient::beginJob(void){

  if ( verbose_ ) cout << "EBTestPulseClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBTestPulseClient::beginRun(void){

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
    sprintf(histo, "EBTPT test pulse quality G01 SM%02d", ism);
    g01_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g02_[ism-1] ) delete g02_[ism-1];
    sprintf(histo, "EBTPT test pulse quality G06 SM%02d", ism);
    g02_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    if ( g03_[ism-1] ) delete g03_[ism-1];
    sprintf(histo, "EBTPT test pulse quality G12 SM%02d", ism);
    g03_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( a01_[ism-1] ) delete a01_[ism-1];
    sprintf(histo, "EBTPT test pulse amplitude G01 SM%02d", ism);
    a01_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( a02_[ism-1] ) delete a02_[ism-1];
    sprintf(histo, "EBTPT test pulse amplitude G06 SM%02d", ism);
    a02_[ism-1] = new TH1F(histo, histo, 1700, 0., 1700.);
    if ( a03_[ism-1] ) delete a03_[ism-1];
    sprintf(histo, "EBTPT test pulse amplitude G12 SM%02d", ism);
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

    if ( cloneME_ ) {
      if ( ha01_[ism-1] ) delete ha01_[ism-1];
      if ( ha02_[ism-1] ) delete ha02_[ism-1];
      if ( ha03_[ism-1] ) delete ha03_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];
      if ( hs02_[ism-1] ) delete hs02_[ism-1];
      if ( hs03_[ism-1] ) delete hs03_[ism-1];

      if ( he01_[ism-1] ) delete he01_[ism-1];
      if ( he02_[ism-1] ) delete he02_[ism-1];
      if ( he03_[ism-1] ) delete he03_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
      if ( i03_[ism-1] ) delete i03_[ism-1];
      if ( i04_[ism-1] ) delete i04_[ism-1];
    }

    ha01_[ism-1] = 0;
    ha02_[ism-1] = 0;
    ha03_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;

    he01_[ism-1] = 0;
    he02_[ism-1] = 0;
    he03_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;

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

void EBTestPulseClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov) {

  EcalLogicID ecid;
  MonTestPulseDat adc;
  map<EcalLogicID, MonTestPulseDat> dataset1;
  MonPulseShapeDat shape;
  map<EcalLogicID, MonPulseShapeDat> dataset2;

  cout << "Creating MonTestPulseDatObjects for the database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 10.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03;
    float mean01, mean02, mean03;
    float rms01, rms02, rms03;

    vector<float> sample01, sample02, sample03;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = -1.;
        mean01 = mean02 = mean03 = -1.;
        rms01  = rms02  = rms03  = -1.;

        sample01.clear();
        sample02.clear();
        sample03.clear();

        bool update_channel = false;

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

            cout << "Preparing dataset for SM=" << ism << endl;
            cout << "G01 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;
            cout << "G06 (" << ie << "," << ip << ") " << num02 << " " << mean02 << " " << rms02 << endl;
            cout << "G12 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;

          }

          adc.setADCMeanG1(mean01);
          adc.setADCRMSG1(rms01);

          adc.setADCMeanG6(mean02);
          adc.setADCRMSG6(rms02);

          adc.setADCMeanG12(mean03);
          adc.setADCRMSG12(rms03);

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
            } else {
              for ( int i = 1; i <= 10; i++ ) { sample01.push_back(-1.); }
            }

            if ( hs02_[ism-1] && hs02_[ism-1]->GetEntries() >= n_min_tot ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample02.push_back(int(hs02_[ism-1]->GetBinContent(hs02_[ism-1]->GetBin(1, i))));
              }
            } else {
              for ( int i = 1; i <= 10; i++ ) { sample02.push_back(-1.); }
            }

            if ( hs03_[ism-1] && hs03_[ism-1]->GetEntries() >= n_min_tot ) {
              for ( int i = 1; i <= 10; i++ ) {
                sample03.push_back(int(hs03_[ism-1]->GetBinContent(hs03_[ism-1]->GetBin(1, i))));
              }
            } else {
              for ( int i = 1; i <= 10; i++ ) { sample03.push_back(-1.); }
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

            shape.setSamples(sample01,  1);
            shape.setSamples(sample02,  6);
            shape.setSamples(sample03, 12);

            int ic = (ip-1) + 20*(ie-1) + 1;

            if ( econn ) {
              try {
                ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
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
      if ( dataset1.size() != 0 ) econn->insertDataSet(&dataset1, moniov);
      if ( dataset2.size() != 0 ) econn->insertDataSet(&dataset2, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  MonPNMGPADat pn;
  map<EcalLogicID, MonPNMGPADat> dataset3;

  cout << "Creating MonPnDatObjects for the database ..." << endl;

  const float m_min_tot = 100.;
  const float m_min_bin = 10.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04;
    float mean01, mean02, mean03, mean04;
    float rms01, rms02, rms03, rms04;

    for ( int i = 1; i <= 10; i++ ) {

      num01  = num02  = num03  = num04  = -1.;
      mean01 = mean02 = mean03 = mean04 = -1.;
      rms01  = rms02  = rms03  = rms04  = -1.;

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

      if ( i03_[ism-1] && i03_[ism-1]->GetEntries() >= m_min_tot ) {
        num03 = i03_[ism-1]->GetBinEntries(i03_[ism-1]->GetBin(i));
        if ( num03 >= m_min_bin ) {
          mean03 = i03_[ism-1]->GetBinContent(i03_[ism-1]->GetBin(1, i));
          rms03  = i03_[ism-1]->GetBinError(i03_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( i04_[ism-1] && i04_[ism-1]->GetEntries() >= m_min_tot ) {
        num04 = i04_[ism-1]->GetBinEntries(i04_[ism-1]->GetBin(1, i));
        if ( num04 >= m_min_bin ) {
          mean04 = i04_[ism-1]->GetBinContent(i04_[ism-1]->GetBin(1, i));
          rms04  = i04_[ism-1]->GetBinError(i04_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( update_channel ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") G01 " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "PNs (" << i << ") G16 " << num02  << " " << mean02 << " " << rms02  << endl;

        }

        pn.setADCMeanG1(mean01);
        pn.setADCRMSG1(rms01);

        pn.setPedMeanG1(mean03);
        pn.setPedRMSG1(rms03);

        pn.setADCMeanG16(mean02);
        pn.setADCRMSG16(rms02);

        pn.setPedMeanG16(mean04);
        pn.setPedRMSG16(rms04);

        if ( mean01 > 200. ) {
          pn.setTaskStatus(true);
        } else {
          pn.setTaskStatus(false);
        }

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset3[ecid] = pn;
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
      if ( dataset3.size() != 0 ) econn->insertDataSet(&dataset3, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBTestPulseClient::subscribe(void){

  if ( verbose_ ) cout << "EBTestPulseClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude error SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude error SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude error SM*");

  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBTestPulseClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBTPT amplitude SM%02d G01", ism);
      me_ha01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude SM%02d G01", ism);
      mui_->add(me_ha01_[ism-1], histo);

      sprintf(histo, "EBTPT amplitude SM%02d G06", ism);
      me_ha02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude SM%02d G06", ism);
      mui_->add(me_ha02_[ism-1], histo);

      sprintf(histo, "EBTPT amplitude SM%02d G12", ism);
      me_ha03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude SM%02d G12", ism);
      mui_->add(me_ha03_[ism-1], histo);

      sprintf(histo, "EBTPT shape SM%02d G01", ism);
      me_hs01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT shape SM%02d G01", ism);
      mui_->add(me_hs01_[ism-1], histo);

      sprintf(histo, "EBTPT shape SM%02d G06", ism);
      me_hs02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT shape SM%02d G06", ism);
      mui_->add(me_hs02_[ism-1], histo);

      sprintf(histo, "EBTPT shape SM%02d G12", ism);
      me_hs03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT shape SM%02d G12", ism);
      mui_->add(me_hs03_[ism-1], histo);

      sprintf(histo, "EBTPT amplitude error SM%02d G01", ism);
      me_he01_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude error SM%02d G01", ism);
      mui_->add(me_he01_[ism-1], histo);

      sprintf(histo, "EBTPT amplitude error SM%02d G06", ism);
      me_he02_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude error SM%02d G06", ism);
      mui_->add(me_he02_[ism-1], histo);

      sprintf(histo, "EBTPT amplitude error SM%02d G12", ism);
      me_he03_[ism-1] = mui_->collate2D(histo, histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude error SM%02d G12", ism);
      mui_->add(me_he03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G01", ism);
      me_i01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs amplitude SM%02d G01", ism);
      mui_->add(me_i01_[ism-1], histo);

      sprintf(histo, "EBPDT PNs amplitude SM%02d G16", ism);
      me_i02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs amplitude SM%02d G16", ism);
      mui_->add(me_i02_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G01", ism);
      me_i03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain01");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM%02d G01", ism);
      mui_->add(me_i03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs pedestal SM%02d G16", ism);
      me_i04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain16");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM%02d G16", ism);
      mui_->add(me_i04_[ism-1], histo);

    }

  }

}

void EBTestPulseClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT shape SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude error SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT shape SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude error SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT shape SM*");
  mui_->subscribeNew("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude error SM*");

  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs amplitude SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM*");

}

void EBTestPulseClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBTestPulseClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBTestPulseClient: uncollate" << endl;

    if ( mui_ ) {

      for ( int ism = 1; ism <= 36; ism++ ) {

        mui_->removeCollate(me_ha01_[ism-1]);
        mui_->removeCollate(me_ha02_[ism-1]);
        mui_->removeCollate(me_ha03_[ism-1]);

        mui_->removeCollate(me_hs01_[ism-1]);
        mui_->removeCollate(me_hs02_[ism-1]);
        mui_->removeCollate(me_hs03_[ism-1]);

        mui_->removeCollate(me_he01_[ism-1]);
        mui_->removeCollate(me_he02_[ism-1]);
        mui_->removeCollate(me_he03_[ism-1]);

        mui_->removeCollate(me_i01_[ism-1]);
        mui_->removeCollate(me_i02_[ism-1]);
        mui_->removeCollate(me_i03_[ism-1]);
        mui_->removeCollate(me_i04_[ism-1]);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT shape SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude error SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT shape SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude error SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT shape SM*");
  mui_->unsubscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude error SM*");

  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs amplitude SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs pedestal SM*");

}

void EBTestPulseClient::analyze(void){

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
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01/EBTPT amplitude SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( ha01_[ism-1] ) delete ha01_[ism-1];
          sprintf(histo, "ME EBTPT amplitude SM%02d G01", ism);
          ha01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          ha01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06/EBTPT amplitude SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( ha02_[ism-1] ) delete ha02_[ism-1];
          sprintf(histo, "ME EBTPT amplitude SM%02d G06", ism);
          ha02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          ha02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12/EBTPT amplitude SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( ha03_[ism-1] ) delete ha03_[ism-1];
          sprintf(histo, "ME EBTPT amplitude SM%02d G12", ism);
          ha03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          ha03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01/EBTPT shape SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTPT shape SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( hs01_[ism-1] ) delete hs01_[ism-1];
          sprintf(histo, "ME EBTPT shape SM%02d G01", ism);
          hs01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          hs01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06/EBTPT shape SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTPT shape SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( hs02_[ism-1] ) delete hs02_[ism-1];
          sprintf(histo, "ME EBTPT shape SM%02d G06", ism);
          hs02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          hs02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12/EBTPT shape SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTPT shape SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( hs03_[ism-1] ) delete hs03_[ism-1];
          sprintf(histo, "ME EBTPT shape SM%02d G12", ism);
          hs03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          hs03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain01/EBTPT amplitude error SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTPT amplitude error SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( he01_[ism-1] ) delete he01_[ism-1];
          sprintf(histo, "ME EBTPT amplitude error SM%02d G01", ism);
          he01_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
        } else {
          he01_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain06/EBTPT amplitude error SM%02d G06", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTPT amplitude error SM%02d G06", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( he02_[ism-1] ) delete he02_[ism-1];
          sprintf(histo, "ME EBTPT amplitude error SM%02d G06", ism);
          he02_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
        } else {
          he02_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBTestPulseTask/Gain12/EBTPT amplitude error SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTPT amplitude error SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( he03_[ism-1] ) delete he03_[ism-1];
          sprintf(histo, "ME EBTPT amplitude error SM%02d G12", ism);
          he03_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
        } else {
          he03_[ism-1] = dynamic_cast<TH2F*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain01/EBPDT PNs amplitude SM%02d G01", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Gain01/EBPDT PNs amplitude SM%02d G01", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( i01_[ism-1] ) delete i01_[ism-1];
          sprintf(histo, "ME EBPDT PNs amplitude SM%02d G01", ism);
          i01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          i01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Gain16/EBPDT PNs amplitude SM%02d G16", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Gain16/EBPDT PNs amplitude SM%02d G16", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( i02_[ism-1] ) delete i02_[ism-1];
          sprintf(histo, "ME EBPDT PNs amplitude SM%02d G16", ism);
          i02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          i02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
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
          if ( i03_[ism-1] ) delete i03_[ism-1];
          sprintf(histo, "ME EBPDT PNs pedestal SM%02d G01", ism);
          i03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          i03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
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
          if ( i04_[ism-1] ) delete i04_[ism-1];
          sprintf(histo, "ME EBPDT PNs pedestal SM%02d G16", ism);
          i04_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          i04_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 10.;

    float num01, num02, num03;
    float mean01, mean02, mean03;
    float rms01, rms02, rms03;

    if ( g01_[ism-1] ) g01_[ism-1]->Reset();
    if ( g02_[ism-1] ) g02_[ism-1]->Reset();
    if ( g03_[ism-1] ) g03_[ism-1]->Reset();

    if ( a01_[ism-1] ) a01_[ism-1]->Reset();
    if ( a02_[ism-1] ) a02_[ism-1]->Reset();
    if ( a03_[ism-1] ) a03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = -1.;
        mean01 = mean02 = mean03 = -1.;
        rms01  = rms02  = rms03  = -1.;

        if ( g01_[ism-1] ) g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);
        if ( g02_[ism-1] ) g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), 2.);
        if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

        bool update_channel1 = false;
        bool update_channel2 = false;
        bool update_channel3 = false;

        float numEventsinCry[3] = {0., 0., 0.};

        if ( ha01_[ism-1] ) numEventsinCry[0] = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
        if ( ha02_[ism-1] ) numEventsinCry[1] = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
        if ( ha03_[ism-1] ) numEventsinCry[2] = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));

        if ( ha01_[ism-1] && ha01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = ha01_[ism-1]->GetBinContent(ha01_[ism-1]->GetBin(ie, ip));
            rms01  = ha01_[ism-1]->GetBinError(ha01_[ism-1]->GetBin(ie, ip));
            update_channel1 = true;
          }
        }

        if ( ha02_[ism-1] && ha02_[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = ha02_[ism-1]->GetBinContent(ha02_[ism-1]->GetBin(ie, ip));
            rms02  = ha02_[ism-1]->GetBinError(ha02_[ism-1]->GetBin(ie, ip));
            update_channel2 = true;
          }
        }

        if ( ha03_[ism-1] && ha03_[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = ha03_[ism-1]->GetBinContent(ha03_[ism-1]->GetBin(ie, ip));
            rms03  = ha03_[ism-1]->GetBinError(ha03_[ism-1]->GetBin(ie, ip));
            update_channel3 = true;
          }
        }

        if ( update_channel1 ) {

          float val;

          val = 1.;
          if ( mean01 < amplitudeThreshold_ )
            val = 0.;
          if ( rms01 > RMSThreshold_ )
            val = 0.;
          if ( he01_[ism-1] && numEventsinCry[0] > 0 ) {
            float errorRate = he01_[ism-1]->GetBinContent(he01_[ism-1]->GetBin(ie, ip)) / numEventsinCry[0];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          if ( g01_[ism-1] ) g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);

          if ( a01_[ism-1] ) a01_[ism-1]->SetBinContent(ip+20*(ie-1), mean01);
          if ( a01_[ism-1] ) a01_[ism-1]->SetBinError(ip+20*(ie-1), rms01);

        }

        if ( update_channel2 ) {

          float val;

          val = 1.;
          if ( mean02 < amplitudeThreshold_ )
            val = 0.;
          if ( rms02 > RMSThreshold_ )
            val = 0.;
          if ( he02_[ism-1] && numEventsinCry[1] > 0 ) {
            float errorRate = he02_[ism-1]->GetBinContent(he02_[ism-1]->GetBin(ie, ip)) / numEventsinCry[1];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          if ( g02_[ism-1] ) g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), val);

          if ( a02_[ism-1] ) a02_[ism-1]->SetBinContent(ip+20*(ie-1), mean02);
          if ( a02_[ism-1] ) a02_[ism-1]->SetBinError(ip+20*(ie-1), rms02);

        }

        if ( update_channel3 ) {

          float val;

          val = 1.;
          if ( mean03 < amplitudeThreshold_ )
            val = 0.;
          if ( rms03 > RMSThreshold_ )
            val = 0.;
          if ( he03_[ism-1] && numEventsinCry[2] > 0 ) {
            float errorRate = he03_[ism-1]->GetBinContent(he03_[ism-1]->GetBin(ie, ip)) / numEventsinCry[2];
            if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
          }
          if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);

          if ( a03_[ism-1] ) a03_[ism-1]->SetBinContent(ip+20*(ie-1), mean03);
          if ( a03_[ism-1] ) a03_[ism-1]->SetBinError(ip+20*(ie-1), rms03);

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

  string imgNameQual[3], imgNameAmp[3], imgNameShape[3], imgNameMEPn[2], imgNameMEPnPed[2], imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);
  TCanvas* cShape = new TCanvas("cShape", "Temp", csize, csize);
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
        case 3:
          obj1f = a03_[ism-1];
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

      // Shape distributions

      imgNameShape[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( hs01_[ism-1] ) obj1d = hs01_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
          if ( hs02_[ism-1] ) obj1d = hs02_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 3:
          if ( hs03_[ism-1] ) obj1d = hs03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
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
        cShape->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

    }

    // Loop on gain

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      // Monitoring elements plots

      imgNameMEPn[iCanvas-1] = "";

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
        imgNameMEPn[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameMEPn[iCanvas-1];

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

      imgNameMEPnPed[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( i03_[ism-1] ) obj1d = i03_[ism-1]->ProjectionY("_py", 1, 1, "e");
          break;
        case 2:
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

      if ( imgNameAmp[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameAmp[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameShape[iCanvas-1].size() != 0 )
        htmlFile << "<td><img src=\"" << imgNameShape[iCanvas-1] << "\"></td>" << endl;
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

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameMEPnPed[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPnPed[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

      if ( imgNameMEPn[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameMEPn[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td colspan=\"2\"><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"center\"><td colspan=\"4\">Gain 1</td><td colspan=\"4\">Gain 16</td></tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cAmp;
  delete cShape;
  delete cPed;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

