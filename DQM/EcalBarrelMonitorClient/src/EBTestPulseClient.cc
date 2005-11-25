/*
 * \file EBTestPulseClient.cc
 * 
 * $Date: 2005/11/24 18:13:16 $
 * $Revision: 1.26 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBTestPulseClient.h>

EBTestPulseClient::EBTestPulseClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    ha01_[i] = 0;
    ha02_[i] = 0;
    ha03_[i] = 0;

    hs01_[i] = 0;
    hs02_[i] = 0;
    hs03_[i] = 0;

    he01_[i] = 0;
    he02_[i] = 0;
    he03_[i] = 0;

    sprintf(histo, "EBPT test pulse quality G01 SM%02d", i+1);
    g01_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBPT test pulse quality G06 SM%02d", i+1);
    g02_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBPT test pulse quality G12 SM%02d", i+1);
    g03_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBPT test pulse amplitude G01 SM%02d", i+1);
    a01_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBPT test pulse amplitude G06 SM%02d", i+1);
    a02_[i] = new TH1F(histo, histo, 1700, 0., 1700.);
    sprintf(histo, "EBPT test pulse amplitude G12 SM%02d", i+1);
    a03_[i] = new TH1F(histo, histo, 1700, 0., 1700.);

    sprintf(histo, "EBPT test pulse shape G01 SM%02d", i+1);
    s01_[i] = new TH1D(histo, histo, 10, 0., 10.);
    sprintf(histo, "EBPT test pulse shape G06 SM%02d", i+1);
    s02_[i] = new TH1D(histo, histo, 10, 0., 10.);
    sprintf(histo, "EBPT test pulse shape G12 SM%02d", i+1);
    s03_[i] = new TH1D(histo, histo, 10, 0., 10.);

  }

  amplitudeThreshold_ = 200;
  RMSThreshold_ = 20 ;                
  threshold_on_AmplitudeErrorsNumber_ = 0.02;

}

EBTestPulseClient::~EBTestPulseClient(){

  this->unsubscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( ha01_[i] ) delete ha01_[i];
    if ( ha02_[i] ) delete ha02_[i];
    if ( ha03_[i] ) delete ha03_[i];

    if ( hs01_[i] ) delete hs01_[i];
    if ( hs02_[i] ) delete hs02_[i];
    if ( hs03_[i] ) delete hs03_[i];

    if ( he01_[i] ) delete he01_[i];
    if ( he02_[i] ) delete he02_[i];
    if ( he03_[i] ) delete he03_[i];

    delete g01_[i];
    delete g02_[i];
    delete g03_[i];

    delete a01_[i];
    delete a02_[i];
    delete a03_[i];

    delete s01_[i];
    delete s02_[i];
    delete s03_[i];

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

    if ( ha01_[i] ) delete ha01_[i];
    if ( ha02_[i] ) delete ha02_[i];
    if ( ha03_[i] ) delete ha03_[i];
    ha01_[i] = 0;
    ha02_[i] = 0;
    ha03_[i] = 0;

    if ( hs01_[i] ) delete hs01_[i];
    if ( hs02_[i] ) delete hs02_[i];
    if ( hs03_[i] ) delete hs03_[i];
    hs01_[i] = 0;
    hs02_[i] = 0;
    hs03_[i] = 0;

    if ( he01_[i] ) delete he01_[i];
    if ( he02_[i] ) delete he02_[i];
    if ( he03_[i] ) delete he03_[i];
    he01_[i] = 0;
    he02_[i] = 0;
    he03_[i] = 0;

    g01_[i]->Reset();
    g02_[i]->Reset();
    g03_[i]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g01_[i]->SetBinContent(g01_[i]->GetBin(ie, ip), 2.);
        g02_[i]->SetBinContent(g02_[i]->GetBin(ie, ip), 2.);
        g03_[i]->SetBinContent(g03_[i]->GetBin(ie, ip), 2.);

      }
    }

    a01_[i]->Reset();
    a02_[i]->Reset();
    a03_[i]->Reset();

    s01_[i]->Reset();
    s02_[i]->Reset();
    s03_[i]->Reset();

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

  const float n_min_tot = 1000.;
  const float n_min_bin = 30.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01;
    float mean01;
    float rms01;

    float num02;
    float mean02;
    float rms02;

    float num03;
    float mean03;
    float rms03;

    vector<int> sample01, sample02, sample03;

    for ( int ie = 1; ie <= 85; ie++ ) { 
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = -1.;
        mean01 = -1.;
        rms01  = -1.;

        num02  = -1.;
        mean02 = -1.;
        rms02  = -1.;

        num03  = -1.;
        mean03 = -1.;
        rms03  = -1.;

        sample01.clear();
        sample02.clear();
        sample03.clear();

        bool update_channel = false;

        float numEventsinCry[3] = {0., 0., 0.};

        if ( ha01_[ism-1] ) numEventsinCry[1] = ha01_[ism-1]->GetBinEntries(ha01_[ism-1]->GetBin(ie, ip));
        if ( ha02_[ism-1] ) numEventsinCry[2] = ha02_[ism-1]->GetBinEntries(ha02_[ism-1]->GetBin(ie, ip));
        if ( ha03_[ism-1] ) numEventsinCry[3] = ha03_[ism-1]->GetBinEntries(ha03_[ism-1]->GetBin(ie, ip));

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

          float val;

          if ( g01_[ism-1] ) {
            val = 1.;
            if ( mean01 > amplitudeThreshold_ )
              val = 0.;
            if ( rms01 > RMSThreshold_ )
              val = 0.;
            if ( he01_[ism-1] ) {
              float errorRate = (float) (he01_[ism-1]->GetBinContent(he01_[ism-1]->GetBin(ie, ip)) / numEventsinCry[1]) ;
              if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
            }
            g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);
          }

          if ( a01_[ism-1] ) {
            a01_[ism-1]->SetBinContent(ip+20*(ie-1), mean01);
            a01_[ism-1]->SetBinError(ip+20*(ie-1), rms01);
          }
          
          if ( hs01_[ism-1] && s01_[ism-1] ) s01_[ism-1] = hs01_[ism-1]->ProjectionY("_py", 1, 10, "e");

          // adc.setADCMean(mean02);
          // adc.setADCRMS(rms02);

          if ( g02_[ism-1] ) {
            val = 1.;
            if ( mean02 > amplitudeThreshold_ )
              val = 0.;
            if ( rms02 > RMSThreshold_ )
              val = 0.;
            if ( he02_[ism-1] ) {
              float errorRate = (float) (he02_[ism-1]->GetBinContent(he02_[ism-1]->GetBin(ie, ip)) / numEventsinCry[2]) ;
              if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
            }
            g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), val);
          }

          if ( a02_[ism-1] ) {
            a02_[ism-1]->SetBinContent(ip+20*(ie-1), mean02);
            a02_[ism-1]->SetBinError(ip+20*(ie-1), rms02);
          }
          
          if ( hs02_[ism-1] && s02_[ism-1] ) s02_[ism-1] = hs02_[ism-1]->ProjectionY("_py", 1, 10, "e");

          // adc.setADCMean(mean03);
          // adc.setADCRMS(rms03);

          if ( g03_[ism-1] ) {
            val = 1.;
            if ( mean03 > amplitudeThreshold_ )
              val = 0.;
            if ( rms03 > RMSThreshold_ )
              val = 0.;
            if ( he03_[ism-1] ) {
              float errorRate = (float) (he03_[ism-1]->GetBinContent(he03_[ism-1]->GetBin(ie, ip)) / numEventsinCry[3]) ;
              if ( errorRate > threshold_on_AmplitudeErrorsNumber_ ) val = 0.;
            }
            g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);
          }

          if ( a03_[ism-1] ) {
            a03_[ism-1]->SetBinContent(ip+20*(ie-1), mean03);
            a03_[ism-1]->SetBinError(ip+20*(ie-1), rms03);
          }
          
          if ( hs03_[ism-1] && s03_[ism-1] ) s03_[ism-1] = hs03_[ism-1]->ProjectionY("_py", 1, 10, "e");

          adc.setTaskStatus(1);

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

            cout << "sample01= " << flush;
            for ( unsigned int i = 0; i < sample01.size(); i++ ) {
              cout << sample01[i] << " " << flush;
            }
            cout << endl;

            cout << "sample02= " << flush;
            for ( unsigned int i = 0; i < sample02.size(); i++ ) {
              cout << sample02[i] << " " << flush;
            }
            cout << endl;

            cout << "sample03= " << flush;
            for ( unsigned int i = 0; i < sample03.size(); i++ ) {
              cout << sample03[i] << " " << flush;
            }
            cout << endl;

            shape.setSamples(sample01);
            

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
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude error SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude error SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT shape SM*");
  mui_->subscribe("*/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude error SM*");

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
        if ( ha01_[ism-1] ) delete ha01_[ism-1];
        ha01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude SM%02d G06", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( ha02_[ism-1] ) delete ha02_[ism-1];
        ha02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude SM%02d G12", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( ha03_[ism-1] ) delete ha03_[ism-1];
        ha03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTT shape SM%02d G01", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( hs01_[ism-1] ) delete hs01_[ism-1];
        hs01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTT shape SM%02d G06", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( hs02_[ism-1] ) delete hs02_[ism-1];
        hs02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTT shape SM%02d G12", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( hs03_[ism-1] ) delete hs03_[ism-1];
        hs03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain01/EBTT amplitude error SM%02d G01", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( he01_[ism-1] ) delete he01_[ism-1];
        he01_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain06/EBTT amplitude error SM%02d G06", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( he02_[ism-1] ) delete he02_[ism-1];
        he02_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBTestPulseTask/Gain12/EBTT amplitude error SM%02d G12", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( he03_[ism-1] ) delete he03_[ism-1];
        he03_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone());
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

  string imgNameQual[3] , imgNameAmp[3] , imgNameShape[3] , imgName , meName;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {
    
    if ( g01_[ism-1] && g02_[ism-1] && g03_[ism-1] &&
         a01_[ism-1] && a02_[ism-1] && a03_[ism-1] &&
         s01_[ism-1] && s02_[ism-1] && s03_[ism-1] ) {

      // Loop on gains

      for ( int iCanvas=1 ; iCanvas <= 3 ; iCanvas++ ) {

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
        case 3:
          meName = g03_[ism-1]->GetName();
          obj2f = g03_[ism-1];
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
          case 3:
            meName = a03_[ism-1]->GetName();
            obj1f = a03_[ism-1];
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
        
        // Shape distributions
        
        TH1D* obj1d = 0;

        switch ( iCanvas ) {
          case 1:
            meName = s01_[ism-1]->GetName();
            obj1d = s01_[ism-1];
            break;
          case 2:
            meName = s02_[ism-1]->GetName();
            obj1d = s02_[ism-1];
            break;
          case 3:
            meName = s03_[ism-1]->GetName();
            obj1d = s03_[ism-1];
            break;
          default:
            break;
          }
        
        TCanvas *cShape = new TCanvas("cShape" , "Temp", csize , csize );
        for ( unsigned int iShape=0 ; iShape < meName.size(); iShape++ ) {
          if ( meName.substr(iShape,1) == " " )  {
            meName.replace(iShape, 1, "_");
          }
        }
        imgNameShape[iCanvas-1] = meName + ".jpg";
        imgName = htmlDir + imgNameShape[iCanvas-1];
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
        cShape->SaveAs(imgName.c_str());
        gPad->SetLogy(0);
        delete cShape;
        
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

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

