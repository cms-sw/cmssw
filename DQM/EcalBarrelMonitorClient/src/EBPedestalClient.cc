/*
 * \file EBPedestalClient.cc
 * 
 * $Date: 2005/11/14 13:33:33 $
 * $Revision: 1.11 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>

EBPedestalClient::EBPedestalClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    sprintf(histo, "EBPT pedestal quality G01 SM%02d", i+1);
    g01[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBPT pedestal quality G06 SM%02d", i+1);
    g02[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBPT pedestal quality G12 SM%02d", i+1);
    g03[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBPT pedestal mean G01 SM%02d", i+1);
    p01[i] = new TH1F(histo, histo, 100, 150., 250.);
    sprintf(histo, "EBPT pedestal mean G06 SM%02d", i+1);
    p02[i] = new TH1F(histo, histo, 100, 150., 250.);
    sprintf(histo, "EBPT pedestal mean G12 SM%02d", i+1);
    p03[i] = new TH1F(histo, histo, 100, 150., 250.);

    sprintf(histo, "EBPT pedestal rms G01 SM%02d", i+1);
    r01[i] = new TH1F(histo, histo, 100, 0., 10.);
    sprintf(histo, "EBPT pedestal rms G06 SM%02d", i+1);
    r02[i] = new TH1F(histo, histo, 100, 0., 10.);
    sprintf(histo, "EBPT pedestal rms G12 SM%02d", i+1);
    r03[i] = new TH1F(histo, histo, 100, 0., 10.);
  }

}

EBPedestalClient::~EBPedestalClient(){

  this->unsubscribe();

}

void EBPedestalClient::beginJob(const edm::EventSetup& c){

  cout << "EBPedestalClient: beginJob" << endl;

  ievt_ = 0;

}

void EBPedestalClient::beginRun(const edm::EventSetup& c){

  cout << "EBPedestalClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  for ( int ism = 1; ism <= 36; ism++ ) {
    h01[ism-1] = 0;
    h02[ism-1] = 0;
    h03[ism-1] = 0;
  }

}

void EBPedestalClient::endJob(void) {

  cout << "EBPedestalClient: endJob, ievt = " << ievt_ << endl;

}

void EBPedestalClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBPedestalClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  MonPedestalsDat p;
  map<EcalLogicID, MonPedestalsDat> dataset;

  cout << "Writing MonPedestalsDatObjects to database ..." << endl;

  float n_min_tot = 1000.;
  float n_min_bin = 50.;

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

        if ( h01[ism-1] && h01[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01[ism-1]->GetBinEntries(h01[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01[ism-1]->GetBinContent(h01[ism-1]->GetBin(ie, ip));
            rms01  = h01[ism-1]->GetBinError(h01[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }
  
        if ( h02[ism-1] && h02[ism-1]->GetEntries() >= n_min_tot ) {
          num02 = h02[ism-1]->GetBinEntries(h02[ism-1]->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02[ism-1]->GetBinContent(h02[ism-1]->GetBin(ie, ip));
            rms02  = h02[ism-1]->GetBinError(h02[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }
  
        if ( h03[ism-1] && h03[ism-1]->GetEntries() >= n_min_tot ) {
          num03 = h03[ism-1]->GetBinEntries(h03[ism-1]->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03[ism-1]->GetBinContent(h03[ism-1]->GetBin(ie, ip));
            rms03  = h03[ism-1]->GetBinError(h03[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "G01 (" << ie << "," << ip << ") " << num01  << " "
                                                       << mean01 << " "
                                                       << rms01  << endl;
            cout << "G06 (" << ie << "," << ip << ") " << num02  << " "
                                                       << mean02 << " "
                                                       << rms02  << endl;
            cout << "G12 (" << ie << "," << ip << ") " << num03  << " "
                                                       << mean03 << " "
                                                       << rms03  << endl;

          }

          p.setPedMeanG1(mean01);
          p.setPedRMSG1(rms01);

          if ( g01[ism-1] ) {
            if ( rms01 == 0 ) {
              g01[ism-1]->SetBinContent(g01[ism-1]->GetBin(ie, ip), 0.);
            }
          }

          if ( p01[ism-1] ) p01[ism-1]->Fill(mean01);
          if ( r01[ism-1] ) r01[ism-1]->Fill(rms01);

          p.setPedMeanG6(mean02);
          p.setPedRMSG6(rms02);

          if ( p02[ism-1] ) p02[ism-1]->Fill(mean02);
          if ( r02[ism-1] ) r02[ism-1]->Fill(rms02);

          p.setPedMeanG12(mean03);
          p.setPedRMSG12(rms03);

          if ( p03[ism-1] ) p03[ism-1]->Fill(mean03);
          if ( r03[ism-1] ) r03[ism-1]->Fill(rms03);

          p.setTaskStatus(1);

          try {
            if ( econn ) ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
            dataset[ecid] = p;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }

        }

      }
    }

  }

  try {
    cout << "Inserting dataset ... " << flush;
    if ( econn ) econn->insertDataSet(&dataset, runiov, runtag );
    cout << "done." << endl;
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
  }

}

void EBPedestalClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

void EBPedestalClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

void EBPedestalClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

void EBPedestalClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )  
  cout << "EBPedestalClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    h01[ism-1] = 0;
    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM%02d G01", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) h01[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
    }

    h02[ism-1] = 0;
    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM%02d G06", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) h02[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
    }

    h03[ism-1] = 0;
    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM%02d G12", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) h03[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
    }

  }

}

void EBPedestalClient::htmlOutput(int run, string htmlDir){

  cout << "Preparing EBPedestalClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + "EBPedestalClient.html").c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PedestalTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PEDESTAL</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  string gifname01 , gifname02 , gifname03;
  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    if ( g01[ism-1] && g02[ism-1] && g03[ism-1] ) {

      htmlFile << "</h3>Supermodule&nbsp;&nbsp;" << ism << "</h3>" << endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\" align=center> " << endl;
      htmlFile << "<tr><td>" << endl;

      htmlFile << "</td></tr>" << endl;
      htmlFile << "<tr><td>Gain 1</td><td>Gain 6</td><td>Gain 12</td><tr>" << endl;
      htmlFile << "</table>" << endl;
      htmlFile << "<br>" << endl;
    
    }

  }

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

