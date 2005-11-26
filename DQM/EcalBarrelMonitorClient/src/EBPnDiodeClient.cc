/*
 * \file EBPnDiodeClient.cc
 * 
 * $Date: 2005/11/26 18:43:10 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPnDiodeClient.h>

EBPnDiodeClient::EBPnDiodeClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  for ( int i = 0; i < 36; i++ ) {

    h01_[i] = 0;

  }

}

EBPnDiodeClient::~EBPnDiodeClient(){

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];

  }

}

void EBPnDiodeClient::beginJob(const edm::EventSetup& c){

  cout << "EBPnDiodeClient: beginJob" << endl;

  ievt_ = 0;

  this->subscribe();

}

void EBPnDiodeClient::beginRun(const edm::EventSetup& c){

  cout << "EBPnDiodeClient: beginRun" << endl;

  jevt_ = 0;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( h01_[ism-1] ) delete h01_[ism-1];
    h01_[ism-1] = 0;

  }

}

void EBPnDiodeClient::endJob(void) {

  cout << "EBPnDiodeClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

}

void EBPnDiodeClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBPnDiodeClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  MonPNDat p;
  map<EcalLogicID, MonPNDat> dataset;

  cout << "Writing MonPnDatObjects to database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01;
    float mean01;
    float rms01;

    for ( int i = 1; i <= 10; i++ ) {

      num01  = -1.;
      mean01 = -1.;
      rms01  = -1.;

      bool update_channel = false;

      if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
        num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(i));
        if ( num01 >= n_min_bin ) {
          mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(i));
          rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(i));
          update_channel = true;
        }
      }

      if ( update_channel ) {

        if ( i == 1 ) {

          cout << "Inserting dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") " << num01  << " "
                                       << mean01 << " "
                                       << rms01  << endl;
        }

        p.setADCMean(mean01);
        p.setADCRMS(rms01);

        p.setTaskStatus(1);

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset[ecid] = p;
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

void EBPnDiodeClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/EBPT PNs SM*");

}

void EBPnDiodeClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/EBPT PNs SM*");

}

void EBPnDiodeClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/EBPT PNs SM*");

}

void EBPnDiodeClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )  
    cout << "EBPnDiodeClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/EBPT PNs SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        sprintf(histo, "ME EBPT PNs SM%02d", ism);
        h01_[ism-1] = dynamic_cast<TProfile*> ((ob->operator->())->Clone(histo));
      }
    }

  }

}

void EBPnDiodeClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBPnDiodeClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PnDiodeTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PNDIODE</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=white>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .jpg files from existing histograms

  int csize = 250;

//  double histMax = 1.e15;

  string imgNameME, imgName, meName;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    if ( h01_[ism-1] ) {

      // Monitoring elements plots

      TProfile* objp = 0;

      meName = h01_[ism-1]->GetName();
      objp = h01_[ism-1];

      TCanvas *cAmp = new TCanvas("cAmp" , "Temp", csize , csize );
      for ( unsigned int iAmp=0 ; iAmp < meName.size(); iAmp++ ) {
        if ( meName.substr(iAmp,1) == " " )  {
          meName.replace(iAmp, 1 ,"_" );
        }
      }
      imgNameME = meName + ".jpg";
      imgName = htmlDir + imgNameME;
      gStyle->SetOptStat("euomr");
      objp->SetStats(kTRUE);
//      if ( objp->GetMaximum(histMax) > 0. ) {
//        gPad->SetLogy(1);
//      } else {
//        gPad->SetLogy(0);
//      }
      objp->Draw();
      cAmp->Update();
      TPaveStats* stAmp = dynamic_cast<TPaveStats*>(objp->FindObject("stats"));
      if ( stAmp ) {
        stAmp->SetX1NDC(0.6);
        stAmp->SetY1NDC(0.75);
      }
      cAmp->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
      delete cAmp;

    }

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    if ( imgNameME.size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameME << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

