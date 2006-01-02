/*
 * \file EBCosmicClient.cc
 * 
 * $Date: 2006/01/02 14:04:38 $
 * $Revision: 1.30 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBCosmicClient.h>

EBCosmicClient::EBCosmicClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  for ( int ism = 1; ism <= 36; ism++ ) {

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

  }

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBCosmicClient::~EBCosmicClient(){

  this->cleanup();

}

void EBCosmicClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBCosmicClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBCosmicClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBCosmicClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBCosmicClient::endJob(void) {

  if ( verbose_ ) cout << "EBCosmicClient: endJob, ievt = " << ievt_ << endl;

}

void EBCosmicClient::endRun(void) {

  if ( verbose_ ) cout << "EBCosmicClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBCosmicClient::setup(void) {

}

void EBCosmicClient::cleanup(void) {

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( h01_[ism-1] ) delete h01_[ism-1];
    h01_[ism-1] = 0;
    if ( h02_[ism-1] ) delete h02_[ism-1];
    h02_[ism-1] = 0;
    if ( h03_[ism-1] ) delete h03_[ism-1];
    h03_[ism-1] = 0;

  }

}

void EBCosmicClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov) {

  EcalLogicID ecid;
  MonOccupancyDat o;
  map<EcalLogicID, MonOccupancyDat> dataset;

  cout << "Creating MonOccupancyDatObjects for the database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 10.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02;
    float mean01, mean02;
    float rms01, rms02;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = -1.;
        mean01 = mean02 = -1.;
        rms01  = rms02  = -1.;

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

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Preparing dataset for SM=" << ism << endl;

            cout << "Sel (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "Cut (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << endl;

          }

          o.setEventsOverHighThreshold(int(num01));
          o.setEventsOverLowThreshold(int(num02));

          o.setAvgEnergy(mean01);

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
              dataset[ecid] = o;
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
      if ( dataset.size() != 0 ) econn->insertDataSet(&dataset, moniov);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBCosmicClient::subscribe(void){

  if ( verbose_ ) cout << "EBCosmicClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBCosmicTask/Sel/EBCT amplitude sel SM*");
  mui_->subscribe("*/EcalBarrel/EBCosmicTask/Cut/EBCT amplitude cut SM*");
  mui_->subscribe("*/EcalBarrel/EBCosmicTask/Spectrum/EBCT amplitude spectrum SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBCosmicClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBCT amplitude sel SM%02d", ism);
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBCosmicTask/Sel");
      sprintf(histo, "*/EcalBarrel/EBCosmicTask/Sel/EBCT amplitude sel SM%02d", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBCT amplitude cut SM%02d", ism);
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBCosmicTask/Cut");
      sprintf(histo, "*/EcalBarrel/EBCosmicTask/Cut/EBCT amplitude cut SM%02d", ism);
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EBCT amplitude spectrum SM%02d", ism);
      me_h03_[ism-1] = mui_->collate1D(histo, histo, "EcalBarrel/Sums/EBCosmicTask/Spectrum");
      sprintf(histo, "*/EcalBarrel/EBCosmicTask/Spectrum/EBCT amplitude spectrum SM%02d", ism);
      mui_->add(me_h03_[ism-1], histo);

    }

  }

}

void EBCosmicClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBCosmicTask/Sel/EBCT amplitude sel SM*");
  mui_->subscribeNew("*/EcalBarrel/EBCosmicTask/Cut/EBCT amplitude cut SM*");
  mui_->subscribeNew("*/EcalBarrel/EBCosmicTask/Spectrum/EBCT amplitude spectrum SM*");

}

void EBCosmicClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBCosmicClient: unsubscribe" << endl;

  if ( collateSources_ ) {
  
    if ( verbose_ ) cout << "EBCosmicClient: uncollate" << endl;

    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( bei ) { 

      Char_t histo[80];

      for ( int ism = 1; ism <= 36; ism++ ) {

        sprintf(histo, "EBCT amplitude sel SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBCosmicTask/Sel");
        bei->removeElement(histo);

        sprintf(histo, "EBCT amplitude cut SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBCosmicTask/Cut");
        bei->removeElement(histo);

        sprintf(histo, "EBCT amplitude spectrum SM%02d", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBCosmicTask/Spectrum");
        bei->removeElement(histo);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBCosmicTask/Sel/EBCT amplitude sel SM*");
  mui_->unsubscribe("*/EcalBarrel/EBCosmicTask/Cut/EBCT amplitude cut SM*");
  mui_->unsubscribe("*/EcalBarrel/EBCosmicTask/Spectrum/EBCT amplitude spectrum SM*");

}

void EBCosmicClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBCosmicClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBCosmicTask/Sel/EBCT amplitude sel SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBCosmicTask/Sel/EBCT amplitude sel SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        sprintf(histo, "ME EBCT amplitude sel SM%02d", ism);
        h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBCosmicTask/Cut/EBCT amplitude cut SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBCosmicTask/Cut/EBCT amplitude cut SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        sprintf(histo, "ME EBCT amplitude cut SM%02d", ism);
        h02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBCosmicTask/Spectrum/EBCT amplitude spectrum SM%02d", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBCosmicTask/Spectrum/EBCT amplitude spectrum SM%02d", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        sprintf(histo, "ME EBCT amplitude spectrum SM%02d", ism);
        h03_[ism-1] = dynamic_cast<TH1F*> ((ob->operator->())->Clone(histo));
//        h03_[ism-1] = dynamic_cast<TH1F*> (ob->operator->());
      }
    }

  }

}

void EBCosmicClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBCosmicClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:CosmicTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">COSMIC</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  int csize = 250;

  double histMax = 1.e15;

  int pCol4[10];
  for ( int i = 0; i < 10; i++ ) pCol4[i] = 30+i;

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);

  string imgNameME[3], imgName, meName;

  TCanvas* cMe = new TCanvas("cMe", "Temp", 2*csize, csize);
  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);

  TProfile2D* objp;
  TH1F* obj1f; 

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    // Monitoring elements plots

    for ( int iCanvas = 1; iCanvas <= 2; iCanvas++ ) {

      imgNameME[iCanvas-1] = "";

      objp = 0;
      switch ( iCanvas ) {
        case 1:
          objp = h01_[ism-1];
          break;
        case 2:
          objp = h02_[ism-1];
          break;
        default:
          break;
      }

      if ( objp ) {

        meName = objp->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1, "_");
          }
        }
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cMe->cd();
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(10, pCol4);
        objp->GetXaxis()->SetNdivisions(17);
        objp->GetYaxis()->SetNdivisions(4);
        cMe->SetGridx();
        cMe->SetGridy();
        objp->SetMaximum();
        objp->Draw("colz");
        dummy.Draw("text,same");
        cMe->Update();
        cMe->SaveAs(imgName.c_str());

      }

    }

    // Energy spectrum distributions

    imgNameME[2] = "";

    obj1f = h03_[ism-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameME[2] = meName + ".png";
      imgName = htmlDir + imgNameME[2];

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
      cAmp->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 ) 
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    if ( imgNameME[2].size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameME[2] << "\"></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cMe;
  delete cAmp;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

