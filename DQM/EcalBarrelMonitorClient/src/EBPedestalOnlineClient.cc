/*
 * \file EBPedestalOnlineClient.cc
 *
 * $Date: 2006/01/18 11:40:54 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalOnlineClient.h>

EBPedestalOnlineClient::EBPedestalOnlineClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  for ( int ism = 1; ism <= 36; ism++ ) {

    h03_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g03_[ism-1] = 0;

    p03_[ism-1] = 0;

    r03_[ism-1] = 0;

  }

   expectedMean_ = 200.0;
   discrepancyMean_ = 20.0;
   RMSThreshold_ = 2.0;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBPedestalOnlineClient::~EBPedestalOnlineClient(){

  this->cleanup();

}

void EBPedestalOnlineClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBPedestalOnlineClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBPedestalOnlineClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBPedestalOnlineClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBPedestalOnlineClient::endJob(void) {

  if ( verbose_ ) cout << "EBPedestalOnlineClient: endJob, ievt = " << ievt_ << endl;

}

void EBPedestalOnlineClient::endRun(void) {

  if ( verbose_ ) cout << "EBPedestalOnlineClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBPedestalOnlineClient::setup(void) {

  Char_t histo[50];

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g03_[ism-1] ) delete g03_[ism-1];
    sprintf(histo, "EBPOT pedestal quality G12 SM%02d", ism);
    g03_[ism-1] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    if ( p03_[ism-1] ) delete p03_[ism-1];
    sprintf(histo, "EBPOT pedestal mean G12 SM%02d", ism);
    p03_[ism-1] = new TH1F(histo, histo, 100, 150., 250.);

    if ( r03_[ism-1] ) delete r03_[ism-1];
    sprintf(histo, "EBPOT pedestal rms G12 SM%02d", ism);
    r03_[ism-1] = new TH1F(histo, histo, 100, 0.,  10.);

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    g03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

      }
    }

    p03_[ism-1]->Reset();

    r03_[ism-1]->Reset();

  }

}

void EBPedestalOnlineClient::cleanup(void) {

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( cloneME_ ) {
      if ( h03_[ism-1] ) delete h03_[ism-1];
    }

    h03_[ism-1] = 0;

  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( g03_[ism-1] ) delete g03_[ism-1];
    g03_[ism-1] = 0;

    if ( p03_[ism-1] ) delete p03_[ism-1];
    p03_[ism-1] = 0;

    if ( r03_[ism-1] ) delete r03_[ism-1];
    r03_[ism-1] = 0;

  }

}

void EBPedestalOnlineClient::writeDb(EcalCondDBInterface* econn, MonRunIOV* moniov) {

  EcalLogicID ecid;
  MonPedestalsOnlineDat p;
  map<EcalLogicID, MonPedestalsOnlineDat> dataset;

  cout << "Creating MonPedestalsOnlineDatObjects for the database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num03;
    float mean03;
    float rms03;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num03  = -1.;
        mean03 = -1.;
        rms03  = -1.;

        bool update_channel = false;

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

            cout << "G12 (" << ie << "," << ip << ") " << num03  << " " << mean03 << " " << rms03  << endl;
          }

          p.setADCMeanG12(mean03);
          p.setADCRMSG12(rms03);

          if ( g03_[ism-1]  && g03_[ism-1]->GetBinContent(g03_[ism-1]->GetBin(ie, ip)) == 1. ) {
             p.setTaskStatus(true);
          } else {
             p.setTaskStatus(false);
          }

          int ic = (ip-1) + 20*(ie-1) + 1;

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_number", ism, ic);
              dataset[ecid] = p;
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

void EBPedestalOnlineClient::subscribe(void){

  if ( verbose_ ) cout << "EBPedestalOnlineClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedestalOnlineTask/Gain12/EBPOT pedestal SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPedestalOnlineClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBPOT pedestal SM%02d G12", ism);
      me_h03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedestalOnlineTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBPedestalOnlineTask/Gain12/EBPOT pedestal SM%02d G12", ism);
      mui_->add(me_h03_[ism-1], histo);

    }

  }

}

void EBPedestalOnlineClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedestalOnlineTask/Gain12/EBPOT pedestal SM*");

}

void EBPedestalOnlineClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBPedestalOnlineClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPedestalOnlineClient: uncollate" << endl;

    if ( mui_ ) {

      for ( int ism = 1; ism <= 36; ism++ ) {

        mui_->removeCollate(me_h03_[ism-1]);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedestalOnlineTask/Gain12/EBPOT pedestal SM*");

}

void EBPedestalOnlineClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBPedestalOnlineClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedestalOnlineTask/Gain12/EBPOT pedestal SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalOnlineTask/Gain12/EBPOT pedestal SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( cloneME_ ) {
          if ( h03_[ism-1] ) delete h03_[ism-1];
          sprintf(histo, "ME EBPOT pedestal SM%02d G12", ism);
          h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
        } else {
          h03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
        }
      }
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 50.;

    float num03;
    float mean03;
    float rms03;

    if ( g03_[ism-1] ) g03_[ism-1]->Reset();

    if ( p03_[ism-1] ) p03_[ism-1]->Reset();

    if ( r03_[ism-1] ) r03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num03  = -1.;
        mean03 = -1.;
        rms03  = -1.;

        if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

        bool update_channel = false;

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
          if ( abs(mean03 - expectedMean_) > discrepancyMean_ )
            val = 0.;
          if ( rms03 > RMSThreshold_ )
            val = 0.;
          if ( g03_[ism-1] ) g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);

          if ( p03_[ism-1] ) p03_[ism-1]->Fill(mean03);
          if ( r03_[ism-1] ) r03_[ism-1]->Fill(rms03);

        }

      }
    }

  }

}

void EBPedestalOnlineClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBPedestalOnlineClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PedestalOnlineTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PEDESTAL ONLINE</span></h2> " << endl;
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

  string imgNameQual, imgNameMean, imgNameRMS, imgName, meName;

  TCanvas* cQual = new TCanvas("cQual", "Temp", 2*csize, csize);
  TCanvas* cMean = new TCanvas("cMean", "Temp", csize, csize);
  TCanvas* cRMS = new TCanvas("cRMS", "Temp", csize, csize);

  TH2F* obj2f;
  TH1F* obj1f;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    // Quality plots

    imgNameQual = "";

    obj2f = g03_[ism-1];

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
      dummy.Draw("text,same");
      cQual->Update();
      cQual->SaveAs(imgName.c_str());

    }

    // Mean distributions

    imgNameMean = "";

    obj1f = p03_[ism-1];

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1 ,"_" );
        }
      }
      imgNameMean = meName + ".png";
      imgName = htmlDir + imgNameMean;

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
      cMean->SaveAs(imgName.c_str());
      gPad->SetLogy(0);

    }

    // RMS distributions

    obj1f = r03_[ism-1];

    imgNameRMS = "";

    if ( obj1f ) {

      meName = obj1f->GetName();

      for ( unsigned int i = 0; i < meName.size(); i++ ) {
        if ( meName.substr(i, 1) == " " )  {
          meName.replace(i, 1, "_");
        }
      }
      imgNameRMS = meName + ".png";
      imgName = htmlDir + imgNameRMS;

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

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    if ( imgNameQual.size() != 0 )
      htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameQual << "\"></td>" << endl;
    else
      htmlFile << "<img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;
    htmlFile << "<tr>" << endl;

    if ( imgNameMean.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameMean << "\"></td>" << endl;
    else
      htmlFile << "<img src=\"" << " " << "\"></td>" << endl;

    if ( imgNameRMS.size() != 0 )
      htmlFile << "<td><img src=\"" << imgNameRMS << "\"></td>" << endl;
    else
      htmlFile << "<img src=\"" << " " << "\"></td>" << endl;

    htmlFile << "</tr>" << endl;

    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cQual;
  delete cMean;
  delete cRMS;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

