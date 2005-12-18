/*
 * \file EBPedPreSampleClient.cc
 * 
 * $Date: 2005/12/18 15:28:41 $
 * $Revision: 1.46 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedPreSampleClient.h>

EBPedPreSampleClient::EBPedPreSampleClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    h03_[i] = 0;

    sprintf(histo, "EBPT pedestal PreSample quality G12 SM%02d", i+1);
    g03_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBPT pedestal PreSample mean G12 SM%02d", i+1);
    p03_[i] = new TH1F(histo, histo, 100, 150., 250.);

    sprintf(histo, "EBPT pedestal PreSample rms G12 SM%02d", i+1);
    r03_[i] = new TH1F(histo, histo, 100, 0.,  10.);

  }

   expectedMean_ = 200.0;
   discrepancyMean_ = 20.0;
   RMSThreshold_ = 2.0;

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBPedPreSampleClient::~EBPedPreSampleClient(){

  for ( int i = 0; i < 36; i++ ) {

    if ( h03_[i] ) delete h03_[i];

    delete g03_[i];

    delete p03_[i];

    delete r03_[i];

  }

}

void EBPedPreSampleClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBPedPreSampleClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBPedPreSampleClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBPedPreSampleClient: beginRun" << endl;

  jevt_ = 0;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( h03_[ism-1] ) delete h03_[ism-1];
    h03_[ism-1] = 0;

    g03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

      }
    }

    p03_[ism-1]->Reset();

    r03_[ism-1]->Reset();

  }

  this->subscribe();

}

void EBPedPreSampleClient::endJob(void) {

  if ( verbose_ ) cout << "EBPedPreSampleClient: endJob, ievt = " << ievt_ << endl;

}

void EBPedPreSampleClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  if ( verbose_ ) cout << "EBPedPreSampleClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
//  MonPedestalsDat p;
//  map<EcalLogicID, MonPedestalsDat> dataset;

  cout << "Writing MonPedPreSampleDatObjects to database ..." << endl;

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

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "G12 (" << ie << "," << ip << ") " << num03  << " " << mean03 << " " << rms03  << endl;
          }

//          p.setPedMeanG12(mean03);
//          p.setPedRMSG12(rms03);

//          if ( g03_[ism-1]->GetBinContent(g03_[ism-1]->GetBin(ie, ip)) == 1. ) {
//             p.setTaskStatus(true);
//          } else {
//             p.setTaskStatus(false);
//          }

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
//              dataset[ecid] = p;
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
//      econn->insertDataSet(&dataset, runiov, runtag );
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  this->unsubscribe();

}

void EBPedPreSampleClient::subscribe(void){

  if ( verbose_ ) cout << "EBPedPreSampleClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPedPreSampleClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBPT pedestal PreSample SM%02d G12", ism);
      me_h03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPedPreSampleTask/Gain12");
      sprintf(histo, "*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM%02d G12", ism);
      mui_->add(me_h03_[ism-1], histo);

    }

  }

}

void EBPedPreSampleClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBPedPreSampleClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPedPreSampleClient: uncollate" << endl;

    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( bei ) {

      Char_t histo[80];

      for ( int ism = 1; ism <= 36; ism++ ) {

        sprintf(histo, "EBPT pedestal PreSample SM%02d G12", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBPedPreSampleTask/Gain12");
        bei->removeElement(histo);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBPedPreSampleClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM%02d G12", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM%02d G12", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        sprintf(histo, "ME EBPT pedestal PreSample SM%02d G12", ism);
        h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    const float n_min_tot = 1000.;
    const float n_min_bin = 50.;

    float num03;
    float mean03;
    float rms03;

    g03_[ism-1]->Reset();

    p03_[ism-1]->Reset();

    r03_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num03  = -1.;
        mean03 = -1.;
        rms03  = -1.;

        g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), 2.);

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
          g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);

          p03_[ism-1]->Fill(mean03);
          r03_[ism-1]->Fill(rms03);

        }

      }
    }

  }

}

void EBPedPreSampleClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBPedPreSampleClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PedPreSampleTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PEDESTAL ON PRESAMPLE</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .jpg files from existing histograms

  int csize = 250;

  double histMax = 1.e15;

  int pCol3[3] = { 2, 3, 5 };

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);

  string imgNameQual , imgNameMean , imgNameRMS , imgName , meName;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {
    
    if ( g03_[ism-1] && p03_[ism-1] && r03_[ism-1] ) {

      TH2F* obj2f = 0; 

      meName = g03_[ism-1]->GetName();
      obj2f = g03_[ism-1];

      TCanvas *cQual = new TCanvas("cQual" , "Temp", 2*csize , csize );
      for ( unsigned int iQual = 0 ; iQual < meName.size(); iQual++ ) {
        if ( meName.substr(iQual, 1) == " " )  {
          meName.replace(iQual, 1, "_");
        }
      }
      imgNameQual = meName + ".jpg";
      imgName = htmlDir + imgNameQual;
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

      // Mean distributions
        
      TH1F* obj1f = 0; 
        
      meName = p03_[ism-1]->GetName();
      obj1f = p03_[ism-1];
        
      TCanvas *cMean = new TCanvas("cMean" , "Temp", csize , csize );
      for ( unsigned int iMean=0 ; iMean < meName.size(); iMean++ ) {
        if ( meName.substr(iMean,1) == " " )  {
          meName.replace(iMean, 1 ,"_" );
        }
      }
      imgNameMean = meName + ".jpg";
      imgName = htmlDir + imgNameMean;
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cMean->Update();
      TPaveStats* stMean = dynamic_cast<TPaveStats*>(obj1f->FindObject("stats"));
      if ( stMean ) {
        stMean->SetX1NDC(0.6);
        stMean->SetY1NDC(0.75);
      }
      cMean->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
      delete cMean;
      
      // RMS distributions
      
          meName = r03_[ism-1]->GetName();
          obj1f = r03_[ism-1];

      TCanvas *cRMS = new TCanvas("cRMS" , "Temp", csize , csize );
      for ( unsigned int iRMS=0 ; iRMS < meName.size(); iRMS++ ) {
        if ( meName.substr(iRMS,1) == " " )  {
          meName.replace(iRMS, 1, "_");
        }
      }
      imgNameRMS = meName + ".jpg";
      imgName = htmlDir + imgNameRMS;
      gStyle->SetOptStat("euomr");
      obj1f->SetStats(kTRUE);
      if ( obj1f->GetMaximum(histMax) > 0. ) {
        gPad->SetLogy(1);
      } else {
        gPad->SetLogy(0);
      }
      obj1f->Draw();
      cRMS->Update();
      TPaveStats* stRMS = dynamic_cast<TPaveStats*>(obj1f->FindObject("stats"));
      if ( stRMS ) {
        stRMS->SetX1NDC(0.6);
        stRMS->SetY1NDC(0.75);
      }
      cRMS->SaveAs(imgName.c_str());
      gPad->SetLogy(0);
      delete cRMS;

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

  }

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

