/*
 * \file EBPedestalClient.cc
 * 
 * $Date: 2005/11/26 20:42:49 $
 * $Revision: 1.30 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>

EBPedestalClient::EBPedestalClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    h01_[i] = 0;
    h02_[i] = 0;
    h03_[i] = 0;

    sprintf(histo, "EBPT pedestal quality G01 SM%02d", i+1);
    g01_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBPT pedestal quality G06 SM%02d", i+1);
    g02_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);
    sprintf(histo, "EBPT pedestal quality G12 SM%02d", i+1);
    g03_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBPT pedestal mean G01 SM%02d", i+1);
    p01_[i] = new TH1F(histo, histo, 100, 150., 250.);
    sprintf(histo, "EBPT pedestal mean G06 SM%02d", i+1);
    p02_[i] = new TH1F(histo, histo, 100, 150., 250.);
    sprintf(histo, "EBPT pedestal mean G12 SM%02d", i+1);
    p03_[i] = new TH1F(histo, histo, 100, 150., 250.);

    sprintf(histo, "EBPT pedestal rms G01 SM%02d", i+1);
    r01_[i] = new TH1F(histo, histo, 100, 0., 10.);
    sprintf(histo, "EBPT pedestal rms G06 SM%02d", i+1);
    r02_[i] = new TH1F(histo, histo, 100, 0., 10.);
    sprintf(histo, "EBPT pedestal rms G12 SM%02d", i+1);
    r03_[i] = new TH1F(histo, histo, 100, 0., 10.);

  }

  expectedMean_[1] = 200;
  expectedMean_[2] = 200;
  expectedMean_[3] = 200;

  discrepancyMean_[1] = 20;
  discrepancyMean_[2] = 20;
  discrepancyMean_[3] = 20;

  RMSThreshold_[1] = 1;
  RMSThreshold_[2] = 1;
  RMSThreshold_[3] = 2;
 
}

EBPedestalClient::~EBPedestalClient(){

  for ( int i = 0 ; i < 36 ; i++ ) {

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    if ( h03_[i] ) delete h03_[i];

    delete g01_[i];
    delete g02_[i];
    delete g03_[i];

    delete p01_[i];
    delete p02_[i];
    delete p03_[i];

    delete r01_[i];
    delete r02_[i];
    delete r03_[i];

  }

}

void EBPedestalClient::beginJob(const edm::EventSetup& c){

  cout << "EBPedestalClient: beginJob" << endl;

  ievt_ = 0;

  this->subscribe();

}

void EBPedestalClient::beginRun(const edm::EventSetup& c){

  cout << "EBPedestalClient: beginRun" << endl;

  jevt_ = 0;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( h01_[ism-1] ) delete h01_[ism-1];
    if ( h02_[ism-1] ) delete h02_[ism-1];
    if ( h03_[ism-1] ) delete h03_[ism-1];
    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;

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

  }

}

void EBPedestalClient::endJob(void) {

  cout << "EBPedestalClient: endJob, ievt = " << ievt_ << endl;

  this->unsubscribe();

}

void EBPedestalClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBPedestalClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  MonPedestalsDat p;
  map<EcalLogicID, MonPedestalsDat> dataset;

  cout << "Writing MonPedestalsDatObjects to database ..." << endl;

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

          float val;

          if ( g01_[ism-1] ) {
            val = 1.;
            if ( abs(mean01 - expectedMean_[1]) > discrepancyMean_[1] )
              val = 0.;
            if ( rms01 > RMSThreshold_[1] )
              val = 0.;
            g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);
          }
          
          if ( p01_[ism-1] ) p01_[ism-1]->Fill(mean01);
          if ( r01_[ism-1] ) r01_[ism-1]->Fill(rms01);

          p.setPedMeanG6(mean02);
          p.setPedRMSG6(rms02);

          if ( g02_[ism-1] ) {
            val = 1.;
            if ( abs(mean02 - expectedMean_[2]) > discrepancyMean_[2] )
              val = 0.;
            if ( rms02 > RMSThreshold_[2] )
              val = 0.;
            g02_[ism-1]->SetBinContent(g02_[ism-1]->GetBin(ie, ip), val);
          }

          if ( p02_[ism-1] ) p02_[ism-1]->Fill(mean02);
          if ( r02_[ism-1] ) r02_[ism-1]->Fill(rms02);

          p.setPedMeanG12(mean03);
          p.setPedRMSG12(rms03);

          if ( g03_[ism-1] ) {
            val = 1.;
            if ( abs(mean03 - expectedMean_[3]) > discrepancyMean_[3] )
              val = 0.;
            if ( rms03 > RMSThreshold_[3] )
              val = 0.;
            g03_[ism-1]->SetBinContent(g03_[ism-1]->GetBin(ie, ip), val);
          }

          if ( p03_[ism-1] ) p03_[ism-1]->Fill(mean03);
          if ( r03_[ism-1] ) r03_[ism-1]->Fill(rms03);

          p.setTaskStatus(1);

          if ( econn ) {
            try {
              ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
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
      econn->insertDataSet(&dataset, runiov, runtag );
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBPedestalClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

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

  }

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

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

//    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM%02d G01", ism);
    sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain01/EBPT pedestal SM%02d G01", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        sprintf(histo, "ME EBPT pedestal SM%02d G01", ism);
        h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
      }
    }

//    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM%02d G06", ism);
    sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain06/EBPT pedestal SM%02d G06", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        sprintf(histo, "ME EBPT pedestal SM%02d G06", ism);
        h02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
      }
    }

//    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM%02d G12", ism);
    sprintf(histo, "EcalBarrel/Sums/EBPedestalTask/Gain12/EBPT pedestal SM%02d G12", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        sprintf(histo, "ME EBPT pedestal SM%02d G12", ism);
        h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
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

  string imgNameQual[3] , imgNameMean[3] , imgNameRMS[3] , imgName , meName;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {
    
    if ( g01_[ism-1] && g02_[ism-1] && g03_[ism-1] &&
         p01_[ism-1] && p02_[ism-1] && p03_[ism-1] &&
         r01_[ism-1] && r02_[ism-1] && r03_[ism-1] ) {

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

        // Mean distributions
        
        TH1F* obj1f = 0; 
        
        switch ( iCanvas ) {
          case 1:
            meName = p01_[ism-1]->GetName();
            obj1f = p01_[ism-1];
            break;
          case 2:
            meName = p02_[ism-1]->GetName();
            obj1f = p02_[ism-1];
            break;
          case 3:
            meName = p03_[ism-1]->GetName();
            obj1f = p03_[ism-1];
            break;
          default:
            break;
          }
        
        TCanvas *cMean = new TCanvas("cMean" , "Temp", csize , csize );
        for ( unsigned int iMean=0 ; iMean < meName.size(); iMean++ ) {
          if ( meName.substr(iMean,1) == " " )  {
            meName.replace(iMean, 1 ,"_" );
          }
        }
        imgNameMean[iCanvas-1] = meName + ".jpg";
        imgName = htmlDir + imgNameMean[iCanvas-1];
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
        TPaveStats* stMean = dynamic_cast<TPaveStats*>(obj1f->FindObject("stats"));
        if ( stMean ) {
          stMean->SetX1NDC(0.6);
          stMean->SetY1NDC(0.75);
        }
        cMean->SaveAs(imgName.c_str());
        delete cMean;
        
        // RMS distributions
        
        switch ( iCanvas ) {
          case 1:
            meName = r01_[ism-1]->GetName();
            obj1f = r01_[ism-1];
            break;
          case 2:
            meName = r02_[ism-1]->GetName();
            obj1f = r02_[ism-1];
            break;
          case 3:
            meName = r03_[ism-1]->GetName();
            obj1f = r03_[ism-1];
            break;
          default:
            break;
          }
        
        TCanvas *cRMS = new TCanvas("cRMS" , "Temp", csize , csize );
        for ( unsigned int iRMS=0 ; iRMS < meName.size(); iRMS++ ) {
          if ( meName.substr(iRMS,1) == " " )  {
            meName.replace(iRMS, 1, "_");
          }
        }
        imgNameRMS[iCanvas-1] = meName + ".jpg";
        imgName = htmlDir + imgNameRMS[iCanvas-1];
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

        if ( imgNameMean[iCanvas-1].size() != 0 ) 
          htmlFile << "<td><img src=\"" << imgNameMean[iCanvas-1] << "\"></td>" << endl;
        else
          htmlFile << "<img src=\"" << " " << "\"></td>" << endl;
        
        if ( imgNameRMS[iCanvas-1].size() != 0 ) 
          htmlFile << "<td><img src=\"" << imgNameRMS[iCanvas-1] << "\"></td>" << endl;
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

