/*
 * \file EBPedPreSampleClient.cc
 * 
 * $Date: 2005/11/16 13:40:38 $
 * $Revision: 1.21 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedPreSampleClient.h>

EBPedPreSampleClient::EBPedPreSampleClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  Char_t histo[50];

  for ( int i = 0; i < 36; i++ ) {

    h01_[i] = 0;

    sprintf(histo, "EBPT pedestal PreSample quality G01 SM%02d", i+1);
    g01_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBPT pedestal PreSample mean G01 SM%02d", i+1);
    p01_[i] = new TH1F(histo, histo, 100, 150., 250.);

    sprintf(histo, "EBPT pedestal PreSample rms G01 SM%02d", i+1);
    r01_[i] = new TH1F(histo, histo, 100, 0., 10.);

  }


   expectedMean_ = 2048;
  
   discrepancyMean_ = 2044;
  
   RMSThreshold_ = 2.2;
   
}

EBPedPreSampleClient::~EBPedPreSampleClient(){

  this->unsubscribe();

  for ( int i = 0; i < 36; i++ ) {

    if ( h01_[i] ) delete h01_[i];

    delete g01_[i];

    delete p01_[i];

    delete r01_[i];

  }

}

void EBPedPreSampleClient::beginJob(const edm::EventSetup& c){

  cout << "EBPedPreSampleClient: beginJob" << endl;

  ievt_ = 0;

}

void EBPedPreSampleClient::beginRun(const edm::EventSetup& c){

  cout << "EBPedPreSampleClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( h01_[ism-1] ) delete h01_[ism-1];
    h01_[ism-1] = 0;

    g01_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), 2.);

      }
    }

    p01_[ism-1]->Reset();

    r01_[ism-1]->Reset();

  }

}

void EBPedPreSampleClient::endJob(void) {

  cout << "EBPedPreSampleClient: endJob, ievt = " << ievt_ << endl;

}

void EBPedPreSampleClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBPedPreSampleClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
//  MonPedestalsDat p;
//  map<EcalLogicID, MonPedestalsDat> dataset;

  cout << "Writing MonPedPreSampleDatObjects to database ..." << endl;

  float n_min_tot = 1000.;
  float n_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01;
    float mean01;
    float rms01;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = -1.;
        mean01 = -1.;
        rms01  = -1.;

        bool update_channel = false;

        if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
          num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "G01 (" << ie << "," << ip << ") " << num01  << " "
                                                       << mean01 << " "
                                                       << rms01  << endl;
          }

//          p.setPedMeanG01(mean01);
//          p.setPedRMSG01(rms01);

          float val;

          if ( g01_[ism-1] ) {
            val = 1.;
            if ( abs(mean01 - expectedMean_) > discrepancyMean_ )
              val = 0.;
            if ( rms01 > RMSThreshold_ )
              val = 0.;
            g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);
          }

          if ( p01_[ism-1] ) p01_[ism-1]->Fill(mean01);
          if ( r01_[ism-1] ) r01_[ism-1]->Fill(rms01);

//          p.setTaskStatus(1);

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

}

void EBPedPreSampleClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedPreSampleTask/Gain01/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedPreSampleTask/Gain01/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedPreSampleTask/Gain01/EBPT pedestal PreSample SM*");

}

void EBPedPreSampleClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )  
    cout << "EBPedPreSampleClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedPreSampleTask/Gain01/EBPT pedestal PreSample SM%02d G01", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
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

  string imgNameQual , imgNameMean , imgNameRMS , imgName , meName;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {
    
    if ( g01_[ism-1] && p01_[ism-1] && r01_[ism-1] ) {

      TH2F* obj2f = 0; 

      meName = g01_[ism-1]->GetName();
      obj2f = g01_[ism-1];

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
        
      meName = p01_[ism-1]->GetName();
      obj1f = p01_[ism-1];
        
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
      gPad->SetLogy(0);
      TPaveStats* stMean = dynamic_cast<TPaveStats*>(obj1f->FindObject("stats"));
      if ( stMean ) {
        stMean->SetX1NDC(0.6);
        stMean->SetY1NDC(0.75);
      }
      cMean->SaveAs(imgName.c_str());
      delete cMean;
      
      // RMS distributions
      
          meName = r01_[ism-1]->GetName();
          obj1f = r01_[ism-1];

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
      gPad->SetLogy(0);
      TPaveStats* stRMS = dynamic_cast<TPaveStats*>(obj1f->FindObject("stats"));
      if ( stRMS ) {
        stRMS->SetX1NDC(0.6);
        stRMS->SetY1NDC(0.75);
      }
      cRMS->SaveAs(imgName.c_str());
      delete cRMS;

      htmlFile << "</h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
      htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
      htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
      htmlFile << "<tr align=\"center\">" << endl;

      if ( imgNameQual != " " ) 
        htmlFile << "<td colspan=\"2\"><img src=\" " << imgNameQual << "\"></td>" << endl;
      else
        htmlFile << "<img src=\" " << " " << "\"></td>" << endl;

      htmlFile << "</tr>" << endl;
      htmlFile << "<tr>" << endl;

      if ( imgNameMean != " " ) 
        htmlFile << "<td><img src=\" " << imgNameMean << "\"></td>" << endl;
      else
        htmlFile << "<img src=\" " << " " << "\"></td>" << endl;
      
      if ( imgNameRMS != " " ) 
        htmlFile << "<td><img src=\" " << imgNameRMS << "\"></td>" << endl;
      else
        htmlFile << "<img src=\" " << " " << "\"></td>" << endl;

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

