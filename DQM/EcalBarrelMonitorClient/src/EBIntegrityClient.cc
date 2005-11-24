/*
 * \file EBIntegrityClient.cc
 * 
 * $Date: 2005/11/24 12:43:53 $
 * $Revision: 1.32 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBIntegrityClient.h>

EBIntegrityClient::EBIntegrityClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;


  Char_t histo[50];

  h00_ = 0;
  for ( int i = 0; i < 36; i++ ) {

    h_[i] = 0;

    h01_[i] = 0;
    h02_[i] = 0;
    h03_[i] = 0;
    h04_[i] = 0;

    sprintf(histo, "EBPT data integrity quality SM%02d", i+1);
    g01_[i] = new TH2F(histo, histo, 85, 0., 85., 20, 0., 20.);

  }

  threshCry_ = 0.;

}

EBIntegrityClient::~EBIntegrityClient(){

  this->unsubscribe();


  if ( h00_ ) delete h00_;
  for ( int i = 0; i < 36; i++ ) {
    
    if ( h_[i] ) delete h_[i];
  
    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    if ( h03_[i] ) delete h03_[i];
    if ( h04_[i] ) delete h04_[i];

    delete g01_[i];

  }

}

void EBIntegrityClient::beginJob(const edm::EventSetup& c){

  cout << "EBIntegrityClient: beginJob" << endl;

  ievt_ = 0;

}

void EBIntegrityClient::beginRun(const edm::EventSetup& c){

  cout << "EBIntegrityClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

  if ( h00_ ) delete h00_;
  h00_ = 0;
  for ( int i = 0; i < 36; i++ ) {

    if ( h_[i] ) delete h_[i];

    if ( h01_[i] ) delete h01_[i];
    if ( h02_[i] ) delete h02_[i];
    if ( h03_[i] ) delete h03_[i];
    if ( h04_[i] ) delete h04_[i];
    h_[i] = 0;
    h01_[i] = 0;
    h02_[i] = 0;
    h03_[i] = 0;
    h04_[i] = 0;

    g01_[i]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        g01_[i]->SetBinContent(g01_[i]->GetBin(ie, ip), 2.);

      }
    }

  }

}

void EBIntegrityClient::endJob(void) {

  cout << "EBIntegrityClient: endJob, ievt = " << ievt_ << endl;

}

void EBIntegrityClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBIntegrityClient: endRun, jevt = " << jevt_ << endl;

  if ( jevt_ == 0 ) return;

  EcalLogicID ecid;
  RunConsistencyDat c;
  map<EcalLogicID, RunConsistencyDat> dataset;

  cout << "Writing RunConsistencyDatObjects to database ..." << endl;

  const float n_min_bin = 0.;

  float num00;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        float numEventsinCry = 0.;
        if ( h_[ism-1] ) numEventsinCry = h_[ism-1]->GetBinEntries(h_[ism-1]->GetBin(ie, ip)) / 3.;
        
        // cout << "Number of events per crystal (" << ie << "," << ip << ") SM " << ism << " " << numEventsinCry << endl;

        if ( numEventsinCry > n_min_bin ) {
          
          num00 = -1.;
          
          if ( h00_ ) {
            num00  = h00_->GetBinContent(h00_->GetBin(ie, ip));
          }
          
          num01 = num02 = num03 = num04 = -1.;
          
          bool update_channel = false;
          bool update_channel_db = false;
          
          if ( h01_[ism-1] ) {
            num01  = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(ie, ip));
            update_channel = true;
            if ( num01 > 0 ) update_channel_db = true;
          }

          if ( h02_[ism-1] ) {
            num02  = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(ie, ip));
            update_channel = true;
            if ( num02 > 0 ) update_channel_db = true;
          }

          if ( h03_[ism-1] ) {
            num03  = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(ie, ip));
            update_channel = true;
            if ( num03 > 0 ) update_channel_db = true;
          }

          if ( h04_[ism-1] ) {
            num04  = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(ie, ip));
            update_channel = true;
            if ( num04 > 0 ) update_channel_db = true;
          }

          if ( update_channel ) {

            if ( ie == 1 && ip == 1 ) {

              cout << "Inserting dataset for SM=" << ism << endl;

              cout << "(" << ie << "," << ip << ") " << num00 << " " << num01 << " " << num02 << " " << num03 << " " << num04 << endl;

            }

            if ( update_channel_db ) {
              c.setExpectedEvents(0);
              c.setProblemsInGain(int(num01));
              c.setProblemsInId(int(num02));
              c.setProblemsInSample(int(-999));
              c.setProblemsInADC(int(-999));
            }

            float val;

            if ( g01_[ism-1] ) {
              val = 1.;
              if ( (( num01 + num02 ) / numEventsinCry / 2. ) > threshCry_ ) 
                val = 0.;
              g01_[ism-1]->SetBinContent(g01_[ism-1]->GetBin(ie, ip), val);
            }

            if ( update_channel_db ) {
              if ( econn ) {
                try {
                  ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
                  dataset[ecid] = c;
                } catch (runtime_error &e) {
                  cerr << e.what() << endl;
                }
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
      econn->insertDataSet(&dataset, runiov, runtag );
      cout << "done." << endl; 
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBIntegrityClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/DCC size error");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/Gain/EI gain SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/ChId/EI ChId SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/TTId/EI TTId SM*");
  mui_->subscribe("*/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");

}

void EBIntegrityClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/DCC size error");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/Gain/EI gain SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/ChId/EI ChId SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/TTId/EI TTId SM*");
  mui_->subscribeNew("*/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");

}

void EBIntegrityClient::unsubscribe(void){
  
  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/DCC size error");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/Gain/EI gain SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/ChId/EI ChId SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/TTId/EI TTId SM*");
  mui_->unsubscribe("*/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM*");

}

void EBIntegrityClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 )
    cout << "EBIntegrityClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];
  
  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/DCC size error");
  me = mui_->get(histo);
  if ( me ) {
    cout << "Found '" << histo << "'" << endl;
    ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    if ( ob ) {
      if ( h00_ ) delete h00_;
      h00_ = dynamic_cast<TH1F*> ((ob->operator->())->Clone());
    }
  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedPreSampleTask/Gain12/EBPT pedestal PreSample SM%02d G12", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h_[ism-1] ) delete h_[ism-1];
        h_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/Gain/EI gain SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        h01_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/ChId/EI ChId SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        h02_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/TTId/EI TTId SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        h03_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone());
      }
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EcalIntegrity/TTBlockSize/EI TTBlockSize SM%02d", ism);
    me = mui_->get(histo);
    if ( me ) {
      cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h04_[ism-1] ) delete h04_[ism-1];
        h04_[ism-1] = dynamic_cast<TH2F*> ((ob->operator->())->Clone());
      }
    }

  }

}

void EBIntegrityClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBIntegrityClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:IntegrityTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl; 
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">INTEGRITY</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=white>channel is missing</td></table>" << endl;
  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .jpg files from existing histograms

  int csize = 250;

  int pCol3[3] = { 2, 3, 10 };

  TH2C dummy( "dummy", "dummy for sm", 85, 0., 85., 20, 0., 20. );
  for( int i = 0; i < 68; i++ ) {
    int a = 2 + ( i/4 ) * 5;
    int b = 2 + ( i%4 ) * 5;
    dummy.Fill( a, b, i+1 );
  }
  dummy.SetMarkerSize(2);

  string imgNameDCC, imgNameQual, imgNameME[4], imgName , meName;
  
  if ( h00_ ) {
    
    // DCC size error
    
    TH1F* obj1f = 0; 
    meName = h00_->GetName();
    obj1f = h00_;
    
    TCanvas *cDCC = new TCanvas("cDCC" , "Temp", 2*csize , csize );
    for ( unsigned int iDCC = 0 ; iDCC < meName.size(); iDCC++ ) {
      if ( meName.substr(iDCC, 1) == " " )  {
        meName.replace(iDCC, 1, "_");
      }
    }
    imgNameDCC = meName + ".jpg";
    imgName = htmlDir + imgNameDCC;
    gStyle->SetOptStat(" ");
    gStyle->SetPalette(3, pCol3);
    obj1f->GetXaxis()->SetNdivisions(17);
    obj1f->GetYaxis()->SetNdivisions(4);
    cDCC->SetGridx();
    cDCC->SetGridy();
    obj1f->SetMinimum(-0.00000001);
    obj1f->SetMaximum(2.0);
    obj1f->Draw("col");
    dummy.Draw("text,same");
    cDCC->Update();
    cDCC->SaveAs(imgName.c_str());
    delete cDCC;
    
  }
  
  htmlFile << "<h3><strong>DCC size error</strong></h3>" << endl;
  
  if ( imgNameDCC.size() != 0 ) 
    htmlFile << "<p><img src=\"" << imgNameDCC << "\"></p>" << endl;
  else
    htmlFile << "<p><img src=\"" << " " << "\"></p>" << endl;
  
  htmlFile << "<br>" << endl;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {
    
    if ( g01_[ism-1] && h01_[ism-1] && h02_[ism-1] 
         && h03_[ism-1] && h04_[ism-1] ) {

      // Quality plots
      
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

      // Monitoring elements plots
      
      for ( int iCanvas = 1; iCanvas <= 4; iCanvas++ ) {
      
        switch ( iCanvas ) {
        case 1:
          meName = h01_[ism-1]->GetName();
          obj2f = h01_[ism-1];
          break;
        case 2:
          meName = h02_[ism-1]->GetName();
          obj2f = h02_[ism-1];
          break;
        case 3:
          meName = h03_[ism-1]->GetName();
          obj2f = h03_[ism-1];
          break;
        case 4:
          meName = h04_[ism-1]->GetName();
          obj2f = h04_[ism-1];
          break;
        default:
          break;
        }
        
        TCanvas *cMe = new TCanvas("cMe" , "Temp", 2*csize , csize );
        for ( unsigned int iMe = 0 ; iMe < meName.size(); iMe++ ) {
          if ( meName.substr(iMe, 1) == " " )  {
            meName.replace(iMe, 1, "_");
          }
        }
        imgNameME[iCanvas-1] = meName + ".jpg";
        imgName = htmlDir + imgNameME[iCanvas-1];
        gStyle->SetOptStat(" ");
        gStyle->SetPalette(1, 0);
        obj2f->GetXaxis()->SetNdivisions(17);
        obj2f->GetYaxis()->SetNdivisions(4);
        cMe->SetGridx();
        cMe->SetGridy();
        obj2f->SetMinimum(-0.00000001);
        obj2f->SetMaximum();
        obj2f->Draw("colz");
        cMe->Update();
        cMe->SaveAs(imgName.c_str());
        delete cMe;

      }

    }

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
    
    if ( imgNameQual.size() != 0 ) 
      htmlFile << "<p><img src=\"" << imgNameQual << "\"></p>" << endl;
    else
      htmlFile << "<p><img src=\"" << " " << "\"></p>" << endl;

    htmlFile << "<br>" << endl;
    
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 4 ; iCanvas++ ) {
      
      if ( imgNameME[iCanvas-1].size() != 0 ) 
        htmlFile << "<td><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<img src=\"" << " " << "\"></td>" << endl;
      
    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;
    
  }
  
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
  
}

