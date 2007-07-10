#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalCMCTClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESDQMUtils.h"

#include "TStyle.h"
#include "TH2F.h"
#include "TPaveText.h"

ESPedestalCMCTClient::ESPedestalCMCTClient(const ParameterSet& ps) {
  
  writeHisto_     = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_      = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_       = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESPedestalCMCT");
  rootFolder_     = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_        = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/CT");
  htmlName_       = ps.getUntrackedParameter<string>("htmlName","ESPedestalCMCT.html");  
  cmnThreshold_   = ps.getUntrackedParameter<double>("cmnThreshold", 3);
  sta_            = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;

}

ESPedestalCMCTClient::~ESPedestalCMCTClient(){
}

void ESPedestalCMCTClient::beginJob(const EventSetup& context){

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/PedestalCMCT");
    dbe_->rmdir("ES/QT/PedestalCMCT");
  }

}

void ESPedestalCMCTClient::endJob(){
  
  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }

  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/PedestalCMCT");  

  if ( init_ ) this->cleanup();
}

void ESPedestalCMCTClient::setup() {

  init_ = true;

  Char_t hist[200];
  
  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCMCT");

    for (int i=0; i<3; ++i) {
      sprintf(hist, "ES QT CMCT Mean TS %d", i+1);
      meMean_[i] = dbe_->book1D(hist, hist, 100, -50, 50);
      sprintf(hist, "ES QT CMCT RMS TS %d", i+1);
      meRMS_[i] = dbe_->book1D(hist, hist, 100, -50, 50);
    }

    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {
	for (int k=0; k<3; ++k) {
	  sprintf(hist, "ES CMCT Quality Box %d Plane %d TS %d", i+1, j+1, k+1);
	  meCMCol_[i][j][k] = dbe_->book2D(hist, hist, 2, 0, 2, 5, 0, 5);
	  
	}
      }
    }
  }
  
}

void ESPedestalCMCTClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCMCT");

    for (int i=0; i<3; ++i) {
      if (meMean_[i]) dbe_->removeElement( meMean_[i]->getName() );
      if (meRMS_[i]) dbe_->removeElement( meRMS_[i]->getName() );
      meMean_[i] = 0;
      meRMS_[i] = 0;
    }

    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {
	for (int k=0; k<3; ++k) {
	  if (meCMCol_[i][j][k]) dbe_->removeElement( meCMCol_[i][j][k]->getName() );
	meCMCol_[i][j][k] = 0;
	
	}
      }
    }
  }

  init_ = false;
}

void ESPedestalCMCTClient::analyze(const Event& e, const EventSetup& context){
  
  if ( ! init_ ) this->setup();

  int runNum = e.id().run();

  if (runNum != run_) { 

    if (run_ > 0) {
      Char_t runNum_s[50];
      sprintf(runNum_s, "%08d", run_);
      outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
      
      if (writeHTML_) {
	doQT();
	htmlOutput(run_, htmlDir_, htmlName_);
      }

      if (writeHisto_) dbe_->save(outputFile_);
    }

    run_ = runNum; 
    count_ = 0;
  }

  count_++;

  if ((count_ % dumpRate_) == 0) {
    if (writeHTML_) {
      doQT();
      htmlOutput(runNum, htmlDir_, htmlName_);
    }
    if (writeHisto_) dbe_->save(outputFile_);
  }
  
}

void ESPedestalCMCTClient::doQT() {

  for (int i=0; i<3; ++i) {
    ESDQMUtils::resetME( meMean_[i] );
    ESDQMUtils::resetME( meRMS_[i] );
  }

  int val = 0;
  for (int i=0; i<2; ++i) {    
    for (int j=0; j<6; ++j) {
      for (int k=0; k<2; ++k) {
	for (int m=0; m<5; ++m) {
	  for (int n=0; n<3; ++n) {
	    
	    int zside = (i==0)?1:-1;
	    MonitorElement * senME = dbe_->get(getMEName(zside, j+1, k+1, m+1, 0, n, 1));
	    
	    if (senME) {
	      MonitorElementT<TNamed>* sen = dynamic_cast<MonitorElementT<TNamed>*>(senME);           
	      TH1F *hCMSen = dynamic_cast<TH1F*> (sen->operator->());  	    
	      
	      if (hCMSen->GetRMS()>cmnThreshold_) val = 7;
	      else if (hCMSen->GetEntries() == 0) val = 5;
	      else val = 4;
	      
	      meCMCol_[i][j][n]->setBinContent(k+1, m+1, val) ;  
	      
	      if (hCMSen->GetEntries() != 0) {
		meMean_[n]->Fill(hCMSen->GetMean());
		meRMS_[n]->Fill(hCMSen->GetRMS());
	      }
	      
	    }
	    
	  }	
	}
      }
    }
  }

}

string ESPedestalCMCTClient::getMEName(const int & zside, const int & plane, const int & row, const int & col, const int & strip, const int & slot, const int & type) {
  
  Char_t hist[500];
  if (type == 0)
    sprintf(hist,"%sES/ESpedestalCMCTTask/ES Pedestal CM_S%d Z %d P %d Row %02d Col %02d Str %02d", rootFolder_.c_str(),slot,zside,plane,row,col,strip);
  else 
    sprintf(hist,"%sES/ESPedestalCMCTTask/ES Sensor CM_S%d Z %d P %d Row %02d Col %02d", rootFolder_.c_str(),slot,zside,plane,row,col);

  return hist;
}

void ESPedestalCMCTClient::htmlOutput(int run, string htmlDir, string htmlName) {

  cout<<"Going to output ESPedestalCMCTClient html ..."<<endl;
  
  Char_t run_s[50];
  sprintf(run_s, "%08d", run); 
  htmlDir = htmlDir+"/"+run_s;
  system(("/bin/mkdir -m 777 -p " + htmlDir).c_str());

  ofstream htmlFile;   
  htmlFile.open((htmlDir+"/"+htmlName).c_str()); 

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=UTF-8\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Preshower DQM : PedestalCMCTTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Cosmic Ray Test Common Mode</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>This strip has problems</td>" << endl;
  htmlFile << "<td bgcolor=lime>This strip has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>This strip is missing</td></table>" << endl;
  htmlFile << "<br>" << endl;

  // make plots
  string histName;
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1, 0);
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetGridStyle(1);

  TCanvas *cCMQ = new TCanvas("cCMQ", "cCMQ", 1200, 250);
  TCanvas *cCM  = new TCanvas("cCM",  "cCM",  900, 600);
  
  MonitorElementT<TNamed>* CMQ[2][6][3];
  TH2F* hCMQ[2][6][3];
  for (int i=0; i<2; ++i) {
    for (int j=0; j<6; ++j) {
      for (int k=0; k<3; ++k) {
	CMQ[i][j][k] = dynamic_cast<MonitorElementT<TNamed>*>(meCMCol_[i][j][k]);           
	hCMQ[i][j][k] = dynamic_cast<TH2F*> (CMQ[i][j][k]->operator->());  
      }
    }
  }

  gStyle->SetOptStat("");
  cCMQ->Divide(6,1);
  for (int k=0; k<3; ++k) {
    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {
	cCMQ->cd(j+1);
	gPad->SetGridx();
	gPad->SetGridy();
	hCMQ[i][j][k]->GetXaxis()->SetNdivisions(-102);
	hCMQ[i][j][k]->GetYaxis()->SetNdivisions(-105);
	hCMQ[i][j][k]->GetXaxis()->SetLabelSize(0.08);
	hCMQ[i][j][k]->GetYaxis()->SetLabelSize(0.08);
	hCMQ[i][j][k]->SetMinimum(-0.00000001);
	hCMQ[i][j][k]->SetMaximum(7.0);
	char tit[128]; sprintf(tit,"Box %d P %d TS %d",i+1,j+1,k+1);
	hCMQ[i][j][k]->SetTitle(tit);
	hCMQ[i][j][k]->Draw("col");
	gPad->Update();
	TPaveText *t = (TPaveText*) gPad->GetPrimitive("title");
	t->SetTextColor(4);
	t->SetTextSize(.1);
	t->SetBorderSize(0);
	t->SetX1NDC(0.00); t->SetX2NDC(1);
	t->SetY1NDC(0.93); t->SetY2NDC(1);
      }
      stringstream ts; ts << (k+1);
      histName = (i==0) ? htmlDir+"/PedestalCM_Quality_Box1_TS"+ts.str()+".png":htmlDir+"/PedestalCM_Quality_Box2_TS"+ts.str()+".png";
      cCMQ->SaveAs(histName.c_str());  
    }
  }
  
  // Plot Mean and RMS
  MonitorElementT<TNamed>* Mean[3];
  TH1F *hMean[3];
  MonitorElementT<TNamed>* RMS[3];
  TH1F *hRMS[3];
  for (int i=0; i<3; ++i) {
    Mean[i] = dynamic_cast<MonitorElementT<TNamed>*>(meMean_[i]);
    hMean[i] = dynamic_cast<TH1F*> (Mean[i]->operator->());
    RMS[i] = dynamic_cast<MonitorElementT<TNamed>*>(meRMS_[i]);
    hRMS[i] = dynamic_cast<TH1F*> (RMS[i]->operator->());
  }

  gStyle->SetOptStat(111110);
  cCM->Divide(3,2);
  for (int i=0; i<3; i++) {
    cCM->cd(i+1);
    hMean[i]->Draw();
    cCM->cd(i+4);
    hRMS[i]->Draw();
  }
  histName = htmlDir+"/PedestalCM_Mean_RMS.png";
  cCM->SaveAs(histName.c_str());

  // Show plots
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_Box1_TS1.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_Box2_TS1.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_Box1_TS2.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_Box2_TS2.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_Box1_TS3.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_Box2_TS3.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  htmlFile << "<tr>" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Mean_RMS.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}
