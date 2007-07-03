#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalCTClient.h"

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

ESPedestalCTClient::ESPedestalCTClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESPedestalCT");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/CT");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESPedestalCT.html");  
  sta_        = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;

  fg = new TF1("fg", "gaus");
}

ESPedestalCTClient::~ESPedestalCTClient(){

  delete fg;

}

void ESPedestalCTClient::beginJob(const EventSetup& context){

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/PedestalCT");
    dbe_->rmdir("ES/QT/PedestalCT");
  }

} 

void ESPedestalCTClient::endJob(){

  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }
  
  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/PedestalCT");  

  if ( init_ ) this->cleanup();
}

void ESPedestalCTClient::setup() {

  init_ = true;

  Char_t hist[200];
  
  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCT");
    sprintf(hist, "ES QT PedestalCT Mean");
    meMean_ = dbe_->book1D(hist, hist, 5000, -0.5, 4999.5);
    sprintf(hist, "ES QT PedestalCT RMS");
    meRMS_ = dbe_->book1D(hist, hist, 100, -0.5, 99.5);
    sprintf(hist, "ES QT PedestalCT Fit Mean");
    meFitMean_ = dbe_->book1D(hist, hist, 5000, -0.5, 4999.5);
    sprintf(hist, "ES QT PedestalCT Fit RMS");
    meFitRMS_ = dbe_->book1D(hist, hist, 100, -0.5, 99.5);

    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {

	sprintf(hist, "ES PedestalCT Quality Box %d Plane %d", i+1, j+1);
	mePedCol_[i][j] = dbe_->book2D(hist, hist, 64, 0, 64, 5, 0, 5);

	for (int k=0; k<2; ++k) {
	  for (int m=0; m<5; ++m) {
	    
	    int zside = (i==0)?1:-1;
	    
	    sprintf(hist, "ES Pedestal Mean RMS Z %d P %d Row %02d Col %02d", zside, j+1, k+1, m+1);
	    mePedMeanRMS_[i][j][k][m] = dbe_->book1D(hist, hist, 32, 0, 32);
	    sprintf(hist, "ES Pedestal RMS Z %d P %d Row %02d Col %02d", zside, j+1, k+1, m+1);
	    mePedRMS_[i][j][k][m] = dbe_->book1D(hist, hist, 32, 0, 32);
	    
	    sprintf(hist, "ES Pedestal Fit Mean RMS Z %d P %d Row %02d Col %02d", zside, j+1, k+1, m+1);
	    mePedFitMeanRMS_[i][j][k][m] = dbe_->book1D(hist, hist, 32, 0, 32);
	    sprintf(hist, "ES Pedestal Fit RMS Z %d P %d Row %02d Col %02d", zside, j+1, k+1, m+1);
	    mePedFitRMS_[i][j][k][m] = dbe_->book1D(hist, hist, 32, 0, 32);
	  }
	}
      }
    }
  }

}

void ESPedestalCTClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCT");
    if (meMean_) dbe_->removeElement( meMean_->getName() );
    if (meRMS_) dbe_->removeElement( meRMS_->getName() );
    if (meFitMean_) dbe_->removeElement( meFitMean_->getName() );
    if (meFitRMS_) dbe_->removeElement( meFitRMS_->getName() );
    meMean_ = 0;
    meRMS_ = 0;
    meFitMean_ = 0;
    meFitRMS_ = 0;
    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {

	if (mePedCol_[i][j]) dbe_->removeElement( mePedCol_[i][j]->getName() );
	mePedCol_[i][j] = 0;

	for (int k=0; k<2; ++k) {
	  for (int m=0; m<5; ++m) {
	    if (mePedMeanRMS_[i][j][k][m]) dbe_->removeElement( mePedMeanRMS_[i][j][k][m]->getName() );
	    if (mePedRMS_[i][j][k][m]) dbe_->removeElement( mePedRMS_[i][j][k][m]->getName() );
	    if (mePedFitMeanRMS_[i][j][k][m]) dbe_->removeElement( mePedFitMeanRMS_[i][j][k][m]->getName() );
	    if (mePedFitRMS_[i][j][k][m]) dbe_->removeElement( mePedFitRMS_[i][j][k][m]->getName() );
	    mePedMeanRMS_[i][j][k][m] = 0;
	    mePedRMS_[i][j][k][m] = 0;
	    mePedFitMeanRMS_[i][j][k][m] = 0;
	    mePedFitRMS_[i][j][k][m] = 0;
	  }
	}
      }
    }
  }
  
  init_ = false;
}

void ESPedestalCTClient::analyze(const Event& e, const EventSetup& context){
  
  if ( ! init_ ) this->setup();

  int runNum = e.id().run();

  if (runNum != run_) { 

    if (run_ > 0) {
      Char_t runNum_s[50];
      sprintf(runNum_s, "%08d", runNum);
      outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
      
      if (writeHTML_) htmlOutput(runNum, htmlDir_, htmlName_);
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

void ESPedestalCTClient::doQT() {

  ESDQMUtils::resetME( meMean_ );
  ESDQMUtils::resetME( meRMS_ );
  ESDQMUtils::resetME( meFitMean_ );
  ESDQMUtils::resetME( meFitRMS_ );

  int val = 0;
  for (int i=0; i<2; ++i) {    
    for (int j=0; j<6; ++j) {
      for (int k=0; k<2; ++k) {
	for (int m=0; m<5; ++m) {
	  for (int n=0; n<32; ++n) {
	    
	    int zside = (i==0)?1:-1;
	    MonitorElement * occME = dbe_->get(getMEName(zside, j+1, k+1, m+1, n+1));
	    
	    if (occME) {
	      MonitorElementT<TNamed>* occ = dynamic_cast<MonitorElementT<TNamed>*>(occME);           
	      TH1F *hPedestal = dynamic_cast<TH1F*> (occ->operator->());      
	      
	      if (hPedestal->GetMean()!=0) {
		hPedestal->Fit("fg","Q");
		hPedestal->Fit("fg","RQ","",fg->GetParameter(1)-2.*fg->GetParameter(2),fg->GetParameter(1)+2.*fg->GetParameter(2));
		
		meMean_->Fill(hPedestal->GetMean());
		meRMS_->Fill(hPedestal->GetRMS());
		meFitMean_->Fill(fg->GetParameter(1));
		meFitRMS_->Fill(fg->GetParameter(2));
	      }
	      
	      if (hPedestal->GetRMS()>10) val = 7;
	      else if (hPedestal->GetMean()==0) val = 5;
	      else val = 4;
	      mePedCol_[i][j]->setBinContent(k*32+n+1, m+1, val) ;       
	      
	      if (hPedestal->GetMean()!=0) {
		mePedMeanRMS_[i][j][k][m]->setBinContent(n+1, hPedestal->GetMean());	   
		mePedMeanRMS_[i][j][k][m]->setBinError(n+1, hPedestal->GetRMS());	   
		mePedRMS_[i][j][k][m]->setBinContent(n+1, hPedestal->GetRMS());
		  mePedFitMeanRMS_[i][j][k][m]->setBinContent(n+1, fg->GetParameter(1));	   
		  mePedFitMeanRMS_[i][j][k][m]->setBinError(n+1, fg->GetParameter(2));	   
		  mePedFitRMS_[i][j][k][m]->setBinContent(n+1, fg->GetParameter(2));
	      }
	      
	    } else {
	      mePedCol_[i][j]->setBinContent(k*32+n+1, m+1, 5) ;
	    }
	    
	  }	
	}
      }
    }
  }

}

string ESPedestalCTClient::getMEName(const int & zside, const int & plane, const int & row, const int & col, const int & strip) {
  
  Char_t hist[500];
  sprintf(hist,"%sES/ESPedestalCTTask/ES Pedestal Z %d P %d Row %02d Col %02d Str %02d",rootFolder_.c_str(),zside,plane,row,col,strip);

  return hist;
}

void ESPedestalCTClient::htmlOutput(int run, string htmlDir, string htmlName) {

  cout<<"Going to output ESPedestalCTClient html ..."<<endl;
  
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
  htmlFile << "  <title>Preshower DQM : PedestalCTTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Cosmic Ray Test PEDESTAL</span></h2> " << endl;
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

  TCanvas *cPedQ = new TCanvas("cPedQ", "cPedQ", 1200, 250);
  TCanvas *cPed  = new TCanvas("cPed",  "cPed",  600, 300);
  TCanvas *cPedF = new TCanvas("cPedF", "cPedF", 600, 300);
  
  MonitorElementT<TNamed>* PedQ[2][6];
  TH2F* hPedQ[2][6];
  for (int i=0; i<2; ++i) {
    for (int j=0; j<6; ++j) {
      PedQ[i][j] = dynamic_cast<MonitorElementT<TNamed>*>(mePedCol_[i][j]);           
      hPedQ[i][j] = dynamic_cast<TH2F*> (PedQ[i][j]->operator->());  
    }
  }
  
  gStyle->SetOptStat("");
  cPedQ->Divide(6,1);
  for (int i=0; i<2; ++i) {
    for (int j=0; j<6; ++j) {
      cPedQ->cd(j+1);
      gPad->SetGridx();
      gPad->SetGridy();
      hPedQ[i][j]->GetXaxis()->SetNdivisions(-102);
      hPedQ[i][j]->GetYaxis()->SetNdivisions(-105);
      hPedQ[i][j]->SetMinimum(-0.00000001);
      hPedQ[i][j]->SetMaximum(7.0);
      char tit[128]; sprintf(tit,"Box %d   Plane %d",i+1,j+1);
      hPedQ[i][j]->SetTitle(tit);
      hPedQ[i][j]->Draw("col");
      gPad->Update();
      TPaveText *t = (TPaveText*) gPad->GetPrimitive("title");
      t->SetTextColor(4);
      t->SetTextSize(.1);
      t->SetBorderSize(0);
      t->SetX1NDC(0.00); t->SetX2NDC(1);
      t->SetY1NDC(0.93); t->SetY2NDC(1);
    }
    histName = (i==0) ? htmlDir+"/Pedestal_Quality_Box1.png":htmlDir+"/Pedestal_Quality_Box2.png";
    cPedQ->SaveAs(histName.c_str());  
  }

  // Plot Mean and RMS
  MonitorElementT<TNamed>* Mean = dynamic_cast<MonitorElementT<TNamed>*>(meMean_);
  TH1F *hMean = dynamic_cast<TH1F*> (Mean->operator->());
  MonitorElementT<TNamed>* RMS  = dynamic_cast<MonitorElementT<TNamed>*>(meRMS_);
  TH1F *hRMS = dynamic_cast<TH1F*> (RMS->operator->());

  gStyle->SetOptStat(111110);
  cPed->Divide(2,1);
  cPed->cd(1);
  hMean->GetXaxis()->SetNdivisions(5);
  hMean->Draw();
  cPed->cd(2);
  hRMS->Draw();
  histName = htmlDir+"/Pedestal_Mean_RMS.png";
  cPed->SaveAs(histName.c_str());

  // Plot Fit Mean and RMS
  MonitorElementT<TNamed>* FitMean = dynamic_cast<MonitorElementT<TNamed>*>(meFitMean_);
  TH1F *hFitMean = dynamic_cast<TH1F*> (FitMean->operator->());
  MonitorElementT<TNamed>* FitRMS  = dynamic_cast<MonitorElementT<TNamed>*>(meFitRMS_);
  TH1F *hFitRMS = dynamic_cast<TH1F*> (FitRMS->operator->());

  gStyle->SetOptStat(111110);
  cPedF->Divide(2,1);
  cPedF->cd(1);
  hFitMean->GetXaxis()->SetNdivisions(5);
  hFitMean->Draw();
  cPedF->cd(2);
  hFitRMS->Draw();
  histName = htmlDir+"/Pedestal_Fit_Mean_RMS.png";
  cPedF->SaveAs(histName.c_str());

  // Show plots
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"Pedestal_Quality_Box1.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"Pedestal_Quality_Box2.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"2\"><img src=\"Pedestal_Mean_RMS.png\"></img></td>" << endl;
  htmlFile << "<td colspan=\"2\"><img src=\"Pedestal_Fit_Mean_RMS.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  stringstream run_str; run_str << run;
  system(("/preshower/yannisp1/html/DQM_html_generator "+run_str.str()).c_str());

}
