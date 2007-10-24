#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalCMTBClient.h"

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

ESPedestalCMTBClient::ESPedestalCMTBClient(const ParameterSet& ps) {
  
  writeHisto_     = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_      = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_       = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESPedestalCMTB");
  rootFolder_     = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_        = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/TB");
  htmlName_       = ps.getUntrackedParameter<string>("htmlName","ESPedestalCMTB.html");  
  cmnThreshold_   = ps.getUntrackedParameter<double>("cmnThreshold", 3);
  sta_            = ps.getUntrackedParameter<bool>("RunStandalone", false);
  gain_           = ps.getUntrackedParameter<int>("ESGain", 1);

  count_ = 0;
  run_ = -1;
  init_ = false;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();
}

ESPedestalCMTBClient::~ESPedestalCMTBClient(){
}

void ESPedestalCMTBClient::beginJob(const EventSetup& context){

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/PedestalCMTB");
    dbe_->rmdir("ES/QT/PedestalCMTB");
  }

}

void ESPedestalCMTBClient::endJob(){
  
  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }

  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/PedestalCMTB");  

  if ( init_ ) this->cleanup();
}

void ESPedestalCMTBClient::setup() {

  init_ = true;

  Char_t hist[200];
  
  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCMTB");

    for (int i=0; i<3; ++i) {
      sprintf(hist, "ES QT CMTB Mean TS %d", i+1);
      meMean_[i] = dbe_->book1D(hist, hist, 100, -50, 50);
      sprintf(hist, "ES QT CMTB RMS TS %d", i+1);
      meRMS_[i] = dbe_->book1D(hist, hist, 100, -50, 50);
    }

    for (int i=0; i<2; ++i) {
      for (int j=0; j<3; ++j) {
	sprintf(hist, "ES CMTB Quality Plane %d TS %d", i+1, j+1);
	meCMCol_[i][j] = dbe_->book2D(hist, hist, 4, 0, 4, 4, 0, 4);
      }
    }
  }
  
}

void ESPedestalCMTBClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/PedestalCMTB");

    for (int i=0; i<3; ++i) {
      if (meMean_[i]) dbe_->removeElement( meMean_[i]->getName() );
      if (meRMS_[i]) dbe_->removeElement( meRMS_[i]->getName() );
      meMean_[i] = 0;
      meRMS_[i] = 0;
    }

    for (int i=0; i<2; ++i) {
      for (int j=0; j<3; ++j) {
	if (meCMCol_[i][j]) dbe_->removeElement( meCMCol_[i][j]->getName() );
	meCMCol_[i][j] = 0;
      }
    }
  }
  
  init_ = false;
}

void ESPedestalCMTBClient::analyze(const Event& e, const EventSetup& context){
  
  if ( ! init_ ) this->setup();

  int runNum = e.id().run();
  Char_t runNum_s[50];

  if (runNum != run_) { 

    if (run_ > 0) {

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

    sprintf(runNum_s, "%08d", run_);
    outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";    
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

void ESPedestalCMTBClient::doQT() {

  for (int i=0; i<3; ++i) {
    ESDQMUtils::resetME( meMean_[i] );
    ESDQMUtils::resetME( meRMS_[i] );
  }

  int val = 0;
  for (int i=0; i<2; ++i) {    
    for (int j=0; j<4; ++j) {
      for (int k=0; k<4; ++k) {
	for (int m=0; m<3; ++m) {
	  
	  MonitorElement * senME = dbe_->get(getMEName(i+1, j+1, k+1, 0, m, 1));

	  if (senME) {
	    MonitorElementT<TNamed>* sen = dynamic_cast<MonitorElementT<TNamed>*>(senME);           
	    TH1F *hCMSen = dynamic_cast<TH1F*> (sen->operator->());  	    

	    if (hCMSen->GetRMS()>cmnThreshold_) val = 7;
	    else if (hCMSen->GetEntries() == 0) val = 5;
	    else val = 4;
	    
	    meCMCol_[i][m]->setBinContent(j+1, k+1, val) ;  
	    
	    if (hCMSen->GetEntries() != 0) {
	      meMean_[m]->Fill(hCMSen->GetMean());
	      meRMS_[m]->Fill(hCMSen->GetRMS());
	    }
	    
	  }
	  
	}	
      }
    }
  }

  for (int i=0; i<2; ++i) {
    for (int j=0; j<3; ++j) {

      MonitorElement *adcME = dbe_->get(getMEName(i+1, 0, 0, 0, j+1, 2));
      MonitorElement *adcZSME = dbe_->get(getMEName(i+1, 0, 0, 0, j+1, 3));
      MonitorElement *occME = dbe_->get(getMEName(i+1, 0, 0, 0, j+1, 4));

      if (adcME) {
	MonitorElementT<TNamed>* adc = dynamic_cast<MonitorElementT<TNamed>*>(adcME);
	hADC_[i][j] = dynamic_cast<TH1F*> (adc->operator->());
      }

      if (adcZSME) {
	MonitorElementT<TNamed>* adczs = dynamic_cast<MonitorElementT<TNamed>*>(adcZSME);
	hADCZS_[i][j] = dynamic_cast<TH1F*> (adczs->operator->());
      }

      if (occME) {
	MonitorElementT<TNamed>* occ = dynamic_cast<MonitorElementT<TNamed>*>(occME);
	hOCC_[i][j] = dynamic_cast<TH2F*> (occ->operator->());
      }
    }
  }

}

string ESPedestalCMTBClient::getMEName(const int & plane, const int & col, const int & row, const int & strip, const int & slot, const int & type) {
  
  Char_t hist[500];
  if (type == 0)
    sprintf(hist,"%sES/ESPedestalCMTBTask/ES Pedestal CM_S%d Z 1 P %d Col %02d Row %02d Str %02d", rootFolder_.c_str(),slot,plane,col,row,strip);
  else if (type == 1)
    sprintf(hist,"%sES/ESPedestalCMTBTask/ES Sensor CM_S%d Z 1 P %d Col %02d Row %02d", rootFolder_.c_str(),slot,plane,col,row);
  else if (type==2) 
    sprintf(hist,"%sES/ESPedestalCMTBTask/ES ADC P %d TS %d", rootFolder_.c_str(),plane,slot);
  else if (type==3)
    sprintf(hist,"%sES/ESPedestalCMTBTask/ES ADC ZS P %d TS %d", rootFolder_.c_str(),plane,slot);
  else if (type==4) 
    sprintf(hist,"%sES/ESPedestalCMTBTask/ES Occupancy P %d TS %d", rootFolder_.c_str(),plane,slot);

  return hist;    
}

void ESPedestalCMTBClient::htmlOutput(int run, string htmlDir, string htmlName) {

  cout<<"Going to output ESPedestalCMTBClient html ..."<<endl;
  
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
  htmlFile << "  <title>Preshower DQM : PedestalCMTBTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  if (gain_==1)
    htmlFile << " style=\"color: rgb(0, 0, 153);\">Test Beam Common Mode (Low Gain) </span></h2> " << endl;
  if (gain_==2)
    htmlFile << " style=\"color: rgb(0, 0, 153);\">Test Beam Common Mode (High Gain) </span></h2> " << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr><td bgcolor=red>This strip has problems</td>" << endl;
  htmlFile << "<td bgcolor=lime>This strip has NO problems</td>" << endl;
  htmlFile << "<td bgcolor=yellow>This strip is missing</td></tr>" << endl;
  htmlFile << "<tr><td> \> "<<cmnThreshold_<<" ADC </td>"<<endl;
  htmlFile << "<td> \< "<<cmnThreshold_<<" ADC </td>"<<endl;
  htmlFile << "<td> </td></tr>"<<endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // make plots
  string histName;
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1, 0);
  gStyle->SetStatW(0.3);
  gStyle->SetStatH(0.3);
  gStyle->SetGridStyle(1);

  TCanvas *cCMQ = new TCanvas("cCMQ", "cCMQ", 600, 300);
  TCanvas *cCM  = new TCanvas("cCM",  "cCM",  900, 600);
  
  MonitorElementT<TNamed>* CMQ[2][3];
  TH2F* hCMQ[2][3];
  for (int i=0; i<2; ++i) {
    for (int j=0; j<3; ++j) {
      CMQ[i][j] = dynamic_cast<MonitorElementT<TNamed>*>(meCMCol_[i][j]);           
      hCMQ[i][j] = dynamic_cast<TH2F*> (CMQ[i][j]->operator->());  
    }
  }    

  gStyle->SetOptStat("");
  cCMQ->Divide(2,1);
  for (int j=0; j<3; ++j) {
    for (int i=0; i<2; ++i) {
      cCMQ->cd(i+1);
      gPad->SetGridx();
      gPad->SetGridy();
      hCMQ[i][j]->GetXaxis()->SetNdivisions(-104);
      hCMQ[i][j]->GetYaxis()->SetNdivisions(-104);
      hCMQ[i][j]->SetMinimum(-0.00000001);
      hCMQ[i][j]->SetMaximum(7.0);
      char tit[128]; sprintf(tit,"Plane %d TS %d",i+1,j+1);
      hCMQ[i][j]->SetTitle(tit);
      hCMQ[i][j]->Draw("col");
      gPad->Update();
      TPaveText *t = (TPaveText*) gPad->GetPrimitive("title");
      t->SetTextColor(4);
      t->SetTextSize(.1);
      t->SetBorderSize(0);
      t->SetX1NDC(0.00); t->SetX2NDC(1);
      t->SetY1NDC(0.93); t->SetY2NDC(1);
    }
    stringstream ts; ts << (j+1);
    histName = htmlDir+"/PedestalCM_Quality_TS"+ts.str()+".png";
    cCMQ->SaveAs(histName.c_str());  
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
  for (int i=0; i<3; ++i) {
    cCM->cd(i+1);
    hMean[i]->Draw();
    cCM->cd(i+4);
    hRMS[i]->Draw();
  }
  histName = htmlDir+"/PedestalCM_Mean_RMS.png";
  cCM->SaveAs(histName.c_str());

  gStyle->SetOptStat("");
  TCanvas *cADCZS = new TCanvas("cADCZS", "cADCZS", 1200, 800);
  cADCZS->Divide(3,2);
  for (int i=0; i<2; ++i) {
    for (int j=0; j<3; ++j) {
      cADCZS->cd(1+(i*3)+j);
      gPad->SetLogy(1);
      hADCZS_[i][j]->Draw();
      gPad->Update();
    }
  }

  histName = htmlDir+"/ESADCZS.png";
  cADCZS->SaveAs(histName.c_str());

  htmlFile << "<img src=\"ESADCZS.png\"></img>" <<endl;

  TPaveText *t1;
  TPaveText *t2;
  TPaveText *t3;
  TPaveText *t4;

  TCanvas *cOCC = new TCanvas("cOCC", "cOCC", 1200, 800);
  cOCC->Divide(3,2);
  for (int i=0; i<2; ++i) {
    for (int j=0; j<3; ++j) {
      cOCC->cd(1+(i*3)+j);
      gPad->SetGridx();
      gPad->SetGridy();
      hOCC_[i][j]->GetXaxis()->SetNdivisions(-104);
      hOCC_[i][j]->GetYaxis()->SetNdivisions(-104);
      hOCC_[i][j]->GetXaxis()->SetLabelSize(0);
      hOCC_[i][j]->GetYaxis()->SetLabelSize(0);
      hOCC_[i][j]->Draw("colz");
      gPad->Update();
      t1 = new TPaveText(0.18, .04, 0.23, .09, "NDC");
      t1->SetBorderSize(0);
      t1->SetFillColor(0);
      t1->SetTextSize(0.06);
      t1->AddText("1");
      t1->Draw();
      t2 = new TPaveText(0.38, .04, 0.43, .09, "NDC");
      t2->SetBorderSize(0);
      t2->SetFillColor(0);
      t2->SetTextSize(0.06);
      t2->AddText("2");
      t2->Draw();
      t3 = new TPaveText(0.58, .04, 0.63, .09, "NDC");
      t3->SetBorderSize(0);
      t3->SetFillColor(0);
      t3->SetTextSize(0.06);
      t3->AddText("3");
      t3->Draw();
      t4 = new TPaveText(0.78, .04, 0.83, .09, "NDC");
      t4->SetBorderSize(0);
      t4->SetFillColor(0);
      t4->SetTextSize(0.06);
      t4->AddText("4");
      t4->Draw();
      t1 = new TPaveText(0.05, .785, 0.1, .835, "NDC");
      t1->SetBorderSize(0);
      t1->SetFillColor(0);
      t1->SetTextSize(0.06);
      t1->AddText("4");
      t1->Draw();
      t2 = new TPaveText(0.05, .585, 0.1, .635, "NDC");
      t2->SetBorderSize(0);
      t2->SetFillColor(0);
      t2->SetTextSize(0.06);
      t2->AddText("3");
      t2->Draw();
      t3 = new TPaveText(0.05, .385, 0.1, .435, "NDC");
      t3->SetBorderSize(0);
      t3->SetFillColor(0);
      t3->SetTextSize(0.06);
      t3->AddText("2");
      t3->Draw();
      t4 = new TPaveText(0.05, .185, 0.1, .235, "NDC");
      t4->SetBorderSize(0);
      t4->SetFillColor(0);
      t4->SetTextSize(0.06);
      t4->AddText("1");
      t4->Draw();
    }
  }

  histName = htmlDir+"/ESOCC.png";
  cOCC->SaveAs(histName.c_str());

  htmlFile << "<img src=\"ESOCC.png\"></img>" <<endl;

  delete t1;
  delete t2;
  delete t3;
  delete t4;

  TCanvas *cADC = new TCanvas("cADC", "cADC", 1200, 800);
  cADC->Divide(3,2);
  for (int i=0; i<2; ++i) {
    for (int j=0; j<3; ++j) {
      cADC->cd(1+(i*3)+j);
      gPad->SetLogy(1);
      hADC_[i][j]->Draw();
      gPad->Update();
    }
  }

  histName = htmlDir+"/ESADC_ped_cmn.png";
  cADC->SaveAs(histName.c_str());

  htmlFile << "<img src=\"ESADC_ped_cmn.png\"></img>" <<endl;

  // Show plots
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" align=\"\"> " << endl;
  htmlFile << "<tr align=\"\">" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_TS1.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"\">" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_TS2.png\"></img></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"\">" << endl;
  htmlFile << "<td colspan=\"1\"><img src=\"PedestalCM_Quality_TS3.png\"></img></td>" << endl;
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
