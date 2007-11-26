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
    meRMS_ = dbe_->book1D(hist, hist, 20, -0.5, 19.5);
    sprintf(hist, "ES QT PedestalCT Fit Mean");
    meFitMean_ = dbe_->book1D(hist, hist, 5000, -0.5, 4999.5);
    sprintf(hist, "ES QT PedestalCT Fit RMS");
    meFitRMS_ = dbe_->book1D(hist, hist, 20, -0.5, 19.5);

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
	      
	      //if (hPedestal->GetRMS()>10) val = 7;	      
	      if (fg->GetParameter(2)>10 && hPedestal->GetMean()!=0) val = 7;	      
	      else if (hPedestal->GetMean()==0) val = 5;
	      else val = 4;
	      mePedCol_[i][j]->setBinContent(abs(n-32-k*32), m+1, val) ;       

	      if (hPedestal->GetMean()!=0) {
		mePedMeanRMS_[i][j][k][m]->setBinContent(n+1, hPedestal->GetMean());	   
		mePedMeanRMS_[i][j][k][m]->setBinError(n+1, hPedestal->GetRMS());	   
		mePedRMS_[i][j][k][m]->setBinContent(n+1, hPedestal->GetRMS());
		mePedFitMeanRMS_[i][j][k][m]->setBinContent(n+1, fg->GetParameter(1));	   
		mePedFitMeanRMS_[i][j][k][m]->setBinError(n+1, fg->GetParameter(2));	   
		mePedFitRMS_[i][j][k][m]->setBinContent(n+1, fg->GetParameter(2));
	      }
	      
	    } else {
	      mePedCol_[i][j]->setBinContent(abs(n-32-k*32), m+1, 5) ;
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
  gStyle->SetGridStyle(1);

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
      hPedQ[i][j]->GetXaxis()->SetLabelSize(0.08);
      hPedQ[i][j]->GetYaxis()->SetLabelSize(0.08);
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
  hMean->GetXaxis()->SetNdivisions(10);
  hMean->SetLineColor(4);
  hMean->Draw();
  cPed->cd(2);
  hRMS->SetLineColor(4);
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
  hFitMean->GetXaxis()->SetNdivisions(10);
  //hFitMean->GetXaxis()->SetLimits(0, 1800);
  hFitMean->SetLineColor(4);
  hFitMean->Draw();
  cPedF->cd(2);
  hFitRMS->GetXaxis()->SetLimits(0, 20);
  hFitRMS->SetLineColor(4);
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


  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  // Create the main html file index.html in the appropriate directory.
  // This piece of code was ported from the standalone version
  // of DQM_html_generator.c by Yannis.Papadopoulos@cern.ch

  int trig=1, i;
  char trigger[8]="", fname[1024], cmd[1024];
  FILE* htmlfp;

  if (trig==1)
    sprintf(trigger,"%s","CT");
  else if (trig==2)
    sprintf(trigger,"%s","TB");
  else
    return; // this should never happen...

  sprintf(fname,"%s/index.html",htmlDir.c_str());

  sprintf(cmd,"rm -f %s",fname); // overcome file ownership problems
  system(cmd);

  htmlfp=fopen(fname,"w");

  fprintf(htmlfp,"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n");
  fprintf(htmlfp,"\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">");
  fprintf(htmlfp,"<html xmlns=\"http://www.w3.org/1999/xhtml\">");
  fprintf(htmlfp,"<head>");
  fprintf(htmlfp,"  <title>ES DQM %s: run %08d</title>",trigger,run);
  fprintf(htmlfp,"  <meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />");
  fprintf(htmlfp,"  <meta name=\"author\" content=\"Ioannis PAPADOPOULOS\"/>");
  fprintf(htmlfp,"  <style type=\"text/css\">");
  fprintf(htmlfp,"     body  {background-color: #caf; font-family: sans-serif; font-size: 12px; position: relative;}");
  fprintf(htmlfp,"     #mother {position: relative;}");
  fprintf(htmlfp,"     td:first-child    {text-align: right;}");
  fprintf(htmlfp,"     #plot {position: absolute; left: 200px; top: 0;}");
  fprintf(htmlfp,"     #dfrm {position: absolute; left: 200px; top: 0; visibility: hidden; background-color: white;}");
  fprintf(htmlfp,"     .ciel {background-color: #acf;}");
  fprintf(htmlfp,"     .vert {background-color: #cfa;}");
  fprintf(htmlfp,"  </style>");
  fprintf(htmlfp,"  <script type=\"text/javascript\">");
  fprintf(htmlfp,"  function hideplot()");
  fprintf(htmlfp,"  {");
  fprintf(htmlfp,"    document.getElementById('plot_img').style.visibility = 'hidden' ;");
  fprintf(htmlfp,"    document.getElementById('dfrm').style.visibility     = 'visible' ;");
  fprintf(htmlfp,"  }");
  fprintf(htmlfp,"  function submitform()");
  fprintf(htmlfp,"  {");
  fprintf(htmlfp,"    document.getElementById('plot_img').style.visibility = 'visible' ;");
  fprintf(htmlfp,"    document.getElementById('dfrm').style.visibility     = 'hidden' ;");
  fprintf(htmlfp,"    for (i=0;i<document.getElementById('myform').rad.length;i++) {");
  fprintf(htmlfp,"      if (document.getElementById('myform').rad[i].checked) {");
  fprintf(htmlfp,"        t = document.getElementById('myform').rad[i].value;");
  fprintf(htmlfp,"      }");
  fprintf(htmlfp,"    }");
  fprintf(htmlfp,"    x=document.getElementById('myform').elements['var_ix'].value;");
  fprintf(htmlfp,"    y=document.getElementById('myform').elements['var_iy'].value;");
  if (trig==1)
    fprintf(htmlfp,"    z=document.getElementById('myform').elements['var_z'].value;");
  else
    fprintf(htmlfp,"    z=1;"); // for TB z is not used and is set to 1
  fprintf(htmlfp,"    s=document.getElementById('myform').elements['var_strip'].value;");
  fprintf(htmlfp,"    p=document.getElementById('myform').elements['var_plane'].value;");
  fprintf(htmlfp,"    document.getElementById('plot_img').src=\"/cgi-bin/DQM/DQMimage%s.sh?",trigger);
  fprintf(htmlfp,"t=\"+t+\"&s=\"+s+\"&x=\"+x+\"&y=\"+y+\"&z=\"+z+\"&p=\"+p+\"&r=\"+%d;",run);
  fprintf(htmlfp,"  }");
  fprintf(htmlfp,"  </script>");
  fprintf(htmlfp,"</head>");

  fprintf(htmlfp,"<body>");
  fprintf(htmlfp,"<div id=\"mother\">");

  fprintf(htmlfp,"  <form name=\"myform\" id=\"myform\" action=\"javascript: submitform()\" >");
  fprintf(htmlfp,"    <table>");
  fprintf(htmlfp,"    <tr><td colspan=\"2\" style=\"text-align:center;\">ES DQM");
  fprintf(htmlfp,"      <span style=\"color:blue; font-size:1.5em;\">%s</span><br/>",trigger);
  fprintf(htmlfp,"      run <span style=\"color:red; font-size:1.5em;\">%08d</span><hr/></td></tr>",run);

  fprintf(htmlfp,"    <tr><td>Strip # :</td>");
  fprintf(htmlfp,"    <td><select name=\"var_strip\">");
  for (i=1; i<=32; i++) fprintf(htmlfp,"<option value=\"%d\">%02d</option>",i,i);
  fprintf(htmlfp,"    </select></td></tr>");

  fprintf(htmlfp,"    <tr><td>Plane # :</td>");
  fprintf(htmlfp,"    <td><select name=\"var_plane\">");
  if (trig==1)
    for (i=1; i<=6; i++) fprintf(htmlfp,"<option value=\"%d\">%02d</option>",i,i);
  else
    for (i=1; i<=2; i++) fprintf(htmlfp,"<option value=\"%d\">%02d</option>",i,i);
  fprintf(htmlfp,"    </select></td></tr>");

  fprintf(htmlfp,"    <tr><td>ix :</td>");
  fprintf(htmlfp,"    <td><select name=\"var_ix\">");
  if (trig==1) // CT
    for (i=1; i<=2; i++) fprintf(htmlfp,"<option value=\"%d\">%02d</option>",i,i);
  else         // TB
    for (i=1; i<=4; i++) fprintf(htmlfp,"<option value=\"%d\">%02d</option>",i,i);
  fprintf(htmlfp,"    </select></td></tr>");

  fprintf(htmlfp,"    <tr><td>iy :</td>");
  fprintf(htmlfp,"    <td><select name=\"var_iy\">");
  if (trig==1) // CT
    for (i=1; i<=5; i++) fprintf(htmlfp,"<option value=\"%d\">%02d</option>",i,i);
  else         // TB
    for (i=1; i<=4; i++) fprintf(htmlfp,"<option value=\"%d\">%02d</option>",i,i);
  fprintf(htmlfp,"    </select></td></tr>");

  if (trig==1) { // z is used in CT but not in TB. (In TB it is set to 1 in the javascript code)
    fprintf(htmlfp,"    <tr><td>z :</td>");
    fprintf(htmlfp,"    <td><select name=\"var_z\">");
    for (i=1; i>=-1; i--) if (i) fprintf(htmlfp,"<option value=\"%d\">%2d</option>",i,i);
    fprintf(htmlfp,"    </select></td></tr>");
  }

  fprintf(htmlfp,"    <tr style=\"font-size:4px;\"><td>&nbsp;</td><td>&nbsp;</td></tr>");

  fprintf(htmlfp,"    <tr class=\"ciel\"><td>sensor pedestals:  </td>");
  fprintf(htmlfp,"        <td> <input type=\"radio\" name=\"rad\" value=\"1\" checked=\"checked\"/></td></tr>");
  fprintf(htmlfp,"    <tr class=\"ciel\"><td>sensor noise:      </td>");
  fprintf(htmlfp,"        <td> <input type=\"radio\" name=\"rad\" value=\"2\"/></td></tr>");
  fprintf(htmlfp,"    <tr class=\"ciel\"><td>strip (raw):       </td>");
  fprintf(htmlfp,"        <td> <input type=\"radio\" name=\"rad\" value=\"3\"/><br/></td></tr>");
  fprintf(htmlfp,"    <tr class=\"vert\"><td>sensor CM:         </td>");
  fprintf(htmlfp,"        <td> <input type=\"radio\" name=\"rad\" value=\"4\"/><br/></td></tr>");
  fprintf(htmlfp,"    <tr class=\"vert\"><td>strip (raw-ped-CM):</td>");
  fprintf(htmlfp,"        <td> <input type=\"radio\" name=\"rad\" value=\"5\"/><br/></td></tr>");

  fprintf(htmlfp,"    <tr style=\"font-size:4px;\"><td>&nbsp;</td><td>&nbsp;</td></tr>");

  fprintf(htmlfp,"    <tr><td colspan=\"2\"><input type=\"submit\" value=\"Show the plot!\"></input></td></tr>");
  fprintf(htmlfp,"    <tr><td colspan=\"2\" style=\"text-align: center;\"><hr/>");
  fprintf(htmlfp,"      <a href=\"/DQM/%s/%08d/ESPedestal%s.html\"",trigger,run,trigger);
  fprintf(htmlfp,"         target=frm onclick=\"hideplot();\">");
  fprintf(htmlfp,"        <img src=\"/ESPedestal%s_small.png\"",trigger);
  fprintf(htmlfp,"             title=\"Click here to see the Pedestal summary plots\" border=0>");
  fprintf(htmlfp,"      </a>");
  fprintf(htmlfp,"    </td><tr>");
  fprintf(htmlfp,"    <tr><td colspan=\"2\" style=\"text-align: center;\">");
  fprintf(htmlfp,"      <a href=\"/DQM/%s/%08d/ESPedestalCM%s.html\"",trigger,run,trigger);
  fprintf(htmlfp,"         target=frm onclick=\"hideplot();\">");
  fprintf(htmlfp,"        <img src=\"/ESPedestalCM%s_small.png\"",trigger);
  fprintf(htmlfp,"             title=\"Click here to see the Common Mode noise summary plots\" border=0>");
  fprintf(htmlfp,"      </a>");
  fprintf(htmlfp,"    </td><tr>");
  fprintf(htmlfp,"    <tr><td colspan=\"2\" style=\"text-align: center;\">");
  fprintf(htmlfp,"      <a href=\"/DQM/%s/%08d/ESDataIntegrity.html\"",trigger,run);
  fprintf(htmlfp,"         target=frm onclick=\"hideplot();\">");
  fprintf(htmlfp,"        <img src=\"/ESDataIntegrity_small.png\"");
  fprintf(htmlfp,"             title=\"Click here to see the Data Integrity summary plots\" border=0>");
  fprintf(htmlfp,"      </a>");
  fprintf(htmlfp,"    </td><tr>");
  fprintf(htmlfp,"    <tr><td colspan=\"2\" style=\"text-align: center;\">");
  fprintf(htmlfp,"      <a href=\"/DQM/%s/%08d/ESTDC%s.html\"",trigger,run,trigger);
  fprintf(htmlfp,"         target=frm onclick=\"hideplot();\">");
  fprintf(htmlfp,"        <img src=\"/ESTDC_small.png\"");
  fprintf(htmlfp,"             title=\"Click here to see the TDC summary plots\" border=0>");
  fprintf(htmlfp,"      </a>");
  fprintf(htmlfp,"    </td><tr>");
  fprintf(htmlfp,"    </table>");
  fprintf(htmlfp,"  </form>");

  fprintf(htmlfp,"  <div id=\"plot\">");
  fprintf(htmlfp,"  <img id=\"plot_img\" src=\"/The-CMS-Experiment.jpg\"");
  fprintf(htmlfp,"       alt=\"\" width=\"692\">");
  fprintf(htmlfp,"  </div>");

  fprintf(htmlfp,"  <div id=\"dfrm\">");
  fprintf(htmlfp,"  <iframe name=\"frm\" width=\"1400\" height=1200></iframe>");
  fprintf(htmlfp,"  </div>");

  fprintf(htmlfp,"</div>");
  fprintf(htmlfp,"</body>");
  fprintf(htmlfp,"</html>");

  fclose(htmlfp);
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------

}
