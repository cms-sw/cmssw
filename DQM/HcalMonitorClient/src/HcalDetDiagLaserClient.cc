#include "DQM/HcalMonitorClient/interface/HcalDetDiagLaserClient.h"
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>
#include <TPaveStats.h>

HcalDetDiagLaserClient::HcalDetDiagLaserClient(){}
HcalDetDiagLaserClient::~HcalDetDiagLaserClient(){}
void HcalDetDiagLaserClient::beginJob(){}
void HcalDetDiagLaserClient::beginRun(){}
void HcalDetDiagLaserClient::endJob(){} 
void HcalDetDiagLaserClient::endRun(){} 
void HcalDetDiagLaserClient::cleanup(){} 
void HcalDetDiagLaserClient::analyze(){} 
void HcalDetDiagLaserClient::report(){} 
void HcalDetDiagLaserClient::resetAllME(){} 
void HcalDetDiagLaserClient::createTests(){}
void HcalDetDiagLaserClient::loadHistograms(TFile* infile){}

void HcalDetDiagLaserClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  HcalBaseClient::init(ps,dbe,clientName);
  status=0;
} 

void HcalDetDiagLaserClient::getHistograms(){
  std::string folder="HcalDetDiagLaserMonitor/Summary Plots/";
  Energy        =getHisto(folder+"Laser Energy Distribution",              process_, dbe_, debug_,cloneME_);
  Timing        =getHisto(folder+"Laser Timing Distribution",              process_, dbe_, debug_,cloneME_);
  EnergyRMS     =getHisto(folder+"Laser Energy RMS/Energy Distribution",   process_, dbe_, debug_,cloneME_);
  TimingRMS     =getHisto(folder+"Laser Timing RMS Distribution",          process_, dbe_, debug_,cloneME_);
  Time2Dhbhehf  =getHisto2(folder+"Laser Timing HBHEHF",                   process_, dbe_, debug_,cloneME_);
  Time2Dho      =getHisto2(folder+"Laser Timing HO",                       process_, dbe_, debug_,cloneME_);
  Energy2Dhbhehf=getHisto2(folder+"Laser Energy HBHEHF",                   process_, dbe_, debug_,cloneME_);
  Energy2Dho    =getHisto2(folder+"Laser Energy HO",                       process_, dbe_, debug_,cloneME_);
   
  MonitorElement* me = dbe_->get("Hcal/HcalDetDiagLaserMonitor/HcalDetDiagLaserMonitor Event Number");
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
} 
bool HcalDetDiagLaserClient::haveOutput(){
    getHistograms();
    if(ievt_>=10) return true;
    return false; 
}
int  HcalDetDiagLaserClient::SummaryStatus(){
    status=0;
    return status;
}

void HcalDetDiagLaserClient::htmlOutput(int runNo, string htmlDir, string htmlName){
  if (debug_>0) cout << "<HcalDetDiagLaserClient::htmlOutput> Preparing  html output ..." << endl;
  if(!dbe_) return;
  string client = "HcalDetDiagLaserClient";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  gROOT->SetBatch(true);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadColor(0);
  gStyle->SetOptStat(111110);
  gStyle->SetPalette(1);
  //TPaveStats *ptstats;
  TCanvas *can=new TCanvas("HcalDetDiagLaserClient","HcalDetDiagLaserClient",0,0,500,350);
  can->cd();
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());
  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Detector Diagnostics Laser Monitor</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Detector Diagnostics Laser Monitor</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2 align=\"center\">Summary Laser plots</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  Time2Dhbhehf->SetStats(0);
  Time2Dho->SetStats(0);
  Time2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "laser_timing_hbhehf.gif").c_str());
  Time2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "laser_timing_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"laser_timing_hbhehf.gif\" alt=\"laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"laser_timing_ho.gif\" alt=\"laser timing distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  Energy2Dhbhehf->SetStats(0);
  Energy2Dho->SetStats(0);
  Energy2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "laser_energy_hbhehf.gif").c_str());
  Energy2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "laser_energy_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"laser_energy_hbhehf.gif\" alt=\"laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"laser_energy_ho.gif\" alt=\"laser energy distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;  
  Energy->Draw();    can->SaveAs((htmlDir + "laser_energy_distribution.gif").c_str());
  EnergyRMS->Draw(); can->SaveAs((htmlDir + "laser_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"laser_energy_distribution.gif\" alt=\"laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"laser_energy_rms_distribution.gif\" alt=\"laser energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  Timing->Draw();    can->SaveAs((htmlDir + "laser_timing_distribution.gif").c_str());
  TimingRMS->Draw(); can->SaveAs((htmlDir + "laser_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"laser_timing_distribution.gif\" alt=\"laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"laser_timing_rms_distribution.gif\" alt=\"laser timing rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
    
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  htmlFile.close();
} 

