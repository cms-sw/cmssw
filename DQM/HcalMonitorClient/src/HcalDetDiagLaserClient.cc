#include "DQM/HcalMonitorClient/interface/HcalDetDiagLaserClient.h"
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>
#include <TPaveStats.h>

typedef struct{
int eta;
int phi;
}Raddam_ch;
Raddam_ch RADDAM_CH[56]={{-30,15},{-32,15},{-34,15},{-36,15},{-38,15},{-40,15},{-41,15},
                         {-30,35},{-32,35},{-34,35},{-36,35},{-38,35},{-40,35},{-41,35},
                         {-30,51},{-32,51},{-34,51},{-36,51},{-38,51},{-40,51},{-41,51},
                         {-30,71},{-32,71},{-34,71},{-36,71},{-38,71},{-40,71},{-41,71},
                         {30, 01},{32, 01},{34, 01},{36, 01},{38, 01},{40, 71},{41, 71},
                         {30, 21},{32, 21},{34, 21},{36, 21},{38, 21},{40, 19},{41, 19},
                         {30, 37},{32, 37},{34, 37},{36, 37},{38, 37},{40, 35},{41, 35},
                         {30, 57},{32, 57},{34, 57},{36, 57},{38, 57},{40, 55},{41, 55}};

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
  hbheEnergy      =getHisto(folder+"HBHE Laser Energy Distribution",            process_, dbe_, debug_,cloneME_);
  hbheTiming      =getHisto(folder+"HBHE Laser Timing Distribution",            process_, dbe_, debug_,cloneME_);
  hbheEnergyRMS   =getHisto(folder+"HBHE Laser Energy RMS/Energy Distribution", process_, dbe_, debug_,cloneME_);
  hbheTimingRMS   =getHisto(folder+"HBHE Laser Timing RMS Distribution",        process_, dbe_, debug_,cloneME_);
  hoEnergy        =getHisto(folder+"HO Laser Energy Distribution",              process_, dbe_, debug_,cloneME_);
  hoTiming        =getHisto(folder+"HO Laser Timing Distribution",              process_, dbe_, debug_,cloneME_);
  hoEnergyRMS     =getHisto(folder+"HO Laser Energy RMS/Energy Distribution",   process_, dbe_, debug_,cloneME_);
  hoTimingRMS     =getHisto(folder+"HO Laser Timing RMS Distribution",          process_, dbe_, debug_,cloneME_);
  hfEnergy        =getHisto(folder+"HF Laser Energy Distribution",              process_, dbe_, debug_,cloneME_);
  hfTiming        =getHisto(folder+"HF Laser Timing Distribution",              process_, dbe_, debug_,cloneME_);
  hfEnergyRMS     =getHisto(folder+"HF Laser Energy RMS/Energy Distribution",   process_, dbe_, debug_,cloneME_);
  hfTimingRMS     =getHisto(folder+"HF Laser Timing RMS Distribution",          process_, dbe_, debug_,cloneME_);
  
  Time2Dhbhehf  =getHisto2(folder+"Laser Timing HBHEHF",                   process_, dbe_, debug_,cloneME_);
  Time2Dho      =getHisto2(folder+"Laser Timing HO",                       process_, dbe_, debug_,cloneME_);
  Energy2Dhbhehf=getHisto2(folder+"Laser Energy HBHEHF",                   process_, dbe_, debug_,cloneME_);
  Energy2Dho    =getHisto2(folder+"Laser Energy HO",                       process_, dbe_, debug_,cloneME_);
  refTime2Dhbhehf  =getHisto2(folder+"HBHEHF Laser (Timing-Ref)+1",        process_, dbe_, debug_,cloneME_);
  refTime2Dho      =getHisto2(folder+"HO Laser (Timing-Ref)+1",            process_, dbe_, debug_,cloneME_);
  refEnergy2Dhbhehf=getHisto2(folder+"HBHEHF Laser Energy/Ref",            process_, dbe_, debug_,cloneME_);
  refEnergy2Dho    =getHisto2(folder+"HO Laser Energy/Ref",                process_, dbe_, debug_,cloneME_);
  
  char str[100];
  for(int i=0;i<56;i++){   
        sprintf(str,"RADDAM (%i %i)",RADDAM_CH[i].eta,RADDAM_CH[i].phi);                                             
        Raddam[i] = getHisto(folder+str,  process_, dbe_, debug_,cloneME_);
        sprintf(str,"RADDAM (Eta=%i,Phi=%i)",RADDAM_CH[i].eta,RADDAM_CH[i].phi);     
	Raddam[i]->SetXTitle("TS");
	Raddam[i]->SetTitle(str);
	//if(Raddam[i]->GetEntries()>0)
	//for(int j=0;j<11;j++){ Raddam[i]->SetBinContent(j,(Raddam[i]->GetBinContent(j)/Raddam[i]->GetEntries()/10.0));}
  }
  
   
  MonitorElement* me = dbe_->get("Hcal/HcalDetDiagLaserMonitor/HcalDetDiagLaserMonitor Event Number");
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  me = dbe_->get("Hcal/HcalDetDiagLaserMonitor/HcalDetDiagLaserMonitor Reference Run");
  if(me) {
    string s=me->valueString();
    char str[200]; 
    sscanf((s.substr(2,s.length()-2)).c_str(), "%s", str);
    ref_run=str;
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
  TCanvas *can=new TCanvas("HcalDetDiagLaserClient","HcalDetDiagLaserClient",0,0,500,350);
  can->cd();
  
  if(Raddam[0]->GetEntries()>0){
     ofstream RADDAM;
     RADDAM.open((htmlDir + "RADDAM_"+htmlName).c_str());
     RADDAM << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">"<< endl;
     RADDAM << "<head>"<< endl;
     RADDAM << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>"<< endl;
     RADDAM << "<title>"<< "RADDAM channels" <<"</title>"<< endl;
     RADDAM << "<style type=\"text/css\">"<< endl;
     RADDAM << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"<< endl;
     RADDAM << "   td.s0 { font-family: arial, arial ce, helvetica; }"<< endl;
     RADDAM << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< endl;
     RADDAM << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }"<< endl;
     RADDAM << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }"<< endl;
     RADDAM << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }"<< endl;
     RADDAM << "</style>"<< endl;
     RADDAM << "<body>"<< endl;
     RADDAM << "<h2>Run "<< runNo<<": RADDAM channels event shape </h2>" << endl;
     RADDAM << "<table>"<< endl;
     
     char str[100];
     for(int i=0;i<28;i++){
         RADDAM << "<tr align=\"left\">" << endl;
         //Raddam[2*i]->SetStats(0);
         //Raddam[2*i+1]->SetStats(0);
         Raddam[2*i]->Draw();    sprintf(str,"%02d",2*i);    can->SaveAs((htmlDir + "raddam_ch"+str+".gif").c_str());
         Raddam[2*i+1]->Draw();  sprintf(str,"%02d",2*i+1);  can->SaveAs((htmlDir + "raddam_ch"+str+".gif").c_str());
	 sprintf(str,"raddam_ch%02d.gif",2*i);
         RADDAM << "<td align=\"center\"><img src=\""<<str<<"\" alt=\"raddam channel\">   </td>" << endl;
	 sprintf(str,"raddam_ch%02d.gif",2*i+1);
         RADDAM << "<td align=\"center\"><img src=\""<<str<<"\" alt=\"raddam channel\">   </td>" << endl;
         RADDAM << "</tr>" << endl;
     }

     RADDAM << "</table>"<< endl;
     RADDAM << "</body>"<< endl;
     RADDAM << "</html>"<< endl;
     RADDAM.close();
  }

  Time2Dhbhehf->SetXTitle("i#eta");
  Time2Dhbhehf->SetYTitle("i#phi");
  Time2Dho->SetXTitle("i#eta");
  Time2Dho->SetYTitle("i#phi");
  Energy2Dhbhehf->SetXTitle("i#eta");
  Energy2Dhbhehf->SetYTitle("i#phi");
  Energy2Dho->SetXTitle("i#eta");
  Energy2Dho->SetYTitle("i#phi");
  refTime2Dhbhehf->SetXTitle("i#eta");
  refTime2Dhbhehf->SetYTitle("i#phi");
  refTime2Dho->SetXTitle("i#eta");
  refTime2Dho->SetYTitle("i#phi");
  refEnergy2Dhbhehf->SetXTitle("i#eta");
  refEnergy2Dhbhehf->SetYTitle("i#phi");
  refEnergy2Dho->SetXTitle("i#eta");
  refEnergy2Dho->SetYTitle("i#phi");
  refTime2Dhbhehf->SetMinimum(0);
  refTime2Dhbhehf->SetMaximum(2);
  refTime2Dho->SetMinimum(0);
  refTime2Dho->SetMaximum(2);
  refEnergy2Dhbhehf->SetMinimum(0.5);
  refEnergy2Dhbhehf->SetMaximum(1.5);
  refEnergy2Dho->SetMinimum(0.5);
  refEnergy2Dho->SetMaximum(1.5);
  
  Time2Dhbhehf->SetNdivisions(36,"Y");
  Time2Dho->SetNdivisions(36,"Y");
  Energy2Dhbhehf->SetNdivisions(36,"Y");
  Energy2Dho->SetNdivisions(36,"Y");
  refTime2Dhbhehf->SetNdivisions(36,"Y");
  refTime2Dho->SetNdivisions(36,"Y");
  refEnergy2Dhbhehf->SetNdivisions(36,"Y");
  refEnergy2Dho->SetNdivisions(36,"Y");
  
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
  
  
  if(Raddam[0]->GetEntries()>0){
    htmlFile << "<h2 align=\"center\"><a href=\"" << ("RADDAM_"+htmlName).c_str() <<"\">RADDAM channels</a><h2>";
    htmlFile << "<hr>" << endl;
  }
  
  htmlFile << "<h2 align=\"center\">Stability Laser plots (Reference run "<<ref_run<<")</h2>" << endl;
  htmlFile << "<table width=100% border=0><tr>" << endl;
  
  can->SetGridy();
  can->SetGridx();
  
  htmlFile << "<tr align=\"left\">" << endl;
  refTime2Dhbhehf->SetStats(0);
  refTime2Dho->SetStats(0);
  refTime2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "ref_laser_timing_hbhehf.gif").c_str());
  refTime2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "ref_laser_timing_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_timing_hbhehf.gif\" alt=\"ref laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_timing_ho.gif\" alt=\"ref laser timing distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  htmlFile << "<tr align=\"left\">" << endl;
  refEnergy2Dhbhehf->SetStats(0);
  refEnergy2Dho->SetStats(0);
  refEnergy2Dhbhehf->Draw("COLZ");    can->SaveAs((htmlDir + "ref_laser_energy_hbhehf.gif").c_str());
  refEnergy2Dho->Draw("COLZ");        can->SaveAs((htmlDir + "ref_laser_energy_ho.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_energy_hbhehf.gif\" alt=\"ref laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ref_laser_energy_ho.gif\" alt=\"ref laser energy distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  
  
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
  hbheEnergy->Draw();    can->SaveAs((htmlDir + "hbhe_laser_energy_distribution.gif").c_str());
  hbheEnergyRMS->Draw(); can->SaveAs((htmlDir + "hbhe_laser_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_energy_distribution.gif\" alt=\"hbhe laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_energy_rms_distribution.gif\" alt=\"hbhelaser energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  hbheTiming->Draw();    can->SaveAs((htmlDir + "hbhe_laser_timing_distribution.gif").c_str());
  hbheTimingRMS->Draw(); can->SaveAs((htmlDir + "hbhe_laser_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_timing_distribution.gif\" alt=\"hbhe laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hbhe_laser_timing_rms_distribution.gif\" alt=\"hbhe laser timing rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;  
  hoEnergy->Draw();    can->SaveAs((htmlDir + "ho_laser_energy_distribution.gif").c_str());
  hoEnergyRMS->Draw(); can->SaveAs((htmlDir + "ho_laser_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_energy_distribution.gif\" alt=\"ho laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_energy_rms_distribution.gif\" alt=\"ho laser energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  hoTiming->Draw();    can->SaveAs((htmlDir + "ho_laser_timing_distribution.gif").c_str());
  hoTimingRMS->Draw(); can->SaveAs((htmlDir + "ho_laser_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_timing_distribution.gif\" alt=\"ho laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"ho_laser_timing_rms_distribution.gif\" alt=\"ho laser timing rms distribution\">   </td>" << endl;
  
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;  
  hfEnergy->Draw();    can->SaveAs((htmlDir + "hf_laser_energy_distribution.gif").c_str());
  hfEnergyRMS->Draw(); can->SaveAs((htmlDir + "hf_laser_energy_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_energy_distribution.gif\" alt=\"hf laser energy distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_energy_rms_distribution.gif\" alt=\"hf laser energy rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  hfTiming->Draw();    can->SaveAs((htmlDir + "hf_laser_timing_distribution.gif").c_str());
  hfTimingRMS->Draw(); can->SaveAs((htmlDir + "hf_laser_timing_rms_distribution.gif").c_str());
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_timing_distribution.gif\" alt=\"hf laser timing distribution\">   </td>" << endl;
  htmlFile << "<td align=\"center\"><img src=\"hf_laser_timing_rms_distribution.gif\" alt=\"hf laser timing rms distribution\">   </td>" << endl;
  htmlFile << "</tr>" << endl;
  
  
  
  
  htmlFile << "</table>" << endl;
    
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  htmlFile.close();
} 

