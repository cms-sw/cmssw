#include "DQM/EcalPreshowerMonitorClient/interface/ESDataIntegrityClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"

#include "TStyle.h"
#include "TPaveText.h"

ESDataIntegrityClient::ESDataIntegrityClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESDataIntegrity");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/TB");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESDataIntegrity.html");  
  // 1 : CT, 2 : TB
  detType_    = ps.getUntrackedParameter<int>("DetectorType", 1);
  sta_        = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;
  
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

}

ESDataIntegrityClient::~ESDataIntegrityClient(){
}

void ESDataIntegrityClient::endJob(){
  
  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }

  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/DataIntegrity");  

  if ( init_ ) this->cleanup();
}

void ESDataIntegrityClient::setup() {

   init_ = true;

}

void ESDataIntegrityClient::beginJob(const EventSetup& context){

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/DataIntegrity");
    dbe_->rmdir("ES/QT/DataIntegrity");
  }

}

void ESDataIntegrityClient::cleanup() {

  if (sta_) return;

  if (dbe_) {
    dbe_->setCurrentFolder("ES/QT/DataIntegrity");
  }

  init_ = false;

}

void ESDataIntegrityClient::analyze(const Event& e, const EventSetup& context){
	
  if (! init_) this->setup();

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

void ESDataIntegrityClient::doQT() {

  MonitorElementT<TNamed>* meT;

  MonitorElement * meCRCError = dbe_->get(getMEName("ES CRC Errors"));
  if (meCRCError) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meCRCError);           
    hCRCError_ = dynamic_cast<TH2F*> (meT->operator->());      
  }

  MonitorElement * meDCCError = dbe_->get(getMEName("ES DCC Errors"));
  if (meDCCError) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meDCCError);           
    hDCCError_ = dynamic_cast<TH1F*> (meT->operator->());      
  }
 
  MonitorElement * meGlbBC = dbe_->get(getMEName("ES Global BC Errors"));
  if (meGlbBC) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meGlbBC);           
    hGlbBC_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  MonitorElement * meGlbEC = dbe_->get(getMEName("ES Global EC Errors"));
  if (meGlbEC) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meGlbEC);           
    hGlbEC_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  MonitorElement * meKchipBC = dbe_->get(getMEName("ES Kchip BC Errors"));
  if (meKchipBC) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meKchipBC);           
    hKchipBC_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  MonitorElement * meKchipEC = dbe_->get(getMEName("ES Kchip EC Errors"));
  if (meKchipEC) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meKchipEC);           
    hKchipEC_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  MonitorElement * meFlag1 = dbe_->get(getMEName("ES KCHIP Flag1"));
  if (meFlag1) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meFlag1);           
    hFlag1_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  MonitorElement * meFlag2 = dbe_->get(getMEName("ES KCHIP Flag2"));
  if (meFlag2) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meFlag2);           
    hFlag2_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  MonitorElement * meEvtLen = dbe_->get(getMEName("ES Event Length"));
  if (meEvtLen) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meEvtLen);
    hEvtLen_ = dynamic_cast<TH1F*> (meT->operator->());
  }

  MonitorElement * meFedIds = dbe_->get(getMEName("ES DCC FedId"));
  if (meFedIds) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meFedIds);
    hFedIds_ = dynamic_cast<TH1F*> (meT->operator->());
  }

}

string ESDataIntegrityClient::getMEName(const string & meName) {
  
  string histoname = rootFolder_+"ES/ESDataIntegrityTask/"+meName; 

  return histoname;

}

void ESDataIntegrityClient::htmlOutput(int run, string htmlDir, string htmlName) {

  cout<<"Going to output ESDataIntegrityClient html ..."<<endl;
  
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
  htmlFile << "  <title>Preshower DQM : DataIntegrityTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<a name=""top""></a>" << endl;
  htmlFile << "<h2>Run Number / Num of Analyzed events :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;"<< count_ <<"</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task :&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Data Integrity</span></h2> " << endl;
  htmlFile << "<hr>" << endl;

  // Count errors 
  double crcErrors = 0;
  for (int i=1; i<=36; ++i) 
    crcErrors += hCRCError_->GetBinContent(i, 1);  

  double dccErrors = 0;
  for (int i=2; i<=255; ++i) 
    dccErrors += hDCCError_->GetBinContent(i);  

  double flag1Errors = 0;
  for (int i=2; i<=16; ++i) 
    flag1Errors += hFlag1_->GetBinContent(i);

  double flag2Errors = 0;
  for (int i=2; i<=255; ++i) 
    flag2Errors += hFlag2_->GetBinContent(i);

  // Show results
  htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" > " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"6\">CRC errors</td>" << endl;
  htmlFile << "<td colspan=\"6\">DCC errors</td>" << endl;
  htmlFile << "<td colspan=\"6\">Flag1 errors</td>" << endl;
  htmlFile << "<td colspan=\"6\">Flag2 errors</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << crcErrors <<"</sapn></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << dccErrors <<"</span></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << flag1Errors <<"</span></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << flag2Errors <<"</span></td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" <<endl;
  
  htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" > " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"2\">Event Length</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td>Length : </td>" << endl;
  htmlFile << "<td>contribution : </td>" << endl;
  htmlFile << "</tr>" << endl;
  for (int i=1; i<=1500; ++i) {
    if (hEvtLen_->GetBinContent(i, 1) != 0) {
      htmlFile << "<tr align=\"center\">" << endl;
      htmlFile << "<td>" << i-1 << "</td>" << endl;
      htmlFile << "<td>" << hEvtLen_->GetBinContent(i, 1) << "</td>" << endl;
      htmlFile << "</tr>" << endl; 
    }
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" <<endl;

  // Data from which Fed ID
  htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" > " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"2\">Data from which FED id </td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td>FED id : </td>" << endl;
  htmlFile << "<td>contribution : </td>" << endl;
  htmlFile << "</tr>" << endl;
  for (int i=1; i<=50; ++i) {
    if (hFedIds_->GetBinContent(i, 1) != 0) {
      htmlFile << "<tr align=\"center\">" << endl;
      htmlFile << "<td>" << i-1 << "</td>" << endl;
      htmlFile << "<td>" << hFedIds_->GetBinContent(i, 1) << "</td>" << endl;
      htmlFile << "</tr>" << endl; 
    }
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" <<endl;

  // Global BC and EC check
  htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" > " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"2\">Global BC check</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td>KCHIP id : </td>" << endl;
  htmlFile << "<td>contribution : </td>" << endl;
  htmlFile << "</tr>" << endl;
  for (int i=1; i<=45; ++i) {
    if ( hGlbBC_->GetBinContent(i) != 0) {
      htmlFile << "<tr align=\"center\">" << endl;
      htmlFile << "<td>" << hGlbBC_->GetBinCenter(i) << "</td>" << endl;
      htmlFile << "<td>" << hGlbBC_->GetBinContent(i) << "</td>" << endl;
      htmlFile << "</tr>" << endl; 
    }
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" <<endl;

  htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" > " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"2\">Global EC check</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td>KCHIP id : </td>" << endl;
  htmlFile << "<td>contribution : </td>" << endl;
  htmlFile << "</tr>" << endl;
  for (int i=1; i<=45; ++i) {
    if ( hGlbEC_->GetBinContent(i) != 0) {
      htmlFile << "<tr align=\"center\">" << endl;
      htmlFile << "<td>" << hGlbEC_->GetBinCenter(i) << "</td>" << endl;
      htmlFile << "<td>" << hGlbEC_->GetBinContent(i) << "</td>" << endl;
      htmlFile << "</tr>" << endl; 
    }
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" <<endl;

  // Local(Kchip) BC and EC check
  htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" > " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"2\">KCHIP BC check</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td>KCHIP id : </td>" << endl;
  htmlFile << "<td>contribution : </td>" << endl;
  htmlFile << "</tr>" << endl;
  for (int i=1; i<=45; ++i) {
    if ( hKchipBC_->GetBinContent(i) != 0) {
      htmlFile << "<tr align=\"center\">" << endl;
      htmlFile << "<td>" << hKchipBC_->GetBinCenter(i) << "</td>" << endl;
      htmlFile << "<td>" << hKchipBC_->GetBinContent(i) << "</td>" << endl;
      htmlFile << "</tr>" << endl; 
    }
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" <<endl;

  htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\" > " << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"2\">KCHIP EC check</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td>KCHIP id : </td>" << endl;
  htmlFile << "<td>contribution : </td>" << endl;
  htmlFile << "</tr>" << endl;
  for (int i=1; i<=45; ++i) {
    if ( hKchipEC_->GetBinContent(i) != 0) {
      htmlFile << "<tr align=\"center\">" << endl;
      htmlFile << "<td>" << hKchipEC_->GetBinCenter(i) << "</td>" << endl;
      htmlFile << "<td>" << hKchipEC_->GetBinContent(i) << "</td>" << endl;
      htmlFile << "</tr>" << endl; 
    }
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" <<endl;

  if (crcErrors != 0) {
    htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" > " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td colspan=\"2\">CRC errors</td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td>criminal fiber : </td>" << endl;
    htmlFile << "<td>contribution : </td>" << endl;
    htmlFile << "</tr>" << endl;
    for (int i=1; i<=36; ++i) {
      if (hCRCError_->GetBinContent(i, 1) != 0) {
	htmlFile << "<tr align=\"center\">" << endl;
	htmlFile << "<td>" << i-1 << "</td>" << endl;
	htmlFile << "<td>" << hCRCError_->GetBinContent(i, 1) << "</td>" << endl;
	htmlFile << "</tr>" << endl; 
      }
    }
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" <<endl;
  }
  
  if (dccErrors != 0) {
    htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" >" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td colspan=\"2\">DCC errors</td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td>error code : </td>" << endl; 
    htmlFile << "<td>contribution : </td>" << endl; 
    htmlFile << "</tr>" << endl; 
    for (int i=2; i<=255; ++i) {
      if (hDCCError_->GetBinContent(i) != 0) {
	htmlFile << "<tr align=\"center\">" << endl;
	htmlFile << "<td>" << i-1 << "</td>" << endl;
	htmlFile << "<td>" << hDCCError_->GetBinContent(i) << "</td>" << endl;
	htmlFile << "</tr>" << endl; 
      }
    }
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" <<endl;
  }

  if (flag1Errors != 0) {
    htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" >" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td colspan=\"2\">Flag1 errors</td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td>error code : </td>" << endl; 
    htmlFile << "<td>contribution : </td>" << endl; 
    htmlFile << "</tr>" << endl; 
    for (int i=2; i<=16; ++i) {
      if (hFlag1_->GetBinContent(i) != 0) {
	htmlFile << "<tr align=\"center\">" << endl;
	htmlFile << "<td> 0x" << hex << i-1 << dec << "</td>" << endl;
	htmlFile << "<td>" << hFlag1_->GetBinContent(i) << "</td>" << endl;
	htmlFile << "</tr>" << endl; 
      }
    }
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" <<endl;
  }

  if (flag2Errors != 0) {
    htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" >" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td colspan=\"2\">Flag2 errors</td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td>error code : </td>" << endl; 
    htmlFile << "<td>contribution : </td>" << endl; 
    htmlFile << "</tr>" << endl; 

    for (int i=2; i<=255; ++i) {
      if (hFlag2_->GetBinContent(i) != 0) {
	htmlFile << "<tr align=\"center\">" << endl;
	htmlFile << "<td> 0x" << hex << i-1 << dec << "</td>" << endl;
	htmlFile << "<td>" << hFlag2_->GetBinContent(i) << "</td>" << endl;
	htmlFile << "</tr>" << endl; 
      }
    }
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" <<endl;
  }

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  //--------------------------------------------------------------------------------- 	
  //--------------------------------------------------------------------------------- 	 
  // Create the main html file index.html in the appropriate directory. 	 
  // This piece of code was ported from the standalone version 	 
  // of DQM_html_generator.c by Yannis.Papadopoulos@cern.ch 	 
  
  int trig=detType_, i; 	 
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

