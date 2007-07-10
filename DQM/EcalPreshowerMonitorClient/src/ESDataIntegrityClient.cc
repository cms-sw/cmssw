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

  MonitorElement * meBC = dbe_->get(getMEName("ES BC Errors"));
  if (meBC) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meBC);           
    hBC_ = dynamic_cast<TH1F*> (meT->operator->());      
  }

  MonitorElement * meEC = dbe_->get(getMEName("ES EC Errors"));
  if (meEC) {
    meT = dynamic_cast<MonitorElementT<TNamed>*>(meEC);           
    hEC_ = dynamic_cast<TH1F*> (meT->operator->());      
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

  double bcErrors = hBC_->GetBinContent(1);
  double ecErrors = hEC_->GetBinContent(1);

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
  htmlFile << "<td colspan=\"6\">BC errors</td>" << endl;
  htmlFile << "<td colspan=\"6\">EC errors</td>" << endl;
  htmlFile << "<td colspan=\"6\">Flag1 errors</td>" << endl;
  htmlFile << "<td colspan=\"6\">Flag2 errors</td>" << endl;
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"center\">" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << crcErrors <<"</sapn></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << dccErrors <<"</span></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << bcErrors <<"</span></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << ecErrors <<"</span></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << flag1Errors <<"</span></td>" << endl;
  htmlFile << "<td colspan=\"6\"> <span style=\"color: rgb(0, 0, 153);\">" << flag2Errors <<"</span></td>" << endl;
  htmlFile << "</tr>" << endl;
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

}

