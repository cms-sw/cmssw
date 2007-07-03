#include "DQM/EcalPreshowerMonitorClient/interface/ESDataIntegrityClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElementBaseT.h"

ESDataIntegrityClient::ESDataIntegrityClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESDataIntegrity");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/TB");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESDataIntegrity.html");  
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
  for (int i=1; i<=36; ++i) {    
    crcErrors += hCRCError_->GetBinContent(i, 1);
  }

  double dccErrors = 0;
  for (int i=1; i<=255; ++i) {
    dccErrors += hDCCError_->GetBinContent(i);
  }

  // Show results
  htmlFile << "<h2>CRC errors : &nbsp;&nbsp;&nbsp; <span style=\"color: rgb(0, 0, 153);\">" << crcErrors << "</span></h2>"<<endl;
  if (crcErrors != 0) {
    htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" > " << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td colspan=\"2\">criminal fiber : </td>" << endl;
    htmlFile << "<td colspan=\"2\">contribution : </td>" << endl;
    htmlFile << "</tr>" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    for (int i=1; i<=36; ++i) {
      if (hCRCError_->GetBinContent(i, 1) != 0) {
	htmlFile << "<td colspan=\"2\">" << i-1 << "</td>" << endl;
	htmlFile << "<td colspan=\"2\">" << hCRCError_->GetBinContent(i, 1) << "</td>" << endl;
      }
    }
    htmlFile << "</tr>" << endl; 
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" <<endl;
  }
  
  htmlFile << "<h2>DCC errors : &nbsp;&nbsp;&nbsp; <span style=\"color: rgb(0, 0, 153);\">" << dccErrors << "</span></h2>"<<endl;
  if (dccErrors != 0) {
    htmlFile << "<table border=\"1\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" >" << endl;
    htmlFile << "<tr align=\"center\">" << endl;
    htmlFile << "<td colspan=\"2\">error code : </td>" << endl; 
    htmlFile << "<td colspan=\"2\">contribution : </td>" << endl; 
    htmlFile << "</tr>" << endl; 
    htmlFile << "<tr align=\"center\">" << endl;
    for (int i=1; i<=255; ++i) {
      if (hDCCError_->GetBinContent(i) != 0) {
	htmlFile << "<td colspan=\"2\">" << i << "</td>" << endl;
	htmlFile << "<td colspan=\"2\">" << hDCCError_->GetBinContent(i) << "</td>" << endl;
      }
    }
    htmlFile << "</tr>" << endl; 
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" <<endl;
  }

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

