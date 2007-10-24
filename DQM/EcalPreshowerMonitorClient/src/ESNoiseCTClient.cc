#include "DQM/EcalPreshowerMonitorClient/interface/ESNoiseCTClient.h"

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

ESNoiseCTClient::ESNoiseCTClient(const ParameterSet& ps) {
  
  writeHisto_ = ps.getUntrackedParameter<bool>("writeHisto", true);
  writeHTML_  = ps.getUntrackedParameter<bool>("writeHTML", true);
  dumpRate_   = ps.getUntrackedParameter<int>("dumpRate", 100);
  outputFileName_ = ps.getUntrackedParameter<string>("outputFileName", "ESNoiseCT");
  rootFolder_ = ps.getUntrackedParameter<string>("rootFolder", "");
  htmlDir_    = ps.getUntrackedParameter<string>("htmlDir","/preshower/DQM/CT");
  htmlName_   = ps.getUntrackedParameter<string>("htmlName","ESNoiseCT.html");  
  sta_        = ps.getUntrackedParameter<bool>("RunStandalone", false);

  count_ = 0;
  run_ = -1;
  init_ = false;

  fg = new TF1("fg", "gaus");
}

ESNoiseCTClient::~ESNoiseCTClient(){

  delete fg;

}

void ESNoiseCTClient::beginJob(const EventSetup& context){

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if (dbe_) {
    dbe_->setVerbose(1);
    dbe_->setCurrentFolder("ES/QT/NoiseCT");
    dbe_->rmdir("ES/QT/NoiseCT");
  }

} 

void ESNoiseCTClient::endJob(){

  Char_t runNum_s[50];
  sprintf(runNum_s, "%08d", run_);
  outputFile_ = htmlDir_+"/"+runNum_s+"/"+outputFileName_+"_"+runNum_s+".root";
  
  if (writeHTML_) {
    doQT();
    htmlOutput(run_, htmlDir_, htmlName_);
  }
  
  if (writeHisto_) dbe_->save(outputFile_);
  dbe_->rmdir("ES/QT/NoiseCT");  

  if ( init_ ) this->cleanup();
}

void ESNoiseCTClient::setup() {

  init_ = true;

}

void ESNoiseCTClient::cleanup() {

  if (sta_) return;

  init_ = false;

}

void ESNoiseCTClient::analyze(const Event& e, const EventSetup& context){
  
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

void ESNoiseCTClient::doQT() {

  TH1F* htmp = new TH1F("htmp", "htmp", 4100, 0, 4100);

  int count;
  double mean, rms; 
  for (int i=0; i<1; ++i) {    
    for (int j=0; j<1; ++j) {
      for (int k=0; k<2; ++k) {
	for (int m=0; m<5; ++m) {
	  
	  int zside = (i==0)?1:-1;
	  MonitorElement * occME = dbe_->get(getMEName(zside, j+1, k+1, m+1));
	  count = -1;	 
	  htmp->Clear();

	  if (occME) {
	    MonitorElementT<TNamed>* occ = dynamic_cast<MonitorElementT<TNamed>*>(occME);           
	    TH1F *hNoise = dynamic_cast<TH1F*> (occ->operator->());      
	    
	    for (int n=0; n<32; ++n) {
	      htmp->Fill(hNoise->GetRMS(n+1));	    
	      cout<<"Noise : "<<n+1<<" "<<hNoise->GetRMS(n+1)<<endl;
	    }
	    mean = htmp->GetMean();
	    rms = htmp->GetRMS();

	    htmp->Clear();
	    for (int n=0; n<32; ++n) {
	      if (hNoise->GetRMS(n+1)<mean+3.*rms && hNoise->GetRMS(n+1)>mean-3.*rms)
		htmp->Fill(hNoise->GetRMS(n+1));
	      else 
		cout<<"L1 : "<<i+1<<" "<<j+1<<" "<<k+1<<" "<<m+1<<" "<<n+1<<endl;
	    }
	    
	    mean = htmp->GetMean();
	    rms = htmp->GetRMS();

	    htmp->Clear();
	    for (int n=0; n<32; ++n) {
	      if (hNoise->GetRMS(n+1)<mean+3.*rms && hNoise->GetRMS(n+1)>mean-3.*rms)
		htmp->Fill(hNoise->GetRMS(n+1));
	      else
		cout<<"L2 : "<<i+1<<" "<<j+1<<" "<<k+1<<" "<<m+1<<" "<<n+1<<endl;
	    }
	    mean = htmp->GetMean();
	    rms = htmp->GetRMS();

	    htmp->Clear();
	    for (int n=0; n<32; ++n) {
	      if (hNoise->GetRMS(n+1)<mean+3.*rms && hNoise->GetRMS(n+1)>mean-3.*rms)
		htmp->Fill(hNoise->GetRMS(n+1));
	      else
		cout<<"L3 : "<<i+1<<" "<<j+1<<" "<<k+1<<" "<<m+1<<" "<<n+1<<endl;
	    }
	    
	  }		  
	}	
      }
    }
  }

  delete htmp;
}

string ESNoiseCTClient::getMEName(const int & zside, const int & plane, const int & row, const int & col) {
  
  Char_t hist[500];
  sprintf(hist,"%sES/QT/PedestalCT/ES Pedestal Fit Mean RMS Z %d P %d Row %02d Col %02d",rootFolder_.c_str(),zside,plane,row,col);

  return hist;
}

void ESNoiseCTClient::htmlOutput(int run, string htmlDir, string htmlName) {

  cout<<"Going to output ESNoiseCTClient html ..."<<endl;
  
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
  htmlFile << "  <title>Preshower DQM : NoiseCTTask output</title> " << endl;
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

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}
