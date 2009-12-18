#include <DQM/HcalMonitorClient/interface/HcalDataFormatClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalDataFormatClient::HcalDataFormatClient(){}


void HcalDataFormatClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  CDF_Violation_ = NULL;      //Summary histo of Common Data Format violations by FED ID
  DCCEventFormatError_ = NULL;//Summary histo of DCC Event Format violations by FED ID 
  DCCStatusBits_ = NULL;  
  DCCVersion_ = NULL;
  FEDRawDataSizes_ = NULL;
  EvFragSize_ = NULL;
  EvFragSize2_ = NULL;
  FEDEntries_ = NULL;

  LRBDataCorruptionIndicators_ = NULL; 

  HTRBCN_ = NULL;            // Bunch count number distributions
  dccBCN_ = NULL;         // Bunch count number distributions
  BCNCheck_ = NULL;       // HTR BCN compared to DCC BCN
  BCNSynch_ = NULL;       // htr-htr disagreement location
  EvtNCheck_ = NULL;      // HTR Evt # compared to DCC Evt #
  EvtNumberSynch_ = NULL; // htr-htr disagreement location
  OrNCheck_ = NULL;       // htr OrN compared to dcc OrN
  OrNSynch_ = NULL;       // htr-htr disagreement location
  BCNwhenOrNDiff_ = NULL; // BCN distribution (subset)

  HalfHTRDataCorruptionIndicators_ = NULL;
  DataFlowInd_ = NULL;
  InvHTRData_ = NULL;
  HTRFWVersionByCrate_ = NULL; 
  meUSFractSpigs_ = NULL;
  HTRStatusWdByCrate_ = NULL; //  TH2F* ErrMapbyCrate_ = NULL; //HTR error bits by crate
  HTRStatusCrate0_ = NULL;   //Map of HTR errors into Crate 0
  HTRStatusCrate1_ = NULL;   //Map of HTR errors into Crate 1
  HTRStatusCrate2_ = NULL;   //Map of HTR errors into Crate 2
  HTRStatusCrate3_ = NULL;   //Map of HTR errors into Crate 3
  HTRStatusCrate4_ = NULL;   //Map of HTR errors into Crate 4
  HTRStatusCrate5_ = NULL;   //Map of HTR errors into Crate 5
  HTRStatusCrate6_ = NULL;   //Map of HTR errors into Crate 6
  HTRStatusCrate7_ = NULL;   //Map of HTR errors into Crate 7
  HTRStatusCrate9_ = NULL;   //Map of HTR errors into Crate 9
  HTRStatusCrate10_ = NULL;  //Map of HTR errors into Crate 10
  HTRStatusCrate11_ = NULL;  //Map of HTR errors into Crate 11
  HTRStatusCrate12_ = NULL;  //Map of HTR errors into Crate 12
  HTRStatusCrate13_ = NULL;  //Map of HTR errors into Crate 13
  HTRStatusCrate14_ = NULL;  //Map of HTR errors into Crate 14
  HTRStatusCrate15_ = NULL;  //Map of HTR errors into Crate 15
  HTRStatusCrate17_ = NULL;  //Map of HTR errors into Crate 17
  for(int i=0; i<3; i++){
    HTRStatusWdByPartition_[i] = NULL;
  }

  ChannSumm_DataIntegrityCheck_ = NULL;
  for (int i=0;i<NUMDCCS;i++)
    Chann_DataIntegrityCheck_[i] = NULL;

  FibBCN_ = NULL;
  Fib1OrbMsgBCN_ = NULL;  //BCN of Fiber 1 Orb Msg
  Fib2OrbMsgBCN_ = NULL;  //BCN of Fiber 2 Orb Msg
  Fib3OrbMsgBCN_ = NULL;  //BCN of Fiber 3 Orb Msg
  Fib4OrbMsgBCN_ = NULL;  //BCN of Fiber 4 Orb Msg
  Fib5OrbMsgBCN_ = NULL;  //BCN of Fiber 5 Orb Msg
  Fib6OrbMsgBCN_ = NULL;  //BCN of Fiber 6 Orb Msg
  Fib7OrbMsgBCN_ = NULL;  //BCN of Fiber 7 Orb Msg
  Fib8OrbMsgBCN_ = NULL;  //BCN of Fiber 8 Orb Msg
}

HcalDataFormatClient::~HcalDataFormatClient(){
  this->cleanup();  
}

void HcalDataFormatClient::beginJob(void){
  if ( debug_>0 ) cout << "HcalDataFormatClient: beginJob" << endl;

  ievt_ = 0; jevt_ = 0;
  return;
}

void HcalDataFormatClient::beginRun(void){
  if ( debug_>0 ) cout << "HcalDataFormatClient: beginRun" << endl;

  jevt_ = 0;
  this->resetAllME();
  return;
}

void HcalDataFormatClient::endJob(void) {
  if ( debug_>0 ) cout << "HcalDataFormatClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

  return;
}

void HcalDataFormatClient::endRun(void) {

  if ( debug_>0 ) cout << "HcalDataFormatClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

  return;
}



void HcalDataFormatClient::cleanup(void) {
  return; // cleanup causes error?
  // Used to ask if each member TH1*, and call delete on it.
  // and the, for some reason, set it to NULL.  Clever or mad?
}



void HcalDataFormatClient::analyze(void){
  jevt_++;
  getHistograms();

  int fed2offset=0;
  int fed3offset=0;
  int spg2offset=0;
  int spg3offset=0;
  int chn2offset=0;
  float scale=1.0;
  float val=0.0;
  //Normalize everything by the -1,-1 underflow bin
  for (int fednum=0; fednum<NUMDCCS; fednum++) {
    fed3offset = 1 + (4*fednum); //3 bins, plus one of margin, each DCC
    fed2offset = 1 + (3*fednum); //2 bins, plus one of margin, each DCC
    for (int spgnum=0; spgnum<15; spgnum++) {
      spg3offset = 1 + (4*spgnum); //3 bins, plus one of margin, each spigot
      //Warning! Assumes interchangable scaling factors among these histograms!
      scale= LRBDataCorruptionIndicators_->GetBinContent(-1,-1);
      for (int xbin=1; xbin<=3; xbin++) {
	for (int ybin=1; ybin<=3; ybin++) {
	  if (!LRBDataCorruptionIndicators_) continue;
	  val = LRBDataCorruptionIndicators_->GetBinContent(fed3offset+xbin,
							    spg3offset+ybin);
	  if (val) 
	    LRBDataCorruptionIndicators_->SetBinContent(fed3offset+xbin,
							spg3offset+ybin,
							( (float)val/(float)scale ));
	  if (!HalfHTRDataCorruptionIndicators_) continue;
	  val = HalfHTRDataCorruptionIndicators_->GetBinContent(fed3offset+xbin,
								spg3offset+ybin);
	  if (val) 
	    HalfHTRDataCorruptionIndicators_->SetBinContent(fed3offset+xbin,
							    spg3offset+ybin,
							    ( (float)val/(float)scale ));
	  if (!DataFlowInd_ || xbin>2) continue;  //DataFlowInd_;  2x by 3y
	  val = DataFlowInd_->GetBinContent(fed2offset+xbin,
					    spg3offset+ybin);
	  if (val) 
	    DataFlowInd_->SetBinContent(fed2offset+xbin,
					spg3offset+ybin,
					( (float)val/(float)scale ));
	}
      }
    }
  }

  if (!ChannSumm_DataIntegrityCheck_) return;
  //Normalize by the number of events each channel spake. (Handles ZS!)
  for (int fednum=0;fednum<NUMDCCS;fednum++) {
    fed2offset = 1 + (3*fednum); //2 bins, plus one of margin, each DCC 
    for (int spgnum=0; spgnum<15; spgnum++) {
      spg2offset = 1 + (3*spgnum); //2 bins, plus one of margin, each spigot
      scale = ChannSumm_DataIntegrityCheck_->GetBinContent(fed2offset,
  							   spg2offset);
      for (int xbin=1; xbin<=2; xbin++) {
  	for (int ybin=1; ybin<=2; ybin++) {
  	  val = ChannSumm_DataIntegrityCheck_->GetBinContent(fed2offset+xbin,
  							     spg2offset+ybin);
  	  if ( (val) && (scale) ) {
  	    ChannSumm_DataIntegrityCheck_->SetBinContent(fed2offset+xbin,
  							 spg2offset+ybin,
  							 val/scale);
	    val=0.0;
	  }
  	}
      }
      //Clear the scaler, which clutters the final plot.
      ChannSumm_DataIntegrityCheck_->SetBinContent(fed2offset,
						   spg2offset, 0.0);

      if (!Chann_DataIntegrityCheck_[fednum]) continue;  
      for (int chnnum=0; chnnum<24; chnnum++) {
  	chn2offset = 1 + (3*chnnum); //2 bins, plus one of margin, each channel
	if (! (Chann_DataIntegrityCheck_[fednum]))  
	  continue;
  	scale = Chann_DataIntegrityCheck_[fednum]->GetBinContent(chn2offset,
  								 spg2offset);
  	for (int xbin=1; xbin<=2; xbin++) {
  	  for (int ybin=1; ybin<=2; ybin++) {
  	    val = Chann_DataIntegrityCheck_[fednum]->GetBinContent(chn2offset+xbin,
  								   spg2offset+ybin);
  	    if ( (val) && (scale) )
  	      Chann_DataIntegrityCheck_[fednum]->SetBinContent(chn2offset+xbin,
  							       spg2offset+ybin,
  							       val/scale);
  	  }
  	}
	Chann_DataIntegrityCheck_[fednum]->SetBinContent(chn2offset,
							 spg2offset,0.0);
      }
    }
  }

  int updates = 0;
  if ( updates % 10 == 0 ) {
    if ( debug_>0 ) cout << "HcalDataFormatClient: " << updates << " updates" << endl;
  }
  return;
}

void HcalDataFormatClient::getHistograms(bool getEmAll){

  if(!dbe_) return;
  cloneME_=false;
  char name[150];     

  sprintf(name,"DataFormatMonitor/Corruption/07 LRB Data Corruption Indicators");
  LRBDataCorruptionIndicators_ = 
    getHisto2(name, process_.c_str(), dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Corruption/08 Half-HTR Data Corruption Indicators");
  HalfHTRDataCorruptionIndicators_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Data Flow/01 Data Flow Indicators");
  DataFlowInd_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

  sprintf(name,"DataFormatMonitor/Corruption/09 Channel Integrity Summarized by Spigot");
  ChannSumm_DataIntegrityCheck_  = getHisto2(name, process_, dbe_, debug_,cloneME_);
  
  for(int i=0; i<NUMDCCS; i++) {
    sprintf(name,"DataFormatMonitor/Corruption/Channel Data Integrity/FED %03d Channel Integrity", i+700);
    Chann_DataIntegrityCheck_[i]  = getHisto2(name, process_, dbe_, debug_,cloneME_);
  }


  if (getEmAll) {
    sprintf(name,"DataFormatMonitor/Corruption/01 Common Data Format violations");
    CDF_Violation_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/02 DCC Event Format violation");
    DCCEventFormatError_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/DCC Status Bits");
    DCCStatusBits_ = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/DCC Firmware Version");
    DCCVersion_ = getHistoTProfile(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Data Flow/DCC Data Block Size Distribution");
    FEDRawDataSizes_ = getHisto(name, process_, dbe_, debug_,cloneME_);


    sprintf(name,"DataFormatMonitor/Data Flow/DCC Data Block Size Profile");
    EvFragSize_  = getHistoTProfile(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Data Flow/DCC Data Block Size Each FED");
    EvFragSize2_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Data Flow/DCC Event Counts");
    FEDEntries_ = getHisto(name, process_, dbe_, debug_,cloneME_);

  
    sprintf(name,"DataFormatMonitor/Data Flow/BCN from HTRs");
    HTRBCN_ = getHisto(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Data Flow/BCN from DCCs");
    dccBCN_  = getHisto(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/05 BCN Difference Between Ref HTR and DCC");
    BCNCheck_  = getHisto(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/05 BCN Inconsistent - HTR vs Ref HTR");
    BCNSynch_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/06 EvN Difference Between Ref HTR and DCC");
    EvtNCheck_  = getHisto(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/06 EvN Inconsistent - HTR vs Ref HTR");
    EvtNumberSynch_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/03 OrN Difference Between Ref HTR and DCC");
    OrNCheck_  = getHisto(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/03 OrN Inconsistent - HTR vs Ref HTR");
    OrNSynch_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Corruption/04 HTR BCN when OrN Diff");
    BCNwhenOrNDiff_  = getHisto(name, process_, dbe_, debug_,cloneME_);

  
    sprintf(name,"DataFormatMonitor/Diagnostics/Unpacking - HcalHTRData check failures");
    InvHTRData_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Firmware Version");
    HTRFWVersionByCrate_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR UnSuppressed Event Fractions");
    meUSFractSpigs_  = getHisto(name, process_, dbe_, debug_,cloneME_);


    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word by Crate");
    HTRStatusWdByCrate_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusWdByCrate_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 0");
    HTRStatusCrate0_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate0_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 1");
    HTRStatusCrate1_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate1_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 2");
    HTRStatusCrate2_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate2_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 3");
    HTRStatusCrate3_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate3_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 4");
    HTRStatusCrate4_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate4_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 5");
    HTRStatusCrate5_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate5_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 6");
    HTRStatusCrate6_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate6_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 7");
    HTRStatusCrate7_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate7_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 9");
    HTRStatusCrate9_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate9_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 10");
    HTRStatusCrate10_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate10_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 11");
    HTRStatusCrate11_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate11_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 12");
    HTRStatusCrate12_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate12_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 13");
    HTRStatusCrate13_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate13_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 14");
    HTRStatusCrate14_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate14_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 15");
    HTRStatusCrate15_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate15_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word - Crate 17");
    HTRStatusCrate17_ = getHisto2(name, process_, dbe_, debug_,cloneME_);
    labelyBits(HTRStatusCrate17_);
 
    for(int i=0; i<4; i++){
      if(!subDetsOn_[i]) continue;
      string type = "HBHE";
      if(i==1) type = "HBHE";
      else if(i==2) type = "HF";
      else if(i==3) type = "HO";
      sprintf(name,"DataFormatMonitor/Diagnostics/HTR Status Word %s", type.c_str());
      int ind = i-1;
      if (ind <0) ind = 0;
      HTRStatusWdByPartition_[ind] = getHisto(name, process_, dbe_, debug_,cloneME_);    
      labelxBits(HTRStatusWdByPartition_[ind]);
    }

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber Orbit Message BCN");
    FibBCN_  = getHisto(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 1 Orbit Message BCNs");
    Fib1OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 2 Orbit Message BCNs");
    Fib2OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 3 Orbit Message BCNs");
    Fib3OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 4 Orbit Message BCNs");
    Fib4OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 5 Orbit Message BCNs");
    Fib5OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 6 Orbit Message BCNs");
    Fib6OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 7 Orbit Message BCNs");
    Fib7OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);

    sprintf(name,"DataFormatMonitor/Diagnostics/HTR Fiber 8 Orbit Message BCNs");
    Fib8OrbMsgBCN_  = getHisto2(name, process_, dbe_, debug_,cloneME_);
  }
  return;
}


void HcalDataFormatClient::labelxBits(TH1F* hist){
  
  if(hist==NULL) return;

  //hist->LabelsOption("v","X");

  hist->SetXTitle("Error Bit");
  hist->GetXaxis()->SetBinLabel(1,"Overflow Warn");
  hist->GetXaxis()->SetBinLabel(2,"Buffer Busy");
  hist->GetXaxis()->SetBinLabel(3,"Empty Event");
  hist->GetXaxis()->SetBinLabel(4,"Reject L1A");
  hist->GetXaxis()->SetBinLabel(5,"Latency Err");
  hist->GetXaxis()->SetBinLabel(6,"Latency Warn");
  hist->GetXaxis()->SetBinLabel(7,"OpDat Err");
  hist->GetXaxis()->SetBinLabel(8,"Clock Err");
  hist->GetXaxis()->SetBinLabel(9,"Bunch Err");
  hist->GetXaxis()->SetBinLabel(13,"Test Mode");
  hist->GetXaxis()->SetBinLabel(14,"Histo Mode");
  hist->GetXaxis()->SetBinLabel(15,"Calib Trig");
  hist->GetXaxis()->SetBinLabel(16,"Bit15 Err");
  
  return;
}

void HcalDataFormatClient::labelyBits(TH2F* hist){
  
  if(hist==NULL) return;

  hist->SetYTitle("Error Bit");
  hist->GetYaxis()->SetBinLabel(1,"Overflow Warn");
  hist->GetYaxis()->SetBinLabel(2,"Buffer Busy");
  hist->GetYaxis()->SetBinLabel(3,"Empty Event");
  hist->GetYaxis()->SetBinLabel(4,"Reject L1A");
  hist->GetYaxis()->SetBinLabel(5,"Latency Err");
  hist->GetYaxis()->SetBinLabel(6,"Latency Warn");
  hist->GetYaxis()->SetBinLabel(7,"OpDat Err");
  hist->GetYaxis()->SetBinLabel(8,"Clock Err");
  hist->GetYaxis()->SetBinLabel(9,"Bunch Err");
  hist->GetYaxis()->SetBinLabel(13,"Test Mode");
  hist->GetYaxis()->SetBinLabel(14,"Histo Mode");
  hist->GetYaxis()->SetBinLabel(15,"Calib Trig");
  hist->GetYaxis()->SetBinLabel(16,"Bit15 Err");
  
  return;
}


void HcalDataFormatClient::report(){
  if(!dbe_) return;
  if ( debug_>0 ) cout << "HcalDataFormatClient: report" << endl;
  
  stringstream name;
  name<<process_.c_str()<<rootFolder_.c_str()<<"/DataFormatMonitor/Data Format Task Event Number";
  MonitorElement* me = dbe_->get(name.str().c_str());
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_>0 ) cout << "Found '" << name.str().c_str() << "'" << endl;
  }
  //else printf("Didn't find %s\n",name.str().c_str());
  else if (debug_>0 )
    std::cout <<"Didn't find "<<name.str().c_str()<<endl;
  getHistograms();
  
  return;
}

void HcalDataFormatClient::resetAllME(){

  if(!dbe_) return;
  
  char name[150];     
  sprintf(name,"%s%s/DataFormatMonitor/DCC Plots/Spigot Format Errors",process_.c_str(),rootFolder_.c_str());
  resetME(name,dbe_);

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HBHE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";

    sprintf(name,"%s%s/DataFormatMonitor/HTR Plots/%s Data Format Error Word",process_.c_str(),rootFolder_.c_str(), type.c_str());
    resetME(name,dbe_);

  }
  
  return;
}

void HcalDataFormatClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  if (debug_>0) cout << "Preparing HcalDataFormatClient html output ..." << endl;
  string client = "DataFormatMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Data Format Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Data Format</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DataFormatMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DataFormatMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DataFormatMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Hcal DCC Error Word</strong></h2>" << endl;  
  htmlFile << "<h3>" << endl;
  if(subDetsOn_[0]||subDetsOn_[1]) htmlFile << "<a href=\"#HBHE_Plots\">HBHE Plots </a></br>" << endl;
  //if(subDetsOn_[1]) htmlFile << "<a href=\"#HBHE_Plots\">HBHE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTMLTProfile(runNo,EvFragSize_,"FED ","Ev Frag Size (bytes)", 92, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,FEDEntries_,"HCAL FED ID","Events", 23, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusWdByCrate_,"Crate #"," ", 92, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,LRBDataCorruptionIndicators_,"HCAL FED ID","", 23, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTMLTProfile(runNo,EvFragSize_,"FED ","Ev Frag Size (bytes)", 92, htmlFile,htmlDir);
  histoHTML2       (runNo,EvFragSize2_,"FED ","Ev Frag Size (bytes)",100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusWdByCrate_,"Crate #"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo, FEDRawDataSizes_,"Ev Frag Size (bytes)", "", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,HTRBCN_,"Bunch Counter Number","Events", 23, htmlFile,htmlDir);
  histoHTML(runNo,dccBCN_,"Bunch Counter Number","Events", 23, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,CDF_Violation_,"HCAL FED ID"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,DCCEventFormatError_,"HCAL FED ID","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,DCCStatusBits_,"HCAL FED ID","", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,BCNSynch_,"Slot #","Crate #", 92, htmlFile,htmlDir);
  histoHTML2(runNo,EvtNumberSynch_,"Slot #","Crate #", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,BCNCheck_,"htr BCN - dcc BCN"," ", 92, htmlFile,htmlDir);
  histoHTML(runNo,EvtNCheck_,"htr Evt # - dcc Evt #","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
 
  htmlFile << "<tr align=\"center\">" << endl;
  histoHTML(runNo,FibBCN_,"Fiber Orbit Message BCN","Events", 30, htmlFile,htmlDir);
  histoHTML2(runNo,InvHTRData_,"Spigot #","DCC #", 23, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRFWVersionByCrate_,"Crate #","Firmware Version", 100, htmlFile,htmlDir,true);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate0_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusCrate1_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate2_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusCrate3_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate4_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusCrate5_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate6_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusCrate7_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate9_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate10_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusCrate11_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate12_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusCrate13_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate14_,"Slot #"," ", 92, htmlFile,htmlDir);
  histoHTML2(runNo,HTRStatusCrate15_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,HTRStatusCrate17_,"Slot #"," ", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  bool HBOn_ = subDetsOn_[0];
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    
    string type = "HBHE";
    if(i==1) type = "HBHE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO"; 
    if (i==1 && HBOn_) continue;
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;
    int ind = i-1;
    if (ind<0) ind = 0;
    histoHTML(runNo,HTRStatusWdByPartition_[ind],"Error Bit","Frequency", 92, htmlFile,htmlDir);
    htmlFile << "<tr align=\"left\">" << endl;
    /*
      histoHTML2(runNo,crateErrMap_[i],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
      htmlFile << "</tr>" << endl;   
      htmlFile << "<tr align=\"left\">" << endl;  
      histoHTML2(runNo,spigotErrMap_[i],"Spigot","DCC Id", 100, htmlFile,htmlDir);
      htmlFile << "</tr>" << endl;
    */
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;   
  
  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;
  
  htmlFile.close();
  return;
}

void HcalDataFormatClient::createTests(){
  //obsolete forerunner to the xml-configured Quality Tests.
  return;
}

void HcalDataFormatClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DataFormatMonitor/ZZ DQM Diagnostics/Data Format Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  char name[150]; 
  sprintf(name,"DQMData/Hcal/DataFormatMonitor/Readout Chain DataIntegrity Check");
  LRBDataCorruptionIndicators_=(TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HcalFEDChecking/FEDEntries");
  FEDEntries_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/BCN from HTRs");
  HTRBCN_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/DCC Plots/BCN from DCCs");
  dccBCN_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/BCN Difference Between Ref HTR and DCC");
  BCNCheck_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/EvN Difference Between Ref HTR and DCC");
  EvtNCheck_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/BCN of Fiber Orbit Message");
  FibBCN_ = (TH1F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/EvN Inconsistent - HTR vs Ref HTR");
  EvtNumberSynch_ = (TH2F*)infile->Get(name);
  
  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/BCN Inconsistent - HTR vs Ref HTR");
  BCNSynch_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Firmware Version");
  HTRFWVersionByCrate_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/Invalid HTR Data");
  InvHTRData_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/DCC Plots/Event Fragment Size for each FED");
  EvFragSize_ = (TProfile*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/DCC Plots/All Evt Frag Sizes");
  EvFragSize2_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/HTR Error Word by Crate");
  HTRStatusWdByCrate_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 0");
  HTRStatusCrate0_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 1");
  HTRStatusCrate1_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 2");
  HTRStatusCrate2_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 3");
  HTRStatusCrate3_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 4");
  HTRStatusCrate4_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 5");
  HTRStatusCrate5_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 6");
  HTRStatusCrate6_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 7");
  HTRStatusCrate7_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 9");
  HTRStatusCrate9_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 10");
  HTRStatusCrate10_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 11");
  HTRStatusCrate11_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 12");
  HTRStatusCrate12_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 13");
  HTRStatusCrate13_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 14");
  HTRStatusCrate14_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 15");
  HTRStatusCrate15_ = (TH2F*)infile->Get(name);

  sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/ZZ HTR Expert Plots/HTR Error Word - Crate 17");
  HTRStatusCrate17_ = (TH2F*)infile->Get(name);

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HBHE";
    if(i==1) type = "HBHE";
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";

    sprintf(name,"DQMData/Hcal/DataFormatMonitor/HTR Plots/%s Data Format Error Word", type.c_str());
    int ind = i-1;
    if (i<0) i=0;
    HTRStatusWdByPartition_[ind] = (TH1F*)infile->Get(name);    
    labelxBits(HTRStatusWdByPartition_[ind]);
    /*    
	  sprintf(name,"DQMData/Hcal/DataFormatMonitor/%s Data Format Crate Error Map", type.c_str());
	  crateErrMap_[i] = (TH2F*)infile->Get(name);

	  sprintf(name,"DQMData/Hcal/DataFormatMonitor/%s Data Format Spigot Error Map", type.c_str());
	  spigotErrMap_[i] = (TH2F*)infile->Get(name);
    */
  }

  return;
}
