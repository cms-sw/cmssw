#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/BeamMonitor/plugins/TKStatus.h"
#include <iostream>

using namespace edm;

TKStatus::TKStatus( const ParameterSet& ps ) : 
  checkStatus_(true) {
  parameters_     = ps;
  dcsTkFileName_  = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<std::string>("DIPFileName");
  for (int i=0;i<6;i++) dcsTk[i]=true;
  countLumi_ = lastlumi_ = 0;
  runnum = -1;
}

TKStatus::~TKStatus() {

}

//--------------------------------------------------------
void TKStatus::beginJob() {
}

//--------------------------------------------------------
void TKStatus::beginRun(const edm::Run& r, const EventSetup& context) {
  runnum = r.run();
}

//--------------------------------------------------------
void TKStatus::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
  int nthlumi = lumiSeg.luminosityBlock();
  if (nthlumi <= lastlumi_) return;
  checkStatus_ = true;
  lastlumi_ = nthlumi;
}

// ----------------------------------------------------------
void TKStatus::analyze(const Event& iEvent, 
		       const EventSetup& iSetup ) {
  if (checkStatus_) { // check every LS
    // Checking TK status
    Handle<DcsStatusCollection> dcsStatus;
    iEvent.getByLabel("scalersRawToDigi", dcsStatus);
    for (int i=0;i<6;i++) dcsTk[i]=true;
    for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); 
	 dcsStatusItr != dcsStatus->end(); ++dcsStatusItr) {
      if (!dcsStatusItr->ready(DcsStatus::BPIX))   dcsTk[0]=false;
      if (!dcsStatusItr->ready(DcsStatus::FPIX))   dcsTk[1]=false;
      if (!dcsStatusItr->ready(DcsStatus::TIBTID)) dcsTk[2]=false;
      if (!dcsStatusItr->ready(DcsStatus::TOB))    dcsTk[3]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECp))   dcsTk[4]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECm))   dcsTk[5]=false;
    }
    dumpTkDcsStatus(dcsTkFileName_);
    checkStatus_ = false;
  }
}

//--------------------------------------------------------
void TKStatus::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				  const EventSetup& iSetup) {
  int nlumi = lumiSeg.id().luminosityBlock();
  if (nlumi <= lastlumi_ ) return;

}

//--------------------------------------------------------
void TKStatus::endRun(const Run& r, const EventSetup& context){

}

//--------------------------------------------------------
void TKStatus::endJob(const LuminosityBlock& lumiSeg, 
		      const EventSetup& iSetup){

}

//--------------------------------------------------------
void TKStatus::dumpTkDcsStatus(std::string & fileName){
  std::ofstream outFile;
  std::string tmpname = fileName;
  char index[10];
  sprintf(index,"%s","_TkStatus");
  tmpname.insert(fileName.length()-4,index);

  outFile.open(tmpname.c_str());
  outFile << "BPIX " << (dcsTk[0]?"On":"Off") << std::endl;
  outFile << "FPIX " << (dcsTk[1]?"On":"Off") << std::endl;
  outFile << "TIBTID " << (dcsTk[2]?"On":"Off") << std::endl;
  outFile << "TOB " << (dcsTk[3]?"On":"Off") << std::endl;
  outFile << "TECp " << (dcsTk[4]?"On":"Off") << std::endl;
  outFile << "TECm " << (dcsTk[5]?"On":"Off") << std::endl;
  bool AllTkOn = true;
  for (int i=0; i<5; i++) {
    if (!dcsTk[i]) {
      AllTkOn = false;
      break;
    }
  }
  outFile << "WholeTrackerOn " << (AllTkOn?"Yes":"No") << std::endl;
  outFile << "Runnumber " << runnum << std::endl;
 
  outFile.close();
}

DEFINE_FWK_MODULE(TKStatus);
