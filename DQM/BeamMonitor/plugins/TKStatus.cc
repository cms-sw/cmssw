#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/BeamMonitor/plugins/TKStatus.h"
#include <iostream>

using namespace edm;

TKStatus::TKStatus( const ParameterSet& ps ) {
  dcsTkFileName_  = ps.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<std::string>("DIPFileName");
  {
    std::string tmpname = dcsTkFileName_;
    tmpname.insert(dcsTkFileName_.length()-4,"_TkStatus");
    dcsTkFileName_ = std::move(tmpname);
  }
  dcsStatus_ = consumes<DcsStatusCollection>(
    ps.getUntrackedParameter<std::string>("DCSStatus", "scalersRawToDigi"));
}

// ----------------------------------------------------------
void TKStatus::analyze(const Event& iEvent,
		       const EventSetup& iSetup ) {
  int nthlumi = iEvent.luminosityBlock();
  if (nthlumi > lastlumi_) { // check every LS
    lastlumi_ = nthlumi;
    // Checking TK status
    Handle<DcsStatusCollection> dcsStatus;
    iEvent.getByToken(dcsStatus_, dcsStatus);

    std::array<bool,6> dcsTk;
    for (auto& e: dcsTk) {e=true;}

    for (auto const& status: *dcsStatus) {
      if (!status.ready(DcsStatus::BPIX))   dcsTk[0]=false;
      if (!status.ready(DcsStatus::FPIX))   dcsTk[1]=false;
      if (!status.ready(DcsStatus::TIBTID)) dcsTk[2]=false;
      if (!status.ready(DcsStatus::TOB))    dcsTk[3]=false;
      if (!status.ready(DcsStatus::TECp))   dcsTk[4]=false;
      if (!status.ready(DcsStatus::TECm))   dcsTk[5]=false;
    }
    dumpTkDcsStatus(dcsTkFileName_,iEvent.run(), dcsTk );
  }
}

//--------------------------------------------------------
void TKStatus::dumpTkDcsStatus(std::string const & fileName, edm::RunNumber_t runnum, std::array<bool,6> const& dcsTk){
  std::ofstream outFile;

  outFile.open(fileName.c_str());
  outFile << "BPIX " << (dcsTk[0]?"On":"Off") << std::endl;
  outFile << "FPIX " << (dcsTk[1]?"On":"Off") << std::endl;
  outFile << "TIBTID " << (dcsTk[2]?"On":"Off") << std::endl;
  outFile << "TOB " << (dcsTk[3]?"On":"Off") << std::endl;
  outFile << "TECp " << (dcsTk[4]?"On":"Off") << std::endl;
  outFile << "TECm " << (dcsTk[5]?"On":"Off") << std::endl;
  bool AllTkOn = true;
  for (auto status: dcsTk) {
    if (!status) {
      AllTkOn = false;
      break;
    }
  }
  outFile << "WholeTrackerOn " << (AllTkOn?"Yes":"No") << std::endl;
  outFile << "Runnumber " << runnum << std::endl;

  outFile.close();
}

DEFINE_FWK_MODULE(TKStatus);
