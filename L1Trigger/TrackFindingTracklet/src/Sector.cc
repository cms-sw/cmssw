#include "L1Trigger/TrackFindingTracklet/interface/Sector.h"
#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"

#include "L1Trigger/TrackFindingTracklet/interface/DTCLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/InputLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllInnerStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubTripletsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CandidateMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/FullMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackFitMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CleanTrackMemory.h"

#include "L1Trigger/TrackFindingTracklet/interface/InputRouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMRouterCM.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMRouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngine.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngineDisplaced.h"
#include "L1Trigger/TrackFindingTracklet/interface/TripletEngine.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProcessorDisplaced.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorDisplaced.h"
#include "L1Trigger/TrackFindingTracklet/interface/ProjectionRouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/MatchEngine.h"
#include "L1Trigger/TrackFindingTracklet/interface/MatchCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/MatchProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/FitTrack.h"
#include "L1Trigger/TrackFindingTracklet/interface/PurgeDuplicate.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubStreamData.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"

#include <deque>

using namespace std;
using namespace trklet;

Sector::Sector(Settings const& settings, Globals* globals) : isector_(-1), settings_(settings), globals_(globals) {}

Sector::~Sector() = default;

void Sector::setSector(unsigned int isector) {
  assert(isector < N_SECTOR);
  isector_ = isector;
  double dphi = 2 * M_PI / N_SECTOR;
  double dphiHG = 0.5 * settings_.dphisectorHG() - M_PI / N_SECTOR;
  phimin_ = isector_ * dphi - dphiHG;
  phimax_ = phimin_ + dphi + 2 * dphiHG;
  phimin_ -= M_PI / N_SECTOR;
  phimax_ -= M_PI / N_SECTOR;
  phimin_ = reco::reduceRange(phimin_);
  phimax_ = reco::reduceRange(phimax_);
  if (phimin_ > phimax_) {
    phimin_ -= 2 * M_PI;
  }
}

bool Sector::addStub(L1TStub stub, string dtc) {
  unsigned int layerdisk = stub.layerdisk();
  int nrbits = 3;

  if (layerdisk < N_LAYER && globals_->phiCorr(layerdisk) == nullptr) {
    globals_->phiCorr(layerdisk) = new TrackletLUT(settings_);
    globals_->phiCorr(layerdisk)->initPhiCorrTable(layerdisk, nrbits);
  }

  Stub fpgastub(stub, settings_, *globals_);

  if (layerdisk < N_LAYER) {
    FPGAWord r = fpgastub.r();
    int bendbin = fpgastub.bend().value();
    int rbin = (r.value() + (1 << (r.nbits() - 1))) >> (r.nbits() - nrbits);
    const TrackletLUT& phiCorrTable = *globals_->phiCorr(layerdisk);
    int iphicorr = phiCorrTable.lookup(bendbin * (1 << nrbits) + rbin);
    fpgastub.setPhiCorr(iphicorr);
  }

  int nadd = 0;
  for (unsigned int i = 0; i < DL_.size(); i++) {
    const string& name = DL_[i]->getName();
    if (name.find("_" + dtc) == string::npos)
      continue;
    DL_[i]->addStub(stub, fpgastub);
    nadd++;
  }

  if (!(settings_.reduced()))
    assert(nadd == 1);

  return true;
}

void Sector::addMem(string memType, string memName) {
  if (memType == "DTCLink:") {
    addMemToVec(DL_, memName, settings_, phimin_, phimax_);
  } else if (memType == "InputLink:") {
    addMemToVec(IL_, memName, settings_, phimin_, phimax_);
  } else if (memType == "AllStubs:") {
    addMemToVec(AS_, memName, settings_);
  } else if (memType == "AllInnerStubs:") {
    addMemToVec(AIS_, memName, settings_);
  } else if (memType == "VMStubsTE:") {
    addMemToVec(VMSTE_, memName, settings_);
  } else if (memType == "VMStubsME:") {
    addMemToVec(VMSME_, memName, settings_);
  } else if (memType == "StubPairs:" || memType == "StubPairsDisplaced:") {
    addMemToVec(SP_, memName, settings_);
  } else if (memType == "StubTriplets:") {
    addMemToVec(ST_, memName, settings_);
  } else if (memType == "TrackletParameters:") {
    addMemToVec(TPAR_, memName, settings_);
  } else if (memType == "TrackletProjections:") {
    addMemToVec(TPROJ_, memName, settings_);
  } else if (memType == "AllProj:") {
    addMemToVec(AP_, memName, settings_);
  } else if (memType == "VMProjections:") {
    addMemToVec(VMPROJ_, memName, settings_);
  } else if (memType == "CandidateMatch:") {
    addMemToVec(CM_, memName, settings_);
  } else if (memType == "FullMatch:") {
    addMemToVec(FM_, memName, settings_);
  } else if (memType == "TrackFit:") {
    addMemToVec(TF_, memName, settings_, phimin_, phimax_);
  } else if (memType == "CleanTrack:") {
    addMemToVec(CT_, memName, settings_, phimin_, phimax_);
  } else {
    edm::LogPrint("Tracklet") << "Don't know of memory type: " << memType;
    exit(0);
  }
}

void Sector::addProc(string procType, string procName) {
  if (procType == "InputRouter:") {
    addProcToVec(IR_, procName, settings_, globals_);
  } else if (procType == "VMRouter:") {
    addProcToVec(VMR_, procName, settings_, globals_);
  } else if (procType == "VMRouterCM:") {
    addProcToVec(VMRCM_, procName, settings_, globals_);
  } else if (procType == "TrackletEngine:") {
    addProcToVec(TE_, procName, settings_, globals_);
  } else if (procType == "TrackletEngineDisplaced:") {
    addProcToVec(TED_, procName, settings_, globals_);
  } else if (procType == "TripletEngine:") {
    addProcToVec(TRE_, procName, settings_, globals_);
  } else if (procType == "TrackletCalculator:") {
    addProcToVec(TC_, procName, settings_, globals_);
  } else if (procType == "TrackletProcessor:") {
    addProcToVec(TP_, procName, settings_, globals_);
  } else if (procType == "TrackletProcessorDisplaced:") {
    addProcToVec(TPD_, procName, settings_, globals_);
  } else if (procType == "TrackletCalculatorDisplaced:") {
    addProcToVec(TCD_, procName, settings_, globals_);
  } else if (procType == "ProjectionRouter:") {
    addProcToVec(PR_, procName, settings_, globals_);
  } else if (procType == "MatchEngine:") {
    addProcToVec(ME_, procName, settings_, globals_);
  } else if (procType == "MatchCalculator:" ||
             procType == "DiskMatchCalculator:") {  //TODO should not be used in configurations
    addProcToVec(MC_, procName, settings_, globals_);
  } else if (procType == "MatchProcessor:") {
    addProcToVec(MP_, procName, settings_, globals_);
  } else if (procType == "FitTrack:") {
    addProcToVec(FT_, procName, settings_, globals_);
  } else if (procType == "PurgeDuplicate:") {
    addProcToVec(PD_, procName, settings_, globals_);
  } else {
    edm::LogPrint("Tracklet") << "Don't know of processing type: " << procType;
    exit(0);
  }
}

void Sector::addWire(string mem, string procinfull, string procoutfull) {
  stringstream ss1(procinfull);
  string procin, output;
  getline(ss1, procin, '.');
  getline(ss1, output);

  stringstream ss2(procoutfull);
  string procout, input;
  getline(ss2, procout, '.');
  getline(ss2, input);

  MemoryBase* memory = getMem(mem);

  if (!procin.empty()) {
    ProcessBase* inProc = getProc(procin);
    inProc->addOutput(memory, output);
  }

  if (!procout.empty()) {
    ProcessBase* outProc = getProc(procout);
    outProc->addInput(memory, input);
  }
}

ProcessBase* Sector::getProc(string procName) {
  auto it = Processes_.find(procName);

  if (it != Processes_.end()) {
    return it->second;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find process : " << procName << endl;
  return nullptr;
}

MemoryBase* Sector::getMem(string memName) {
  auto it = Memories_.find(memName);

  if (it != Memories_.end()) {
    return it->second;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find memory : " << memName;
  return nullptr;
}

void Sector::writeDTCStubs(bool first) {
  for (auto& i : DL_) {
    i->writeStubs(first, isector_);
  }
}

void Sector::writeIRStubs(bool first) {
  for (auto& i : IL_) {
    i->writeStubs(first, isector_);
  }
}

void Sector::writeVMSTE(bool first) {
  for (auto& i : VMSTE_) {
    i->writeStubs(first, isector_);
  }
}

void Sector::writeVMSME(bool first) {
  for (auto& i : VMSME_) {
    i->writeStubs(first, isector_);
  }
}

void Sector::writeAS(bool first) {
  for (auto& i : AS_) {
    i->writeStubs(first, isector_);
  }
}

void Sector::writeAIS(bool first) {
  for (auto& i : AIS_) {
    i->writeStubs(first, isector_);
  }
}

void Sector::writeSP(bool first) {
  for (auto& i : SP_) {
    i->writeSP(first, isector_);
  }
}

void Sector::writeST(bool first) {
  for (auto& i : ST_) {
    i->writeST(first, isector_);
  }
}

void Sector::writeTPAR(bool first) {
  for (auto& i : TPAR_) {
    i->writeTPAR(first, isector_);
  }
}

void Sector::writeTPROJ(bool first) {
  for (auto& i : TPROJ_) {
    i->writeTPROJ(first, isector_);
  }
}

void Sector::writeAP(bool first) {
  for (auto& i : AP_) {
    i->writeAP(first, isector_);
  }
}

void Sector::writeVMPROJ(bool first) {
  for (auto& i : VMPROJ_) {
    i->writeVMPROJ(first, isector_);
  }
}

void Sector::writeCM(bool first) {
  for (auto& i : CM_) {
    i->writeCM(first, isector_);
  }
}

void Sector::writeMC(bool first) {
  for (auto& i : FM_) {
    i->writeMC(first, isector_);
  }
}

void Sector::writeTF(bool first) {
  for (auto& i : TF_) {
    i->writeTF(first, isector_);
  }
}

void Sector::writeCT(bool first) {
  for (auto& i : CT_) {
    i->writeCT(first, isector_);
  }
}

void Sector::clean() {
  for (auto& mem : MemoriesV_) {
    mem->clean();
  }
}

void Sector::executeIR() {
  for (auto& i : IR_) {
    i->execute();
  }
}

void Sector::executeVMR() {
  if (settings_.writeMonitorData("IL")) {
    ofstream& out = globals_->ofstream("inputlink.txt");
    for (auto& i : IL_) {
      out << i->getName() << " " << i->nStubs() << endl;
    }
  }
  for (auto& i : VMR_) {
    i->execute();
  }
  for (auto& i : VMRCM_) {
    i->execute(isector_);
  }
}

void Sector::executeTE() {
  for (auto& i : TE_) {
    i->execute();
  }
}

void Sector::executeTED() {
  for (auto& i : TED_) {
    i->execute();
  }
}

void Sector::executeTRE() {
  for (auto& i : TRE_) {
    i->execute();
  }
}

void Sector::executeTP() {
  for (auto& i : TP_) {
    i->execute(isector_, phimin_, phimax_);
  }
}

void Sector::executeTPD() {
  for (auto& i : TPD_) {
    i->execute(isector_, phimin_, phimax_);
  }
}

void Sector::executeTC() {
  for (auto& i : TC_) {
    i->execute(isector_, phimin_, phimax_);
  }

  if (settings_.writeMonitorData("TrackProjOcc")) {
    ofstream& out = globals_->ofstream("trackprojocc.txt");
    for (auto& i : TPROJ_) {
      out << i->getName() << " " << i->nTracklets() << endl;
    }
  }
}

void Sector::executeTCD() {
  for (auto& i : TCD_) {
    i->execute(isector_, phimin_, phimax_);
  }
}

void Sector::executePR() {
  for (auto& i : PR_) {
    i->execute();
  }
}

void Sector::executeME() {
  for (auto& i : ME_) {
    i->execute(isector_);
  }
}

void Sector::executeMC() {
  for (auto& i : MC_) {
    i->execute(isector_, phimin_);
  }
}

void Sector::executeMP() {
  for (auto& i : MP_) {
    i->execute(isector_, phimin_);
  }
}

// Order here reflects Tracklet algo that calls FitTrack before PurgeDuplicates.
// If using Hybrid, then PurgeDuplicates runs both duplicate removal & KF steps.
// (unless duplicate removal disabled, in which case FitTrack runs KF).

void Sector::executeFT(vector<vector<string>>& streamsTrackRaw, vector<vector<StubStreamData>>& streamsStubRaw) {
  const int numChannels = streamsTrackRaw.size() / N_SECTOR;
  const int maxNumProjectionLayers = streamsStubRaw.size() / streamsTrackRaw.size();
  const int offsetTrack = isector_ * numChannels;
  int channelTrack(0);

  for (auto& i : FT_) {
    // Temporary streams for a single TrackBuilder (i.e. seed type)
    deque<string> streamTrackTmp;
    vector<deque<StubStreamData>> streamsStubTmp(maxNumProjectionLayers);
    i->execute(streamTrackTmp, streamsStubTmp, isector_);
    if (!settings_.storeTrackBuilderOutput())
      continue;
    const int offsetStub = (offsetTrack + channelTrack) * maxNumProjectionLayers;
    streamsTrackRaw[offsetTrack + channelTrack] = vector<string>(streamTrackTmp.begin(), streamTrackTmp.end());
    channelTrack++;
    int channelStub(0);
    for (auto& stream : streamsStubTmp)
      streamsStubRaw[offsetStub + channelStub++] = vector<StubStreamData>(stream.begin(), stream.end());
  }
}

// Returns tracks reconstructed by L1 track chain.
void Sector::executePD(std::vector<Track>& tracks) {
  for (auto& i : PD_) {
    i->execute(tracks, isector_);
  }
}

std::vector<Tracklet*> Sector::getAllTracklets() const {
  std::vector<Tracklet*> tmp;
  for (auto& tpar : TPAR_) {
    for (unsigned int j = 0; j < tpar->nTracklets(); j++) {
      tmp.push_back(tpar->getTracklet(j));
    }
  }
  return tmp;
}

std::vector<const Stub*> Sector::getStubs() const {
  std::vector<const Stub*> tmp;

  for (auto& imem : IL_) {
    for (unsigned int istub = 0; istub < imem->nStubs(); istub++) {
      tmp.push_back(imem->getStub(istub));
    }
  }

  return tmp;
}

std::unordered_set<int> Sector::seedMatch(int itp) const {
  std::unordered_set<int> tmpSeeds;
  for (auto& i : TPAR_) {
    unsigned int nTracklet = i->nTracklets();
    for (unsigned int j = 0; j < nTracklet; j++) {
      if (i->getTracklet(j)->tpseed() == itp) {
        tmpSeeds.insert(i->getTracklet(j)->getISeed());
      }
    }
  }
  return tmpSeeds;
}
