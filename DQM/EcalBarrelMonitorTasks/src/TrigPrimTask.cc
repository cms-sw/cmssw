#include "../interface/TrigPrimTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace ecaldqm {

  TrigPrimTask::TrigPrimTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "TrigPrimTask"),
    ttMap_(0),
    realTps_(0),
    runOnEmul_(true),
    expectedTiming_(0),
    HLTCaloPath_(""),
    HLTMuonPath_(""),
    HLTCaloBit_(false),
    HLTMuonBit_(false),
    bxBin_(0.),
    towerReadouts_()
  {
    collectionMask_ = 
      (0x1 << kRun) |
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi) |
      (0x1 << kTrigPrimDigi) |
      (0x1 << kTrigPrimEmulDigi);

    dependencies_.push_back(std::pair<Collections, Collections>(kTrigPrimEmulDigi, kEBDigi));
    dependencies_.push_back(std::pair<Collections, Collections>(kTrigPrimEmulDigi, kEEDigi));
    dependencies_.push_back(std::pair<Collections, Collections>(kTrigPrimEmulDigi, kTrigPrimDigi));

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));

    runOnEmul_ = taskParams.getUntrackedParameter<bool>("runOnEmul");
    expectedTiming_ = taskParams.getUntrackedParameter<int>("expectedTiming");
    HLTCaloPath_ = taskParams.getUntrackedParameter<std::string>("HLTCaloPath");
    HLTMuonPath_ = taskParams.getUntrackedParameter<std::string>("HLTMuonPath");

    // binning in terms of bunch trains
    int binEdges[nBXBins + 1] = {1, 271, 541, 892, 1162, 1432, 1783, 2053, 2323, 2674, 2944, 3214, 3446, 3490, 3491, 3565};
    for(int i(0); i < nBXBins + 1; i++) bxBinEdges_[i] = binEdges[i];

    if(!runOnEmul_) collectionMask_ &= ~(0x1 << kTrigPrimEmulDigi);
  }

  TrigPrimTask::~TrigPrimTask()
  {
  }

  void
  TrigPrimTask::bookMEs()
  {
    std::stringstream ss;

    if(runOnEmul_){
      DQWorker::bookMEs();
      MEs_[kEmulMaxIndex]->setBinLabel(-1, 1, "no emul", 1);
      for(int i(2); i <= 6; i++){
	ss.str("");
	ss << (i - 1);
	MEs_[kEmulMaxIndex]->setBinLabel(-1, i, ss.str(), 1);
      }
    }
    else{
      unsigned bookList[] = {kEtReal, kEtRealMap, kEtSummary, kEtVsBx, kOccVsBx,
			     kLowIntMap, kMedIntMap, kHighIntMap, kTTFlags, kTTFMismatch};
      for(unsigned iME(0); iME < sizeof(bookList) / sizeof(unsigned); iME++)
	MEs_[bookList[iME]]->book();
    }

    int iBin(1);
    for(int i(1); i < nBXBins + 1; i++){
      ss.str("");
      if(bxBinEdges_[i] - bxBinEdges_[i - 1] == 1) ss << bxBinEdges_[i - 1];
      else ss << bxBinEdges_[i - 1] << " - " << (bxBinEdges_[i] - 1);
      MEs_[kEtVsBx]->setBinLabel(-1, iBin, ss.str(), 1);
      MEs_[kOccVsBx]->setBinLabel(-1, iBin, ss.str(), 1);
      iBin++;
    }
  }

  void
  TrigPrimTask::beginRun(const edm::Run &, const edm::EventSetup &_es)
  {
    edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHndl;
    _es.get<IdealGeometryRecord>().get(ttMapHndl);
    ttMap_ = ttMapHndl.product();
  }

  void
  TrigPrimTask::beginEvent(const edm::Event &_evt, const edm::EventSetup &)
  {
    using namespace std;

    towerReadouts_.clear();

    realTps_ = 0;

    HLTCaloBit_ = false;
    HLTMuonBit_ = false;

    int* pBin(std::upper_bound(bxBinEdges_, bxBinEdges_ + nBXBins + 1, _evt.bunchCrossing()));
    bxBin_ = static_cast<int>(pBin - bxBinEdges_) - 0.5;

//     if(HLTCaloPath_.size() || HLTMuonPath_.size()){
//       edm::TriggerResultsByName results(_evt.triggerResultsByName("HLT"));
//       if(!results.isValid()) results = _evt.triggerResultsByName("RECO");
//       if(results.isValid()){
// 	const vector<string>& pathNames(results.triggerNames());

// 	size_t caloStar(HLTCaloPath_.find('*'));
// 	if(caloStar != string::npos){
// 	  string caloSub(HLTCaloPath_.substr(0, caloStar));
// 	  bool found(false);
// 	  for(unsigned iP(0); iP < pathNames.size(); ++iP){
// 	    if(pathNames[iP].substr(0, caloStar) == caloSub){
// 	      HLTCaloPath_ = pathNames[iP];
// 	      found = true;
// 	      break;
// 	    }
// 	  }
// 	  if(!found) HLTCaloPath_ = "";
// 	}

// 	size_t muonStar(HLTMuonPath_.find('*'));
// 	if(muonStar != string::npos){
// 	  string muonSub(HLTMuonPath_.substr(0, muonStar));
// 	  bool found(false);
// 	  for(unsigned iP(0); iP < pathNames.size(); ++iP){
// 	    if(pathNames[iP].substr(0, muonStar) == muonSub){
// 	      HLTMuonPath_ = pathNames[iP];
// 	      found = true;
// 	      break;
// 	    }
// 	  }
// 	  if(!found) HLTMuonPath_ = "";
// 	}

// 	if(HLTCaloPath_.size()){
// 	  try{
// 	    HLTCaloBit_ = results.accept(HLTCaloPath_);
// 	  }
// 	  catch(cms::Exception e){
// 	    if(e.category() != "LogicError") throw e;
// 	    std::cout << e.message() << std::endl;
// 	    HLTCaloPath_ = "";
// 	  }
// 	}
// 	if(HLTMuonPath_.size()){
// 	  try{
// 	    HLTMuonBit_ = results.accept(HLTMuonPath_);
// 	  }
// 	  catch(cms::Exception e){
// 	    if(e.category() != "LogicError") throw e;
// 	    std::cout << e.message() << std::endl;
// 	    HLTMuonPath_ = "";
// 	  }
// 	}
//       }
//     }
  }

  void
  TrigPrimTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      EcalTrigTowerDetId ttid(ttMap_->towerOf(digiItr->id()));
      towerReadouts_[ttid.rawId()]++;
    }
  }

  void
  TrigPrimTask::runOnRealTPs(const EcalTrigPrimDigiCollection &_tps)
  {
    realTps_ = &_tps;

    float nTP(0.);

    for(EcalTrigPrimDigiCollection::const_iterator tpItr(_tps.begin()); tpItr != _tps.end(); ++tpItr){
      EcalTrigTowerDetId ttid(tpItr->id());
      float et(tpItr->compressedEt());

      if(et > 0.){
	nTP += 1.;
	MEs_[kEtVsBx]->fill(ttid, bxBin_, et);
      }

      MEs_[kEtReal]->fill(ttid, et);
      MEs_[kEtRealMap]->fill(ttid, et);
      MEs_[kEtSummary]->fill(ttid, et);

      int interest(tpItr->ttFlag() & 0x3);

      switch(interest){
      case 0:
	MEs_[kLowIntMap]->fill(ttid);
	break;
      case 1:
	MEs_[kMedIntMap]->fill(ttid);
	break;
      case 3:
	MEs_[kHighIntMap]->fill(ttid);
	break;
      default:
	break;
      }

      MEs_[kTTFlags]->fill(ttid, float(tpItr->ttFlag()));

      if((interest == 1 || interest == 3) && towerReadouts_[ttid.rawId()] != ttMap_->constituentsOf(ttid).size())
	MEs_[kTTFMismatch]->fill(ttid);
    }

    MEs_[kOccVsBx]->fill(bxBin_, nTP);
  }

  void
  TrigPrimTask::runOnEmulTPs(const EcalTrigPrimDigiCollection &_tps)
  {
    for(EcalTrigPrimDigiCollection::const_iterator tpItr(_tps.begin()); tpItr != _tps.end(); ++tpItr){
      EcalTrigTowerDetId ttid(tpItr->id());
      int et(tpItr->compressedEt());

      //      MEs_[kEtEmul]->fill(ttid, et);

      //      MEs_[kEtEmulMap]->fill(ttid, et);

      float maxEt(0);
      int iMax(0);
      for(int iDigi(0); iDigi < 5; iDigi++){
	float sampleEt((*tpItr)[iDigi].compressedEt());

	if(sampleEt > maxEt){
	  maxEt = sampleEt;
	  iMax = iDigi + 1;
	}
      }

      MEs_[kEtMaxEmul]->fill(ttid, maxEt);
      MEs_[kEmulMaxIndex]->fill(ttid, iMax + 0.5);

      bool match(true);
      bool matchFG(true);

      EcalTrigPrimDigiCollection::const_iterator realItr(realTps_->find(ttid));
      if(realItr != realTps_->end()){

	int realEt(realItr->compressedEt());

	if(realEt <= 0) continue;

	int interest(realItr->ttFlag() & 0x3);
	if((interest == 1 || interest == 3) && towerReadouts_[ttid.rawId()] == ttMap_->constituentsOf(ttid).size()){

	  if(et != realEt) match = false;
	  if(tpItr->fineGrain() != realItr->fineGrain()) matchFG = false;

	  std::vector<int> matchedIndex(0);
	  for(int iDigi(0); iDigi < 5; iDigi++){
	    if((*tpItr)[iDigi].compressedEt() == realEt)
	      matchedIndex.push_back(iDigi + 1);
	  }

	  if(!matchedIndex.size()) matchedIndex.push_back(0);
	  for(std::vector<int>::iterator matchItr(matchedIndex.begin()); matchItr != matchedIndex.end(); ++matchItr){
	    MEs_[kMatchedIndex]->fill(ttid, *matchItr + 0.5);

	    // timing information is only within emulated TPs (real TPs have one time sample)
// 	    if(HLTCaloBit_) MEs_[kTimingCalo]->fill(ttid, float(*matchItr));
// 	    if(HLTMuonBit_) MEs_[kTimingMuon]->fill(ttid, float(*matchItr));

	    if(*matchItr != expectedTiming_) MEs_[kTimingError]->fill(ttid);
	  }

	}
      }
      else{
	match = false;
	matchFG = false;
      }

      if(!match) MEs_[kEtEmulError]->fill(ttid);
      if(!matchFG) MEs_[kFGEmulError]->fill(ttid);
    }
  }

  /*static*/
  void
  TrigPrimTask::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs indexAxis;
    indexAxis.nbins = 6;
    indexAxis.low = 0.;
    indexAxis.high = 6.;
    indexAxis.title = "TP index";

    BinService::AxisSpecs etAxis;
    etAxis.nbins = 128;
    etAxis.low = 0.;
    etAxis.high = 256.;
    etAxis.title = "TP Et";

    BinService::AxisSpecs bxAxis;
    bxAxis.nbins = 15;
    bxAxis.low = 0.;
    bxAxis.high = bxAxis.nbins;

    BinService::AxisSpecs flagAxis;
    flagAxis.nbins = 8;
    flagAxis.low = 0.;
    flagAxis.high = 8.;
    flagAxis.title = "TT flag";

    _data[kEtReal] = MEData("EtReal", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &etAxis);
    _data[kEtMaxEmul] = MEData("EtMaxEmul", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &etAxis);
    _data[kEtRealMap] = MEData("EtRealMap", BinService::kSM, BinService::kTriggerTower, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &etAxis);
    //    _data[kEtEmulMap] = MEData("EtEmulMap", BinService::kSM, BinService::kTriggerTower, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &etAxis);
    _data[kEtSummary] = MEData("EtRealMap", BinService::kEcal2P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &etAxis);
    _data[kMatchedIndex] = MEData("MatchedIndex", BinService::kSM, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F, 0, &indexAxis);
    _data[kEmulMaxIndex] = MEData("EmulMaxIndex", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &indexAxis);
    _data[kTimingError] = MEData("TimingError", BinService::kChannel, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH1F);
    _data[kEtVsBx] = MEData("EtVsBx", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TPROFILE, &bxAxis);
    _data[kOccVsBx] = MEData("OccVsBx", BinService::kEcal, BinService::kUser, MonitorElement::DQM_KIND_TPROFILE, &bxAxis);
    _data[kLowIntMap] = MEData("LowIntMap", BinService::kEcal3P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
    _data[kMedIntMap] = MEData("MedIntMap", BinService::kEcal3P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
    _data[kHighIntMap] = MEData("HighIntMap", BinService::kEcal3P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
    _data[kTTFlags] = MEData("TTFlags", BinService::kEcal2P, BinService::kDCC, MonitorElement::DQM_KIND_TH2F, 0, &flagAxis);
    _data[kTTFMismatch] = MEData("TTFMismatch", BinService::kEcal2P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
//     _data[kTimingCalo] = MEData("TimingCalo", BinService::kEcal2P, BinService::kTCC, MonitorElement::DQM_KIND_TH2F);
//     _data[kTimingMuon] = MEData("TimingMuon", BinService::kEcal2P, BinService::kTCC, MonitorElement::DQM_KIND_TH2F);
    _data[kEtEmulError] = MEData("EtEmulError", BinService::kChannel, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH1F);
    _data[kFGEmulError] = MEData("FGEmulError", BinService::kChannel, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH1F);
  }

  DEFINE_ECALDQM_WORKER(TrigPrimTask);
}


