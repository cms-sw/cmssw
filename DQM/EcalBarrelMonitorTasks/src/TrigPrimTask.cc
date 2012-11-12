#include "../interface/TrigPrimTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"

#include <iomanip>

namespace ecaldqm {

  TrigPrimTask::TrigPrimTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "TrigPrimTask"),
    realTps_(0),
    runOnEmul_(_workerParams.getUntrackedParameter<bool>("runOnEmul")),
//     HLTCaloPath_(_workerParams.getUntrackedParameter<std::string>("HLTCaloPath")),
//     HLTMuonPath_(_workerParams.getUntrackedParameter<std::string>("HLTMuonPath")),
//     HLTCaloBit_(false),
//     HLTMuonBit_(false),
    bxBin_(0.),
    towerReadouts_()
  {
    collectionMask_[kRun] = true;
    collectionMask_[kEBDigi] = true;
    collectionMask_[kEEDigi] = true;
    collectionMask_[kTrigPrimDigi] = true;
    collectionMask_[kTrigPrimEmulDigi] = runOnEmul_;

    // binning in terms of bunch trains
    int binEdges[nBXBins + 1] = {1, 271, 541, 892, 1162, 1432, 1783, 2053, 2323, 2674, 2944, 3214, 3446, 3490, 3491, 3565};
    for(int i(0); i < nBXBins + 1; i++) bxBinEdges_[i] = binEdges[i];
  }

  void
  TrigPrimTask::setDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kTrigPrimEmulDigi, kEBDigi, kEEDigi, kTrigPrimDigi));
  }

  void
  TrigPrimTask::bookMEs()
  {
    std::stringstream ss;

    if(runOnEmul_){
      DQWorker::bookMEs();
      MEs_["EmulMaxIndex"]->setBinLabel(-1, 1, "no maximum", 1);
      MEs_["MatchedIndex"]->setBinLabel(-1, 1, "no emul", 2);
      for(int i(2); i <= 6; i++){
	ss.str("");
	ss << (i - 1);
	MEs_["EmulMaxIndex"]->setBinLabel(-1, i, ss.str(), 1);
	MEs_["MatchedIndex"]->setBinLabel(-1, i, ss.str(), 2);
      }
    }
    else{
      std::string bookList[] = {"EtReal", "EtRealMap", "EtSummary", "EtVsBx", "OccVsBx",
			     "LowIntMap", "MedIntMap", "HighIntMap", "TTFlags", "TTFMismatch"};
      for(unsigned iME(0); iME < sizeof(bookList) / sizeof(std::string); iME++)
	MEs_[bookList[iME]]->book();
    }

    int iBin(1);
    for(int i(1); i < nBXBins + 1; i++){
      ss.str("");
      if(bxBinEdges_[i] - bxBinEdges_[i - 1] == 1) ss << bxBinEdges_[i - 1];
      else ss << bxBinEdges_[i - 1] << " - " << (bxBinEdges_[i] - 1);
      MEs_["EtVsBx"]->setBinLabel(-1, iBin, ss.str(), 1);
      MEs_["OccVsBx"]->setBinLabel(-1, iBin, ss.str(), 1);
      iBin++;
    }

    for(int iBin(1); iBin <= 8; ++iBin){
      ss.str("");
      ss << iBin - 1;
      MEs_["TTFlags"]->setBinLabel(-1, iBin, ss.str(), 2);
    }
  }

  void
  TrigPrimTask::beginEvent(const edm::Event &_evt, const edm::EventSetup &)
  {
    using namespace std;

    towerReadouts_.clear();

    realTps_ = 0;

//     HLTCaloBit_ = false;
//     HLTMuonBit_ = false;

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
      EcalTrigTowerDetId ttid(getTrigTowerMap()->towerOf(digiItr->id()));
      towerReadouts_[ttid.rawId()]++;
    }
  }

  void
  TrigPrimTask::runOnRealTPs(const EcalTrigPrimDigiCollection &_tps)
  {
    MESet* meEtVsBx(MEs_["EtVsBx"]);
    MESet* meEtReal(MEs_["EtReal"]);
    MESet* meEtRealMap(MEs_["EtRealMap"]);
    MESet* meEtSummary(MEs_["EtSummary"]);
    MESet* meLowIntMap(MEs_["LowIntMap"]);
    MESet* meMedIntMap(MEs_["MedIntMap"]);
    MESet* meHighIntMap(MEs_["HighIntMap"]);
    MESet* meTTFlags(MEs_["TTFlags"]);
    MESet* meTTFMismatch(MEs_["TTFMismatch"]);
    MESet* meOccVsBx(MEs_["OccVsBx"]);

    realTps_ = &_tps;

    double nTP[] = {0., 0., 0.};

    for(EcalTrigPrimDigiCollection::const_iterator tpItr(_tps.begin()); tpItr != _tps.end(); ++tpItr){
      EcalTrigTowerDetId ttid(tpItr->id());
      float et(tpItr->compressedEt());

      if(et > 0.){
        if(ttid.subDet() == EcalBarrel)
          nTP[0] += 1.;
        else if(ttid.zside() < 0)
          nTP[1] += 1.;
        else
          nTP[2] += 2.;
	meEtVsBx->fill(ttid, bxBin_, et);
      }

      meEtReal->fill(ttid, et);
      meEtRealMap->fill(ttid, et);
      meEtSummary->fill(ttid, et);

      int interest(tpItr->ttFlag() & 0x3);

      switch(interest){
      case 0:
	meLowIntMap->fill(ttid);
	break;
      case 1:
	meMedIntMap->fill(ttid);
	break;
      case 3:
	meHighIntMap->fill(ttid);
	break;
      default:
	break;
      }

      meTTFlags->fill(ttid, float(tpItr->ttFlag()));

      if((interest == 1 || interest == 3) && towerReadouts_[ttid.rawId()] != getTrigTowerMap()->constituentsOf(ttid).size())
	meTTFMismatch->fill(ttid);
    }

    meOccVsBx->fill(unsigned(BinService::kEB + 1), bxBin_, nTP[0]);
    meOccVsBx->fill(unsigned(BinService::kEEm + 1), bxBin_, nTP[1]);
    meOccVsBx->fill(unsigned(BinService::kEEp + 1), bxBin_, nTP[2]);
  }

  void
  TrigPrimTask::runOnEmulTPs(const EcalTrigPrimDigiCollection &_tps)
  {
    MESet* meEtMaxEmul(MEs_["EtMaxEmul"]);
    MESet* meEmulMaxIndex(MEs_["EmulMaxIndex"]);
    MESet* meMatchedIndex(MEs_["MatchedIndex"]);
    MESet* meEtEmulError(MEs_["EtEmulError"]);
    MESet* meFGEmulError(MEs_["FGEmulError"]);

    for(EcalTrigPrimDigiCollection::const_iterator tpItr(_tps.begin()); tpItr != _tps.end(); ++tpItr){
      EcalTrigTowerDetId ttid(tpItr->id());

      int et(tpItr->compressedEt());

      float maxEt(0.);
      int iMax(0);
      for(int iDigi(0); iDigi < 5; iDigi++){
	float sampleEt((*tpItr)[iDigi].compressedEt());

	if(sampleEt > maxEt){
	  maxEt = sampleEt;
	  iMax = iDigi + 1;
	}
      }

      meEtMaxEmul->fill(ttid, maxEt);
      if(maxEt > 0.)
        meEmulMaxIndex->fill(ttid, iMax + 0.5);

      bool match(true);
      bool matchFG(true);

      EcalTrigPrimDigiCollection::const_iterator realItr(realTps_->find(ttid));
      if(realItr != realTps_->end()){

	int realEt(realItr->compressedEt());

	if(realEt > 0){

          int interest(realItr->ttFlag() & 0x3);
          if((interest == 1 || interest == 3) && towerReadouts_[ttid.rawId()] == getTrigTowerMap()->constituentsOf(ttid).size()){

            if(et != realEt) match = false;
            if(tpItr->fineGrain() != realItr->fineGrain()) matchFG = false;

            std::vector<int> matchedIndex(0);
            for(int iDigi(0); iDigi < 5; iDigi++){
              if((*tpItr)[iDigi].compressedEt() == realEt)
                matchedIndex.push_back(iDigi + 1);
            }

            if(!matchedIndex.size()) matchedIndex.push_back(0);
            for(std::vector<int>::iterator matchItr(matchedIndex.begin()); matchItr != matchedIndex.end(); ++matchItr){
              meMatchedIndex->fill(ttid, *matchItr + 0.5);

              // timing information is only within emulated TPs (real TPs have one time sample)
              // 	    if(HLTCaloBit_) MEs_[kTimingCalo]->fill(ttid, float(*matchItr));
              // 	    if(HLTMuonBit_) MEs_[kTimingMuon]->fill(ttid, float(*matchItr));
            }
          }

	}
      }
      else{
	match = false;
	matchFG = false;
      }

      if(!match) meEtEmulError->fill(ttid);
      if(!matchFG) meFGEmulError->fill(ttid);
    }
  }

  DEFINE_ECALDQM_WORKER(TrigPrimTask);
}

