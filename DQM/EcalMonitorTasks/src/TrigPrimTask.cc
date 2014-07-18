#include "../interface/TrigPrimTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iomanip>

namespace ecaldqm
{
  TrigPrimTask::TrigPrimTask() :
    DQWorkerTask(),
    realTps_(0),
    runOnEmul_(false),
    //     HLTCaloPath_(""),
    //     HLTMuonPath_(""),
    //     HLTCaloBit_(false),
    //     HLTMuonBit_(false),
    bxBinEdges_{1, 271, 541, 892, 1162, 1432, 1783, 2053, 2323, 2674, 2944, 3214, 3446, 3490, 3491, 3565},
    bxBin_(0.),
    towerReadouts_()
  {
  }

  void
  TrigPrimTask::setParams(edm::ParameterSet const& _params)
  {
    runOnEmul_ = _params.getUntrackedParameter<bool>("runOnEmul");
    if(!runOnEmul_){
      MEs_.erase(std::string("EtMaxEmul"));
      MEs_.erase(std::string("EmulMaxIndex"));
      MEs_.erase(std::string("MatchedIndex"));
      MEs_.erase(std::string("EtEmulError"));
      MEs_.erase(std::string("FGEmulError"));
    }
  }

  void
  TrigPrimTask::addDependencies(DependencySet& _dependencies)
  {
    if(runOnEmul_) _dependencies.push_back(Dependency(kTrigPrimEmulDigi, kEBDigi, kEEDigi, kTrigPrimDigi));
  }

  void
  TrigPrimTask::beginEvent(edm::Event const& _evt, edm::EventSetup const&)
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
    //  const vector<string>& pathNames(results.triggerNames());

    //  size_t caloStar(HLTCaloPath_.find('*'));
    //  if(caloStar != string::npos){
    //    string caloSub(HLTCaloPath_.substr(0, caloStar));
    //    bool found(false);
    //    for(unsigned iP(0); iP < pathNames.size(); ++iP){
    //      if(pathNames[iP].substr(0, caloStar) == caloSub){
    //        HLTCaloPath_ = pathNames[iP];
    //        found = true;
    //        break;
    //      }
    //    }
    //    if(!found) HLTCaloPath_ = "";
    //  }

    //  size_t muonStar(HLTMuonPath_.find('*'));
    //  if(muonStar != string::npos){
    //    string muonSub(HLTMuonPath_.substr(0, muonStar));
    //    bool found(false);
    //    for(unsigned iP(0); iP < pathNames.size(); ++iP){
    //      if(pathNames[iP].substr(0, muonStar) == muonSub){
    //        HLTMuonPath_ = pathNames[iP];
    //        found = true;
    //        break;
    //      }
    //    }
    //    if(!found) HLTMuonPath_ = "";
    //  }

    //  if(HLTCaloPath_.size()){
    //    try{
    //      HLTCaloBit_ = results.accept(HLTCaloPath_);
    //    }
    //    catch(cms::Exception e){
    //      if(e.category() != "LogicError") throw e;
    //      HLTCaloPath_ = "";
    //    }
    //  }
    //  if(HLTMuonPath_.size()){
    //    try{
    //      HLTMuonBit_ = results.accept(HLTMuonPath_);
    //    }
    //    catch(cms::Exception e){
    //      if(e.category() != "LogicError") throw e;
    //      HLTMuonPath_ = "";
    //    }
    //  }
    //       }
    //     }
  }

  template<typename DigiCollection>
  void
  TrigPrimTask::runOnDigis(DigiCollection const& _digis)
  {
    for(typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      EcalTrigTowerDetId ttid(getTrigTowerMap()->towerOf(digiItr->id()));
      towerReadouts_[ttid.rawId()]++;
    }
  }

  void
  TrigPrimTask::runOnRealTPs(EcalTrigPrimDigiCollection const& _tps)
  {
    MESet& meEtVsBx(MEs_.at("EtVsBx"));
    MESet& meEtReal(MEs_.at("EtReal"));
    MESet& meEtRealMap(MEs_.at("EtRealMap"));
    MESet& meEtSummary(MEs_.at("EtSummary"));
    MESet& meLowIntMap(MEs_.at("LowIntMap"));
    MESet& meMedIntMap(MEs_.at("MedIntMap"));
    MESet& meHighIntMap(MEs_.at("HighIntMap"));
    MESet& meTTFlags(MEs_.at("TTFlags"));
    MESet& meTTFMismatch(MEs_.at("TTFMismatch"));
    MESet& meOccVsBx(MEs_.at("OccVsBx"));

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
        meEtVsBx.fill(ttid, bxBin_, et);
      }

      meEtReal.fill(ttid, et);
      meEtRealMap.fill(ttid, et);
      meEtSummary.fill(ttid, et);

      int interest(tpItr->ttFlag() & 0x3);

      switch(interest){
      case 0:
        meLowIntMap.fill(ttid);
        break;
      case 1:
        meMedIntMap.fill(ttid);
        break;
      case 3:
        meHighIntMap.fill(ttid);
        break;
      default:
        break;
      }

      meTTFlags.fill(ttid, float(tpItr->ttFlag()));

      if((interest == 1 || interest == 3) && towerReadouts_[ttid.rawId()] != getTrigTowerMap()->constituentsOf(ttid).size())
        meTTFMismatch.fill(ttid);
    }

    meOccVsBx.fill(EcalBarrel, bxBin_, nTP[0]);
    meOccVsBx.fill(-EcalEndcap, bxBin_, nTP[1]);
    meOccVsBx.fill(EcalEndcap, bxBin_, nTP[2]);
  }

  void
  TrigPrimTask::runOnEmulTPs(EcalTrigPrimDigiCollection const& _tps)
  {
    MESet& meEtMaxEmul(MEs_.at("EtMaxEmul"));
    MESet& meEmulMaxIndex(MEs_.at("EmulMaxIndex"));
    MESet& meMatchedIndex(MEs_.at("MatchedIndex"));
    MESet& meEtEmulError(MEs_.at("EtEmulError"));
    MESet& meFGEmulError(MEs_.at("FGEmulError"));

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

      meEtMaxEmul.fill(ttid, maxEt);
      if(maxEt > 0.)
        meEmulMaxIndex.fill(ttid, iMax);

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
              meMatchedIndex.fill(ttid, *matchItr + 0.5);

              // timing information is only within emulated TPs (real TPs have one time sample)
              //      if(HLTCaloBit_) MEs_[kTimingCalo].fill(ttid, float(*matchItr));
              //      if(HLTMuonBit_) MEs_[kTimingMuon].fill(ttid, float(*matchItr));
            }
          }

        }
      }
      else{
        match = false;
        matchFG = false;
      }

      if(!match) meEtEmulError.fill(ttid);
      if(!matchFG) meFGEmulError.fill(ttid);
    }
  }

  DEFINE_ECALDQM_WORKER(TrigPrimTask);
}

