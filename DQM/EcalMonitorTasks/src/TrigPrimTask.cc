#include "DQM/EcalMonitorTasks/interface/TrigPrimTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iomanip>

namespace ecaldqm {
  TrigPrimTask::TrigPrimTask()
      : DQWorkerTask(),
        realTps_(nullptr),
        runOnEmul_(false),
        //     HLTCaloPath_(""),
        //     HLTMuonPath_(""),
        //     HLTCaloBit_(false),
        //     HLTMuonBit_(false),
        bxBinEdges_(),
        bxBinEdgesFine_(),
        bxBin_(0.),
        bxBinFine_(0.),
        towerReadouts_(),
        lhcStatusInfoCollectionTag_() {}

  void TrigPrimTask::setParams(edm::ParameterSet const& _params) {
    runOnEmul_ = _params.getUntrackedParameter<bool>("runOnEmul");
    if (!runOnEmul_) {
      MEs_.erase(std::string("EtMaxEmul"));
      MEs_.erase(std::string("EmulMaxIndex"));
      MEs_.erase(std::string("MatchedIndex"));
      MEs_.erase(std::string("EtEmulError"));
      MEs_.erase(std::string("FGEmulError"));
      MEs_.erase(std::string("RealvEmulEt"));
    }
    lhcStatusInfoCollectionTag_ = _params.getUntrackedParameter<edm::InputTag>(
        "lhcStatusInfoCollectionTag", edm::InputTag("tcdsDigis", "tcdsRecord"));
    bxBinEdges_ = _params.getUntrackedParameter<std::vector<int> >("bxBins");
    bxBinEdgesFine_ = _params.getUntrackedParameter<std::vector<int> >("bxBinsFine");
  }

  void TrigPrimTask::addDependencies(DependencySet& _dependencies) {
    if (runOnEmul_)
      _dependencies.push_back(Dependency(kTrigPrimEmulDigi, kEBDigi, kEEDigi, kTrigPrimDigi));
  }

  void TrigPrimTask::beginRun(edm::Run const&, edm::EventSetup const& _es) {
    // Read-in Status records:
    // Status records stay constant over run so they are read-in only once here
    // but filled by LS in runOnRealTPs() because MEs are not yet booked at beginRun()
    TTStatus = &_es.getData(TTStatusRcd_);
    StripStatus = &_es.getData(StripStatusRcd_);
  }

  void TrigPrimTask::beginEvent(edm::Event const& _evt,
                                edm::EventSetup const& _es,
                                bool const& ByLumiResetSwitch,
                                bool& lhcStatusSet) {
    using namespace std;

    towerReadouts_.clear();

    if (ByLumiResetSwitch) {
      MEs_.at("EtSummaryByLumi").reset(GetElectronicsMap());
      MEs_.at("TTFlags4ByLumi").reset(GetElectronicsMap());
      MEs_.at("LHCStatusByLumi").reset(GetElectronicsMap(), -1);
    }

    if (!lhcStatusSet) {
      // Update LHC status once each LS
      MESet& meLHCStatusByLumi(static_cast<MESet&>(MEs_.at("LHCStatusByLumi")));
      edm::Handle<TCDSRecord> tcdsData;
      _evt.getByToken(lhcStatusInfoRecordToken_, tcdsData);
      if (tcdsData.isValid()) {
        meLHCStatusByLumi.fill(getEcalDQMSetupObjects(), double(tcdsData->getBST().getBeamMode()));
        lhcStatusSet = true;
      }
    }

    realTps_ = nullptr;

    //     HLTCaloBit_ = false;
    //     HLTMuonBit_ = false;

    std::vector<int>::iterator pBin(std::upper_bound(bxBinEdges_.begin(), bxBinEdges_.end(), _evt.bunchCrossing()));
    bxBin_ = static_cast<int>(pBin - bxBinEdges_.begin()) - 0.5;
    // fine binning for TP Occ vs BX plot as requested by DAQ in March 2021
    std::vector<int>::iterator pBinFine(
        std::upper_bound(bxBinEdgesFine_.begin(), bxBinEdgesFine_.end(), _evt.bunchCrossing()));
    bxBinFine_ = static_cast<int>(pBinFine - bxBinEdgesFine_.begin()) - 0.5;

    const EcalTPGTowerStatusMap& towerMap = TTStatus->getMap();
    const EcalTPGStripStatusMap& stripMap = StripStatus->getMap();

    MESet& meTTMaskMap(MEs_.at("TTMaskMap"));

    for (EcalTPGTowerStatusMap::const_iterator ttItr(towerMap.begin()); ttItr != towerMap.end(); ++ttItr) {
      if ((*ttItr).second > 0) {
        const EcalTrigTowerDetId ttid((*ttItr).first);
        //if(ttid.subDet() == EcalBarrel)
        meTTMaskMap.fill(getEcalDQMSetupObjects(), ttid, 1);
      }  //masked
    }  //loop on towers

    for (EcalTPGStripStatusMap::const_iterator stItr(stripMap.begin()); stItr != stripMap.end(); ++stItr) {
      if ((*stItr).second > 0) {
        const EcalElectronicsId stid((*stItr).first);
        //if(stid.subdet() == EcalEndcap);
        meTTMaskMap.fill(getEcalDQMSetupObjects(), stid, 1);
      }  //masked
    }  //loop on pseudo-strips

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

  template <typename DigiCollection>
  void TrigPrimTask::runOnDigis(DigiCollection const& _digis) {
    for (typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr) {
      EcalTrigTowerDetId ttid(GetTrigTowerMap()->towerOf(digiItr->id()));
      towerReadouts_[ttid.rawId()]++;
    }
  }

  void TrigPrimTask::setTokens(edm::ConsumesCollector& _collector) {
    lhcStatusInfoRecordToken_ = _collector.consumes<TCDSRecord>(lhcStatusInfoCollectionTag_);
    TTStatusRcd_ = _collector.esConsumes<edm::Transition::BeginRun>();
    StripStatusRcd_ = _collector.esConsumes<edm::Transition::BeginRun>();
  }

  void TrigPrimTask::runOnRealTPs(EcalTrigPrimDigiCollection const& _tps) {
    MESet& meEtVsBx(MEs_.at("EtVsBx"));
    MESet& meEtReal(MEs_.at("EtReal"));
    MESet& meEtRealMap(MEs_.at("EtRealMap"));
    MESet& meEtSummary(MEs_.at("EtSummary"));
    MESet& meEtSummaryByLumi(MEs_.at("EtSummaryByLumi"));
    MESet& meLowIntMap(MEs_.at("LowIntMap"));
    MESet& meMedIntMap(MEs_.at("MedIntMap"));
    MESet& meHighIntMap(MEs_.at("HighIntMap"));
    MESet& meTTFlags(MEs_.at("TTFlags"));
    MESet& meTTFlagsVsEt(MEs_.at("TTFlagsVsEt"));
    MESet& meTTFlags4(MEs_.at("TTFlags4"));
    MESet& meTTFlags4ByLumi(MEs_.at("TTFlags4ByLumi"));
    MESet& meTTFMismatch(MEs_.at("TTFMismatch"));
    MESet& meOccVsBx(MEs_.at("OccVsBx"));

    realTps_ = &_tps;

    double nTP[] = {0., 0., 0.};

    for (EcalTrigPrimDigiCollection::const_iterator tpItr(_tps.begin()); tpItr != _tps.end(); ++tpItr) {
      EcalTrigTowerDetId ttid(tpItr->id());
      float et(tpItr->compressedEt());

      if (et > 0.) {
        if (ttid.subDet() == EcalBarrel)
          nTP[0] += 1.;
        else if (ttid.zside() < 0)
          nTP[1] += 1.;
        else
          nTP[2] += 2.;
        meEtVsBx.fill(getEcalDQMSetupObjects(), ttid, bxBin_, et);
      }

      meEtReal.fill(getEcalDQMSetupObjects(), ttid, et);
      meEtRealMap.fill(getEcalDQMSetupObjects(), ttid, et);
      meEtSummary.fill(getEcalDQMSetupObjects(), ttid, et);
      meEtSummaryByLumi.fill(getEcalDQMSetupObjects(), ttid, et);

      int interest(tpItr->ttFlag() & 0x3);

      switch (interest) {
        case 0:
          meLowIntMap.fill(getEcalDQMSetupObjects(), ttid);
          break;
        case 1:
          meMedIntMap.fill(getEcalDQMSetupObjects(), ttid);
          break;
        case 3:
          meHighIntMap.fill(getEcalDQMSetupObjects(), ttid);
          break;
        default:
          break;
      }

      // Fill TT Flag MEs
      int ttF(tpItr->ttFlag());
      meTTFlags.fill(getEcalDQMSetupObjects(), ttid, 1.0 * ttF);
      meTTFlagsVsEt.fill(getEcalDQMSetupObjects(), ttid, et, 1.0 * ttF);
      // Monitor occupancy of TTF=4
      // which contains info about TT auto-masking
      if (ttF >= 4) {
        meTTFlags4.fill(getEcalDQMSetupObjects(), ttid);
        meTTFlags4ByLumi.fill(getEcalDQMSetupObjects(), ttid);
      }
      if ((ttF == 1 || ttF == 3) && towerReadouts_[ttid.rawId()] != GetTrigTowerMap()->constituentsOf(ttid).size())
        meTTFMismatch.fill(getEcalDQMSetupObjects(), ttid);
    }

    meOccVsBx.fill(getEcalDQMSetupObjects(), EcalBarrel, bxBinFine_, nTP[0]);
    meOccVsBx.fill(getEcalDQMSetupObjects(), -EcalEndcap, bxBinFine_, nTP[1]);
    meOccVsBx.fill(getEcalDQMSetupObjects(), EcalEndcap, bxBinFine_, nTP[2]);

    // Set TT/Strip Masking status in Ecal3P view
    // Status Records are read-in at beginRun() but filled here
    // Requestied by ECAL Trigger in addition to TTMaskMap plots in SM view
    MESet& meTTMaskMapAll(MEs_.at("TTMaskMapAll"));

    // Fill from TT Status Rcd
    const EcalTPGTowerStatusMap& TTStatusMap(TTStatus->getMap());
    for (EcalTPGTowerStatusMap::const_iterator ttItr(TTStatusMap.begin()); ttItr != TTStatusMap.end(); ++ttItr) {
      const EcalTrigTowerDetId ttid(ttItr->first);
      if (ttItr->second > 0)
        meTTMaskMapAll.setBinContent(getEcalDQMSetupObjects(), ttid, 1);  // TT is masked
    }  // TTs

    // Fill from Strip Status Rcd
    const EcalTPGStripStatusMap& StripStatusMap(StripStatus->getMap());
    for (EcalTPGStripStatusMap::const_iterator stItr(StripStatusMap.begin()); stItr != StripStatusMap.end(); ++stItr) {
      const EcalTriggerElectronicsId stid(stItr->first);
      // Since ME has kTriggerTower binning, convert to EcalTrigTowerDetId first
      // In principle, setBinContent() could be implemented for EcalTriggerElectronicsId class as well
      const EcalTrigTowerDetId ttid(GetElectronicsMap()->getTrigTowerDetId(stid.tccId(), stid.ttId()));
      if (stItr->second > 0)
        meTTMaskMapAll.setBinContent(getEcalDQMSetupObjects(), ttid, 1);  // PseudoStrip is masked
    }  // PseudoStrips

  }  // TrigPrimTask::runOnRealTPs()

  void TrigPrimTask::runOnEmulTPs(EcalTrigPrimDigiCollection const& _tps) {
    MESet& meEtMaxEmul(MEs_.at("EtMaxEmul"));
    MESet& meEmulMaxIndex(MEs_.at("EmulMaxIndex"));
    MESet& meMatchedIndex(MEs_.at("MatchedIndex"));
    MESet& meEtEmulError(MEs_.at("EtEmulError"));
    MESet& meFGEmulError(MEs_.at("FGEmulError"));
    MESet& meRealvEmulEt(MEs_.at("RealvEmulEt"));

    for (EcalTrigPrimDigiCollection::const_iterator tpItr(_tps.begin()); tpItr != _tps.end(); ++tpItr) {
      EcalTrigTowerDetId ttid(tpItr->id());

      int et(tpItr->compressedEt());

      float maxEt(0.);
      int iMax(0);
      for (int iDigi(0); iDigi < 5; iDigi++) {
        float sampleEt((*tpItr)[iDigi].compressedEt());

        if (sampleEt > maxEt) {
          maxEt = sampleEt;
          iMax = iDigi + 1;
        }
      }

      meEtMaxEmul.fill(getEcalDQMSetupObjects(), ttid, maxEt);
      if (maxEt > 0.)
        meEmulMaxIndex.fill(getEcalDQMSetupObjects(), ttid, iMax);

      bool match(true);
      bool matchFG(true);

      // Loop over real TPs and look for an emulated TP index with matching Et:
      // If an Et match is found, return TP index correpsonding to BX of emulated TP where match was found
      // Standard TPG comparison: { TP index:matched BX } = { no emul:No Et match, 0:BX-2, 1:BX-1, 2:in-time, 3:BX+1, 4:BX+2 }
      EcalTrigPrimDigiCollection::const_iterator realItr(realTps_->find(ttid));
      if (realItr != realTps_->end()) {
        int realEt(realItr->compressedEt());

        if (realEt > 0) {
          int ttF(realItr->ttFlag());
          if ((ttF == 1 || ttF == 3) &&
              towerReadouts_[ttid.rawId()] == GetTrigTowerMap()->constituentsOf(ttid).size()) {
            if (et != realEt)
              match = false;
            if (tpItr->fineGrain() != realItr->fineGrain())
              matchFG = false;

            // NOTE: matchedIndex comparison differs from Standard TPG comparison:
            // { matchedIndex:TP index } = { 0:no emul, 1:BX-2, 2:BX-1, 3:in-time, 4:BX+1, 5:BX+2 }
            std::vector<int> matchedIndex(0);
            // iDigi only loops over explicit Et matches:
            // { iDigi:TP index } = { 0:BX-2, 1:BX-1, 2:in-time, 3:BX+1, 4:BX+2 }
            for (int iDigi(0); iDigi < 5; iDigi++) {
              if ((*tpItr)[iDigi].compressedEt() == realEt) {
                // matchedIndex = iDigi + 1
                if (iDigi != 2) {
                  matchedIndex.push_back(iDigi + 1);
                }
                // If an in-time match is found, exit loop and clear out any other matches:
                // Ensures multiple matches are not returned (e.g. during saturation)
                else {
                  matchedIndex.clear();
                  matchedIndex.push_back(3);  // Et match is to in-time emulated TP
                  break;
                }
              }  // Et match found
            }  // iDigi
            if (matchedIndex.empty())
              matchedIndex.push_back(0);  // no Et match found => no emul

            // Fill Real vs Emulated TP Et
            meRealvEmulEt.fill(
                getEcalDQMSetupObjects(), ttid, realEt, (*tpItr)[2].compressedEt());  // iDigi=2:in-time BX

            // Fill matchedIndex ME
            for (std::vector<int>::iterator matchItr(matchedIndex.begin()); matchItr != matchedIndex.end();
                 ++matchItr) {
              meMatchedIndex.fill(getEcalDQMSetupObjects(), ttid, *matchItr + 0.5);

              // timing information is only within emulated TPs (real TPs have one time sample)
              //      if(HLTCaloBit_) MEs_[kTimingCalo].fill(ttid, float(*matchItr));
              //      if(HLTMuonBit_) MEs_[kTimingMuon].fill(ttid, float(*matchItr));
            }
          }
        }
      } else {
        match = false;
        matchFG = false;
      }

      if (!match)
        meEtEmulError.fill(getEcalDQMSetupObjects(), ttid);
      if (!matchFG)
        meFGEmulError.fill(getEcalDQMSetupObjects(), ttid);
    }
  }

  DEFINE_ECALDQM_WORKER(TrigPrimTask);
}  // namespace ecaldqm
