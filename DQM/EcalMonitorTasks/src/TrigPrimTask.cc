#include "../interface/TrigPrimTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"

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
  TrigPrimTask::beginRun(edm::Run const&, edm::EventSetup const& _es)
  {
    // Read-in Status records:
    // Status records stay constant over run so they are read-in only once here
    // but filled by LS in runOnRealTPs() because MEs are not yet booked at beginRun()
    _es.get<EcalTPGTowerStatusRcd>().get( TTStatusRcd );
    _es.get<EcalTPGStripStatusRcd>().get( StripStatusRcd );
  }

  void
  TrigPrimTask::beginEvent(edm::Event const& _evt, edm::EventSetup const&  _es)
  {
    using namespace std;

    towerReadouts_.clear();

    realTps_ = 0;

    //     HLTCaloBit_ = false;
    //     HLTMuonBit_ = false;

    int* pBin(std::upper_bound(bxBinEdges_, bxBinEdges_ + nBXBins + 1, _evt.bunchCrossing()));
    bxBin_ = static_cast<int>(pBin - bxBinEdges_) - 0.5;

    edm::ESHandle<EcalTPGTowerStatus> TTStatusRcd_;
    _es.get<EcalTPGTowerStatusRcd>().get(TTStatusRcd_);
    const EcalTPGTowerStatus * TTStatus=TTStatusRcd_.product();
    const EcalTPGTowerStatusMap &towerMap=TTStatus->getMap();

    edm::ESHandle<EcalTPGStripStatus> StripStatusRcd_;
    _es.get<EcalTPGStripStatusRcd>().get(StripStatusRcd_);
    const EcalTPGStripStatus * StripStatus=StripStatusRcd_.product();
    const EcalTPGStripStatusMap &stripMap=StripStatus->getMap();

    MESet& meTTMaskMap(MEs_.at("TTMaskMap"));

    for(EcalTPGTowerStatusMap::const_iterator ttItr(towerMap.begin()); ttItr != towerMap.end(); ++ttItr){

       if ((*ttItr).second > 0)
       {
         const EcalTrigTowerDetId  ttid((*ttItr).first);
         //if(ttid.subDet() == EcalBarrel)
            meTTMaskMap.fill(ttid,1);
       }//masked   
    }//loop on towers
  
    for(EcalTPGStripStatusMap::const_iterator stItr(stripMap.begin()); stItr != stripMap.end(); ++stItr){

       if ((*stItr).second > 0)
       {
         const EcalElectronicsId stid((*stItr).first);
         //if(stid.subdet() == EcalEndcap);
            meTTMaskMap.fill(stid,1);
       }//masked   
    }//loop on pseudo-strips
  
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
    MESet& meTTFlags4( MEs_.at("TTFlags4") );
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

      // Fill TT Flag MEs
      float ttF( tpItr->ttFlag() );
      meTTFlags.fill( ttid, ttF );
      // Monitor occupancy of TTF=4
      // which contains info about TT auto-masking
      if ( ttF == 4. )
        meTTFlags4.fill( ttid );

      if((interest == 1 || interest == 3) && towerReadouts_[ttid.rawId()] != getTrigTowerMap()->constituentsOf(ttid).size())
        meTTFMismatch.fill(ttid);
    }

    meOccVsBx.fill( EcalBarrel, bxBin_, nTP[0]);
    meOccVsBx.fill(-EcalEndcap, bxBin_, nTP[1]);
    meOccVsBx.fill( EcalEndcap, bxBin_, nTP[2]);

    // Set TT/Strip Masking status in Ecal3P view
    // Status Records are read-in at beginRun() but filled here
    // Requestied by ECAL Trigger in addition to TTMaskMap plots in SM view
    MESet& meTTMaskMapAll(MEs_.at("TTMaskMapAll"));

    // Fill from TT Status Rcd
    const EcalTPGTowerStatus *TTStatus( TTStatusRcd.product() );
    const EcalTPGTowerStatusMap &TTStatusMap( TTStatus->getMap() );
    for( EcalTPGTowerStatusMap::const_iterator ttItr(TTStatusMap.begin()); ttItr != TTStatusMap.end(); ++ttItr ){
      const EcalTrigTowerDetId ttid( ttItr->first );
      if ( ttItr->second > 0 )
        meTTMaskMapAll.setBinContent( ttid,1 ); // TT is masked
    } // TTs

    // Fill from Strip Status Rcd
    const EcalTPGStripStatus *StripStatus( StripStatusRcd.product() );
    const EcalTPGStripStatusMap &StripStatusMap( StripStatus->getMap() );
    for( EcalTPGStripStatusMap::const_iterator stItr(StripStatusMap.begin()); stItr != StripStatusMap.end(); ++stItr ){
      const EcalTriggerElectronicsId stid( stItr->first );
      // Since ME has kTriggerTower binning, convert to EcalTrigTowerDetId first
      // In principle, setBinContent() could be implemented for EcalTriggerElectronicsId class as well
      const EcalTrigTowerDetId ttid( getElectronicsMap()->getTrigTowerDetId(stid.tccId(), stid.ttId()) );
      if ( stItr->second > 0 )
        meTTMaskMapAll.setBinContent( ttid,1 ); // PseudoStrip is masked
    } // PseudoStrips

  } // TrigPrimTask::runOnRealTPs()

  void
  TrigPrimTask::runOnEmulTPs(EcalTrigPrimDigiCollection const& _tps)
  {
    MESet& meEtMaxEmul(MEs_.at("EtMaxEmul"));
    MESet& meEmulMaxIndex(MEs_.at("EmulMaxIndex"));
    MESet& meMatchedIndex(MEs_.at("MatchedIndex"));
    MESet& meEtEmulError(MEs_.at("EtEmulError"));
    MESet& meFGEmulError(MEs_.at("FGEmulError"));
    MESet& meRealvEmulEt(MEs_.at("RealvEmulEt"));

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

      // Loop over real TPs and look for an emulated TP index with matching Et:
      // If an Et match is found, return TP index correpsonding to BX of emulated TP where match was found
      // Standard TPG comparison: { TP index:matched BX } = { no emul:No Et match, 0:BX-2, 1:BX-1, 2:in-time, 3:BX+1, 4:BX+2 }
      EcalTrigPrimDigiCollection::const_iterator realItr(realTps_->find(ttid));
      if(realItr != realTps_->end()){

        int realEt(realItr->compressedEt());

        if(realEt > 0){

          int interest(realItr->ttFlag() & 0x3);
          if((interest == 1 || interest == 3) && towerReadouts_[ttid.rawId()] == getTrigTowerMap()->constituentsOf(ttid).size()){

            if(et != realEt) match = false;
            if(tpItr->fineGrain() != realItr->fineGrain()) matchFG = false;

	    // NOTE: matchedIndex comparison differs from Standard TPG comparison:
	    // { matchedIndex:TP index } = { 0:no emul, 1:BX-2, 2:BX-1, 3:in-time, 4:BX+1, 5:BX+2 }
	    std::vector<int> matchedIndex(0);
	    // iDigi only loops over explicit Et matches:
	    // { iDigi:TP index } = { 0:BX-2, 1:BX-1, 2:in-time, 3:BX+1, 4:BX+2 }
	    for(int iDigi(0); iDigi < 5; iDigi++){
	      if((*tpItr)[iDigi].compressedEt() == realEt) {
		// matchedIndex = iDigi + 1
		if (iDigi != 2) {
		  matchedIndex.push_back(iDigi + 1);
		}
		// If an in-time match is found, exit loop and clear out any other matches:
		// Ensures multiple matches are not returned (e.g. during saturation)
		else {
		  matchedIndex.clear();
		  matchedIndex.push_back(3); // Et match is to in-time emulated TP
		  break;
		}
	      } // Et match found
	    } // iDigi
	    if(!matchedIndex.size()) matchedIndex.push_back(0); // no Et match found => no emul

	    // Fill Real vs Emulated TP Et
	    meRealvEmulEt.fill( ttid,realEt,(*tpItr)[2].compressedEt() ); // iDigi=2:in-time BX

	    // Fill matchedIndex ME
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

