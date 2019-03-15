/*
 * OMTFProcessorTTMerger.cpp
 *
 *  Created on: Oct 7, 2017
 *      Author: kbunkow
 */

#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternWithStat.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFProcessorTTMerger.h>
#include <iostream>
#include <algorithm>
#include <strstream>

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

///////////////////////////////////////////////
///////////////////////////////////////////////
template <class GoldenPatternType>
OMTFProcessorTTMerger<GoldenPatternType>::OMTFProcessorTTMerger(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, const edm::EventSetup & evSetup, const L1TMuonOverlapParams* omtfPatterns):
OMTFProcessor<GoldenPatternType>(omtfConfig, edmCfg, evSetup, omtfPatterns),
  ghostBustFunc(ghostBust3)
{
  init(edmCfg);
};

template <class GoldenPatternType>
OMTFProcessorTTMerger<GoldenPatternType>::OMTFProcessorTTMerger(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, const edm::EventSetup& evSetup, const typename ProcessorBase<GoldenPatternType>::GoldenPatternVec& gps):
OMTFProcessor<GoldenPatternType>(omtfConfig, edmCfg, evSetup, gps),
  ghostBustFunc(ghostBust3)
{
  init(edmCfg);
};

template <class GoldenPatternType>
OMTFProcessorTTMerger<GoldenPatternType>::~OMTFProcessorTTMerger() {

}

template<class GoldenPatternType>
void OMTFProcessorTTMerger<GoldenPatternType>::init(const edm::ParameterSet& edmCfg) {
  if(edmCfg.exists("ttTracksSource") ) {
    std::string trackSrc = edmCfg.getParameter<std::string>("ttTracksSource");
    if(trackSrc == "SIM_TRACKS")
      ttTracksSource = SIM_TRACKS;
    else if(trackSrc == "L1_TRACKER") {
      ttTracksSource = L1_TRACKER;
      if(edmCfg.exists("l1Tk_nPar") ) {
        l1Tk_nPar = edmCfg.getParameter<int>("l1Tk_nPar");
      }
    }
  }

  if(edmCfg.exists("refLayerMustBeValid") ) {
    refLayerMustBeValid = edmCfg.getParameter<bool>("refLayerMustBeValid");
  }
  //modifyPatterns(); //TODO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<remove<<<<<<<<<<<<<<<<<<<<<<<
  GoldenPatternResult::setFinalizeFunction(1); //TODO movo to config!!!!
  //edm::LogImportant("OMTFProcessorTTMerger") << "ttTracksSource "<<ttTracksSource << std::endl;
}

template <class GoldenPatternType>
std::vector<l1t::RegionalMuonCand> OMTFProcessorTTMerger<GoldenPatternType>::getFinalcandidates(unsigned int iProcessor, l1t::tftype mtfType, AlgoMuons& algoCands) {
//TODO optimize for the TTTracks
  std::vector<l1t::RegionalMuonCand> result;

  for(auto myCand: algoCands){
    l1t::RegionalMuonCand candidate;
    TTAlgoMuon* ttAlgoMuon = static_cast<TTAlgoMuon*>(myCand.get());
    candidate.setHwPt(ttAlgoMuon->getTtTrack().getPtHw());
    candidate.setHwEta(ttAlgoMuon->getTtTrack().getEtaHw());
    candidate.setHwPhi(this->myOmtfConfig->phiToGlobalHwPhi(ttAlgoMuon->getTtTrack().getPhi()));
/*    int phiValue = myCand->getPhi();
    if(phiValue>= int(this->myOmtfConfig->nPhiBins()) )
      phiValue -= this->myOmtfConfig->nPhiBins();
    ///conversion factor from OMTF to uGMT scale is  5400/576 i.e. phiValue/=9.375;
    phiValue = floor(phiValue*437./pow(2,12));    // ie. use as in hw: 9.3729977
    candidate.setHwPhi(phiValue);*/

    candidate.setHwSign(ttAlgoMuon->getTtTrack().getCharge()<0 ? 1:0  );
    candidate.setHwSignValid(1);

    unsigned int quality = checkHitPatternValidity(myCand->getFiredLayerBits()) ? 0 | (1 << 2) | (1 << 3) //=12
                                                                     : 0 | (1 << 2);                      //=4
    if (    abs(myCand->getEtaHw()) == 115
        && (    static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001110000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000001110000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000110000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001100000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001010000000").to_ulong()
           )
       ) quality =4;
    if( this->myOmtfConfig->fwVersion() >= 5 ) {
      if (    static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000110000").to_ulong()
         ) quality = 1;
    }
//  if (abs(myCand->getEta()) == 121) quality = 4;
    if (abs(myCand->getEtaHw()) == 121) quality = 0; // changed on request from HI

    candidate.setHwQual (quality);

    std::map<int, int> trackAddr;
    trackAddr[0] = myCand->getFiredLayerBits();
    trackAddr[1] = myCand->getRefLayer();
    trackAddr[2] = myCand->getDisc();
    trackAddr[3] = ttAlgoMuon->getTtTrack().getIndex();
    candidate.setTrackAddress(trackAddr);
    candidate.setTFIdentifiers(iProcessor,mtfType);

    if (candidate.hwPt() > 0 && quality > 0)  //rejecting here the candidates with eta 121, i.e. > 1.31
      result.push_back(candidate);
  }
  return result;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
template<class GoldenPatternType>
bool OMTFProcessorTTMerger<GoldenPatternType>::checkHitPatternValidity(unsigned int hits) {
  ///FIXME: read the list from configuration so this can be controlled at runtime.
  std::vector<unsigned int> badPatterns = {99840, 34304, 3075, 36928, 12300, 98816, 98944, 33408, 66688, 66176, 7171, 20528, 33856, 35840, 4156, 34880};
//TODO optimize for the TTTracks
/*
99840 01100001 1000 000000
34304 00100001 1000 000000
 3075 00000011 0000 000011
36928 00100100 0001 000000
12300 00001100 0000 001100
98816 01100000 1000 000000
98944 01100000 1010 000000
33408 00100000 1010 000000
66688 01000001 0010 000000
66176 01000000 1010 000000
 7171 00000111 0000 000011
20528 00010100 0000 110000
33856 00100001 0001 000000
35840 00100011 0000 000000
 4156 00000100 0000 111100
34880 00100010 0001 000000*/

  for(auto aHitPattern: badPatterns){
    if(hits==aHitPattern) return false;
  }

  return true;
}


template<class GoldenPatternType>
void OMTFProcessorTTMerger<GoldenPatternType>::laodTTTracks(const edm::Event &event, const edm::ParameterSet& edmCfg) {
  ttTracks.clear();
  //cout<<__FUNCTION__<<":"<<__LINE__<<endl;

  if(ttTracksSource == SIM_TRACKS) {
    edm::Handle<edm::SimTrackContainer> simTks;
    event.getByLabel(edmCfg.getParameter<edm::InputTag>("g4SimTrackSrc"), simTks);

    unsigned int index = 0;
    for (std::vector<SimTrack>::const_iterator it=simTks->begin(); it< simTks->end(); it++) {
      const SimTrack& simMuon = *it;
      index++;
      if ( !(simMuon.type() == 13 || simMuon.type() == -13) )
        continue;

      ttTracks.emplace_back(simMuon, index-1);
    }
  }
  else if(ttTracksSource == L1_TRACKER) {
	  edm::Handle< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > tTTrackHandle;
	  event.getByLabel(edmCfg.getParameter<edm::InputTag>("L1TrackInputTag"), tTTrackHandle);
    //cout << __FUNCTION__<<":"<<__LINE__ << " LTTTrackHandle->size() "<<tTTrackHandle->size() << endl;

	  unsigned int index = 0;
    for (auto iterL1Track = tTTrackHandle->begin(); iterL1Track != tTTrackHandle->end(); iterL1Track++ ) {
      ttTracks.emplace_back(*iterL1Track, index, l1Tk_nPar);
      index++;
      //cout<<__FUNCTION__<<":"<<__LINE__<<" "<<*iterL1Track<<" Momentum "<<iterL1Track->getMomentum(l1Tk_nPar)<<" RInv "<<iterL1Track->getRInv(l1Tk_nPar)<<endl;
    }
  }
  //cout<<__FUNCTION__<<":"<<__LINE__<<" ttTracks.size() "<<ttTracks.size()<<endl;
}

template<class GoldenPatternType>
TTTracks OMTFProcessorTTMerger<GoldenPatternType>::getTTTrackForProcessor(unsigned int iProcessor, l1t::tftype mtfType, const TTTracks& eventTTTRacks) {
  TTTracks procTTTRacks;

  double phiUnit = 2*M_PI/this->myOmtfConfig->nPhiBins();
  int marginLeft =  - 2*M_PI * 70. / 360. / phiUnit; //70 deg margin in phiUnits - TODO move to config, find optimal value
  int marginRight =   2*M_PI / this->myOmtfConfig->nProcessors() / phiUnit - marginLeft; //10 deg margin in phiUnits - TODO move to config, find optimal value

  //TODO move to config
  double etaCutFrom = 0.82;
  double etaCutTo = 1.24;
  if(mtfType == l1t::tftype::omtf_neg) {
    etaCutFrom = -1.24;
    etaCutTo = -0.82;
  }

  for(auto& ttTrack : eventTTTRacks) {
    TrackingTriggerTrack acceptedTrak = ttTrack;
    // adjust [0,2pi] and [-pi,pi] to get deltaPhi difference properly
    double phi = acceptedTrak.getPhi();

    // local angle in CSC halfStrip phi units
    int phiHw = this->myOmtfConfig->getProcScalePhi(iProcessor, phi);

    if(acceptedTrak.getEta()  > etaCutFrom && acceptedTrak.getEta() < etaCutTo ) {
      if(phiHw > marginLeft && phiHw < marginRight) {
        acceptedTrak.setPhiHw(phiHw);
        acceptedTrak.setEtaHw(this->myOmtfConfig->etaToHwEta( acceptedTrak.getEta() ));
        acceptedTrak.setPtHw(this->myOmtfConfig->ptGevToHw(acceptedTrak.getPt() ) );

        procTTTRacks.push_back(acceptedTrak);
      }
    }
  }

  return procTTTRacks;
}


///////////////////////////////////////////////
///////////////////////////////////////////////
int printCtn = 0; //TODO remove, it is just for debug
template<class GoldenPatternType>
const void OMTFProcessorTTMerger<GoldenPatternType>::processInput(unsigned int iProcessor, l1t::tftype mtfType,
    const OMTFinput & aInput, const TTTracks& ttTracks)
{
  /*
  unsigned int procIndx = this->myOmtfConfig->getProcIndx(iProcessor, mtfType);
  for(auto& itGP: this->theGPs) {
    for(auto& result : itGP->getResults()[procIndx]) {
      result.reset();
    }
    the results kept by the goldenPatterns are useless here, because can altered because for each event many ttTracks have to be analysed
  }*/

  //////////////////////////////////////
  //////////////////////////////////////
  std::bitset<128> refHitsBits = aInput.getRefHits(iProcessor);
  if(refHitsBits.none())
    return; // myResults;

	ttMuons.clear();
  for(auto& ttTrack : ttTracks) {
    unsigned int testedRefHitNum = 0;

    GoldenPatternResult* bestResult = nullptr;
    unsigned int bestResultIRefHit = 0;
    unsigned int patNum = this->myOmtfConfig->getPatternNum(ttTrack.getPt(), ttTrack.getCharge()); //TODO use hardware pt scale
    auto& itGP = this->theGPs[patNum];

    //cout<<__FUNCTION__<<":"<<__LINE__<<" iProcessor "<<iProcessor<<" theGPs.size() "<<this->theGPs.size()<<" selected pattern "<<itGP->key()<<endl;

    std::vector<std::shared_ptr<GoldenPatternResult> > gpResults;
    for(unsigned int iRefHit = 0; iRefHit < this->myOmtfConfig->nRefHits(); ++iRefHit) { //loop over all possible refHits, i.e. 128
      if(!refHitsBits[iRefHit])
        continue;

      if(testedRefHitNum == this->myOmtfConfig->nTestRefHits() -1)
        break;

      testedRefHitNum++;

      const RefHitDef& aRefHitDef = this->myOmtfConfig->getRefHitsDefs()[iProcessor][iRefHit];

      unsigned int refLayerLogicNum = this->myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer];

      const MuonStubPtr refStub = aInput.getMuonStub(refLayerLogicNum, aRefHitDef.iInput);
      int phiRef = refStub->phiHw;
      int etaRef = refStub->etaHw;

      unsigned int iRegion = aRefHitDef.iRegion;

      bool refLayerValid = false;

      gpResults.emplace_back(make_shared<GoldenPatternResult>(this->myOmtfConfig) );
      GoldenPatternResult* gpResult = gpResults.back().get();

      for(unsigned int iLayer=0; iLayer < this->myOmtfConfig->nLayers(); ++iLayer) {
        MuonStubPtr newRefStubPtr = refStub;
        if(iLayer == refLayerLogicNum) { //we include the ttTrack phi only for the reference hit
          MuonStub newRefStub;
          newRefStub.phiHw = ttTrack.getPhiHw();
          newRefStub.type = MuonStub::TTTRACK_REF;
          newRefStubPtr = std::make_shared<MuonStub>(newRefStub);
        }

        MuonStubPtrs1D restrictedLayerStubs = this->restrictInput(iProcessor, iRegion, iLayer, aInput);

        StubResult stubResult = itGP->process1Layer1RefLayer(aRefHitDef.iRefLayer, iLayer,
            restrictedLayerStubs,
            newRefStubPtr);

				gpResult->setStubResult(iLayer, stubResult);

        //if for the ref layer the result is not valid (i.e. the dist_phi is outside of range or the pdfVal = 0, we drop this refHit
        if(iLayer == refLayerLogicNum) {
          refLayerValid = stubResult.getValid();
        }
      }

      //TODO set phi and eta from ttTrack???
      int phiRefSt2 = itGP->propagateRefPhi(phiRef, etaRef, aRefHitDef.iRefLayer); //todo check which phi and eta - from ttTrack or frmo refHit should be here

      gpResult->set(aRefHitDef.iRefLayer, phiRefSt2, etaRef, phiRef);

      gpResult->finalise(); //this sets result to valid, without any conditions

      if(!refLayerMustBeValid)
        refLayerValid = true;
      if( !refLayerValid || gpResult->getFiredLayerCnt() < 2) { //TODO optimize the cut on the FiredLayerCnt
        gpResult->setValid(false);
      }

     /* if(printCtn++ < 200)
        cout<<__FUNCTION__<<":"<<__LINE__<<" iProcessor "<<iProcessor<<" ttTrack Pt "<<ttTrack.getPt()<<" charge "<<ttTrack.getCharge()<<" refLayerLogicNum "<<refLayerLogicNum
          <<" iRefHit "<<iRefHit<<"\n"<<*gpResult<<endl;*/

      if(gpResult->isValid())
      {
        if( bestResult == nullptr ||
            (gpResult->getFiredLayerCnt() >  bestResult->getFiredLayerCnt() ) ||
            (gpResult->getFiredLayerCnt() == bestResult->getFiredLayerCnt() && gpResult->getPdfSum() > bestResult->getPdfSum() )
        )
        {
          bestResult = gpResult;
          bestResultIRefHit =  testedRefHitNum-1;//-1 Because it was ++ already
        }
      }
    }
    if(bestResult != nullptr) {
      LogTrace("omtfEventPrintout")<<">>>>>>>>>>>>>>>>>>>>>bestResult: \niProcessor "<<iProcessor
          <<" ttTrack Pt "<<ttTrack.getPt()<<" charge "<<ttTrack.getCharge()
          <<" eta "<<ttTrack.getEta()<<" phi "<<ttTrack.getPhi()<<" index  "<<ttTrack.getIndex()
          <<" bestResultIRefHit "<<bestResultIRefHit<<"\n"<<*bestResult<<endl;

      ttMuons.emplace_back(std::make_shared<TTAlgoMuon>(ttTrack, *bestResult, itGP.get(), gpResults, bestResultIRefHit ) );
      //N.B. gpResults is empty after that
    }
  }

  //ghostbusting
  for(auto& ttMuon : ttMuons) {
    for(auto& selected : selectedTTMuons) {
      int ghostBustResult = ghostBustFunc(selected, ttMuon);
      if(ghostBustResult == 0) { //selected kills ttMuon
        ttMuon->kill();
      }
      else if(ghostBustResult == 1) {//ttMuon kills selected
        selected->kill();
      }
      else {
        //ttMuon neither kills nor is killed
      }
    }

    selectedTTMuons.erase(std::remove_if(selectedTTMuons.begin(), selectedTTMuons.end(),
                                  [](shared_ptr<AlgoMuon>& x){return x->isKilled();} ), selectedTTMuons.end());

    if( !ttMuon->isKilled()) {
      selectedTTMuons.push_back(ttMuon);
    }
  }

/*  std::ostringstream myStr;
  myStr<<"iProcessor: "<<iProcessor<<std::endl;
  myStr<<"Input: ------------"<<std::endl;
  myStr<<aInput<<std::endl;
  edm::LogInfo("OMTF processor")<<myStr.str();*/

  return;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/**should return:
 * 0 if first kills second
 * 1 if second kills first
 * 2 otherwise (none is killed)
 */
template<class GoldenPatternType>
int OMTFProcessorTTMerger<GoldenPatternType>::ghostBust1(std::shared_ptr<AlgoMuon> first, std::shared_ptr<AlgoMuon> second) {
  //good ghost bust function looks on the hits indexes in each candidate and check how many hits are common, kill one of them if more then e.g. 1
  int commonHits = 0;
  for(unsigned int iLayer=0; iLayer < first->getGpResult().getStubResults().size(); ++iLayer) {
    if( first->getGpResult().isLayerFired(iLayer) &&
       second->getGpResult().isLayerFired(iLayer) &&
       first->getGpResult().getStubResults()[iLayer].getMuonStub() == second->getGpResult().getStubResults()[iLayer].getMuonStub() ) { //TODO this is hit phi, not the hit index, but in principle the result should be the same
      commonHits++;
    }
  }

  if(commonHits >= 1) { //probably to sharp...
    if(      first->getGpResult().getFiredLayerCnt() >  second->getGpResult().getFiredLayerCnt()  ) {
      return 0;
    }
    else if( first->getGpResult().getFiredLayerCnt() <  second->getGpResult().getFiredLayerCnt()  ) {
      return 1;
    }
    else { //first->getGpResult().getFiredLayerCnt() == second->getGpResult().getFiredLayerCnt()
      if(      first->getGpResult().getPdfSum() > second->getGpResult().getPdfSum() ) {
        return 0;
      }
      else if( first->getGpResult().getPdfSum() < second->getGpResult().getPdfSum() ) {
        return 1;
      }
      else {// first->getGpResult().getPdfSum()== second->getGpResult().getPdfSum()
        if( first->getPt() > second->getPt() )
          return 0;
        else
          return 1;
      }
    }
  }
  return 2;
}

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/**should return:
 * 0 if first kills second
 * 1 if second kills first
 * 2 otherwise (none is killed)
 */
template<class GoldenPatternType>
int OMTFProcessorTTMerger<GoldenPatternType>::ghostBust2(std::shared_ptr<AlgoMuon> first, std::shared_ptr<AlgoMuon> second) {
  //cout<<__FUNCTION__<<":"<<__LINE__<<endl;
  //good ghost bust function looks on the hits indexes in each candidate and check how many hits are common, kill one of them if more then e.g. 1
  int commonHits = 0;
  for(unsigned int iLayer=0; iLayer < first->getGpResult().getStubResults().size(); ++iLayer) {
    if( first->getGpResult().isLayerFired(iLayer) &&
       second->getGpResult().isLayerFired(iLayer) &&
       first->getGpResult().getStubResults()[iLayer].getMuonStub() == second->getGpResult().getStubResults()[iLayer].getMuonStub() ) { //TODO this is hit phi, not the hit index, but in principle the result should be the same
      commonHits++;
    }
  }

  if(commonHits >= 1) { //probably to sharp...
    if( abs( (int)first->getGpResult().getFiredLayerCnt() - (int)second->getGpResult().getFiredLayerCnt()) <= 1 ) {
      if(      first->getGpResult().getPdfSum() > second->getGpResult().getPdfSum() ) {
        return 0;
      }
      else if( first->getGpResult().getPdfSum() < second->getGpResult().getPdfSum() ) {
        return 1;
      }
      else {// first->getGpResult().getPdfSum()== second->getGpResult().getPdfSum()
        if( first->getPt() > second->getPt() )
          return 0;
        else
          return 1;
      }
    }
    else if(      first->getGpResult().getFiredLayerCnt() >  second->getGpResult().getFiredLayerCnt()  ) {
      return 0;
    }
    else if( first->getGpResult().getFiredLayerCnt() <  second->getGpResult().getFiredLayerCnt()  ) {
      return 1;
    }
  }
  return 2;
}

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
/**should return:
 * 0 if first kills second
 * 1 if second kills first
 * 2 otherwise (none is killed)
 */
template<class GoldenPatternType>
int OMTFProcessorTTMerger<GoldenPatternType>::ghostBust3(std::shared_ptr<AlgoMuon> first, std::shared_ptr<AlgoMuon> second) {
  //cout<<__FUNCTION__<<":"<<__LINE__<<endl;
  //good ghost bust function looks on the hits indexes in each candidate and check how many hits are common, kill one of them if more then e.g. 1
  int commonHits = 0;
  for(unsigned int iLayer=0; iLayer < first->getGpResult().getStubResults().size(); ++iLayer) {
    if( first->getGpResult().isLayerFired(iLayer) &&
       second->getGpResult().isLayerFired(iLayer) &&
       first->getGpResult().getStubResults()[iLayer].getMuonStub() == second->getGpResult().getStubResults()[iLayer].getMuonStub()) { //TODO this is hit phi, not the hit index, but in principle the result should be the same
      commonHits++;
    }
  }

  if(commonHits >= 1) { //probably to sharp...
    if(      first->getGpResult().getPdfSum() > second->getGpResult().getPdfSum() ) {
      return 0;
    }
    else if( first->getGpResult().getPdfSum() < second->getGpResult().getPdfSum() ) {
      return 1;
    }
    else {// first->getGpResult().getPdfSum()== second->getGpResult().getPdfSum()
      if( first->getPt() > second->getPt() )
        return 0;
      else
        return 1;
    }
  }
  return 2;
}

template<class GoldenPatternType>
std::vector<l1t::RegionalMuonCand> OMTFProcessorTTMerger<GoldenPatternType>::
run(unsigned int iProcessor, l1t::tftype mtfType, int bx, std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  //boost::timer::auto_cpu_timer t("%ws wall, %us user in getProcessorCandidates\n");
  this->inputMaker.setFlag(0);
  OMTFinput input = this->inputMaker.buildInputForProcessor(
      this->dtPhDigis.product(),
      this->dtThDigis.product(),
      this->cscDigis.product(),
      this->rpcDigis.product(),
                iProcessor, mtfType, bx);
  int flag = this->inputMaker.getFlag();

  TTTracks procTTTracks = getTTTrackForProcessor(iProcessor, mtfType, ttTracks);

  //cout<<__FUNCTION__<<":"<<__LINE__<<" iProcessor "<<iProcessor<<" procTTTracks.size() "<<procTTTracks.size()<<endl;

  //cout<<"buildInputForProce "; t.report();
  selectedTTMuons.clear();

  processInput(iProcessor, mtfType, input, procTTTracks);

  //cout<<__FUNCTION__<<":"<<__LINE__<<" iProcessor "<<iProcessor<<" selectedTTMuons.size() "<<selectedTTMuons.size()<<endl;

  //cout<<"processInput       "; t.report();

  std::vector<l1t::RegionalMuonCand> candMuons = this->getFinalcandidates(iProcessor, mtfType, selectedTTMuons);
  //cout<<"getFinalcandidates "; t.report();
  //fill outgoing collection
  for (auto & candMuon :  candMuons) {
     candMuon.setHwQual( candMuon.hwQual() | flag);         //FIXME temporary debug fix
  }

  //dump to XML
  //if(bx==0) writeResultToXML(iProcessor, mtfType,  input, algoCandidates, candMuons); //TODO handle bx
  //if(bx==0)
  for(auto& obs : observers) {
    obs->observeProcesorEmulation(iProcessor, mtfType,  input, ttMuons, selectedTTMuons, candMuons);
  }

  return candMuons;
}


template<class GoldenPatternType>
void OMTFProcessorTTMerger<GoldenPatternType>::modifyPatterns() {
  for(auto& gp : this->theGPs) {
    if(gp->key().thePt == 0)
      continue;
    for(unsigned int iLayer = 0; iLayer < gp->getPdf().size(); ++iLayer) {
      for(unsigned int iRefLayer = 0; iRefLayer < gp->getPdf()[iLayer].size(); ++iRefLayer) {
        //unsigned int refLayerLogicNum = omtfConfig->getRefToLogicNumber()[iRefLayer];
        //if(refLayerLogicNum == iLayer)
        {
          int binsToSet = 20; //TODO
          unsigned int pdfSize = gp->getPdf()[iLayer][iRefLayer].size();

          unsigned int iBinMax = 0;
          double maxVal = 0;
          for(unsigned int iBin = 0; iBin < pdfSize; iBin++) {
            if(maxVal < gp->getPdf()[iLayer][iRefLayer][iBin] ) {
              maxVal = gp->getPdf()[iLayer][iRefLayer][iBin];
              iBinMax = iBin;
            }
          }

          for(unsigned int iBin = iBinMax; iBin < pdfSize; iBin++) {
            if( gp->getPdf()[iLayer][iRefLayer][iBin] == 0) {
              gp->setPdfValue(1, iLayer, iRefLayer, iBin);
              binsToSet--;
              if(binsToSet == 0)
                break;
            }
          }

          binsToSet = 20; //TODO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,
          for(int iBin = iBinMax; iBin >= 0; iBin--) {
            if( gp->getPdf()[iLayer][iRefLayer][iBin] == 0) {
              gp->setPdfValue(1, iLayer, iRefLayer, iBin);
              binsToSet--;
              if(binsToSet == 0)
                break;
            }
          }
        }
      }
    }
  }
}


template class OMTFProcessorTTMerger<GoldenPattern>;
template class OMTFProcessorTTMerger<GoldenPatternWithStat>;
template class OMTFProcessorTTMerger<GoldenPatternWithThresh>;
