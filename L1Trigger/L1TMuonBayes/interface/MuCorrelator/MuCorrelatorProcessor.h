/*
 * MuCorrelatorProcessor.h
 *
 *  Created on: Jan 18, 2019
 *      Author: kbunkow
 */

#ifndef MUCORRELATOR_MUCORRELATORPROCESSOR_H_
#define MUCORRELATOR_MUCORRELATORPROCESSOR_H_

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include "L1Trigger/L1TMuonBayes/interface/MuonStub.h"
#include "L1Trigger/L1TMuonBayes/interface/TrackingTriggerTrack.h"

#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/MuonStubsInput.h"
#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/MuCorrelatorConfig.h"
#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/AlgoTTMuon.h"
#include "L1Trigger/L1TMuonBayes/interface/MuCorrelator/PdfModule.h"
#include "L1Trigger/L1TMuonBayes/interface/MuTimingModule.h"

#include "boost/multi_array.hpp"



class StandaloneCandWithStubs {
public:
  l1t::RegionalMuonCand regionalCand;
  MuonStubsInput stubs;

  unsigned int index;
};


typedef std::vector<StandaloneCandWithStubs> StandaloneCandWithStubsVec;

class CorrelatorMuCandidate {
  //TODO
};

typedef std::vector<CorrelatorMuCandidate> CorrelatorMuCandidates;


class MuCorrelatorProcessor {
public:
  //default pdfModuleType is PdfModule
  MuCorrelatorProcessor(MuCorrelatorConfigPtr& config, std::string pdfModuleType = "");

  //takes the ownership of the pdfModule
  MuCorrelatorProcessor(MuCorrelatorConfigPtr& config, unique_ptr<IPdfModule> pdfModule);

  virtual ~MuCorrelatorProcessor();

  virtual CorrelatorMuCandidates run(int bx) {
    //TODO implement;
    CorrelatorMuCandidates muCandidates;
    return muCandidates;
  }

  virtual AlgoTTMuons processTracks(const MuonStubsInput& muonStubs, const TrackingTriggerTracks& ttTracks);

  virtual AlgoTTMuonPtr processTrack(const MuonStubsInput& muonStubs, const TrackingTriggerTrackPtr& ttTrack);
  virtual AlgoTTMuonPtr processTrackUsingRefStubs(const MuonStubsInput& muonStubs, const TrackingTriggerTrackPtr& ttTrack);

  virtual MuonStubsInput selectStubs(const MuonStubsInput& muonStubs, const TrackingTriggerTrackPtr& ttTrack);

  virtual MuonStubPtrs1D selectRefStubs(const MuonStubsInput& muonStubs, const TrackingTriggerTrackPtr& ttTrack);

  //adds the StubResult to the algoTTMuon
  virtual void processStubs(const MuonStubsInput& muonStubs, unsigned int layer, const TrackingTriggerTrackPtr& ttTrack, const MuonStubPtr refStub, AlgoTTMuonPtr algoTTMuon);

  virtual AlgoTTMuons ghostBust(AlgoTTMuons& algoTTMuons);

  static int ghostBust3(std::shared_ptr<AlgoTTMuon> first, std::shared_ptr<AlgoTTMuon> second);

  virtual AlgoTTMuons processTracks(const StandaloneCandWithStubsVec& candsWithStubs, const TrackingTriggerTracks& ttTracks);

  AlgoTTMuonPtr processTrack(const StandaloneCandWithStubsVec& candsWithStubs, const TrackingTriggerTrackPtr& ttTrack);

  ///initial selection of the standalone candidates compatibile with a given ttTrack
  ///return vector to allow for an option when ex. two close stand alone candidates are selected for a given ttTrack, then final one is selected based on stubs
  virtual StandaloneCandWithStubsVec selectCandsWithStubs(const StandaloneCandWithStubsVec& candsWithStubs, const TrackingTriggerTrackPtr& ttTrack);

  IPdfModule* getPdfModule() {
    return pdfModule.get();
  }

  virtual std::vector<l1t::RegionalMuonCand> getFinalCandidates(unsigned int iProcessor, l1t::tftype mtfType, AlgoTTMuons& algoTTMuons);

  virtual bool assignQuality(AlgoTTMuons& algoTTMuons);

  //takes the ownership
  void setMuTimingModule(unique_ptr<MuTimingModule>& muTimingModule) {
    this->muTimingModule = std::move(muTimingModule);
  }

  MuTimingModule* getMuTimingModule() {
    return muTimingModule.get();
  }

private:
  MuCorrelatorConfigPtr config;

  /**should return:
   * 0 if first kills second
   * 1 if second kills first
   * 2 otherwise (none is killed)
   */
  std::function<int (AlgoTTMuonPtr first, AlgoTTMuonPtr second)> ghostBustFunc;

  unique_ptr<IPdfModule> pdfModule;
  unique_ptr<MuTimingModule> muTimingModule;

  std::vector< std::pair<int, boost::dynamic_bitset<> > > lowQualityHitPatterns;
};

#endif /* INTERFACE_MUCORRELATOR_MUCORRELATORPROCESSOR_H_ */
