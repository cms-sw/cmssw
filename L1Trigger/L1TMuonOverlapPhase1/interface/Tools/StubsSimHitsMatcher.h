/*
 * StubsSimHitsMatcher.h
 *
 *  Created on: Jan 13, 2021
 *      Author: kbunkow
 */

#ifndef L1T_OmtfP1_TOOLS_STUBSSIMHITSMATCHER_H_
#define L1T_OmtfP1_TOOLS_STUBSSIMHITSMATCHER_H_

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "TH1I.h"
#include "TH2I.h"

class RPCGeometry;
class CSCGeometry;
class DTGeometry;

struct MuonGeometryTokens;

class StubsSimHitsMatcher {
public:
  StubsSimHitsMatcher(const edm::ParameterSet& edmCfg,
                      const OMTFConfiguration* omtfConfig,
                      const MuonGeometryTokens& muonGeometryTokens);
  virtual ~StubsSimHitsMatcher();

  void beginRun(edm::EventSetup const& eventSetup);

  void endJob();

  void match(const edm::Event& iEvent,
             const l1t::RegionalMuonCand* omtfCand,
             const AlgoMuonPtr& procMuon,
             std::ostringstream& ostr);  //Processor gbCandidate);

  class MatchedTrackInfo {
  public:
    union {
      struct {
        int32_t eventNum = 0;
        uint32_t trackId = 0;
      };
      uint64_t eventTrackNum;
    };

    mutable std::vector<unsigned int> matchedDigiCnt;

    MatchedTrackInfo(uint32_t eventNum, uint32_t trackId) : matchedDigiCnt(18, 0) {
      this->eventNum = eventNum;
      this->trackId = trackId;
      //eventTrackNum = this->eventNum << 32 | trackId;
    }

    bool operator<(const MatchedTrackInfo& b) const { return this->eventTrackNum < b.eventTrackNum; }

    bool operator>(const MatchedTrackInfo& b) const { return this->eventTrackNum > b.eventTrackNum; }

    bool operator==(const MatchedTrackInfo& b) const { return this->eventTrackNum == b.eventTrackNum; }
  };

private:
  const OMTFConfiguration* omtfConfig;

  edm::InputTag rpcSimHitsInputTag;
  edm::InputTag cscSimHitsInputTag;
  edm::InputTag dtSimHitsInputTag;

  edm::InputTag rpcDigiSimLinkInputTag;
  edm::InputTag cscStripDigiSimLinksInputTag;
  edm::InputTag dtDigiSimLinksInputTag;

  edm::InputTag trackingParticleTag;

  const MuonGeometryTokens& muonGeometryTokens;

  edm::ESWatcher<MuonGeometryRecord> muonGeometryRecordWatcher;

  // pointers to the current geometry records
  unsigned long long _geom_cache_id = 0;
  edm::ESHandle<RPCGeometry> _georpc;
  edm::ESHandle<CSCGeometry> _geocsc;
  edm::ESHandle<DTGeometry> _geodt;

  TH1I* allMatchedTracksPdgIds = nullptr;   //[pdgId] = tracksCnt
  TH1I* bestMatchedTracksPdgIds = nullptr;  //[pdgId] = tracksCnt

  TH2I* stubsInLayersCntByPdgId = nullptr;
  TH2I* firedLayersCntByPdgId = nullptr;
  TH2I* ptByPdgId = nullptr;

  TH2I* rhoByPdgId = nullptr;
};

#endif /* L1T_OmtfP1_TOOLS_STUBSSIMHITSMATCHER_H_ */
