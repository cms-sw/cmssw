#include "HLTCSCRing2or3Filter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

HLTCSCRing2or3Filter::HLTCSCRing2or3Filter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      m_input(iConfig.getParameter<edm::InputTag>("input")),
      m_minHits(iConfig.getParameter<unsigned int>("minHits")),
      m_xWindow(iConfig.getParameter<double>("xWindow")),
      m_yWindow(iConfig.getParameter<double>("yWindow")) {
  cscrechitsToken = consumes<CSCRecHit2DCollection>(m_input);
}

HLTCSCRing2or3Filter::~HLTCSCRing2or3Filter() = default;

void HLTCSCRing2or3Filter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("input", edm::InputTag("hltCsc2DRecHits"));
  desc.add<unsigned int>("minHits", 4);
  desc.add<double>("xWindow", 2.);
  desc.add<double>("yWindow", 2.);
  descriptions.add("hltCSCRing2or3Filter", desc);
}

bool HLTCSCRing2or3Filter::hltFilter(edm::Event& iEvent,
                                     const edm::EventSetup& iSetup,
                                     trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  edm::Handle<CSCRecHit2DCollection> hits;
  iEvent.getByToken(cscrechitsToken, hits);

  edm::ESHandle<CSCGeometry> cscGeometry;
  bool got_cscGeometry = false;

  std::map<int, std::vector<const CSCRecHit2D*> > chamber_tohit;

  for (auto const& hit : *hits) {
    CSCDetId id(hit.geographicalId());
    int chamber_id = CSCDetId(id.endcap(), id.station(), id.ring(), id.chamber(), 0).rawId();

    if (id.ring() == 2 || id.ring() == 3) {
      std::map<int, std::vector<const CSCRecHit2D*> >::const_iterator chamber_iter = chamber_tohit.find(chamber_id);
      if (chamber_iter == chamber_tohit.end()) {
        std::vector<const CSCRecHit2D*> newlist;
        newlist.push_back(&hit);
      }
      chamber_tohit[chamber_id].push_back(&hit);
    }  // end if this ring is selected
  }    // end loop over hits

  unsigned int minHitsAlmostSquared = (m_minHits - 1) * (m_minHits - 2);
  for (std::map<int, std::vector<const CSCRecHit2D*> >::const_iterator chamber_iter = chamber_tohit.begin();
       chamber_iter != chamber_tohit.end();
       ++chamber_iter) {
    if (chamber_iter->second.size() >= m_minHits) {
      if (!got_cscGeometry) {
        iSetup.get<MuonGeometryRecord>().get(cscGeometry);
        got_cscGeometry = true;
      }

      unsigned int pairs_in_window = 0;
      for (auto hit1 = chamber_iter->second.begin(); hit1 != chamber_iter->second.end(); ++hit1) {
        for (auto hit2 = chamber_iter->second.begin(); hit2 != hit1; ++hit2) {
          GlobalPoint pos1 =
              cscGeometry->idToDet((*hit1)->geographicalId())->surface().toGlobal((*hit1)->localPosition());
          GlobalPoint pos2 =
              cscGeometry->idToDet((*hit2)->geographicalId())->surface().toGlobal((*hit2)->localPosition());

          if (fabs(pos1.x() - pos2.x()) < m_xWindow && fabs(pos1.y() - pos2.y()) < m_yWindow)
            pairs_in_window++;

          if (pairs_in_window >= minHitsAlmostSquared)
            return true;
        }  // end loop over hits
      }    // end loop over hits

    }  // end if chamber has enough hits
  }    // end loop over chambers

  return false;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTCSCRing2or3Filter);
