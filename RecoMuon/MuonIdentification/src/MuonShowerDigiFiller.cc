//
// Package:    MuonShowerDigiFiller
// Class:      MuonShowerDigiFiller
//
/**\class MuonShowerDigiFiller MuonShowerDigiFiller.cc RecoMuon/MuonIdentification/src/MuonShowerDigiFiller.cc

 Description: Class filling shower information using DT and CSC digis 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Carlo Battilana, INFN BO
//         Created:  Sat Mar 23 14:36:22 CET 2019
//
//

// system include files

// user include files
#include "RecoMuon/MuonIdentification/interface/MuonShowerDigiFiller.h"

//
// constructors and destructor
//

MuonShowerDigiFiller::MuonShowerDigiFiller(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : m_digiMaxDistanceX(iConfig.getParameter<double>("digiMaxDistanceX")),
      m_dtDigisToken(iC.consumes<DTDigiCollection>(iConfig.getParameter<edm::InputTag>("dtDigiCollectionLabel"))),
      m_cscDigisToken(
          iC.consumes<CSCStripDigiCollection>(iConfig.getParameter<edm::InputTag>("cscDigiCollectionLabel"))),
      m_dtGeometryToken(iC.esConsumes<edm::Transition::BeginRun>()),
      m_cscGeometryToken(iC.esConsumes<edm::Transition::BeginRun>()) {}

//
// member functions
//

void MuonShowerDigiFiller::getES(const edm::EventSetup& iSetup) {
  m_dtGeometry = iSetup.getHandle(m_dtGeometryToken);
  m_cscGeometry = iSetup.getHandle(m_cscGeometryToken);
}

void MuonShowerDigiFiller::getDigis(edm::Event& iEvent) {
  iEvent.getByToken(m_dtDigisToken, m_dtDigis);
  iEvent.getByToken(m_cscDigisToken, m_cscDigis);
}

void MuonShowerDigiFiller::fill(reco::MuonChamberMatch& muChMatch) const {
  int nDigisInRange = 0;

  // DT chamber
  if (muChMatch.detector() == MuonSubdetId::DT) {
    double xTrack = muChMatch.x;

    for (int sl = 1; sl <= DTChamberId::maxSuperLayerId; sl += 2) {
      for (int layer = 1; layer <= DTChamberId::maxLayerId; ++layer) {
        const DTLayerId layerId(DTChamberId(muChMatch.id.rawId()), sl, layer);

        auto range = m_dtDigis->get(layerId);

        for (auto digiIt = range.first; digiIt != range.second; ++digiIt) {
          const auto topo = m_dtGeometry->layer(layerId)->specificTopology();

          double xWire = topo.wirePosition((*digiIt).wire());
          double dX = std::abs(xWire - xTrack);

          if (dX < m_digiMaxDistanceX)
            nDigisInRange++;
        }
      }
    }
  }

  else if (muChMatch.detector() == MuonSubdetId::CSC) {
    double xTrack = muChMatch.x;
    double yTrack = muChMatch.y;

    for (int iLayer = 1; iLayer <= CSCDetId::maxLayerId(); ++iLayer) {
      const CSCDetId chId(muChMatch.id.rawId());
      const CSCDetId layerId(chId.endcap(), chId.station(), chId.ring(), chId.chamber(), iLayer);

      auto range = m_cscDigis->get(layerId);

      for (auto digiIt = range.first; digiIt != range.second; ++digiIt) {
        std::vector<int> adcVals = digiIt->getADCCounts();
        bool hasFired = false;
        float pedestal = 0.5 * (float)(adcVals[0] + adcVals[1]);
        float threshold = 13.3;
        float diff = 0.;
        for (const auto& adcVal : adcVals) {
          diff = (float)adcVal - pedestal;
          if (diff > threshold) {
            hasFired = true;
            break;
          }
        }

        if (!hasFired)
          continue;

        const CSCLayerGeometry* layerGeom = m_cscGeometry->layer(layerId)->geometry();

        Float_t xStrip = layerGeom->xOfStrip(digiIt->getStrip(), yTrack);
        float dX = std::abs(xStrip - xTrack);

        if (dX < m_digiMaxDistanceX)
          nDigisInRange++;
      }
    }
  }

  muChMatch.nDigisInRange = nDigisInRange;
}

void MuonShowerDigiFiller::fillDefault(reco::MuonChamberMatch& muChMatch) const { muChMatch.nDigisInRange = 0; }
