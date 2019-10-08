// -*- C++ -*-
//
// Package:    RecoBTag/PixelCluster
// Class:      PixelClusterTagInfoProducer
//
/**\class PixelClusterTagInfoProducer PixelCluster RecoBTag/PixelCluster/plugins/PixelClusterTagInfoProducer.cc

 Description: Produces a collection of PixelClusterTagInfo objects,
  that contain the pixel cluster hit multiplicity in each pixel layer or disk
  in a narrow cone around the jet axis.

 Implementation:
     If the event does not fulfill minimum conditions (at least one jet above threshold,
     and a valid primary vertex) and empty collection is filled. Otherwise, a loop over
     the pixel cluster collection fills a vector of reco::PixelClusterProperties that
     contains the geometrical position and the charge of the cluster above threshold.
     A second loop on jets performs the dR association, and fills the TagInfo collection.
*/
//
// Original Author:  Alberto Zucchetta, Manuel Sommerhalder (UniZ) [zucchett]
//         Created:  Wed, 03 Jul 2019 12:37:30 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// TagInfo
#include "DataFormats/BTauReco/interface/PixelClusterTagInfo.h"

// For vertices
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// For jet
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

// For pixel clusters and topology
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class PixelClusterTagInfoProducer : public edm::global::EDProducer<> {
public:
  explicit PixelClusterTagInfoProducer(const edm::ParameterSet&);
  ~PixelClusterTagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::ParameterSet iConfig;

  const edm::EDGetTokenT<edm::View<reco::Jet> > m_jets;
  const edm::EDGetTokenT<reco::VertexCollection> m_vertices;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > m_pixelhit;
  const bool m_isPhase1;
  const bool m_addFPIX;
  const int m_minADC;
  const float m_minJetPt;
  const float m_maxJetEta;
  const float m_hadronMass;
  const unsigned int m_nLayers;
};

PixelClusterTagInfoProducer::PixelClusterTagInfoProducer(const edm::ParameterSet& iConfig)
    : m_jets(consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("jets"))),
      m_vertices(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      m_pixelhit(consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelhit"))),
      m_isPhase1(iConfig.getParameter<bool>("isPhase1")),
      m_addFPIX(iConfig.getParameter<bool>("addForward")),
      m_minADC(iConfig.getParameter<int>("minAdcCount")),
      m_minJetPt(iConfig.getParameter<double>("minJetPtCut")),
      m_maxJetEta(iConfig.getParameter<double>("maxJetEtaCut")),
      m_hadronMass(iConfig.getParameter<double>("hadronMass")),
      m_nLayers(m_isPhase1 ? 4 : 3) {
  produces<reco::PixelClusterTagInfoCollection>();
}

PixelClusterTagInfoProducer::~PixelClusterTagInfoProducer() {}

// ------------ method called to produce the data  ------------
void PixelClusterTagInfoProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Declare produced collection
  auto collectionTagInfo = std::make_unique<reco::PixelClusterTagInfoCollection>();

  // Open jet collection
  edm::Handle<edm::View<reco::Jet> > collectionJets;
  iEvent.getByToken(m_jets, collectionJets);

  // Get primary vertex in the event
  edm::Handle<reco::VertexCollection> collectionPVs;
  iEvent.getByToken(m_vertices, collectionPVs);

  // Count jets above threshold
  int nJets(0);
  for (auto jetIt = collectionJets->begin(); jetIt != collectionJets->end(); ++jetIt) {
    if (jetIt->pt() > m_minJetPt)
      nJets++;
  }

  // If no suitable Jet and PV is available, skip the event without opening pixel collection
  if (collectionPVs->empty() || nJets <= 0) {
    iEvent.put(std::move(collectionTagInfo));
    return;
  }

  // Get primary vertex 3D position
  reco::VertexCollection::const_iterator firstPV = collectionPVs->begin();
  GlobalPoint v3(firstPV->x(), firstPV->y(), firstPV->z());

  // Open Pixel Cluster collection
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > collectionHandle;
  iEvent.getByToken(m_pixelhit, collectionHandle);
  const edmNew::DetSetVector<SiPixelCluster>& collectionClusters(*collectionHandle);

  // Open Geometry
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom);
  const TrackerGeometry& theTracker(*geom);

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoH;
  iSetup.get<TrackerTopologyRcd>().get(tTopoH);
  const TrackerTopology* tTopo = tTopoH.product();

  std::vector<reco::PixelClusterProperties> clusters;

  // Get vector of detunit ids, and fill a vector of PixelClusterProperties in the loop
  for (auto const& detUnit : collectionClusters) {
    if (detUnit.empty())
      continue;
    DetId detId = DetId(detUnit.detId());  // Get the Detid object for pixel detector selection
    if (detId.det() != DetId::Tracker)
      continue;
    if (!(detId.subdetId() == PixelSubdetector::PixelBarrel ||
          (m_addFPIX && detId.subdetId() == PixelSubdetector::PixelEndcap)))
      continue;
    unsigned int layer = tTopo->layer(detId);  // The layer index is in range 1-4 or 1-3
    if (layer == 0 || layer > m_nLayers)
      continue;

    // Get the geom-detector
    const auto* geomDet = theTracker.idToDet(detId);
    const auto* topol = &geomDet->topology();

    for (auto const& clUnit : detUnit) {
      if (m_minADC > 0 and clUnit.charge() < m_minADC)
        continue;  // skip cluster if below threshold
      // Get global position of the cluster
      LocalPoint lp = topol->localPosition(MeasurementPoint(clUnit.x(), clUnit.y()));
      GlobalPoint gp = geomDet->surface().toGlobal(lp);
      // Fill PixelClusterProperties vector for matching
      reco::PixelClusterProperties cp = {gp.x(), gp.y(), gp.z(), clUnit.charge(), layer};
      clusters.push_back(cp);
    }
  }

  // Loop over jets and perform geometrical matching with pixel clusters
  for (unsigned int j = 0, nj = collectionJets->size(); j < nj; j++) {
    if (collectionJets->at(j).pt() < m_minJetPt)
      continue;

    float cR = m_hadronMass * 2. / (collectionJets->at(j).pt());  // 2 mX / pT
    edm::RefToBase<reco::Jet> jetRef = collectionJets->refAt(j);  // Get jet RefToBase
    reco::PixelClusterData data(m_nLayers);

    for (auto const& cluster : clusters) {
      GlobalPoint c3(cluster.x, cluster.y, cluster.z);  // Get cluster 3D position
      float dR = reco::deltaR(c3 - v3, jetRef->momentum());
      // Match pixel clusters to jets and fill Data struct
      if (cluster.layer >= 1 && cluster.layer <= m_nLayers) {
        int idx(cluster.layer - 1);
        if (dR < 0.16) {
          data.r016[idx]++;
          if (dR < 0.10) {
            data.r010[idx]++;
            if (dR < 0.08) {
              data.r008[idx]++;
              if (dR < 0.06) {
                data.r006[idx]++;
                if (dR < 0.04)
                  data.r004[idx]++;
              }
            }
          }
        }
        if (dR < cR) {
          data.rvar[idx]++;
          data.rvwt[idx] += cluster.charge;
        }
      }
    }
    // Create tagInfo object and fill the collection
    reco::PixelClusterTagInfo tagInfo(data, jetRef);
    collectionTagInfo->push_back(tagInfo);
  }

  // Put the TagInfo collection in the event
  iEvent.put(std::move(collectionTagInfo));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PixelClusterTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("pixelhit", edm::InputTag("siPixelClusters"));
  desc.add<bool>("isPhase1", true);
  desc.add<bool>("addForward", true);
  desc.add<int>("minAdcCount", -1);
  desc.add<double>("minJetPtCut", 100.);
  desc.add<double>("maxJetEtaCut", 2.5);
  desc.add<double>("hadronMass", 12.);
  descriptions.add("pixelClusterTagInfos", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelClusterTagInfoProducer);
