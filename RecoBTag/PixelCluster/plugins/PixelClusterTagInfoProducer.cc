#include "RecoBTag/PixelCluster/interface/PixelClusterTagInfoProducer.h"

PixelClusterTagInfoProducer::PixelClusterTagInfoProducer(const edm::ParameterSet& iConfig)
    : m_jets(consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("jets"))),
      m_vertices(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      m_pixelhit(consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelhit"))),
      m_isPhase1(iConfig.getParameter<bool>("isPhase1")),
      m_addFPIX(iConfig.getParameter<bool>("addForward")),
      m_minADC(iConfig.getParameter<int>("minAdcCount")),
      m_minJetPt(iConfig.getParameter<double>("minJetPtCut")),
      m_maxJetEta(iConfig.getParameter<double>("maxJetEtaCut")) {
  produces<reco::PixelClusterTagInfoCollection>();

  m_nLayers = (m_isPhase1 ? 4 : 3);
}

PixelClusterTagInfoProducer::~PixelClusterTagInfoProducer() {}

// ------------ method called to produce the data  ------------
void PixelClusterTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Declare produced collection
  auto pixelTagInfo = std::make_unique<reco::PixelClusterTagInfoCollection>();

  // Open jet collection
  edm::Handle<edm::View<reco::Jet> > collectionJets;
  iEvent.getByToken(m_jets, collectionJets);

  // Count jets above threshold
  int nJets(0);
  for (auto jetIt = collectionJets->begin(); jetIt != collectionJets->end(); ++jetIt) {
    if (jetIt->pt() > m_minJetPt)
      nJets++;
  }

  // Get primary vertex in the event
  edm::Handle<reco::VertexCollection> collectionPVs;
  iEvent.getByToken(m_vertices, collectionPVs);
  reco::VertexCollection::const_iterator firstPV = collectionPVs->begin();

  for (reco::VertexCollection::const_iterator vtxIt = collectionPVs->begin(); vtxIt != collectionPVs->end(); ++vtxIt) {
    firstPV = vtxIt;
    break;
  }

  // If no suitable Jet and PV is available, skip the event without opening pixel collection
  if (collectionPVs->empty() || nJets <= 0) {
    iEvent.put(std::move(pixelTagInfo));
    return;
  }

  // Open Pixel Cluster collection
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > collectionClusters;
  iEvent.getByToken(m_pixelhit, collectionClusters);

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
  for (edmNew::DetSetVector<SiPixelCluster>::const_iterator detUnit = collectionClusters->begin();
       detUnit != collectionClusters->end();
       ++detUnit) {
    if (detUnit->empty())
      continue;
    unsigned int detid = detUnit->detId();
    DetId detId = DetId(detid);          // Get the Detid object
    unsigned int detType = detId.det();  // det type, pixel = 1
    if (detType != 1)
      continue;                             // Consider only pixels
    unsigned int subid = detId.subdetId();  // Subdetector type, pix barrel = 1, forward = 2

    // Get the geom-detector
    const PixelGeomDetUnit* geomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId));
    const PixelTopology* topol = &(geomDet->specificTopology());
    int layer = 0;  // The layer index is in range 1-4

    if (subid == 1) {  // pixel barrel
      PixelBarrelName pbn(detid, tTopo, m_isPhase1);
      layer = pbn.layerName();
    } else if (m_addFPIX && subid == 2) {  // pixel forward
      PixelEndcapName pen(detid, tTopo, m_isPhase1);
      layer = pen.diskName();
    }
    if (layer == 0 || layer > m_nLayers)
      continue;

    for (edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = detUnit->begin(); clustIt != detUnit->end();
         ++clustIt) {
      // get global position of the cluster
      LocalPoint lp = topol->localPosition(MeasurementPoint(clustIt->x(), clustIt->y()));
      GlobalPoint clustgp = geomDet->surface().toGlobal(lp);
      if (m_minADC > 0 and clustIt->charge() < m_minADC)
        continue;  // skip cluster if below threshold
      reco::PixelClusterProperties cp = {clustgp.x(), clustgp.y(), clustgp.z(), clustIt->charge(), layer};
      clusters.push_back(cp);
    }
  }

  // Loop over jets and perform geometrical matching with pixel clusters
  for (unsigned int j = 0, nj = collectionJets->size(); j < nj; j++) {
    if (collectionJets->at(j).pt() < m_minJetPt)
      continue;

    edm::RefToBase<reco::Jet> jetRef = collectionJets->refAt(j);  // Get jet RefToBase

    reco::PixelClusterData data = {};  // Initialize new Data to 0
    reco::PixelClusterTagInfo tagInfo = {};

    for (auto cluIt = clusters.begin(); cluIt != clusters.end(); ++cluIt) {
      TVector3 c3(cluIt->x - firstPV->x(), cluIt->y - firstPV->y(), cluIt->z - firstPV->z());
      TVector3 j3(jetRef->px(), jetRef->py(), jetRef->pz());
      float dR = j3.DeltaR(c3);
      float sC = 12. * 2. / (jetRef->pt());

      // Match pixel clusters to jets and fill Data struct
      if (cluIt->layer >= 1 && cluIt->layer <= m_nLayers) {
        if (dR < 0.04)
          data.R004[cluIt->layer]++;
        if (dR < 0.06)
          data.R006[cluIt->layer]++;
        if (dR < 0.08)
          data.R008[cluIt->layer]++;
        if (dR < 0.10)
          data.R010[cluIt->layer]++;
        if (dR < 0.16)
          data.R016[cluIt->layer]++;
        if (dR < sC)
          data.RVAR[cluIt->layer]++;
        if (dR < sC)
          data.RVWT[cluIt->layer] += cluIt->charge;
      }
    }

    tagInfo.setJetRef(jetRef);
    tagInfo.setData(data);

    pixelTagInfo->push_back(tagInfo);
  }

  // Put the TagInfo collection in the event
  iEvent.put(std::move(pixelTagInfo));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PixelClusterTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelClusterTagInfoProducer);
