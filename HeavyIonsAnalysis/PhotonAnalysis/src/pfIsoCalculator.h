#ifndef pfIsoCalculator_h
#define pfIsoCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/HeavyIonEvent/interface/VoronoiBackground.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"



#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class pfIsoCalculator
{

 public:

  pfIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::InputTag &pfCandidateLabel_, const edm::InputTag &pfVoroniBkgLabel_, const edm::InputTag &vtxLabel_) ;
  pfIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::EDGetTokenT<edm::View<reco::PFCandidate> > pfCandidates, const edm::EDGetTokenT<edm::ValueMap<reco::VoronoiBackground> > pfVoronoiBkg, const math::XYZPoint& pv) ;
  double getPfIso (const reco::Photon& photon,  int pfId, double r1=0.4, double r2=0.00, double jWidth=0.00, double threshold=0);
  double getVsPfIso(const reco::Photon& photon,  int pfId, double r1=0.4, double r2=0.00, double jWidth=0.00, double threshold=0, bool isVsCorrected=true);
  double getPfIso (const reco::GsfElectron& ele, int pfId, double r = 0.4, double r2=0.00, double threshold = 0);

 private:
  //const reco::PFCandidateCollection *pfCandidateColl;
  edm::Handle<edm::View<reco::PFCandidate> > candidatesView;
  edm::Handle<reco::VoronoiMap> pfVoronoiBkg;
  edm::Handle<reco::VertexCollection> vtxs;
  reco::Vertex::Point vtx_;

};

#endif
