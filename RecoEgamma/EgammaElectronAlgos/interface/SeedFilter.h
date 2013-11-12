#ifndef SeedFilter_H
#define SeedFilter_H

/** \class SeedFilter
 * originally  Matteo Sani's SubSeedGenerator
 *
 ************************************************************/
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <TMath.h>

#include <Math/VectorUtil.h>
#include <Math/Point3D.h>

class SeedGeneratorFromRegionHits;
class MagneticField;
namespace edm { class ConsumesCollector; }

class SeedFilter {
 public:

  struct Tokens {
    edm::EDGetTokenT<std::vector<reco::Vertex> > token_vtx;
    edm::EDGetTokenT<reco::BeamSpot> token_bs;
  };

  SeedFilter(const edm::ParameterSet& conf,
	     const Tokens& tokens,
	     edm::ConsumesCollector& iC);
  ~SeedFilter();

  void seeds(edm::Event&, const edm::EventSetup&, const reco::SuperClusterRef &, TrajectorySeedCollection *);

 private:
  SeedGeneratorFromRegionHits *combinatorialSeedGenerator;

  // remove them FIXME
  double dr_, deta_, dphi_, pt_;

  double ptmin_, vertexz_, originradius_,  halflength_, deltaEta_, deltaPhi_;
  bool useZvertex_;
  //  edm::InputTag BSProducer_;  //FIXME?
  edm::EDGetTokenT<std::vector<reco::Vertex> > vertexSrc_;

  edm::ESHandle<MagneticField> theMagField;

  int hitsfactoryMode_;

  edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;

  std::string measurementTrackerName_;
};

#endif // SeedFilter_H


