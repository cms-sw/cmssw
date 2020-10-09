#ifndef RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitsBuilderValidation_H
#define RecoLocalTracker_SiPhase2VectorHitBuilder_VectorHitsBuilderValidation_H

#include <map>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include <TH1F.h>
#include <TH2D.h>
#include <TGraph.h>
#include <THStack.h>
#include <TCanvas.h>
#include <TTree.h>
#include <TArrow.h>

struct VHHistos {
  THStack* numberVHsMixed;
  TH1F* numberVHsPS;
  TH1F* numberVHs2S;

  TGraph* globalPosXY[3];
  TGraph* localPosXY[3];

  TH1F* deltaXVHSimHits[3];
  TH1F* deltaYVHSimHits[3];

  TH1F* deltaXVHSimHits_P[3];
  TH1F* deltaYVHSimHits_P[3];

  TH1F* digiEfficiency[3];

  TH1F* totalSimHits;
  TH1F* primarySimHits;
  TH1F* otherSimHits;

  TH1F* curvature;
  TH1F* width;
  TH1F* deltaXlocal;
};

class VectorHitsBuilderValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2TrackerCluster1DRef;

  typedef std::map<unsigned int, std::vector<PSimHit> > SimHitsMap;
  typedef std::map<unsigned int, SimTrack> SimTracksMap;

  explicit VectorHitsBuilderValidation(const edm::ParameterSet&);
  ~VectorHitsBuilderValidation();
  void beginJob();
  void endJob();
  void analyze(const edm::Event&, const edm::EventSetup&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::map<unsigned int, VHHistos>::iterator createLayerHistograms(unsigned int);
  void CreateVHsXYGraph(const std::vector<Global3DPoint>, const std::vector<Global3DVector>);
  void CreateVHsRZGraph(const std::vector<Global3DPoint>, const std::vector<Global3DVector>);
  void CreateWindowCorrGraph();

  unsigned int getLayerNumber(const DetId&);
  unsigned int getModuleNumber(const DetId& detid);
  void printCluster(const GeomDetUnit* geomDetUnit, const OmniClusterRef cluster);

  std::pair<bool, uint32_t> isTrue(const VectorHit vh,
                                   const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& siphase2SimLinks,
                                   DetId& detId) const;
  std::vector<std::pair<uint32_t, EncodedEventId> > getSimTrackIds(
      const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >&, const DetId&, uint32_t) const;
  unsigned int getSimTrackId(const edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& pixelSimLinks,
                             const DetId& detId,
                             unsigned int channel) const;

  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D> > srcClu_;
  edm::EDGetTokenT<VectorHitCollection> VHacc_;
  edm::EDGetTokenT<VectorHitCollection> VHrej_;
  edm::ESInputTag cpeTag_;
  const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe_;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > siphase2OTSimLinksToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> simHitsToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> simVerticesToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;

  const TrackerGeometry* tkGeom_;
  const TrackerTopology* tkTopo_;
  const MagneticField* magField_;

  TTree* tree_;
  TGraph* trackerLayoutRZ_[3];
  TGraph* trackerLayoutXY_[3];
  TGraph* trackerLayoutXYBar_;
  TGraph* trackerLayoutXYEC_;
  TGraph* localPosXvsDeltaX_[3];
  TGraph* localPosYvsDeltaY_[3];
  TCanvas* VHXY_;
  TCanvas* VHRZ_;
  std::vector<TArrow*> arrowVHs_;

  TH2D* ParallaxCorrectionRZ_;
  TH1F* VHaccLayer_;
  TH1F* VHrejLayer_;
  TH1F* VHaccTrueLayer_;
  TH1F* VHrejTrueLayer_;
  TH1F* VHaccTrue_signal_Layer_;
  TH1F* VHrejTrue_signal_Layer_;

  std::map<unsigned int, VHHistos> histograms_;
};
#endif
