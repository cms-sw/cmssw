#ifndef RecoAlgos_MultiTrackSelector_h
#define RecoAlgos_MultiTrackSelector_h
/** \class MultiTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author David Lange
 *
 *
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"

class dso_hidden MultiTrackSelector : public edm::stream::EDProducer<> {
private:
public:
  /// constructor
  explicit MultiTrackSelector();
  explicit MultiTrackSelector(const edm::ParameterSet &cfg);
  /// destructor
  ~MultiTrackSelector() override;

  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

protected:
  void beginStream(edm::StreamID) final;

  // void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final {
  //  init();
  //}
  //void beginRun(edm::Run const&, edm::EventSetup const&) final { init(); }
  // void init(edm::EventSetup const& es) const;

  typedef math::XYZPoint Point;
  /// process one event
  void produce(edm::Event &evt, const edm::EventSetup &es) final { run(evt, es); }
  virtual void run(edm::Event &evt, const edm::EventSetup &es) const;

  /// return class, or -1 if rejected
  bool select(unsigned tsNum,
              const reco::BeamSpot &vertexBeamSpot,
              const TrackingRecHitCollection &recHits,
              const reco::Track &tk,
              const std::vector<Point> &points,
              std::vector<float> &vterr,
              std::vector<float> &vzerr,
              double mvaVal) const;
  void selectVertices(unsigned int tsNum,
                      const reco::VertexCollection &vtxs,
                      std::vector<Point> &points,
                      std::vector<float> &vterr,
                      std::vector<float> &vzerr) const;

  void processMVA(edm::Event &evt,
                  const edm::EventSetup &es,
                  const reco::BeamSpot &beamspot,
                  const reco::VertexCollection &vertices,
                  int selIndex,
                  std::vector<float> &mvaVals_,
                  bool writeIt = false) const;
  Point getBestVertex(const reco::TrackBaseRef, const reco::VertexCollection) const;

  /// source collection label
  edm::EDGetTokenT<reco::TrackCollection> src_;
  edm::EDGetTokenT<TrackingRecHitCollection> hSrc_;
  edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  bool useVertices_;
  bool useVtxError_;
  bool useAnyMVA_;
  edm::EDGetTokenT<reco::VertexCollection> vertices_;

  /// do I have to set a quality bit?
  std::vector<bool> setQualityBit_;
  std::vector<reco::TrackBase::TrackQuality> qualityToSet_;

  /// vertex cuts
  std::vector<int32_t> vtxNumber_;
  //StringCutObjectSelector is not const thread safe
  std::vector<StringCutObjectSelector<reco::Vertex> > vertexCut_;

  //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
  std::vector<std::vector<double> > res_par_;
  std::vector<double> chi2n_par_;
  std::vector<double> chi2n_no1Dmod_par_;
  std::vector<std::vector<double> > d0_par1_;
  std::vector<std::vector<double> > dz_par1_;
  std::vector<std::vector<double> > d0_par2_;
  std::vector<std::vector<double> > dz_par2_;
  // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
  std::vector<bool> applyAdaptedPVCuts_;

  /// Impact parameter absolute cuts
  std::vector<double> max_d0_;
  std::vector<double> max_z0_;
  std::vector<double> nSigmaZ_;

  /// Cuts on numbers of layers with hits/3D hits/lost hits.
  std::vector<uint32_t> min_layers_;
  std::vector<uint32_t> min_3Dlayers_;
  std::vector<uint32_t> max_lostLayers_;
  std::vector<uint32_t> min_hits_bypass_;

  // pterror and nvalid hits cuts
  std::vector<double> max_relpterr_;
  std::vector<uint32_t> min_nhits_;

  std::vector<int32_t> max_minMissHitOutOrIn_;
  std::vector<int32_t> max_lostHitFraction_;

  std::vector<double> min_eta_;
  std::vector<double> max_eta_;

  // Flag and absolute cuts if no PV passes the selection
  std::vector<double> max_d0NoPV_;
  std::vector<double> max_z0NoPV_;
  std::vector<bool> applyAbsCutsIfNoPV_;
  //if true, selector flags but does not select
  std::vector<bool> keepAllTracks_;

  // allow one of the previous psets to be used as a prefilter
  std::vector<unsigned int> preFilter_;
  std::vector<std::string> name_;

  //setup mva selector
  std::vector<bool> useMVA_;
  std::vector<bool> useMVAonly_;

  std::vector<double> min_MVA_;

  std::vector<std::string> mvaType_;
  std::vector<std::string> forestLabel_;
  std::vector<GBRForest *> forest_;
  bool useForestFromDB_;
  std::string dbFileName_;
};

#endif
