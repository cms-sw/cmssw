#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include "L1Trigger/L1TTrackMatch/interface/pTFrom2Stubs.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "L1TCaloTriggerNtupleBase.h"

class L1TriggerNtupleTrackTrigger : public L1TCaloTriggerNtupleBase {
public:
  L1TriggerNtupleTrackTrigger(const edm::ParameterSet& conf);
  ~L1TriggerNtupleTrackTrigger() override{};
  void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
  void fill(const edm::Event& e, const edm::EventSetup& es) final;
  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;

private:
  void clear() final;
  std::pair<float, float> propagateToCalo(const math::XYZTLorentzVector& iMom,
                                          const math::XYZTLorentzVector& iVtx,
                                          double iCharge,
                                          double iBField);

  edm::EDGetToken track_token_;

  int l1track_n_;
  std::vector<float> l1track_pt_;
  std::vector<float> l1track_pt2stubs_;
  std::vector<float> l1track_eta_;
  std::vector<float> l1track_phi_;
  std::vector<float> l1track_curv_;
  std::vector<float> l1track_chi2_;
  std::vector<float> l1track_chi2Red_;
  std::vector<int> l1track_nStubs_;
  std::vector<float> l1track_z0_;
  std::vector<int> l1track_charge_;
  std::vector<float> l1track_caloeta_;
  std::vector<float> l1track_calophi_;

  edm::ESWatcher<IdealMagneticFieldRecord> magfield_watcher_;
  HGCalTriggerTools triggerTools_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory, L1TriggerNtupleTrackTrigger, "L1TriggerNtupleTrackTrigger");

L1TriggerNtupleTrackTrigger::L1TriggerNtupleTrackTrigger(const edm::ParameterSet& conf)
    : L1TCaloTriggerNtupleBase(conf) {}

void L1TriggerNtupleTrackTrigger::initialize(TTree& tree,
                                             const edm::ParameterSet& conf,
                                             edm::ConsumesCollector&& collector) {
  track_token_ =
      collector.consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(conf.getParameter<edm::InputTag>("TTTracks"));

  tree.Branch(branch_name_w_prefix("n").c_str(), &l1track_n_, branch_name_w_prefix("n/I").c_str());
  tree.Branch(branch_name_w_prefix("pt").c_str(), &l1track_pt_);
  tree.Branch(branch_name_w_prefix("pt2stubs").c_str(), &l1track_pt2stubs_);
  tree.Branch(branch_name_w_prefix("eta").c_str(), &l1track_eta_);
  tree.Branch(branch_name_w_prefix("phi").c_str(), &l1track_phi_);
  tree.Branch(branch_name_w_prefix("curv").c_str(), &l1track_curv_);
  tree.Branch(branch_name_w_prefix("chi2").c_str(), &l1track_chi2_);
  tree.Branch(branch_name_w_prefix("chi2Red").c_str(), &l1track_chi2Red_);
  tree.Branch(branch_name_w_prefix("nStubs").c_str(), &l1track_nStubs_);
  tree.Branch(branch_name_w_prefix("z0").c_str(), &l1track_z0_);
  tree.Branch(branch_name_w_prefix("charge").c_str(), &l1track_charge_);
  tree.Branch(branch_name_w_prefix("caloeta").c_str(), &l1track_caloeta_);
  tree.Branch(branch_name_w_prefix("calophi").c_str(), &l1track_calophi_);
}

void L1TriggerNtupleTrackTrigger::fill(const edm::Event& ev, const edm::EventSetup& es) {
  // the L1Tracks
  edm::Handle<std::vector<L1TTTrackType>> l1TTTrackHandle;
  ev.getByToken(track_token_, l1TTTrackHandle);

  float fBz = 0;
  if (magfield_watcher_.check(es)) {
    edm::ESHandle<MagneticField> magfield;
    es.get<IdealMagneticFieldRecord>().get(magfield);
    fBz = magfield->inTesla(GlobalPoint(0, 0, 0)).z();
  }

  // geometry needed to call pTFrom2Stubs
  edm::ESHandle<TrackerGeometry> geomHandle;
  es.get<TrackerDigiGeometryRecord>().get("idealForDigi", geomHandle);
  const TrackerGeometry* tGeom = geomHandle.product();

  triggerTools_.eventSetup(es);

  clear();
  for (auto trackIter = l1TTTrackHandle->begin(); trackIter != l1TTTrackHandle->end(); ++trackIter) {
    l1track_n_++;
    l1track_pt_.emplace_back(trackIter->momentum().perp());
    l1track_pt2stubs_.emplace_back(pTFrom2Stubs::pTFrom2(trackIter, tGeom));
    l1track_eta_.emplace_back(trackIter->momentum().eta());
    l1track_phi_.emplace_back(trackIter->momentum().phi());
    l1track_curv_.emplace_back(trackIter->rInv());
    l1track_chi2_.emplace_back(trackIter->chi2());
    l1track_chi2Red_.emplace_back(trackIter->chi2Red());
    l1track_nStubs_.emplace_back(trackIter->getStubRefs().size());
    float z0 = trackIter->POCA().z();  //cm
    int charge = trackIter->rInv() > 0 ? +1 : -1;

    reco::Candidate::PolarLorentzVector p4p(
        trackIter->momentum().perp(), trackIter->momentum().eta(), trackIter->momentum().phi(), 0);  // no mass ?
    reco::Particle::LorentzVector p4(p4p.X(), p4p.Y(), p4p.Z(), p4p.E());
    reco::Particle::Point vtx(0., 0., z0);

    auto caloetaphi = propagateToCalo(p4, math::XYZTLorentzVector(0., 0., z0, 0.), charge, fBz);

    l1track_z0_.emplace_back(z0);
    l1track_charge_.emplace_back(charge);
    l1track_caloeta_.emplace_back(caloetaphi.first);
    l1track_calophi_.emplace_back(caloetaphi.second);
  }
}

void L1TriggerNtupleTrackTrigger::clear() {
  l1track_n_ = 0;
  l1track_pt_.clear();
  l1track_pt2stubs_.clear();
  l1track_eta_.clear();
  l1track_phi_.clear();
  l1track_curv_.clear();
  l1track_chi2_.clear();
  l1track_chi2Red_.clear();
  l1track_nStubs_.clear();
  l1track_z0_.clear();
  l1track_charge_.clear();
  l1track_caloeta_.clear();
  l1track_calophi_.clear();
}

std::pair<float, float> L1TriggerNtupleTrackTrigger::propagateToCalo(const math::XYZTLorentzVector& iMom,
                                                                     const math::XYZTLorentzVector& iVtx,
                                                                     double iCharge,
                                                                     double iBField) {
  BaseParticlePropagator prop = BaseParticlePropagator(RawParticle(iMom, iVtx, iCharge), 0., 0., iBField);
  prop.setPropagationConditions(129.0, triggerTools_.getLayerZ(1), false);
  prop.propagate();
  double ecalShowerDepth = reco::PFCluster::getDepthCorrection(prop.particle().momentum().E(), false, false);
  math::XYZVector point = math::XYZVector(prop.particle().vertex()) +
                          math::XYZTLorentzVector(prop.particle().momentum()).Vect().Unit() * ecalShowerDepth;
  return std::make_pair(point.eta(), point.phi());
}
