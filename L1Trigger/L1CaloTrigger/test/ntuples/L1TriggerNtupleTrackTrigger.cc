#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FWCore/Framework/interface/ESWatcher.h"



class L1TriggerNtupleTrackTrigger : public HGCalTriggerNtupleBase {

  public:
    L1TriggerNtupleTrackTrigger(const edm::ParameterSet& conf);
    ~L1TriggerNtupleTrackTrigger() override{};
    void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) final;
    void fill(const edm::Event& e, const edm::EventSetup& es) final;
    typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;

  private:
    void clear() final;
    // HGCalTriggerTools triggerTools_;
    std::pair<float,float> propagateToCalo(const math::XYZTLorentzVector& iMom,
                                           const math::XYZTLorentzVector& iVtx,
                                           double iCharge,
                                           double iBField);

    edm::EDGetToken track_token_;

    int l1track_n_ ;
    std::vector<float> l1track_pt_;
    // std::vector<float> l1track_energy_;
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

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
    L1TriggerNtupleTrackTrigger,
    "L1TriggerNtupleTrackTrigger" );


L1TriggerNtupleTrackTrigger::
L1TriggerNtupleTrackTrigger(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

void
L1TriggerNtupleTrackTrigger::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{
  track_token_ = collector.consumes<std::vector<TTTrack< Ref_Phase2TrackerDigi_> > >(conf.getParameter<edm::InputTag>("TTTracks"));

  tree.Branch("l1track_n",       &l1track_n_, "l1track_n/I");
  tree.Branch("l1track_pt",      &l1track_pt_);
  // tree.Branch("l1track_energy", &l1track_energy_);
  tree.Branch("l1track_eta",     &l1track_eta_);
  tree.Branch("l1track_phi",     &l1track_phi_);
  tree.Branch("l1track_curv",    &l1track_curv_);
  tree.Branch("l1track_chi2",    &l1track_chi2_);
  tree.Branch("l1track_chi2Red",    &l1track_chi2Red_);
  tree.Branch("l1track_nStubs",  &l1track_nStubs_);
  tree.Branch("l1track_z0",      &l1track_z0_);
  tree.Branch("l1track_charge",  &l1track_charge_);
  tree.Branch("l1track_caloeta", &l1track_caloeta_);
  tree.Branch("l1track_calophi", &l1track_calophi_);

}



void
L1TriggerNtupleTrackTrigger::fill(const edm::Event& ev, const edm::EventSetup& es) {

  // the L1Tracks
  edm::Handle<std::vector< L1TTTrackType >> l1TTTrackHandle;
  ev.getByToken(track_token_, l1TTTrackHandle);

  float fBz = 0;
  if(magfield_watcher_.check(es)) {
    edm::ESHandle<MagneticField> magfield;
    es.get<IdealMagneticFieldRecord>().get(magfield);
    // aField_ = &(*magfield);
    fBz = magfield->inTesla(GlobalPoint(0,0,0)).z();
  }

  triggerTools_.eventSetup(es);

  clear();
  for (auto trackIter = l1TTTrackHandle->begin(); trackIter != l1TTTrackHandle->end(); ++trackIter) {
    l1track_n_++;
    // NOTE: filter on fabs(eta) in EE to reduce the size of the collection
    if(fabs(trackIter->getMomentum().eta()) < 1.3) continue;
    // physical values

    l1track_pt_.emplace_back(trackIter->getMomentum().perp());
    // l1track_energy_.emplace_back(trackIter->energy());
    l1track_eta_.emplace_back(trackIter->getMomentum().eta());
    l1track_phi_.emplace_back(trackIter->getMomentum().phi());
    l1track_curv_.emplace_back(trackIter->getRInv());
    l1track_chi2_.emplace_back(trackIter->getChi2());
    l1track_chi2Red_.emplace_back(trackIter->getChi2Red());
    l1track_nStubs_.emplace_back(trackIter->getStubRefs().size());
    // FIXME: need to be configuratble?
    int nParam_ = 4;
    float z0   = trackIter->getPOCA(nParam_).z(); //cm
    int charge = trackIter->getRInv() > 0 ? +1 : -1;

    reco::Candidate::PolarLorentzVector p4p(trackIter->getMomentum().perp(),
                                            trackIter->getMomentum().eta(),
                                            trackIter->getMomentum().phi(), 0); // no mass ?
    reco::Particle::LorentzVector p4(p4p.X(), p4p.Y(), p4p.Z(), p4p.E());
    reco::Particle::Point vtx(0.,0.,z0);

    auto caloetaphi = propagateToCalo(p4, math::XYZTLorentzVector(0.,0.,z0,0.), charge, fBz);

    l1track_z0_.emplace_back(z0);
    l1track_charge_.emplace_back(charge);
    l1track_caloeta_.emplace_back(caloetaphi.first);
    l1track_calophi_.emplace_back(caloetaphi.second);
  }
}


void
L1TriggerNtupleTrackTrigger::clear() {
  l1track_n_ = 0;
  l1track_pt_.clear();
  // l1track_energy_.clear();
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




// #include "FastSimulation/Particle/interface/RawParticle.h"

std::pair<float,float> L1TriggerNtupleTrackTrigger::propagateToCalo(const math::XYZTLorentzVector& iMom,
                                                                    const math::XYZTLorentzVector& iVtx,
                                                                    double iCharge,
                                                                    double iBField) {
    BaseParticlePropagator particle = BaseParticlePropagator(RawParticle(iMom,iVtx),0.,0.,iBField);
    particle.setCharge(iCharge);
    // particle.propagateToEcalEntrance(false);
    particle.setPropagationConditions(129.0 , triggerTools_.getLayerZ(1) , false);
    particle.propagate();
    double ecalShowerDepth = reco::PFCluster::getDepthCorrection(particle.momentum().E(),false,false);
    math::XYZVector point = math::XYZVector(particle.vertex())+math::XYZTLorentzVector(particle.momentum()).Vect().Unit()*ecalShowerDepth;
    // math::XYZVector point  = particle.vertex();
    // math::XYZVector point = math::XYZVector(particle.vertex())+math::XYZTLorentzVector(particle.momentum()).Vect().Unit()*ecalShowerDepth;
    return std::make_pair(point.eta(), point.phi());
}
