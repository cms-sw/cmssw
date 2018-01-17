#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/CaloGeometryTools/interface/Transform3DPJ.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// NOTE: most of this code is borrowed by https://github.com/CMS-HGCAL/reco-ntuples
// kudos goes to the original authors. Ideally the 2 repos should be merged since they share part of the use case
#include <iostream>

namespace HGCal_helpers {

  class coordinates {
   public:
    coordinates() : x(0), y(0), z(0), eta(100), phi(0) {}
    float x, y, z, eta, phi;
    inline math::XYZTLorentzVectorD toVector() { return math::XYZTLorentzVectorD(x, y, z, 0); }
  };


  class simpleTrackPropagator {
   public:
    simpleTrackPropagator(MagneticField const *f)
        : field_(f), prod_(field_, alongMomentum, 5.e-5), absz_target_(0) {
      ROOT::Math::SMatrixIdentity id;
      AlgebraicSymMatrix55 C(id);
      C *= 0.001;
      err_ = CurvilinearTrajectoryError(C);
    }
    void setPropagationTargetZ(const float &z);

    bool propagate(const double px, const double py, const double pz, const double x, const double y,
                   const double z, const float charge, coordinates &coords) const;

    bool propagate(const math::XYZTLorentzVectorD &momentum, const math::XYZTLorentzVectorD &position,
                   const float charge, coordinates &coords) const;

   private:
    simpleTrackPropagator() : field_(0), prod_(field_, alongMomentum, 5.e-5), absz_target_(0) {}
    const RKPropagatorInS &RKProp() const { return prod_.propagator; }
    Plane::PlanePointer targetPlaneForward_, targetPlaneBackward_;
    MagneticField const *field_;
    CurvilinearTrajectoryError err_;
    defaultRKPropagator::Product prod_;
    float absz_target_;
  };

  void simpleTrackPropagator::setPropagationTargetZ(const float &z) {
    targetPlaneForward_ = Plane::build(Plane::PositionType(0, 0, std::abs(z)), Plane::RotationType());
    targetPlaneBackward_ =
        Plane::build(Plane::PositionType(0, 0, -std::abs(z)), Plane::RotationType());
    absz_target_ = std::abs(z);
  }
  bool simpleTrackPropagator::propagate(const double px, const double py, const double pz,
                                        const double x, const double y, const double z,
                                        const float charge, coordinates &output) const {
    output = coordinates();

    typedef TrajectoryStateOnSurface TSOS;
    GlobalPoint startingPosition(x, y, z);
    GlobalVector startingMomentum(px, py, pz);
    Plane::PlanePointer startingPlane =
        Plane::build(Plane::PositionType(x, y, z), Plane::RotationType());
    TSOS startingStateP(
        GlobalTrajectoryParameters(startingPosition, startingMomentum, charge, field_), err_,
        *startingPlane);

    TSOS trackStateP;
    if (pz > 0) {
      trackStateP = RKProp().propagate(startingStateP, *targetPlaneForward_);
    } else {
      trackStateP = RKProp().propagate(startingStateP, *targetPlaneBackward_);
    }
    if (trackStateP.isValid()) {
      output.x = trackStateP.globalPosition().x();
      output.y = trackStateP.globalPosition().y();
      output.z = trackStateP.globalPosition().z();
      output.phi = trackStateP.globalPosition().phi();
      output.eta = trackStateP.globalPosition().eta();
      return true;
    }
    return false;
  }

  bool simpleTrackPropagator::propagate(const math::XYZTLorentzVectorD &momentum,
                                        const math::XYZTLorentzVectorD &position, const float charge,
                                        coordinates &output) const {
    return propagate(momentum.px(), momentum.py(), momentum.pz(), position.x(), position.y(),
                     position.z(), charge, output);
  }

}  // HGCal_helpers


class HGCalTriggerNtupleGen : public HGCalTriggerNtupleBase
{

    public:
        HGCalTriggerNtupleGen(const edm::ParameterSet& );

        virtual void initialize(TTree&, const edm::ParameterSet&, edm::ConsumesCollector&&) override final;
        virtual void fill(const edm::Event&, const edm::EventSetup& ) override final;

        typedef ROOT::Math::Transform3DPJ Transform3D;
        typedef ROOT::Math::Transform3DPJ::Point Point;

    private:
        virtual void clear() override final;
        void retrieveLayerPositions(const edm::EventSetup &es, unsigned layers);

        edm::EDGetToken gen_token_;
        edm::EDGetToken gen_PU_token_;
        
        int gen_n_;
        int gen_PUNumInt_;
        float gen_TrueNumInt_;

        float vtx_x_;
        float vtx_y_;
        float vtx_z_;


        ////////////////////
        // GenParticles
        //
        std::vector<float> genpart_eta_;
        std::vector<float> genpart_phi_;
        std::vector<float> genpart_pt_;
        std::vector<float> genpart_energy_;
        std::vector<float> genpart_dvx_;
        std::vector<float> genpart_dvy_;
        std::vector<float> genpart_dvz_;
        std::vector<float> genpart_ovx_;
        std::vector<float> genpart_ovy_;
        std::vector<float> genpart_ovz_;
        std::vector<float> genpart_exx_;
        std::vector<float> genpart_exy_;
        std::vector<int> genpart_mother_;
        std::vector<float> genpart_exphi_;
        std::vector<float> genpart_exeta_;
        std::vector<float> genpart_fbrem_;
        std::vector<int> genpart_pid_;
        std::vector<int> genpart_gen_;
        std::vector<int> genpart_reachedEE_;
        std::vector<bool> genpart_fromBeamPipe_;
        std::vector<std::vector<float>> genpart_posx_;
        std::vector<std::vector<float>> genpart_posy_;
        std::vector<std::vector<float>> genpart_posz_;

        ////////////////////
        // reco::GenParticles
        //
        std::vector<float> gen_eta_;
        std::vector<float> gen_phi_;
        std::vector<float> gen_pt_;
        std::vector<float> gen_energy_;
        std::vector<int> gen_charge_;
        std::vector<int> gen_pdgid_;
        std::vector<int> gen_status_;
        std::vector<std::vector<int>> gen_daughters_;


        // -------convenient tool to deal with simulated tracks
        FSimEvent *mySimEvent_;

        std::vector<float> layerPositions_;
        //std::vector<double> dEdXWeights_;
        //std::vector<double> invThicknessCorrection_;

        // and also the magnetic field
        MagneticField const *aField_;

        HGCalTriggerTools triggerTools_;

        // edm::EDGetTokenT<std::vector<reco::GenParticle> > genParticles_;
        edm::EDGetToken simTracks_token_;
        edm::EDGetToken simVertices_token_;
        edm::EDGetToken hev_token_;


};

DEFINE_EDM_PLUGIN(HGCalTriggerNtupleFactory,
        HGCalTriggerNtupleGen,
        "HGCalTriggerNtupleGen" );


HGCalTriggerNtupleGen::
HGCalTriggerNtupleGen(const edm::ParameterSet& conf):HGCalTriggerNtupleBase(conf)
{
}

//FIXME: it should be called once per run...
void HGCalTriggerNtupleGen::retrieveLayerPositions(const edm::EventSetup &es, unsigned layers) {
  triggerTools_.setEventSetup(es);

  DetId id;
  for (unsigned ilayer = 1; ilayer <= layers; ++ilayer) {
    if (ilayer <= triggerTools_.lastLayerEE()) {
      id = HGCalDetId(ForwardSubdetector::HGCEE, 1, ilayer, 1, 50, 1);
    } else if (ilayer > triggerTools_.lastLayerEE() && ilayer <= triggerTools_.lastLayerFH()) {
      id = HGCalDetId(ForwardSubdetector::HGCHEF, 1, ilayer - triggerTools_.lastLayerEE(), 1, 50, 1);
    } else if (ilayer > triggerTools_.lastLayerFH()) {
      id = HcalDetId(HcalSubdetector::HcalEndcap, 50, 100, ilayer - triggerTools_.lastLayerFH());
    }
    const GlobalPoint pos = triggerTools_.getPosition(id);
    layerPositions_.push_back(pos.z());
  }
}


void
HGCalTriggerNtupleGen::
initialize(TTree& tree, const edm::ParameterSet& conf, edm::ConsumesCollector&& collector)
{

    edm::ParameterSet particleFilter_(conf.getParameter<edm::ParameterSet>("TestParticleFilter"));
    mySimEvent_ = new FSimEvent(particleFilter_);

    gen_token_ = collector.consumes<reco::GenParticleCollection>(conf.getParameter<edm::InputTag>("GenParticles"));

    gen_PU_token_ = collector.consumes<std::vector<PileupSummaryInfo>>(conf.getParameter<edm::InputTag>("GenPU"));
    tree.Branch("gen_n", &gen_n_, "gen_n/I");
    tree.Branch("gen_PUNumInt", &gen_PUNumInt_ ,"gen_PUNumInt/I");
    tree.Branch("gen_TrueNumInt", &gen_TrueNumInt_ ,"gen_TrueNumInt/F");

    hev_token_ = collector.consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));
    simTracks_token_ = collector.consumes<std::vector<SimTrack>>(edm::InputTag("g4SimHits"));
    simVertices_token_ = collector.consumes<std::vector<SimVertex>>(edm::InputTag("g4SimHits"));

    tree.Branch("vtx_x", &vtx_x_);
    tree.Branch("vtx_y", &vtx_y_);
    tree.Branch("vtx_z", &vtx_z_);


    tree.Branch("gen_eta", &gen_eta_);
    tree.Branch("gen_phi", &gen_phi_);
    tree.Branch("gen_pt", &gen_pt_);
    tree.Branch("gen_energy", &gen_energy_);
    tree.Branch("gen_charge", &gen_charge_);
    tree.Branch("gen_pdgid", &gen_pdgid_);
    tree.Branch("gen_status", &gen_status_);
    tree.Branch("gen_daughters", &gen_daughters_);

    tree.Branch("genpart_eta", &genpart_eta_);
    tree.Branch("genpart_phi", &genpart_phi_);
    tree.Branch("genpart_pt", &genpart_pt_);
    tree.Branch("genpart_energy", &genpart_energy_);
    tree.Branch("genpart_dvx", &genpart_dvx_);
    tree.Branch("genpart_dvy", &genpart_dvy_);
    tree.Branch("genpart_dvz", &genpart_dvz_);
    tree.Branch("genpart_ovx", &genpart_ovx_);
    tree.Branch("genpart_ovy", &genpart_ovy_);
    tree.Branch("genpart_ovz", &genpart_ovz_);
    tree.Branch("genpart_mother", &genpart_mother_);
    tree.Branch("genpart_exphi", &genpart_exphi_);
    tree.Branch("genpart_exeta", &genpart_exeta_);
    tree.Branch("genpart_exx", &genpart_exx_);
    tree.Branch("genpart_exy", &genpart_exy_);
    tree.Branch("genpart_fbrem", &genpart_fbrem_);
    tree.Branch("genpart_pid", &genpart_pid_);
    tree.Branch("genpart_gen", &genpart_gen_);
    tree.Branch("genpart_reachedEE", &genpart_reachedEE_);
    tree.Branch("genpart_fromBeamPipe", &genpart_fromBeamPipe_);
    tree.Branch("genpart_posx", &genpart_posx_);
    tree.Branch("genpart_posy", &genpart_posy_);
    tree.Branch("genpart_posz", &genpart_posz_);


}

void
HGCalTriggerNtupleGen::
fill(const edm::Event& iEvent, const edm::EventSetup& es)
{
    clear();

    edm::Handle<std::vector< PileupSummaryInfo > >  PupInfo_h;
    e.getByToken(gen_PU_token_, PupInfo_h);
    const std::vector< PileupSummaryInfo >& PupInfo = *PupInfo_h;


    // FIXME: this part could go in begin run
    edm::ESHandle<HepPDT::ParticleDataTable> pdt;
    es.getData(pdt);
    mySimEvent_->initializePdt(&(*pdt));

    retrieveLayerPositions(es, 52);

    edm::ESHandle<MagneticField> magfield;
    es.get<IdealMagneticFieldRecord>().get(magfield);
    aField_ = &(*magfield);
    // up to here...could go in the beginRun

    // This balck magic is needed to use the mySimEvent_
    ParticleTable::Sentry ptable(mySimEvent_->theTable());
    edm::Handle<edm::HepMCProduct> hevH;
    edm::Handle<std::vector<SimTrack>> simTracksHandle;
    edm::Handle<std::vector<SimVertex>> simVerticesHandle;

    iEvent.getByToken(hev_token_, hevH);
    iEvent.getByToken(simTracks_token_, simTracksHandle);
    iEvent.getByToken(simVertices_token_, simVerticesHandle);
    mySimEvent_->fill(*simTracksHandle, *simVerticesHandle);

    HepMC::GenVertex *primaryVertex = *(hevH)->GetEvent()->vertices_begin();
    vtx_x_ = primaryVertex->position().x() / 10.;  // to put in official units
    vtx_y_ = primaryVertex->position().y() / 10.;
    vtx_z_ = primaryVertex->position().z() / 10.;
    Point sim_pv(vtx_x_, vtx_y_, vtx_z_);


    HGCal_helpers::simpleTrackPropagator toHGCalPropagator(aField_);
    toHGCalPropagator.setPropagationTargetZ(layerPositions_[0]);
    std::vector<FSimTrack *> allselectedgentracks;
    unsigned int npart = mySimEvent_->nTracks();
    for (unsigned int i = 0; i < npart; ++i) {
      std::vector<float> xp, yp, zp;
      FSimTrack &myTrack(mySimEvent_->track(i));
      math::XYZTLorentzVectorD vtx(0, 0, 0, 0);

      int reachedEE = 0;  // compute the extrapolations for the particles reaching EE
                          // and for the gen particles
      double fbrem = -1;

      if (std::abs(myTrack.vertex().position().z()) >= layerPositions_[0]) continue;

      unsigned nlayers = 40;
      if (myTrack.noEndVertex())  // || myTrack.genpartIndex()>=0)
      {
        HGCal_helpers::coordinates propcoords;
        bool reachesHGCal = toHGCalPropagator.propagate(
            myTrack.momentum(), myTrack.vertex().position(), myTrack.charge(), propcoords);
        vtx = propcoords.toVector();

        if (reachesHGCal && vtx.Rho() < 160 && vtx.Rho() > 25) {
          reachedEE = 2;
          double dpt = 0;

          for (int i = 0; i < myTrack.nDaughters(); ++i) dpt += myTrack.daughter(i).momentum().pt();
          if (abs(myTrack.type()) == 11) fbrem = dpt / myTrack.momentum().pt();
        } else if (reachesHGCal && vtx.Rho() > 160)
          reachedEE = 1;

        HGCal_helpers::simpleTrackPropagator indiv_particleProp(aField_);
        for (unsigned il = 0; il < nlayers; ++il) {
          const float charge = myTrack.charge();
          indiv_particleProp.setPropagationTargetZ(layerPositions_[il]);
          HGCal_helpers::coordinates propCoords;
          indiv_particleProp.propagate(myTrack.momentum(), myTrack.vertex().position(), charge,
                                       propCoords);

          xp.push_back(propCoords.x);
          yp.push_back(propCoords.y);
          zp.push_back(propCoords.z);
        }
      } else {
        vtx = myTrack.endVertex().position();
      }
      auto orig_vtx = myTrack.vertex().position();

      allselectedgentracks.push_back(&mySimEvent_->track(i));
      // fill branches
      genpart_eta_.push_back(myTrack.momentum().eta());
      genpart_phi_.push_back(myTrack.momentum().phi());
      genpart_pt_.push_back(myTrack.momentum().pt());
      genpart_energy_.push_back(myTrack.momentum().energy());
      genpart_dvx_.push_back(vtx.x());
      genpart_dvy_.push_back(vtx.y());
      genpart_dvz_.push_back(vtx.z());

      genpart_ovx_.push_back(orig_vtx.x());
      genpart_ovy_.push_back(orig_vtx.y());
      genpart_ovz_.push_back(orig_vtx.z());

      HGCal_helpers::coordinates hitsHGCal;
      toHGCalPropagator.propagate(myTrack.momentum(), orig_vtx, myTrack.charge(), hitsHGCal);

      genpart_exphi_.push_back(hitsHGCal.phi);
      genpart_exeta_.push_back(hitsHGCal.eta);
      genpart_exx_.push_back(hitsHGCal.x);
      genpart_exy_.push_back(hitsHGCal.y);

      genpart_fbrem_.push_back(fbrem);
      genpart_pid_.push_back(myTrack.type());
      genpart_gen_.push_back(myTrack.genpartIndex());
      genpart_reachedEE_.push_back(reachedEE);
      genpart_fromBeamPipe_.push_back(true);

      genpart_posx_.push_back(xp);
      genpart_posy_.push_back(yp);
      genpart_posz_.push_back(zp);
    }


    edm::Handle<std::vector<reco::GenParticle>> genParticlesHandle;
    iEvent.getByToken(gen_token_, genParticlesHandle);
    for (std::vector<reco::GenParticle>::const_iterator it_p = genParticlesHandle->begin();
         it_p != genParticlesHandle->end(); ++it_p) {
      gen_eta_.push_back(it_p->eta());
      gen_phi_.push_back(it_p->phi());
      gen_pt_.push_back(it_p->pt());
      gen_energy_.push_back(it_p->energy());
      gen_charge_.push_back(it_p->charge());
      gen_pdgid_.push_back(it_p->pdgId());
      gen_status_.push_back(it_p->status());
      std::vector<int> daughters(it_p->daughterRefVector().size(), 0);
      for (unsigned j = 0; j < it_p->daughterRefVector().size(); ++j) {
        daughters[j] = static_cast<int>(it_p->daughterRefVector().at(j).key());
      }
      gen_daughters_.push_back(daughters);
    }


    // associate gen particles to mothers
    genpart_mother_.resize(genpart_posz_.size(), -1);
    for (size_t i = 0; i < allselectedgentracks.size(); i++) {
      const auto tracki = allselectedgentracks.at(i);

      for (size_t j = i + 1; j < allselectedgentracks.size(); j++) {
        const auto trackj = allselectedgentracks.at(j);

        if (!tracki->noMother()) {
          if (&tracki->mother() == trackj) genpart_mother_.at(i) = j;
        }
        if (!trackj->noMother()) {
          if (&trackj->mother() == tracki) genpart_mother_.at(j) = i;
        }
      }
    }

    for(const auto& PVI : PupInfo)
    {
        if(PVI.getBunchCrossing() == 0)
        { 
            gen_PUNumInt_ = PVI.getPU_NumInteractions();
            gen_TrueNumInt_ = PVI.getTrueNumInteractions();
        }
    }

}


void
HGCalTriggerNtupleGen::
clear()
{

    gen_n_ = 0;
    gen_PUNumInt_ = 0;
    gen_TrueNumInt_ = 0.;

    vtx_x_ = 0;
    vtx_y_ = 0;
    vtx_z_ = 0;


    //
    genpart_eta_.clear();
    genpart_phi_.clear();
    genpart_pt_.clear();
    genpart_energy_.clear();
    genpart_dvx_.clear();
    genpart_dvy_.clear();
    genpart_dvz_.clear();
    genpart_ovx_.clear();
    genpart_ovy_.clear();
    genpart_ovz_.clear();
    genpart_exx_.clear();
    genpart_exy_.clear();
    genpart_mother_.clear();
    genpart_exphi_.clear();
    genpart_exeta_.clear();
    genpart_fbrem_.clear();
    genpart_pid_.clear();
    genpart_gen_.clear();
    genpart_reachedEE_.clear();
    genpart_fromBeamPipe_.clear();
    genpart_posx_.clear();
    genpart_posy_.clear();
    genpart_posz_.clear();

    ////////////////////
    // reco::GenParticles
    //
    gen_eta_.clear();
    gen_phi_.clear();
    gen_pt_.clear();
    gen_energy_.clear();
    gen_charge_.clear();
    gen_pdgid_.clear();
    gen_status_.clear();
    gen_daughters_.clear();
}
