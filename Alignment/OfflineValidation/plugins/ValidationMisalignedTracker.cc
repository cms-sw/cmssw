// -*- C++ -*-
//
// Package:    ValidationMisalignedTracker
// Class:      ValidationMisalignedTracker
//
/**\class ValidationMisalignedTracker ValidationMisalignedTracker.cc Alignment/OfflineValidation/src/ValidationMisalignedTracker.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Nicola De Filippis
//         Created:  Thu Dec 14 13:13:32 CET 2006
// $Id: ValidationMisalignedTracker.cc,v 1.8 2013/01/07 20:46:23 wmtan Exp $
//
//

#include "Alignment/OfflineValidation/plugins/ValidationMisalignedTracker.h"

// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/transform.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

//
// constructors and destructor
//
ValidationMisalignedTracker::ValidationMisalignedTracker(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      magFieldToken_(esConsumes()),
      selection_eff(iConfig.getUntrackedParameter<bool>("selection_eff", false)),
      selection_fake(iConfig.getUntrackedParameter<bool>("selection_fake", true)),
      ZmassSelection_(iConfig.getUntrackedParameter<bool>("ZmassSelection", false)),
      simobject(iConfig.getUntrackedParameter<std::string>("simobject", "g4SimHits")),
      trackassociator(iConfig.getUntrackedParameter<std::string>("TrackAssociator", "ByHits")),
      associators(iConfig.getParameter<std::vector<std::string> >("associators")),
      label(iConfig.getParameter<std::vector<edm::InputTag> >("label")),
      label_tp_effic(iConfig.getParameter<edm::InputTag>("label_tp_effic")),
      label_tp_fake(iConfig.getParameter<edm::InputTag>("label_tp_fake")),

      rootfile_(iConfig.getUntrackedParameter<std::string>("rootfile", "myroot.root")),
      evtToken_(consumes<edm::HepMCProduct>(edm::InputTag("source"))),
      tpeffToken_(consumes<TrackingParticleCollection>(label_tp_effic)),
      tpfakeToken_(consumes<TrackingParticleCollection>(label_tp_fake)),
      trackToken_(consumes<edm::View<reco::Track> >(label[0])),
      assocToken_{edm::vector_transform(
          label, [this](const edm::InputTag& lab) { return consumes<reco::TrackToTrackingParticleAssociator>(lab); })} {
  //now do what ever initialization is needed
  mzmu = 0., recmzmu = 0., ptzmu = 0., recptzmu = 0., etazmu = 0., recetazmu = 0., thetazmu = 0., recthetazmu = 0.,
  phizmu = 0., recphizmu = 0.;
  recenezmu = 0., enezmu = 0., pLzmu = 0., recpLzmu = 0., yzmu = 0., recyzmu = 0., mxptmu = 0., recmxptmu = 0.,
  minptmu = 0., recminptmu = 0.;
  // mzele=0.,recmzele=0.

  flag = 0, flagrec = 0, count = 0, countrec = 0;
  nAssoc = 0;

  for (int i = 0; i < 2; i++) {
    countpart[i] = 0;
    countpartrec[i] = 0;
    for (int j = 0; j < 2; j++) {
      ene[i][j] = 0.;
      p[i][j] = 0.;
      px[i][j] = 0.;
      py[i][j] = 0.;
      pz[i][j] = 0.;
      ptmu[i][j] = 0.;
      recene[i][j] = 0.;
      recp[i][j] = 0.;
      recpx[i][j] = 0.;
      recpy[i][j] = 0.;
      recpz[i][j] = 0.;
      recptmu[i][j] = 0.;
    }
  }

  eventCount_ = 0;

  file_ = new TFile(rootfile_.c_str(), "RECREATE");

  // initialize the tree
  tree_eff = new TTree("EffTracks", "Efficiency Tracks Tree");

  tree_eff->Branch("Run", &irun, "irun/i");
  tree_eff->Branch("Event", &ievt, "ievt/i");

  // SimTrack
  tree_eff->Branch("TrackID", &trackType, "trackType/i");
  tree_eff->Branch("pt", &pt, "pt/F");
  tree_eff->Branch("eta", &eta, "eta/F");
  tree_eff->Branch("CotTheta", &cottheta, "cottheta/F");
  tree_eff->Branch("phi", &phi, "phi/F");
  tree_eff->Branch("d0", &d0, "d0/F");
  tree_eff->Branch("z0", &z0, "z0/F");
  tree_eff->Branch("nhit", &nhit, "nhit/i");

  // RecTrack
  tree_eff->Branch("recpt", &recpt, "recpt/F");
  tree_eff->Branch("receta", &receta, "receta/F");
  tree_eff->Branch("CotRecTheta", &reccottheta, "reccottheta/F");
  tree_eff->Branch("recphi", &recphi, "recphi/F");
  tree_eff->Branch("recd0", &recd0, "recd0/F");
  tree_eff->Branch("recz0", &recz0, "recz0/F");
  tree_eff->Branch("nAssoc", &nAssoc, "nAssoc/i");
  tree_eff->Branch("recnhit", &recnhit, "recnhit/i");
  tree_eff->Branch("CHISQ", &recchiq, "recchiq/F");

  tree_eff->Branch("reseta", &reseta, "reseta/F");
  tree_eff->Branch("respt", &respt, "respt/F");
  tree_eff->Branch("resd0", &resd0, "resd0/F");
  tree_eff->Branch("resz0", &resz0, "resz0/F");
  tree_eff->Branch("resphi", &resphi, "resphi/F");
  tree_eff->Branch("rescottheta", &rescottheta, "rescottheta/F");
  tree_eff->Branch("eff", &eff, "eff/F");

  // Invariant masses, pt of Z
  tree_eff->Branch("mzmu", &mzmu, "mzmu/F");
  tree_eff->Branch("ptzmu", &ptzmu, "ptzmu/F");
  tree_eff->Branch("pLzmu", &pLzmu, "pLzmu/F");
  tree_eff->Branch("enezmu", &enezmu, "enezmu/F");
  tree_eff->Branch("etazmu", &etazmu, "etazmu/F");
  tree_eff->Branch("thetazmu", &thetazmu, "thetazmu/F");
  tree_eff->Branch("phizmu", &phizmu, "phizmu/F");
  tree_eff->Branch("yzmu", &yzmu, "yzmu/F");
  tree_eff->Branch("mxptmu", &mxptmu, "mxptmu/F");
  tree_eff->Branch("minptmu", &minptmu, "minptmu/F");

  tree_eff->Branch("recmzmu", &recmzmu, "recmzmu/F");
  tree_eff->Branch("recptzmu", &recptzmu, "recptzmu/F");
  tree_eff->Branch("recpLzmu", &recpLzmu, "recpLzmu/F");
  tree_eff->Branch("recenezmu", &recenezmu, "recenezmu/F");
  tree_eff->Branch("recetazmu", &recetazmu, "recetazmu/F");
  tree_eff->Branch("recthetazmu", &recthetazmu, "recthetazmu/F");
  tree_eff->Branch("recphizmu", &recphizmu, "recphizmu/F");
  tree_eff->Branch("recyzmu", &recyzmu, "recyzmu/F");
  tree_eff->Branch("recmxptmu", &recmxptmu, "recmxptmu/F");
  tree_eff->Branch("recminptmu", &recminptmu, "recminptmu/F");

  //tree->Branch("mzele",&ntmzele,"ntmzele/F");
  //tree->Branch("recmzele",&ntmzeleRec,"ntmzeleRec/F");
  tree_eff->Branch("chi2Associator", &recchiq, "recchiq/F");

  // Fake

  tree_fake = new TTree("FakeTracks", "Fake Rate Tracks Tree");

  tree_fake->Branch("Run", &irun, "irun/i");
  tree_fake->Branch("Event", &ievt, "ievt/i");

  // SimTrack
  tree_fake->Branch("fakeTrackID", &faketrackType, "faketrackType/i");
  tree_fake->Branch("fakept", &fakept, "fakept/F");
  tree_fake->Branch("fakeeta", &fakeeta, "fakeeta/F");
  tree_fake->Branch("fakeCotTheta", &fakecottheta, "fakecottheta/F");
  tree_fake->Branch("fakephi", &fakephi, "fakephi/F");
  tree_fake->Branch("faked0", &faked0, "faked0/F");
  tree_fake->Branch("fakez0", &fakez0, "fakez0/F");
  tree_fake->Branch("fakenhit", &fakenhit, "fakenhit/i");

  // RecTrack
  tree_fake->Branch("fakerecpt", &fakerecpt, "fakerecpt/F");
  tree_fake->Branch("fakereceta", &fakereceta, "fakereceta/F");
  tree_fake->Branch("fakeCotRecTheta", &fakereccottheta, "fakereccottheta/F");
  tree_fake->Branch("fakerecphi", &fakerecphi, "fakerecphi/F");
  tree_fake->Branch("fakerecd0", &fakerecd0, "fakerecd0/F");
  tree_fake->Branch("fakerecz0", &fakerecz0, "fakerecz0/F");
  tree_fake->Branch("fakenAssoc", &fakenAssoc, "fakenAssoc/i");
  tree_fake->Branch("fakerecnhit", &fakerecnhit, "fakerecnhit/i");
  tree_fake->Branch("fakeCHISQ", &fakerecchiq, "fakerecchiq/F");

  tree_fake->Branch("fakereseta", &fakereseta, "fakereseta/F");
  tree_fake->Branch("fakerespt", &fakerespt, "fakerespt/F");
  tree_fake->Branch("fakeresd0", &fakeresd0, "fakeresd0/F");
  tree_fake->Branch("fakeresz0", &fakeresz0, "fakeresz0/F");
  tree_fake->Branch("fakeresphi", &fakeresphi, "fakeresphi/F");
  tree_fake->Branch("fakerescottheta", &fakerescottheta, "fakerescottheta/F");
  tree_fake->Branch("fake", &fake, "fake/F");

  // Invariant masses, pt of Z
  tree_fake->Branch("fakemzmu", &fakemzmu, "fakemzmu/F");
  tree_fake->Branch("fakeptzmu", &fakeptzmu, "fakeptzmu/F");
  tree_fake->Branch("fakepLzmu", &fakepLzmu, "fakepLzmu/F");
  tree_fake->Branch("fakeenezmu", &fakeenezmu, "fakeenezmu/F");
  tree_fake->Branch("fakeetazmu", &fakeetazmu, "fakeetazmu/F");
  tree_fake->Branch("fakethetazmu", &fakethetazmu, "fakethetazmu/F");
  tree_fake->Branch("fakephizmu", &fakephizmu, "fakephizmu/F");
  tree_fake->Branch("fakeyzmu", &fakeyzmu, "fakeyzmu/F");
  tree_fake->Branch("fakemxptmu", &fakemxptmu, "fakemxptmu/F");
  tree_fake->Branch("fakeminptmu", &fakeminptmu, "fakeminptmu/F");

  tree_fake->Branch("fakerecmzmu", &fakerecmzmu, "fakerecmzmu/F");
  tree_fake->Branch("fakerecptzmu", &fakerecptzmu, "fakerecptzmu/F");
  tree_fake->Branch("fakerecpLzmu", &fakerecpLzmu, "fakerecpLzmu/F");
  tree_fake->Branch("fakerecenezmu", &fakerecenezmu, "fakerecenezmu/F");
  tree_fake->Branch("fakerecetazmu", &fakerecetazmu, "fakerecetazmu/F");
  tree_fake->Branch("fakerecthetazmu", &fakerecthetazmu, "fakerecthetazmu/F");
  tree_fake->Branch("fakerecphizmu", &fakerecphizmu, "fakerecphizmu/F");
  tree_fake->Branch("fakerecyzmu", &fakerecyzmu, "fakerecyzmu/F");
  tree_fake->Branch("fakerecmxptmu", &fakerecmxptmu, "fakerecmxptmu/F");
  tree_fake->Branch("fakerecminptmu", &fakerecminptmu, "fakerecminptmu/F");

  tree_fake->Branch("fakechi2Associator", &fakerecchiq, "fakerecchiq/F");
}

ValidationMisalignedTracker::~ValidationMisalignedTracker() {
  edm::LogVerbatim("ValidationMisalignedTracker")
      << "ValidationMisalignedTracker::endJob Processed " << eventCount_ << " events";

  // store the tree in the output file
  file_->Write();

  // Closing the file deletes the tree.
  file_->Close();
  tree_eff = nullptr;
  tree_fake = nullptr;
}

//
// member functions
//
void ValidationMisalignedTracker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag> >("label", {});
  desc.add<edm::InputTag>("label_tp_effic", edm::InputTag("FinalTrackRefitter"));
  desc.add<edm::InputTag>("label_tp_fake", edm::InputTag("TrackRefitter"));
  desc.add<std::vector<std::string> >("associators", {});
  desc.addUntracked<bool>("selection_eff", false);
  desc.addUntracked<bool>("selection_fake", true);
  desc.addUntracked<bool>("ZmassSelection", false);
  desc.addUntracked<std::string>("simobject", "g4SimHits");
  desc.addUntracked<std::string>("TrackAssociator", "ByHits");
  desc.addUntracked<std::string>("rootfile", "myroot.root");
  descriptions.add("validationMisAlignedTracker", desc);
}

// ------------ method called to for each event  ------------
void ValidationMisalignedTracker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<const reco::TrackToTrackingParticleAssociator*> associatore;

  {
    for (unsigned int w = 0; w < associators.size(); w++) {
      const edm::Handle<reco::TrackToTrackingParticleAssociator>& theAssociator = iEvent.getHandle(assocToken_[w]);
      associatore.push_back(theAssociator.product());
    }
  }

  edm::LogInfo("Tracker Misalignment Validation") << "\n Starting!";

  // Monte Carlo Z selection
  skip = false;
  std::vector<int> indmu;

  if (selection_eff && ZmassSelection_) {
    const edm::Handle<edm::HepMCProduct>& evt = iEvent.getHandle(evtToken_);
    bool accepted = false;
    bool foundmuons = false;
    HepMC::GenEvent* myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

    for (HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {
      if (!accepted && ((*p)->pdg_id() == 23) && (*p)->status() == 3) {
        accepted = true;
        for (HepMC::GenVertex::particle_iterator aDaughter = (*p)->end_vertex()->particles_begin(HepMC::descendants);
             aDaughter != (*p)->end_vertex()->particles_end(HepMC::descendants);
             aDaughter++) {
          if (abs((*aDaughter)->pdg_id()) == 13) {
            foundmuons = true;
            if ((*aDaughter)->status() != 1) {
              for (HepMC::GenVertex::particle_iterator byaDaughter =
                       (*aDaughter)->end_vertex()->particles_begin(HepMC::descendants);
                   byaDaughter != (*aDaughter)->end_vertex()->particles_end(HepMC::descendants);
                   byaDaughter++) {
                if ((*byaDaughter)->status() == 1 && abs((*byaDaughter)->pdg_id()) == 13) {
                  indmu.push_back((*byaDaughter)->barcode());
                  edm::LogVerbatim("ValidationMisalignedTracker")
                      << "Stable muon from Z with charge " << (*byaDaughter)->pdg_id() << " and index "
                      << (*byaDaughter)->barcode();
                }
              }
            } else {
              indmu.push_back((*aDaughter)->barcode());
              edm::LogVerbatim("ValidationMisalignedTracker")
                  << "Stable muon from Z with charge " << (*aDaughter)->pdg_id() << " and index "
                  << (*aDaughter)->barcode();
            }
          }
        }
        if (!foundmuons) {
          edm::LogVerbatim("ValidationMisalignedTracker") << "No muons from Z ...skip event";
          skip = true;
        }
      }
    }
    if (!accepted) {
      edm::LogVerbatim("ValidationMisalignedTracker") << "No Z particles in the event ...skip event";
      skip = true;
    }
  } else {
    skip = false;
  }

  //
  // Retrieve tracker geometry from event setup
  //
  const TrackerGeometry* trackerGeometry = &iSetup.getData(geomToken_);
  auto testGeomDet = trackerGeometry->detsTOB().front();
  edm::LogVerbatim("ValidationMisalignedTracker") << testGeomDet->position();

  //Dump Run and Event
  irun = iEvent.id().run();
  ievt = iEvent.id().event();

  // Reset tree variables
  int countpart[2] = {0, 0}, countpartrec[2] = {0, 0}, flag = 0, flagrec = 0, count = 0, countrec = 0;
  //int countsim=0;
  float ene[2][2], px[2][2], py[2][2], pz[2][2], ptmu[2][2];
  float recene[2][2], recp[2][2], recpx[2][2], recpy[2][2], recpz[2][2], recptmu[2][2];

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      ene[i][j] = 0.;
      px[i][j] = 0.;
      py[i][j] = 0.;
      pz[i][j] = 0.;
      ptmu[i][j] = 0.;
      recene[i][j] = 0.;
      recp[i][j] = 0.;
      recpx[i][j] = 0.;
      recpy[i][j] = 0.;
      recpz[i][j] = 0.;
      recptmu[i][j] = 0.;
    }
  }

  const edm::Handle<TrackingParticleCollection>& TPCollectionHeff = iEvent.getHandle(tpeffToken_);
  const TrackingParticleCollection tPCeff = *(TPCollectionHeff.product());

  const edm::Handle<TrackingParticleCollection>& TPCollectionHfake = iEvent.getHandle(tpfakeToken_);
  const TrackingParticleCollection tPCfake = *(TPCollectionHfake.product());

  for (unsigned int ww = 0; ww < associators.size(); ww++) {
    //
    //get collections from the event
    //

    const edm::Handle<edm::View<reco::Track> >& trackCollection = iEvent.getHandle(trackToken_);
    const edm::View<reco::Track> tC = *(trackCollection.product());

    //associate tracks
    LogTrace("TrackValidator") << "Calling associateRecoToSim method\n";
    reco::RecoToSimCollection recSimColl = associatore[ww]->associateRecoToSim(trackCollection, TPCollectionHfake);

    LogTrace("TrackValidator") << "Calling associateSimToReco method\n";
    reco::SimToRecoCollection simRecColl = associatore[ww]->associateSimToReco(trackCollection, TPCollectionHeff);

    //
    //compute number of tracks per eta interval
    //

    if (selection_eff && !skip) {
      edm::LogVerbatim("ValidationMisalignedTracker") << "Computing Efficiency";

      edm::LogVerbatim("TrackValidator") << "\n# of TrackingParticles (before cuts): " << tPCeff.size() << "\n";
      int st = 0;
      for (TrackingParticleCollection::size_type i = 0; i < tPCeff.size(); i++) {
        // Initialize variables
        eta = 0., theta = 0., phi = 0., pt = 0., cottheta = 0., costheta = 0.;
        d0 = 0., z0 = 0.;
        nhit = 0;
        receta = 0., rectheta = 0., recphi = 0., recpt = 0., reccottheta = 0., recd0 = 0., recz0 = 0.;
        respt = 0., resd0 = 0., resz0 = 0., reseta = 0., resphi = 0., rescottheta = 0.;
        recchiq = 0.;
        recnhit = 0;
        trackType = 0;
        eff = 0;

        // typedef edm::Ref<TrackingParticleCollection> TrackingParticleRef;
        TrackingParticleRef tp(TPCollectionHeff, i);
        if (tp->charge() == 0)
          continue;
        st++;
        //pt=sqrt(tp->momentum().perp2());
        //eta=tp->momentum().eta();
        //vpos=tp->vertex().perp2()));

        const SimTrack* simulatedTrack = &(*tp->g4Track_begin());

        const MagneticField* theMF = &iSetup.getData(magFieldToken_);
        FreeTrajectoryState ftsAtProduction(
            GlobalPoint(tp->vertex().x(), tp->vertex().y(), tp->vertex().z()),
            GlobalVector(
                simulatedTrack->momentum().x(), simulatedTrack->momentum().y(), simulatedTrack->momentum().z()),
            TrackCharge(tp->charge()),
            theMF);
        TSCPBuilderNoMaterial tscpBuilder;
        TrajectoryStateClosestToPoint tsAtClosestApproach =
            tscpBuilder(ftsAtProduction, GlobalPoint(0, 0, 0));  //as in TrackProducerAlgorithm
        GlobalPoint v = tsAtClosestApproach.theState().position();
        GlobalVector p = tsAtClosestApproach.theState().momentum();

        //  double qoverpSim = tsAtClosestApproach.charge()/p.mag();
        //  double lambdaSim = M_PI/2-p.theta();
        //  double phiSim    = p.phi();
        double dxySim = (-v.x() * sin(p.phi()) + v.y() * cos(p.phi()));
        double dszSim = v.z() * p.perp() / p.mag() - (v.x() * p.x() + v.y() * p.y()) / p.perp() * p.z() / p.mag();
        d0 = float(-dxySim);
        z0 = float(dszSim * p.mag() / p.perp());

        if (abs(simulatedTrack->type()) == 13 && simulatedTrack->genpartIndex() != -1) {
          edm::LogVerbatim("ValidationMisalignedTracker") << " TRACCIA SIM DI MUONI ";
          edm::LogVerbatim("ValidationMisalignedTracker") << "Gen part " << simulatedTrack->genpartIndex();
          trackType = simulatedTrack->type();
          theta = simulatedTrack->momentum().theta();
          costheta = cos(theta);
          cottheta = 1. / tan(theta);

          eta = simulatedTrack->momentum().eta();
          phi = simulatedTrack->momentum().phi();
          pt = simulatedTrack->momentum().pt();
          nhit = tp->matchedHit();

          edm::LogVerbatim("ValidationMisalignedTracker")
              << "3) Before assoc: SimTrack of type = " << simulatedTrack->type() << " ,at eta = " << eta
              << " ,with pt at vertex = " << simulatedTrack->momentum().pt() << " GeV/c"
              << " ,d0 =" << d0 << " ,z0 =" << z0 << " ,nhit=" << nhit;

          if (ZmassSelection_) {
            if (abs(trackType) == 13 &&
                (simulatedTrack->genpartIndex() == indmu[0] || simulatedTrack->genpartIndex() == indmu[1])) {
              edm::LogVerbatim("ValidationMisalignedTracker") << " TRACK sim of muons from Z ";
              flag = 0;
              count = countpart[0];
              countpart[0]++;
            } else if (abs(trackType) == 11) {
              //edm::LogVerbatim("ValidationMisalignedTracker") << " TRACCIA SIM DI ELETTRONI ";
              flag = 1;
              count = countpart[1];
              countpart[1]++;
            }

            px[flag][count] = simulatedTrack->momentum().x();
            py[flag][count] = simulatedTrack->momentum().y();
            pz[flag][count] = simulatedTrack->momentum().z();
            ptmu[flag][count] = simulatedTrack->momentum().pt();
            ene[flag][count] = simulatedTrack->momentum().e();
          }

          std::vector<std::pair<edm::RefToBase<reco::Track>, double> > rt;
          if (simRecColl.find(tp) != simRecColl.end()) {
            rt = simRecColl[tp];
            if (!rt.empty()) {
              edm::RefToBase<reco::Track> t = rt.begin()->first;

              // bool flagptused=false;
              // for (unsigned int j=0;j<ptused.size();j++){
              //   if (fabs(t->pt()-ptused[j])<0.001) {
              //     flagptused=true;
              //   }
              // }

              edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st << " with pt=" << t->pt()
                                                 << " associated with quality:" << rt.begin()->second << "\n";
              edm::LogVerbatim("ValidationMisalignedTracker") << "Reconstructed Track:" << t->pt();
              edm::LogVerbatim("ValidationMisalignedTracker") << "\tpT: " << t->pt();
              edm::LogVerbatim("ValidationMisalignedTracker") << "\timpact parameter:d0: " << t->d0();
              edm::LogVerbatim("ValidationMisalignedTracker") << "\timpact parameter:z0: " << t->dz();
              edm::LogVerbatim("ValidationMisalignedTracker")
                  << "\tAzimuthal angle of point of closest approach:" << t->phi();
              edm::LogVerbatim("ValidationMisalignedTracker") << "\tcharge: " << t->charge();
              edm::LogVerbatim("ValidationMisalignedTracker") << "\teta: " << t->eta();
              edm::LogVerbatim("ValidationMisalignedTracker") << "\tnormalizedChi2: " << t->normalizedChi2();

              recnhit = t->numberOfValidHits();
              recchiq = t->normalizedChi2();
              rectheta = t->theta();
              reccottheta = 1. / tan(rectheta);
              //receta=-log(tan(rectheta/2.));
              receta = t->momentum().eta();
              //	   reccostheta=cos(matchedrectrack->momentum().theta());
              recphi = t->phi();
              recpt = t->pt();
              ptused.push_back(recpt);
              recd0 = t->d0();
              recz0 = t->dz();

              edm::LogVerbatim("ValidationMisalignedTracker")
                  << "5) After call to associator: the best match has " << recnhit << " hits, Chi2 = " << recchiq
                  << ", pt at vertex = " << recpt << " GeV/c, "
                  << ", recd0 = " << recd0 << ", recz0= " << recz0;

              respt = recpt - pt;
              resd0 = recd0 - d0;
              resz0 = recz0 - z0;
              reseta = receta - eta;
              resphi = recphi - phi;
              rescottheta = reccottheta - cottheta;
              eff = 1;

              edm::LogVerbatim("ValidationMisalignedTracker")
                  << "6) Transverse momentum residual=" << respt << " ,d0 residual=" << resd0
                  << " ,z0 residual=" << resz0 << " with eff=" << eff;

              if (ZmassSelection_) {
                if (abs(trackType) == 13) {
                  edm::LogVerbatim("ValidationMisalignedTracker") << " TRACCIA RECO DI MUONI ";
                  flagrec = 0;
                  countrec = countpartrec[0];
                  countpartrec[0]++;
                } else if (abs(trackType) == 11) {
                  edm::LogVerbatim("ValidationMisalignedTracker") << " TRACCIA RECO DI ELETTRONI ";
                  flagrec = 1;
                  countrec = countpartrec[1];
                  countpartrec[1]++;
                }

                recp[flagrec][countrec] = sqrt(t->momentum().mag2());
                recpx[flagrec][countrec] = t->momentum().x();
                recpy[flagrec][countrec] = t->momentum().y();
                recpz[flagrec][countrec] = t->momentum().z();
                recptmu[flagrec][countrec] =
                    sqrt((t->momentum().x() * t->momentum().x()) + (t->momentum().y() * t->momentum().y()));
                if (abs(trackType) == 13)
                  recene[flagrec][countrec] = sqrt(recp[flagrec][countrec] * recp[flagrec][countrec] + 0.105 * 0.105);
                if (abs(trackType) == 11)
                  recene[flagrec][countrec] = sqrt(recp[flagrec][countrec] * recp[flagrec][countrec] + 0.0005 * 0.0005);
              }

              edm::LogVerbatim("ValidationMisalignedTracker") << "7) Transverse momentum reconstructed =" << recpt
                                                              << " at  eta= " << receta << " and phi= " << recphi;
            }
          } else {
            edm::LogVerbatim("TrackValidator")
                << "TrackingParticle #" << st << " with pt=" << sqrt(tp->momentum().perp2())
                << " NOT associated to any reco::Track"
                << "\n";
            receta = -100.;
            recphi = -100.;
            recpt = -100.;
            recd0 = -100.;
            recz0 = -100;
            respt = -100.;
            resd0 = -100.;
            resz0 = -100.;
            resphi = -100.;
            reseta = -100.;
            rescottheta = -100.;
            recnhit = 100;
            recchiq = -100;
            eff = 0;
            flagrec = 100;
          }

          edm::LogVerbatim("ValidationMisalignedTracker") << "Eff=" << eff;

          // simulated muons

          edm::LogVerbatim("ValidationMisalignedTracker") << "Flag is" << flag;
          edm::LogVerbatim("ValidationMisalignedTracker") << "RecFlag is" << flagrec;

          if (countpart[0] == 2 && flag == 0) {
            mzmu =
                sqrt((ene[0][0] + ene[0][1]) * (ene[0][0] + ene[0][1]) - (px[0][0] + px[0][1]) * (px[0][0] + px[0][1]) -
                     (py[0][0] + py[0][1]) * (py[0][0] + py[0][1]) - (pz[0][0] + pz[0][1]) * (pz[0][0] + pz[0][1]));
            edm::LogVerbatim("ValidationMisalignedTracker") << "Mzmu " << mzmu;
            ptzmu = sqrt((px[0][0] + px[0][1]) * (px[0][0] + px[0][1]) + (py[0][0] + py[0][1]) * (py[0][0] + py[0][1]));

            pLzmu = pz[0][0] + pz[0][1];
            enezmu = ene[0][0] + ene[0][1];
            phizmu = atan2((py[0][0] + py[0][1]), (px[0][0] + px[0][1]));
            thetazmu = atan2(ptzmu, (pz[0][0] + pz[0][1]));
            etazmu = -log(tan(thetazmu * 3.14 / 360.));
            yzmu = 0.5 * log((enezmu + pLzmu) / (enezmu - pLzmu));
            mxptmu = std::max(ptmu[0][0], ptmu[0][1]);
            minptmu = std::min(ptmu[0][0], ptmu[0][1]);
          } else {
            mzmu = -100.;
            ptzmu = -100.;
            pLzmu = -100.;
            enezmu = -100.;
            etazmu = -100.;
            phizmu = -100.;
            thetazmu = -100.;
            yzmu = -100.;
            mxptmu = -100.;
            minptmu = -100.;
          }

          // reconstructed muons
          if (countpartrec[0] == 2 && flagrec == 0) {
            recmzmu = sqrt((recene[0][0] + recene[0][1]) * (recene[0][0] + recene[0][1]) -
                           (recpx[0][0] + recpx[0][1]) * (recpx[0][0] + recpx[0][1]) -
                           (recpy[0][0] + recpy[0][1]) * (recpy[0][0] + recpy[0][1]) -
                           (recpz[0][0] + recpz[0][1]) * (recpz[0][0] + recpz[0][1]));
            edm::LogVerbatim("ValidationMisalignedTracker") << "RecMzmu " << recmzmu;
            recptzmu = sqrt((recpx[0][0] + recpx[0][1]) * (recpx[0][0] + recpx[0][1]) +
                            (recpy[0][0] + recpy[0][1]) * (recpy[0][0] + recpy[0][1]));

            recpLzmu = recpz[0][0] + recpz[0][1];
            recenezmu = recene[0][0] + recene[0][1];
            recphizmu = atan2((recpy[0][0] + recpy[0][1]), (recpx[0][0] + recpx[0][1]));
            recthetazmu = atan2(recptzmu, (recpz[0][0] + recpz[0][1]));
            recetazmu = -log(tan(recthetazmu * 3.14 / 360.));
            recyzmu = 0.5 * log((recenezmu + recpLzmu) / (recenezmu - recpLzmu));
            recmxptmu = std::max(recptmu[0][0], recptmu[0][1]);
            recminptmu = std::min(recptmu[0][0], recptmu[0][1]);
          } else {
            recmzmu = -100.;
            recptzmu = -100.;
            recpLzmu = -100.;
            recenezmu = -100.;
            recetazmu = -100.;
            recphizmu = -100.;
            recthetazmu = -100.;
            recyzmu = -100.;
            recmxptmu = -100;
            recminptmu = -100.;
          }

          tree_eff->Fill();

        }  // end of loop on muons
      }    // end of loop for tracking particle
    }      // end of loop for efficiency

    //
    // Fake Rate
    //
    if (selection_fake) {
      edm::LogVerbatim("ValidationMisalignedTracker") << "Computing Fake Rate";

      fakeeta = 0., faketheta = 0., fakephi = 0., fakept = 0., fakecottheta = 0., fakecostheta = 0.;
      faked0 = 0., fakez0 = 0.;
      fakenhit = 0;
      fakereceta = 0., fakerectheta = 0., fakerecphi = 0., fakerecpt = 0., fakereccottheta = 0., fakerecd0 = 0.,
      fakerecz0 = 0.;
      fakerespt = 0., fakeresd0 = 0., fakeresz0 = 0., fakereseta = 0., fakeresphi = 0., fakerescottheta = 0.;
      fakerecchiq = 0.;
      fakerecnhit = 0;
      faketrackType = 0;
      fake = 0;

      //      int at=0;
      int rT = 0;
      for (reco::TrackCollection::size_type i = 0; i < tC.size(); ++i) {
        edm::RefToBase<reco::Track> track(trackCollection, i);
        rT++;

        fakeeta = 0., faketheta = 0., fakephi = 0., fakept = 0., fakecottheta = 0., fakecostheta = 0.;
        faked0 = 0., fakez0 = 0.;
        fakenhit = 0;
        fakereceta = 0., fakerectheta = 0., fakerecphi = 0., fakerecpt = 0., fakereccottheta = 0., fakerecd0 = 0.,
        fakerecz0 = 0.;
        fakerespt = 0., fakeresd0 = 0., fakeresz0 = 0., fakereseta = 0., fakeresphi = 0., fakerescottheta = 0.;
        fakerecchiq = 0.;
        fakerecnhit = 0;
        faketrackType = 0;
        fake = 0;

        fakerecnhit = track->numberOfValidHits();
        fakerecchiq = track->normalizedChi2();
        fakerectheta = track->theta();
        fakereccottheta = 1. / tan(rectheta);
        //fakereceta=-log(tan(rectheta/2.));
        fakereceta = track->momentum().eta();
        //	   fakereccostheta=cos(track->momentum().theta());
        fakerecphi = track->phi();
        fakerecpt = track->pt();
        fakerecd0 = track->d0();
        fakerecz0 = track->dz();

        edm::LogVerbatim("ValidationMisalignedTracker") << "1) Before assoc: TkRecTrack at eta = " << fakereceta;
        edm::LogVerbatim("ValidationMisalignedTracker") << "Track number " << i;
        edm::LogVerbatim("ValidationMisalignedTracker") << "\tPT: " << track->pt();
        edm::LogVerbatim("ValidationMisalignedTracker") << "\timpact parameter:d0: " << track->d0();
        edm::LogVerbatim("ValidationMisalignedTracker") << "\timpact parameter:z0: " << track->dz();
        edm::LogVerbatim("ValidationMisalignedTracker")
            << "\tAzimuthal angle of point of closest approach:" << track->phi();
        edm::LogVerbatim("ValidationMisalignedTracker") << "\tcharge: " << track->charge();
        edm::LogVerbatim("ValidationMisalignedTracker") << "\teta: " << track->eta();
        edm::LogVerbatim("ValidationMisalignedTracker") << "\tnormalizedChi2: " << track->normalizedChi2();

        std::vector<std::pair<TrackingParticleRef, double> > tp;

        //Compute fake rate vs eta
        if (recSimColl.find(track) != recSimColl.end()) {
          tp = recSimColl[track];
          if (!tp.empty()) {
            edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
                                               << " associated with quality:" << tp.begin()->second << "\n";

            TrackingParticleRef tpr = tp.begin()->first;
            const SimTrack* fakeassocTrack = &(*tpr->g4Track_begin());

            const MagneticField* theMF = &iSetup.getData(magFieldToken_);
            FreeTrajectoryState ftsAtProduction(
                GlobalPoint(tpr->vertex().x(), tpr->vertex().y(), tpr->vertex().z()),
                GlobalVector(
                    fakeassocTrack->momentum().x(), fakeassocTrack->momentum().y(), fakeassocTrack->momentum().z()),
                TrackCharge(tpr->charge()),
                theMF);
            TSCPBuilderNoMaterial tscpBuilder;
            TrajectoryStateClosestToPoint tsAtClosestApproach =
                tscpBuilder(ftsAtProduction, GlobalPoint(0, 0, 0));  //as in TrackProducerAlgorithm
            GlobalPoint v = tsAtClosestApproach.theState().position();
            GlobalVector p = tsAtClosestApproach.theState().momentum();

            //  double qoverpSim = tsAtClosestApproach.charge()/p.mag();
            //  double lambdaSim = M_PI/2-p.theta();
            //  double phiSim    = p.phi();
            double dxySim = (-v.x() * sin(p.phi()) + v.y() * cos(p.phi()));
            double dszSim = v.z() * p.perp() / p.mag() - (v.x() * p.x() + v.y() * p.y()) / p.perp() * p.z() / p.mag();
            faked0 = float(-dxySim);
            fakez0 = float(dszSim * p.mag() / p.perp());

            faketrackType = fakeassocTrack->type();
            faketheta = fakeassocTrack->momentum().theta();
            fakecottheta = 1. / tan(faketheta);
            fakeeta = fakeassocTrack->momentum().eta();
            fakephi = fakeassocTrack->momentum().phi();
            fakept = fakeassocTrack->momentum().pt();
            fakenhit = tpr->matchedHit();

            edm::LogVerbatim("ValidationMisalignedTracker")
                << "4) After call to associator: the best SimTrack match is of type" << fakeassocTrack->type()
                << " ,at eta = " << fakeeta << " and phi = " << fakephi << " ,with pt at vertex = " << fakept
                << " GeV/c"
                << " ,d0 global = " << faked0 << " ,z0 = " << fakez0;
            fake = 1;

            fakerespt = fakerecpt - fakept;
            fakeresd0 = fakerecd0 - faked0;
            fakeresz0 = fakerecz0 - fakez0;
            fakereseta = -log(tan(fakerectheta / 2.)) - (-log(tan(faketheta / 2.)));
            fakeresphi = fakerecphi - fakephi;
            fakerescottheta = fakereccottheta - fakecottheta;
          }
        } else {
          edm::LogVerbatim("TrackValidator")
              << "reco::Track #" << rT << " with pt=" << track->pt() << " NOT associated to any TrackingParticle"
              << "\n";

          fakeeta = -100.;
          faketheta = -100;
          fakephi = -100.;
          fakept = -100.;
          faked0 = -100.;
          fakez0 = -100;
          fakerespt = -100.;
          fakeresd0 = -100.;
          fakeresz0 = -100.;
          fakeresphi = -100.;
          fakereseta = -100.;
          fakerescottheta = -100.;
          fakenhit = 100;
          fake = 0;
        }

        tree_fake->Fill();
      }

    }  // End of loop on fakerate

  }  // End of loop on associators
}

// ------------ method called once each job just after ending the event loop  ------------
void ValidationMisalignedTracker::endJob() {
  edm::LogVerbatim("ValidationMisalignedTracker") << "\t Misalignment analysis completed \n";
}

DEFINE_FWK_MODULE(ValidationMisalignedTracker);
