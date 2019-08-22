#include <ostream>
#include "IOMC/ParticleGuns/interface/BeamMomentumGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

#include "TFile.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  BeamMomentumGunProducer::BeamMomentumGunProducer(const edm::ParameterSet& pset)
      : FlatBaseThetaGunProducer(pset),
        parPDGId_(nullptr),
        parX_(nullptr),
        parY_(nullptr),
        parZ_(nullptr),
        parPx_(nullptr),
        parPy_(nullptr),
        parPz_(nullptr),
        b_npar_(nullptr),
        b_eventId_(nullptr),
        b_parPDGId_(nullptr),
        b_parX_(nullptr),
        b_parY_(nullptr),
        b_parZ_(nullptr),
        b_parPx_(nullptr),
        b_parPy_(nullptr),
        b_parPz_(nullptr) {
    edm::ParameterSet pgun_params = pset.getParameter<edm::ParameterSet>("PGunParameters");

    // doesn't seem necessary to check if pset is empty
    xoff_ = pgun_params.getParameter<double>("XOffset");
    yoff_ = pgun_params.getParameter<double>("YOffset");
    zpos_ = pgun_params.getParameter<double>("ZPosition");
    if (fVerbosity > 0)
      edm::LogVerbatim("BeamMomentumGun") << "Beam vertex offset (cm) " << xoff_ << ":" << yoff_ << " and z position "
					  << zpos_;

    edm::FileInPath fp = pgun_params.getParameter<edm::FileInPath>("FileName");
    std::string infileName = fp.fullPath();

    fFile_ = new TFile(infileName.c_str());
    fFile_->GetObject("EventTree", fTree_);
    nentries_ = fTree_->GetEntriesFast();
    if (fVerbosity > 0)
      edm::LogVerbatim("BeamMomentumGun") << "Total Events: " << nentries_ << " in " << infileName;

    // Set branch addresses and branch pointers
    int npart = fTree_->SetBranchAddress("npar", &npar_, &b_npar_);
    int event = fTree_->SetBranchAddress("eventId", &eventId_, &b_eventId_);
    int pdgid = fTree_->SetBranchAddress("parPDGId", &parPDGId_, &b_parPDGId_);
    int parxx = fTree_->SetBranchAddress("parX", &parX_, &b_parX_);
    int paryy = fTree_->SetBranchAddress("parY", &parY_, &b_parY_);
    int parzz = fTree_->SetBranchAddress("parZ", &parZ_, &b_parZ_);
    int parpx = fTree_->SetBranchAddress("parPx", &parPx_, &b_parPx_);
    int parpy = fTree_->SetBranchAddress("parPy", &parPy_, &b_parPy_);
    int parpz = fTree_->SetBranchAddress("parPz", &parPz_, &b_parPz_);
    if ((npart != 0) || (event != 0) || (pdgid != 0) || (parxx != 0) || (paryy != 0) ||
	(parzz != 0) || (parpx != 0) || (parpy != 0) || (parpz != 0))
      throw cms::Exception("GenException") << "Branch address wrong in i/p file\n";

    produces<HepMCProduct>("unsmeared");
    produces<GenEventInfoProduct>();

    if (fVerbosity > 0)
      edm::LogVerbatim("BeamMomentumGun") << "BeamMonetumGun is initialzed";
  }

  void BeamMomentumGunProducer::produce(edm::Event& e, const edm::EventSetup& es) {
    if (fVerbosity > 0)
      edm::LogVerbatim("BeamMomentumGun") << "BeamMomentumGunProducer : Begin New Event Generation";

    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    // event loop (well, another step in it...)
    // no need to clean up GenEvent memory - done in HepMCProduct
    // here re-create fEvt (memory)
    //
    fEvt = new HepMC::GenEvent();

    // random entry generation for peaking event randomly from tree
    long int rjentry = static_cast<long int>(CLHEP::RandFlat::shoot(engine, 0, nentries_ - 1));
    fTree_->GetEntry(rjentry);
    if (fVerbosity > 0)
      edm::LogVerbatim("BeamMomentumGun") << "Entry " << rjentry << " : " << eventId_ << "  :  " << npar_;

    // loop over particles
    int barcode = 1;
    for (unsigned int ip = 0; ip < parPDGId_->size(); ip++) {
      int partID = parPDGId_->at(ip);
      const HepPDT::ParticleData* pData = fPDGTable->particle(HepPDT::ParticleID(std::abs(partID)));
      double mass = pData->mass().value();
      if (fVerbosity > 0)
        edm::LogVerbatim("BeamMomentumGun") << "PDGId: " << partID << "   mass: " << mass;
      double xp = (xoff_ + mm2cm_ * parX_->at(ip));
      double yp = (yoff_ + mm2cm_ * parY_->at(ip));
      HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(xp, yp, zpos_));
      double pxGeV = MeV2GeV_ * parPx_->at(ip);
      double pyGeV = MeV2GeV_ * parPy_->at(ip);
      double pzGeV = MeV2GeV_ * parPz_->at(ip);
      double momRand2 = pxGeV * pxGeV + pyGeV * pyGeV + pzGeV * pzGeV;
      double energy = std::sqrt(momRand2 + mass * mass);
      double mom = std::sqrt(momRand2);
      double theta = CLHEP::RandFlat::shoot(engine, fMinTheta, fMaxTheta);
      double phi = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
      double px = mom * sin(theta) * cos(phi);
      double py = mom * sin(theta) * sin(phi);
      double pz = mom * cos(theta);

      if (fVerbosity > 0)
        edm::LogVerbatim("BeamMomentumGun") << "px:py:pz " << px << ":" << py << ":" << pz;

      HepMC::FourVector p(px, py, pz, energy);
      HepMC::GenParticle* part = new HepMC::GenParticle(p, partID, 1);
      part->suggest_barcode(barcode);
      barcode++;
      Vtx->add_particle_out(part);

      if (fAddAntiParticle) {
        HepMC::FourVector ap(-px, -py, -pz, energy);
        int apartID = (partID == 22 || partID == 23) ? partID : -partID;
        HepMC::GenParticle* apart = new HepMC::GenParticle(ap, apartID, 1);
        apart->suggest_barcode(barcode);
        if (fVerbosity > 0)
          edm::LogVerbatim("BeamMomentumGun")
              << "Add anti-particle " << apartID << ":" << -px << ":" << -py << ":" << -pz;
        barcode++;
        Vtx->add_particle_out(apart);
      }

      fEvt->add_vertex(Vtx);
    }

    fEvt->set_event_number(e.id().event());
    fEvt->set_signal_process_id(20);

    if (fVerbosity > 0)
      fEvt->print();

    std::unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
    BProduct->addHepMCData(fEvt);
    e.put(std::move(BProduct), "unsmeared");

    std::unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
    e.put(std::move(genEventInfo));

    if (fVerbosity > 0)
      edm::LogVerbatim("BeamMomentumGun") << "BeamMomentumGunProducer : Event Generation Done";
  }
}  // namespace edm
