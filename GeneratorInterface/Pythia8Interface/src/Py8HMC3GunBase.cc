#include <memory>

#include "GeneratorInterface/Pythia8Interface/interface/Py8HMC3GunBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Concurrency/interface/SharedResourceNames.h"

#include "HepMC3/Print.h"

// EvtGen plugin
//
//#include "Pythia8Plugins/EvtGen.h"

using namespace Pythia8;

const std::vector<std::string> gen::Py8HMC3GunBase::p8SharedResources = {edm::SharedResourceNames::kPythia8};

namespace gen {

  Py8HMC3GunBase::Py8HMC3GunBase(edm::ParameterSet const& ps) : Py8HMC3InterfaceBase(ps) {
    // PGun specs
    //
    edm::ParameterSet pgun_params = ps.getParameter<edm::ParameterSet>("PGunParameters");

    // although there's the method ParameterSet::empty(),
    // it looks like it's NOT even necessary to check if it is,
    // before trying to extract parameters - if it is empty,
    // the default values seem to be taken
    //
    fPartIDs = pgun_params.getParameter<std::vector<int> >("ParticleID");
    fMinPhi = pgun_params.getParameter<double>("MinPhi");  // ,-3.14159265358979323846);
    fMaxPhi = pgun_params.getParameter<double>("MaxPhi");  // , 3.14159265358979323846);
    ivhepmc = 3;
  }

  // specific to Py8GunHad !!!
  //
  bool Py8HMC3GunBase::initializeForInternalPartons() {
    // NO MATTER what was this setting below, override it before init
    // - this is essencial for the PGun mode

    // Key requirement: switch off ProcessLevel, and thereby also PartonLevel.
    fMasterGen->readString("ProcessLevel:all = off");
    fMasterGen->readString("ProcessLevel::resonanceDecays=on");
    fMasterGen->init();

    // init decayer
    fDecayer->readString("ProcessLevel:all = off");  // Same trick!
    fDecayer->readString("ProcessLevel::resonanceDecays=on");
    fDecayer->init();

#if 0
    if (useEvtGen) {
      edm::LogInfo("Pythia8Interface") << "Creating and initializing pythia8 EvtGen plugin";
      evtgenDecays = std::make_shared<EvtGenDecays>(fMasterGen.get(), evtgenDecFile, evtgenPdlFile);
      for (unsigned int i = 0; i < evtgenUserFiles.size(); i++)
        evtgenDecays->readDecayFile(evtgenUserFiles.at(i));
    }
#endif

    return true;
  }

  bool Py8HMC3GunBase::residualDecay() {
    Event* pythiaEvent = &(fMasterGen->event);

    int NPartsBeforeDecays = pythiaEvent->size() - 1;  // do NOT count the very 1st "system" particle
                                                       // in Pythia8::Event record; it does NOT even
                                                       // get translated by the HepMCInterface to the
                                                       // HepMC::GenEvent record !!!
    int NPartsAfterDecays = ((event3().get())->particles()).size();

    if (NPartsAfterDecays == NPartsBeforeDecays)
      return true;

    bool result = true;

    for (const auto &p : (event3().get())->particles()) {
      if (p->id() > NPartsBeforeDecays) {
        if (p->status() == 1 && (fDecayer->particleData).canDecay(p->pid())) {
          fDecayer->event.reset();
          Particle py8part(p->pid(),
                         93,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         p->momentum().x(),
                         p->momentum().y(),
                         p->momentum().z(),
                         p->momentum().t(),
                         p->generated_mass());

          py8part.vProd(p->production_vertex()->position().x(),
                        p->production_vertex()->position().y(),
                        p->production_vertex()->position().z(),
                        p->production_vertex()->position().t());

          py8part.tau((fDecayer->particleData).tau0(p->pid()));
          fDecayer->event.append(py8part);
          int nentries = fDecayer->event.size();
          if (!fDecayer->event[nentries - 1].mayDecay())
            continue;
          result = fDecayer->next();
          int nentries1 = fDecayer->event.size();
          if (nentries1 <= nentries)
            continue;  //same number of particles, no decays...

          p->set_status(2);

          HepMC3::GenVertexPtr prod_vtx0 = make_shared<HepMC3::GenVertex>(  // neglect particle path to decay
              HepMC3::FourVector(p->production_vertex()->position().x(),
                                 p->production_vertex()->position().y(),
                                 p->production_vertex()->position().z(),
                                 p->production_vertex()->position().t()));
          prod_vtx0->add_particle_in(p);
          (event3().get())->add_vertex(prod_vtx0);
          Pythia8::Event pyev = fDecayer->event;
          double momFac = 1.;
          for (int i = 2; i < pyev.size(); ++i) {
            // Fill the particle.
            HepMC3::GenParticlePtr pnew = make_shared<HepMC3::GenParticle>(
                HepMC3::FourVector(
                    momFac * pyev[i].px(), momFac * pyev[i].py(), momFac * pyev[i].pz(), momFac * pyev[i].e()),
                pyev[i].id(),
                pyev[i].statusHepMC());
            pnew->set_generated_mass(momFac * pyev[i].m());
            prod_vtx0->add_particle_out(pnew);
          }
        }
      }
    }

    return result;
  }

  void Py8HMC3GunBase::finalizeEvent() {
    //******** Verbosity ********

    if (maxEventsToPrint > 0 && (pythiaPylistVerbosity || pythiaHepMCVerbosity || pythiaHepMCVerbosityParticles)) {
      maxEventsToPrint--;
      if (pythiaPylistVerbosity) {
        fMasterGen->info.list();
        fMasterGen->event.list();
      }

      if (pythiaHepMCVerbosity) {
        std::cout << "Event process = " << fMasterGen->info.code() << "\n"
                  << "----------------------" << std::endl;
        HepMC3::Print::listing(*(event3().get()));
      }
      if (pythiaHepMCVerbosityParticles) {
        std::cout << "Event process = " << fMasterGen->info.code() << "\n"
                  << "----------------------" << std::endl;
        for (const auto &p : (event3().get())->particles()) {
          HepMC3::Print::line(p, true);
        }
      }
    }
    return;
  }

  void Py8HMC3GunBase::statistics() {
    fMasterGen->stat();

    double xsec = fMasterGen->info.sigmaGen();  // cross section in mb
    xsec *= 1.0e9;                              // translate to pb (CMS/Gen "convention" as of May 2009)
    runInfo().setInternalXSec(xsec);
    return;
  }

#if 0
  void Py8HMC3GunBase::evtGenDecay() {
    if (evtgenDecays.get())
      evtgenDecays->decay();
  }
#endif

}  // namespace gen
