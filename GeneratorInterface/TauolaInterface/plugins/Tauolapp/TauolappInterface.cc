/* this is the code for the new Tauola++ */

#include <iostream>

#include "GeneratorInterface/TauolaInterface/interface/TauolappInterface.h"

#include "Tauola/Tauola.h"
#include "Tauola/TauolaHepMCEvent.h"
#include "Tauola/Log.h"
#include "Tauola/TauolaHepMCParticle.h"
#include "Tauola/TauolaParticle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandomEngine.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// LHE Run
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

using namespace gen;
using namespace edm;
using namespace std;

CLHEP::HepRandomEngine* TauolappInterface::fRandomEngine = nullptr;

extern "C" {

void gen::ranmar_(float* rvec, int* lenv) {
  for (int i = 0; i < *lenv; i++)
    *rvec++ = TauolappInterface::flat();
  return;
}

void gen::rmarin_(int*, int*, int*) { return; }
}

TauolappInterface::TauolappInterface(const edm::ParameterSet& pset, edm::ConsumesCollector iCollector)
    : fPolarization(false),
      fPDGTableToken(iCollector.esConsumes<edm::Transition::BeginLuminosityBlock>()),
      fPSet(nullptr),
      fIsInitialized(false),
      fMDTAU(-1),
      fSelectDecayByEvent(false),
      lhe(nullptr),
      dmMatch(0.5),
      dolhe(false),
      dolheBosonCorr(false),
      ntries(10) {
  fPSet = new ParameterSet(pset);
}

TauolappInterface::~TauolappInterface() {
  if (fPSet != nullptr)
    delete fPSet;
}

void TauolappInterface::init(const edm::EventSetup& es) {
  if (fIsInitialized)
    return;  // do init only once
  if (fPSet == nullptr)
    throw cms::Exception("TauolappInterfaceError") << "Attempt to initialize Tauola with an empty ParameterSet\n"
                                                   << std::endl;

  fIsInitialized = true;

  fPDGTable = es.getHandle(fPDGTableToken);

  Tauolapp::Tauola::setDecayingParticle(15);

  // LHE Information
  dmMatch = fPSet->getUntrackedParameter<double>("dmMatch", 0.5);
  dolhe = fPSet->getUntrackedParameter<bool>("dolhe", false);
  dolheBosonCorr = fPSet->getUntrackedParameter<bool>("dolheBosonCorr", true);
  ntries = fPSet->getUntrackedParameter<int>("ntries", 10);

  // polarization switch
  // fPolarization = fPSet->getParameter<bool>("UseTauolaPolarization") ? 1 : 0 ;
  fPolarization = fPSet->getParameter<bool>("UseTauolaPolarization");

  // read tau decay mode switches
  //
  ParameterSet cards = fPSet->getParameter<ParameterSet>("InputCards");

  fMDTAU = cards.getParameter<int>("mdtau");

  if (fMDTAU == 0 || fMDTAU == 1) {
    Tauolapp::Tauola::setSameParticleDecayMode(cards.getParameter<int>("pjak1"));
    Tauolapp::Tauola::setOppositeParticleDecayMode(cards.getParameter<int>("pjak2"));
  }

  Tauolapp::Tauola::spin_correlation.setAll(fPolarization);

  // some more options, copied over from an example
  // Default values
  //Tauola::setEtaK0sPi(0,0,0); // switches to decay eta K0_S and pi0 1/0 on/off.

  const HepPDT::ParticleData* PData =
      fPDGTable->particle(HepPDT::ParticleID(abs(Tauolapp::Tauola::getDecayingParticle())));
  double lifetime = PData->lifetime().value();
  Tauolapp::Tauola::setTauLifetime(lifetime);

  fPDGs.push_back(Tauolapp::Tauola::getDecayingParticle());

  Tauolapp::Tauola::setRandomGenerator(gen::TauolappInterface::flat);

  if (fPSet->exists("parameterSets")) {
    std::vector<std::string> par = fPSet->getParameter<std::vector<std::string> >("parameterSets");
    for (unsigned int ip = 0; ip < par.size(); ++ip) {
      std::string curSet = par[ip];
      if (curSet == "setNewCurrents")
        Tauolapp::Tauola::setNewCurrents(fPSet->getParameter<int>(curSet));
    }
  }

  Tauolapp::Tauola::initialize();

  Tauolapp::Tauola::spin_correlation.setAll(
      fPolarization);  // Tauola switches this on during Tauola::initialise(); so we add this here to keep it on/off

  if (fPSet->exists("parameterSets")) {
    std::vector<std::string> par = fPSet->getParameter<std::vector<std::string> >("parameterSets");
    for (unsigned int ip = 0; ip < par.size(); ++ip) {
      std::string curSet = par[ip];
      if (curSet == "spinCorrelationSetAll")
        Tauolapp::Tauola::spin_correlation.setAll(fPSet->getParameter<bool>(curSet));
      if (curSet == "spinCorrelationGAMMA")
        Tauolapp::Tauola::spin_correlation.GAMMA = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationZ0")
        Tauolapp::Tauola::spin_correlation.Z0 = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationHIGGS")
        Tauolapp::Tauola::spin_correlation.HIGGS = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationHIGGSH")
        Tauolapp::Tauola::spin_correlation.HIGGS_H = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationHIGGSA")
        Tauolapp::Tauola::spin_correlation.HIGGS_A = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationHIGGSPLUS")
        Tauolapp::Tauola::spin_correlation.HIGGS_PLUS = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationHIGGSMINUS")
        Tauolapp::Tauola::spin_correlation.HIGGS_MINUS = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationWPLUS")
        Tauolapp::Tauola::spin_correlation.W_PLUS = fPSet->getParameter<bool>(curSet);
      if (curSet == "spinCorrelationWMINUS")
        Tauolapp::Tauola::spin_correlation.W_MINUS = fPSet->getParameter<bool>(curSet);

      if (curSet == "setHiggsScalarPseudoscalarPDG")
        Tauolapp::Tauola::setHiggsScalarPseudoscalarPDG(fPSet->getParameter<int>(curSet));
      if (curSet == "setHiggsScalarPseudoscalarMixingAngle")
        Tauolapp::Tauola::setHiggsScalarPseudoscalarMixingAngle(fPSet->getParameter<double>(curSet));

      if (curSet == "setRadiation")
        Tauolapp::Tauola::setRadiation(fPSet->getParameter<bool>(curSet));
      if (curSet == "setRadiationCutOff")
        Tauolapp::Tauola::setRadiationCutOff(fPSet->getParameter<double>(curSet));

      if (curSet == "setEtaK0sPi") {
        std::vector<int> vpar = fPSet->getParameter<std::vector<int> >(curSet);
        if (vpar.size() == 3)
          Tauolapp::Tauola::setEtaK0sPi(vpar[0], vpar[1], vpar[2]);
        else {
          std::cout << "WARNING invalid size for setEtaK0sPi: " << vpar.size() << " Require 3 elements " << std::endl;
        }
      }

      if (curSet == "setTaukle") {
        std::vector<double> vpar = fPSet->getParameter<std::vector<double> >(curSet);
        if (vpar.size() == 4)
          Tauolapp::Tauola::setTaukle(vpar[0], vpar[1], vpar[2], vpar[3]);
        else {
          std::cout << "WARNING invalid size for setTaukle: " << vpar.size() << " Require 4 elements " << std::endl;
        }
      }

      if (curSet == "setTauBr") {
        edm::ParameterSet cfg = fPSet->getParameter<edm::ParameterSet>(curSet);
        std::vector<int> vJAK = cfg.getParameter<std::vector<int> >("JAK");
        std::vector<double> vBR = cfg.getParameter<std::vector<double> >("BR");
        if (vJAK.size() == vBR.size()) {
          for (unsigned int i = 0; i < vJAK.size(); i++)
            Tauolapp::Tauola::setTauBr(vJAK[i], vBR[i]);
        } else {
          std::cout << "WARNING invalid size for setTauBr - JAK: " << vJAK.size() << " BR: " << vBR.size() << std::endl;
        }
      }
    }
  }

  // override decay modes if needs be
  //
  // we have to do it AFTER init because otherwises branching ratios are NOT filled in
  //
  if (fMDTAU != 0 && fMDTAU != 1) {
    decodeMDTAU(fMDTAU);
  }

  Tauolapp::Log::LogWarning(false);
  Tauolapp::Log::IgnoreRedirection(true);

  return;
}

double TauolappInterface::flat() {
  if (!fRandomEngine) {
    throw cms::Exception("LogicError")
        << "TauolaInterface::flat: Attempt to generate random number when engine pointer is null\n"
        << "This might mean that the code was modified to generate a random number outside the\n"
        << "event and beginLuminosityBlock methods, which is not allowed.\n";
  }
  return fRandomEngine->flat();
}

HepMC::GenEvent* TauolappInterface::decay(HepMC::GenEvent* evt) {
  if (!fIsInitialized)
    return evt;
  Tauolapp::Tauola::setRandomGenerator(
      gen::TauolappInterface::flat);  // rest tauola++ random number incase other modules use tauola++
  int NPartBefore = evt->particles_size();
  int NVtxBefore = evt->vertices_size();

  // what do we do if Hep::GenEvent size is larger than 10K ???
  // Tauola (& Photos, BTW) can only handle up to 10K via HEPEVT,
  // and in case of CMS, it's only up to 4K !!!
  // override decay mode if needs be

  if (fSelectDecayByEvent) {
    selectDecayByMDTAU();
  }
  if (dolhe && lhe != nullptr) {
    std::vector<HepMC::GenParticle> particles;
    std::vector<int> m_idx;
    std::vector<double> spinup = lhe->getHEPEUP()->SPINUP;
    std::vector<int> pdg = lhe->getHEPEUP()->IDUP;
    for (unsigned int i = 0; i < spinup.size(); i++) {
      particles.push_back(HepMC::GenParticle(HepMC::FourVector(lhe->getHEPEUP()->PUP.at(i)[0],
                                                               lhe->getHEPEUP()->PUP.at(i)[1],
                                                               lhe->getHEPEUP()->PUP.at(i)[2],
                                                               lhe->getHEPEUP()->PUP.at(i)[3]),
                                             lhe->getHEPEUP()->IDUP.at(i)));
      int status = lhe->getHEPEUP()->ISTUP.at(i);
      particles.at(particles.size() - 1).set_generated_mass(lhe->getHEPEUP()->PUP.at(i)[4]);
      particles.at(particles.size() - 1).set_status(status > 0 ? (status == 2 ? 3 : status) : 3);
      m_idx.push_back(lhe->getHEPEUP()->MOTHUP.at(i).first - 1);  // correct for fortran index offset
    }
    // match to taus in hepmc and identify mother of taus
    bool hastaus(false);
    std::vector<HepMC::GenParticle*> match;
    for (HepMC::GenEvent::particle_const_iterator iter = evt->particles_begin(); iter != evt->particles_end(); iter++) {
      if (abs((*iter)->pdg_id()) == 15) {
        hastaus = true;
        int mother_pid(0);
        // check imediate parent to avoid parent tau ie tau->taugamma
        for (HepMC::GenVertex::particle_iterator mother = (*iter)->production_vertex()->particles_begin(HepMC::parents);
             mother != (*iter)->production_vertex()->particles_end(HepMC::parents);
             mother++) {
          mother_pid = (*mother)->pdg_id();
          if (mother_pid != (*iter)->pdg_id()) {
            // match against lhe record
            if (abs(mother_pid) == 24 ||  // W
                abs(mother_pid) == 37 ||  // H+/-
                abs(mother_pid) == 23 ||  // Z
                abs(mother_pid) == 22 ||  // gamma
                abs(mother_pid) == 25 ||  // H0 SM
                abs(mother_pid) == 35 ||  // H0
                abs(mother_pid) == 36     // A0
            ) {
              bool isfound = false;
              for (unsigned int k = 0; k < match.size(); k++) {
                if ((*mother) == match.at(k))
                  isfound = true;
              }
              if (!isfound)
                match.push_back(*mother);
            }
          }
        }
      }
    }
    if (hastaus) {
      // if is single gauge boson decay and match helicities
      if (match.size() == 1 && dolheBosonCorr) {
        for (int i = 0; i < ntries; i++) {
          // re-decay taus then check if helicities match
          auto* t_event = new Tauolapp::TauolaHepMCEvent(evt);
          t_event->undecayTaus();
          t_event->decayTaus();
          bool ismatch = true;
          for (unsigned int j = 0; j < spinup.size(); j++) {
            if (abs(pdg.at(j)) == 15) {
              double diffhelminus = (-1.0 * (double)Tauolapp::Tauola::getHelMinus() -
                                     spinup.at(j));  // -1.0 to correct for tauola feature
              double diffhelplus = ((double)Tauolapp::Tauola::getHelPlus() - spinup.at(j));
              if (pdg.at(j) == 15 && diffhelminus > 0.5)
                ismatch = false;
              if (pdg.at(j) == -15 && diffhelplus > 0.5)
                ismatch = false;
            }
          }
          delete t_event;
          if (ismatch)
            break;
        }
      } else {
        // If the event does not contain a single gauge boson the code will be run with
        // remove all tau decays
        auto* t_event = new Tauolapp::TauolaHepMCEvent(evt);
        t_event->undecayTaus();
        delete t_event;
        // decay all taus manually based on the helicity
        for (HepMC::GenEvent::particle_const_iterator iter = evt->particles_begin(); iter != evt->particles_end();
             iter++) {
          if (abs((*iter)->pdg_id()) == 15 && isLastTauInChain(*iter)) {
            TLorentzVector ltau(
                (*iter)->momentum().px(), (*iter)->momentum().py(), (*iter)->momentum().pz(), (*iter)->momentum().e());
            HepMC::GenParticle* m = GetMother(*iter);
            TLorentzVector mother(m->momentum().px(), m->momentum().py(), m->momentum().pz(), m->momentum().e());
            TVector3 boost = -1.0 * mother.BoostVector();  // boost into mother's CM frame
            TLorentzVector ltau_lab = ltau;
            ltau.Boost(boost);
            mother.Boost(boost);
            HepMC::GenEvent* tauevt = make_simple_tau_event(ltau, (*iter)->pdg_id(), (*iter)->status());
            HepMC::GenParticle* p = (*(tauevt->particles_begin()));
            Tauolapp::TauolaParticle* tp = new Tauolapp::TauolaHepMCParticle(p);
            double helicity = MatchedLHESpinUp(*iter, particles, spinup, m_idx);  // get helicity from lhe
            if ((*iter)->pdg_id() == 15)
              helicity *= -1.0;
            tp->undecay();
            // use |S_{tau}|=0.999999 to avoid issues with numerical roundoff
            Tauolapp::Tauola::decayOne(tp, true, 0, 0, ((double)helicity) * 0.999999);
            boost *= -1.0;  // boost back to lab frame
            mother.Boost(boost);
            update_particles((*iter), evt, p, boost);
            //correct tau liftetime for boost (change rest frame from mothers to taus)
            BoostProdToLabLifeTimeInDecays((*iter), ltau_lab, ltau);
            delete tauevt;
          }
        }
      }
    }
  } else {
    //construct tmp TAUOLA event
    auto* t_event = new Tauolapp::TauolaHepMCEvent(evt);
    //t_event->undecayTaus();
    t_event->decayTaus();
    delete t_event;
  }

  for (int iv = NVtxBefore + 1; iv <= evt->vertices_size(); iv++) {
    HepMC::GenVertex* GenVtx = evt->barcode_to_vertex(-iv);
    //
    // now find decay products with funky barcode, weed out and replace with clones of sensible barcode
    // we can NOT change the barcode while iterating, because iterators do depend on the barcoding
    // thus we have to take a 2-step procedure
    //
    std::vector<int> BCodes;
    BCodes.clear();
    for (HepMC::GenVertex::particle_iterator pitr = GenVtx->particles_begin(HepMC::children);
         pitr != GenVtx->particles_end(HepMC::children);
         ++pitr) {
      if ((*pitr)->barcode() > 10000) {
        BCodes.push_back((*pitr)->barcode());
      }
    }
    if (!BCodes.empty()) {
      for (size_t ibc = 0; ibc < BCodes.size(); ibc++) {
        HepMC::GenParticle* p1 = evt->barcode_to_particle(BCodes[ibc]);
        int nbc = p1->barcode() - 10000 + NPartBefore;
        p1->suggest_barcode(nbc);
      }
    }
  }

  for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {
    if ((*p)->end_vertex() && (*p)->status() == 1)
      (*p)->set_status(2);
    if ((*p)->end_vertex() && (*p)->end_vertex()->particles_out_size() == 0)
      edm::LogWarning("TauolappInterface::decay error: empty end vertex!");
  }

  return evt;
}

void TauolappInterface::statistics() { return; }

void TauolappInterface::decodeMDTAU(int mdtau) {
  // Note-1:
  // I have to hack the common block directly because set<...>DecayMode(...)
  // only changes it in the Tauola++ instance but does NOT passes it over
  // to the Fortran core - this it does only one, via initialize() stuff...
  //
  // So I'll do both ways of settings, just for consistency...
  // but I probably need to communicate it to the Tauola(++) team...
  //

  // Note-2:
  // originally, the 1xx settings are meant for tau's from hard event,
  // and the 2xx settings are for any tau in the event record;
  //
  // later one, we'll have to take this into account...
  // but first I'll have to sort out what happens in the 1xx case
  // to tau's coming outside of hard event (if any in the record)
  //

  if (mdtau == 101 || mdtau == 201) {
    // override with electron mode for both tau's
    //
    Tauolapp::jaki_.jak1 = 1;
    Tauolapp::jaki_.jak2 = 1;
    Tauolapp::Tauola::setSameParticleDecayMode(1);
    Tauolapp::Tauola::setOppositeParticleDecayMode(1);
    return;
  }

  if (mdtau == 102 || mdtau == 202) {
    // override with muon mode for both tau's
    //
    Tauolapp::jaki_.jak1 = 2;
    Tauolapp::jaki_.jak2 = 2;
    Tauolapp::Tauola::setSameParticleDecayMode(2);
    Tauolapp::Tauola::setOppositeParticleDecayMode(2);
    return;
  }

  if (mdtau == 111 || mdtau == 211) {
    // override with electron mode for 1st tau
    // and any mode for 2nd tau
    //
    Tauolapp::jaki_.jak1 = 1;
    Tauolapp::jaki_.jak2 = 0;
    Tauolapp::Tauola::setSameParticleDecayMode(1);
    Tauolapp::Tauola::setOppositeParticleDecayMode(0);
    return;
  }

  if (mdtau == 112 || mdtau == 212) {
    // override with muon mode for the 1st tau
    // and any mode for the 2nd tau
    //
    Tauolapp::jaki_.jak1 = 2;
    Tauolapp::jaki_.jak2 = 0;
    Tauolapp::Tauola::setSameParticleDecayMode(2);
    Tauolapp::Tauola::setOppositeParticleDecayMode(0);
    return;
  }

  if (mdtau == 121 || mdtau == 221) {
    // override with any mode for the 1st tau
    // and electron mode for the 2nd tau
    //
    Tauolapp::jaki_.jak1 = 0;
    Tauolapp::jaki_.jak2 = 1;
    Tauolapp::Tauola::setSameParticleDecayMode(0);
    Tauolapp::Tauola::setOppositeParticleDecayMode(1);
    return;
  }

  if (mdtau == 122 || mdtau == 222) {
    // override with any mode for the 1st tau
    // and muon mode for the 2nd tau
    //
    Tauolapp::jaki_.jak1 = 0;
    Tauolapp::jaki_.jak2 = 2;
    Tauolapp::Tauola::setSameParticleDecayMode(0);
    Tauolapp::Tauola::setOppositeParticleDecayMode(2);
    return;
  }

  if (mdtau == 140 || mdtau == 240) {
    // override with pi+/- nutau mode for both tau's
    //
    Tauolapp::jaki_.jak1 = 3;
    Tauolapp::jaki_.jak2 = 3;
    Tauolapp::Tauola::setSameParticleDecayMode(3);
    Tauolapp::Tauola::setOppositeParticleDecayMode(3);
    return;
  }

  if (mdtau == 141 || mdtau == 241) {
    // override with pi+/- nutau mode for the 1st tau
    // and any mode for the 2nd tau
    //
    Tauolapp::jaki_.jak1 = 3;
    Tauolapp::jaki_.jak2 = 0;
    Tauolapp::Tauola::setSameParticleDecayMode(3);
    Tauolapp::Tauola::setOppositeParticleDecayMode(0);
    return;
  }

  if (mdtau == 142 || mdtau == 242) {
    // override with any mode for the 1st tau
    // and pi+/- nutau mode for 2nd tau
    //
    Tauolapp::jaki_.jak1 = 0;
    Tauolapp::jaki_.jak2 = 3;
    Tauolapp::Tauola::setSameParticleDecayMode(0);
    Tauolapp::Tauola::setOppositeParticleDecayMode(3);
    return;
  }

  // OK, we come here for semi-inclusive modes
  //

  // First of all, leptons and hadron modes sums
  //
  // re-scale branching ratios, just in case...
  //
  double sumBra = 0;

  // the number of decay modes is hardcoded at 22 because that's what it is right now in Tauola
  // in the future, perhaps an asscess method would be useful - communicate to Tauola team...
  //

  for (int i = 0; i < 22; i++) {
    sumBra += Tauolapp::taubra_.gamprt[i];
  }
  if (sumBra == 0.)
    return;  // perhaps need to throw ?
  for (int i = 0; i < 22; i++) {
    double newBra = Tauolapp::taubra_.gamprt[i] / sumBra;
    Tauolapp::Tauola::setTauBr(i + 1, newBra);
  }
  sumBra = 1.0;

  double sumLeptonBra = Tauolapp::taubra_.gamprt[0] + Tauolapp::taubra_.gamprt[1];
  double sumHadronBra = sumBra - sumLeptonBra;

  for (int i = 0; i < 2; i++) {
    fLeptonModes.push_back(i + 1);
    fScaledLeptonBrRatios.push_back((Tauolapp::taubra_.gamprt[i] / sumLeptonBra));
  }
  for (int i = 2; i < 22; i++) {
    fHadronModes.push_back(i + 1);
    fScaledHadronBrRatios.push_back((Tauolapp::taubra_.gamprt[i] / sumHadronBra));
  }

  fSelectDecayByEvent = true;
  return;
}

void TauolappInterface::selectDecayByMDTAU() {
  if (fMDTAU == 100 || fMDTAU == 200) {
    int mode = selectLeptonic();
    Tauolapp::jaki_.jak1 = mode;
    Tauolapp::Tauola::setSameParticleDecayMode(mode);
    mode = selectLeptonic();
    Tauolapp::jaki_.jak2 = mode;
    Tauolapp::Tauola::setOppositeParticleDecayMode(mode);
    return;
  }

  int modeL = selectLeptonic();
  int modeH = selectHadronic();

  if (fMDTAU == 110 || fMDTAU == 210) {
    Tauolapp::jaki_.jak1 = modeL;
    Tauolapp::jaki_.jak2 = 0;
    Tauolapp::Tauola::setSameParticleDecayMode(modeL);
    Tauolapp::Tauola::setOppositeParticleDecayMode(0);
    return;
  }

  if (fMDTAU == 120 || fMDTAU == 22) {
    Tauolapp::jaki_.jak1 = 0;
    Tauolapp::jaki_.jak2 = modeL;
    Tauolapp::Tauola::setSameParticleDecayMode(0);
    Tauolapp::Tauola::setOppositeParticleDecayMode(modeL);
    return;
  }

  if (fMDTAU == 114 || fMDTAU == 214) {
    Tauolapp::jaki_.jak1 = modeL;
    Tauolapp::jaki_.jak2 = modeH;
    Tauolapp::Tauola::setSameParticleDecayMode(modeL);
    Tauolapp::Tauola::setOppositeParticleDecayMode(modeH);
    return;
  }

  if (fMDTAU == 124 || fMDTAU == 224) {
    Tauolapp::jaki_.jak1 = modeH;
    Tauolapp::jaki_.jak2 = modeL;
    Tauolapp::Tauola::setSameParticleDecayMode(modeH);
    Tauolapp::Tauola::setOppositeParticleDecayMode(modeL);
    return;
  }

  if (fMDTAU == 115 || fMDTAU == 215) {
    Tauolapp::jaki_.jak1 = 1;
    Tauolapp::jaki_.jak2 = modeH;
    Tauolapp::Tauola::setSameParticleDecayMode(1);
    Tauolapp::Tauola::setOppositeParticleDecayMode(modeH);
    return;
  }

  if (fMDTAU == 125 || fMDTAU == 225) {
    Tauolapp::jaki_.jak1 = modeH;
    Tauolapp::jaki_.jak2 = 1;
    Tauolapp::Tauola::setSameParticleDecayMode(modeH);
    Tauolapp::Tauola::setOppositeParticleDecayMode(1);
    return;
  }

  if (fMDTAU == 116 || fMDTAU == 216) {
    Tauolapp::jaki_.jak1 = 2;
    Tauolapp::jaki_.jak2 = modeH;
    Tauolapp::Tauola::setSameParticleDecayMode(2);
    Tauolapp::Tauola::setOppositeParticleDecayMode(modeH);
    return;
  }

  if (fMDTAU == 126 || fMDTAU == 226) {
    Tauolapp::jaki_.jak1 = modeH;
    Tauolapp::jaki_.jak2 = 2;
    Tauolapp::Tauola::setSameParticleDecayMode(modeH);
    Tauolapp::Tauola::setOppositeParticleDecayMode(2);
    return;
  }

  if (fMDTAU == 130 || fMDTAU == 230) {
    Tauolapp::jaki_.jak1 = modeH;
    Tauolapp::jaki_.jak2 = selectHadronic();
    Tauolapp::Tauola::setSameParticleDecayMode(modeH);
    Tauolapp::Tauola::setOppositeParticleDecayMode(Tauolapp::jaki_.jak2);
    return;
  }

  if (fMDTAU == 131 || fMDTAU == 231) {
    Tauolapp::jaki_.jak1 = modeH;
    Tauolapp::jaki_.jak2 = 0;
    Tauolapp::Tauola::setSameParticleDecayMode(modeH);
    Tauolapp::Tauola::setOppositeParticleDecayMode(0);
    return;
  }

  if (fMDTAU == 132 || fMDTAU == 232) {
    Tauolapp::jaki_.jak1 = 0;
    Tauolapp::jaki_.jak2 = modeH;
    Tauolapp::Tauola::setSameParticleDecayMode(0);
    Tauolapp::Tauola::setOppositeParticleDecayMode(modeH);
    return;
  }

  // unlikely that we get here on unknown mdtau
  // - there's a protection earlier
  // but if we do, just set defaults
  // probably need to spit a warning...
  //
  Tauolapp::Tauola::setSameParticleDecayMode(0);
  Tauolapp::Tauola::setOppositeParticleDecayMode(0);

  return;
}

int TauolappInterface::selectLeptonic() {
  float prob = flat();

  if (prob > 0. && prob <= fScaledLeptonBrRatios[0]) {
    return 1;
  } else if (prob > fScaledLeptonBrRatios[1] && prob <= 1.) {
    return 2;
  }

  return 0;
}

int TauolappInterface::selectHadronic() {
  float prob = 0.;
  int len = 1;
  ranmar_(&prob, &len);

  double sumBra = fScaledHadronBrRatios[0];
  if (prob > 0. && prob <= sumBra) {
    return fHadronModes[0];
  } else {
    int NN = fScaledHadronBrRatios.size();
    for (int i = 1; i < NN; i++) {
      if (prob > sumBra && prob <= (sumBra + fScaledHadronBrRatios[i])) {
        return fHadronModes[i];
      }
      sumBra += fScaledHadronBrRatios[i];
    }
  }

  return 0;
}

HepMC::GenEvent* TauolappInterface::make_simple_tau_event(const TLorentzVector& l, int pdgid, int status) {
  HepMC::GenEvent* event = new HepMC::GenEvent();
  // make tau's four vector
  HepMC::FourVector momentum_tau1(l.Px(), l.Py(), l.Pz(), l.E());
  // make particles
  HepMC::GenParticle* tau1 = new HepMC::GenParticle(momentum_tau1, pdgid, status);
  // make the vertex
  HepMC::GenVertex* vertex = new HepMC::GenVertex();
  vertex->add_particle_out(tau1);
  event->add_vertex(vertex);
  return event;
}

void TauolappInterface::update_particles(HepMC::GenParticle* partHep,
                                         HepMC::GenEvent* theEvent,
                                         HepMC::GenParticle* p,
                                         TVector3& boost) {
  partHep->set_status(p->status());
  if (p->end_vertex()) {
    if (!partHep->end_vertex()) {
      HepMC::GenVertex* vtx = new HepMC::GenVertex(p->end_vertex()->position());
      theEvent->add_vertex(vtx);
      vtx->add_particle_in(partHep);
    }
    if (p->end_vertex()->particles_out_size() != 0) {
      for (HepMC::GenVertex::particles_out_const_iterator d = p->end_vertex()->particles_out_const_begin();
           d != p->end_vertex()->particles_out_const_end();
           d++) {
        // Create daughter and add to event
        TLorentzVector l((*d)->momentum().px(), (*d)->momentum().py(), (*d)->momentum().pz(), (*d)->momentum().e());
        l.Boost(boost);
        HepMC::FourVector momentum(l.Px(), l.Py(), l.Pz(), l.E());
        HepMC::GenParticle* daughter = new HepMC::GenParticle(momentum, (*d)->pdg_id(), (*d)->status());
        daughter->suggest_barcode(theEvent->particles_size() + 1);
        partHep->end_vertex()->add_particle_out(daughter);
        if ((*d)->end_vertex())
          update_particles(daughter, theEvent, (*d), boost);
      }
    }
  }
}

bool TauolappInterface::isLastTauInChain(const HepMC::GenParticle* tau) {
  if (tau->end_vertex()) {
    HepMC::GenVertex::particle_iterator dau;
    for (dau = tau->end_vertex()->particles_begin(HepMC::children);
         dau != tau->end_vertex()->particles_end(HepMC::children);
         dau++) {
      int dau_pid = (*dau)->pdg_id();
      if (dau_pid == tau->pdg_id())
        return false;
    }
  }
  return true;
}

double TauolappInterface::MatchedLHESpinUp(HepMC::GenParticle* tau,
                                           std::vector<HepMC::GenParticle>& p,
                                           std::vector<double>& spinup,
                                           std::vector<int>& m_idx) {
  HepMC::GenParticle* Tau = FirstTauInChain(tau);
  HepMC::GenParticle* mother = GetMother(Tau);
  TLorentzVector t(tau->momentum().px(), tau->momentum().py(), tau->momentum().pz(), tau->momentum().e());
  TLorentzVector m(mother->momentum().px(), mother->momentum().py(), mother->momentum().pz(), mother->momentum().e());
  for (unsigned int i = 0; i < p.size(); i++) {
    if (tau->pdg_id() == p.at(i).pdg_id()) {
      if (mother->pdg_id() == p.at(m_idx.at(i)).pdg_id()) {
        TLorentzVector pm(p.at(m_idx.at(i)).momentum().px(),
                          p.at(m_idx.at(i)).momentum().py(),
                          p.at(m_idx.at(i)).momentum().pz(),
                          p.at(m_idx.at(i)).momentum().e());
        if (fabs(m.M() - pm.M()) < dmMatch)
          return spinup.at(i);
      }
    }
  }
  return 0;
}

HepMC::GenParticle* TauolappInterface::FirstTauInChain(HepMC::GenParticle* tau) {
  if (tau->production_vertex()) {
    HepMC::GenVertex::particle_iterator mother;
    for (mother = tau->production_vertex()->particles_begin(HepMC::parents);
         mother != tau->production_vertex()->particles_end(HepMC::parents);
         mother++) {
      if ((*mother)->pdg_id() == tau->pdg_id())
        return FirstTauInChain(*mother);  // recursive call to get mother with different pdgid
    }
  }
  return tau;
}

HepMC::GenParticle* TauolappInterface::GetMother(HepMC::GenParticle* tau) {
  if (tau->production_vertex()) {
    HepMC::GenVertex::particle_iterator mother;
    for (mother = tau->production_vertex()->particles_begin(HepMC::parents);
         mother != tau->production_vertex()->particles_end(HepMC::parents);
         mother++) {
      if ((*mother)->pdg_id() == tau->pdg_id())
        return GetMother(*mother);  // recursive call to get mother with different pdgid
      return (*mother);
    }
  }
  return tau;
}

void TauolappInterface::BoostProdToLabLifeTimeInDecays(HepMC::GenParticle* p,
                                                       TLorentzVector& lab,
                                                       TLorentzVector& prod) {
  if (p->end_vertex() && p->production_vertex()) {
    HepMC::GenVertex* PGenVtx = p->production_vertex();
    HepMC::GenVertex* EGenVtx = p->end_vertex();
    double VxDec = PGenVtx->position().x() + lab.Px() / prod.Px() * (EGenVtx->position().x() - PGenVtx->position().x());
    double VyDec = PGenVtx->position().y() + lab.Py() / prod.Py() * (EGenVtx->position().y() - PGenVtx->position().y());
    double VzDec = PGenVtx->position().z() + lab.Pz() / prod.Pz() * (EGenVtx->position().z() - PGenVtx->position().z());
    double VtDec = PGenVtx->position().t() + lab.Pt() / prod.Pt() * (EGenVtx->position().t() - PGenVtx->position().t());
    EGenVtx->set_position(HepMC::FourVector(VxDec, VyDec, VzDec, VtDec));
    for (HepMC::GenVertex::particle_iterator dau = p->end_vertex()->particles_begin(HepMC::children);
         dau != p->end_vertex()->particles_end(HepMC::children);
         dau++) {
      BoostProdToLabLifeTimeInDecays((*dau), lab, prod);  //recursively modify everything in the decay chain
    }
  }
}
