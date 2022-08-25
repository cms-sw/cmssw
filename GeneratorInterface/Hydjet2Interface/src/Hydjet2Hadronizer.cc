/**
 *    \brief Interface to the HYDJET++ (Hydjet2) generator (since core v. 2.4.3), produces HepMC events
 *    \version 1.3
 *    \author Andrey Belyaev
 */

#include <TLorentzVector.h>
#include <TMath.h>
#include <TVector3.h>

#include "GeneratorInterface/Hydjet2Interface/interface/Hydjet2Hadronizer.h"
#include <cmath>
#include <fstream>
#include <iostream>

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/PythiaWrapper6_4.h"
#include "HepMC/SimpleVector.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

CLHEP::HepRandomEngine *hjRandomEngine;

using namespace edm;
using namespace std;
using namespace gen;

int Hydjet2Hadronizer::convertStatusForComponents(int sta, int typ, int pySta) {
  int st = -1;
  if (typ == 0)  //soft
    st = 2 - sta;
  else if (typ == 1)
    st = convertStatus(pySta);

  if (st == -1)
    throw cms::Exception("ConvertStatus") << "Wrong status code!" << endl;

  if (separateHydjetComponents_) {
    if (st == 1 && typ == 0)
      return 6;
    if (st == 1 && typ == 1)
      return 7;
    if (st == 2 && typ == 0)
      return 16;
    if (st == 2 && typ == 1)
      return 17;
  }
  return st;
}

int Hydjet2Hadronizer::convertStatus(int st) {
  if (st <= 0)
    return 0;
  if (st <= 10)
    return 1;
  if (st <= 20)
    return 2;
  if (st <= 30)
    return 3;
  else
    return -1;
}

const std::vector<std::string> Hydjet2Hadronizer::theSharedResources = {edm::SharedResourceNames::kPythia6};

//____________________________________________________________________________________________
Hydjet2Hadronizer::Hydjet2Hadronizer(const edm::ParameterSet &pset, edm::ConsumesCollector &&iC)
    : BaseHadronizer(pset),
      rotate_(pset.getParameter<bool>("rotateEventPlane")),
      evt(nullptr),
      nsub_(0),
      nhard_(0),
      nsoft_(0),
      phi0_(0.),
      sinphi0_(0.),
      cosphi0_(1.),
      fVertex_(nullptr),
      pythia6Service_(new Pythia6Service(pset))

{
  fParams.doPrintInfo = false;
  fParams.allowEmptyEvent = false;
  fParams.fNevnt = 0;                                      //not used in CMSSW
  fParams.femb = pset.getParameter<int>("embeddingMode");  //
  fParams.fSqrtS = pset.getParameter<double>("fSqrtS");    // C.m.s. energy per nucleon pair
  fParams.fAw = pset.getParameter<double>("fAw");          // Atomic weigth of nuclei, fAw
  fParams.fIfb = pset.getParameter<int>(
      "fIfb");  // Flag of type of centrality generation, fBfix (=0 is fixed by fBfix, >0 distributed [fBfmin, fBmax])
  fParams.fBmin = pset.getParameter<double>("fBmin");  // Minimum impact parameter in units of nuclear radius, fBmin
  fParams.fBmax = pset.getParameter<double>("fBmax");  // Maximum impact parameter in units of nuclear radius, fBmax
  fParams.fBfix = pset.getParameter<double>("fBfix");  // Fixed impact parameter in units of nuclear radius, fBfix
  fParams.fT = pset.getParameter<double>("fT");        // Temperature at chemical freeze-out, fT [GeV]
  fParams.fMuB = pset.getParameter<double>("fMuB");    // Chemical baryon potential per unit charge, fMuB [GeV]
  fParams.fMuS = pset.getParameter<double>("fMuS");    // Chemical strangeness potential per unit charge, fMuS [GeV]
  fParams.fMuC = pset.getParameter<double>(
      "fMuC");  // Chemical charm potential per unit charge, fMuC [GeV] (used if charm production is turned on)
  fParams.fMuI3 = pset.getParameter<double>("fMuI3");  // Chemical isospin potential per unit charge, fMuI3 [GeV]
  fParams.fThFO = pset.getParameter<double>("fThFO");  // Temperature at thermal freeze-out, fThFO [GeV]
  fParams.fMu_th_pip =
      pset.getParameter<double>("fMu_th_pip");  // Chemical potential of pi+ at thermal freeze-out, fMu_th_pip [GeV]
  fParams.fTau = pset.getParameter<double>(
      "fTau");  // Proper time proper at thermal freeze-out for central collisions, fTau [fm/c]
  fParams.fSigmaTau = pset.getParameter<double>(
      "fSigmaTau");  // Duration of emission at thermal freeze-out for central collisions, fSigmaTau [fm/c]
  fParams.fR = pset.getParameter<double>(
      "fR");  // Maximal transverse radius at thermal freeze-out for central collisions, fR [fm]
  fParams.fYlmax =
      pset.getParameter<double>("fYlmax");  // Maximal longitudinal flow rapidity at thermal freeze-out, fYlmax
  fParams.fUmax = pset.getParameter<double>(
      "fUmax");  // Maximal transverse flow rapidity at thermal freeze-out for central collisions, fUmax
  fParams.frhou2 = pset.getParameter<double>("fRhou2");  //parameter to swich ON/OFF = 0) rhou2
  fParams.frhou3 = pset.getParameter<double>("fRhou3");  //parameter to swich ON/OFF(0) rhou3
  fParams.frhou4 = pset.getParameter<double>("fRhou4");  //parameter to swich ON/OFF(0) rhou4
  fParams.fDelta =
      pset.getParameter<double>("fDelta");  // Momentum azimuthal anizotropy parameter at thermal freeze-out, fDelta
  fParams.fEpsilon =
      pset.getParameter<double>("fEpsilon");  // Spatial azimuthal anisotropy parameter at thermal freeze-out, fEpsilon
  fParams.fv2 = pset.getParameter<double>("fKeps2");  //parameter to swich ON/OFF(0) epsilon2 fluctuations
  fParams.fv3 = pset.getParameter<double>("fKeps3");  //parameter to swich ON/OFF(0) epsilon3 fluctuations
  fParams.fIfDeltaEpsilon = pset.getParameter<double>(
      "fIfDeltaEpsilon");  // Flag to specify fDelta and fEpsilon values, fIfDeltaEpsilon (=0 user's ones, >=1 calculated)
  fParams.fDecay =
      pset.getParameter<int>("fDecay");  // Flag to switch on/off hadron decays, fDecay (=0 decays off, >=1 decays on)
  fParams.fWeakDecay = pset.getParameter<double>(
      "fWeakDecay");  // Low decay width threshold fWeakDecay[GeV]: width<fWeakDecay decay off, width>=fDecayWidth decay on; can be used to switch off weak decays
  fParams.fEtaType = pset.getParameter<double>(
      "fEtaType");  // Flag to choose longitudinal flow rapidity distribution, fEtaType (=0 uniform, >0 Gaussian with the dispersion Ylmax)
  fParams.fTMuType = pset.getParameter<double>(
      "fTMuType");  // Flag to use calculated T_ch, mu_B and mu_S as a function of fSqrtS, fTMuType (=0 user's ones, >0 calculated)
  fParams.fCorrS = pset.getParameter<double>(
      "fCorrS");  // Strangeness supression factor gamma_s with fCorrS value (0<fCorrS <=1, if fCorrS <= 0 then it is calculated)
  fParams.fCharmProd = pset.getParameter<int>(
      "fCharmProd");  // Flag to include thermal charm production, fCharmProd (=0 no charm production, >=1 charm production)
  fParams.fCorrC = pset.getParameter<double>(
      "fCorrC");  // Charmness enhancement factor gamma_c with fCorrC value (fCorrC >0, if fCorrC<0 then it is calculated)
  fParams.fNhsel = pset.getParameter<int>(
      "fNhsel");  //Flag to include jet (J)/jet quenching (JQ) and hydro (H) state production, fNhsel (0 H on & J off, 1 H/J on & JQ off, 2 H/J/HQ on, 3 J on & H/JQ off, 4 H off & J/JQ on)
  fParams.fPyhist = pset.getParameter<int>(
      "fPyhist");  // Flag to suppress the output of particle history from PYTHIA, fPyhist (=1 only final state particles; =0 full particle history from PYTHIA)
  fParams.fIshad = pset.getParameter<int>(
      "fIshad");  // Flag to switch on/off nuclear shadowing, fIshad (0 shadowing off, 1 shadowing on)
  fParams.fPtmin =
      pset.getParameter<double>("fPtmin");  // Minimal pt of parton-parton scattering in PYTHIA event, fPtmin [GeV/c]
  fParams.fT0 = pset.getParameter<double>(
      "fT0");  // Initial QGP temperature for central Pb+Pb collisions in mid-rapidity, fT0 [GeV]
  fParams.fTau0 = pset.getParameter<double>("fTau0");  // Proper QGP formation time in fm/c, fTau0 (0.01<fTau0<10)
  fParams.fNf = pset.getParameter<int>("fNf");         // Number of active quark flavours in QGP, fNf (0, 1, 2 or 3)
  fParams.fIenglu = pset.getParameter<int>(
      "fIenglu");  // Flag to fix type of partonic energy loss, fIenglu (0 radiative and collisional loss, 1 radiative loss only, 2 collisional loss only)
  fParams.fIanglu = pset.getParameter<int>(
      "fIanglu");  // Flag to fix type of angular distribution of in-medium emitted gluons, fIanglu (0 small-angular, 1 wide-angular, 2 collinear).

  edm::FileInPath f1("externals/hydjet2/particles.data");
  strcpy(fParams.partDat, (f1.fullPath()).c_str());

  edm::FileInPath f2("externals/hydjet2/tabledecay.txt");
  strcpy(fParams.tabDecay, (f2.fullPath()).c_str());

  fParams.fPythiaTune = false;

  if (pset.exists("signalVtx"))
    signalVtx_ = pset.getUntrackedParameter<std::vector<double>>("signalVtx");

  if (signalVtx_.size() == 4) {
    if (!fVertex_)
      fVertex_ = new HepMC::FourVector();
    LogDebug("EventSignalVertex") << "Setting event signal vertex "
                                  << " x = " << signalVtx_.at(0) << " y = " << signalVtx_.at(1)
                                  << "  z= " << signalVtx_.at(2) << " t = " << signalVtx_.at(3) << endl;
    fVertex_->set(signalVtx_.at(0), signalVtx_.at(1), signalVtx_.at(2), signalVtx_.at(3));
  }

  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity", 0);
  LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_;
  //Max number of events printed on verbosity level
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint", 0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_;
  if (fParams.femb == 1) {
    fParams.fIfb = 0;
    src_ = iC.consumes<CrossingFrame<edm::HepMCProduct>>(
        pset.getUntrackedParameter<edm::InputTag>("backgroundLabel", edm::InputTag("mix", "generatorSmeared")));
  }

  separateHydjetComponents_ = pset.getUntrackedParameter<bool>("separateHydjetComponents", false);
}
//__________________________________________________________________________________________
Hydjet2Hadronizer::~Hydjet2Hadronizer() {
  call_pystat(1);
  delete pythia6Service_;
}

//_____________________________________________________________________
void Hydjet2Hadronizer::doSetRandomEngine(CLHEP::HepRandomEngine *v) {
  pythia6Service_->setRandomEngine(v);
  hjRandomEngine = v;
}

//______________________________________________________________________________________________________
bool Hydjet2Hadronizer::readSettings(int) {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);
  pythia6Service_->setGeneralParams();

  fParams.fSeed = hjRandomEngine->CLHEP::HepRandomEngine::getSeed();
  LogInfo("Hydjet2Hadronizer|GenSeed") << "Seed for random number generation: "
                                       << hjRandomEngine->CLHEP::HepRandomEngine::getSeed();

  return kTRUE;
}

//______________________________________________________________________________________________________
bool Hydjet2Hadronizer::initializeForInternalPartons() {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);

  // the input impact parameter (bxx_) is in [fm]; transform in [fm/RA] for hydjet usage
  const double ra = nuclear_radius();
  LogInfo("Hydjet2Hadronizer|RAScaling") << "Nuclear radius(RA) =  " << ra;
  fParams.fBmin /= ra;
  fParams.fBmax /= ra;
  fParams.fBfix /= ra;

  hj2 = new Hydjet2(fParams);

  return kTRUE;
}

//__________________________________________________________________________________________
bool Hydjet2Hadronizer::generatePartonsAndHadronize() {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);

  // generate single event
  if (fParams.femb == 1) {
    const edm::Event &e = getEDMEvent();
    HepMC::GenVertex *genvtx = nullptr;
    const HepMC::GenEvent *inev = nullptr;
    Handle<CrossingFrame<HepMCProduct>> cf;
    e.getByToken(src_, cf);
    MixCollection<HepMCProduct> mix(cf.product());
    if (mix.size() < 1) {
      throw cms::Exception("MatchVtx") << "Mixing has " << mix.size() << " sub-events, should have been at least 1"
                                       << endl;
    }
    const HepMCProduct &bkg = mix.getObject(0);
    if (!(bkg.isVtxGenApplied())) {
      throw cms::Exception("MatchVtx") << "Input background does not have smeared vertex!" << endl;
    } else {
      inev = bkg.GetEvent();
    }

    genvtx = inev->signal_process_vertex();

    if (!genvtx)
      throw cms::Exception("MatchVtx") << "Input background does not have signal process vertex!" << endl;

    double aX, aY, aZ, aT;

    aX = genvtx->position().x();
    aY = genvtx->position().y();
    aZ = genvtx->position().z();
    aT = genvtx->position().t();

    if (!fVertex_) {
      fVertex_ = new HepMC::FourVector();
    }
    LogInfo("MatchVtx") << " setting vertex "
                        << " aX " << aX << " aY " << aY << " aZ " << aZ << " aT " << aT << endl;
    fVertex_->set(aX, aY, aZ, aT);

    const HepMC::HeavyIon *hi = inev->heavy_ion();

    if (hi) {
      fParams.fBfix = (hi->impact_parameter()) / nuclear_radius();
      phi0_ = hi->event_plane_angle();
      sinphi0_ = sin(phi0_);
      cosphi0_ = cos(phi0_);
    } else {
      LogWarning("EventEmbedding") << "Background event does not have heavy ion record!";
    }

  } else if (rotate_)
    rotateEvtPlane();

  nsoft_ = 0;
  nhard_ = 0;

  // generate one HYDJET event
  int ntry = 0;

  while (nsoft_ == 0 && nhard_ == 0) {
    if (ntry > 100) {
      LogError("Hydjet2EmptyEvent") << "##### HYDJET2: No Particles generated, Number of tries =" << ntry;
      // Throw an exception. Use the EventCorruption exception since it maps onto SkipEvent
      // which is what we want to do here.
      std::ostringstream sstr;
      sstr << "Hydjet2HadronizerProducer: No particles generated after " << ntry << " tries.\n";
      edm::Exception except(edm::errors::EventCorruption, sstr.str());
      throw except;
    } else {
      hj2->GenerateEvent(fParams.fBfix);

      if (hj2->IsEmpty()) {
        continue;
      }

      nsoft_ = hj2->GetNhyd();
      nsub_ = hj2->GetNjet();
      nhard_ = hj2->GetNpyt();

      //100 trys
      ++ntry;
    }
  }

  if (ev == 0) {
    Sigin = hj2->GetSigin();
    Sigjet = hj2->GetSigjet();
  }
  ev = true;

  if (fParams.fNhsel < 3)
    nsub_++;

  // event information
  std::unique_ptr<HepMC::GenEvent> evt = std::make_unique<HepMC::GenEvent>();
  std::unique_ptr<edm::HepMCProduct> HepMCEvt = std::make_unique<edm::HepMCProduct>();

  if (nhard_ > 0 || nsoft_ > 0)
    get_particles(evt.get());

  evt->set_signal_process_id(pypars.msti[0]);  // type of the process
  evt->set_event_scale(pypars.pari[16]);       // Q^2
  add_heavy_ion_rec(evt.get());

  if (fVertex_) {
    // generate new vertex & apply the shift
    // Copy the HepMC::GenEvent
    HepMCEvt = std::make_unique<edm::HepMCProduct>(evt.get());
    HepMCEvt->applyVtxGen(fVertex_);
    evt = std::make_unique<HepMC::GenEvent>(*HepMCEvt->GetEvent());
  }

  HepMC::HEPEVT_Wrapper::check_hepevt_consistency();
  LogDebug("HEPEVT_info") << "Ev numb: " << HepMC::HEPEVT_Wrapper::event_number()
                          << " Entries number: " << HepMC::HEPEVT_Wrapper::number_entries() << " Max. entries "
                          << HepMC::HEPEVT_Wrapper::max_number_entries() << std::endl;

  event() = std::move(evt);
  return kTRUE;
}

//________________________________________________________________
bool Hydjet2Hadronizer::declareStableParticles(const std::vector<int> &_pdg) {
  std::vector<int> pdg = _pdg;
  for (size_t i = 0; i < pdg.size(); i++) {
    int pyCode = pycomp_(pdg[i]);
    std::ostringstream pyCard;
    pyCard << "MDCY(" << pyCode << ",1)=0";
    std::cout << pyCard.str() << std::endl;
    call_pygive(pyCard.str());
  }
  return true;
}
//________________________________________________________________
bool Hydjet2Hadronizer::hadronize() { return false; }
bool Hydjet2Hadronizer::decay() { return true; }
bool Hydjet2Hadronizer::residualDecay() { return true; }
void Hydjet2Hadronizer::finalizeEvent() {}
void Hydjet2Hadronizer::statistics() {}
const char *Hydjet2Hadronizer::classname() const { return "gen::Hydjet2Hadronizer"; }

//________________________________________________________________
void Hydjet2Hadronizer::rotateEvtPlane() {
  const double pi = 3.14159265358979;
  phi0_ = 2. * pi * gen::pyr_(nullptr) - pi;
  sinphi0_ = sin(phi0_);
  cosphi0_ = cos(phi0_);
}

//_____________________________________________________________________
bool Hydjet2Hadronizer::get_particles(HepMC::GenEvent *evt) {
  LogDebug("Hydjet2") << " Number of sub events " << nsub_;
  LogDebug("Hydjet2") << " Number of hard events " << hj2->GetNjet();
  LogDebug("Hydjet2") << " Number of hard particles " << nhard_;
  LogDebug("Hydjet2") << " Number of soft particles " << nsoft_;
  LogDebug("Hydjet2") << " nhard_ + nsoft_ = " << nhard_ + nsoft_ << " Ntot = " << hj2->GetNtot() << endl;

  int ihy = 0;
  int isub_l = -1;
  int stab = 0;

  vector<HepMC::GenParticle *> particle(hj2->GetNtot());
  HepMC::GenVertex *sub_vertices = nullptr;

  while (ihy < hj2->GetNtot()) {
    if ((hj2->GetiJet().at(ihy)) != isub_l) {
      sub_vertices = new HepMC::GenVertex(HepMC::FourVector(0, 0, 0, 0), hj2->GetiJet().at(ihy));
      evt->add_vertex(sub_vertices);
      if (!evt->signal_process_vertex())
        evt->set_signal_process_vertex(sub_vertices);
      isub_l = hj2->GetiJet().at(ihy);
    }

    if ((convertStatusForComponents(
            (hj2->GetFinal()).at(ihy), (hj2->GetType()).at(ihy), (hj2->GetPythiaStatus().at(ihy)))) == 1)
      stab++;
    LogDebug("Hydjet2_array") << ihy << " MULTin ev.:" << hj2->GetNtot() << " SubEv.#" << hj2->GetiJet().at(ihy)
                              << " Part #" << ihy + 1 << ", PDG: " << hj2->GetPdg().at(ihy) << " (st. "
                              << convertStatus(hj2->GetPythiaStatus().at(ihy))
                              << ") mother=" << hj2->GetMotherIndex().at(ihy) + 1 << ", childs ("
                              << hj2->GetFirstDaughterIndex().at(ihy) + 1 << "-"
                              << hj2->GetLastDaughterIndex().at(ihy) + 1 << "), vtx (" << hj2->GetX().at(ihy) << ","
                              << hj2->GetY().at(ihy) << "," << hj2->GetZ().at(ihy) << ") " << std::endl;

    if ((hj2->GetMotherIndex().at(ihy)) <= 0) {
      particle.at(ihy) = build_hyjet2(ihy, ihy + 1);
      if (!sub_vertices)
        LogError("Hydjet2_array") << "##### HYDJET2: Vertex not initialized!";
      else
        sub_vertices->add_particle_out(particle.at(ihy));
      LogDebug("Hydjet2_array") << " ---> " << ihy + 1 << std::endl;
    } else {
      particle.at(ihy) = build_hyjet2(ihy, ihy + 1);
      int mid = hj2->GetMotherIndex().at(ihy);

      while (((mid + 1) < ihy) && (std::abs(hj2->GetPdg().at(mid)) < 100) &&
             ((hj2->GetFirstDaughterIndex().at(mid + 1)) <= ihy)) {
        mid++;
        LogDebug("Hydjet2_array") << "======== MID changed to " << mid
                                  << " ======== PDG(mid) = " << hj2->GetPdg().at(mid) << std::endl;
      }

      if (std::abs(hj2->GetPdg().at(mid)) < 100) {
        mid = hj2->GetMotherIndex().at(ihy);
        LogDebug("Hydjet2_array") << "======== MID changed BACK to " << mid
                                  << " ======== PDG(mid) = " << hj2->GetPdg().at(mid) << std::endl;
      }

      HepMC::GenParticle *mother = particle.at(mid);
      HepMC::GenVertex *prod_vertex = mother->end_vertex();

      if (!prod_vertex) {
        prod_vertex = build_hyjet2_vertex(ihy, (hj2->GetiJet().at(ihy)));
        prod_vertex->add_particle_in(mother);
        LogDebug("Hydjet2_array") << " <--- " << mid + 1 << std::endl;
        evt->add_vertex(prod_vertex);
      }
      prod_vertex->add_particle_out(particle.at(ihy));
      LogDebug("Hydjet2_array") << " ---" << mid + 1 << "---> " << ihy + 1 << std::endl;
    }
    ihy++;
  }

  LogDebug("Hydjet2_array") << " MULTin ev.:" << hj2->GetNtot() << ", last index: " << ihy - 1
                            << ", stable particles: " << stab << std::endl;
  return kTRUE;
}

//___________________________________________________________________
HepMC::GenParticle *Hydjet2Hadronizer::build_hyjet2(int index, int barcode) {
  // Build particle object corresponding to index in hyjets (soft+hard)
  double px0 = (hj2->GetPx()).at(index);
  double py0 = (hj2->GetPy()).at(index);

  double px = px0 * cosphi0_ - py0 * sinphi0_;
  double py = py0 * cosphi0_ + px0 * sinphi0_;

  HepMC::GenParticle *p = new HepMC::GenParticle(
      HepMC::FourVector(px,                        // px
                        py,                        // py
                        (hj2->GetPz()).at(index),  // pz
                        (hj2->GetE()).at(index)),  // E
      (hj2->GetPdg()).at(index),                   // id
      convertStatusForComponents(
          (hj2->GetFinal()).at(index), (hj2->GetType()).at(index), (hj2->GetPythiaStatus()).at(index))  // status
  );

  p->suggest_barcode(barcode);
  return p;
}

//___________________________________________________________________
HepMC::GenVertex *Hydjet2Hadronizer::build_hyjet2_vertex(int i, int id) {
  // build verteces for the hyjets stored events
  double x0 = (hj2->GetX()).at(i);
  double y0 = (hj2->GetY()).at(i);

  // convert to mm (as in PYTHIA6)
  const double fm_to_mm = 1e-12;
  double x = fm_to_mm * (x0 * cosphi0_ - y0 * sinphi0_);
  double y = fm_to_mm * (y0 * cosphi0_ + x0 * sinphi0_);
  double z = fm_to_mm * (hj2->GetZ()).at(i);
  double t = fm_to_mm * (hj2->GetT()).at(i);

  HepMC::GenVertex *vertex = new HepMC::GenVertex(HepMC::FourVector(x, y, z, t), id);

  return vertex;
}

//_____________________________________________________________________
void Hydjet2Hadronizer::add_heavy_ion_rec(HepMC::GenEvent *evt) {
  // heavy ion record in the final CMSSW Event
  int nproj = static_cast<int>((hj2->GetNpart()) / 2);
  int ntarg = static_cast<int>((hj2->GetNpart()) - nproj);

  HepMC::HeavyIon *hi = new HepMC::HeavyIon(nsub_,                              // Ncoll_hard/N of SubEvents
                                            nproj,                              // Npart_proj
                                            ntarg,                              // Npart_targ
                                            hj2->GetNbcol(),                    // Ncoll
                                            0,                                  // spectator_neutrons
                                            0,                                  // spectator_protons
                                            0,                                  // N_Nwounded_collisions
                                            0,                                  // Nwounded_N_collisions
                                            0,                                  // Nwounded_Nwounded_collisions
                                            hj2->GetBgen() * nuclear_radius(),  // impact_parameter in [fm]
                                            phi0_,                              // event_plane_angle
                                            hj2->GetPsiv3(),                    // eccentricity <<<---- psi for v3!!!
                                            Sigin                               // sigma_inel_NN
  );

  evt->set_heavy_ion(*hi);
  delete hi;
}
