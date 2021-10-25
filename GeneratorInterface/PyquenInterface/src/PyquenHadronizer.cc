#include <iostream>
#include <ctime>

#include "GeneratorInterface/PyquenInterface/interface/PyquenHadronizer.h"

#include "GeneratorInterface/Core/interface/FortranInstance.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenWrapper.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/HiGenCommon/interface/HiGenEvtSelectorFactory.h"

#include "HepMC/IO_HEPEVT.h"
#include "HepMC/PythiaWrapper.h"

using namespace gen;
using namespace edm;
using namespace std;

HepMC::IO_HEPEVT pyquen_hepevtio;

const std::vector<std::string> PyquenHadronizer::theSharedResources = {edm::SharedResourceNames::kPythia6,
                                                                       gen::FortranInstance::kFortranInstance};

PyquenHadronizer ::PyquenHadronizer(const ParameterSet& pset, edm::ConsumesCollector&& iC)
    : BaseHadronizer(pset),
      pset_(pset),
      abeamtarget_(pset.getParameter<double>("aBeamTarget")),
      angularspecselector_(pset.getParameter<int>("angularSpectrumSelector")),
      bmin_(pset.getParameter<double>("bMin")),
      bmax_(pset.getParameter<double>("bMax")),
      bfixed_(pset.getParameter<double>("bFixed")),
      cflag_(pset.getParameter<int>("cFlag")),
      comenergy(pset.getParameter<double>("comEnergy")),
      doquench_(pset.getParameter<bool>("doQuench")),
      doradiativeenloss_(pset.getParameter<bool>("doRadiativeEnLoss")),
      docollisionalenloss_(pset.getParameter<bool>("doCollisionalEnLoss")),
      doIsospin_(pset.getParameter<bool>("doIsospin")),
      protonSide_(pset.getUntrackedParameter<int>("protonSide", 0)),
      embedding_(pset.getParameter<bool>("embeddingMode")),
      evtPlane_(0),
      nquarkflavor_(pset.getParameter<int>("qgpNumQuarkFlavor")),
      qgpt0_(pset.getParameter<double>("qgpInitialTemperature")),
      qgptau0_(pset.getParameter<double>("qgpProperTimeFormation")),
      maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint", 1)),
      fVertex_(nullptr),
      pythiaHepMCVerbosity_(pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false)),
      pythiaPylistVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
      pythia6Service_(new Pythia6Service(pset)),
      filterType_(pset.getUntrackedParameter<string>("filterType", "None")) {
  if (pset.exists("signalVtx"))
    signalVtx_ = pset.getUntrackedParameter<std::vector<double> >("signalVtx");

  if (signalVtx_.size() == 4) {
    if (!fVertex_)
      fVertex_ = new HepMC::FourVector();
    LogDebug("EventSignalVertex") << "Setting event signal vertex "
                                  << " x = " << signalVtx_.at(0) << " y = " << signalVtx_.at(1)
                                  << "  z= " << signalVtx_.at(2) << " t = " << signalVtx_.at(3) << endl;
    fVertex_->set(signalVtx_.at(0), signalVtx_.at(1), signalVtx_.at(2), signalVtx_.at(3));
  }

  // Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_ << endl;
  // HepMC event verbosity Level
  pythiaHepMCVerbosity_ = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false);
  LogDebug("HepMCverbosity") << "Pythia HepMC verbosity = " << pythiaHepMCVerbosity_ << endl;

  //Max number of events printed on verbosity level
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint", 0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_ << endl;

  if (embedding_) {
    cflag_ = 0;
    src_ = iC.consumes<CrossingFrame<edm::HepMCProduct> >(
        pset.getUntrackedParameter<edm::InputTag>("backgroundLabel", edm::InputTag("mix", "generatorSmeared")));
  }
  selector_ = HiGenEvtSelectorFactory::get(filterType_, pset);

  int cm = 1, va, vb, vc;
  PYQVER(cm, va, vb, vc);
  //HepMC::HEPEVT_Wrapper::set_max_number_entries(4000);
}

//_____________________________________________________________________
PyquenHadronizer::~PyquenHadronizer() {
  // distructor
  call_pystat(1);

  delete pythia6Service_;
}

//_____________________________________________________________________
void PyquenHadronizer::doSetRandomEngine(CLHEP::HepRandomEngine* v) { pythia6Service_->setRandomEngine(v); }

//_____________________________________________________________________
void PyquenHadronizer::add_heavy_ion_rec(HepMC::GenEvent* evt) {
  HepMC::HeavyIon* hi = new HepMC::HeavyIon(1,            // Ncoll_hard
                                            -1,           // Npart_proj
                                            -1,           // Npart_targ
                                            1,            // Ncoll
                                            -1,           // spectator_neutrons
                                            -1,           // spectator_protons
                                            -1,           // N_Nwounded_collisions
                                            -1,           // Nwounded_N_collisions
                                            -1,           // Nwounded_Nwounded_collisions
                                            plfpar.bgen,  // impact_parameter in [fm]
                                            evtPlane_,    // event_plane_angle
                                            0,            // eccentricity
                                            -1            // sigma_inel_NN
  );

  evt->set_heavy_ion(*hi);

  delete hi;
}

//_____________________________________________________________________
bool PyquenHadronizer::generatePartonsAndHadronize() {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);

  // Not possible to retrieve impact paramter and event plane info
  // at this part, need to overwrite filter() in
  // PyquenGeneratorFilter

  if (embedding_) {
    const edm::Event& e = getEDMEvent();
    HepMC::GenVertex* genvtx = nullptr;
    const HepMC::GenEvent* inev = nullptr;
    Handle<CrossingFrame<HepMCProduct> > cf;
    e.getByToken(src_, cf);
    MixCollection<HepMCProduct> mix(cf.product());
    if (mix.size() < 1) {
      throw cms::Exception("MatchVtx") << "Mixing has " << mix.size() << " sub-events, should have been at least 1"
                                       << endl;
    }
    const HepMCProduct& bkg = mix.getObject(0);
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

    //LogInfo("MatchVtx")
    std::cout << " setting vertex "
              << " aX " << aX << " aY " << aY << " aZ " << aZ << " aT " << aT << endl;
    fVertex_->set(aX, aY, aZ, aT);

    const HepMC::HeavyIon* hi = inev->heavy_ion();

    if (hi) {
      bfixed_ = hi->impact_parameter();
      evtPlane_ = hi->event_plane_angle();
    } else {
      LogWarning("EventEmbedding") << "Background event does not have heavy ion record!";
    }
  }

  // Generate PYQUEN event
  // generate single partonic PYTHIA jet event

  // Take into account whether it's a nn or pp or pn interaction
  if (doIsospin_) {
    string projN = "p";
    string targN = "p";
    if (protonSide_ != 1)
      projN = nucleon();
    if (protonSide_ != 2)
      targN = nucleon();
    call_pyinit("CMS", projN.data(), targN.data(), comenergy);
  }
  call_pyevnt();

  // call PYQUEN to apply parton rescattering and energy loss
  // if doQuench=FALSE, it is pure PYTHIA
  if (doquench_) {
    PYQUEN(abeamtarget_, cflag_, bfixed_, bmin_, bmax_);
    edm::LogInfo("PYQUENinAction") << "##### Calling PYQUEN(" << abeamtarget_ << "," << cflag_ << "," << bfixed_
                                   << ") ####";
  } else {
    edm::LogInfo("PYQUENinAction") << "##### Calling PYQUEN: QUENCHING OFF!! This is just PYTHIA !!!! ####";
  }

  // call PYTHIA to finish the hadronization
  pyexec_();

  // fill the HEPEVT with the PYJETS event record
  call_pyhepc(1);

  // event information
  // pyquen_hepevtio.set_trust_mothers_before_daughters(true);
  HepMC::GenEvent* evt = pyquen_hepevtio.read_next_event();

  evt->set_signal_process_id(pypars.msti[0]);  // type of the process
  evt->set_event_scale(pypars.pari[16]);       // Q^2

  if (embedding_)
    rotateEvtPlane(evt, evtPlane_);
  add_heavy_ion_rec(evt);

  if (fVertex_) {
    // Copy the HepMC::GenEvent
    std::unique_ptr<edm::HepMCProduct> HepMCEvt(new edm::HepMCProduct(evt));

    HepMCEvt->applyVtxGen(fVertex_);
    evt = new HepMC::GenEvent((*HepMCEvt->GetEvent()));
  }

  //  HepMC::HEPEVT_Wrapper::check_hepevt_consistency();

  event().reset(evt);

  return true;
}

bool PyquenHadronizer::readSettings(int) {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);
  pythia6Service_->setGeneralParams();
  pythia6Service_->setCSAParams();

  //Proton to Nucleon fraction
  pfrac_ = 1. / (1.98 + 0.015 * pow(abeamtarget_, 2. / 3));

  //initialize pythia
  pyqpythia_init(pset_);

  //initilize pyquen
  pyquen_init(pset_);

  return true;
}

bool PyquenHadronizer::initializeForInternalPartons() {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);

  // Call PYTHIA
  call_pyinit("CMS", "p", "p", comenergy);

  return true;
}

//_____________________________________________________________________
bool PyquenHadronizer::pyqpythia_init(const ParameterSet& pset) {
  //Turn Hadronization Off whether or not there is quenching
  // PYEXEC is called later anyway
  string sHadOff("MSTP(111)=0");
  gen::call_pygive(sHadOff);

  return true;
}

//_________________________________________________________________
bool PyquenHadronizer::pyquen_init(const ParameterSet& pset) {
  // PYQUEN initialization

  // angular emitted gluon  spectrum selection
  pyqpar.ianglu = angularspecselector_;

  // type of medium induced partonic energy loss
  if (doradiativeenloss_ && docollisionalenloss_) {
    edm::LogInfo("PYQUENinEnLoss") << "##### PYQUEN: Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0;
  } else if (doradiativeenloss_) {
    edm::LogInfo("PYQUENinRad") << "##### PYQUEN: Only RADIATIVE partonic energy loss ON ####";
    pyqpar.ienglu = 1;
  } else if (docollisionalenloss_) {
    edm::LogInfo("PYQUENinColl") << "##### PYQUEN: Only COLLISIONAL partonic energy loss ON ####";
    pyqpar.ienglu = 2;
  } else {
    edm::LogInfo("PYQUENinEnLoss") << "##### PYQUEN: Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0;
  }

  // number of active quark flavors in qgp
  pyqpar.nfu = nquarkflavor_;
  // initial temperature of QGP
  pyqpar.T0u = qgpt0_;
  // proper time of QGP formation
  pyqpar.tau0u = qgptau0_;

  return true;
}

const char* PyquenHadronizer::nucleon() {
  int* dummy = nullptr;
  double random = gen::pyr_(dummy);
  const char* nuc = nullptr;
  if (random > pfrac_)
    nuc = "n";
  else
    nuc = "p";

  return nuc;
}

void PyquenHadronizer::rotateEvtPlane(HepMC::GenEvent* evt, double angle) {
  double sinphi0 = sin(angle);
  double cosphi0 = cos(angle);

  for (HepMC::GenEvent::vertex_iterator vt = evt->vertices_begin(); vt != evt->vertices_end(); ++vt) {
    double x0 = (*vt)->position().x();
    double y0 = (*vt)->position().y();
    double z = (*vt)->position().z();
    double t = (*vt)->position().t();

    double x = x0 * cosphi0 - y0 * sinphi0;
    double y = y0 * cosphi0 + x0 * sinphi0;

    (*vt)->set_position(HepMC::FourVector(x, y, z, t));
  }

  for (HepMC::GenEvent::particle_iterator vt = evt->particles_begin(); vt != evt->particles_end(); ++vt) {
    double x0 = (*vt)->momentum().x();
    double y0 = (*vt)->momentum().y();
    double z = (*vt)->momentum().z();
    double t = (*vt)->momentum().t();

    double x = x0 * cosphi0 - y0 * sinphi0;
    double y = y0 * cosphi0 + x0 * sinphi0;

    (*vt)->set_momentum(HepMC::FourVector(x, y, z, t));
  }
}

bool PyquenHadronizer::declareStableParticles(const std::vector<int>& _pdg) {
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

//____________________________________________________________________

bool PyquenHadronizer::hadronize() { return false; }

bool PyquenHadronizer::decay() { return true; }

bool PyquenHadronizer::residualDecay() { return true; }

void PyquenHadronizer::finalizeEvent() {}

void PyquenHadronizer::statistics() {}

const char* PyquenHadronizer::classname() const { return "gen::PyquenHadronizer"; }
