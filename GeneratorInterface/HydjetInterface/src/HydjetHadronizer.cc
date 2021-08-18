/**
   \brief Interface to the HYDJET generator (since core v. 1.9.1), produces HepMC events
   \version 2.0
   \authors Camelia Mironov, Andrey Belyaev
*/

#include <iostream>
#include <cmath>

#include "boost/lexical_cast.hpp"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "GeneratorInterface/Core/interface/FortranInstance.h"
#include "GeneratorInterface/HydjetInterface/interface/HydjetHadronizer.h"
#include "GeneratorInterface/HydjetInterface/interface/HydjetWrapper.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "HepMC/IO_HEPEVT.h"
#include "HepMC/PythiaWrapper6_4.h"
#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"
#include "HepMC/SimpleVector.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

using namespace edm;
using namespace std;
using namespace gen;

namespace {
  int convertStatus(int st) {
    if (st <= 0)
      return 0;
    if (st <= 10)
      return 1;
    if (st <= 20)
      return 2;
    if (st <= 30)
      return 3;
    else
      return st;
  }
}  // namespace

const std::vector<std::string> HydjetHadronizer::theSharedResources = {edm::SharedResourceNames::kPythia6,
                                                                       gen::FortranInstance::kFortranInstance};

//_____________________________________________________________________
HydjetHadronizer::HydjetHadronizer(const ParameterSet& pset, edm::ConsumesCollector&& iC)
    : BaseHadronizer(pset),
      evt(nullptr),
      pset_(pset),
      abeamtarget_(pset.getParameter<double>("aBeamTarget")),
      angularspecselector_(pset.getParameter<int>("angularSpectrumSelector")),
      bfixed_(pset.getParameter<double>("bFixed")),
      bmax_(pset.getParameter<double>("bMax")),
      bmin_(pset.getParameter<double>("bMin")),
      cflag_(pset.getParameter<int>("cFlag")),
      embedding_(pset.getParameter<bool>("embeddingMode")),
      comenergy(pset.getParameter<double>("comEnergy")),
      doradiativeenloss_(pset.getParameter<bool>("doRadiativeEnLoss")),
      docollisionalenloss_(pset.getParameter<bool>("doCollisionalEnLoss")),
      fracsoftmult_(pset.getParameter<double>("fracSoftMultiplicity")),
      hadfreeztemp_(pset.getParameter<double>("hadronFreezoutTemperature")),
      hymode_(pset.getParameter<string>("hydjetMode")),
      maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint", 1)),
      maxlongy_(pset.getParameter<double>("maxLongitudinalRapidity")),
      maxtrany_(pset.getParameter<double>("maxTransverseRapidity")),
      nsub_(0),
      nhard_(0),
      nmultiplicity_(pset.getParameter<int>("nMultiplicity")),
      nsoft_(0),
      nquarkflavor_(pset.getParameter<int>("qgpNumQuarkFlavor")),
      pythiaPylistVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
      qgpt0_(pset.getParameter<double>("qgpInitialTemperature")),
      qgptau0_(pset.getParameter<double>("qgpProperTimeFormation")),
      phi0_(0.),
      sinphi0_(0.),
      cosphi0_(1.),
      rotate_(pset.getParameter<bool>("rotateEventPlane")),
      shadowingswitch_(pset.getParameter<int>("shadowingSwitch")),
      signn_(pset.getParameter<double>("sigmaInelNN")),
      fVertex_(nullptr),
      pythia6Service_(new Pythia6Service(pset)) {
  // Default constructor

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

  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity", 0);
  LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_;

  //Max number of events printed on verbosity level
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint", 0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_;

  if (embedding_) {
    cflag_ = 0;
    src_ = iC.consumes<CrossingFrame<edm::HepMCProduct> >(
        pset.getUntrackedParameter<edm::InputTag>("backgroundLabel", edm::InputTag("mix", "generatorSmeared")));
  }

  int cm = 1, va, vb, vc;
  HYJVER(cm, va, vb, vc);
  HepMC::HEPEVT_Wrapper::set_max_number_entries(4000);
}

//_____________________________________________________________________
HydjetHadronizer::~HydjetHadronizer() {
  // destructor
  call_pystat(1);
  delete pythia6Service_;
}

//_____________________________________________________________________
void HydjetHadronizer::doSetRandomEngine(CLHEP::HepRandomEngine* v) { pythia6Service_->setRandomEngine(v); }

//_____________________________________________________________________
void HydjetHadronizer::add_heavy_ion_rec(HepMC::GenEvent* evt) {
  // heavy ion record in the final CMSSW Event
  double npart = hyfpar.npart;
  int nproj = static_cast<int>(npart / 2);
  int ntarg = static_cast<int>(npart - nproj);

  HepMC::HeavyIon* hi = new HepMC::HeavyIon(nsub_,                           // Ncoll_hard/N of SubEvents
                                            nproj,                           // Npart_proj
                                            ntarg,                           // Npart_targ
                                            static_cast<int>(hyfpar.nbcol),  // Ncoll
                                            0,                               // spectator_neutrons
                                            0,                               // spectator_protons
                                            0,                               // N_Nwounded_collisions
                                            0,                               // Nwounded_N_collisions
                                            0,                               // Nwounded_Nwounded_collisions
                                            hyfpar.bgen * nuclear_radius(),  // impact_parameter in [fm]
                                            phi0_,                           // event_plane_angle
                                            0,  //hypsi3.psi3,                                   // eccentricity
                                            hyjpar.sigin  // sigma_inel_NN
  );

  evt->set_heavy_ion(*hi);
  delete hi;
}

//___________________________________________________________________
HepMC::GenParticle* HydjetHadronizer::build_hyjet(int index, int barcode) {
  // Build particle object corresponding to index in hyjets (soft+hard)
  double x0 = hyjets.phj[0][index];
  double y0 = hyjets.phj[1][index];

  double x = x0 * cosphi0_ - y0 * sinphi0_;
  double y = y0 * cosphi0_ + x0 * sinphi0_;

  HepMC::GenParticle* p = new HepMC::GenParticle(HepMC::FourVector(x,                      // px
                                                                   y,                      // py
                                                                   hyjets.phj[2][index],   // pz
                                                                   hyjets.phj[3][index]),  // E
                                                 hyjets.khj[1][index],                     // id
                                                 convertStatus(hyjets.khj[0][index]        // status
                                                               ));

  p->suggest_barcode(barcode);
  return p;
}

//___________________________________________________________________
HepMC::GenVertex* HydjetHadronizer::build_hyjet_vertex(int i, int id) {
  // build verteces for the hyjets stored events
  double x0 = hyjets.vhj[0][i];
  double y0 = hyjets.vhj[1][i];
  double x = x0 * cosphi0_ - y0 * sinphi0_;
  double y = y0 * cosphi0_ + x0 * sinphi0_;
  double z = hyjets.vhj[2][i];
  double t = hyjets.vhj[4][i];

  HepMC::GenVertex* vertex = new HepMC::GenVertex(HepMC::FourVector(x, y, z, t), id);
  return vertex;
}

//___________________________________________________________________

bool HydjetHadronizer::generatePartonsAndHadronize() {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);

  // generate single event
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
    LogInfo("MatchVtx") << " setting vertex "
                        << " aX " << aX << " aY " << aY << " aZ " << aZ << " aT " << aT << endl;
    fVertex_->set(aX, aY, aZ, aT);

    const HepMC::HeavyIon* hi = inev->heavy_ion();

    if (hi) {
      bfixed_ = (hi->impact_parameter()) / nuclear_radius();
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

  edm::LogInfo("HYDJETmode") << "##### HYDJET  nhsel = " << hyjpar.nhsel;
  edm::LogInfo("HYDJETfpart") << "##### HYDJET fpart = " << hyflow.fpart;
  edm::LogInfo("HYDJETtf") << "##### HYDJET hadron freez-out temp, Tf = " << hyflow.Tf;
  edm::LogInfo("HYDJETinTemp") << "##### HYDJET: QGP init temperature, T0 =" << pyqpar.T0u;
  edm::LogInfo("HYDJETinTau") << "##### HYDJET: QGP formation time,tau0 =" << pyqpar.tau0u;

  int ntry = 0;
  while (nsoft_ == 0 && nhard_ == 0) {
    if (ntry > 100) {
      edm::LogError("HydjetEmptyEvent") << "##### HYDJET: No Particles generated, Number of tries =" << ntry;

      // Throw an exception.  Use the EventCorruption exception since it maps onto SkipEvent
      // which is what we want to do here.

      std::ostringstream sstr;
      sstr << "HydjetHadronizerProducer: No particles generated after " << ntry << " tries.\n";
      edm::Exception except(edm::errors::EventCorruption, sstr.str());
      throw except;
    } else {
      HYEVNT(bfixed_);
      nsoft_ = hyfpar.nhyd;
      nsub_ = hyjpar.njet;
      nhard_ = hyfpar.npyt;
      ++ntry;
    }
  }

  if (hyjpar.nhsel < 3)
    nsub_++;

  // event information
  HepMC::GenEvent* evt = new HepMC::GenEvent();

  if (nhard_ > 0 || nsoft_ > 0)
    get_particles(evt);

  evt->set_signal_process_id(pypars.msti[0]);  // type of the process
  evt->set_event_scale(pypars.pari[16]);       // Q^2
  add_heavy_ion_rec(evt);

  if (fVertex_) {
    // generate new vertex & apply the shift

    // Copy the HepMC::GenEvent
    std::unique_ptr<edm::HepMCProduct> HepMCEvt(new edm::HepMCProduct(evt));

    HepMCEvt->applyVtxGen(fVertex_);
    evt = new HepMC::GenEvent((*HepMCEvt->GetEvent()));
  }

  HepMC::HEPEVT_Wrapper::check_hepevt_consistency();
  LogDebug("HEPEVT_info") << "Ev numb: " << HepMC::HEPEVT_Wrapper::event_number()
                          << " Entries number: " << HepMC::HEPEVT_Wrapper::number_entries() << " Max. entries "
                          << HepMC::HEPEVT_Wrapper::max_number_entries() << std::endl;

  event().reset(evt);
  return true;
}

//_____________________________________________________________________
bool HydjetHadronizer::get_particles(HepMC::GenEvent* evt) {
  // Hard particles. The first nhard_ lines from hyjets array.
  // Pythia/Pyquen sub-events (sub-collisions) for a given event
  // Return T/F if success/failure
  // Create particles from lujet entries, assign them into vertices and
  // put the vertices in the GenEvent, for each SubEvent
  // The SubEvent information is kept by storing indeces of main vertices
  // of subevents as a vector in GenHIEvent.

  LogDebug("SubEvent") << " Number of sub events " << nsub_;
  LogDebug("Hydjet") << " Number of hard events " << hyjpar.njet;
  LogDebug("Hydjet") << " Number of hard particles " << nhard_;
  LogDebug("Hydjet") << " Number of soft particles " << nsoft_;
  LogDebug("Hydjet") << " nhard_ + nsoft_ = " << nhard_ + nsoft_ << " hyjets.nhj = " << hyjets.nhj << endl;

  int ihy = 0;
  int isub = -1;
  int isub_l = -1;
  int stab = 0;

  vector<HepMC::GenParticle*> primary_particle(hyjets.nhj);
  vector<HepMC::GenParticle*> particle(hyjets.nhj);

  HepMC::GenVertex* sub_vertices = new HepMC::GenVertex(HepMC::FourVector(0, 0, 0, 0), 0);  // just initialization

  // contain the last index in for each subevent
  vector<int> index(nsub_);

  while (ihy < hyjets.nhj) {
    isub = std::floor((hyjets.khj[2][ihy] / 50000));
    int hjoffset = isub * 50000;

    if (isub != isub_l) {
      sub_vertices = new HepMC::GenVertex(HepMC::FourVector(0, 0, 0, 0), isub);
      evt->add_vertex(sub_vertices);
      if (!evt->signal_process_vertex())
        evt->set_signal_process_vertex(sub_vertices);
      index[isub] = ihy - 1;
      isub_l = isub;
    }

    if (convertStatus(hyjets.khj[0][ihy]) == 1)
      stab++;
    LogDebug("Hydjet_array") << ihy << " MULTin ev.:" << hyjets.nhj << " SubEv.#" << isub << " Part #" << ihy + 1
                             << ", PDG: " << hyjets.khj[1][ihy] << " (st. " << convertStatus(hyjets.khj[0][ihy])
                             << ") mother=" << hyjets.khj[2][ihy] - (isub * 50000) + index[isub] + 1 << " ("
                             << hyjets.khj[2][ihy] << "), childs ("
                             << hyjets.khj[3][ihy] - (isub * 50000) + index[isub] + 1 << "-"
                             << hyjets.khj[4][ihy] - (isub * 50000) + index[isub] + 1 << "), vtx ("
                             << hyjets.vhj[0][ihy] << "," << hyjets.vhj[1][ihy] << "," << hyjets.vhj[2][ihy] << ") "
                             << std::endl;

    if (hyjets.khj[2][ihy] == 0) {
      primary_particle[ihy] = build_hyjet(ihy, ihy + 1);
      sub_vertices->add_particle_out(primary_particle[ihy]);
      LogDebug("Hydjet_array") << " ---> " << ihy + 1 << std::endl;
    } else {
      particle[ihy] = build_hyjet(ihy, ihy + 1);
      int mid = hyjets.khj[2][ihy] - hjoffset + index[isub];
      int mid_t = mid;
      while ((mid < ihy) && (hyjets.khj[1][mid] < 100) && (hyjets.khj[3][mid + 1] - hjoffset + index[isub] == ihy))
        mid++;
      if (hyjets.khj[1][mid] < 100)
        mid = mid_t;

      HepMC::GenParticle* mother = primary_particle.at(mid);
      HepMC::GenVertex* prods = build_hyjet_vertex(ihy, isub);

      if (!mother) {
        mother = particle[mid];
        primary_particle[mid] = mother;
      }

      HepMC::GenVertex* prod_vertex = mother->end_vertex();
      if (!prod_vertex) {
        prod_vertex = prods;
        prod_vertex->add_particle_in(mother);
        LogDebug("Hydjet_array") << " <--- " << mid + 1 << std::endl;
        evt->add_vertex(prod_vertex);
        prods = nullptr;
      }

      prod_vertex->add_particle_out(particle[ihy]);
      LogDebug("Hydjet_array") << " ---" << mid + 1 << "---> " << ihy + 1 << std::endl;

      if (prods)
        delete prods;
    }
    ihy++;
  }
  LogDebug("Hydjet_array") << " MULTin ev.:" << hyjets.nhj << ", last index: " << ihy - 1
                           << ", Sub events: " << isub + 1 << ", stable particles: " << stab << std::endl;

  return true;
}

//______________________________________________________________
bool HydjetHadronizer::call_hyinit(double energy, double a, int ifb, double bmin, double bmax, double bfix, int nh) {
  // initialize hydjet

  pydatr.mrpy[2] = 1;
  HYINIT(energy, a, ifb, bmin, bmax, bfix, nh);
  return true;
}

//______________________________________________________________
bool HydjetHadronizer::hydjet_init(const ParameterSet& pset) {
  // set hydjet options

  // hydjet running mode mode
  // kHydroOnly --- nhsel=0 jet production off (pure HYDRO event), nhsel=0
  // kHydroJets --- nhsle=1 jet production on, jet quenching off (HYDRO+njet*PYTHIA events)
  // kHydroQJet --- nhsel=2 jet production & jet quenching on (HYDRO+njet*PYQUEN events)
  // kJetsOnly  --- nhsel=3 jet production on, jet quenching off, HYDRO off (njet*PYTHIA events)
  // kQJetsOnly --- nhsel=4 jet production & jet quenching on, HYDRO off (njet*PYQUEN events)

  if (hymode_ == "kHydroOnly")
    hyjpar.nhsel = 0;
  else if (hymode_ == "kHydroJets")
    hyjpar.nhsel = 1;
  else if (hymode_ == "kHydroQJets")
    hyjpar.nhsel = 2;
  else if (hymode_ == "kJetsOnly")
    hyjpar.nhsel = 3;
  else if (hymode_ == "kQJetsOnly")
    hyjpar.nhsel = 4;
  else
    hyjpar.nhsel = 2;

  // fraction of soft hydro induced multiplicity
  hyflow.fpart = fracsoftmult_;

  // hadron freez-out temperature
  hyflow.Tf = hadfreeztemp_;

  // maximum longitudinal collective rapidity
  hyflow.ylfl = maxlongy_;

  // maximum transverse collective rapidity
  hyflow.ytfl = maxtrany_;

  // shadowing on=1, off=0
  hyjpar.ishad = shadowingswitch_;

  // set inelastic nucleon-nucleon cross section
  hyjpar.sigin = signn_;

  // angular emitted gluon spectrum selection
  pyqpar.ianglu = angularspecselector_;

  // number of active quark flavors in qgp
  pyqpar.nfu = nquarkflavor_;

  // initial temperature of QGP
  pyqpar.T0u = qgpt0_;

  // proper time of QGP formation
  pyqpar.tau0u = qgptau0_;

  // type of medium induced partonic energy loss
  if (doradiativeenloss_ && docollisionalenloss_) {
    edm::LogInfo("HydjetEnLoss") << "##### Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0;
  } else if (doradiativeenloss_) {
    edm::LogInfo("HydjetenLoss") << "##### Only RADIATIVE partonic energy loss ON ####";
    pyqpar.ienglu = 1;
  } else if (docollisionalenloss_) {
    edm::LogInfo("HydjetEnLoss") << "##### Only COLLISIONAL partonic energy loss ON ####";
    pyqpar.ienglu = 2;
  } else {
    edm::LogInfo("HydjetEnLoss") << "##### Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0;
  }
  return true;
}

//_____________________________________________________________________

bool HydjetHadronizer::readSettings(int) {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);
  pythia6Service_->setGeneralParams();

  return true;
}

//_____________________________________________________________________

bool HydjetHadronizer::initializeForInternalPartons() {
  Pythia6Service::InstanceWrapper guard(pythia6Service_);
  // pythia6Service_->setGeneralParams();

  // the input impact parameter (bxx_) is in [fm]; transform in [fm/RA] for hydjet usage
  const float ra = nuclear_radius();
  LogInfo("RAScaling") << "Nuclear radius(RA) =  " << ra;
  bmin_ /= ra;
  bmax_ /= ra;
  bfixed_ /= ra;

  // hydjet running options
  hydjet_init(pset_);
  // initialize hydjet
  LogInfo("HYDJETinAction") << "##### Calling HYINIT(" << comenergy << "," << abeamtarget_ << "," << cflag_ << ","
                            << bmin_ << "," << bmax_ << "," << bfixed_ << "," << nmultiplicity_ << ") ####";
  call_hyinit(comenergy, abeamtarget_, cflag_, bmin_, bmax_, bfixed_, nmultiplicity_);
  return true;
}

bool HydjetHadronizer::declareStableParticles(const std::vector<int>& _pdg) {
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
void HydjetHadronizer::rotateEvtPlane() {
  const double pi = 3.14159265358979;
  phi0_ = 2. * pi * gen::pyr_(nullptr) - pi;
  sinphi0_ = sin(phi0_);
  cosphi0_ = cos(phi0_);
}

//________________________________________________________________
bool HydjetHadronizer::hadronize() { return false; }

bool HydjetHadronizer::decay() { return true; }

bool HydjetHadronizer::residualDecay() { return true; }

void HydjetHadronizer::finalizeEvent() {}

void HydjetHadronizer::statistics() {}

const char* HydjetHadronizer::classname() const { return "gen::HydjetHadronizer"; }
