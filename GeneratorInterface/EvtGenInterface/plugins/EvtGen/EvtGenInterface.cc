// Class Based on EvtGenInterface(LHC).
//
// Created March 2014
//
// This class is a modification of the original EvtGenInterface which was developed for EvtGenLHC 9.1.
// The modifications for EvtGen 1.3.0 are implemented by Ian M. Nugent
// I would like to thank the EvtGen developers, in particular John Black, and Mikhail Kirsanov for their assistance.
//
// January 2015: Setting of coherent or incoherent B mixing included by Eduard Burelo
// January 2015: Adding new feature to allow users to provide new evtgen models

#include "GeneratorInterface/EvtGenInterface/plugins/EvtGenUserModels/EvtModelUserReg.h"
#include "GeneratorInterface/EvtGenInterface/plugins/EvtGen/EvtGenInterface.h"

#include "GeneratorInterface/EvtGenInterface/interface/EvtGenFactory.h"
#include "GeneratorInterface/EvtGenInterface/interface/myEvtRandomEngine.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandFlat.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "EvtGen/EvtGen.hh"
#include "EvtGenExternal/EvtExternalGenList.hh"
#include "EvtGenBase/EvtAbsRadCorr.hh"
#include "EvtGenBase/EvtDecayBase.hh"
#include "EvtGenBase/EvtId.hh"
#include "EvtGenBase/EvtDecayTable.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtParticleFactory.hh"
#include "EvtGenBase/EvtHepMCEvent.hh"

#include "HepPID/ParticleIDTranslations.hh"

#include "TString.h"
#include <string>
#include <cstdlib>
#include <cstdio>

using namespace gen;
using namespace edm;

CLHEP::HepRandomEngine* EvtGenInterface::fRandomEngine;

EvtGenInterface::EvtGenInterface(const ParameterSet& pset) {
  fPSet = new ParameterSet(pset);
  the_engine = new myEvtRandomEngine(nullptr);
}

void EvtGenInterface::SetDefault_m_PDGs() {
  // fill up default list of particles to be declared stable in the "master generator"
  // these are assumed to be PDG ID's
  //
  // Note: Pythia6's kc=43, 44, and 84 commented out because they're obsolete (per S.Mrenna)
  //
  m_PDGs.push_back(300553);
  m_PDGs.push_back(511);
  m_PDGs.push_back(521);
  m_PDGs.push_back(523);
  m_PDGs.push_back(513);
  m_PDGs.push_back(533);
  m_PDGs.push_back(531);  //B_s0

  m_PDGs.push_back(15);

  m_PDGs.push_back(413);
  m_PDGs.push_back(423);
  m_PDGs.push_back(433);
  m_PDGs.push_back(411);
  m_PDGs.push_back(421);
  m_PDGs.push_back(431);
  m_PDGs.push_back(10411);
  m_PDGs.push_back(10421);
  m_PDGs.push_back(10413);
  m_PDGs.push_back(10423);
  m_PDGs.push_back(20413);
  m_PDGs.push_back(20423);

  m_PDGs.push_back(415);
  m_PDGs.push_back(425);
  m_PDGs.push_back(10431);
  m_PDGs.push_back(20433);
  m_PDGs.push_back(10433);
  m_PDGs.push_back(435);

  m_PDGs.push_back(310);
  m_PDGs.push_back(311);
  m_PDGs.push_back(313);
  m_PDGs.push_back(323);
  m_PDGs.push_back(10321);
  m_PDGs.push_back(10311);
  m_PDGs.push_back(10313);
  m_PDGs.push_back(10323);
  m_PDGs.push_back(20323);
  m_PDGs.push_back(20313);
  m_PDGs.push_back(325);
  m_PDGs.push_back(315);

  m_PDGs.push_back(100313);
  m_PDGs.push_back(100323);
  m_PDGs.push_back(30313);
  m_PDGs.push_back(30323);
  m_PDGs.push_back(30343);
  m_PDGs.push_back(30353);
  m_PDGs.push_back(30363);

  m_PDGs.push_back(111);
  m_PDGs.push_back(221);
  m_PDGs.push_back(113);
  m_PDGs.push_back(213);
  m_PDGs.push_back(223);
  m_PDGs.push_back(331);
  m_PDGs.push_back(333);
  m_PDGs.push_back(20213);
  m_PDGs.push_back(20113);
  m_PDGs.push_back(215);
  m_PDGs.push_back(115);
  m_PDGs.push_back(10213);
  m_PDGs.push_back(10113);
  m_PDGs.push_back(9000111);  // PDG ID = 9000111, Pythia6 ID = 10111
  m_PDGs.push_back(9000211);  // PDG ID = 9000211, Pythia6 ID = 10211
  m_PDGs.push_back(9010221);  // PDG ID = 9010211, Pythia6 ID = ???
  m_PDGs.push_back(10221);
  m_PDGs.push_back(20223);
  m_PDGs.push_back(20333);
  m_PDGs.push_back(225);
  m_PDGs.push_back(9020221);  // PDG ID = 9020211, Pythia6 ID = ???
  m_PDGs.push_back(335);
  m_PDGs.push_back(10223);
  m_PDGs.push_back(10333);
  m_PDGs.push_back(100213);
  m_PDGs.push_back(100113);

  m_PDGs.push_back(441);
  m_PDGs.push_back(100441);
  m_PDGs.push_back(443);
  m_PDGs.push_back(100443);
  m_PDGs.push_back(9000443);
  m_PDGs.push_back(9010443);
  m_PDGs.push_back(9020443);
  m_PDGs.push_back(10441);
  m_PDGs.push_back(20443);
  m_PDGs.push_back(445);

  m_PDGs.push_back(30443);
  m_PDGs.push_back(551);
  m_PDGs.push_back(553);
  m_PDGs.push_back(100553);
  m_PDGs.push_back(200553);
  m_PDGs.push_back(10551);
  m_PDGs.push_back(20553);
  m_PDGs.push_back(555);
  m_PDGs.push_back(10553);

  m_PDGs.push_back(110551);
  m_PDGs.push_back(120553);
  m_PDGs.push_back(100555);
  m_PDGs.push_back(210551);
  m_PDGs.push_back(220553);
  m_PDGs.push_back(200555);
  m_PDGs.push_back(30553);
  m_PDGs.push_back(20555);

  m_PDGs.push_back(557);
  m_PDGs.push_back(130553);
  m_PDGs.push_back(120555);
  m_PDGs.push_back(100557);
  m_PDGs.push_back(110553);
  m_PDGs.push_back(210553);
  m_PDGs.push_back(10555);
  m_PDGs.push_back(110555);

  m_PDGs.push_back(4122);
  m_PDGs.push_back(4132);
  // m_PDGs.push_back( 84 ); // obsolete
  m_PDGs.push_back(4112);
  m_PDGs.push_back(4212);
  m_PDGs.push_back(4232);
  m_PDGs.push_back(4222);
  m_PDGs.push_back(4322);
  m_PDGs.push_back(4312);

  m_PDGs.push_back(102132);
  m_PDGs.push_back(103124);
  m_PDGs.push_back(203122);
  m_PDGs.push_back(103122);
  m_PDGs.push_back(123122);
  m_PDGs.push_back(213122);
  m_PDGs.push_back(103126);
  m_PDGs.push_back(13212);
  //m_PDGs.push_back( 13241 ); unknown particle -typo?

  m_PDGs.push_back(203126);
  m_PDGs.push_back(102134);
  m_PDGs.push_back(3122);
  m_PDGs.push_back(3222);
  m_PDGs.push_back(2214);
  m_PDGs.push_back(2224);
  m_PDGs.push_back(3324);
  m_PDGs.push_back(2114);
  m_PDGs.push_back(1114);
  m_PDGs.push_back(3112);
  m_PDGs.push_back(3212);
  m_PDGs.push_back(3114);
  m_PDGs.push_back(3224);
  m_PDGs.push_back(3214);
  m_PDGs.push_back(3216);
  m_PDGs.push_back(3322);
  m_PDGs.push_back(3312);
  m_PDGs.push_back(3314);
  m_PDGs.push_back(3334);

  m_PDGs.push_back(4114);
  m_PDGs.push_back(4214);
  m_PDGs.push_back(4224);
  m_PDGs.push_back(4314);
  m_PDGs.push_back(4324);
  m_PDGs.push_back(4332);
  m_PDGs.push_back(4334);
  //m_PDGs.push_back( 43 ); // obsolete (?)
  //m_PDGs.push_back( 44 ); // obsolete (?)
  m_PDGs.push_back(10443);

  m_PDGs.push_back(5122);
  m_PDGs.push_back(5132);
  m_PDGs.push_back(5232);
  m_PDGs.push_back(5332);
  m_PDGs.push_back(5222);
  m_PDGs.push_back(5112);
  m_PDGs.push_back(5212);
  m_PDGs.push_back(541);
  m_PDGs.push_back(14122);
  m_PDGs.push_back(14124);
  m_PDGs.push_back(5312);
  m_PDGs.push_back(5322);
  m_PDGs.push_back(10521);
  m_PDGs.push_back(20523);
  m_PDGs.push_back(10523);

  m_PDGs.push_back(525);
  m_PDGs.push_back(10511);
  m_PDGs.push_back(20513);
  m_PDGs.push_back(10513);
  m_PDGs.push_back(515);
  m_PDGs.push_back(10531);
  m_PDGs.push_back(20533);
  m_PDGs.push_back(10533);
  m_PDGs.push_back(535);
  m_PDGs.push_back(543);
  m_PDGs.push_back(545);
  m_PDGs.push_back(5114);
  m_PDGs.push_back(5224);
  m_PDGs.push_back(5214);
  m_PDGs.push_back(5314);
  m_PDGs.push_back(5324);
  m_PDGs.push_back(5334);
  m_PDGs.push_back(10541);
  m_PDGs.push_back(10543);
  m_PDGs.push_back(20543);

  m_PDGs.push_back(4424);
  m_PDGs.push_back(4422);
  m_PDGs.push_back(4414);
  m_PDGs.push_back(4412);
  m_PDGs.push_back(4432);
  m_PDGs.push_back(4434);

  m_PDGs.push_back(130);

  for (unsigned int i = 0; i < m_PDGs.size(); i++) {
    int pdt = HepPID::translatePythiatoPDT(m_PDGs.at(i));
    if (pdt != m_PDGs.at(i))
      m_PDGs.at(i) = pdt;
  }
}

EvtGenInterface::~EvtGenInterface() {}

void EvtGenInterface::init() {
  // flags for pythia8
  fSpecialSettings.push_back("Pythia8:ParticleDecays:mixB = off");
  //

  edm::FileInPath decay_table(fPSet->getParameter<std::string>("decay_table"));
  edm::FileInPath pdt(fPSet->getParameter<edm::FileInPath>("particle_property_file"));

  bool usePythia = fPSet->getUntrackedParameter<bool>("use_internal_pythia", true);
  bool useTauola = fPSet->getUntrackedParameter<bool>("use_internal_tauola", true);
  bool usePhotos = fPSet->getUntrackedParameter<bool>("use_internal_photos", true);

  //Setup evtGen following instructions on http://evtgen.warwick.ac.uk/docs/external/
  bool convertPythiaCodes = fPSet->getUntrackedParameter<bool>(
      "convertPythiaCodes", true);  // Specify if we want to use Pythia 6 physics codes for decays
  std::string pythiaDir =
      std::getenv("PYTHIA8DATA");  // Specify the pythia xml data directory to use the default PYTHIA8DATA location
  if (pythiaDir.empty()) {
    edm::LogError("EvtGenInterface::~EvtGenInterface")
        << "EvtGenInterface::init() PYTHIA8DATA not defined. Terminating program ";
    exit(0);
  }
  std::string photonType("gamma");  // Specify the photon type for Photos
  bool useEvtGenRandom(true);       // Specify if we want to use the EvtGen random number engine for these generators

  // Set up the default external generator list: Photos, Pythia and/or Tauola
  EvtExternalGenList genList(convertPythiaCodes, pythiaDir, photonType, useEvtGenRandom);
  EvtAbsRadCorr* radCorrEngine = nullptr;
  if (usePhotos)
    radCorrEngine = genList.getPhotosModel();                        // Get interface to radiative correction engine
  std::list<EvtDecayBase*> extraModels = genList.getListOfModels();  // get interface to Pythia and Tauola
  std::list<EvtDecayBase*> myExtraModels;
  for (unsigned int i = 0; i < extraModels.size(); i++) {
    std::list<EvtDecayBase*>::iterator it = extraModels.begin();
    std::advance(it, i);
    TString name = (*it)->getName();
    if (name.Contains("PYTHIA") && usePythia)
      myExtraModels.push_back(*it);
    if (name.Contains("TAUOLA") && useTauola)
      myExtraModels.push_back(*it);
  }

  //Set up user evtgen models

  EvtModelUserReg userList;
  std::list<EvtDecayBase*> userModels = userList.getUserModels();  // get interface to user models
  for (unsigned int i = 0; i < userModels.size(); i++) {
    std::list<EvtDecayBase*>::iterator it = userModels.begin();
    std::advance(it, i);
    TString name = (*it)->getName();
    edm::LogInfo("EvtGenInterface::~EvtGenInterface") << "Adding user model: " << name;
    myExtraModels.push_back(*it);
  }

  // Set up the incoherent (1) or coherent (0) B mixing option
  BmixingOption = fPSet->getUntrackedParameter<int>("B_Mixing", 1);
  if (BmixingOption != 0 && BmixingOption != 1) {
    throw cms::Exception("Configuration") << "EvtGenProducer requires B_Mixing to be 0 (coherent) or 1 (incoherent) \n"
                                             "Please fix this in your configuration.";
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Create the EvtGen generator object, passing the external generators
  m_EvtGen = new EvtGen(
      decay_table.fullPath().c_str(), pdt.fullPath().c_str(), the_engine, radCorrEngine, &myExtraModels, BmixingOption);

  // Add additional user information
  if (fPSet->exists("user_decay_file")) {
    std::vector<std::string> user_decays = fPSet->getParameter<std::vector<std::string> >("user_decay_file");
    for (unsigned int i = 0; i < user_decays.size(); i++) {
      edm::FileInPath user_decay(user_decays.at(i));
      m_EvtGen->readUDecay(user_decay.fullPath().c_str());
    }
  }

  if (fPSet->exists("user_decay_embedded")) {
    std::vector<std::string> user_decay_lines = fPSet->getParameter<std::vector<std::string> >("user_decay_embedded");
    char user_decay_tmp[] = "/tmp/user_decay_tmpfileXXXXXX";
    int tmp_creation = mkstemp(user_decay_tmp);
    FILE* tmpf = std::fopen(user_decay_tmp, "w");
    if (!tmpf || (tmp_creation == -1)) {
      edm::LogError("EvtGenInterface::~EvtGenInterface")
          << "EvtGenInterface::init() fails when trying to open a temporary file for embedded user.dec. Terminating "
             "program ";
      exit(0);
    }
    for (unsigned int i = 0; i < user_decay_lines.size(); i++) {
      user_decay_lines.at(i) += "\n";
      std::fputs(user_decay_lines.at(i).c_str(), tmpf);
    }
    std::fclose(tmpf);
    m_EvtGen->readUDecay(user_decay_tmp);
  }

  // setup pdgid which the generator/hadronizer should not decay
  if (fPSet->exists("operates_on_particles")) {
    std::vector<int> tmpPIDs = fPSet->getParameter<std::vector<int> >("operates_on_particles");
    m_PDGs.clear();
    bool goodinput = false;
    if (!tmpPIDs.empty()) {
      if (tmpPIDs.size() == 1 && tmpPIDs[0] == 0)
        goodinput = false;
      else
        goodinput = true;
    } else {
      goodinput = false;
    }
    if (goodinput)
      m_PDGs = tmpPIDs;
    else
      SetDefault_m_PDGs();
  } else
    SetDefault_m_PDGs();

  for (unsigned int i = 0; i < m_PDGs.size(); i++) {
    edm::LogInfo("EvtGenInterface::~EvtGenInterface")
        << "EvtGenInterface::init() Particles to Operate on: " << m_PDGs[i];
  }

  // Obtain information to set polarization of particles
  polarize_ids = fPSet->getUntrackedParameter<std::vector<int> >("particles_to_polarize", std::vector<int>());
  polarize_pol = fPSet->getUntrackedParameter<std::vector<double> >("particle_polarizations", std::vector<double>());
  if (polarize_ids.size() != polarize_pol.size()) {
    throw cms::Exception("Configuration")
        << "EvtGenProducer requires that the particles_to_polarize and particle_polarization\n"
           "vectors be the same size.  Please fix this in your configuration.";
  }
  for (unsigned int ndx = 0; ndx < polarize_ids.size(); ndx++) {
    if (polarize_pol[ndx] < -1. || polarize_pol[ndx] > 1.) {
      throw cms::Exception("Configuration")
          << "EvtGenProducer error: particle polarizations must be in the range -1 < P < 1";
    }
    polarizations.insert(std::pair<int, float>(polarize_ids[ndx], polarize_pol[ndx]));
  }

  // Forced decays are particles that are aliased and forced to be decayed by EvtGen
  if (fPSet->exists("list_forced_decays")) {
    std::vector<std::string> forced_names = fPSet->getParameter<std::vector<std::string> >("list_forced_decays");
    for (unsigned int i = 0; i < forced_names.size(); i++) {
      EvtId found = EvtPDL::getId(forced_names[i]);
      if (found.getId() == -1)
        throw cms::Exception("Configuration") << "name in part list for ignored decays not found: " << forced_names[i];
      if (found.getId() == found.getAlias())
        throw cms::Exception("Configuration") << "name of ignored decays is not an alias: " << forced_names[i];
      forced_id.push_back(found);
      forced_pdgids.push_back(EvtPDL::getStdHep(found));  // force_pdgids is the list of stdhep codes
    }
  }
  edm::LogInfo("EvtGenInterface::~EvtGenInterface")
      << "Number of Forced Paricles is: " << forced_pdgids.size() << std::endl;
  for (unsigned int j = 0; j < forced_id.size(); j++) {
    edm::LogInfo("EvtGenInterface::~EvtGenInterface")
        << "Forced Paricles are: " << forced_pdgids.at(j) << " " << forced_id.at(j) << std::endl;
  }
  // Ignore decays are particles that are not to be decayed by EvtGen
  if (fPSet->exists("list_ignored_pdgids")) {
    ignore_pdgids = fPSet->getUntrackedParameter<std::vector<int> >("list_ignored_pdgids");
  }

  return;
}

HepMC::GenEvent* EvtGenInterface::decay(HepMC::GenEvent* evt) {
  if (the_engine->engine() == nullptr) {
    throw edm::Exception(edm::errors::LogicError)
        << "The EvtGen code attempted to use a random number engine while\n"
        << "the engine pointer was null in EvtGenInterface::decay. This might\n"
        << "mean that the code was modified to generate a random number outside\n"
        << "the event and beginLuminosityBlock methods, which is not allowed.\n";
  }
  CLHEP::RandFlat m_flat(*the_engine->engine(), 0., 1.);

  // decay all request unforced particles and store the forced decays to later decay one per event
  unsigned int nisforced = 0;
  std::vector<std::vector<HepMC::GenParticle*> > forcedparticles;
  for (unsigned int i = 0; i < forced_pdgids.size(); i++)
    forcedparticles.push_back(std::vector<HepMC::GenParticle*>());

  // notice this is a dynamic loop
  for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {
    if ((*p)->status() == 1) {  // all particles to be decays are set to status 1 by generator.hadronizer
      int idHep = (*p)->pdg_id();
      bool isignore = false;
      for (unsigned int i = 0; i < ignore_pdgids.size(); i++) {
        if (idHep == ignore_pdgids[i])
          isignore = true;
      }
      if (!isignore) {
        bool isforced = false;
        for (unsigned int i = 0; i < forced_pdgids.size(); i++) {
          if (idHep == forced_pdgids[i]) {
            forcedparticles.at(i).push_back(*p);  // storing and counting forced decays.
            nisforced++;
            isforced = true;
            break;
          }
        }
        bool isDefaultEvtGen = false;
        for (unsigned int i = 0; i < m_PDGs.size(); i++) {
          if (abs(idHep) == abs(m_PDGs[i])) {
            isDefaultEvtGen = true;
            break;
          }
        }
        EvtId idEvt = EvtPDL::evtIdFromStdHep(idHep);
        int ipart = idEvt.getId();
        EvtDecayTable* evtDecayTable = EvtDecayTable::getInstance();
        if (!isforced && isDefaultEvtGen && ipart != -1 && evtDecayTable->getNMode(ipart) != 0) {
          addToHepMC(*p, idEvt, evt, true);  // generate decay and remove daugther if they are forced
        }
      }
    }
  }

  // decay all forced particles (only 1/event is forced)... with no mixing allowed
  unsigned int which = (unsigned int)(nisforced * flat());
  if (which == nisforced && nisforced > 0)
    which = nisforced - 1;

  unsigned int idx = 0;
  for (unsigned int i = 0; i < forcedparticles.size(); i++) {
    for (unsigned int j = 0; j < forcedparticles.at(i).size(); j++) {
      EvtId idEvt = EvtPDL::evtIdFromStdHep(forcedparticles.at(i).at(j)->pdg_id());  // "standard" decay Id
      if (idx == which) {
        idEvt = forced_id[i];  // force decay Id
        edm::LogInfo("EvtGenInterface::decay ")
            << EvtPDL::getStdHep(idEvt) << " will force to decay " << idx + 1 << " out of " << nisforced << std::endl;
      }
      bool decayed = false;
      while (!decayed) {
        decayed = addToHepMC(forcedparticles.at(i).at(j), idEvt, evt, false);
      }
      idx++;
    }
  }

  // add code to ensure all particles have an end vertex and if they are undecayed with no end vertes set to status 1
  for (HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p) {
    if ((*p)->end_vertex() && (*p)->status() == 1)
      edm::LogWarning("EvtGenInterface::decay error: incorrect status!");  //(*p)->set_status(2);
    if ((*p)->end_vertex() && (*p)->end_vertex()->particles_out_size() == 0)
      edm::LogWarning("EvtGenInterface::decay error: empty end vertex!");
  }
  return evt;
}

// Add particle to MC
bool EvtGenInterface::addToHepMC(HepMC::GenParticle* partHep,
                                 const EvtId& idEvt,
                                 HepMC::GenEvent* theEvent,
                                 bool del_daug) {
  // Set up the parent particle from the HepMC GenEvent tree.
  //EvtVector4R pInit(EvtPDL::getMass(idEvt),partHep->momentum().px(),partHep->momentum().py(),partHep->momentum().pz());
  EvtVector4R pInit(
      partHep->momentum().e(), partHep->momentum().px(), partHep->momentum().py(), partHep->momentum().pz());
  EvtParticle* parent = EvtParticleFactory::particleFactory(idEvt, pInit);
  HepMC::FourVector posHep = (partHep->production_vertex())->position();
  EvtVector4R vInit(posHep.t(), posHep.x(), posHep.y(), posHep.z());
  // Reset polarization if requested....
  if (EvtPDL::getSpinType(idEvt) == EvtSpinType::DIRAC && polarizations.count(partHep->pdg_id()) > 0) {
    const HepMC::FourVector& momHep = partHep->momentum();
    EvtVector4R momEvt;
    momEvt.set(momHep.t(), momHep.x(), momHep.y(), momHep.z());
    // Particle is spin 1/2, so we can polarize it. Check polarizations map for particle, grab its polarization if it exists
    // and make the spin density matrix
    float pol = polarizations.find(partHep->pdg_id())->second;
    GlobalVector pPart(momHep.x(), momHep.y(), momHep.z());
    GlobalVector zHat(0., 0., 1.);
    GlobalVector zCrossP = zHat.cross(pPart);
    GlobalVector polVec = pol * zCrossP.unit();
    EvtSpinDensity theSpinDensity;
    theSpinDensity.setDim(2);
    theSpinDensity.set(0, 0, EvtComplex(1. / 2. + polVec.z() / 2., 0.));
    theSpinDensity.set(0, 1, EvtComplex(polVec.x() / 2., -polVec.y() / 2.));
    theSpinDensity.set(1, 0, EvtComplex(polVec.x() / 2., polVec.y() / 2.));
    theSpinDensity.set(1, 1, EvtComplex(1. / 2. - polVec.z() / 2., 0.));
    parent->setSpinDensityForwardHelicityBasis(theSpinDensity);
  }
  if (parent) {
    // Generate the event
    m_EvtGen->generateDecay(parent);
    if (del_daug)
      go_through_daughters(parent);  // will delete all daugthers which are listed as forced, to allow decay them later

    // create HepMCTree
    EvtHepMCEvent evtHepMCEvent;
    evtHepMCEvent.constructEvent(parent, vInit);
    HepMC::GenEvent* evtGenHepMCTree = evtHepMCEvent.getEvent();
    parent->deleteTree();

    // update the event using a recursive function
    if (!evtGenHepMCTree->particles_empty())
      update_particles(partHep, theEvent, (*evtGenHepMCTree->particles_begin()));

  } else {
    // this should never hapend, in such case, speak out.
    return false;
  }
  return true;
}

//Recursivley add EvtGen decay to to Event Decy tree
void EvtGenInterface::update_particles(HepMC::GenParticle* partHep, HepMC::GenEvent* theEvent, HepMC::GenParticle* p) {
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
        HepMC::GenParticle* daughter = new HepMC::GenParticle((*d)->momentum(), (*d)->pdg_id(), (*d)->status());
        daughter->suggest_barcode(theEvent->particles_size() + 1);
        partHep->end_vertex()->add_particle_out(daughter);
        if ((*d)->end_vertex())
          update_particles(daughter, theEvent, (*d));  // if daugthers add them as well
      }
      partHep->set_status(p->status());
    }
  }
}

void EvtGenInterface::setRandomEngine(CLHEP::HepRandomEngine* v) {
  the_engine->setRandomEngine(v);
  fRandomEngine = v;
}

double EvtGenInterface::flat() {
  if (!fRandomEngine) {
    throw cms::Exception("LogicError")
        << "EvtGenInterface::flat: Attempt to generate random number when engine pointer is null\n"
        << "This might mean that the code was modified to generate a random number outside the\n"
        << "event and beginLuminosityBlock methods, which is not allowed.\n";
  }
  return fRandomEngine->flat();
}

bool EvtGenInterface::findLastinChain(HepMC::GenParticle*& p) {
  if (p->end_vertex()) {
    if (p->end_vertex()->particles_out_size() != 0) {
      for (HepMC::GenVertex::particles_out_const_iterator d = p->end_vertex()->particles_out_const_begin();
           d != p->end_vertex()->particles_out_const_end();
           d++) {
        if (abs((*d)->pdg_id()) == abs(p->pdg_id())) {
          p = *d;
          findLastinChain(p);
          return false;
        }
      }
    }
  }
  return true;
}

bool EvtGenInterface::hasnoDaughter(HepMC::GenParticle* p) {
  if (p->end_vertex()) {
    if (p->end_vertex()->particles_out_size() != 0) {
      return false;
    }
  }
  return true;
}

void EvtGenInterface::go_through_daughters(EvtParticle* part) {
  int NDaug = part->getNDaug();
  if (NDaug) {
    EvtParticle* Daughter;
    for (int i = 0; i < NDaug; i++) {
      Daughter = part->getDaug(i);
      int idHep = Daughter->getPDGId();
      int found = 0;
      for (unsigned int j = 0; j < forced_pdgids.size(); j++) {
        if (idHep == forced_pdgids[j]) {
          found = 1;
          Daughter->deleteDaughters();
          break;
        }
      }
      if (!found)
        go_through_daughters(Daughter);
    }
  }
}

DEFINE_EDM_PLUGIN(EvtGenFactory, gen::EvtGenInterface, "EvtGen130");
