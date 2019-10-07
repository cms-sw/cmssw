#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/HerwigWrapper.h>
#include <HepMC/HEPEVT_Wrapper.h>
#include <HepMC/IO_HERWIG.h>
#include "HepPID/ParticleIDTranslations.hh"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "GeneratorInterface/Core/interface/FortranInstance.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"
#include "GeneratorInterface/Herwig6Interface/interface/herwig.h"

#include "DataFormats/Math/interface/LorentzVector.h"

namespace CLHEP {
  class HepRandomEngine;
}

extern "C" {
void hwuidt_(int *iopt, int *ipdg, int *iwig, char nwig[8]);
double hwualf_(int *mode, double *scale);
double hwuaem_(double *scale);
}

// helpers
namespace {
  inline bool call_hwmatch() {
    int result;
    hwmatch(&result);
    return result;
  }
  inline bool call_hwmsct() {
    int result;
    hwmsct(&result);
    return result;
  }

  int pdgToHerwig(int ipdg, char *nwig) {
    int iopt = 1;
    int iwig = 0;
    hwuidt_(&iopt, &ipdg, &iwig, nwig);
    return ipdg ? iwig : 0;
  }

  bool markStable(int pdgId) {
    char nwig[9] = "        ";
    if (!pdgToHerwig(pdgId, nwig))
      return false;
    hwusta(nwig, 1);
    return true;
  }
}  // namespace

class Herwig6Hadronizer : public gen::BaseHadronizer, public gen::Herwig6Instance {
public:
  Herwig6Hadronizer(const edm::ParameterSet &params);
  ~Herwig6Hadronizer() override;

  void setSLHAFromHeader(const std::vector<std::string> &lines);
  bool initialize(const lhef::HEPRUP *heprup);

  bool readSettings(int);

  // bool initializeForInternalPartons() { return initialize(0); }
  // bool initializeForExternalPartons() { return initialize(lheRunInfo()->getHEPRUP()); }

  bool initializeForInternalPartons();
  bool initializeForExternalPartons() { return initializeForInternalPartons(); }

  bool declareStableParticles(const std::vector<int> &pdgIds);
  bool declareSpecialSettings(const std::vector<std::string> &);

  void statistics();

  bool generatePartonsAndHadronize() { return hadronize(); }
  bool hadronize();
  bool decay();
  bool residualDecay();
  void finalizeEvent();

  const char *classname() const { return "Herwig6Hadronizer"; }

private:
  void doSetRandomEngine(CLHEP::HepRandomEngine *v) override;
  std::vector<std::string> const &doSharedResources() const override { return theSharedResources; }

  void clear();

  int pythiaStatusCode(const HepMC::GenParticle *p) const;
  void pythiaStatusCodes();

  void upInit() override;
  void upEvnt() override;

  static const std::vector<std::string> theSharedResources;

  HepMC::IO_HERWIG conv;
  bool needClear;
  bool externalPartons;

  gen::ParameterCollector parameters;
  int herwigVerbosity;
  int hepmcVerbosity;
  int maxEventsToPrint;
  bool printCards;
  bool emulatePythiaStatusCodes;
  double comEnergy;
  bool useJimmy;
  bool doMPInteraction;
  int numTrials;
  bool fConvertToPDG;  // convert PIDs
  bool doMatching;
  bool inclusiveMatching;
  int nMatch;
  double matchingScale;

  bool readMCatNLOfile;

  // -------------------------------------------------------------------------------
  std::string particleSpecFileName;  //Lars 20/Jul/2011
  bool readParticleSpecFile;
  // -------------------------------------------------------------------------------
};

extern "C" {
void hwwarn_(const char *method, int *id);
void setherwpdf_(void);
void mysetpdfpath_(const char *path);
}  // extern "C"

const std::vector<std::string> Herwig6Hadronizer::theSharedResources = {edm::SharedResourceNames::kHerwig6,
                                                                        gen::FortranInstance::kFortranInstance};

Herwig6Hadronizer::Herwig6Hadronizer(const edm::ParameterSet &params)
    : BaseHadronizer(params),
      needClear(false),
      parameters(params.getParameter<edm::ParameterSet>("HerwigParameters")),
      herwigVerbosity(params.getUntrackedParameter<int>("herwigVerbosity", 0)),
      hepmcVerbosity(params.getUntrackedParameter<int>("hepmcVerbosity", 0)),
      maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0)),
      printCards(params.getUntrackedParameter<bool>("printCards", false)),
      emulatePythiaStatusCodes(params.getUntrackedParameter<bool>("emulatePythiaStatusCodes", false)),
      comEnergy(params.getParameter<double>("comEnergy")),
      useJimmy(params.getParameter<bool>("useJimmy")),
      doMPInteraction(params.getParameter<bool>("doMPInteraction")),
      numTrials(params.getUntrackedParameter<int>("numTrialsMPI", 100)),
      doMatching(params.getUntrackedParameter<bool>("doMatching", false)),
      inclusiveMatching(params.getUntrackedParameter<bool>("inclusiveMatching", true)),
      nMatch(params.getUntrackedParameter<int>("nMatch", 0)),
      matchingScale(params.getUntrackedParameter<double>("matchingScale", 0.0)),
      readMCatNLOfile(false),

      // added to be able to read external particle spectrum file
      particleSpecFileName(params.getUntrackedParameter<std::string>("ParticleSpectrumFileName", "")),
      readParticleSpecFile(params.getUntrackedParameter<bool>("readParticleSpecFile", false))

{
  fConvertToPDG = false;
  if (params.exists("doPDGConvert"))
    fConvertToPDG = params.getParameter<bool>("doPDGConvert");
}

Herwig6Hadronizer::~Herwig6Hadronizer() { clear(); }

void Herwig6Hadronizer::doSetRandomEngine(CLHEP::HepRandomEngine *v) { setHerwigRandomEngine(v); }

void Herwig6Hadronizer::clear() {
  if (!needClear)
    return;

  // teminate elementary process
  call(hwefin);
  if (useJimmy)
    call(jmefin);

  needClear = false;
}

void Herwig6Hadronizer::setSLHAFromHeader(const std::vector<std::string> &lines) {
  std::set<std::string> blocks;
  std::string block;
  for (std::vector<std::string>::const_iterator iter = lines.begin(); iter != lines.end(); ++iter) {
    std::string line = *iter;
    std::transform(line.begin(), line.end(), line.begin(), (int (*)(int))std::toupper);
    std::string::size_type pos = line.find('#');
    if (pos != std::string::npos)
      line.resize(pos);

    if (line.empty())
      continue;

    if (!boost::algorithm::is_space()(line[0])) {
      std::vector<std::string> tokens;
      boost::split(tokens, line, boost::algorithm::is_space(), boost::token_compress_on);
      if (tokens.empty())
        continue;
      block.clear();
      if (tokens.size() < 2)
        continue;
      if (tokens[0] == "BLOCK")
        block = tokens[1];
      else if (tokens[0] == "DECAY")
        block = "DECAY";

      if (block.empty())
        continue;

      if (!blocks.count(block)) {
        blocks.insert(block);
        edm::LogWarning("Generator|Herwig6Hadronzier")
            << "Unsupported SLHA block \"" << block << "\".  It will be ignored." << std::endl;
      }
    }
  }
}

bool Herwig6Hadronizer::readSettings(int key) {
  clear();
  const lhef::HEPRUP *heprup = lheRunInfo()->getHEPRUP();
  externalPartons = (heprup != nullptr);

  if (key == 0 && externalPartons)
    return false;
  if (key > 0 && !externalPartons)
    return false;

  std::ostringstream info;
  info << "---------------------------------------------------\n";
  info << "Taking in settinsg for Herwig6Hadronizer for " << (externalPartons ? "external" : "internal")
       << " partons\n";
  info << "---------------------------------------------------\n";

  info << "   Herwig verbosity level         = " << herwigVerbosity << "\n";
  info << "   HepMC verbosity                = " << hepmcVerbosity << "\n";
  info << "   Number of events to be printed = " << maxEventsToPrint << "\n";

  // Call hwudat to set up HERWIG block data
  hwudat();

  // Setting basic parameters
  if (externalPartons) {
    hwproc.PBEAM1 = heprup->EBMUP.first;
    hwproc.PBEAM2 = heprup->EBMUP.second;
    pdgToHerwig(heprup->IDBMUP.first, hwbmch.PART1);
    pdgToHerwig(heprup->IDBMUP.second, hwbmch.PART2);
  } else {
    hwproc.PBEAM1 = 0.5 * comEnergy;
    hwproc.PBEAM2 = 0.5 * comEnergy;
    pdgToHerwig(2212, hwbmch.PART1);
    pdgToHerwig(2212, hwbmch.PART2);
  }

  if (doMatching) {
    hwmatchpram.n_match = nMatch;
    hwmatchpram.matching_scale = matchingScale;

    if (inclusiveMatching)
      hwmatchpram.max_multiplicity_flag = 1;
    else
      hwmatchpram.max_multiplicity_flag = 0;
  }

  if (useJimmy) {
    info << "   HERWIG will be using JIMMY for UE/MI.\n";
    jmparm.MSFLAG = 1;
    if (doMPInteraction)
      info << "   JIMMY trying to generate multiple interactions.\n";
  }

  // set the IPROC already here... needed for VB pairs

  bool iprocFound = false;

  for (gen::ParameterCollector::const_iterator line = parameters.begin(); line != parameters.end(); ++line) {
    if (!strcmp((line->substr(0, 5)).c_str(), "IPROC")) {
      if (!give(*line))
        throw edm::Exception(edm::errors::Configuration)
            << "Herwig 6 did not accept the following: \"" << *line << "\"." << std::endl;
      else
        iprocFound = true;
    }
  }

  if (!iprocFound && !externalPartons)
    throw edm::Exception(edm::errors::Configuration) << "You have to define the process with IPROC." << std::endl;

  // initialize other common blocks ...
  call(hwigin);  // default init

  hwevnt.MAXER = 100000000;  // O(inf)
  hwpram.LWSUD = 0;          // don't write Sudakov form factors
  hwdspn.LWDEC = 0;          // don't write three/four body decays
                             // (no fort.77 and fort.88 ...)
  // init LHAPDF glue
  std::memset(hwprch.AUTPDF, ' ', sizeof hwprch.AUTPDF);
  for (unsigned int i = 0; i < 2; i++) {
    hwpram.MODPDF[i] = -111;
    std::memcpy(hwprch.AUTPDF[i], "HWLHAPDF", 8);
  }

  hwevnt.MAXPR = maxEventsToPrint;
  hwpram.IPRINT = herwigVerbosity;
  //	hwprop.RMASS[6] = 175.0;	//FIXME

  if (printCards) {
    info << "\n";
    info << "------------------------------------\n";
    info << "Reading HERWIG parameters\n";
    info << "------------------------------------\n";
  }
  for (gen::ParameterCollector::const_iterator line = parameters.begin(); line != parameters.end(); ++line) {
    if (printCards)
      info << "   " << *line << "\n";
    if (!give(*line))
      throw edm::Exception(edm::errors::Configuration)
          << "Herwig 6 did not accept the following: \"" << *line << "\"." << std::endl;
  }

  if (printCards)
    info << "\n";

  if (externalPartons) {
    std::vector<std::string> slha = lheRunInfo()->findHeader("slha");
    if (!slha.empty())
      setSLHAFromHeader(slha);
  }

  needClear = true;

  std::pair<int, int> pdfs(-1, -1);
  if (externalPartons)
    pdfs = lheRunInfo()->pdfSetTranslation();

  if (hwpram.MODPDF[0] != -111 || hwpram.MODPDF[1] != -111) {
    for (unsigned int i = 0; i < 2; i++)
      if (hwpram.MODPDF[i] == -111)
        hwpram.MODPDF[i] = -1;

    if (pdfs.first != -1 || pdfs.second != -1)
      edm::LogError("Generator|Herwig6Hadronzier") << "Both external Les Houches event and "
                                                      "config file specify a PDF set.  "
                                                      "User PDF will override external one."
                                                   << std::endl;

    pdfs.first = hwpram.MODPDF[0] != -111 ? hwpram.MODPDF[0] : -1;
    pdfs.second = hwpram.MODPDF[1] != -111 ? hwpram.MODPDF[1] : -1;
  }

  printf("pdfs.first = %i, pdfs.second = %i\n", pdfs.first, pdfs.second);

  hwpram.MODPDF[0] = pdfs.first;
  hwpram.MODPDF[1] = pdfs.second;

  if (externalPartons && hwproc.IPROC >= 0)
    hwproc.IPROC = -1;

  //Lars: lower EFFMIN threshold, to continue execution of IPROC=4000, lambda'_211=0.01 at LM7,10
  if (readParticleSpecFile) {
    openParticleSpecFile(particleSpecFileName);
    hwpram.EFFMIN = 1e-5;
  }

  edm::LogInfo(info.str());

  return true;
}

bool Herwig6Hadronizer::initializeForInternalPartons() {
  if (useJimmy)
    call(jimmin);

  call(hwuinc);

  // initialize HERWIG event generation
  call(hweini);

  if (useJimmy) {
    call(jminit);
  }

  return true;
}

bool Herwig6Hadronizer::initialize(const lhef::HEPRUP *heprup) {
  clear();

  externalPartons = (heprup != nullptr);

  std::ostringstream info;
  info << "---------------------------------------------------\n";
  info << "Initializing Herwig6Hadronizer for " << (externalPartons ? "external" : "internal") << " partons\n";
  info << "---------------------------------------------------\n";

  info << "   Herwig verbosity level         = " << herwigVerbosity << "\n";
  info << "   HepMC verbosity                = " << hepmcVerbosity << "\n";
  info << "   Number of events to be printed = " << maxEventsToPrint << "\n";

  // Call hwudat to set up HERWIG block data
  hwudat();

  // Setting basic parameters
  if (externalPartons) {
    hwproc.PBEAM1 = heprup->EBMUP.first;
    hwproc.PBEAM2 = heprup->EBMUP.second;
    pdgToHerwig(heprup->IDBMUP.first, hwbmch.PART1);
    pdgToHerwig(heprup->IDBMUP.second, hwbmch.PART2);
  } else {
    hwproc.PBEAM1 = 0.5 * comEnergy;
    hwproc.PBEAM2 = 0.5 * comEnergy;
    pdgToHerwig(2212, hwbmch.PART1);
    pdgToHerwig(2212, hwbmch.PART2);
  }

  if (useJimmy) {
    info << "   HERWIG will be using JIMMY for UE/MI.\n";
    jmparm.MSFLAG = 1;
    if (doMPInteraction)
      info << "   JIMMY trying to generate multiple interactions.\n";
  }

  // set the IPROC already here... needed for VB pairs

  bool iprocFound = false;

  for (gen::ParameterCollector::const_iterator line = parameters.begin(); line != parameters.end(); ++line) {
    if (!strcmp((line->substr(0, 5)).c_str(), "IPROC")) {
      if (!give(*line))
        throw edm::Exception(edm::errors::Configuration)
            << "Herwig 6 did not accept the following: \"" << *line << "\"." << std::endl;
      else
        iprocFound = true;
    }
  }

  if (!iprocFound && !externalPartons)
    throw edm::Exception(edm::errors::Configuration) << "You have to define the process with IPROC." << std::endl;

  // initialize other common blocks ...
  call(hwigin);
  hwevnt.MAXER = 100000000;  // O(inf)
  hwpram.LWSUD = 0;          // don't write Sudakov form factors
  hwdspn.LWDEC = 0;          // don't write three/four body decays
                             // (no fort.77 and fort.88 ...)

  // init LHAPDF glue

  std::memset(hwprch.AUTPDF, ' ', sizeof hwprch.AUTPDF);
  for (unsigned int i = 0; i < 2; i++) {
    hwpram.MODPDF[i] = -111;
    std::memcpy(hwprch.AUTPDF[i], "HWLHAPDF", 8);
  }

  if (useJimmy)
    call(jimmin);

  hwevnt.MAXPR = maxEventsToPrint;
  hwpram.IPRINT = herwigVerbosity;
  //	hwprop.RMASS[6] = 175.0;	//FIXME

  if (printCards) {
    info << "\n";
    info << "------------------------------------\n";
    info << "Reading HERWIG parameters\n";
    info << "------------------------------------\n";
  }
  for (gen::ParameterCollector::const_iterator line = parameters.begin(); line != parameters.end(); ++line) {
    if (printCards)
      info << "   " << *line << "\n";
    if (!give(*line))
      throw edm::Exception(edm::errors::Configuration)
          << "Herwig 6 did not accept the following: \"" << *line << "\"." << std::endl;
  }

  if (printCards)
    info << "\n";

  if (externalPartons) {
    std::vector<std::string> slha = lheRunInfo()->findHeader("slha");
    if (!slha.empty())
      setSLHAFromHeader(slha);
  }

  needClear = true;

  std::pair<int, int> pdfs(-1, -1);
  if (externalPartons)
    pdfs = lheRunInfo()->pdfSetTranslation();

  if (hwpram.MODPDF[0] != -111 || hwpram.MODPDF[1] != -111) {
    for (unsigned int i = 0; i < 2; i++)
      if (hwpram.MODPDF[i] == -111)
        hwpram.MODPDF[i] = -1;

    if (pdfs.first != -1 || pdfs.second != -1)
      edm::LogError("Generator|Herwig6Hadronzier") << "Both external Les Houches event and "
                                                      "config file specify a PDF set.  "
                                                      "User PDF will override external one."
                                                   << std::endl;

    pdfs.first = hwpram.MODPDF[0] != -111 ? hwpram.MODPDF[0] : -1;
    pdfs.second = hwpram.MODPDF[1] != -111 ? hwpram.MODPDF[1] : -1;
  }

  printf("pdfs.first = %i, pdfs.second = %i\n", pdfs.first, pdfs.second);

  hwpram.MODPDF[0] = pdfs.first;
  hwpram.MODPDF[1] = pdfs.second;

  if (externalPartons)
    hwproc.IPROC = -1;

  //Lars: lower EFFMIN threshold, to continue execution of IPROC=4000, lambda'_211=0.01 at LM7,10
  if (readParticleSpecFile) {
    openParticleSpecFile(particleSpecFileName);
    hwpram.EFFMIN = 1e-5;
  }

  // HERWIG preparations ...
  call(hwuinc);
  markStable(13);     // MU+
  markStable(-13);    // MU-
  markStable(3112);   // SIGMA+
  markStable(-3112);  // SIGMABAR+
  markStable(3222);   // SIGMA-
  markStable(-3222);  // SIGMABAR-
  markStable(3122);   // LAMBDA0
  markStable(-3122);  // LAMBDABAR0
  markStable(3312);   // XI-
  markStable(-3312);  // XIBAR+
  markStable(3322);   // XI0
  markStable(-3322);  // XI0BAR
  markStable(3334);   // OMEGA-
  markStable(-3334);  // OMEGABAR+
  markStable(211);    // PI+
  markStable(-211);   // PI-
  markStable(321);    // K+
  markStable(-321);   // K-
  markStable(310);    // K_S0
  markStable(130);    // K_L0

  // better: merge with declareStableParticles
  // and get the list from configuration / Geant4 / Core somewhere

  // initialize HERWIG event generation
  call(hweini);

  if (useJimmy)
    call(jminit);

  edm::LogInfo(info.str());

  return true;
}

bool Herwig6Hadronizer::declareStableParticles(const std::vector<int> &pdgIds) {
  markStable(13);     // MU+
  markStable(-13);    // MU-
  markStable(3112);   // SIGMA+
  markStable(-3112);  // SIGMABAR+
  markStable(3222);   // SIGMA-
  markStable(-3222);  // SIGMABAR-
  markStable(3122);   // LAMBDA0
  markStable(-3122);  // LAMBDABAR0
  markStable(3312);   // XI-
  markStable(-3312);  // XIBAR+
  markStable(3322);   // XI0
  markStable(-3322);  // XI0BAR
  markStable(3334);   // OMEGA-
  markStable(-3334);  // OMEGABAR+
  markStable(211);    // PI+
  markStable(-211);   // PI-
  markStable(321);    // K+
  markStable(-321);   // K-
  markStable(310);    // K_S0
  markStable(130);    // K_L0

  for (std::vector<int>::const_iterator iter = pdgIds.begin(); iter != pdgIds.end(); ++iter)
    if (!markStable(*iter))
      return false;
  return true;
}

bool Herwig6Hadronizer::declareSpecialSettings(const std::vector<std::string> &) { return true; }

void Herwig6Hadronizer::statistics() {
  if (!runInfo().internalXSec()) {
    // not set via LHE, so get it from HERWIG
    // the reason is that HERWIG doesn't compute the xsec
    // in all LHE modes

    double RNWGT = 1. / hwevnt.NWGTS;
    double AVWGT = hwevnt.WGTSUM * RNWGT;

    double xsec = 1.0e3 * AVWGT;

    runInfo().setInternalXSec(xsec);
  }
}

bool Herwig6Hadronizer::hadronize() {
  // hard process generation, parton shower, hadron formation

  InstanceWrapper wrapper(this);  // safe guard

  event().reset();

  // call herwig routines to create HEPEVT

  hwuine();  // initialize event

  if (callWithTimeout(10, hwepro)) {  // process event and PS
    // We hung for more than 10 seconds
    int error = 199;
    hwwarn_("HWHGUP", &error);
  }

  hwbgen();  // parton cascades

  // call jimmy ... only if event is not killed yet by HERWIG
  if (useJimmy && doMPInteraction && !hwevnt.IERROR && call_hwmsct()) {
    if (lheEvent())
      lheEvent()->count(lhef::LHERunInfo::kKilled);
    return false;
  }

  hwdhob();  // heavy quark decays
  hwcfor();  // cluster formation
  hwcdec();  // cluster decays

  // 	// if event *not* killed by HERWIG, return true
  // 	if (hwevnt.IERROR) {
  // 	  hwufne();	// finalize event, to keep system clean
  // 	  if (lheEvent()) lheEvent()->count(lhef::LHERunInfo::kKilled);
  // 	  return false;
  // 	}

  //if (lheEvent()) lheEvent()->count(lhef::LHERunInfo::kAccepted);

  hwdhad();  // unstable particle decays
  hwdhvy();  // heavy flavour decays
  hwmevt();  // soft underlying event

  hwufne();  // finalize event

  // if event *not* killed by HERWIG, return true
  if (hwevnt.IERROR) {
    if (lheEvent())
      lheEvent()->count(lhef::LHERunInfo::kKilled);
    return false;
  }

  if (doMatching) {
    bool pass = call_hwmatch();
    if (!pass) {
      printf("Event failed MLM matching\n");
      if (lheEvent())
        lheEvent()->count(lhef::LHERunInfo::kSelected);
      return false;
    }
  }

  if (lheEvent())
    lheEvent()->count(lhef::LHERunInfo::kAccepted);

  event().reset(new HepMC::GenEvent);
  if (!conv.fill_next_event(event().get()))
    throw cms::Exception("Herwig6Error") << "HepMC Conversion problems in event." << std::endl;

  // do particle ID conversion Herwig->PDG, if requested
  if (fConvertToPDG) {
    for (HepMC::GenEvent::particle_iterator part = event()->particles_begin(); part != event()->particles_end();
         ++part) {
      if ((*part)->pdg_id() != HepPID::translateHerwigtoPDT((*part)->pdg_id()))
        (*part)->set_pdg_id(HepPID::translateHerwigtoPDT((*part)->pdg_id()));
    }
  }

  return true;
}

void Herwig6Hadronizer::finalizeEvent() {
  lhef::LHEEvent::fixHepMCEventTimeOrdering(event().get());

  HepMC::PdfInfo pdfInfo;
  if (externalPartons) {
    lheEvent()->fillEventInfo(event().get());
    lheEvent()->fillPdfInfo(&pdfInfo);

    // for MC@NLO: IDWRUP is not filled...
    if (event()->signal_process_id() == 0)
      event()->set_signal_process_id(abs(hwproc.IPROC));
  }

  HepMC::GenParticle *incomingParton = nullptr;
  HepMC::GenParticle *targetParton = nullptr;

  HepMC::GenParticle *incomingProton = nullptr;
  HepMC::GenParticle *targetProton = nullptr;

  // find incoming parton (first entry with IST=121)
  for (HepMC::GenEvent::particle_const_iterator it = event()->particles_begin();
       (it != event()->particles_end() && incomingParton == nullptr);
       it++)
    if ((*it)->status() == 121)
      incomingParton = (*it);

  // find target parton (first entry with IST=122)
  for (HepMC::GenEvent::particle_const_iterator it = event()->particles_begin();
       (it != event()->particles_end() && targetParton == nullptr);
       it++)
    if ((*it)->status() == 122)
      targetParton = (*it);

  // find incoming Proton (first entry ID=2212, IST=101)
  for (HepMC::GenEvent::particle_const_iterator it = event()->particles_begin();
       (it != event()->particles_end() && incomingProton == nullptr);
       it++)
    if ((*it)->status() == 101 && (*it)->pdg_id() == 2212)
      incomingProton = (*it);

  // find target Proton (first entry ID=2212, IST=102)
  for (HepMC::GenEvent::particle_const_iterator it = event()->particles_begin();
       (it != event()->particles_end() && targetProton == nullptr);
       it++)
    if ((*it)->status() == 102 && (*it)->pdg_id() == 2212)
      targetProton = (*it);

  // find hard scale Q (computed from colliding partons)
  if (incomingParton && targetParton) {
    math::XYZTLorentzVector totMomentum(0, 0, 0, 0);
    totMomentum += incomingParton->momentum();
    totMomentum += targetParton->momentum();
    double evScale = totMomentum.mass();
    double evScale2 = evScale * evScale;

    // find alpha_QED & alpha_QCD
    int one = 1;
    double alphaQCD = hwualf_(&one, &evScale);
    double alphaQED = hwuaem_(&evScale2);

    if (!externalPartons || event()->event_scale() < 0)
      event()->set_event_scale(evScale);
    if (!externalPartons || event()->alphaQCD() < 0)
      event()->set_alphaQCD(alphaQCD);
    if (!externalPartons || event()->alphaQED() < 0)
      event()->set_alphaQED(alphaQED);

    if (!externalPartons || pdfInfo.x1() < 0) {
      // get the PDF information
      pdfInfo.set_id1(incomingParton->pdg_id() == 21 ? 0 : incomingParton->pdg_id());
      pdfInfo.set_id2(targetParton->pdg_id() == 21 ? 0 : targetParton->pdg_id());
      if (incomingProton && targetProton) {
        double x1 = incomingParton->momentum().pz() / incomingProton->momentum().pz();
        double x2 = targetParton->momentum().pz() / targetProton->momentum().pz();
        pdfInfo.set_x1(x1);
        pdfInfo.set_x2(x2);
      }
      // we do not fill pdf1 & pdf2, since they are not easily available (what are they needed for anyways???)
      pdfInfo.set_scalePDF(evScale);  // the same as Q above... does this make sense?
    }

    if (!externalPartons || event()->signal_process_id() < 0)
      event()->set_signal_process_id(abs(hwproc.IPROC));
    event()->set_pdf_info(pdfInfo);
  }

  // add event weight & PDF information
  if (lheRunInfo() != nullptr && std::abs(lheRunInfo()->getHEPRUP()->IDWTUP) == 4)
    // in LHE weighting mode 4 the weight is an xsec, so convert form HERWIG
    // to standard CMS unit "picobarn"
    event()->weights().push_back(1.0e3 * hwevnt.EVWGT);
  else
    event()->weights().push_back(hwevnt.EVWGT);

  // find final parton (first entry with IST=123)
  HepMC::GenParticle *finalParton = nullptr;
  for (HepMC::GenEvent::particle_const_iterator it = event()->particles_begin();
       (it != event()->particles_end() && finalParton == nullptr);
       it++)
    if ((*it)->status() == 123)
      finalParton = (*it);

  // add GenEventInfo & binning Values
  eventInfo().reset(new GenEventInfoProduct(event().get()));
  if (finalParton) {
    double thisPt = finalParton->momentum().perp();
    eventInfo()->setBinningValues(std::vector<double>(1, thisPt));
  }

  // emulate PY6 status codes, if switched on...
  if (emulatePythiaStatusCodes)
    pythiaStatusCodes();
}

bool Herwig6Hadronizer::decay() {
  // hadron decays

  // 	InstanceWrapper wrapper(this);	// safe guard
  //
  // 	//int iproc = hwproc.IPROC;
  // 	//hwproc.IPROC = 312;
  // 	hwdhad();	// unstable particle decays
  // 	//hwproc.IPROC = iproc;
  //
  // 	hwdhvy();	// heavy flavour decays
  // 	hwmevt();	// soft underlying event
  //
  // 	if (doMatching) {
  // 	  bool pass = call_hwmatch();
  // 	  if (!pass) {
  // 	    printf("Event failed MLM matching\n");
  // 	    hwufne();
  // 	    if (lheEvent()) lheEvent()->count(lhef::LHERunInfo::kKilled);
  // 	    return false;
  // 	  }
  // 	}
  //
  // 	hwufne();	// finalize event
  //
  // 	if (hwevnt.IERROR)
  // 		return false;
  //
  // 	if (lheEvent()) lheEvent()->count(lhef::LHERunInfo::kAccepted);
  //
  // 	event().reset(new HepMC::GenEvent);
  // 	if (!conv.fill_next_event(event().get()))
  // 		throw cms::Exception("Herwig6Error")
  // 			<< "HepMC Conversion problems in event." << std::endl;
  //
  //         // do particle ID conversion Herwig->PDG, if requested
  //         if ( fConvertToPDG ) {
  //            for ( HepMC::GenEvent::particle_iterator part = event()->particles_begin(); part != event()->particles_end(); ++part) {
  //              if ((*part)->pdg_id() != HepPID::translateHerwigtoPDT((*part)->pdg_id()))
  //                (*part)->set_pdg_id(HepPID::translateHerwigtoPDT((*part)->pdg_id()));
  //            }
  //         }

  return true;
}

bool Herwig6Hadronizer::residualDecay() { return true; }

void Herwig6Hadronizer::upInit() {
  lhef::CommonBlocks::fillHEPRUP(lheRunInfo()->getHEPRUP());
  heprup_.pdfgup[0] = heprup_.pdfgup[1] = -1;
  heprup_.pdfsup[0] = heprup_.pdfsup[1] = -1;
  // we set up the PDFs ourselves

  // pass HERWIG paramaters fomr header (if present)
  std::string mcnloHeader = "herwig6header";
  std::vector<lhef::LHERunInfo::Header> headers = lheRunInfo()->getHeaders();
  for (std::vector<lhef::LHERunInfo::Header>::const_iterator hIter = headers.begin(); hIter != headers.end(); ++hIter) {
    if (hIter->tag() == mcnloHeader) {
      readMCatNLOfile = true;
      for (lhef::LHERunInfo::Header::const_iterator lIter = hIter->begin(); lIter != hIter->end(); ++lIter) {
        if ((lIter->c_str())[0] != '#' && (lIter->c_str())[0] != '\n') {  // it's not a comment)
          if (!give(*lIter))
            throw edm::Exception(edm::errors::Configuration)
                << "Herwig 6 did not accept the following: \"" << *lIter << "\"." << std::endl;
        }
      }
    }
  }
}

void Herwig6Hadronizer::upEvnt() {
  lhef::CommonBlocks::fillHEPEUP(lheEvent()->getHEPEUP());

  // if MCatNLO external file is read, read comment & pass IHPRO to HERWIG
  if (readMCatNLOfile) {
    for (std::vector<std::string>::const_iterator iter = lheEvent()->getComments().begin();
         iter != lheEvent()->getComments().end();
         ++iter) {
      std::string toParse(iter->substr(1));
      if (!give(toParse))
        throw edm::Exception(edm::errors::Configuration)
            << "Herwig 6 did not accept the following: \"" << toParse << "\"." << std::endl;
    }
  }
}

int Herwig6Hadronizer::pythiaStatusCode(const HepMC::GenParticle *p) const {
  int status = p->status();

  // weird 9922212 particles...
  if (status == 3 && !p->end_vertex())
    return 2;

  if (status >= 1 && status <= 3)
    return status;

  if (!p->end_vertex())
    return 1;

  // let's prevent particles having status 3, if the identical
  // particle downstream is a better status 3 candidate
  int currentId = p->pdg_id();
  int orig = status;
  if (status == 123 || status == 124 || status == 155 || status == 156 || status == 160 ||
      (status >= 195 && status <= 197)) {
    for (const HepMC::GenParticle *q = p;;) {
      const HepMC::GenVertex *vtx = q->end_vertex();
      if (!vtx)
        break;

      HepMC::GenVertex::particles_out_const_iterator iter;
      for (iter = vtx->particles_out_const_begin(); iter != vtx->particles_out_const_end(); ++iter)
        if ((*iter)->pdg_id() == currentId)
          break;

      if (iter == vtx->particles_out_const_end())
        break;

      q = *iter;
      if (q->status() == 3 || ((status == 120 || status == 123 || status == 124) && orig > 124))
        return 4;
    }
  }

  int nesting = 0;
  for (;;) {
    if ((status >= 120 && status <= 122) || status == 3) {
      // avoid flagging status 3 if there is a
      // better status 3 candidate upstream
      if (externalPartons)
        return ((orig >= 121 && orig <= 124) || orig == 3) ? 3 : 4;
      else
        return (nesting || (status != 3 && orig <= 124)) ? 3 : 4;
    }

    // check whether we are leaving the hard process
    // including heavy resonance decays
    if (!(status == 4 || status == 123 || status == 124 || status == 155 || status == 156 || status == 160 ||
          (status >= 195 && status <= 197)))
      break;

    const HepMC::GenVertex *vtx = p->production_vertex();
    if (!vtx || !vtx->particles_in_size())
      break;

    p = *vtx->particles_in_const_begin();
    status = p->status();

    int newId = p->pdg_id();

    if (!newId)
      break;

    // nesting increases if we move to the next-best mother
    if (newId != currentId) {
      if (++nesting > 1 && externalPartons)
        break;
      currentId = newId;
    }
  }

  return 2;
}

void Herwig6Hadronizer::pythiaStatusCodes() {
  for (HepMC::GenEvent::particle_iterator iter = event()->particles_begin(); iter != event()->particles_end(); iter++)
    (*iter)->set_status(pythiaStatusCode(*iter));

  for (HepMC::GenEvent::particle_iterator iter = event()->particles_begin(); iter != event()->particles_end(); iter++)
    if ((*iter)->status() == 4)
      (*iter)->set_status(2);
}

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

typedef edm::GeneratorFilter<Herwig6Hadronizer, gen::ExternalDecayDriver> Herwig6GeneratorFilter;
DEFINE_FWK_MODULE(Herwig6GeneratorFilter);

typedef edm::HadronizerFilter<Herwig6Hadronizer, gen::ExternalDecayDriver> Herwig6HadronizerFilter;
DEFINE_FWK_MODULE(Herwig6HadronizerFilter);
