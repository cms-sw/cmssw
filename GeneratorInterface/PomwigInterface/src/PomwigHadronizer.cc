#include "GeneratorInterface/PomwigInterface/interface/PomwigHadronizer.h"

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
#include "GeneratorInterface/Core/interface/FortranInstance.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"
#include "GeneratorInterface/Herwig6Interface/interface/herwig.h"

namespace gen {
  extern "C" {
  void hwuidt_(int *iopt, int *ipdg, int *iwig, char nwig[8]);
  }

  // helpers
  namespace {
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

#define qcd_1994 qcd_1994_
  extern "C" {
  void qcd_1994(double &, double &, double *, int &);
  }
// For H1 2006 fits
#define qcd_2006 qcd_2006_
  extern "C" {
  void qcd_2006(double &, double &, int &, double *, double *, double *, double *, double *);
  }

  extern "C" {
  void hwwarn_(const char *method, int *id);
  void setherwpdf_(void);
  void mysetpdfpath_(const char *path);
  }

  const std::vector<std::string> PomwigHadronizer::theSharedResources = {edm::SharedResourceNames::kHerwig6,
                                                                         gen::FortranInstance::kFortranInstance};

  PomwigHadronizer::PomwigHadronizer(const edm::ParameterSet &params)
      : BaseHadronizer(params),
        needClear(false),
        parameters(params.getParameter<edm::ParameterSet>("HerwigParameters")),
        herwigVerbosity(params.getUntrackedParameter<int>("herwigVerbosity", 0)),
        hepmcVerbosity(params.getUntrackedParameter<int>("hepmcVerbosity", 0)),
        maxEventsToPrint(params.getUntrackedParameter<int>("maxEventsToPrint", 0)),
        printCards(params.getUntrackedParameter<bool>("printCards", false)),
        comEnergy(params.getParameter<double>("comEnergy")),
        survivalProbability(params.getParameter<double>("survivalProbability")),
        diffTopology(params.getParameter<int>("diffTopology")),
        h1fit(params.getParameter<int>("h1fit")),
        useJimmy(params.getParameter<bool>("useJimmy")),
        doMPInteraction(params.getParameter<bool>("doMPInteraction")),
        numTrials(params.getUntrackedParameter<int>("numTrialsMPI", 100)),
        doPDGConvert(false) {
    if (params.exists("doPDGConvert"))
      doPDGConvert = params.getParameter<bool>("doPDGConvert");
  }

  PomwigHadronizer::~PomwigHadronizer() { clear(); }

  void PomwigHadronizer::doSetRandomEngine(CLHEP::HepRandomEngine *v) { setHerwigRandomEngine(v); }

  void PomwigHadronizer::clear() {
    if (!needClear)
      return;

    // teminate elementary process
    call(hwefin);
    if (useJimmy)
      call(jmefin);

    needClear = false;
  }

  bool PomwigHadronizer::initializeForExternalPartons() { return false; }

  bool PomwigHadronizer::readSettings(int) {
    clear();

    edm::LogVerbatim("") << "----------------------------------------------\n"
                         << "Initializing PomwigHadronizer\n"
                         << "----------------------------------------------\n";

    // Call hwudat to set up HERWIG block data
    hwudat();

    // Setting basic parameters ...
    hwproc.PBEAM1 = comEnergy / 2.;
    hwproc.PBEAM2 = comEnergy / 2.;
    // Choose beam particles for POMWIG depending on topology
    switch (diffTopology) {
      case 0:  //DPE
        hwbmch.PART1[0] = 'E';
        hwbmch.PART1[1] = '-';
        hwbmch.PART2[0] = 'E';
        hwbmch.PART2[1] = '-';
        break;
      case 1:  //SD survive PART1
        hwbmch.PART1[0] = 'E';
        hwbmch.PART1[1] = '-';
        hwbmch.PART2[0] = 'P';
        hwbmch.PART2[1] = ' ';
        break;
      case 2:  //SD survive PART2
        hwbmch.PART1[0] = 'P';
        hwbmch.PART1[1] = ' ';
        hwbmch.PART2[0] = 'E';
        hwbmch.PART2[1] = '-';
        break;
      case 3:  //Non diffractive
        hwbmch.PART1[0] = 'P';
        hwbmch.PART1[1] = ' ';
        hwbmch.PART2[0] = 'P';
        hwbmch.PART2[1] = ' ';
        break;
      default:
        throw edm::Exception(edm::errors::Configuration, "PomwigError")
            << " Invalid Diff. Topology. Must be DPE(diffTopology = 0), SD particle 1 (diffTopology = 1), SD particle "
               "2 (diffTopology = 2) and Non diffractive (diffTopology = 3)";
        break;
    }
    for (int i = 2; i < 8; ++i) {
      hwbmch.PART1[i] = ' ';
      hwbmch.PART2[i] = ' ';
    }

    // initialize other common blocks ...
    call(hwigin);

    hwevnt.MAXER = 100000000;  // O(inf)
    hwpram.LWSUD = 0;          // don't write Sudakov form factors
    hwdspn.LWDEC = 0;          // don't write three/four body decays
                               // (no fort.77 and fort.88 ...)a

    std::memset(hwprch.AUTPDF, ' ', sizeof hwprch.AUTPDF);
    for (unsigned int i = 0; i < 2; i++) {
      hwpram.MODPDF[i] = -111;
      std::memcpy(hwprch.AUTPDF[i], "HWLHAPDF", 8);
    }

    hwevnt.MAXPR = maxEventsToPrint;
    hwpram.IPRINT = herwigVerbosity;

    edm::LogVerbatim("") << "------------------------------------\n"
                         << "Reading HERWIG parameters\n"
                         << "------------------------------------\n";

    for (gen::ParameterCollector::const_iterator line = parameters.begin(); line != parameters.end(); ++line) {
      edm::LogVerbatim("") << "   " << *line;
      if (!give(*line))
        throw edm::Exception(edm::errors::Configuration)
            << "Herwig 6 did not accept the following: \"" << *line << "\"." << std::endl;
    }

    needClear = true;

    return true;
  }

  bool PomwigHadronizer::initializeForInternalPartons() {
    call(hwuinc);

    hwusta("PI0     ", 1);

    if (!initializeDPDF())
      return false;

    call(hweini);

    return true;
  }

  bool PomwigHadronizer::initializeDPDF() {
    // Initialize H1 pomeron/reggeon

    if (diffTopology == 3)
      return true;

    if ((diffTopology != 0) && (diffTopology != 1) && (diffTopology != 2))
      return false;

    int nstru = hwpram.NSTRU;
    int ifit = h1fit;
    if ((nstru == 9) || (nstru == 10)) {
      if ((ifit <= 0) || (ifit >= 7)) {
        throw edm::Exception(edm::errors::Configuration, "PomwigError")
            << " Attempted to set non existant H1 1997 fit index. Has to be 1...6";
      }
      std::string aux((nstru == 9) ? "Pomeron" : "Reggeon");
      edm::LogVerbatim("") << "   H1 1997 pdf's: " << aux << "\n"
                           << "   IFIT = " << ifit;
      double xp = 0.1;
      double Q2 = 75.0;
      double xpq[13];
      qcd_1994(xp, Q2, xpq, ifit);
    } else if ((nstru >= 12) && (nstru <= 15)) {
      bool isPom = (nstru == 12) || (nstru == 14);
      bool isFitA = (nstru == 12) || (nstru == 13);
      ifit = (isFitA) ? 1 : 2;
      std::string aux_0((isFitA) ? "A" : "B");
      std::string aux_1((isPom) ? "Pomeron" : "Reggeon");
      edm::LogVerbatim("") << "   H1 2006 Fit " << aux_0 << " " << aux_1 << "\n"
                           << "   IFIT = " << ifit;
      double xp = 0.1;
      double Q2 = 75.0;
      double xpq[13];
      double f2[2];
      double fl[2];
      double c2[2];
      double cl[2];
      qcd_2006(xp, Q2, ifit, xpq, f2, fl, c2, cl);
    } else {
      throw edm::Exception(edm::errors::Configuration, "PomwigError")
          << " Only running Pomeron H1 1997 (NSTRU=9), H1 2006 fit A (NSTRU=12) and H1 2006 fit B (NSTRU=14) or "
             "Reggeon H1 1997 (NSTRU=10), H1 2006 fit A (NSTRU=13) and H1 2006 fit B (NSTRU=15)";
    }

    return true;
  }

  bool PomwigHadronizer::declareStableParticles(const std::vector<int> &pdgIds) {
    for (std::vector<int>::const_iterator iter = pdgIds.begin(); iter != pdgIds.end(); ++iter)
      if (!markStable(*iter))
        return false;
    return true;
  }

  void PomwigHadronizer::statistics() {
    double RNWGT = 1. / hwevnt.NWGTS;
    double AVWGT = hwevnt.WGTSUM * RNWGT;

    double xsec = 1.0e3 * AVWGT;
    xsec = survivalProbability * xsec;

    runInfo().setInternalXSec(xsec);
  }

  bool PomwigHadronizer::hadronize() { return false; }

  bool PomwigHadronizer::generatePartonsAndHadronize() {
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
    if (useJimmy && doMPInteraction && !hwevnt.IERROR && call_hwmsct())
      return false;

    hwdhob();  // heavy quark decays
    hwcfor();  // cluster formation
    hwcdec();  // cluster decays

    // if event *not* killed by HERWIG, return true
    if (!hwevnt.IERROR)
      return true;

    hwufne();  // finalize event
    return false;
  }

  void PomwigHadronizer::finalizeEvent() {
    lhef::LHEEvent::fixHepMCEventTimeOrdering(event().get());

    event()->set_signal_process_id(hwproc.IPROC);

    event()->weights().push_back(hwevnt.EVWGT);
  }

  bool PomwigHadronizer::decay() {
    // hadron decays

    InstanceWrapper wrapper(this);  // safe guard

    hwdhad();  // unstable particle decays
    hwdhvy();  // heavy flavour decays
    hwmevt();  // soft underlying event

    hwufne();  // finalize event

    if (hwevnt.IERROR)
      return false;

    event().reset(new HepMC::GenEvent);
    if (!conv.fill_next_event(event().get()))
      throw cms::Exception("PomwigError") << "HepMC Conversion problems in event." << std::endl;

    // do particle ID conversion Herwig->PDG, if requested
    if (doPDGConvert) {
      for (HepMC::GenEvent::particle_iterator part = event()->particles_begin(); part != event()->particles_end();
           ++part) {
        if ((*part)->pdg_id() != HepPID::translateHerwigtoPDT((*part)->pdg_id()))
          (*part)->set_pdg_id(HepPID::translateHerwigtoPDT((*part)->pdg_id()));
      }
    }

    return true;
  }

  bool PomwigHadronizer::residualDecay() { return true; }

}  //namespace gen
