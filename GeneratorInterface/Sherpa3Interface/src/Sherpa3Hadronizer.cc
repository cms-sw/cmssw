#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <cstdint>
#include <map>
#include <regex>
#include <set>
#include <vector>

#include <dlfcn.h>
#include <glob.h>

#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Math/Random.H"

#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/MyStrStream.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Org/CXXFLAGS_PACKAGES.H"
#include "ATOOLS/Org/My_MPI.H"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "GeneratorInterface/Sherpa3Interface/interface/Sherpa3Utils.h"

#include "CLHEP/Random/RandomEngine.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <HepMC3/GenEvent.h>
#include "HepMC3/Print.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct3.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

//This unnamed namespace is used (instead of static variables) to pass the
//randomEngine passed to doSetRandomEngine to the External Random
//Number Generator CMS_SHERPA_HepMC3_RNG of sherpa
//The advantage of the unnamed namespace over static variables is
//that it is only accessible in this file

namespace {
  CLHEP::HepRandomEngine *ExternalEngine = nullptr;
  CLHEP::HepRandomEngine *GetExternalEngine() { return ExternalEngine; }
  void SetExternalEngine(CLHEP::HepRandomEngine *v) { ExternalEngine = v; }

  // ATOOLS::Exception does not derive from std::exception, so a Sherpa error that
  // escapes into the framework is reported as "an exception of unknown type was
  // thrown" without any message. Every call into Sherpa goes through this wrapper,
  // which turns it into a cms::Exception carrying the Sherpa diagnostics.
  template <typename F>
  auto callSherpa(const char *context, F &&f) -> decltype(f()) {
    try {
      return f();
    } catch (const ATOOLS::normal_exit &e) {
      std::ostringstream sherpaMessage;
      sherpaMessage << e;
      throw cms::Exception("Sherpa3Interface")
          << "Sherpa requested a normal exit while " << context << ":\n"
          << sherpaMessage.str();
    } catch (const ATOOLS::Exception &e) {
      std::ostringstream sherpaMessage;
      sherpaMessage << e;
      throw cms::Exception("Sherpa3Interface") << "Sherpa raised an exception while " << context << ":\n"
                                               << sherpaMessage.str();
    }
  }

  // Sherpa loads its optional plugin libraries itself, on demand: the model via
  // Library_Loader::LoadLibrary("Sherpa" + MODEL), the matrix element generators via
  // LoadLibrary("Sherpa" + generator), analyses and event outputs likewise - always
  // with RTLD_GLOBAL, and from SHERPA_LIBRARY_PATH, which the CMS build of Sherpa 3
  // takes from SHERPA3_LIBRARY_PATH. Preloading those libraries here instead would
  // register their getters in every job, whether the runcard asks for them or not:
  // the model libraries then re-register identical Lorentz calculators (the
  // 'Doubled identifier "DHVV*"' warnings), and libSherpaHiggs.so registers a
  // Tree_ME2 getter that reads HIGGS_INTERFERENCE_ONLY before the Higgs interface
  // has declared its defaults, which aborts the run. So preload everything except
  // the libSherpa*.so plugins, keeping only the core Sherpa libraries listed here.
  const std::array<std::string, 6> kCoreSherpaLibraries{{"libSherpaMain.so",
                                                         "libSherpaInitialization.so",
                                                         "libSherpaSingleEvents.so",
                                                         "libSherpaSoftPhysics.so",
                                                         "libSherpaPerturbativePhysics.so",
                                                         "libSherpaTools.so"}};

  bool isOnDemandSherpaPlugin(const std::string &libPath) {
    const std::string libName = libPath.substr(libPath.find_last_of('/') + 1);
    if (libName.compare(0, 9, "libSherpa") != 0)
      return false;
    return std::find(kCoreSherpaLibraries.begin(), kCoreSherpaLibraries.end(), libName) == kCoreSherpaLibraries.end();
  }

  // EDM-facing name for a Sherpa weight. Downstream CMSSW consumers mangle
  // characters outside [A-Za-z0-9._=]: RivetAnalyzer replaces each of them with
  // '_' before calling HepMC3::GenRunInfo::set_weight_names, which rejects
  // duplicate names. Sherpa's polarised cross-section weights use '+' and '-'
  // (e.g. "PolWeight_COM.W+.+_W+.-") and collide after that cleaning
  // ("W_._W_._" for all four +/- combinations), so they are renamed here:
  // '+' -> 'p', '-' -> 'm'. The original Sherpa name remains the lookup key
  // into the event weights; only the stored name changes.
  std::string edmWeightName(const std::string &sherpaName) {
    std::string name = sherpaName;
    std::replace(name.begin(), name.end(), '+', 'p');
    std::replace(name.begin(), name.end(), '-', 'm');
    return name;
  }
}  // namespace

class Sherpa3Hadronizer : public gen::BaseHadronizer {
public:
  Sherpa3Hadronizer(const edm::ParameterSet &params);
  ~Sherpa3Hadronizer() override;

  bool readSettings(int) { return true; }
  bool initializeForInternalPartons();
  bool declareStableParticles(const std::vector<int> &pdgIds);
  bool declareSpecialSettings(const std::vector<std::string> &) { return true; }
  void statistics();
  bool generatePartonsAndHadronize();
  bool decay();
  bool residualDecay();
  void finalizeEvent();
  std::unique_ptr<GenLumiInfoHeader> getGenLumiInfoHeader() const override;
  const char *classname() const { return "Sherpa3Hadronizer"; }

private:
  void doSetRandomEngine(CLHEP::HepRandomEngine *v) override;
  void preloadSherpaLibraries() const;

  std::string SherpaProcess;
  std::string EvtGenDirectory;
  std::string SherpaResultDir;
  edm::ParameterSet SherpaParameterSet;
  unsigned int maxEventsToPrint;
  std::vector<std::string> arguments;
  // Sherpa 3 has no default constructor; the generator is created in
  // initializeForInternalPartons() once the command-line arguments are known
  std::unique_ptr<SHERPA::Sherpa> Generator;
  bool isInitialized;
  bool isRNGinitialized;
  bool rearrangeWeights;
  bool warnedAboutUnnamedWeights;
  std::vector<std::string> weightlist;
  std::vector<std::string> variationweightlist;
  // EDM-facing names, one per weightlist/variationweightlist entry, in the same
  // order: either the optional SherpaRenamedWeights/SherpaRenamedVariationWeights
  // from the SherpaWeightsBlock, or derived from the Sherpa names with
  // edmWeightName. Used for the lumi header and the HepMC3 run info.
  std::vector<std::string> storedWeightNames;
  std::vector<std::string> storedVariationWeightNames;
};

class CMS_SHERPA_HepMC3_RNG : public ATOOLS::External_RNG {
public:
  CMS_SHERPA_HepMC3_RNG() : randomEngine(nullptr) {
    edm::LogVerbatim("Sherpa3Hadronizer") << "Use stored reference for the external RNG";
    setRandomEngine(GetExternalEngine());
  }
  void setRandomEngine(CLHEP::HepRandomEngine *v) { randomEngine = v; }

private:
  double Get() override;
  CLHEP::HepRandomEngine *randomEngine;
};

void Sherpa3Hadronizer::doSetRandomEngine(CLHEP::HepRandomEngine *v) {
  // In Sherpa 3 the global ATOOLS::ran is only created by the SHERPA::Sherpa
  // constructor, which runs later (in initializeForInternalPartons); it is
  // still null when the framework calls this method for the first time
  CMS_SHERPA_HepMC3_RNG *cmsSherpaRng =
      ATOOLS::ran ? dynamic_cast<CMS_SHERPA_HepMC3_RNG *>(ATOOLS::ran->GetExternalRng()) : nullptr;
  if (cmsSherpaRng == nullptr) {
    //First time call to this function makes the interface store the reference in the unnamed namespace
    if (!isRNGinitialized) {
      isRNGinitialized = true;
      edm::LogVerbatim("Sherpa3Hadronizer") << "Store assigned reference of the randomEngine";
      SetExternalEngine(v);
      // Throw exception if there is no reference to an external RNG and it is not the first call!
    } else {
      if (isInitialized and v != nullptr) {
        throw edm::Exception(edm::errors::LogicError)
            << "The Sherpa interface got a randomEngine reference but there is "
               "no reference to the external RNG to hand it over to\n";
      }
    }
  } else {
    cmsSherpaRng->setRandomEngine(v);
  }
}

Sherpa3Hadronizer::Sherpa3Hadronizer(const edm::ParameterSet &params)
    : BaseHadronizer(params),
      SherpaParameterSet(params.getParameter<edm::ParameterSet>("Sherpa3Parameters")),
      isInitialized(false),
      isRNGinitialized(false),
      rearrangeWeights(false),
      warnedAboutUnnamedWeights(false) {
  // Set the HepMC version to 3 for the Sherpa 3 interface. BaseHadronizer defaults
  // this to 2, and GeneratorFilter reads it once at initialization to decide whether
  // to produce HepMCProduct/GenEventInfoProduct or HepMC3Product/GenEventInfoProduct3.
  // Without this the filter looks for a HepMC2 event, never finds one, and rejects
  // every event.
  ivhepmc = 3;
  if (!params.exists("Sherpa3Process"))
    SherpaProcess = "";
  else
    SherpaProcess = params.getParameter<std::string>("Sherpa3Process");
  // directory where Sherpa.yaml is written and from which Sherpa reads it
  // (the gridpack is unpacked into the current working directory)
  if (!params.exists("EvtGenDirectory"))
    EvtGenDirectory = "./SHERPA3GEN";
  else
    EvtGenDirectory = params.getParameter<std::string>("EvtGenDirectory");
  // directory of the Sherpa results DB (<RESULT_DIRECTORY>.zip); the
  // Sherpa3 gridpacks store it with the Sherpa default "Results"
  // (Results.zip), so do not change this unless the gridpacks are
  // produced with a matching setting
  if (!params.exists("SherpaResultDir"))
    SherpaResultDir = "Results";
  else
    SherpaResultDir = params.getParameter<std::string>("SherpaResultDir");
  if (!params.exists("maxEventsToPrint")) {
    maxEventsToPrint = 0;
  } else {
    const int nEventsToPrint = params.getParameter<int>("maxEventsToPrint");
    if (nEventsToPrint < 0)
      throw cms::Exception("Sherpa3Interface")
          << "maxEventsToPrint must not be negative, got " << nEventsToPrint << std::endl;
    maxEventsToPrint = nEventsToPrint;
  }
  // if hepmcextendedweights is used the event weights have to be reordered ( unordered list can be accessed via event->weights().write() )
  // two lists have to be provided:
  // 1) SherpaWeights
  // - containing nominal event weight, combined matrix element and phase space weight, event normalization, and possibly other sherpa weights
  // 2) SherpaVariationsWeights
  // - containing weights from scale and PDF variations ( have to be defined in the runcard )
  // - in case of unweighted events these weights are also divided by the event normalization (see list 1 )
  // Sherpa Documentation: http://sherpa.hepforge.org/doc/SHERPA-MC-2.2.0.html#Scale-and-PDF-variations
  if (!params.exists("SherpaWeightsBlock")) {
    rearrangeWeights = false;
  } else {
    rearrangeWeights = true;
    edm::ParameterSet WeightsBlock = params.getParameter<edm::ParameterSet>("SherpaWeightsBlock");
    if (WeightsBlock.exists("SherpaWeights"))
      weightlist = WeightsBlock.getParameter<std::vector<std::string> >("SherpaWeights");
    else
      throw cms::Exception("Sherpa3Interface") << "SherpaWeights does not exists in SherpaWeightsBlock" << std::endl;
    if (WeightsBlock.exists("SherpaVariationWeights"))
      variationweightlist = WeightsBlock.getParameter<std::vector<std::string> >("SherpaVariationWeights");
    else
      throw cms::Exception("Sherpa3Interface")
          << "SherpaVariationWeights does not exists in SherpaWeightsBlock" << std::endl;
    // EDM-facing names for the weights, optional SherpaRenamedWeights /
    // SherpaRenamedVariationWeights with one entry per SherpaWeights /
    // SherpaVariationWeights entry in the same order; if absent, the names are
    // derived from the Sherpa names with edmWeightName ('+' -> 'p', '-' -> 'm')
    if (WeightsBlock.exists("SherpaRenamedWeights")) {
      storedWeightNames = WeightsBlock.getParameter<std::vector<std::string> >("SherpaRenamedWeights");
      if (storedWeightNames.size() != weightlist.size())
        throw cms::Exception("Sherpa3Interface")
            << "SherpaRenamedWeights has " << storedWeightNames.size() << " entries but SherpaWeights has "
            << weightlist.size() << std::endl;
    } else {
      storedWeightNames.reserve(weightlist.size());
      for (const auto &i : weightlist)
        storedWeightNames.push_back(edmWeightName(i));
    }
    if (WeightsBlock.exists("SherpaRenamedVariationWeights")) {
      storedVariationWeightNames =
          WeightsBlock.getParameter<std::vector<std::string> >("SherpaRenamedVariationWeights");
      if (storedVariationWeightNames.size() != variationweightlist.size())
        throw cms::Exception("Sherpa3Interface")
            << "SherpaRenamedVariationWeights has " << storedVariationWeightNames.size()
            << " entries but SherpaVariationWeights has " << variationweightlist.size() << std::endl;
    } else {
      storedVariationWeightNames.reserve(variationweightlist.size());
      for (const auto &i : variationweightlist)
        storedVariationWeightNames.push_back(edmWeightName(i));
    }
    // the stored names go into the lumi header and the HepMC3 run info, where
    // duplicates are rejected (HepMC3::GenRunInfo::set_weight_names throws)
    std::set<std::string> storedNames;
    for (const auto &i : storedWeightNames)
      if (!storedNames.insert(i).second)
        throw cms::Exception("Sherpa3Interface")
            << "Duplicate stored weight name '" << i << "' in SherpaWeights" << std::endl;
    for (const auto &i : storedVariationWeightNames)
      if (!storedNames.insert(i).second)
        throw cms::Exception("Sherpa3Interface")
            << "Duplicate stored weight name '" << i << "' in SherpaVariationWeights" << std::endl;
    edm::LogVerbatim("Sherpa3Hadronizer")
        << "Sherpa3Hadronizer will try rearrange the event weights according to "
           "SherpaWeights and SherpaVariationWeights";
  }

  sh3utils::Sherpa3Utils Fetcher(params);
  int retval = Fetcher.Fetch();
  if (retval != 0) {
    throw cms::Exception("Sherpa3Interface") << "Sherpa3Hadronizer: Preparation of Gridpack failed ... ";
  }
  // Sherpa 3 reads a single YAML runcard (Sherpa.yaml in the PATH directory)
  // instead of the per-section *.dat files used by Sherpa 2.
  // The ids (names) of parameter sets to be read are given as a vstring;
  // the contents of all sets are concatenated, in the given order, into Sherpa.yaml.
  std::vector<std::string> setNames = SherpaParameterSet.getParameter<std::vector<std::string> >("parameterSets");
  std::string datfile = EvtGenDirectory + "/Sherpa.yaml";
  std::ofstream os(datfile.c_str());
  if (!os.is_open())
    throw cms::Exception("Sherpa3Interface") << "Could not open the Sherpa runcard " << datfile << " for writing"
                                             << std::endl;
  edm::LogVerbatim("Sherpa3Hadronizer") << "Write Sherpa parameter sets to " << datfile;
  // OpenLoops ships with the CMSSW release, so an OL_PREFIX baked into the
  // fragment points at the wrong installation once the job runs with a
  // different release. Unless the fragment opts out with
  // FIXED_OL_PREFIX = cms.bool(True) in Sherpa3Parameters, replace it with the
  // prefix of the running release (CMS_OPENLOOPS_PREFIX is set by the
  // openloops scram tool).
  const char *openloopsPrefix = std::getenv("CMS_OPENLOOPS_PREFIX");
  const bool fixedOLPrefix =
      SherpaParameterSet.exists("FIXED_OL_PREFIX") && SherpaParameterSet.getParameter<bool>("FIXED_OL_PREFIX");
  const std::regex olPrefixSetting("^\\s*OL_PREFIX\\s*:.*");
  //Loop all set names...
  for (unsigned i = 0; i < setNames.size(); ++i) {
    // ...and read the parameters for each set given in vstrings
    std::vector<std::string> pars = SherpaParameterSet.getParameter<std::vector<std::string> >(setNames[i]);
    // Loop over all strings and write them to the runcard
    for (std::vector<std::string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar) {
      std::string line = *itPar;
      if (std::regex_match(line, olPrefixSetting)) {
        if (fixedOLPrefix) {
          edm::LogVerbatim("Sherpa3Hadronizer")
              << "FIXED_OL_PREFIX is set in Sherpa3Parameters, keeping '" << line << "' from the fragment"
              << std::endl;
        } else if (openloopsPrefix != nullptr) {
          edm::LogVerbatim("Sherpa3Hadronizer")
              << "Overriding '" << line
              << "' from the fragment with the OpenLoops installation of the running release" << std::endl;
          line = std::string("OL_PREFIX: ") + openloopsPrefix;
        } else {
          edm::LogWarning("Sherpa3Hadronizer")
              << "The fragment sets '" << line
              << "' but CMS_OPENLOOPS_PREFIX is not set (openloops scram tool not in the environment);"
                 " keeping the fragment value" << std::endl;
        }
      }
      os << line << std::endl;
    }
  }
  os.close();
  if (os.fail())
    throw cms::Exception("Sherpa3Interface") << "Failed to write the Sherpa runcard " << datfile << std::endl;

  //To be conform to the default Sherpa usage create a command line:
  //name of executable  (only for demonstration, could also be empty)
  std::string shRun = "Sherpa3";
  //Path where Sherpa.yaml is read from (Sherpa 3 setting PATH)
  std::string shPath = "PATH=" + EvtGenDirectory;
  //Path where results are stored
  std::string shRes = "RESULT_DIRECTORY=" + SherpaResultDir;
  //Name of the external random number class
  std::string shRng = "EXTERNAL_RNG=CMS_SHERPA_HepMC3_RNG";

  //create the command line
  arguments.push_back(shRun);
  arguments.push_back(shPath);
  arguments.push_back(shRes);
  arguments.push_back(shRng);
  //initialization of Sherpa moved to initializeForInternalPartons
#ifdef USING__MPI
  // FIXME this should be replaced with a call to the MPIService
  int argc = 0;
  char **argv = nullptr;
  MPI_Init(&argc, &argv);
#endif
}

Sherpa3Hadronizer::~Sherpa3Hadronizer() {
  // ~Sherpa tears down the ATOOLS globals (and, with MPI, calls MPI_Barrier), so it
  // has to run before MPI_Finalize below
  Generator.reset();
#ifdef USING__MPI
  MPI_Finalize();
#endif
}

void Sherpa3Hadronizer::preloadSherpaLibraries() const {
  // Sherpa 3 libraries do not record inter-library dependencies (libYFSMain.so, for
  // instance, does not list libYFSTools.so, which defines the symbols it calls), and
  // CMSSW loads this plugin in a local symbol scope, so cross-library calls would
  // fail when first used. Preload the Sherpa libraries into the global symbol scope
  // before initializing Sherpa. Data relocations are resolved eagerly at dlopen time,
  // so a library whose dependencies have not been loaded yet fails; iterate until no
  // more progress is made.
  const char *sherpaLibPath = std::getenv("SHERPA3_LIBRARY_PATH");
  // No fallback to SHERPA_LIBRARY_PATH: with the sherpa (Sherpa 2) tool in the
  // runtime environment that variable points at the Sherpa 2 libraries, and pulling
  // those into the global scope would be worse than not preloading at all.
  if (sherpaLibPath == nullptr)
    throw cms::Exception("Sherpa3Interface")
        << "SHERPA3_LIBRARY_PATH is not set. The sherpa3 scram tool environment is required to run\n"
        << "Sherpa 3; run cmsenv in the CMSSW area before cmsRun. Without the library preload the\n"
        << "Sherpa 3 libraries cannot resolve each other's symbols and the job dies with a symbol\n"
        << "lookup error." << std::endl;

  const std::string pattern = std::string(sherpaLibPath) + "/lib*.so";
  std::vector<std::string> libraries;
  glob_t globResult;
  if (glob(pattern.c_str(), 0, nullptr, &globResult) == 0) {
    for (size_t i = 0; i < globResult.gl_pathc; ++i)
      libraries.emplace_back(globResult.gl_pathv[i]);
    globfree(&globResult);
  }
  if (libraries.empty())
    throw cms::Exception("Sherpa3Interface")
        << "No Sherpa libraries found in " << sherpaLibPath << " (looking for " << pattern << ")" << std::endl;

  std::vector<std::string> librariesToPreload;
  std::copy_if(libraries.begin(),
               libraries.end(),
               std::back_inserter(librariesToPreload),
               [](const std::string &lib) { return !isOnDemandSherpaPlugin(lib); });

  size_t loaded = 0;
  size_t previouslyLoaded = 0;
  std::string lastError;
  do {
    previouslyLoaded = loaded;
    loaded = 0;
    lastError.clear();
    for (const auto &lib : librariesToPreload) {
      if (dlopen(lib.c_str(), RTLD_LAZY | RTLD_GLOBAL) != nullptr) {
        ++loaded;
      } else {
        // dlerror() has to be read right after the failing call, and may be null
        const char *error = dlerror();
        lastError = (error != nullptr) ? error : "unknown dlopen error";
      }
    }
  } while (loaded > previouslyLoaded && loaded < librariesToPreload.size());

  edm::LogVerbatim("Sherpa3Hadronizer") << "Preloaded " << loaded << " of " << librariesToPreload.size()
                                        << " Sherpa libraries from " << sherpaLibPath << " ("
                                        << (libraries.size() - librariesToPreload.size())
                                        << " on-demand plugin libraries skipped, Sherpa loads those itself)";
  if (loaded < librariesToPreload.size())
    edm::LogWarning("Sherpa3Hadronizer") << "Only " << loaded << " of " << librariesToPreload.size()
                                         << " Sherpa libraries in " << sherpaLibPath
                                         << " could be preloaded; last dlopen error: " << lastError;
}

bool Sherpa3Hadronizer::initializeForInternalPartons() {
  //initialize Sherpa but only once
  if (!isInitialized) {
    preloadSherpaLibraries();

    std::vector<char *> argv;
    argv.reserve(arguments.size());
    for (auto &argument : arguments)
      argv.push_back(const_cast<char *>(argument.c_str()));
    // Sherpa 3 takes the command-line arguments in the constructor
    Generator = callSherpa("constructing the Sherpa generator", [&argv]() {
      return std::make_unique<SHERPA::Sherpa>(static_cast<int>(argv.size()), argv.data());
    });
    if (!callSherpa("initializing the Sherpa run", [this]() { return Generator->InitializeTheRun(); }))
      throw cms::Exception("Sherpa3Interface") << "Sherpa::InitializeTheRun() failed" << std::endl;
    if (!callSherpa("initializing the Sherpa event handler",
                    [this]() { return Generator->InitializeTheEventHandler(); }))
      throw cms::Exception("Sherpa3Interface") << "Sherpa::InitializeTheEventHandler() failed" << std::endl;
    isInitialized = true;
  }
  return true;
}

bool Sherpa3Hadronizer::declareStableParticles(const std::vector<int> &pdgIds) { return false; }

void Sherpa3Hadronizer::statistics() {
  // statistics() is called at the end of the job whether or not the generator was
  // ever initialized
  if (!Generator) {
    edm::LogWarning("Sherpa3Hadronizer") << "No Sherpa generator, skipping the run summary";
    return;
  }

  //calculate statistics
  callSherpa("summarizing the Sherpa run", [this]() { return Generator->SummarizeRun(); });

  //get the xsec & err
  double xsec_val = Generator->TotalXS();
  double xsec_err = Generator->TotalErr();

  //set the internal cross section in pb in GenRunInfoProduct
  runInfo().setInternalXSec(GenRunInfoProduct::XSec(xsec_val, xsec_err));
}

bool Sherpa3Hadronizer::generatePartonsAndHadronize() {
  //get the next event and check if it produced
  bool rc = false;
  std::string lastError;
  for (int itry = 1; itry <= 3; ++itry) {
    try {
      rc = Generator->GenerateOneEvent();
      lastError.clear();
      break;
    } catch (const ATOOLS::Exception &e) {
      std::ostringstream sherpaMessage;
      sherpaMessage << e;
      lastError = sherpaMessage.str();
    } catch (...) {
      lastError = "exception of unknown type";
    }
    edm::LogWarning("Sherpa3Hadronizer") << "Exception from Generator->GenerateOneEvent(), call # " << itry
                                         << " for this event:\n"
                                         << lastError;
  }
  if (!lastError.empty())
    throw cms::Exception("Sherpa3Interface") << "Sherpa failed to generate an event in 3 attempts, last error:\n"
                                             << lastError << std::endl;
  if (rc) {
    //convert it to HepMC3. Sherpa attaches its own GenRunInfo to the event
    //(HepMC3_Interface::Sherpa2HepMC calls set_run_info), so do not create one here.
    auto evt = std::make_unique<HepMC3::GenEvent>();
    callSherpa("filling the HepMC3 event", [this, &evt]() { Generator->FillHepMCEvent(*evt); });

    // in case of unweighted events sherpa puts the max weight as event weight.
    // this is not optimal, we want 1 for unweighted events, so we check
    // whether we are producing unweighted events ("EVENT_GENERATION_MODE" == "1")
    // In Sherpa 3 the HepMC3 output uses named weights by default
    // (HEPMC_USE_NAMED_WEIGHTS: true):
    //   "Weight"                     event weight
    //   <variation names>            on-the-fly scale/PDF variation weights
    //   "EXTRA__MEWeight"            combined matrix element and phase space weight
    //                                (missing only PDF information, thus directly
    //                                suitable for PDF reweighting)
    //   "EXTRA__WeightNormalisation" event weight normalisation (in case of
    //                                unweighted events event weights of ~ +/-1 can
    //                                be obtained by (event weight)/(normalisation))
    //   "EXTRA__NTrials"             number of trials
    // With unnamed weights the legacy index layout [0..3] applies instead.
    if (!evt->run_info())
      throw cms::Exception("Sherpa3Interface") << "Sherpa did not attach a GenRunInfo to the HepMC3 event"
                                               << std::endl;

    // Build name->index map
    const std::vector<std::string> &weight_list = evt->run_info()->weight_names();
    std::map<std::string, std::size_t> nameToIndex;
    for (std::size_t i = 0; i < weight_list.size(); ++i) {
      nameToIndex[weight_list[i]] = i;
    }
    // Helper lambda: get weight by name; fall back to the legacy index layout
    // only when no named weights are available at all
    auto getWeightByName = [&](const std::string &name, std::size_t fallbackIdx) -> double {
      if (!nameToIndex.empty()) {
        auto it = nameToIndex.find(name);
        if (it != nameToIndex.end() && it->second < evt->weights().size())
          return evt->weights()[it->second];
        throw cms::Exception("Sherpa3Interface")
            << "Weight '" << name << "' not found in the named HepMC weights, please check the runcard!" << std::endl;
      }
      if (fallbackIdx < evt->weights().size())
        return evt->weights()[fallbackIdx];
      throw cms::Exception("Sherpa3Interface") << "Missing weight at index " << fallbackIdx << std::endl;
    };

    bool unweighted = false;
    double weight_normalization = -1;
    int EVENT_GENERATION_MODE = ATOOLS::ToType<int>(ATOOLS::rpa->gen.Variable("EVENT_GENERATION_MODE"));
    if ((EVENT_GENERATION_MODE == 1) || (EVENT_GENERATION_MODE == 2)) {
      // EVENT_GENERATION_MODE: 1->Unweighted; 2->PartiallyUnweighted;
      if (evt->weights().size() > 2) {
        unweighted = true;
        weight_normalization = getWeightByName("EXTRA__WeightNormalisation", 2);
        if (weight_normalization == 0.)
          throw cms::Exception("Sherpa3Interface")
              << "Requested unweighted production but the event weight normalization is zero." << std::endl;
      } else {
        throw cms::Exception("Sherpa3Interface")
            << "Requested unweighted production. Missing normalization weight." << std::endl;
      }
    }
    // vector to fill new weights in correct order
    std::vector<double> newWeights;
    if (rearrangeWeights) {
      // the Sherpa names are the lookup keys; the EDM-facing names
      // (storedWeightNames/storedVariationWeightNames) go into the run info
      std::vector<std::string> newWeightNames;
      for (std::size_t iw = 0; iw < weightlist.size(); ++iw) {
        auto it = nameToIndex.find(weightlist[iw]);
        if (it != nameToIndex.end()) {
          newWeights.push_back(evt->weights()[it->second]);
          newWeightNames.push_back(storedWeightNames[iw]);
        } else {
          throw cms::Exception("Sherpa3Interface")
              << "Missing weights! Key " << weightlist[iw]
              << " not found, please check the weight definition!" << std::endl;
        }
      }
      for (std::size_t iw = 0; iw < variationweightlist.size(); ++iw) {
        auto it = nameToIndex.find(variationweightlist[iw]);
        if (it != nameToIndex.end()) {
          double w = evt->weights()[it->second];
          newWeights.push_back(unweighted ? w / weight_normalization : w);
          newWeightNames.push_back(storedVariationWeightNames[iw]);
        } else {
          throw cms::Exception("Sherpa3Interface")
              << "Missing weights! Key " << variationweightlist[iw]
              << " not found, please check the weight definition!" << std::endl;
        }
      }

      //Change original weights for reordered ones, and keep the names in the run
      //info in step with them (Sherpa rewrites the names for every event, so this
      //has to be redone here for every event as well)
      evt->weights() = newWeights;
      evt->run_info()->set_weight_names(newWeightNames);
    } else if (!warnedAboutUnnamedWeights && evt->weights().size() > 1) {
      // Nothing downstream carries the Sherpa weight names: HepMC3Product stores
      // no GenRunInfo and GenEventInfoProduct3 stores values only, so without a
      // SherpaWeightsBlock the weights are written unlabelled and cannot be
      // interpreted later. Say so once, with the names actually on offer.
      warnedAboutUnnamedWeights = true;
      edm::LogWarning("Sherpa3Hadronizer")
          << evt->weights().size() << " event weights are being stored without names, because no "
          << "SherpaWeightsBlock is configured.\nThe names are known only to Sherpa (they depend on "
          << "the runcard) and are lost in the output.\nRegenerate the fragment with "
          << "mkSherpa3Gridpack.sh (or mkSherpa3cff.sh -w) to embed them. Sherpa offers:\n"
          << [&weight_list]() {
               std::ostringstream names;
               for (const auto &name : weight_list)
                 names << "  " << name << "\n";
               return names.str();
             }();
    }
    if (unweighted) {
      if (evt->weights().empty())
        throw cms::Exception("Sherpa3Interface")
            << "Requested unweighted production but the event has no weights left to normalize." << std::endl;
      evt->weights()[0] /= weight_normalization;
    }
    resetEvent3(std::move(evt));
    return true;
  } else {
    return false;
  }
}

bool Sherpa3Hadronizer::decay() { return true; }

bool Sherpa3Hadronizer::residualDecay() { return true; }

void Sherpa3Hadronizer::finalizeEvent() {
  eventInfo3() = std::make_unique<GenEventInfoProduct3>(event3().get());
  //******** Verbosity *******
  if (maxEventsToPrint > 0) {
    maxEventsToPrint--;
    HepMC3::Print::listing(*(event3().get()));
  }
}

//GETTER for the external random numbers
DECLARE_GETTER(CMS_SHERPA_HepMC3_RNG, "CMS_SHERPA_HepMC3_RNG", ATOOLS::External_RNG, ATOOLS::RNG_Key);

ATOOLS::External_RNG *ATOOLS::Getter<ATOOLS::External_RNG, ATOOLS::RNG_Key, CMS_SHERPA_HepMC3_RNG>::operator()(
    const ATOOLS::RNG_Key &) const {
  return new CMS_SHERPA_HepMC3_RNG();
}

void ATOOLS::Getter<ATOOLS::External_RNG, ATOOLS::RNG_Key, CMS_SHERPA_HepMC3_RNG>::PrintInfo(std::ostream &str,
                                                                                             const size_t) const {
  str << "CMS_SHERPA_HepMC3_RNG interface";
}

double CMS_SHERPA_HepMC3_RNG::Get() {
  if (randomEngine == nullptr) {
    throw edm::Exception(edm::errors::LogicError) << "The Sherpa code attempted to a generate random number while\n"
                                                  << "the engine pointer was null. This might mean that the code\n"
                                                  << "was modified to generate a random number outside the event and\n"
                                                  << "beginLuminosityBlock methods, which is not allowed.\n";
  }
  return randomEngine->flat();
}
std::unique_ptr<GenLumiInfoHeader> Sherpa3Hadronizer::getGenLumiInfoHeader() const {
  auto genLumiInfoHeader = BaseHadronizer::getGenLumiInfoHeader();

  if (rearrangeWeights) {
    edm::LogPrint("Sherpa3Hadronizer") << "The order of event weights was changed!";
    // store the EDM-facing names; log the Sherpa name alongside when it differs
    auto pushNames = [&genLumiInfoHeader](const std::vector<std::string> &sherpaNames,
                                          const std::vector<std::string> &storedNames) {
      for (std::size_t i = 0; i < sherpaNames.size(); ++i) {
        genLumiInfoHeader->weightNames().push_back(storedNames[i]);
        if (storedNames[i] == sherpaNames[i])
          edm::LogVerbatim("Sherpa3Hadronizer") << storedNames[i];
        else
          edm::LogVerbatim("Sherpa3Hadronizer") << storedNames[i] << "  (Sherpa name: " << sherpaNames[i] << ")";
      }
    };
    pushNames(weightlist, storedWeightNames);
    pushNames(variationweightlist, storedVariationWeightNames);
  }

  return genLumiInfoHeader;
}

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

typedef edm::GeneratorFilter<Sherpa3Hadronizer, gen::ExternalDecayDriver> Sherpa3GeneratorFilter;
DEFINE_FWK_MODULE(Sherpa3GeneratorFilter);
