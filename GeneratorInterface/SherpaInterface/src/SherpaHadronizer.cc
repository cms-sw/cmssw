#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <cstdint>
#include <vector>

#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Math/Random.H"

#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/MyStrStream.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Org/CXXFLAGS_PACKAGES.H"
#include "ATOOLS/Org/My_MPI.H"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "GeneratorInterface/SherpaInterface/interface/SherpackFetcher.h"

#include "CLHEP/Random/RandomEngine.h"

//This unnamed namespace is used (instead of static variables) to pass the
//randomEngine passed to doSetRandomEngine to the External Random
//Number Generator CMS_SHERPA_RNG of sherpa
//The advantage of the unnamed namespace over static variables is
//that it is only accessible in this file

namespace {
  CLHEP::HepRandomEngine *ExternalEngine = nullptr;
  CLHEP::HepRandomEngine *GetExternalEngine() { return ExternalEngine; }
  void SetExternalEngine(CLHEP::HepRandomEngine *v) { ExternalEngine = v; }
}  // namespace

class SherpaHadronizer : public gen::BaseHadronizer {
public:
  SherpaHadronizer(const edm::ParameterSet &params);
  ~SherpaHadronizer() override;

  bool readSettings(int) { return true; }
  bool initializeForInternalPartons();
  bool declareStableParticles(const std::vector<int> &pdgIds);
  bool declareSpecialSettings(const std::vector<std::string> &) { return true; }
  void statistics();
  bool generatePartonsAndHadronize();
  bool decay();
  bool rearrangeWeights;
  bool residualDecay();
  void finalizeEvent();
  std::unique_ptr<GenLumiInfoHeader> getGenLumiInfoHeader() const override;
  const char *classname() const { return "SherpaHadronizer"; }

private:
  void doSetRandomEngine(CLHEP::HepRandomEngine *v) override;

  std::string SherpaProcess;
  std::string SherpaChecksum;
  std::string SherpaPath;
  std::string SherpaPathPiece;
  std::string SherpaResultDir;
  double SherpaDefaultWeight;
  edm::ParameterSet SherpaParameterSet;
  unsigned int maxEventsToPrint;
  std::vector<std::string> arguments;
  SHERPA::Sherpa *Generator = new SHERPA::Sherpa();
  bool isInitialized;
  bool isRNGinitialized;
  std::vector<std::string> weightlist;
  std::vector<std::string> variationweightlist;
};

class CMS_SHERPA_RNG : public ATOOLS::External_RNG {
public:
  CMS_SHERPA_RNG() : randomEngine(nullptr) {
    edm::LogVerbatim("SherpaHadronizer") << "Use stored reference for the external RNG";
    setRandomEngine(GetExternalEngine());
  }
  void setRandomEngine(CLHEP::HepRandomEngine *v) { randomEngine = v; }

private:
  double Get() override;
  CLHEP::HepRandomEngine *randomEngine;
};

void SherpaHadronizer::doSetRandomEngine(CLHEP::HepRandomEngine *v) {
  CMS_SHERPA_RNG *cmsSherpaRng = dynamic_cast<CMS_SHERPA_RNG *>(ATOOLS::ran->GetExternalRng());
  if (cmsSherpaRng == nullptr) {
    //First time call to this function makes the interface store the reference in the unnamed namespace
    if (!isRNGinitialized) {
      isRNGinitialized = true;
      edm::LogVerbatim("SherpaHadronizer") << "Store assigned reference of the randomEngine";
      SetExternalEngine(v);
      // Throw exception if there is no reference to an external RNG and it is not the first call!
    } else {
      throw edm::Exception(edm::errors::LogicError) << "The Sherpa interface got a randomEngine reference but there is "
                                                       "no reference to the external RNG to hand it over to\n";
    }
  } else {
    cmsSherpaRng->setRandomEngine(v);
  }
}

SherpaHadronizer::SherpaHadronizer(const edm::ParameterSet &params)
    : BaseHadronizer(params),
      SherpaParameterSet(params.getParameter<edm::ParameterSet>("SherpaParameters")),
      isRNGinitialized(false) {
  if (!params.exists("SherpaProcess"))
    SherpaProcess = "";
  else
    SherpaProcess = params.getParameter<std::string>("SherpaProcess");
  if (!params.exists("SherpaPath"))
    SherpaPath = "";
  else
    SherpaPath = params.getParameter<std::string>("SherpaPath");
  if (!params.exists("SherpaPathPiece"))
    SherpaPathPiece = "";
  else
    SherpaPathPiece = params.getParameter<std::string>("SherpaPathPiece");
  if (!params.exists("SherpaResultDir"))
    SherpaResultDir = "Result";
  else
    SherpaResultDir = params.getParameter<std::string>("SherpaResultDir");
  if (!params.exists("SherpaDefaultWeight"))
    SherpaDefaultWeight = 1.;
  else
    SherpaDefaultWeight = params.getParameter<double>("SherpaDefaultWeight");
  if (!params.exists("maxEventsToPrint"))
    maxEventsToPrint = 0;
  else
    maxEventsToPrint = params.getParameter<int>("maxEventsToPrint");
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
      throw cms::Exception("SherpaInterface") << "SherpaWeights does not exists in SherpaWeightsBlock" << std::endl;
    if (WeightsBlock.exists("SherpaVariationWeights"))
      variationweightlist = WeightsBlock.getParameter<std::vector<std::string> >("SherpaVariationWeights");
    else
      throw cms::Exception("SherpaInterface")
          << "SherpaVariationWeights does not exists in SherpaWeightsBlock" << std::endl;
    edm::LogVerbatim("SherpaHadronizer") << "SherpaHadronizer will try rearrange the event weights according to "
                                            "SherpaWeights and SherpaVariationWeights";
  }

  spf::SherpackFetcher Fetcher(params);
  int retval = Fetcher.Fetch();
  if (retval != 0) {
    throw cms::Exception("SherpaInterface") << "SherpaHadronizer: Preparation of Sherpack failed ... ";
  }
  // The ids (names) of parameter sets to be read (Analysis,Run) to create Analysis.dat, Run.dat
  //They are given as a vstring.
  std::vector<std::string> setNames = SherpaParameterSet.getParameter<std::vector<std::string> >("parameterSets");
  //Loop all set names...
  for (unsigned i = 0; i < setNames.size(); ++i) {
    // ...and read the parameters for each set given in vstrings
    std::vector<std::string> pars = SherpaParameterSet.getParameter<std::vector<std::string> >(setNames[i]);
    edm::LogVerbatim("SherpaHadronizer") << "Write Sherpa parameter set " << setNames[i] << " to " << setNames[i]
                                         << ".dat ";
    std::string datfile = SherpaPath + "/" + setNames[i] + ".dat";
    std::ofstream os(datfile.c_str());
    // Loop over all strings and write the according *.dat
    for (std::vector<std::string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar) {
      os << (*itPar) << std::endl;
    }
  }

  //To be conform to the default Sherpa usage create a command line:
  //name of executable  (only for demonstration, could also be empty)
  std::string shRun = "./Sherpa";
  //Path where the Sherpa libraries are stored
  std::string shPath = "PATH=" + SherpaPath;
  // new for Sherpa 1.3.0, suggested by authors
  std::string shPathPiece = "PATH_PIECE=" + SherpaPathPiece;
  //Path where results are stored
  std::string shRes = "RESULT_DIRECTORY=" + SherpaResultDir;  // from Sherpa 1.2.0 on
  //Name of the external random number class
  std::string shRng = "EXTERNAL_RNG=CMS_SHERPA_RNG";

  //create the command line
  arguments.push_back(shRun);
  arguments.push_back(shPath);
  arguments.push_back(shPathPiece);
  arguments.push_back(shRes);
  arguments.push_back(shRng);
  isInitialized = false;
  //initialization of Sherpa moved to initializeForInternalPartons
}

SherpaHadronizer::~SherpaHadronizer() {
  Generator->~Sherpa();
#ifdef USING__MPI
  MPI::Finalize();
#endif
}

bool SherpaHadronizer::initializeForInternalPartons() {
  //initialize Sherpa but only once
  if (!isInitialized) {
    int argc = arguments.size();
    char *argv[argc];
    for (int l = 0; l < argc; l++)
      argv[l] = (char *)arguments[l].c_str();
#ifdef USING__MPI
    MPI::Init();
#endif
    Generator->InitializeTheRun(argc, argv);
    Generator->InitializeTheEventHandler();
    isInitialized = true;
  }
  return true;
}

bool SherpaHadronizer::declareStableParticles(const std::vector<int> &pdgIds) { return false; }

void SherpaHadronizer::statistics() {
  //calculate statistics
  Generator->SummarizeRun();

  //get the xsec & err
  double xsec_val = Generator->TotalXS();
  double xsec_err = Generator->TotalErr();

  //set the internal cross section in pb in GenRunInfoProduct
  runInfo().setInternalXSec(GenRunInfoProduct::XSec(xsec_val, xsec_err));
}

bool SherpaHadronizer::generatePartonsAndHadronize() {
  //get the next event and check if it produced
  bool rc = false;
  int itry = 0;
  bool gen_event = true;
  while ((itry < 3) && gen_event) {
    try {
      rc = Generator->GenerateOneEvent();
      gen_event = false;
    } catch (...) {
      ++itry;
      std::cerr << "Exception from Generator->GenerateOneEvent() catch. Call # " << itry << " for this event\n";
    }
  }
  if (rc) {
    //convert it to HepMC2
    auto evt = std::make_unique<HepMC::GenEvent>();
    Generator->FillHepMCEvent(*evt);

    // in case of unweighted events sherpa puts the max weight as event weight.
    // this is not optimal, we want 1 for unweighted events, so we check
    // whether we are producing unweighted events ("EVENT_GENERATION_MODE" == "1")
    // the information about the weights to the HepMC weight vector:
    // [0] event weight
    // [1] combined matrix element and phase space weight (missing only PDF information, thus directly suitable for PDF reweighting)
    // [2] event weight normalisation (in case of unweighted events event weights of ~ +/-1 can be obtained by (event weight)/(event weight normalisation))
    // [3] number of trials.
    // see also: https://sherpa.hepforge.org/doc/SHERPA-MC-2.1.0.html#Event-output-formats
    bool unweighted = false;
    double weight_normalization = -1;
    if (ATOOLS::ToType<int>(ATOOLS::rpa->gen.Variable("EVENT_GENERATION_MODE")) == 1) {
      if (evt->weights().size() > 2) {
        unweighted = true;
        weight_normalization = evt->weights()[2];
      } else {
        throw cms::Exception("SherpaInterface")
            << "Requested unweighted production. Missing normalization weight." << std::endl;
      }
    }

    // vector to fill new weights in correct order
    std::vector<double> newWeights;
    if (rearrangeWeights) {
      for (auto &i : weightlist) {
        if (evt->weights().has_key(i)) {
          newWeights.push_back(evt->weights()[i]);
        } else {
          throw cms::Exception("SherpaInterface")
              << "Missing weights! Key " << i << " not found, please check the weight definition!" << std::endl;
        }
      }
      for (auto &i : variationweightlist) {
        if (evt->weights().has_key(i)) {
          if (unweighted) {
            newWeights.push_back(evt->weights()[i] / weight_normalization);
          } else {
            newWeights.push_back(evt->weights()[i]);
          }
        } else {
          throw cms::Exception("SherpaInterface")
              << "Missing weights! Key " << i << " not found, please check the weight definition!" << std::endl;
        }
      }

      //Change original weights for reordered ones
      evt->weights().clear();
      for (auto &elem : newWeights) {
        evt->weights().push_back(elem);
      }
    }

    if (unweighted) {
      evt->weights()[0] /= weight_normalization;
    }
    resetEvent(std::move(evt));
    return true;
  } else {
    return false;
  }
}

bool SherpaHadronizer::decay() { return true; }

bool SherpaHadronizer::residualDecay() { return true; }

void SherpaHadronizer::finalizeEvent() {
  //******** Verbosity *******
  if (maxEventsToPrint > 0) {
    maxEventsToPrint--;
    event()->print();
  }
}

//GETTER for the external random numbers
DECLARE_GETTER(CMS_SHERPA_RNG, "CMS_SHERPA_RNG", ATOOLS::External_RNG, ATOOLS::RNG_Key);

ATOOLS::External_RNG *ATOOLS::Getter<ATOOLS::External_RNG, ATOOLS::RNG_Key, CMS_SHERPA_RNG>::operator()(
    const ATOOLS::RNG_Key &) const {
  return new CMS_SHERPA_RNG();
}

void ATOOLS::Getter<ATOOLS::External_RNG, ATOOLS::RNG_Key, CMS_SHERPA_RNG>::PrintInfo(std::ostream &str,
                                                                                      const size_t) const {
  str << "CMS_SHERPA_RNG interface";
}

double CMS_SHERPA_RNG::Get() {
  if (randomEngine == nullptr) {
    throw edm::Exception(edm::errors::LogicError) << "The Sherpa code attempted to a generate random number while\n"
                                                  << "the engine pointer was null. This might mean that the code\n"
                                                  << "was modified to generate a random number outside the event and\n"
                                                  << "beginLuminosityBlock methods, which is not allowed.\n";
  }
  return randomEngine->flat();
}
std::unique_ptr<GenLumiInfoHeader> SherpaHadronizer::getGenLumiInfoHeader() const {
  auto genLumiInfoHeader = BaseHadronizer::getGenLumiInfoHeader();

  if (rearrangeWeights) {
    edm::LogPrint("SherpaHadronizer") << "The order of event weights was changed!";
    for (auto &i : weightlist) {
      genLumiInfoHeader->weightNames().push_back(i);
      edm::LogVerbatim("SherpaHadronizer") << i;
    }
    for (auto &i : variationweightlist) {
      genLumiInfoHeader->weightNames().push_back(i);
      edm::LogVerbatim("SherpaHadronizer") << i;
    }
  }
  return genLumiInfoHeader;
}
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

typedef edm::GeneratorFilter<SherpaHadronizer, gen::ExternalDecayDriver> SherpaGeneratorFilter;
DEFINE_FWK_MODULE(SherpaGeneratorFilter);
