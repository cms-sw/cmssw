#include "GeneratorInterface/RivetInterface/interface/RivetAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Rivet/Run.hh"
#include "Rivet/AnalysisHandler.hh"
#include "Rivet/Analysis.hh"

#include <regex>

using namespace Rivet;
using namespace edm;

RivetAnalyzer::RivetAnalyzer(const edm::ParameterSet& pset)
    : _isFirstEvent(true),
      _outFileName(pset.getParameter<std::string>("OutputFile")),
      _analysisNames(pset.getParameter<std::vector<std::string> >("AnalysisNames")),
      //decide whether to finalize the plots or not.
      //deciding not to finalize them can be useful for further harvesting of many jobs
      _doFinalize(pset.getParameter<bool>("DoFinalize")),
      _lheLabel(pset.getParameter<edm::InputTag>("LHECollection")),
      _xsection(-1.) {
  usesResource("Rivet");

  _hepmcCollection = consumes<HepMC3Product>(pset.getParameter<edm::InputTag>("HepMCCollection"));
  _genLumiInfoToken = consumes<GenLumiInfoHeader, edm::InLumi>(pset.getParameter<edm::InputTag>("genLumiInfo"));

  _useLHEweights = pset.getParameter<bool>("useLHEweights");
  if (_useLHEweights) {
    _lheRunInfoToken = consumes<LHERunInfoProduct, edm::InRun>(_lheLabel);
    _LHECollection = consumes<LHEEventProduct>(_lheLabel);
  }

  _weightCap = pset.getParameter<double>("weightCap");
  _NLOSmearing = pset.getParameter<double>("NLOSmearing");
  _setIgnoreBeams = pset.getParameter<bool>("setIgnoreBeams");
  _skipMultiWeights = pset.getParameter<bool>("skipMultiWeights");
  _selectMultiWeights = pset.getParameter<std::string>("selectMultiWeights");
  _deselectMultiWeights = pset.getParameter<std::string>("deselectMultiWeights");
  _setNominalWeightName = pset.getParameter<std::string>("setNominalWeightName");

  //set user cross section if needed
  _xsection = pset.getParameter<double>("CrossSection");
}

RivetAnalyzer::~RivetAnalyzer() {}

void RivetAnalyzer::beginJob() {
  //set the environment, very ugly but rivet is monolithic when it comes to paths
  char* cmsswbase = std::getenv("CMSSW_BASE");
  char* cmsswrelease = std::getenv("CMSSW_RELEASE_BASE");
  // These should never fail
  assert(cmsswbase);
  assert(cmsswrelease);
  if (!std::getenv("RIVET_REF_PATH")) {
    const std::string rivetref = string(cmsswbase) +
                                 "/src/GeneratorInterface/RivetInterface/data:" + string(cmsswrelease) +
                                 "/src/GeneratorInterface/RivetInterface/data:.";
    char* rivetrefCstr = strdup(rivetref.c_str());
    setenv("RIVET_REF_PATH", rivetrefCstr, 1);
    free(rivetrefCstr);
  }
  if (!std::getenv("RIVET_INFO_PATH")) {
    const std::string rivetinfo = string(cmsswbase) +
                                  "/src/GeneratorInterface/RivetInterface/data:" + string(cmsswrelease) +
                                  "/src/GeneratorInterface/RivetInterface/data:.";
    char* rivetinfoCstr = strdup(rivetinfo.c_str());
    setenv("RIVET_INFO_PATH", rivetinfoCstr, 1);
    free(rivetinfoCstr);
  }
}

void RivetAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (_useLHEweights) {
    edm::Handle<LHERunInfoProduct> lheRunInfoHandle;
    iRun.getByLabel(_lheLabel, lheRunInfoHandle);
    typedef std::vector<LHERunInfoProduct::Header>::const_iterator headers_const_iterator;

    std::regex reg("<weight.*> ?(.*?) ?<\\/weight>");
    for (headers_const_iterator iter = lheRunInfoHandle->headers_begin(); iter != lheRunInfoHandle->headers_end();
         iter++) {
      std::vector<std::string> lines = iter->lines();
      for (unsigned int iLine = 0; iLine < lines.size(); iLine++) {
        std::smatch match;
        std::regex_search(lines.at(iLine), match, reg);
        if (!match.empty()) {
          _lheWeightNames.push_back(match[1]);
        }
      }
    }
  }
}

void RivetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //finalize weight names on the first event
  if (_isFirstEvent) {
    auto genLumiInfoHandle = iEvent.getLuminosityBlock().getHandle(_genLumiInfoToken);
    if (genLumiInfoHandle.isValid()) {
      _weightNames = genLumiInfoHandle->weightNames();
    }

    // need to reset the default weight name (or plotting will fail)
    if (!_weightNames.empty()) {
      _weightNames[0] = "";
    } else {  // Summer16 samples have 1 weight stored in HepMC but no weightNames
      _weightNames.push_back("");
    }
    if (_useLHEweights) {
      // Some samples have weights but no weight names -> assign generic names lheN
      if (_lheWeightNames.empty()) {
        edm::Handle<LHEEventProduct> lheEventHandle;
        iEvent.getByToken(_LHECollection, lheEventHandle);
        for (unsigned int i = 0; i < lheEventHandle->weights().size(); i++) {
          _lheWeightNames.push_back("lhe" + std::to_string(i + 1));  // start with 1 to match LHE weight IDs
        }
      }
      _weightNames.insert(_weightNames.end(), _lheWeightNames.begin(), _lheWeightNames.end());
    }
    // clean weight names to be accepted by Rivet plotting
    for (const std::string& wn : _weightNames) {
      _cleanedWeightNames.push_back(std::regex_replace(wn, std::regex("[^A-Za-z\\d\\._=]"), "_"));
    }
    runinfo = make_shared<HepMC3::GenRunInfo>();
    runinfo->set_weight_names(_cleanedWeightNames);
  }

  //get the hepmc product from the event
  edm::Handle<HepMC3Product> evt;
  iEvent.getByToken(_hepmcCollection, evt);

  // get HepMC GenEvent
  const HepMC3::GenEventData* genEventData = evt->GetEvent();
  std::unique_ptr<HepMC3::GenEvent> genEvent = std::make_unique<HepMC3::GenEvent>();
  genEvent->read_data(*genEventData);

  std::vector<double> mergedWeights;
  for (unsigned int i = 0; i < genEvent->weights().size(); i++) {
    mergedWeights.push_back(genEvent->weights()[i]);
  }

  if (_useLHEweights) {
    edm::Handle<LHEEventProduct> lheEventHandle;
    iEvent.getByToken(_LHECollection, lheEventHandle);
    for (unsigned int i = 0; i < _lheWeightNames.size(); i++) {
      mergedWeights.push_back(genEvent->weights()[0] * lheEventHandle->weights().at(i).wgt /
                              lheEventHandle->originalXWGTUP());
    }
  }

  double xsection = _xsection > 0 ? _xsection : genEvent->cross_section()->xsecs()[0];
  HepMC3::GenCrossSectionPtr xsec = make_shared<HepMC3::GenCrossSection>();
  xsec->set_cross_section(std::vector<double>(mergedWeights.size(), xsection),
                          std::vector<double>(mergedWeights.size(), 0.));
  genEvent->set_cross_section(xsec);
  genEvent->set_run_info(runinfo);
  genEvent->weights() = mergedWeights;

  //apply the beams initialization on the first event
  if (_isFirstEvent) {
    _analysisHandler = std::make_unique<Rivet::AnalysisHandler>();
    _analysisHandler->addAnalyses(_analysisNames);

    /// Set analysis handler weight options
    _analysisHandler->setCheckBeams(!_setIgnoreBeams);
    _analysisHandler->skipMultiWeights(_skipMultiWeights);
    _analysisHandler->matchWeightNames(_selectMultiWeights);
    _analysisHandler->unmatchWeightNames(_deselectMultiWeights);
    _analysisHandler->setNominalWeightName(_setNominalWeightName);
    _analysisHandler->setWeightCap(_weightCap);
    _analysisHandler->setNLOSmearing(_NLOSmearing);

    _analysisHandler->init(*genEvent);

    _isFirstEvent = false;
  }

  //run the analysis
  _analysisHandler->analyze(const_cast<GenEvent&>(*genEvent));
}

void RivetAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (_doFinalize)
    _analysisHandler->finalize();
  _analysisHandler->writeData(_outFileName);

  return;
}

void RivetAnalyzer::endJob() {}

DEFINE_FWK_MODULE(RivetAnalyzer);
