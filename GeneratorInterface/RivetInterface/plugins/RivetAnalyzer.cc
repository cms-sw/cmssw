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
      _produceDQM(pset.getParameter<bool>("ProduceDQMOutput")),
      _lheLabel(pset.getParameter<edm::InputTag>("LHECollection")),
      _xsection(-1.) {
  usesResource("Rivet");

  _hepmcCollection = consumes<HepMCProduct>(pset.getParameter<edm::InputTag>("HepMCCollection"));
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

  if (_produceDQM) {
    // book stuff needed for DQM
    dbe = nullptr;
    dbe = edm::Service<DQMStore>().operator->();
  }
}

RivetAnalyzer::~RivetAnalyzer() {}

void RivetAnalyzer::beginJob() {
  //set the environment, very ugly but rivet is monolithic when it comes to paths
  char* cmsswbase = std::getenv("CMSSW_BASE");
  char* cmsswrelease = std::getenv("CMSSW_RELEASE_BASE");
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
      if (_lheWeightNames.size() == 0) {
        edm::Handle<LHEEventProduct> lheEventHandle;
        iEvent.getByToken(_LHECollection, lheEventHandle);
        for (unsigned int i = 0; i < lheEventHandle->weights().size(); i++) {
          _lheWeightNames.push_back("lhe"+std::to_string(i+1)); // start with 1 to match LHE weight IDs
        }
      }
      _weightNames.insert(_weightNames.end(), _lheWeightNames.begin(), _lheWeightNames.end());
    }
    // clean weight names to be accepted by Rivet plotting
    for (const std::string& wn : _weightNames) {
      _cleanedWeightNames.push_back(std::regex_replace(wn, std::regex("[^A-Za-z\\d\\._=]"), "_"));
    }
  }

  //get the hepmc product from the event
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(_hepmcCollection, evt);

  // get HepMC GenEvent
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();
  std::unique_ptr<HepMC::GenEvent> tmpGenEvtPtr;
  //if you want to use an external weight or set the cross section we have to clone the GenEvent and change the weight
  tmpGenEvtPtr = std::make_unique<HepMC::GenEvent>(*(evt->GetEvent()));

  if (_xsection > 0) {
    HepMC::GenCrossSection xsec;
    xsec.set_cross_section(_xsection);
    tmpGenEvtPtr->set_cross_section(xsec);
  }

  std::vector<double> mergedWeights;
  for (unsigned int i = 0; i < tmpGenEvtPtr->weights().size(); i++) {
    mergedWeights.push_back(tmpGenEvtPtr->weights()[i]);
  }

  if (_useLHEweights) {
    edm::Handle<LHEEventProduct> lheEventHandle;
    iEvent.getByToken(_LHECollection, lheEventHandle);
    for (unsigned int i = 0; i < _lheWeightNames.size(); i++) {
      mergedWeights.push_back(tmpGenEvtPtr->weights()[0] * lheEventHandle->weights().at(i).wgt /
                              lheEventHandle->originalXWGTUP());
    }
  }

  tmpGenEvtPtr->weights().clear();
  for (unsigned int i = 0; i < _cleanedWeightNames.size(); i++) {
    tmpGenEvtPtr->weights()[_cleanedWeightNames[i]] = mergedWeights[i];
  }
  myGenEvent = tmpGenEvtPtr.get();

  //apply the beams initialization on the first event
  if (_isFirstEvent) {
    _analysisHandler = std::make_unique<Rivet::AnalysisHandler>();
    _analysisHandler->addAnalyses(_analysisNames);

    /// Set analysis handler weight options
    _analysisHandler->setIgnoreBeams(_setIgnoreBeams);
    _analysisHandler->skipMultiWeights(_skipMultiWeights);
    _analysisHandler->selectMultiWeights(_selectMultiWeights);
    _analysisHandler->deselectMultiWeights(_deselectMultiWeights);
    _analysisHandler->setNominalWeightName(_setNominalWeightName);
    _analysisHandler->setWeightCap(_weightCap);
    _analysisHandler->setNLOSmearing(_NLOSmearing);

    _analysisHandler->init(*myGenEvent);

    _isFirstEvent = false;
  }

  //run the analysis
  _analysisHandler->analyze(*myGenEvent);
}

void RivetAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (_doFinalize)
    _analysisHandler->finalize();
  else {
    //if we don't finalize we just want to do the transformation from histograms to DPS
    ////normalizeTree(_analysisHandler->tree());
    //normalizeTree();
  }
  _analysisHandler->writeData(_outFileName);

  return;
}

//from Rivet 2.X: Analysis.hh (cls 18Feb2014)
/// List of registered analysis data objects
//const vector<AnalysisObjectPtr>& analysisObjects() const {
//return _analysisobjects;
//}

void RivetAnalyzer::endJob() {}

void RivetAnalyzer::normalizeTree() {
  using namespace YODA;
  std::vector<string> analyses = _analysisHandler->analysisNames();

  //tree.ls(".", true);
  const string tmpdir = "/RivetNormalizeTmp";
  //tree.mkdir(tmpdir);
  for (const string& analysis : analyses) {
    if (_produceDQM) {
      dbe->setCurrentFolder("Rivet/" + analysis);
      //global variables that are always present
      //sumOfWeights
      TH1F nevent("nEvt", "n analyzed Events", 1, 0., 1.);
      nevent.SetBinContent(1, _analysisHandler->sumW());
      _mes.push_back(dbe->book1D("nEvt", &nevent));
    }
    //cross section
    //TH1F xsection("xSection", "Cross Section", 1, 0., 1.);
    //xsection.SetBinContent(1,_analysisHandler->crossSection());
    //_mes.push_back(dbe->book1D("xSection",&xsection));
    //now loop over the histograms

    /*
    const vector<string> paths = tree.listObjectNames("/"+analysis, true); // args set recursive listing
    std::cout << "Number of objects in YODA tree for analysis " << analysis << " = " << paths.size() << std::endl;
    foreach (const string& path, paths) {
      IManagedObject* hobj = tree.find(path);
      if (hobj) {
        // Weird seg fault on SLC4 when trying to dyn cast an IProfile ptr to a IHistogram
        // Fix by attempting to cast to IProfile first, only try IHistogram if it fails.
        IHistogram1D* histo = 0;
        IProfile1D* prof = dynamic_cast<IProfile1D*>(hobj);
        if (!prof) histo = dynamic_cast<IHistogram1D*>(hobj);

        std::cout << "Converting histo " << path << " to DPS" << std::endl;
        tree.mv(path, tmpdir);
        const size_t lastslash = path.find_last_of("/");
        const string basename = path.substr(lastslash+1, path.length() - (lastslash+1));
        const string tmppath = tmpdir + "/" + basename;

        // If it's a normal histo:
        if (histo) {
          IHistogram1D* tmphisto = dynamic_cast<IHistogram1D*>(tree.find(tmppath));
          if (tmphisto) {
            _analysisHandler->datapointsetFactory().create(path, *tmphisto);
          }
          //now convert to root and then ME
    //need aida2flat (from Rivet 1.X) & flat2root here
          TH1F* h = aida2root<IHistogram1D, TH1F>(histo, basename);
          if (_produceDQM)
            _mes.push_back(dbe->book1D(h->GetName(), h));
          delete h;
          tree.rm(tmppath);
        }
        // If it's a profile histo:
        else if (prof) {
          IProfile1D* tmpprof = dynamic_cast<IProfile1D*>(tree.find(tmppath));
          if (tmpprof) {
            _analysisHandler->datapointsetFactory().create(path, *tmpprof);
          }
          //now convert to root and then ME
    //need aida2flat (from Rivet 1.X) & flat2root here
          TProfile* p = aida2root<IProfile1D, TProfile>(prof, basename);
          if (_produceDQM)
            _mes.push_back(dbe->bookProfile(p->GetName(), p));
          delete p;
          tree.rm(tmppath);
        }
      }
    }
    */
  }
  //tree.rmdir(tmpdir);
}

DEFINE_FWK_MODULE(RivetAnalyzer);
