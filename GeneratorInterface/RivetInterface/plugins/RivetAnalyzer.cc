#include "GeneratorInterface/RivetInterface/interface/RivetAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Rivet/AnalysisHandler.hh"
#include "Rivet/Analysis.hh"

using namespace Rivet;
using namespace edm;

RivetAnalyzer::RivetAnalyzer(const edm::ParameterSet& pset) : 
_analysisHandler(),
_isFirstEvent(true),
_outFileName(pset.getParameter<std::string>("OutputFile")),
//decide whether to finlaize tthe plots or not.
//deciding not to finalize them can be useful for further harvesting of many jobs
_doFinalize(pset.getParameter<bool>("DoFinalize")),
_produceDQM(pset.getParameter<bool>("ProduceDQMOutput")),
_xsection(-1.)
{
  //retrive the analysis name from paarmeter set
  std::vector<std::string> analysisNames = pset.getParameter<std::vector<std::string> >("AnalysisNames");
  
  _hepmcCollection = consumes<HepMCProduct>(pset.getParameter<edm::InputTag>("HepMCCollection"));

  _useExternalWeight = pset.getParameter<bool>("UseExternalWeight");
  if (_useExternalWeight) {
    if (!pset.exists("GenEventInfoCollection")){
      throw cms::Exception("RivetAnalyzer") << "when using an external event weight you have to specify the GenEventInfoProduct collection from which the weight has to be taken " ; 
    }
    _LHECollection          = consumes<LHEEventProduct>(pset.getParameter<edm::InputTag>("LHECollection"));
    _useLHEweights          = pset.getParameter<bool>("useLHEweights");
    _LHEweightNumber        = pset.getParameter<int>("LHEweightNumber");    
    
    _genEventInfoCollection = consumes<GenEventInfoProduct>(pset.getParameter<edm::InputTag>("GenEventInfoCollection"));
    _LHECollection          = consumes<LHEEventProduct>(pset.getParameter<edm::InputTag>("LHECollection"));
    _useLHEweights          = pset.getParameter<bool>("useLHEweights");
    _LHEweightNumber        = pset.getParameter<int>("LHEweightNumber");    
    
  }

  //get the analyses
  _analysisHandler.addAnalyses(analysisNames);

  //go through the analyses and check those that need the cross section
  const std::set< AnaHandle, CmpAnaHandle > & analyses = _analysisHandler.analyses();

  std::set< AnaHandle, CmpAnaHandle >::const_iterator ibeg = analyses.begin();
  std::set< AnaHandle, CmpAnaHandle >::const_iterator iend = analyses.end();
  std::set< AnaHandle, CmpAnaHandle >::const_iterator iana; 
  _xsection = pset.getParameter<double>("CrossSection");
  for (iana = ibeg; iana != iend; ++iana){
    if ((*iana)->needsCrossSection())
      (*iana)->setCrossSection(_xsection);
  }
  if (_produceDQM){
    // book stuff needed for DQM
    dbe = 0;
    dbe = edm::Service<DQMStore>().operator->();
    dbe->setVerbose(50);
  }  

}

RivetAnalyzer::~RivetAnalyzer(){
}

void RivetAnalyzer::beginJob(){
  //set the environment, very ugly but rivet is monolithic when it comes to paths
  char * cmsswbase    = getenv("CMSSW_BASE");
  char * cmsswrelease = getenv("CMSSW_RELEASE_BASE");
  if ( !getenv("RIVET_REF_PATH") )
  {
    const std::string rivetref = "RIVET_REF_PATH=" + string(cmsswbase) + "/src/GeneratorInterface/RivetInterface/data:" + string(cmsswrelease) + "/src/GeneratorInterface/RivetInterface/data";
    putenv(strdup(rivetref.c_str()));
  }
  if ( !getenv("RIVET_INFO_PATH") )
  {
    const std::string rivetinfo = "RIVET_INFO_PATH=" + string(cmsswbase) + "/src/GeneratorInterface/RivetInterface/data:" + string(cmsswrelease) + "/src/GeneratorInterface/RivetInterface/data";
    putenv(strdup(rivetinfo.c_str()));
  }
}

void RivetAnalyzer::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  return;
}

void RivetAnalyzer::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup){
  
  //get the hepmc product from the event
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(_hepmcCollection, evt);

  // get HepMC GenEvent
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();
  std::unique_ptr<HepMC::GenEvent> tmpGenEvtPtr;
  //if you want to use an external weight or set the cross section we have to clone the GenEvent and change the weight  
  if ( _useExternalWeight || _xsection > 0 ){
    tmpGenEvtPtr = std::make_unique<HepMC::GenEvent>(*(evt->GetEvent()));

    if (_xsection > 0){
      HepMC::GenCrossSection xsec;
      xsec.set_cross_section(_xsection);
      tmpGenEvtPtr->set_cross_section(xsec);
    } 

    if ( _useExternalWeight ){
      if (tmpGenEvtPtr->weights().size() == 0) {
	throw cms::Exception("RivetAnalyzer") << "Original weight container has 0 size ";
      }
      if (tmpGenEvtPtr->weights().size() > 1) {
	edm::LogWarning("RivetAnalyzer") << "Original event weight size is " << tmpGenEvtPtr->weights().size() << ". Will change only the first one ";  
      }
    
      if(!_useLHEweights){
	edm::Handle<GenEventInfoProduct> genEventInfoProduct;
	iEvent.getByToken(_genEventInfoCollection, genEventInfoProduct);
	tmpGenEvtPtr->weights()[0] = genEventInfoProduct->weight();
      }else{
	edm::Handle<LHEEventProduct> lheEventHandle;
	iEvent.getByToken(_LHECollection,lheEventHandle);
	const LHEEventProduct::WGT& wgt = lheEventHandle->weights().at(_LHEweightNumber);
	tmpGenEvtPtr->weights()[0] = wgt.wgt;
      }
    }
    myGenEvent = tmpGenEvtPtr.get();

  }
  

  //aaply the beams initialization on the first event
  if (_isFirstEvent){
    _analysisHandler.init(*myGenEvent);
    _isFirstEvent = false;
  }

  //run the analysis
  _analysisHandler.analyze(*myGenEvent);

}


void RivetAnalyzer::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  if (_doFinalize)
  _analysisHandler.finalize();
  else {
    //if we don't finalize we just want to do the transformation from histograms to DPS
    ////normalizeTree(_analysisHandler.tree());
    //normalizeTree();

  }
  _analysisHandler.writeData(_outFileName);

  return;
}



//from Rivet 2.X: Analysis.hh (cls 18Feb2014)
/// List of registered analysis data objects
//const vector<AnalysisObjectPtr>& analysisObjects() const {
//return _analysisobjects;
//}



void RivetAnalyzer::endJob(){
}


void RivetAnalyzer::normalizeTree()    {
  using namespace YODA;
  std::vector<string> analyses = _analysisHandler.analysisNames();
  
  //tree.ls(".", true);
  const string tmpdir = "/RivetNormalizeTmp";
  //tree.mkdir(tmpdir);
  foreach (const string& analysis, analyses) {
    if (_produceDQM){
      dbe->setCurrentFolder(("Rivet/"+analysis).c_str());
      //global variables that are always present
      //sumOfWeights
      TH1F nevent("nEvt", "n analyzed Events", 1, 0., 1.);
      nevent.SetBinContent(1,_analysisHandler.sumOfWeights());
      _mes.push_back(dbe->book1D("nEvt",&nevent));
    }  
    //cross section
    //TH1F xsection("xSection", "Cross Section", 1, 0., 1.);
    //xsection.SetBinContent(1,_analysisHandler.crossSection());
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
            _analysisHandler.datapointsetFactory().create(path, *tmphisto);
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
            _analysisHandler.datapointsetFactory().create(path, *tmpprof);
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
