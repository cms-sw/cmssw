#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DQMOffline/RecoB/plugins/BTagPerformanceAnalyzerOnData.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "DataFormats/Common/interface/View.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace reco;
using namespace edm;
using namespace std;
using namespace RecoBTag;

BTagPerformanceAnalyzerOnData::BTagPerformanceAnalyzerOnData(const edm::ParameterSet& pSet) :
  partonKinematics(pSet.getParameter< bool >("partonKinematics")),
  jetSelector(
    pSet.getParameter<double>("etaMin"),
    pSet.getParameter<double>("etaMax"),
    pSet.getParameter<double>("ptRecJetMin"),
    pSet.getParameter<double>("ptRecJetMax"),
    0.0, 99999.0,
    pSet.getParameter<double>("ratioMin"),
    pSet.getParameter<double>("ratioMax")
  ),
  etaRanges(pSet.getParameter< vector<double> >("etaRanges")),
  ptRanges(pSet.getParameter< vector<double> >("ptRanges")),
  produceEps(pSet.getParameter< bool >("produceEps")),
  producePs(pSet.getParameter< bool >("producePs")),
  psBaseName(pSet.getParameter<std::string>( "psBaseName" )),
  epsBaseName(pSet.getParameter<std::string>( "epsBaseName" )),
  inputFile(pSet.getParameter<std::string>( "inputfile" )),
  update(pSet.getParameter<bool>( "update" )),
  allHisto(pSet.getParameter<bool>( "allHistograms" )),
  finalize(pSet.getParameter< bool >("finalizePlots")),
  finalizeOnly(pSet.getParameter< bool >("finalizeOnly")),
  slInfoTag(pSet.getParameter<edm::InputTag>("softLeptonInfo")),
  moduleConfig(pSet.getParameter< vector<edm::ParameterSet> >("tagConfig")),
  mcPlots_(pSet.getParameter< unsigned int >("mcPlots"))
{
  if(!finalizeOnly) mcPlots_ = 0; //analyzer not designed to produce flavour histograms but could be used for harvesting 
  bookHistos(pSet);
}

void BTagPerformanceAnalyzerOnData::bookHistos(const edm::ParameterSet& pSet)
{
  //
  // Book all histograms.
  //

  if (update) {
    //
        // append the DQM file ... we should consider this experimental
    //    edm::Service<DQMStore>().operator->()->open(std::string((const char *)(inputFile)),"/");
    // removed, a module will take care
  }

  // parton p
//   double pPartonMin = 0.0    ;
//   double pPartonMax = 99999.9 ;

  // iterate over ranges:
  const int iEtaStart = -1                   ;  // this will be the inactive one
  const int iEtaEnd   = etaRanges.size() - 1 ;
  const int iPtStart  = -1                   ;  // this will be the inactive one
  const int iPtEnd    = ptRanges.size() - 1  ;
  setTDRStyle();

  TagInfoPlotterFactory theFactory;

  for (vector<edm::ParameterSet>::const_iterator iModule = moduleConfig.begin(); 
       iModule != moduleConfig.end(); ++iModule) {

    const string& dataFormatType = iModule->exists("type") ? 
                                   iModule->getParameter<string>("type") :
                                   "JetTag";
    if (dataFormatType == "JetTag") {
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      const string& folderName    = iModule->getParameter<string>("folder");
      jetTagInputTags.push_back(moduleLabel);
      binJetTagPlotters.push_back(vector<JetTagPlotter*>()) ;
      // Contains plots for each bin of rapidity and pt.
	vector<BTagDifferentialPlot*> * differentialPlotsConstantEta = new vector<BTagDifferentialPlot*> () ;
	vector<BTagDifferentialPlot*> * differentialPlotsConstantPt  = new vector<BTagDifferentialPlot*> () ;
      if (finalize && mcPlots_==4){
	differentialPlots.push_back(vector<BTagDifferentialPlot*>());
	
	// the constant b-efficiency for the differential plots versus pt and eta
	const double& effBConst = 
	  iModule->getParameter<edm::ParameterSet>("parameters").getParameter<double>("effBConst");
	
	// the objects for the differential plots vs. eta,pt for
	
	for ( int iEta = iEtaStart ; iEta < iEtaEnd ; ++iEta ) {
	  BTagDifferentialPlot * etaConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constETA, moduleLabel.label());
	  differentialPlotsConstantEta->push_back ( etaConstDifferentialPlot );
	}
	
	for ( int iPt = iPtStart ; iPt < iPtEnd ; ++iPt ) {
	  // differentialPlots for this pt bin
	  BTagDifferentialPlot * ptConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constPT, moduleLabel.label());
	  differentialPlotsConstantPt->push_back ( ptConstDifferentialPlot );
	}
      }
      
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; ++iEta ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; ++iPt ) {
	  
	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
	  
	  // Instantiate the genertic b tag plotter
	  JetTagPlotter *jetTagPlotter = new JetTagPlotter(folderName, etaPtBin,
							   iModule->getParameter<edm::ParameterSet>("parameters"), mcPlots_, update, finalize);
	  binJetTagPlotters.back().push_back ( jetTagPlotter ) ;
	  
	  // Add to the corresponding differential plotters
	  if (finalize && mcPlots_==4){	
	    (*differentialPlotsConstantEta)[iEta+1]->addBinPlotter ( jetTagPlotter ) ;
	    (*differentialPlotsConstantPt )[iPt+1] ->addBinPlotter ( jetTagPlotter ) ;
	  }
	}
      }
      
      // the objects for the differential plots vs. eta, pt: collect all from constant eta and constant pt
      if (finalize && mcPlots_==4){
	differentialPlots.back().reserve(differentialPlotsConstantEta->size()+differentialPlotsConstantPt->size()) ;
	differentialPlots.back().insert(differentialPlots.back().end(), differentialPlotsConstantEta->begin(), differentialPlotsConstantEta->end());
	differentialPlots.back().insert(differentialPlots.back().end(), differentialPlotsConstantPt->begin(), differentialPlotsConstantPt->end());
	
	edm::LogInfo("Info")
	  << "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();
	
	// the intermediate ones are no longer needed
	delete differentialPlotsConstantEta ;
	delete differentialPlotsConstantPt  ;
      }
    } else if(dataFormatType == "TagCorrelation") { 
        const InputTag& label1 = iModule->getParameter<InputTag>("label1");
        const InputTag& label2 = iModule->getParameter<InputTag>("label2");
        tagCorrelationInputTags.push_back(std::pair<edm::InputTag, edm::InputTag>(label1, label2));
        binTagCorrelationPlotters.push_back(vector<TagCorrelationPlotter*>());

        // eta loop
        for ( int iEta = iEtaStart ; iEta != iEtaEnd ; ++iEta) {
          // pt loop
          for( int iPt = iPtStart ; iPt != iPtEnd ; ++iPt) {
            const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
            // Instantiate the generic b tag correlation plotter
            TagCorrelationPlotter* tagCorrelationPlotter = new TagCorrelationPlotter(label1.label(), label2.label(), etaPtBin,
                                                                                     iModule->getParameter<edm::ParameterSet>("parameters"),
                                                                                     mcPlots_, update);
            binTagCorrelationPlotters.back().push_back(tagCorrelationPlotter);
          }
        }
    } else {
      // tag info retrievel is deferred (needs availability of EventSetup)
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      const string& folderName    = iModule->getParameter<string>("folder");

      tagInfoInputTags.push_back(vector<edm::InputTag>());
      tiDataFormatType.push_back(dataFormatType);
      binTagInfoPlotters.push_back(vector<BaseTagInfoPlotter*>()) ;
      
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; ++iEta ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; ++iPt ) {
	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
	  
	  // Instantiate the tagInfo plotter
	  
	  BaseTagInfoPlotter *jetTagPlotter = theFactory.buildPlotter(dataFormatType, moduleLabel.label(),
		      etaPtBin, iModule->getParameter<edm::ParameterSet>("parameters"), folderName,
                      update, mcPlots_, finalize);
	  binTagInfoPlotters.back().push_back ( jetTagPlotter ) ;
          binTagInfoPlottersToModuleConfig.insert(make_pair(jetTagPlotter, iModule - moduleConfig.begin()));
	}
      }
      
      edm::LogInfo("Info")
	<< "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();
    }
  }
  
  
}

EtaPtBin BTagPerformanceAnalyzerOnData::getEtaPtBin(const int& iEta, const int& iPt)
{
  // DEFINE BTagBin:
  bool    etaActive_ , ptActive_;
  double  etaMin_, etaMax_, ptMin_, ptMax_ ;
  
  if ( iEta != -1 ) {
    etaActive_ = true ;
    etaMin_    = etaRanges[iEta]   ;
    etaMax_    = etaRanges[iEta+1] ;
  }
  else {
    etaActive_ = false ;
    etaMin_    = etaRanges[0]   ;
    etaMax_    = etaRanges[etaRanges.size() - 1]   ;
  }

  if ( iPt != -1 ) {
    ptActive_ = true ;
    ptMin_    = ptRanges[iPt]   ;
    ptMax_    = ptRanges[iPt+1] ;
  }
  else {
    ptActive_ = false ;
    ptMin_    = ptRanges[0]	;
    ptMax_    = ptRanges[ptRanges.size() - 1]	;
  }
  return EtaPtBin(etaActive_ , etaMin_ , etaMax_ ,
			ptActive_  , ptMin_  , ptMax_ );
}

BTagPerformanceAnalyzerOnData::~BTagPerformanceAnalyzerOnData()
{
  for (unsigned int iJetLabel = 0; iJetLabel != binJetTagPlotters.size(); ++iJetLabel) {
    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      delete binJetTagPlotters[iJetLabel][iPlotter];
    }
    if (finalize && mcPlots_==4){
      for (vector<BTagDifferentialPlot *>::iterator iPlotter = differentialPlots[iJetLabel].begin();
	   iPlotter != differentialPlots[iJetLabel].end(); ++ iPlotter) {
	delete *iPlotter;
      }
    }
  }

  for (vector<vector<TagCorrelationPlotter*> >::iterator iJetLabel = binTagCorrelationPlotters.begin();
       iJetLabel != binTagCorrelationPlotters.end(); ++iJetLabel) 
    for(vector<TagCorrelationPlotter*>::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter)
      delete *iPlotter;

  for (unsigned int iJetLabel = 0; iJetLabel != binTagInfoPlotters.size(); ++iJetLabel) {
    int plotterSize =  binTagInfoPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      delete binTagInfoPlotters[iJetLabel][iPlotter];
    }
  }
}

void BTagPerformanceAnalyzerOnData::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if (finalizeOnly) return;
  //
  //no flavour map needed here

  edm::Handle<reco::SoftLeptonTagInfoCollection> infoHandle;
  iEvent.getByLabel(slInfoTag, infoHandle);

// Look first at the jetTags

  for (unsigned int iJetLabel = 0; iJetLabel != jetTagInputTags.size(); ++iJetLabel) {
    edm::Handle<reco::JetTagCollection> tagHandle;
    iEvent.getByLabel(jetTagInputTags[iJetLabel], tagHandle);
    //
    // insert check on the presence of the collections
    //

    if (!tagHandle.isValid()){
      edm::LogWarning("BTagPerformanceAnalyzerOnData")<<" Collection "<<jetTagInputTags[iJetLabel]<<" not present. Skipping it for this event.";
      continue;
    }
    
    const reco::JetTagCollection & tagColl = *(tagHandle.product());
    LogDebug("Info") << "Found " << tagColl.size() << " B candidates in collection " << jetTagInputTags[iJetLabel];

    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (JetTagCollection::const_iterator tagI = tagColl.begin();
	 tagI != tagColl.end(); ++tagI) {
      
      if (!jetSelector(*(tagI->first), -1, infoHandle)) continue;
      for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
	bool inBin = binJetTagPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(*tagI->first);
	// Fill histograms if in desired pt/rapidity bin.
	if (inBin)
	  binJetTagPlotters[iJetLabel][iPlotter]->analyzeTag(*tagI, -1);
      }
    }
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binJetTagPlotters[iJetLabel][iPlotter]->analyzeTag();
    }
  }

// Now look at Tag Correlations
  for (unsigned int iJetLabel = 0; iJetLabel != tagCorrelationInputTags.size(); ++iJetLabel) {
    const std::pair<edm::InputTag, edm::InputTag>& inputTags = tagCorrelationInputTags[iJetLabel];
    edm::Handle<reco::JetTagCollection> tagHandle1;
    iEvent.getByLabel(inputTags.first, tagHandle1);
    const reco::JetTagCollection& tagColl1 = *(tagHandle1.product());

    edm::Handle<reco::JetTagCollection> tagHandle2;
    iEvent.getByLabel(inputTags.second, tagHandle2);
    const reco::JetTagCollection& tagColl2 = *(tagHandle2.product());

    int plotterSize = binTagCorrelationPlotters[iJetLabel].size();
    for (JetTagCollection::const_iterator tagI = tagColl1.begin(); tagI != tagColl1.end(); ++tagI) {
      if (!jetSelector(*(tagI->first), -1, infoHandle))
        continue;

      for(int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
        bool inBin = binTagCorrelationPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(*(tagI->first));

        if(inBin)
        {
          double discr2 = tagColl2[tagI->first];
          binTagCorrelationPlotters[iJetLabel][iPlotter]->analyzeTags(tagI->second, discr2, -1);
        }
      }
    }
  }
// Now look at the TagInfos

  for (unsigned int iJetLabel = 0; iJetLabel != tiDataFormatType.size(); ++iJetLabel) {
    int plotterSize = binTagInfoPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter)
      binTagInfoPlotters[iJetLabel][iPlotter]->setEventSetup(iSetup);

    vector<edm::InputTag> & inputTags = tagInfoInputTags[iJetLabel];
    if (inputTags.empty()) {
      // deferred retrieval of input tags
      BaseTagInfoPlotter *firstPlotter = binTagInfoPlotters[iJetLabel][0];
      int iModule = binTagInfoPlottersToModuleConfig[firstPlotter];
      vector<string> labels = firstPlotter->tagInfoRequirements();
      if (labels.empty())
        labels.push_back("label");
      for (vector<string>::const_iterator iLabels = labels.begin();
           iLabels != labels.end(); ++iLabels) {
        edm::InputTag inputTag =
        	moduleConfig[iModule].getParameter<InputTag>(*iLabels);
        inputTags.push_back(inputTag);
      }
    }

    unsigned int nInputTags = inputTags.size();
    vector< edm::Handle< View<BaseTagInfo> > > tagInfoHandles(nInputTags);
    edm::ProductID jetProductID;
    unsigned int nTagInfos = 0;
    for (unsigned int iInputTags = 0; iInputTags < inputTags.size(); ++iInputTags) {
      edm::Handle< View<BaseTagInfo> > & tagInfoHandle = tagInfoHandles[iInputTags];
      iEvent.getByLabel(inputTags[iInputTags], tagInfoHandle);
      //
      // protect against missing products
      //
    if (tagInfoHandle.isValid() == false){
      edm::LogWarning("BTagPerformanceAnalyzerOnData")<<" Collection "<<inputTags[iInputTags]<<" not present. Skipping it for this event.";
      continue;
    }


      unsigned int size = tagInfoHandle->size();
      LogDebug("Info") << "Found " << size << " B candidates in collection " << inputTags[iInputTags];
      edm::ProductID thisProductID = (size > 0) ? (*tagInfoHandle)[0].jet().id() : edm::ProductID();
      if (iInputTags == 0) {
        jetProductID = thisProductID;
        nTagInfos = size;
      } else if (jetProductID != thisProductID)
        throw cms::Exception("Configuration") << "TagInfos are referencing a different jet collection." << endl;
      else if (nTagInfos != size)
        throw cms::Exception("Configuration") << "TagInfo collections are having a different size." << endl;
    }

    for (unsigned int iTagInfos = 0; iTagInfos < nTagInfos; ++iTagInfos) {
      vector<const BaseTagInfo*> baseTagInfos(nInputTags);
      edm::RefToBase<Jet> jetRef;
      for (unsigned int iTagInfo = 0; iTagInfo < nInputTags; iTagInfo++) {
        const BaseTagInfo &baseTagInfo = (*tagInfoHandles[iTagInfo])[iTagInfos];
        if (iTagInfo == 0)
          jetRef = baseTagInfo.jet();
        else if (baseTagInfo.jet() != jetRef)
          throw cms::Exception("Configuration") << "TagInfos pointing to different jets." << endl;
        baseTagInfos[iTagInfo] = &baseTagInfo;
      }

      if (!jetSelector(*jetRef, -1, infoHandle))
        continue;

      for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
	bool inBin = binTagInfoPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(*jetRef);
	// Fill histograms if in desired pt/rapidity bin.
	if (inBin)
	  binTagInfoPlotters[iJetLabel][iPlotter]->analyzeTag(baseTagInfos, -1);
      }
    }
  }
}

void BTagPerformanceAnalyzerOnData::endRun(const edm::Run & run, const edm::EventSetup & es){

  if (finalize == false) return;
  setTDRStyle();
  for (unsigned int iJetLabel = 0; iJetLabel != binJetTagPlotters.size(); ++iJetLabel) {
    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binJetTagPlotters[iJetLabel][iPlotter]->finalize();
      //      binJetTagPlotters[iJetLabel][iPlotter]->write(allHisto);
      if (producePs)  (*binJetTagPlotters[iJetLabel][iPlotter]).psPlot(psBaseName);
      if (produceEps) (*binJetTagPlotters[iJetLabel][iPlotter]).epsPlot(epsBaseName);
    }
    
    if (mcPlots_==4){
      for (vector<BTagDifferentialPlot *>::iterator iPlotter = differentialPlots[iJetLabel].begin();
	   iPlotter != differentialPlots[iJetLabel].end(); ++ iPlotter) {
	(**iPlotter).process();
	if (producePs)  (**iPlotter).psPlot(psBaseName);
	if (produceEps) (**iPlotter).epsPlot(epsBaseName);
	//      (**iPlotter).write(allHisto);
      }
    }
  }
  for (unsigned int iJetLabel = 0; iJetLabel != binTagInfoPlotters.size(); ++iJetLabel) {
    int plotterSize =  binTagInfoPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
  binTagInfoPlotters[iJetLabel][iPlotter]->finalize();

      //      binTagInfoPlotters[iJetLabel][iPlotter]->write(allHisto);
      if (producePs)  (*binTagInfoPlotters[iJetLabel][iPlotter]).psPlot(psBaseName);
      if (produceEps) (*binTagInfoPlotters[iJetLabel][iPlotter]).epsPlot(epsBaseName);
    }
  }
  
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformanceAnalyzerOnData);
