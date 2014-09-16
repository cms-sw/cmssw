#include "DQMOffline/RecoB/plugins/BTagPerformanceAnalyzerOnData.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

using namespace reco;
using namespace edm;
using namespace std;
using namespace RecoBTag;

BTagPerformanceAnalyzerOnData::BTagPerformanceAnalyzerOnData(const edm::ParameterSet& pSet) :
  jetSelector(
    pSet.getParameter<double>("etaMin"),
    pSet.getParameter<double>("etaMax"),
    pSet.getParameter<double>("ptRecJetMin"),
    pSet.getParameter<double>("ptRecJetMax"),
    0.0, 99999.0,
    pSet.getParameter<double>("ratioMin"),
    pSet.getParameter<double>("ratioMax"),
    pSet.getParameter<bool>( "doJetID" )
  ),
  etaRanges(pSet.getParameter< vector<double> >("etaRanges")),
  ptRanges(pSet.getParameter< vector<double> >("ptRanges")),
  JECsource(pSet.getParameter<std::string>( "JECsource" )),
  doJEC(pSet.getParameter<bool>( "doJEC" )),
  moduleConfig(pSet.getParameter< vector<edm::ParameterSet> >("tagConfig"))
{
  genToken = mayConsume<GenEventInfoProduct>(edm::InputTag("generator"));
  slInfoToken = consumes<SoftLeptonTagInfoCollection>(pSet.getParameter<InputTag>("softLeptonInfo"));
  for (vector<edm::ParameterSet>::const_iterator iModule = moduleConfig.begin();
       iModule != moduleConfig.end(); ++iModule) {

    const string& dataFormatType = iModule->exists("type") ?
      iModule->getParameter<string>("type") :
      "JetTag";
    if (dataFormatType == "JetTag") {
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      jetTagInputTags.push_back(moduleLabel);
      binJetTagPlotters.push_back(vector<JetTagPlotter*>()) ;
      jetTagToken.push_back(consumes<JetTagCollection>(moduleLabel));
    }
    else if(dataFormatType == "TagCorrelation") {
      const InputTag& label1 = iModule->getParameter<InputTag>("label1");
      const InputTag& label2 = iModule->getParameter<InputTag>("label2");
      tagCorrelationInputTags.push_back(std::pair<edm::InputTag, edm::InputTag>(label1, label2));
      binTagCorrelationPlotters.push_back(vector<TagCorrelationPlotter*>());
      tagCorrelationToken.push_back(std::pair< edm::EDGetTokenT<reco::JetTagCollection>, edm::EDGetTokenT<reco::JetTagCollection> >(consumes<JetTagCollection>(label1), consumes<JetTagCollection>(label2)));
    }
    else {
      tagInfoInputTags.push_back(vector<edm::InputTag>());
      tiDataFormatType.push_back(dataFormatType);
      binTagInfoPlotters.push_back(vector<BaseTagInfoPlotter*>()) ;
      std::vector< edm::EDGetTokenT<edm::View<reco::BaseTagInfo>> > tokens;
      if(dataFormatType == "GenericMVA") {
        const InputTag& ipinfo = iModule->getParameter<InputTag>("ipTagInfos");
        const InputTag& svinfo = iModule->getParameter<InputTag>("svTagInfos");
        tokens.push_back(consumes< View<BaseTagInfo> >(ipinfo));
        tokens.push_back(consumes< View<BaseTagInfo> >(svinfo));
      }
      else {
        const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
        tokens.push_back(consumes< View<BaseTagInfo> >(moduleLabel));
      }
      tagInfoToken.push_back(tokens);
    }
  }
}

void BTagPerformanceAnalyzerOnData::bookHistograms(DQMStore::IBooker & ibook, edm::Run const & run, edm::EventSetup const & es)
{
  //
  // Book all histograms.
  //

  // iterate over ranges:
  const int iEtaStart = -1                   ;  // this will be the inactive one
  const int iEtaEnd   = etaRanges.size() - 1 ;
  const int iPtStart  = -1                   ;  // this will be the inactive one
  const int iPtEnd    = ptRanges.size() - 1  ;
  setTDRStyle();

  TagInfoPlotterFactory theFactory;
  int iTag = -1; int iTagCorr = -1; int iInfoTag = -1;
  for (vector<edm::ParameterSet>::const_iterator iModule = moduleConfig.begin(); 
       iModule != moduleConfig.end(); ++iModule) {

    const string& dataFormatType = iModule->exists("type") ? 
                                   iModule->getParameter<string>("type") :
                                   "JetTag";
    if (dataFormatType == "JetTag") {
      iTag++;
      const string& folderName    = iModule->getParameter<string>("folder");
      
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; ++iEta ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; ++iPt ) {
	  
	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
	  
	  // Instantiate the genertic b tag plotter
	  JetTagPlotter *jetTagPlotter = new JetTagPlotter(folderName, etaPtBin,
							   iModule->getParameter<edm::ParameterSet>("parameters"), 0, false, ibook);
	  binJetTagPlotters.at(iTag).push_back ( jetTagPlotter ) ;
	  
	}
      }
      
    } else if(dataFormatType == "TagCorrelation") { 
        iTagCorr++;
        const InputTag& label1 = iModule->getParameter<InputTag>("label1");
        const InputTag& label2 = iModule->getParameter<InputTag>("label2");

        // eta loop
        for ( int iEta = iEtaStart ; iEta != iEtaEnd ; ++iEta) {
          // pt loop
          for( int iPt = iPtStart ; iPt != iPtEnd ; ++iPt) {
            const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
            // Instantiate the generic b tag correlation plotter
            TagCorrelationPlotter* tagCorrelationPlotter = new TagCorrelationPlotter(label1.label(), label2.label(), etaPtBin,
                                                                                     iModule->getParameter<edm::ParameterSet>("parameters"),
                                                                                     0, ibook);
            binTagCorrelationPlotters.at(iTagCorr).push_back(tagCorrelationPlotter);
          }
        }
    } else {
      iInfoTag++;
      // tag info retrievel is deferred (needs availability of EventSetup)
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      const string& folderName    = iModule->getParameter<string>("folder");
      
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; ++iEta ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; ++iPt ) {
	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
	  
	  // Instantiate the tagInfo plotter
	  
	  BaseTagInfoPlotter *jetTagPlotter = theFactory.buildPlotter(dataFormatType, moduleLabel.label(),
								      etaPtBin, iModule->getParameter<edm::ParameterSet>("parameters"), folderName,
								      0, false, ibook);
	  binTagInfoPlotters.at(iInfoTag).push_back ( jetTagPlotter ) ;
          binTagInfoPlottersToModuleConfig.insert(make_pair(jetTagPlotter, iModule - moduleConfig.begin()));
	}
      }
      
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

  edm::Handle<reco::SoftLeptonTagInfoCollection> infoHandle;
  iEvent.getByToken(slInfoToken, infoHandle);

  //Get JEC
  const JetCorrector* corrector = 0;
  if(doJEC) {
    edm::Handle<GenEventInfoProduct> genInfoHandle; //check if data or MC                                                                                                                                 
    iEvent.getByToken(genToken, genInfoHandle);
    std::string allJECsource = JECsource;
    if( !genInfoHandle.isValid() ) allJECsource += "Residual";
    corrector = JetCorrector::getJetCorrector (allJECsource,iSetup);   //Get the jet corrector from the event setup
  }
  //

// Look first at the jetTags

  for (unsigned int iJetLabel = 0; iJetLabel != jetTagInputTags.size(); ++iJetLabel) {
    edm::Handle<reco::JetTagCollection> tagHandle;
    iEvent.getByToken(jetTagToken[iJetLabel], tagHandle);
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
      
      //JEC
      reco::Jet correctedJet = *(tagI->first);     
      double jec = 1.0;
      if(doJEC && corrector) {
	jec = corrector->correction(*(tagI->first),iEvent,iSetup);
      }
      //

      if (!jetSelector(*(tagI->first), -1, infoHandle, jec)) 
	continue;
      
      for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
	bool inBin = binJetTagPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(*tagI->first, jec);
	// Fill histograms if in desired pt/rapidity bin.
	if (inBin)
	  binJetTagPlotters[iJetLabel][iPlotter]->analyzeTag(*tagI, jec, -1);
      }
    }
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binJetTagPlotters[iJetLabel][iPlotter]->analyzeTag();
    }
  }

// Now look at Tag Correlations
  for (unsigned int iJetLabel = 0; iJetLabel != tagCorrelationInputTags.size(); ++iJetLabel) {
    const std::pair< edm::EDGetTokenT<reco::JetTagCollection>, edm::EDGetTokenT<reco::JetTagCollection> >& inputTokens = tagCorrelationToken[iJetLabel];
    edm::Handle<reco::JetTagCollection> tagHandle1;
    iEvent.getByToken(inputTokens.first, tagHandle1);
    const reco::JetTagCollection& tagColl1 = *(tagHandle1.product());

    edm::Handle<reco::JetTagCollection> tagHandle2;
    iEvent.getByToken(inputTokens.second, tagHandle2);
    const reco::JetTagCollection& tagColl2 = *(tagHandle2.product());

    int plotterSize = binTagCorrelationPlotters[iJetLabel].size();
    for (JetTagCollection::const_iterator tagI = tagColl1.begin(); tagI != tagColl1.end(); ++tagI) {
      //JEC
      reco::Jet correctedJet = *(tagI->first);     
      double jec = 1.0;
      if(doJEC && corrector) {
	jec = corrector->correction(*(tagI->first),iEvent,iSetup);
      }
      //

      if (!jetSelector(*(tagI->first), -1, infoHandle, jec))
        continue;
      
      for(int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
        bool inBin = binTagCorrelationPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(*(tagI->first), jec);

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
    vector<edm::EDGetTokenT<edm::View<reco::BaseTagInfo>> > & tokens = tagInfoToken[iJetLabel];
    if(nInputTags!=tokens.size()) throw cms::Exception("Configuration") << "Different number of Tag Infos than expected" << endl;
    for (unsigned int iInputTags = 0; iInputTags < tokens.size(); ++iInputTags) {
      edm::Handle< View<BaseTagInfo> > & tagInfoHandle = tagInfoHandles[iInputTags];
      iEvent.getByToken(tokens[iInputTags], tagInfoHandle); 
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

      //JEC
      reco::Jet correctedJet = *(jetRef);     
      double jec = 1.0;
      if(doJEC && corrector) {
	jec = corrector->correction(*(jetRef),iEvent,iSetup);
      }
      //

      if (!jetSelector(*jetRef, -1, infoHandle, jec))
        continue;
      
      for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
	bool inBin = binTagInfoPlotters[iJetLabel][iPlotter]->etaPtBin().inBin(*jetRef, jec);
	// Fill histograms if in desired pt/rapidity bin.
	if (inBin)
	  binTagInfoPlotters[iJetLabel][iPlotter]->analyzeTag(baseTagInfos, jec, -1);
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformanceAnalyzerOnData);
