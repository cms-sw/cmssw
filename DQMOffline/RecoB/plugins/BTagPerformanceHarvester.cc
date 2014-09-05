#include "DQMOffline/RecoB/plugins/BTagPerformanceHarvester.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace std;
using namespace RecoBTag;

BTagPerformanceHarvester::BTagPerformanceHarvester(const edm::ParameterSet& pSet) :
  etaRanges(pSet.getParameter< vector<double> >("etaRanges")),
  ptRanges(pSet.getParameter< vector<double> >("ptRanges")),
  produceEps(pSet.getParameter< bool >("produceEps")),
  producePs(pSet.getParameter< bool >("producePs")),
  moduleConfig(pSet.getParameter< vector<edm::ParameterSet> >("tagConfig")),
  flavPlots_(pSet.getParameter< std::string >("flavPlots")),
  makeDiffPlots_(pSet.getParameter< bool >("differentialPlots"))
{
  //mcPlots_ : 1=b+c+l+ni; 2=all+1; 3=1+d+u+s+g; 4=3+all . Default is 2. Don't use 0.
  if(flavPlots_.find("dusg")<15){
    if(flavPlots_.find("all")<15) mcPlots_ = 4;
    else mcPlots_ = 3;
  }
  else if(flavPlots_.find("bcl")<15){
    if(flavPlots_.find("all")<15) mcPlots_ = 2;
    else mcPlots_ = 1;
  }
  else mcPlots_ = 0;

  for (vector<edm::ParameterSet>::const_iterator iModule = moduleConfig.begin();
       iModule != moduleConfig.end(); ++iModule) {
    const string& dataFormatType = iModule->exists("type") ?
      iModule->getParameter<string>("type") :
      "JetTag";
    if (dataFormatType == "JetTag") {
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      jetTagInputTags.push_back(moduleLabel);
      binJetTagPlotters.push_back(vector<JetTagPlotter*>()) ;
      if (mcPlots_ && makeDiffPlots_){
        differentialPlots.push_back(vector<BTagDifferentialPlot*>());
      }
    }
    else if(dataFormatType == "TagCorrelation") {
      const InputTag& label1 = iModule->getParameter<InputTag>("label1");
      const InputTag& label2 = iModule->getParameter<InputTag>("label2");
      tagCorrelationInputTags.push_back(std::pair<edm::InputTag, edm::InputTag>(label1, label2));
      binTagCorrelationPlotters.push_back(vector<TagCorrelationPlotter*>());
    }
    else {
      tagInfoInputTags.push_back(vector<edm::InputTag>());
      tiDataFormatType.push_back(dataFormatType);
      binTagInfoPlotters.push_back(vector<BaseTagInfoPlotter*>()) ;
    }
  }

}

EtaPtBin BTagPerformanceHarvester::getEtaPtBin(const int& iEta, const int& iPt)
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

BTagPerformanceHarvester::~BTagPerformanceHarvester()
{
  for (vector<vector<JetTagPlotter*> >::iterator iJetLabel = binJetTagPlotters.begin();
       iJetLabel != binJetTagPlotters.end(); ++iJetLabel) 
    for (vector<JetTagPlotter*>::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) 
      delete *iPlotter;

  if (mcPlots_ && makeDiffPlots_) {
    for(vector<vector<BTagDifferentialPlot*> >::iterator iJetLabel = differentialPlots.begin();
        iJetLabel != differentialPlots.end(); ++iJetLabel)
      for (vector<BTagDifferentialPlot *>::iterator iPlotter = iJetLabel->begin();
           iPlotter != iJetLabel->end(); ++ iPlotter) 
	delete *iPlotter;
  }

  for (vector<vector<TagCorrelationPlotter*> >::iterator iJetLabel = binTagCorrelationPlotters.begin(); 
       iJetLabel != binTagCorrelationPlotters.end(); ++iJetLabel) 
    for (vector<TagCorrelationPlotter* >::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) 
      delete *iPlotter;
    
  
  for (vector<vector<BaseTagInfoPlotter*> >::iterator iJetLabel = binTagInfoPlotters.begin(); 
       iJetLabel != binTagInfoPlotters.end(); ++iJetLabel) 
    for (vector<BaseTagInfoPlotter*>::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) 
      delete *iPlotter;
    
}

void BTagPerformanceHarvester::dqmEndJob(DQMStore::IBooker & ibook, DQMStore::IGetter & iget)
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

      // Contains plots for each bin of rapidity and pt.
      vector<BTagDifferentialPlot*> * differentialPlotsConstantEta = new vector<BTagDifferentialPlot*> () ;
      vector<BTagDifferentialPlot*> * differentialPlotsConstantPt  = new vector<BTagDifferentialPlot*> () ;
      if (mcPlots_ && makeDiffPlots_){
	// the constant b-efficiency for the differential plots versus pt and eta
	const double& effBConst =
	  			iModule->getParameter<edm::ParameterSet>("parameters").getParameter<double>("effBConst");

	// the objects for the differential plots vs. eta,pt for
	for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	  BTagDifferentialPlot * etaConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constETA, folderName, mcPlots_);
	  differentialPlotsConstantEta->push_back ( etaConstDifferentialPlot );
	}

	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {
	  // differentialPlots for this pt bin
	  BTagDifferentialPlot * ptConstDifferentialPlot = new BTagDifferentialPlot
	    (effBConst, BTagDifferentialPlot::constPT, folderName, mcPlots_);
	  differentialPlotsConstantPt->push_back ( ptConstDifferentialPlot );
	}
      }
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {

	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);

	  // Instantiate the genertic b tag plotter
	  JetTagPlotter *jetTagPlotter = new JetTagPlotter(folderName, etaPtBin,
							   iModule->getParameter<edm::ParameterSet>("parameters"),mcPlots_,true, ibook);
	  binJetTagPlotters.at(iTag).push_back ( jetTagPlotter ) ;

	  // Add to the corresponding differential plotters
	  if (mcPlots_ && makeDiffPlots_){
	    (*differentialPlotsConstantEta)[iEta+1]->addBinPlotter ( jetTagPlotter ) ;
	    (*differentialPlotsConstantPt )[iPt+1] ->addBinPlotter ( jetTagPlotter ) ;
	  }
	}
      }
      // the objects for the differential plots vs. eta, pt: collect all from constant eta and constant pt
      if (mcPlots_ && makeDiffPlots_){
	differentialPlots.at(iTag).reserve(differentialPlotsConstantEta->size()+differentialPlotsConstantPt->size()) ;
	differentialPlots.at(iTag).insert(differentialPlots.at(iTag).end(), differentialPlotsConstantEta->begin(), differentialPlotsConstantEta->end());
	differentialPlots.at(iTag).insert(differentialPlots.at(iTag).end(), differentialPlotsConstantPt->begin(), differentialPlotsConstantPt->end());

	edm::LogInfo("Info")
	  << "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();

	// the intermediate ones are no longer needed
	delete differentialPlotsConstantEta ;
	delete differentialPlotsConstantPt  ;
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
                                                                                     mcPlots_,  ibook);
            binTagCorrelationPlotters.at(iTagCorr).push_back(tagCorrelationPlotter);
          }
        }
    } else {
      iInfoTag++;
      // tag info retrievel is deferred (needs availability of EventSetup)
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      const string& folderName    = iModule->getParameter<string>("folder");
      // eta loop
      for ( int iEta = iEtaStart ; iEta < iEtaEnd ; iEta++ ) {
	// pt loop
	for ( int iPt = iPtStart ; iPt < iPtEnd ; iPt++ ) {
	  const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);

	  // Instantiate the tagInfo plotter

	  BaseTagInfoPlotter *jetTagPlotter = theFactory.buildPlotter(dataFormatType, moduleLabel.label(), 
								      etaPtBin, iModule->getParameter<edm::ParameterSet>("parameters"), folderName, 
								      mcPlots_, true, ibook);
	  binTagInfoPlotters.at(iInfoTag).push_back ( jetTagPlotter ) ;
          binTagInfoPlottersToModuleConfig.insert(make_pair(jetTagPlotter, iModule - moduleConfig.begin()));
	}
      }
      edm::LogInfo("Info")
	<< "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();
    }
  }

  ///////

  setTDRStyle();
  for (unsigned int iJetLabel = 0; iJetLabel != binJetTagPlotters.size(); ++iJetLabel) {
    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binJetTagPlotters[iJetLabel][iPlotter]->finalize(ibook, iget);
      if (producePs)  (*binJetTagPlotters[iJetLabel][iPlotter]).psPlot(psBaseName);
      if (produceEps) (*binJetTagPlotters[iJetLabel][iPlotter]).epsPlot(epsBaseName);
    }
   
      if(makeDiffPlots_) { 
        for (vector<BTagDifferentialPlot *>::iterator iPlotter = differentialPlots[iJetLabel].begin();
	     iPlotter != differentialPlots[iJetLabel].end(); ++ iPlotter) {
	  (*iPlotter)->process(ibook);
	  if (producePs)  (*iPlotter)->psPlot(psBaseName);
	  if (produceEps) (*iPlotter)->epsPlot(epsBaseName);
        }
      }
  }
  for (vector<vector<BaseTagInfoPlotter*> >::iterator iJetLabel = binTagInfoPlotters.begin();
       iJetLabel != binTagInfoPlotters.end(); ++iJetLabel) {
    for (vector<BaseTagInfoPlotter*>::iterator iPlotter = iJetLabel->begin(); iPlotter != iJetLabel->end(); ++iPlotter) {
      (*iPlotter)->finalize(ibook, iget);
      if (producePs)  (*iPlotter)->psPlot(psBaseName);
      if (produceEps) (*iPlotter)->epsPlot(epsBaseName);
    }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformanceHarvester);
