#include "DQMOffline/RecoB/plugins/BTagPerformanceHarvester.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace edm;
using namespace std;
using namespace RecoBTag;

BTagPerformanceHarvester::BTagPerformanceHarvester(const edm::ParameterSet& pSet):
  etaRanges(pSet.getParameter< vector<double> >("etaRanges")),
  ptRanges(pSet.getParameter< vector<double> >("ptRanges")),
  produceEps(pSet.getParameter< bool >("produceEps")),
  producePs(pSet.getParameter< bool >("producePs")),
  moduleConfig(pSet.getParameter< vector<edm::ParameterSet> >("tagConfig")),
  flavPlots_(pSet.getParameter< std::string >("flavPlots")),
  makeDiffPlots_(pSet.getParameter< bool >("differentialPlots"))
{
  //mcPlots_ : 1=b+c+l+ni; 2=all+1; 3=1+d+u+s+g; 4=3+all . Default is 2. Don't use 0.
  if (flavPlots_.find("dusg")<15) {
    if (flavPlots_.find("all")<15)
        mcPlots_ = 4;
    else
        mcPlots_ = 3;
  } else if (flavPlots_.find("bcl")<15) {
    if (flavPlots_.find("all")<15)
        mcPlots_ = 2;
    else
        mcPlots_ = 1;
  } else
      mcPlots_ = 0;

  if (etaRanges.size() <= 1)
      etaRanges = { pSet.getParameter<double>("etaMin"), pSet.getParameter<double>("etaMax") };
  if (ptRanges.size() <= 1)
      ptRanges = { pSet.getParameter<double>("ptRecJetMin"), pSet.getParameter<double>("ptRecJetMax") };
  
  for (vector<edm::ParameterSet>::const_iterator iModule = moduleConfig.begin();
       iModule != moduleConfig.end(); ++iModule) {
    const string& dataFormatType = iModule->exists("type") ?
      iModule->getParameter<string>("type") :
      "JetTag";
    if (dataFormatType == "JetTag") {
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      jetTagInputTags.push_back(moduleLabel);
      binJetTagPlotters.push_back(vector<std::shared_ptr<JetTagPlotter>>());
      if (mcPlots_ && makeDiffPlots_) {
        differentialPlots.push_back(vector<std::unique_ptr<BTagDifferentialPlot>>());
      }
    } else if (dataFormatType == "TagCorrelation") {
      const InputTag& label1 = iModule->getParameter<InputTag>("label1");
      const InputTag& label2 = iModule->getParameter<InputTag>("label2");
      tagCorrelationInputTags.push_back(std::pair<edm::InputTag, edm::InputTag>(label1, label2));
      binTagCorrelationPlotters.push_back(vector<std::unique_ptr<TagCorrelationPlotter>>());
    } else {
      tagInfoInputTags.push_back(vector<edm::InputTag>());
      tiDataFormatType.push_back(dataFormatType);
      binTagInfoPlotters.push_back(vector<std::unique_ptr<BaseTagInfoPlotter>>());
    }
  }

}

EtaPtBin BTagPerformanceHarvester::getEtaPtBin(const int& iEta, const int& iPt)
{
  // DEFINE BTagBin:
  bool    etaActive_ , ptActive_;
  double  etaMin_, etaMax_, ptMin_, ptMax_;

  if (iEta != -1) {
    etaActive_ = true;
    etaMin_    = etaRanges[iEta];
    etaMax_    = etaRanges[iEta+1];
  }
  else {
    etaActive_ = false;
    etaMin_    = etaRanges[0];
    etaMax_    = etaRanges[etaRanges.size() - 1];
  }

  if (iPt != -1) {
    ptActive_ = true;
    ptMin_    = ptRanges[iPt];
    ptMax_    = ptRanges[iPt+1];
  }
  else {
    ptActive_ = false;
    ptMin_    = ptRanges[0];
    ptMax_    = ptRanges[ptRanges.size() - 1];
  }
  return EtaPtBin(etaActive_, etaMin_, etaMax_, ptActive_, ptMin_, ptMax_);
}

BTagPerformanceHarvester::~BTagPerformanceHarvester() {}

void BTagPerformanceHarvester::dqmEndJob(DQMStore::IBooker & ibook, DQMStore::IGetter & iget)
{
  // Book all histograms.

  // iterate over ranges:
  const int iEtaStart = -1                  ;  // this will be the inactive one
  const int iEtaEnd   = etaRanges.size() > 2 ? etaRanges.size() - 1 : 0; // if there is only one bin defined, leave it as the inactive one
  const int iPtStart  = -1                  ;  // this will be the inactive one
  const int iPtEnd    = ptRanges.size() > 2 ? ptRanges.size() - 1 : 0; // if there is only one bin defined, leave it as the inactive one
  setTDRStyle();

  TagInfoPlotterFactory theFactory;
  int iTag = -1; int iTagCorr = -1; int iInfoTag = -1;
  for (vector<edm::ParameterSet>::const_iterator iModule = moduleConfig.begin();
       iModule != moduleConfig.end(); ++iModule) {

    const string& dataFormatType = iModule->exists("type") ?
                                   iModule->getParameter<string>("type") :
                                   "JetTag";
    const bool& doCTagPlots = iModule->exists("doCTagPlots") ?
                                   iModule->getParameter<bool>("doCTagPlots") :
                                   false;
             
    if (dataFormatType == "JetTag") {
      iTag++;
      const string& folderName = iModule->getParameter<string>("folder");

      // Contains plots for each bin of rapidity and pt.
      auto differentialPlotsConstantEta = std::make_unique< std::vector<std::unique_ptr<BTagDifferentialPlot>> >();
      auto differentialPlotsConstantPt  = std::make_unique< std::vector<std::unique_ptr<BTagDifferentialPlot>> >();
      if (mcPlots_ && makeDiffPlots_) {
        // the constant b-efficiency for the differential plots versus pt and eta
        const double& effBConst =
                      iModule->getParameter<edm::ParameterSet>("parameters").getParameter<double>("effBConst");

        // the objects for the differential plots vs. eta,pt for
        for (int iEta = iEtaStart; iEta < iEtaEnd; iEta++) {
            std::unique_ptr<BTagDifferentialPlot> etaConstDifferentialPlot = std::make_unique<BTagDifferentialPlot>(effBConst, BTagDifferentialPlot::constETA, folderName, mcPlots_);
          differentialPlotsConstantEta->push_back(std::move(etaConstDifferentialPlot));
        }

        for (int iPt = iPtStart; iPt < iPtEnd; iPt++) {
          // differentialPlots for this pt bin
          std::unique_ptr<BTagDifferentialPlot> ptConstDifferentialPlot = std::make_unique<BTagDifferentialPlot>(effBConst, BTagDifferentialPlot::constPT, folderName, mcPlots_);
          differentialPlotsConstantPt->push_back(std::move(ptConstDifferentialPlot));
        }
      }
      // eta loop
      for (int iEta = iEtaStart; iEta < iEtaEnd; iEta++) {
        // pt loop
        for (int iPt = iPtStart; iPt < iPtEnd; iPt++) {

          const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);

          // Instantiate the generic b tag plotter
          bool doDifferentialPlots = iModule->exists("differentialPlots") && iModule->getParameter<bool>("differentialPlots") == true;
          std::shared_ptr<JetTagPlotter> jetTagPlotter = std::make_shared<JetTagPlotter>(folderName, etaPtBin,
                                   iModule->getParameter<edm::ParameterSet>("parameters"),mcPlots_,true, ibook, doCTagPlots, doDifferentialPlots);
          binJetTagPlotters.at(iTag).push_back(jetTagPlotter);

          // Add to the corresponding differential plotters
          if (mcPlots_ && makeDiffPlots_) {
            (*differentialPlotsConstantEta)[iEta+1]->addBinPlotter(jetTagPlotter);
            (*differentialPlotsConstantPt)[iPt+1] ->addBinPlotter(jetTagPlotter);
          }
        }
      }
      // the objects for the differential plots vs. eta, pt: collect all from constant eta and constant pt
      if (mcPlots_ && makeDiffPlots_) {
        differentialPlots.at(iTag).reserve(differentialPlotsConstantEta->size() + differentialPlotsConstantPt->size());
        differentialPlots.at(iTag).insert(differentialPlots.at(iTag).end(), std::make_move_iterator(differentialPlotsConstantEta->begin()), std::make_move_iterator(differentialPlotsConstantEta->end()));
        differentialPlots.at(iTag).insert(differentialPlots.at(iTag).end(), std::make_move_iterator(differentialPlotsConstantPt->begin()), std::make_move_iterator(differentialPlotsConstantPt->end()));

        edm::LogInfo("Info")
          << "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();
      }
    } else if (dataFormatType == "TagCorrelation") {
        iTagCorr++;
        const InputTag& label1 = iModule->getParameter<InputTag>("label1");
        const InputTag& label2 = iModule->getParameter<InputTag>("label2");

        // eta loop
        for (int iEta = iEtaStart; iEta != iEtaEnd; ++iEta) {
          // pt loop
          for (int iPt = iPtStart; iPt != iPtEnd; ++iPt) {
            const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);
            // Instantiate the generic b tag correlation plotter
            std::unique_ptr<TagCorrelationPlotter> tagCorrelationPlotter = std::make_unique<TagCorrelationPlotter>(label1.label(), label2.label(), etaPtBin,
                                                                                     iModule->getParameter<edm::ParameterSet>("parameters"),
                                                                                     mcPlots_, doCTagPlots, true, ibook);
            binTagCorrelationPlotters.at(iTagCorr).push_back(std::move(tagCorrelationPlotter));
          }
        }
    } else {
      iInfoTag++;
      // tag info retrievel is deferred(needs availability of EventSetup)
      const InputTag& moduleLabel = iModule->getParameter<InputTag>("label");
      const string& folderName    = iModule->getParameter<string>("folder");
      // eta loop
      for (int iEta = iEtaStart; iEta < iEtaEnd; iEta++) {
        // pt loop
        for (int iPt = iPtStart; iPt < iPtEnd; iPt++) {
            const EtaPtBin& etaPtBin = getEtaPtBin(iEta, iPt);

            // Instantiate the tagInfo plotter

            std::unique_ptr<BaseTagInfoPlotter> jetTagPlotter = theFactory.buildPlotter(dataFormatType, moduleLabel.label(), 
                                          etaPtBin, iModule->getParameter<edm::ParameterSet>("parameters"), folderName, 
                                          mcPlots_, true, ibook);
            binTagInfoPlotters.at(iInfoTag).push_back(std::move(jetTagPlotter));
        }
      }
      edm::LogInfo("Info")
    << "====>>>> ## sizeof differentialPlots = " << differentialPlots.size();
    }
  }

  setTDRStyle();
  for (unsigned int iJetLabel = 0; iJetLabel != binJetTagPlotters.size(); ++iJetLabel) {
    int plotterSize =  binJetTagPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binJetTagPlotters[iJetLabel][iPlotter]->finalize(ibook, iget);
      if (producePs) (*binJetTagPlotters[iJetLabel][iPlotter]).psPlot(psBaseName);
      if (produceEps)(*binJetTagPlotters[iJetLabel][iPlotter]).epsPlot(epsBaseName);
    }
   
    if (makeDiffPlots_) {
        for (auto& iPlotter: differentialPlots[iJetLabel]) {
          iPlotter->process(ibook);
          if (producePs)  iPlotter->psPlot(psBaseName);
          if (produceEps) iPlotter->epsPlot(epsBaseName);
        }
    }
  }
  
  for (auto& iJetLabel: binTagInfoPlotters) {
    for (auto& iPlotter: iJetLabel) {
      iPlotter->finalize(ibook, iget);
      if (producePs)  iPlotter->psPlot(psBaseName);
      if (produceEps) iPlotter->epsPlot(epsBaseName);
    }
  }
  for (unsigned int iJetLabel = 0; iJetLabel != binTagCorrelationPlotters.size(); ++iJetLabel) {
    int plotterSize =  binTagCorrelationPlotters[iJetLabel].size();
    for (int iPlotter = 0; iPlotter != plotterSize; ++iPlotter) {
      binTagCorrelationPlotters[iJetLabel][iPlotter]->finalize(ibook, iget);
      if (producePs) (*binTagCorrelationPlotters[iJetLabel][iPlotter]).psPlot(psBaseName);
      if (produceEps)(*binTagCorrelationPlotters[iJetLabel][iPlotter]).epsPlot(epsBaseName);
    }
  } 

}


//define this as a plug-in
DEFINE_FWK_MODULE(BTagPerformanceHarvester);
