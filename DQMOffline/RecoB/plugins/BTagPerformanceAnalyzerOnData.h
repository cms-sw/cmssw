#ifndef BTagPerformanceAnalyzerOnData_H
#define BTagPerformanceAnalyzerOnData_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQMOffline/RecoB/interface/BTagDifferentialPlot.h"
#include "DQMOffline/RecoB/interface/AcceptJet.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/TagCorrelationPlotter.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <map>

//class CaloJetRef;

/** \class BTagPerformanceAnalyzerOnData
 *
 *  Top level steering routine for b tag performance analysis.
 *
 */

class BTagPerformanceAnalyzerOnData : public edm::EDAnalyzer {
   public:
      explicit BTagPerformanceAnalyzerOnData(const edm::ParameterSet& pSet);

      ~BTagPerformanceAnalyzerOnData();

      virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

      void endRun(const edm::Run & run, const edm::EventSetup & es);

   private:

  struct JetRefCompare :
       public std::binary_function<edm::RefToBase<reco::Jet>, edm::RefToBase<reco::Jet>, bool> {
    inline bool operator () (const edm::RefToBase<reco::Jet> &j1,
                             const edm::RefToBase<reco::Jet> &j2) const
    { return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key()); }
  };

  // Get histogram plotting options from configuration.
  void bookHistos(const edm::ParameterSet& pSet);
  EtaPtBin getEtaPtBin(const int& iEta, const int& iPt);

  std::vector<std::string> tiDataFormatType;
  bool partonKinematics;
  AcceptJet jetSelector;   // Decides if jet and parton satisfy kinematic cuts.
  std::vector<double> etaRanges, ptRanges;
  bool produceEps, producePs;
  std::string psBaseName, epsBaseName, inputFile;
  bool update, allHisto;
  bool finalize;
  bool finalizeOnly;
  edm::InputTag slInfoTag;

  std::vector< std::vector<JetTagPlotter*> > binJetTagPlotters;
  std::vector< std::vector<TagCorrelationPlotter*> > binTagCorrelationPlotters;
  std::vector< std::vector<BaseTagInfoPlotter*> > binTagInfoPlotters;
  std::vector<edm::InputTag> jetTagInputTags;
  std::vector< std::pair<edm::InputTag, edm::InputTag> > tagCorrelationInputTags;
  std::vector< std::vector<edm::InputTag> > tagInfoInputTags;
  // Contains plots for each bin of rapidity and pt.
  std::vector< std::vector<BTagDifferentialPlot*> > differentialPlots;
  std::vector<edm::ParameterSet> moduleConfig;
  std::map<BaseTagInfoPlotter*, size_t> binTagInfoPlottersToModuleConfig;

  unsigned int mcPlots_;

};


#endif
