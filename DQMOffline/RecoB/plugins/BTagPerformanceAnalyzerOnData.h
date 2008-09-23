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
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

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

      void endLuminosityBlock(const edm::LuminosityBlock & lumiBlock, const edm::EventSetup & es);

   private:

  struct JetRefCompare :
       public std::binary_function<edm::RefToBase<reco::Jet>, edm::RefToBase<reco::Jet>, bool> {
    inline bool operator () (const edm::RefToBase<reco::Jet> &j1,
                             const edm::RefToBase<reco::Jet> &j2) const
    { return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key()); }
  };

  // Get histogram plotting options from configuration.
  void init(const edm::ParameterSet& iConfig);
  void bookHistos(const edm::ParameterSet& pSet);
  EtaPtBin getEtaPtBin(int iEta, int iPt);

  vector<std::string> tiDataFormatType;
  bool partonKinematics;
  double etaMin, etaMax;
  double ptRecJetMin, ptRecJetMax;
  AcceptJet jetSelector;   // Decides if jet and parton satisfy kinematic cuts.
  std::vector<double> etaRanges, ptRanges;
  bool produceEps, producePs;
  TString psBaseName, epsBaseName, inputFile;
  bool update, allHisto;
  bool finalize;
  bool finalizeOnly;

  vector< vector<JetTagPlotter*> > binJetTagPlotters;
  vector< vector<BaseTagInfoPlotter*> > binTagInfoPlotters;
  vector<edm::InputTag> jetTagInputTags;
  vector< vector<edm::InputTag> > tagInfoInputTags;
  // Contains plots for each bin of rapidity and pt.
  vector< vector<BTagDifferentialPlot*> > differentialPlots;
  vector<edm::ParameterSet> moduleConfig;
  map<BaseTagInfoPlotter*, size_t> binTagInfoPlottersToModuleConfig;

  bool mcPlots_;

};


#endif
