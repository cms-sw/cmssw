#ifndef BTagPerformanceAnalyzerOnData_H
#define BTagPerformanceAnalyzerOnData_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DQMOffline/RecoB/interface/AcceptJet.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/TagCorrelationPlotter.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

/** \class BTagPerformanceAnalyzerOnData
 *
 *  Top level steering routine for b tag performance analysis.
 *
 */

class BTagPerformanceAnalyzerOnData : public DQMEDAnalyzer {
   public:
      explicit BTagPerformanceAnalyzerOnData(const edm::ParameterSet& pSet);

      ~BTagPerformanceAnalyzerOnData();

      virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);


   private:

  struct JetRefCompare :
       public std::binary_function<edm::RefToBase<reco::Jet>, edm::RefToBase<reco::Jet>, bool> {
    inline bool operator () (const edm::RefToBase<reco::Jet> &j1,
                             const edm::RefToBase<reco::Jet> &j2) const
    { return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key()); }
  };

  // Get histogram plotting options from configuration.
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  EtaPtBin getEtaPtBin(const int& iEta, const int& iPt);

  std::vector<std::string> tiDataFormatType;
  AcceptJet jetSelector;   // Decides if jet and parton satisfy kinematic cuts.
  std::vector<double> etaRanges, ptRanges;
  bool doJEC;
  edm::InputTag slInfoTag;

  std::vector< std::vector<JetTagPlotter*> > binJetTagPlotters;
  std::vector< std::vector<TagCorrelationPlotter*> > binTagCorrelationPlotters;
  std::vector< std::vector<BaseTagInfoPlotter*> > binTagInfoPlotters;
  std::vector<edm::InputTag> jetTagInputTags;
  std::vector< std::pair<edm::InputTag, edm::InputTag> > tagCorrelationInputTags;
  std::vector< std::vector<edm::InputTag> > tagInfoInputTags;
  // Contains plots for each bin of rapidity and pt.
  std::vector<edm::ParameterSet> moduleConfig;
  std::map<BaseTagInfoPlotter*, size_t> binTagInfoPlottersToModuleConfig;

  //add consumes
  edm::EDGetTokenT<reco::JetCorrector> jecMCToken;
  edm::EDGetTokenT<reco::JetCorrector> jecDataToken;
  edm::EDGetTokenT<GenEventInfoProduct> genToken;
  edm::EDGetTokenT<reco::SoftLeptonTagInfoCollection> slInfoToken;
  std::vector< edm::EDGetTokenT<reco::JetTagCollection> > jetTagToken;
  std::vector< std::pair<edm::EDGetTokenT<reco::JetTagCollection>, edm::EDGetTokenT<reco::JetTagCollection>> > tagCorrelationToken;
  std::vector<std::vector <edm::EDGetTokenT<edm::View<reco::BaseTagInfo>> >> tagInfoToken;

};

#endif
