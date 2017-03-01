#ifndef BTagPerformanceHarvester_H
#define BTagPerformanceHarvester_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/TagCorrelationPlotter.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DQMOffline/RecoB/interface/BTagDifferentialPlot.h"

/** \class BTagPerformanceHarvester
 *
 *  Top level steering routine for b tag performance harvesting.
 *
 */

class BTagPerformanceHarvester : public DQMEDHarvester {
   public:
      explicit BTagPerformanceHarvester(const edm::ParameterSet& pSet);
      ~BTagPerformanceHarvester();

   private:
      void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
      EtaPtBin getEtaPtBin(const int& iEta, const int& iPt);

      // Get histogram plotting options from configuration.
      std::vector<double> etaRanges, ptRanges;
      bool produceEps, producePs;
      std::string psBaseName, epsBaseName;
      std::vector<std::string> tiDataFormatType;

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
      
      std::string flavPlots_;
      unsigned int mcPlots_;
      bool makeDiffPlots_;
};

#endif
