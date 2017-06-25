#ifndef TaggingVariablePlotter_H
#define TaggingVariablePlotter_H

#include <string>
#include <vector>

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TaggingVariablePlotter: public BaseTagInfoPlotter {

 public:

  TaggingVariablePlotter(const std::string & tagName, const EtaPtBin & etaPtBin,
              const edm::ParameterSet& pSet,
              unsigned int mc, bool willFinalize, DQMStore::IBooker & ibook,
              const std::string &category = std::string());

  ~TaggingVariablePlotter ();

  void analyzeTag(const reco::BaseTagInfo * baseTagInfo, double jec, int jetFlavour, float w=1);
  void analyzeTag(const reco::TaggingVariableList & variables, int jetFlavour, float w=1);

  virtual void finalize(DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_) {}

  void epsPlot(const std::string & name) {}

  void psPlot(const std::string & name) {}

 private:

  unsigned int mcPlots_;

  struct VariableConfig {
    VariableConfig(const std::string &name, const edm::ParameterSet& pSet,
                   const std::string &category, const std::string& label, 
                   unsigned int mc, DQMStore::IBooker & ibook);

    reco::TaggingVariableName var;
    unsigned int nBins;
    double min, max;
    bool logScale;

    struct Plot {
      std::shared_ptr< FlavourHistograms<double> > histo;
      unsigned int index;
    };

    std::vector<Plot> plots;
    std::string label;
  };

  std::vector<VariableConfig> variables;
};

#endif
