#ifndef TaggingVariablePlotter_H
#define TaggingVariablePlotter_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TaggingVariablePlotter : public BaseTagInfoPlotter {

 public:

  TaggingVariablePlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
			  const edm::ParameterSet& pSet,
			  const unsigned int& mc, const bool& willFinalize, DQMStore::IBooker & ibook,
			  const std::string &category = std::string());

  ~TaggingVariablePlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour);

  void analyzeTag (const reco::TaggingVariableList & variables, const int & jetFlavour);

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour, const float & w);

  void analyzeTag (const reco::TaggingVariableList & variables, const int & jetFlavour, const float & w);

  virtual void finalize (DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_) {}

  void epsPlot(const std::string & name) {}

  void psPlot(const std::string & name) {}

 private:

  unsigned int mcPlots_;

  struct VariableConfig {
    VariableConfig(const std::string &name, const edm::ParameterSet& pSet,
                   const std::string &category, const std::string& label, 
		   const unsigned int& mc, DQMStore::IBooker & ibook);

    reco::TaggingVariableName	var;
    unsigned int		nBins;
    double			min, max;
    bool			logScale;

    struct Plot {
      boost::shared_ptr< FlavourHistograms<double> >	histo;
      unsigned int					index;
    } ;

    std::vector<Plot> plots;
    std::string label;
  } ;

  std::vector<VariableConfig>	variables;
} ;

#endif
