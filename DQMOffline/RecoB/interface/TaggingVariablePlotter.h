#ifndef TaggingVariablePlotter_H
#define TaggingVariablePlotter_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TaggingVariablePlotter : public BaseTagInfoPlotter {

 public:

  TaggingVariablePlotter (const TString & tagName, const EtaPtBin & etaPtBin,
	const edm::ParameterSet& pSet, bool update = false,
	const std::string &category = std::string());

  ~TaggingVariablePlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const int & jetFlavour);

  void analyzeTag (const reco::TaggingVariableList & variables, const int & jetFlavour);

  virtual void finalize ();


  void epsPlot(const TString & name);

  void psPlot(const TString & name);

 private:

  struct VariableConfig {
    VariableConfig(const std::string &name, const edm::ParameterSet& pSet,
                   bool update, const std::string &category, std::string label);

    reco::TaggingVariableName	var;
    unsigned int		nBins;
    double			min, max;
    bool			logScale;

    struct Plot {
      boost::shared_ptr< FlavourHistograms<double> >	histo;
      unsigned int					index;
    } ;

    std::vector<Plot>		plots;
    std::string label;
  } ;

  std::vector<VariableConfig>	variables;
} ;

#endif
