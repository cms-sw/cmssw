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

  TaggingVariablePlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
			  const edm::ParameterSet& pSet, const bool& update, const unsigned int& mc,
	const std::string &category = std::string());

  ~TaggingVariablePlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const int & jetFlavour);

  void analyzeTag (const reco::TaggingVariableList & variables, const int & jetFlavour);

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const int & jetFlavour, const float & w);

  void analyzeTag (const reco::TaggingVariableList & variables, const int & jetFlavour, const float & w);

  virtual void finalize ();


  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

 private:

  unsigned int mcPlots_;

  struct VariableConfig {
    VariableConfig(const std::string &name, const edm::ParameterSet& pSet,
                   const bool& update, const std::string &category, const std::string& label, const unsigned int& mc);

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
