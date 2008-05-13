#ifndef TagInfoPlotterFactory_H
#define TagInfoPlotterFactory_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class TagInfoPlotterFactory  {
 public:
   BaseTagInfoPlotter* buildPlotter(string dataFormatType, const TString & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, bool update, bool mc);
};


#endif
