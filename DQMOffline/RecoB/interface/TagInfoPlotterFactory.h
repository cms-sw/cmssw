#ifndef TagInfoPlotterFactory_H
#define TagInfoPlotterFactory_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class TagInfoPlotterFactory  {
 public:
   BaseTagInfoPlotter* buildPlotter(const std::string& dataFormatType, const std::string & tagName,
				    const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, 
				    const std::string& folderName, const bool& update, const unsigned int& mc,
				    const bool& wf, DQMStore::IBooker & ibook);
};


#endif
