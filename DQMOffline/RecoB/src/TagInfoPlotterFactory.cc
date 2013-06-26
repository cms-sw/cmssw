#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "DQMOffline/RecoB/interface/TrackCountingTagPlotter.h"
#include "DQMOffline/RecoB/interface/TrackProbabilityTagPlotter.h"
#include "DQMOffline/RecoB/interface/SoftLeptonTagPlotter.h"
#include "DQMOffline/RecoB/interface/TrackIPTagPlotter.h"
#include "DQMOffline/RecoB/interface/TaggingVariablePlotter.h"
#include "DQMOffline/RecoB/interface/MVAJetTagPlotter.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

BaseTagInfoPlotter*  TagInfoPlotterFactory::buildPlotter(const string& dataFormatType, const std::string & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, 
        const std::string& folderName, const bool& update, const unsigned int& mc, 
        const bool& wf)
{
  if (dataFormatType == "TrackCounting") {
    return new TrackCountingTagPlotter(folderName, etaPtBin, pSet, update, mc, wf);
  } else if (dataFormatType == "TrackProbability") {
    return new TrackProbabilityTagPlotter(folderName, etaPtBin, pSet, update, mc, wf);
  } else if (dataFormatType == "SoftLepton") {
    return new SoftLeptonTagPlotter(folderName, etaPtBin, pSet, mc, update);
  } else if (dataFormatType == "TrackIP") {
    return new TrackIPTagPlotter(folderName, etaPtBin, pSet, update, mc,wf);
  } else if (dataFormatType == "TaggingVariable") {
    return new TaggingVariablePlotter(folderName, etaPtBin, pSet, update, mc);
  } else if (dataFormatType == "GenericMVA") {
    return new MVAJetTagPlotter(tagName, etaPtBin, pSet, folderName, update, mc);
  }
  throw cms::Exception("Configuration")
    << "BTagPerformanceAnalysis: Unknown ExtendedTagInfo " << dataFormatType << endl
    << "Choose between TrackCounting, TrackProbability, SoftLepton, TrackIP, TaggingVariable, GenericMVA\n";
}
