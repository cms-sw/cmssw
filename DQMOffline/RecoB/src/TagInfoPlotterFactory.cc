#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "DQMOffline/RecoB/interface/TrackCountingTagPlotter.h"
#include "DQMOffline/RecoB/interface/TrackProbabilityTagPlotter.h"
#include "DQMOffline/RecoB/interface/SoftLeptonTagPlotter.h"
#include "DQMOffline/RecoB/interface/TrackIPTagPlotter.h"
#include "DQMOffline/RecoB/interface/TaggingVariablePlotter.h"
#include "DQMOffline/RecoB/interface/MVAJetTagPlotter.h"
#include "FWCore/Utilities/interface/CodedException.h"

using namespace std;

BaseTagInfoPlotter*  TagInfoPlotterFactory::buildPlotter(string dataFormatType, const TString & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, bool update)
{
  if (dataFormatType == "TrackCounting") {
    return new TrackCountingTagPlotter(tagName, etaPtBin, pSet, update);
  } else if (dataFormatType == "TrackProbability") {
    return new TrackProbabilityTagPlotter(tagName, etaPtBin, pSet, update);
  } else if (dataFormatType == "SoftLepton") {
    return new SoftLeptonTagPlotter(tagName, etaPtBin, pSet, update);
  } else if (dataFormatType == "TrackIP") {
    return new TrackIPTagPlotter(tagName, etaPtBin, pSet, update);
  } else if (dataFormatType == "TaggingVariable") {
    return new TaggingVariablePlotter(tagName, etaPtBin, pSet, update);
  } else if (dataFormatType == "GenericMVA") {
    return new MVAJetTagPlotter(tagName, etaPtBin, pSet, update);
  }
  throw cms::Exception("Configuration")
    << "BTagPerformanceAnalysis: Unknown ExtendedTagInfo " << dataFormatType << endl
    << "Choose between TrackCounting, TrackProbability, SoftLepton, TrackIP, TaggingVariable, GenericMVA\n";
}
