#include "DQMOffline/RecoB/interface/TagInfoPlotterFactory.h"
#include "DQMOffline/RecoB/interface/TrackCountingTagPlotter.h"
#include "DQMOffline/RecoB/interface/TrackProbabilityTagPlotter.h"
#include "DQMOffline/RecoB/interface/SoftLeptonTagPlotter.h"
#include "DQMOffline/RecoB/interface/IPTagPlotter.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DQMOffline/RecoB/interface/TaggingVariablePlotter.h"
#include "FWCore/Utilities/interface/Exception.h"
//#include "DQMOffline/RecoB/interface/Track2IPTagPlotter.h"

using namespace std;

std::unique_ptr<BaseTagInfoPlotter> TagInfoPlotterFactory::buildPlotter(const string& dataFormatType,
                                                                        const std::string& tagName,
                                                                        const EtaPtBin& etaPtBin,
                                                                        const edm::ParameterSet& pSet,
                                                                        const std::string& folderName,
                                                                        unsigned int mc,
                                                                        bool wf,
                                                                        DQMStore::IBooker& ibook) {
  if (dataFormatType == "TrackCounting") {
    return std::make_unique<TrackCountingTagPlotter>(folderName, etaPtBin, pSet, mc, wf, ibook);
  } else if (dataFormatType == "TrackProbability") {
    return std::make_unique<TrackProbabilityTagPlotter>(folderName, etaPtBin, pSet, mc, wf, ibook);
  } else if (dataFormatType == "SoftLepton") {
    return std::make_unique<SoftLeptonTagPlotter>(folderName, etaPtBin, pSet, mc, wf, ibook);
  } else if (dataFormatType == "TrackIP") {
    return std::make_unique<IPTagPlotter<reco::TrackRefVector, reco::JTATagInfo>>(
        folderName, etaPtBin, pSet, mc, wf, ibook);
  } else if (dataFormatType == "CandIP") {
    return std::make_unique<IPTagPlotter<std::vector<reco::CandidatePtr>, reco::JetTagInfo>>(
        folderName, etaPtBin, pSet, mc, wf, ibook);
  } else if (dataFormatType == "TaggingVariable") {
    return std::make_unique<TaggingVariablePlotter>(folderName, etaPtBin, pSet, mc, wf, ibook);
  }
  throw cms::Exception("Configuration")
      << "BTagPerformanceAnalysis: Unknown ExtendedTagInfo " << dataFormatType << endl
      << "Choose between TrackCounting, TrackProbability, SoftLepton, TrackIP, CandIP, TaggingVariable, GenericMVA\n";
}
