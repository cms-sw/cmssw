/** \class HLTGetRaw
 *
 * See header file for documentation
 *
 *
 *  \author various
 *
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HLTGetRaw.h"

//
// constructors and destructor
//
HLTGetRaw::HLTGetRaw(const edm::ParameterSet& ps)
    : rawDataCollection_(ps.getParameter<edm::InputTag>("RawDataCollection")),
      rawDataToken_(consumes<FEDRawDataCollection>(rawDataCollection_)) {}

HLTGetRaw::~HLTGetRaw() = default;

void HLTGetRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("RawDataCollection", edm::InputTag("rawDataCollector"));
  descriptions.add("hltGetRaw", desc);
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void HLTGetRaw::analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& setup) const {
  edm::Handle<FEDRawDataCollection> rawDataHandle;
  event.getByToken(rawDataToken_, rawDataHandle);

  LogDebug("DigiInfo") << "Loaded Raw Data Collection: " << rawDataCollection_;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTGetRaw);
