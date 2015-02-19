/** \class HLTGetRaw
 *
 * See header file for documentation
 *
 *
 *  \author various
 *
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "HLTrigger/HLTanalyzers/interface/HLTGetRaw.h"

//
// constructors and destructor
//
HLTGetRaw::HLTGetRaw(const edm::ParameterSet& ps) :
  rawDataCollection_( ps.getParameter<edm::InputTag>("RawDataCollection") ),
  rawDataToken_(      consumes<FEDRawDataCollection>(rawDataCollection_) )
{
}

HLTGetRaw::~HLTGetRaw()
{
}

void
HLTGetRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("RawDataCollection", edm::InputTag("rawDataCollector"));
  descriptions.add("hltGetRaw", desc);
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void
HLTGetRaw::analyze(edm::StreamID sid, edm::Event const & event, edm::EventSetup const & setup) const
{
  edm::Handle<FEDRawDataCollection> rawDataHandle ;
  event.getByToken(rawDataToken_, rawDataHandle );

  LogDebug("DigiInfo") << "Loaded Raw Data Collection: " << rawDataCollection_;
}
