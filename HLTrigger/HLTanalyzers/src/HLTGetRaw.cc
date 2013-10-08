/** \class HLTGetRaw
 *
 * See header file for documentation
 *
 *
 *  \author various
 *
 */

#include "HLTrigger/HLTanalyzers/interface/HLTGetRaw.h"

#include "DataFormats/Common/interface/Handle.h"

// system include files
#include <memory>
#include <vector>
#include <map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVector.h"

// using namespace edm;
// using namespace std;

//
// constructors and destructor
//
HLTGetRaw::HLTGetRaw(const edm::ParameterSet& ps)
{
  RawDataCollection_ = ps.getParameter<edm::InputTag>("RawDataCollection");
  RawDataToken_ = consumes<FEDRawDataCollection>(RawDataCollection_);
}

HLTGetRaw::~HLTGetRaw()
{ }

void
HLTGetRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("RawDataCollection",edm::InputTag("rawDataCollector"));
  descriptions.add("hltgetRaw",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTGetRaw::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//    using namespace edm;

    edm::Handle<FEDRawDataCollection> RawDataHandle ; 
    iEvent.getByToken(RawDataToken_, RawDataHandle );

    LogDebug("DigiInfo") << "Loaded Raw Data Collection: " << RawDataCollection_ ; 

    
}
