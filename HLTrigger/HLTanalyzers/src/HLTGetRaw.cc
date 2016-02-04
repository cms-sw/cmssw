/** \class HLTGetRaw
 *
 * See header file for documentation
 *
 *  $Date: 2011/01/27 10:38:51 $
 *  $Revision: 1.5 $
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

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

// using namespace edm;
// using namespace std;

//
// constructors and destructor
//
HLTGetRaw::HLTGetRaw(const edm::ParameterSet& ps)
{
  RawDataCollection_ = ps.getParameter<edm::InputTag>("RawDataCollection");
}

HLTGetRaw::~HLTGetRaw()
{ }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTGetRaw::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//    using namespace edm;

    edm::Handle<FEDRawDataCollection> RawDataHandle ; 
    iEvent.getByLabel(RawDataCollection_, RawDataHandle );

    LogDebug("DigiInfo") << "Loaded Raw Data Collection: " << RawDataCollection_ ; 

    
}
