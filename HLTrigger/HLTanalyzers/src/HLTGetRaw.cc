/** \class HLTGetRaw
 *
 * See header file for documentation
 *
 *  $Date: 2007/04/12 09:57:13 $
 *  $Revision: 1.2 $
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVector.h"

using namespace edm;
using namespace std;

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
    using namespace edm;

    /*

    Handle<RawDataCollection> RawDataHandle ; 

    iEvent.getByLabel(RawDataCollection_, RawDataHandle );

    LogDebug("DigiInfo") << "total # RawData: " << RawDataHandle->size();

    */
}
