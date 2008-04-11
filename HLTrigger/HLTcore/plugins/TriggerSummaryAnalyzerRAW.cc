/** \class TriggerSummaryAnalyzerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2008/01/12 16:53:57 $
 *  $Revision: 1.14 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerRAW.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<string>

//
// constructors and destructor
//
TriggerSummaryAnalyzerRAW::TriggerSummaryAnalyzerRAW(const edm::ParameterSet& ps) : 
  inputTag_(ps.getParameter<edm::InputTag>("inputTag"))
{
  LogDebug("") << "Using process name: '" << pn_ <<"'";

  LogTrace("") << "Number of unique collections requested " << nc1;

}

TriggerSummaryAnalyzerRAW::~TriggerSummaryAnalyzerRAW()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
TriggerSummaryAnalyzerRAW::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace l1extra;
   using namespace trigger;
}
