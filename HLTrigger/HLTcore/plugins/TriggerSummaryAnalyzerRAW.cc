/** \class TriggerSummaryAnalyzerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2008/04/11 18:09:12 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerRAW.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

//
// constructors and destructor
//
TriggerSummaryAnalyzerRAW::TriggerSummaryAnalyzerRAW(const edm::ParameterSet& ps) : 
  inputTag_(ps.getParameter<edm::InputTag>("inputTag"))
{ }

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

   cout << endl;
   cout << "TriggerSummaryAnalyzerRAW: content of TriggerEventWithRefs: " << inputTag_.encode();

   Handle<TriggerEventWithRefs> handle;
   iEvent.getByLabel(inputTag_,handle);
   if (handle.isValid()) {
     cout << "Used Processname: " << handle->usedProcessName() << endl;
     const size_type nFO(handle->size());
     cout << "Number of TriggerFilterObjects: " << nFO << endl;
     cout << "The TriggerFilterObjects: #, label" << endl;
     for (size_type iFO=0; iFO!=nFO; ++iFO) {
       cout << iFO << " " << handle->filterLabel(iFO) << " " << endl;
     }
   } else {
     cout << "Handle invalid! Check InputTag provided." << endl;
   }
   cout << endl;
   
   return;
}
