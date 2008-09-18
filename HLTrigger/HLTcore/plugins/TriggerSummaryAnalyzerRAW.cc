/** \class TriggerSummaryAnalyzerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2008/09/18 11:55:42 $
 *  $Revision: 1.5 $
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
     cout << "The TriggerFilterObjects: #, tag" << endl;
     for (size_type iFO=0; iFO!=nFO; ++iFO) {
       cout << iFO << " " << handle->filterTag(iFO).encode()
	    << "  # of objects:" << endl;
       cout << "  " << handle->photonSlice(iFO).second-
	               handle->photonSlice(iFO).first
	    << "  " << handle->electronSlice(iFO).second-
                       handle->electronSlice(iFO).first
	    << "  " << handle->muonSlice(iFO).second-
                       handle->muonSlice(iFO).first
	    << "  " << handle->jetSlice(iFO).second-
                       handle->jetSlice(iFO).first
	    << "  " << handle->compositeSlice(iFO).second-
                       handle->compositeSlice(iFO).first
	    << "  " << handle->metSlice(iFO).second-
                       handle->metSlice(iFO).first
	    << "  " << handle->htSlice(iFO).second-
                       handle->htSlice(iFO).first
	    << "  " << handle->pixtrackSlice(iFO).second-
                       handle->pixtrackSlice(iFO).first
	    << "  " << handle->l1emSlice(iFO).second-
                       handle->l1emSlice(iFO).first
	    << "  " << handle->l1muonSlice(iFO).second-
                       handle->l1muonSlice(iFO).first
	    << "  " << handle->l1jetSlice(iFO).second-
                       handle->l1jetSlice(iFO).first
	    << "  " << handle->l1etmissSlice(iFO).second-
                       handle->l1etmissSlice(iFO).first
	    << endl;
     }
     cout << "Elements in linearised collections of Refs: " << endl;
     cout << "  Photons:    " << handle->photonSize()    << endl;
     cout << "  Electrons:  " << handle->electronSize()  << endl;
     cout << "  Muons:      " << handle->muonSize()      << endl;
     cout << "  Jets:       " << handle->jetSize()       << endl;
     cout << "  Composites: " << handle->compositeSize() << endl;
     cout << "  METs:       " << handle->metSize()       << endl;
     cout << "  HTs:        " << handle->htSize()        << endl;
     cout << "  Pixtracks:  " << handle->pixtrackSize()  << endl;
     cout << "  L1EMs:      " << handle->l1emSize()      << endl;
     cout << "  L1Muons:    " << handle->l1muonSize()    << endl;
     cout << "  L1Jets:     " << handle->l1jetSize()     << endl;
     cout << "  L1EtMiss:   " << handle->l1etmissSize()  << endl;
   } else {
     cout << "Handle invalid! Check InputTag provided." << endl;
   }
   cout << endl;
   
   return;
}
