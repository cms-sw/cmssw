/** \class TriggerSummaryAnalyzerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2008/05/02 12:13:28 $
 *  $Revision: 1.4 $
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
       cout << iFO << " " << handle->filterTag(iFO).encode() << " " << endl;
     }
     cout << "Linearised collections of Refs: " << endl;
     cout << "  Photons: "
	  << handle->photonIds().size() << "/"
	  << handle->photonRefs().size() << endl;
     cout << "  Electrons: "
	  << handle->electronIds().size() << "/"
	  << handle->electronRefs().size() << endl;
     cout << "  Muons: "
	  << handle->muonIds().size() << "/"
	  << handle->muonRefs().size() << endl;
     cout << "  Jets: "
	  << handle->jetIds().size() << "/"
	  << handle->jetRefs().size() << endl;
     cout << "  Composites: "
	  << handle->compositeIds().size() << "/"
	  << handle->compositeRefs().size() << endl;
     cout << "  METs: "
	  << handle->metIds().size() << "/"
	  << handle->metRefs().size() << endl;
     cout << "  HTs: "
	  << handle->htIds().size() << "/"
	  << handle->htRefs().size() << endl;
     cout << "  Pixtracks: "
	  << handle->pixtrackIds().size() << "/"
	  << handle->pixtrackRefs().size() << endl;
     cout << "  L1EMs: "
	  << handle->l1emIds().size() << "/"
	  << handle->l1emRefs().size() << endl;
     cout << "  L1Muons: "
	  << handle->l1muonIds().size() << "/"
	  << handle->l1muonRefs().size() << endl;
     cout << "  L1Jets: "
	  << handle->l1jetIds().size() << "/"
	  << handle->l1jetRefs().size() << endl;
     cout << "  L1EtMiss: "
	  << handle->l1etmissIds().size() << "/"
	  << handle->l1etmissRefs().size() << endl;
   } else {
     cout << "Handle invalid! Check InputTag provided." << endl;
   }
   cout << endl;
   
   return;
}
