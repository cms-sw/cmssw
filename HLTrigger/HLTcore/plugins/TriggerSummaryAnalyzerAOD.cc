/** \class TriggerSummaryAnalyzerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2008/05/19 13:16:47 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerAOD.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

//
// constructors and destructor
//
TriggerSummaryAnalyzerAOD::TriggerSummaryAnalyzerAOD(const edm::ParameterSet& ps) : 
  inputTag_(ps.getParameter<edm::InputTag>("inputTag"))
{ }

TriggerSummaryAnalyzerAOD::~TriggerSummaryAnalyzerAOD()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
TriggerSummaryAnalyzerAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;


   cout << endl;
   cout << "TriggerSummaryAnalyzerAOD: content of TriggerEvent: " << inputTag_.encode() << endl;

   Handle<TriggerEvent> handle;
   iEvent.getByLabel(inputTag_,handle);
   if (handle.isValid()) {
     cout << "Used Processname: " << handle->usedProcessName() << endl;
     const size_type nC(handle->sizeCollections());
     cout << "Number of packed Collections: " << nC << endl;
     cout << "The Collections: #, tag, 1-past-end index" << endl;
     for (size_type iC=0; iC!=nC; ++iC) {
       cout << iC << " "
	    << handle->collectionTag(iC).encode() << " "
	    << handle->collectionKey(iC) << endl;
     }
     const size_type nO(handle->sizeObjects());
     cout << "Number of TriggerObjects: " << nO << endl;
     cout << "The TriggerObjects: #, id, pt, eta, phi, mass" << endl;
     const TriggerObjectCollection& TOC(handle->getObjects());
     for (size_type iO=0; iO!=nO; ++iO) {
       const TriggerObject& TO(TOC[iO]);
       cout << iO << " " << TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass() << endl;
     }
     const size_type nF(handle->sizeFilters());
     cout << "Number of TriggerFilters: " << nF << endl;
     cout << "The Filters: #, tag, #ids/#keys, the id/key pairs" << endl;
     for (size_type iF=0; iF!=nF; ++iF) {
       const Vids& VIDS (handle->filterIds(iF));
       const Keys& KEYS(handle->filterKeys(iF));
       const size_type nI(VIDS.size());
       const size_type nK(KEYS.size());
       cout << iF << " " << handle->filterTag(iF).encode()
	    << " " << nI << "/" << nK
	    << " the pairs: ";
       const size_type n(max(nI,nK));
       for (size_type i=0; i!=n; ++i) {
	 cout << " " << VIDS[i] << "/" << KEYS[i];
       }
       cout << endl;
       assert (nI==nK);
     }
   } else {
     cout << "Handle invalid! Check InputTag provided." << endl;
   }
   cout << endl;
   
   return;
}
