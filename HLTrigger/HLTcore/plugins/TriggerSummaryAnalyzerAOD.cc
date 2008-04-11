/** \class TriggerSummaryAnalyzerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2008/04/11 17:08:14 $
 *  $Revision: 1.1 $
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

   cout << "TriggerSummaryAnalyzerAOD: content of TriggerEvent: " << inputTag_.encode();

   Handle<TriggerEvent> handle;
   iEvent.getByLabel(inputTag_,handle);
   if (handle.isValid()) {
     cout << endl;
     cout << "Used Processname: " << handle->usedProcessName() << endl;
     cout << endl;
     const size_type nO(handle->sizeObjects());
     cout << "Number of TriggerObjects: " << nO << endl;
     cout << "The TriggerObjects: #, id, pt, eta, phi, mass" << endl;
     const TriggerObjectCollection& TOC(handle->getObjects());
     for (size_type iO=0; iO!=nO; ++iO) {
       const TriggerObject& TO(TOC[iO]);
       cout << iO << " " << TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass() << endl;
     }
     cout << endl;
     const size_type nF(handle->sizeFilters());
     cout << "Number of TriggerFilters: " << nF << endl;
     cout << "The Filters: #, label, #ids, #keys" << endl;
     for (size_type iF=0; iF!=nF; ++iF) {
       const Vids& VIDS (handle->filterIds(iF));
       const Keys& KEYS(handle->filterKeys(iF));
       const size_type nI(VIDS.size());
       const size_type nK(KEYS.size());
       cout << iF << " " << handle->filterLabel(iF) << " " << nI << " " << nK << endl;
       cout << "  Ids:";
       for (size_type iI=0; iI!=nI; ++iI) { cout << " " << VIDS[iI];}
       cout << endl;
       cout << " Keys:";
       for (size_type iK=0; iK!=nK; ++iK) { cout << " " << KEYS[iK];}
       cout << endl;
     }
     cout << endl;
   } else {
     cout << "Handle invalid!" << endl;
   }
   
   return;
}
