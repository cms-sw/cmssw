#include "PhysicsTools/TagAndProbe/interface/eTriggerCandProducer.h"
#include <cmath>
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <string>
#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerAOD.h"
#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerRAW.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"


eTriggerCandProducer::eTriggerCandProducer(const edm::ParameterSet& iConfig )
{

  _inputProducer = iConfig.getParameter<std::string>("InputProducer");

   // **************** Trigger ******************* //
   const edm::InputTag dTriggerEventTag("hltTriggerSummaryAOD");
   triggerEventTag_ = 
      iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag",
						   dTriggerEventTag);

   const edm::InputTag dHLTTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter");
   hltTag_ = iConfig.getUntrackedParameter<edm::InputTag>("hltTag",dHLTTag);

   delRMatchingCut_ = iConfig.getUntrackedParameter<double>("triggerDelRMatch",
							    0.15);
   // ******************************************** //



   produces<reco::GsfElectronCollection>();
}




eTriggerCandProducer::~eTriggerCandProducer()
{

}


//
// member functions
//


// ------------ method called to produce the data  ------------

void eTriggerCandProducer::produce(edm::Event &event, 
				   const edm::EventSetup &eventSetup)
{

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;


   // Create the output collection
   std::auto_ptr<reco::GsfElectronCollection> 
     outCol(new reco::GsfElectronCollection);



   // Get the input GsfElectron collection
   edm::Handle<reco::GsfElectronCollection> eleCandidatesHandle;
   try
   {
      event.getByLabel(_inputProducer, eleCandidatesHandle);
   }
   catch(cms::Exception &ex)
   {
      edm::LogError("GsfElectron ") << "Error! Can't get collection " << 
	_inputProducer;
      throw ex;
   }

   
   // Trigger Info
   edm::Handle<trigger::TriggerEvent> trgEvent;
   event.getByLabel(triggerEventTag_,trgEvent);

   // Some sanity checks
   if (not trgEvent.isValid()) {
     edm::LogInfo("info")<< "********NO TRIGGER INFO*********** ";
     return;
   }

   // find how many relevant
   const size_type index = trgEvent->filterIndex(hltTag_.label());
   if( !(index < trgEvent->sizeFilters())) {
     edm::LogInfo("info")<< "********NO TRIGGER INFO*********** ";
     return;
   }

   // find how many objects there are
   const trigger::Keys& KEYS(trgEvent->filterKeys(index));
   const size_type nK(KEYS.size());
   // loop over these objects to see whether they match
   const trigger::TriggerObjectCollection& TOC = trgEvent->getObjects();
     

   ///// /////// For debugging /////////////////////
//    const size_type nF(trgEvent->sizeFilters());
//    if(nF < index)
//      edm::LogInfo("info")<< "**** TRIGGER index is larger than MAX *****";
//      cout << "Number of TriggerFilters: " << nF << endl;
//      cout << "The Filters: #, label, #ids/#keys, the id/key pairs" << endl;
//      for (size_type iF=0; iF!=nF; ++iF) {
//        const Vids& VIDS (trgEvent->filterIds(iF));
//        const Keys& KEYS(trgEvent->filterKeys(iF));
//        const size_type nI(VIDS.size());
//        const size_type nK(KEYS.size());
//        cout << iF << " " << trgEvent->filterLabel(iF)
// 	    << " " << nI << "/" << nK
// 	    << " the pairs: ";
//        const size_type n(max(nI,nK));
//        for (size_type i=0; i!=n; ++i) {
// 	 cout << " " << VIDS[i] << "/" << KEYS[i];
//        }
//        cout << endl;
//        assert (nI==nK);
//      }
   ///// //////////////////////////////////////////////////



   // Loop over electrons
   for(unsigned int i = 0; i < eleCandidatesHandle->size(); ++i) {
     // Get cut decision for each electron
     edm::Ref<reco::GsfElectronCollection> electronRef(eleCandidatesHandle, i);
     reco::GsfElectron electron = *electronRef;


     // Did this tag cause a HLT trigger?
     bool hltTrigger = false;


     for(int ipart = 0; ipart != nK; ++ipart) { 

       const trigger::TriggerObject& TO = TOC[KEYS[ipart]];	
       double dRval = deltaR((float)electron.eta(), (float)electron.phi(), 
			     TO.eta(), TO.phi());	
       hltTrigger = (abs(TO.id())==11) && (dRval < delRMatchingCut_);
       if( hltTrigger ) break;
     }       


     if(hltTrigger) outCol->push_back(*electronRef);
   } 

   event.put(outCol);
}











// ---- method called once each job just before starting event loop  ---



void eTriggerCandProducer::beginJob(const edm::EventSetup &eventSetup) {
}




void eTriggerCandProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE( eTriggerCandProducer );
