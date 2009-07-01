#include "PhysicsTools/TagAndProbe/interface/eTriggerCandProducer.h"
#include <cmath>
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <string>

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"


eTriggerCandProducer::eTriggerCandProducer(const edm::ParameterSet& iConfig )
{

  _inputProducer = iConfig.getParameter<edm::InputTag>("InputProducer");

   // **************** Trigger ******************* //
   const edm::InputTag dTriggerEventTag("hltTriggerSummaryAOD","","HLT");
   triggerEventTag_ = 
      iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag",
						   dTriggerEventTag);

   const edm::InputTag dHLTTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter",
			       "","HLT");
   hltTag_ = iConfig.getUntrackedParameter<edm::InputTag>("hltTag",dHLTTag);

   delRMatchingCut_ = iConfig.getUntrackedParameter<double>("triggerDelRMatch",
							    0.30);
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





   // ############# Trigger Path Debug #################

//    const edm::InputTag dTriggerResultTag("TriggerResults","","HLT");
//    edm::Handle<TriggerResults> hltResults;
//    bool b = event.getByLabel(dTriggerResultTag, hltResults);
//    TriggerNames names(*hltResults);
//    int q =0;
//    for ( TriggerNames::Strings::const_iterator 
//            j = names.triggerNames().begin();
//          j !=names.triggerNames().end(); ++j ) {
     
//      std::cout << q << ": " << *j << std::endl;
//      q++;
//    }
 

   // ############# TriggerEvent Debug #################

//    edm::ParameterSet ps;
//    const std::string processName = "HLT";
//    if (event.getProcessParameterSet(processName,ps))
//      {
//        cout << ps << endl;
//        vector< std::string> paths;
//        const std::string pathspar = "@paths";
//        paths = ps.getParameter<vector< std::string> >(pathspar);
//        for (std::vector<string>::const_iterator path = paths.begin(); path
// 	      !=paths.end(); ++path ) {

// 	 cout << *path << endl;
// 	 vector< std::string> modules;
// 	 modules = ps.getParameter<vector< std::string> >(*path);
// 	 for (std::vector<string>::const_iterator module = modules.begin();
// 	      module !=modules.end(); ++module ) {     
// 	   cout << *module << endl;
// 	 }
//        }
//      }



   // Trigger Info
   edm::Handle<trigger::TriggerEvent> trgEvent;
   event.getByLabel(triggerEventTag_,trgEvent);

   // Some sanity checks
   if (not trgEvent.isValid()) {
     edm::LogInfo("info")<< "******** Following Trigger Summary Object Not Found: " << 
       triggerEventTag_;
     event.put(outCol);
     return;
   }


   // loop over these objects to see whether they match
   const trigger::TriggerObjectCollection& TOC = trgEvent->getObjects();
     


   // find how many relevant
   // const size_type index = trgEvent->filterIndex( hltTag_ );

   int index  = trgEvent->sizeFilters();


   //  workaround for the time-being for the Relval_2_1_2 samples
   for(int i=0; i != trgEvent->sizeFilters(); ++i) {
     std::string label(trgEvent->filterTag(i).label());
     if( label == hltTag_.label() ) index = i;
   }
    


   if( index >= trgEvent->sizeFilters() ) {
     // edm::LogInfo("info")<< "******** Following TRIGGER Name Not in Dataset: " << 
     //  hltTag_.label();
     event.put(outCol);
     return;
   }


   // find how many objects there are
   const trigger::Keys& KEYS(trgEvent->filterKeys(index));
   const size_type nK(KEYS.size());



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



void eTriggerCandProducer::beginJob() {
}




void eTriggerCandProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE( eTriggerCandProducer );
