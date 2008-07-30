#include "PhysicsTools/TagAndProbe/interface/eidCandProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include <string>


eidCandProducer::eidCandProducer(const edm::ParameterSet &params)
{

  _inputProducer = params.getParameter<std::string>("InputProducer");
  _electronIDAssocProducer = params.getParameter<std::string>("ElectronIDAssociationProducer");

   produces<reco::GsfElectronCollection>();
}




eidCandProducer::~eidCandProducer()
{

}


//
// member functions
//


// ------------ method called to produce the data  ------------

void eidCandProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup)
{
   // Create the output collection
   std::auto_ptr<reco::GsfElectronCollection> outCol(new reco::GsfElectronCollection);


   // Run the algoritm and store in the event
   edm::Handle<reco::GsfElectronCollection> eleCandidatesHandle;
   try
   {
      event.getByLabel(_inputProducer, eleCandidatesHandle);
   }
   catch(cms::Exception &ex)
   {
      edm::LogError("GsfElectron ") << "Error! Can't get collection " << _inputProducer;
      throw ex;
   }

   // Get electronId association map
   edm::Handle<reco::ElectronIDAssociationCollection> electronIDAssocHandle;
   try
   {
      event.getByLabel(_electronIDAssocProducer, electronIDAssocHandle);
   }
   catch(cms::Exception &ex)
   {
      edm::LogError("ElectronId ") << "Error! Can't get collection " << 
	_electronIDAssocProducer;
      throw ex;
   }


   // Loop over electrons
   for(unsigned int i = 0; i < eleCandidatesHandle->size(); ++i) {
     // Get cut decision for each electron
     edm::Ref<reco::GsfElectronCollection> electronRef(eleCandidatesHandle, i);
     reco::ElectronIDAssociationCollection::const_iterator electronIDAssocItr;
     electronIDAssocItr = electronIDAssocHandle->find(electronRef);
     const reco::ElectronIDRef &id = electronIDAssocItr->val;
     bool cutDecision = id->cutBasedDecision();
     double lh = id->likelihood();
     double nn = id->neuralNetOutput();


     if(cutDecision) outCol->push_back(*electronRef);
     else if(lh != -1.0 || nn != -1.0)  outCol->push_back(*electronRef);
   } 

   event.put(outCol);
}



// ------------ method called once each job just before starting event loop  ---



void eidCandProducer::beginJob(const edm::EventSetup &eventSetup) {
}




void eidCandProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE( eidCandProducer );
