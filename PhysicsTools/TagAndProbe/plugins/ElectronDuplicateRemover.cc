#include "PhysicsTools/TagAndProbe/interface/ElectronDuplicateRemover.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <string>
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"


ElectronDuplicateRemover::ElectronDuplicateRemover(const edm::ParameterSet &params)
{

  _inputProducer = params.getUntrackedParameter<std::string>("src", "pixelMatchGsfElectrons");
  _BarrelMaxEta  = params.getUntrackedParameter<double>("BarrelMaxEta", 1.4442);
  _EndcapMinEta  = params.getUntrackedParameter<double>("EndcapMinEta", 1.560);
  _EndcapMaxEta  = params.getUntrackedParameter<double>("EndcapMaxEta", 2.5);
  _ptMin         = params.getUntrackedParameter<double>("ptMin", 20.0);
  _ptMax         = params.getUntrackedParameter<double>("ptMax", 1000.0);

   produces<reco::GsfElectronCollection>();
}




ElectronDuplicateRemover::~ElectronDuplicateRemover()
{

}


//
// member functions
//


// ------------ method called to produce the data  ------------

void ElectronDuplicateRemover::produce(edm::Event &event, const edm::EventSetup &eventSetup)
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




   //************************************************************************
   // NEW METHOD by David WARDROPE  *****************************************
   //************* DUPLICATE ******  REMOVAL ********************************

   const reco::GsfElectronCollection * electrons = 
     eleCandidatesHandle.product();   
   int index =0;
   
   for(reco::GsfElectronCollection::const_iterator 
	 elec = electrons->begin(); elec != electrons->end();++elec) {
     
     double elecEta = elec->superCluster().get()->eta();
     double elecE   = elec->superCluster().get()->energy();
     double elecPt  = elecE / cosh(elecEta);
    
     bool withinFiducialEta = false;
     bool withinFiducialPt  = false;

     if (fabs(elecEta) < _BarrelMaxEta ||  
	 ( fabs(elecEta) > _EndcapMinEta && fabs(elecEta) < _EndcapMaxEta))
       withinFiducialEta = true;

     if( elecPt > _ptMin && elecPt < _ptMax ) withinFiducialPt = true; 

     if( !(withinFiducialEta && withinFiducialPt) ) continue;



     const reco::GsfElectronRef  
       electronRef(eleCandidatesHandle, index);


     //Remove duplicate electrons which share a supercluster
     bool duplicate = false;
     reco::GsfElectronCollection::const_iterator BestDuplicate = elec;
     int index2 = 0;
     for(reco::GsfElectronCollection::const_iterator
	   elec2 = electrons->begin();
	 elec2 != electrons->end(); ++elec2) {
       if(elec != elec2) {
	 DetId id1= elec->superCluster()->seed()->hitsAndFractions()[0].first;
	 DetId id2 = elec2->superCluster()->seed()->hitsAndFractions()[0].first;
	 if( elec->superCluster() == elec2->superCluster()) {
	   duplicate = true;
	   if(fabs(BestDuplicate->eSuperClusterOverP()-1.0)
	      >= fabs(elec2->eSuperClusterOverP()-1.0)) {
	     BestDuplicate = elec2;
	   }
	 }
       }
       ++index2;
     }

     if(BestDuplicate == elec) outCol->push_back(*electronRef);
     ++index;
   }
   //
   event.put(outCol);
}






// ------------ method called once each job just before starting event loop  ---



void ElectronDuplicateRemover::beginJob() {}




void ElectronDuplicateRemover::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE( ElectronDuplicateRemover );
