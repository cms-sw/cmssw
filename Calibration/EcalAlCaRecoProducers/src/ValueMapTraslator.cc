// -*- C++ -*-
//
// Package:    ValueMapTraslator
// Class:      ValueMapTraslator
// 
/**\class ValueMapTraslator ValueMapTraslator.cc Calibration/ValueMapTraslator/src/ValueMapTraslator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Shervin Nourbakhsh,32 4-C03,+41227672087,
//         Created:  Sat Jul 13 15:40:56 CEST 2013
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

//#define DEBUG
//
// class declaration
//

class ValueMapTraslator : public edm::EDProducer {
  typedef double value_t;
  typedef edm::ValueMap<value_t> Map_t;

   public:
      explicit ValueMapTraslator(const edm::ParameterSet&);
      ~ValueMapTraslator();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
  edm::InputTag referenceCollectionTAG,oldreferenceCollectionTAG;
  edm::InputTag inputCollectionTAG;
  std::string outputCollectionName;
  
  edm::EDGetTokenT<reco::GsfElectronCollection> referenceToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> oldreferenceToken_;
  edm::EDGetTokenT<Map_t> inputToken_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
ValueMapTraslator::ValueMapTraslator(const edm::ParameterSet& iConfig):
  referenceCollectionTAG(iConfig.getParameter<edm::InputTag>("referenceCollection")),
  oldreferenceCollectionTAG(iConfig.getParameter<edm::InputTag>("oldreferenceCollection")),
  inputCollectionTAG(iConfig.getParameter<edm::InputTag>("inputCollection")),
  outputCollectionName(iConfig.getParameter<std::string>("outputCollection"))
{
   //now do what ever other initialization is needed
  referenceToken_    = consumes<reco::GsfElectronCollection>(referenceCollectionTAG);
  oldreferenceToken_ = consumes<reco::GsfElectronCollection>(oldreferenceCollectionTAG);  
  inputToken_        = consumes< Map_t >(inputCollectionTAG);
  /// \todo outputCollectionName = inputCollection+postfix
  produces< Map_t >(outputCollectionName);
  
}


ValueMapTraslator::~ValueMapTraslator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ValueMapTraslator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   std::vector<value_t>  valueVector;
   std::auto_ptr<Map_t> valueVectorPtr(new Map_t());

   //------------------------------ 
   Handle< reco::GsfElectronCollection > referenceHandle;
   Handle< reco::GsfElectronCollection > oldreferenceHandle;
   Handle< Map_t > inputHandle;

   iEvent.getByToken( referenceToken_, referenceHandle);
   iEvent.getByToken( oldreferenceToken_, oldreferenceHandle);
   iEvent.getByToken( inputToken_, inputHandle); 

   for(Map_t::const_iterator valueMap_itr = inputHandle->begin();
       valueMap_itr != inputHandle->end();
       valueMap_itr++){
     for(unsigned int i = 0; i < valueMap_itr.size(); i++){
#ifdef DEBUG
       std::cout << valueMap_itr[i] << std::endl;
#endif
       //       valueVector.push_back((valueMap_itr[i])); //valueMap_itr-inputHandle->begin()]));
     }
     break;
   }

#ifdef DEBUG   
   std::cout << "Size: " << referenceHandle->size() << "\t" << oldreferenceHandle->size() << "\t" << inputHandle->size() << "\t" << valueVector.size() << std::endl;
#endif
   for(reco::GsfElectronCollection::const_iterator electronNew = referenceHandle->begin();
       electronNew!= referenceHandle->end();
       electronNew++){
    
     for(reco::GsfElectronCollection::const_iterator electron = oldreferenceHandle->begin();
	 electron!= oldreferenceHandle->end();
	 electron++){
       //if(electronNew->GsfTrackF
       if(electron->gsfTrack() != electronNew->gsfTrack()) continue; ///< requires that the track is the same, so I'm sure the electron object is the same. This to avoid the case when two electrons have the same eta and phi at the vtx 
       //if(fabs(electronNew->eta() - electron->eta())>0.0001) continue;
       //if(fabs(electronNew->phi() - electron->phi())>0.0001) continue;
       
       reco::GsfElectronRef eleRef(oldreferenceHandle, electron-oldreferenceHandle->begin());
       reco::GsfElectronRef eleRef2(referenceHandle, electronNew-referenceHandle->begin());

#ifdef DEBUG     
       std::cout << eleRef->eta() << "\t" << eleRef2->eta() << "\t" 
		 << eleRef->phi() << "\t" << eleRef2->phi() << "\t"
		 << eleRef->energy() << "\t" << eleRef2->energy() << "\t"
		 << (eleRef->gsfTrack() == eleRef2->gsfTrack()) << "\t" 
		 << (eleRef == eleRef2) << "\t" 
		 <<  (*inputHandle)[eleRef] << std::endl;
#endif
     valueVector.push_back((*inputHandle)[eleRef]); //valueMap_itr-inputHandle->begin()]));
     }
   }
   //#endif
   Map_t::Filler filler(*valueVectorPtr);
   filler.insert(referenceHandle, valueVector.begin(), valueVector.end());
   filler.fill();

   iEvent.put(valueVectorPtr);
   
}

// ------------ method called once each job just before starting event loop  ------------
void 
ValueMapTraslator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ValueMapTraslator::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
ValueMapTraslator::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
ValueMapTraslator::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
ValueMapTraslator::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
ValueMapTraslator::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ValueMapTraslator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ValueMapTraslator);
