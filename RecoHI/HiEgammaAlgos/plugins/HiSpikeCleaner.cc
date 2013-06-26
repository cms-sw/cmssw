// -*- C++ -*-
//111
// Package:    HiSpikeCleaner
// Class:      HiSpikeCleaner
// 
/**\class HiSpikeCleaner HiSpikeCleaner.cc RecoHI/HiSpikeCleaner/src/HiSpikeCleaner.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yong Kim,32 4-A08,+41227673039,
//         Created:  Mon Nov  1 18:22:21 CET 2010
// $Id: HiSpikeCleaner.cc,v 1.10 2012/01/28 10:43:20 eulisse Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"



//
// class declaration
//

class HiSpikeCleaner : public edm::EDProducer {
   public:
      explicit HiSpikeCleaner(const edm::ParameterSet&);
      ~HiSpikeCleaner();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

   edm::InputTag sCInputProducer_;
   edm::InputTag rHInputProducerB_;
   edm::InputTag rHInputProducerE_;

   std::string outputCollection_;
   double TimingCut_;
   double swissCutThr_;
   double etCut_;
   
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
HiSpikeCleaner::HiSpikeCleaner(const edm::ParameterSet& iConfig)
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/
   //now do what ever other initialization is needed
   
   rHInputProducerB_  = iConfig.getParameter<edm::InputTag>("recHitProducerBarrel");
   rHInputProducerE_  = iConfig.getParameter<edm::InputTag>("recHitProducerEndcap");

   sCInputProducer_  = iConfig.getParameter<edm::InputTag>("originalSuperClusterProducer");
   TimingCut_      = iConfig.getUntrackedParameter<double>  ("TimingCut",4.0);
   swissCutThr_      = iConfig.getUntrackedParameter<double>("swissCutThr",0.95);
   etCut_            = iConfig.getParameter<double>("etCut");
   
   outputCollection_ = iConfig.getParameter<std::string>("outputColl");
   produces<reco::SuperClusterCollection>(outputCollection_);
   
   
   
}


HiSpikeCleaner::~HiSpikeCleaner()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HiSpikeCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;


   // Get raw SuperClusters from the event    
   Handle<reco::SuperClusterCollection> pRawSuperClusters;
   try { 
      iEvent.getByLabel(sCInputProducer_, pRawSuperClusters);
   } catch ( cms::Exception& ex ) {
      edm::LogError("EgammaSCCorrectionMakerError") 
	 << "Error! can't get the rawSuperClusters " 
	 << sCInputProducer_.label() ;
   }    
   
   // Get the RecHits from the event
   Handle<EcalRecHitCollection> pRecHitsB;
   try { 
      iEvent.getByLabel(rHInputProducerB_, pRecHitsB);
   } catch ( cms::Exception& ex ) {
      edm::LogError("EgammaSCCorrectionMakerError") 
	 << "Error! can't get the RecHits " 
	 << rHInputProducerB_.label();
   }    
   // Get the RecHits from the event                                                                                                            
   Handle<EcalRecHitCollection> pRecHitsE;
   try {
      iEvent.getByLabel(rHInputProducerE_, pRecHitsE);
   } catch ( cms::Exception& ex ) {
      edm::LogError("EgammaSCCorrectionMakerError")
         << "Error! can't get the RecHits "
         << rHInputProducerE_.label();
   }

   
   // get the channel status from the DB                                                                                                     
   //   edm::ESHandle<EcalChannelStatus> chStatus;
   //   iSetup.get<EcalChannelStatusRcd>().get(chStatus);
   
   edm::ESHandle<EcalSeverityLevelAlgo> ecalSevLvlAlgoHndl;
   iSetup.get<EcalSeverityLevelAlgoRcd>().get(ecalSevLvlAlgoHndl);
   
   
   // Create a pointer to the RecHits and raw SuperClusters
   const reco::SuperClusterCollection *rawClusters = pRawSuperClusters.product();
   
   
   EcalClusterLazyTools lazyTool(iEvent, iSetup, rHInputProducerB_,rHInputProducerE_);

   // Define a collection of corrected SuperClusters to put back into the event
   std::auto_ptr<reco::SuperClusterCollection> corrClusters(new reco::SuperClusterCollection);
   
   //  Loop over raw clusters and make corrected ones
   reco::SuperClusterCollection::const_iterator aClus;
   for(aClus = rawClusters->begin(); aClus != rawClusters->end(); aClus++)
      {
	 double theEt = aClus->energy()/cosh( aClus->eta() ) ;
	 //	 std::cout << " et of SC = " << theEt << std::endl;

	 if ( theEt < etCut_ )  continue;   // cut off low pT superclusters 
	 
	 bool flagS = true;
	 float swissCrx(0);
	 
	 const reco::CaloClusterPtr seed = aClus->seed();
	 DetId id = lazyTool.getMaximum(*seed).first;
	 const EcalRecHitCollection & rechits = *pRecHitsB;
	 EcalRecHitCollection::const_iterator it = rechits.find( id );
	 
	 if( it != rechits.end() ) {
	    ecalSevLvlAlgoHndl->severityLevel(id, rechits);
	    swissCrx = EcalTools::swissCross   (id, rechits, 0.,true);
	    //	    std::cout << "swissCross = " << swissCrx <<std::endl;
	    // std::cout << " timing = " << it->time() << std::endl;
	 }
	 
	 if ( fabs(it->time()) > TimingCut_ ) {
	    flagS = false;
	    //	    std::cout << " timing = " << it->time() << std::endl;
	    //   std::cout << " timing is bad........" << std::endl; 
	 }
	 if ( swissCrx > (float)swissCutThr_ ) {
	    flagS = false ;     // swissCross cut
	    //	    std::cout << "swissCross = " << swissCrx <<std::endl;   
	    //   std::cout << " removed by swiss cross cut" << std::endl;
	 }
	 // - kGood        --> good channel
	 // - kProblematic --> problematic (e.g. noisy)
	 // - kRecovered   --> recovered (e.g. an originally dead or saturated)
	 // - kTime        --> the channel is out of time (e.g. spike)
	 // - kWeird       --> weird (e.g. spike)
	 // - kBad         --> bad, not suitable to be used in the reconstruction
	 //   enum EcalSeverityLevel { kGood=0, kProblematic, kRecovered, kTime, kWeird, kBad };
	    
	 
	 reco::SuperCluster newClus;
	 if ( flagS == true)
	    newClus=*aClus;
	 else
	    continue;
	 corrClusters->push_back(newClus);
      }
   
   // Put collection of corrected SuperClusters into the event
   iEvent.put(corrClusters, outputCollection_);   
   
}

// ------------ method called once each job just before starting event loop  ------------
void 
HiSpikeCleaner::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiSpikeCleaner::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiSpikeCleaner);
