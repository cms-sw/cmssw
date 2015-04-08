// -*- C++ -*-
//
// Package:    HiRecHitSpikeCleaner
// Class:      HiRecHitSpikeCleaner
//
/**\class HiRecHitSpikeCleaner HiRecHitSpikeCleaner.cc RecoHI/HiRecHitSpikeCleaner/src/HiRecHitSpikeCleaner.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yong Kim,32 4-A08,+41227673039,
//         Created:  Mon Nov  1 18:22:21 CET 2010
// $Id: HiRecHitSpikeCleaner.cc,v 1.4 2011/10/17 12:58:04 yjlee Exp $
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

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"

//
// class declaration
//

class HiRecHitSpikeCleaner : public edm::EDProducer {
public:
  explicit HiRecHitSpikeCleaner(const edm::ParameterSet&);
  ~HiRecHitSpikeCleaner();

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  edm::InputTag rHInputProducerB_;

  std::string ebOutputCollection_;
  double TimingCut_;
  double swissCutThr_;
  double etCut_;
  bool avoidIeta85_;

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
HiRecHitSpikeCleaner::HiRecHitSpikeCleaner(const edm::ParameterSet& iConfig)
{
  //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/
  //now do what ever other initialization is needed

  rHInputProducerB_  = iConfig.getParameter<edm::InputTag>("recHitProducerBarrel");

  TimingCut_      = iConfig.getUntrackedParameter<double>  ("TimingCut",4.0);
  swissCutThr_      = iConfig.getUntrackedParameter<double>("swissCutThr",0.95);
  etCut_            = iConfig.getParameter<double>("etCut");
  avoidIeta85_     = iConfig.getUntrackedParameter<bool>("avoidIeta85",true);

  ebOutputCollection_ = iConfig.getParameter<std::string>("ebOutputColl");
  produces<EcalRecHitCollection>(ebOutputCollection_);
}


HiRecHitSpikeCleaner::~HiRecHitSpikeCleaner()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HiRecHitSpikeCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHitsB;
  try {
    iEvent.getByLabel(rHInputProducerB_, pRecHitsB);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EgammaSCCorrectionMakerError")
      << "Error! can't get the RecHits "
      << rHInputProducerB_.label();
  }
  const EcalRecHitCollection *ebRecHits = pRecHitsB.product();

  // Define a collection of corrected SuperClusters to put back into the event
  std::auto_ptr<EcalRecHitCollection> corRecHitsEB(new EcalRecHitCollection);

  //get the rechit geometry
  edm::ESHandle<CaloGeometry> theCaloGeom;
  iSetup.get<CaloGeometryRecord>().get(theCaloGeom);
  const CaloGeometry* caloGeom = theCaloGeom.product();

  //  Loop over raw clusters and make corrected ones
  EcalRecHitCollection::const_iterator it;
  for(it = ebRecHits->begin(); it != ebRecHits->end(); it++)
  {
    const GlobalPoint &position = caloGeom->getPosition(it->id());
    double rhEt = it->energy()/cosh(position.eta());
    //	 std::cout << " et of SC = " << theEt << std::endl;

    bool flagS = true;
    float swissCrx(0);

    swissCrx = EcalTools::swissCross   (it->id(), *ebRecHits, 0.,avoidIeta85_); //EcalSeverityLevelAlgo::swissCross(it->id(), *ebRecHits, 0.,avoidIeta85_);
    //	    std::cout << "swissCross = " << swissCrx <<std::endl;
    // std::cout << " timing = " << it->time() << std::endl;

    if(rhEt > etCut_) {
      if (fabs(it->time()) > TimingCut_ ) {
	flagS = false;
	//std::cout<<"cut on time: " << it->time() << std::endl;
      }
      if ( swissCrx > (float)swissCutThr_ ) {
	flagS = false ;     // swissCross cut
	//std::cout<<"cut on swissCross: " << swissCrx << std::endl;
      }
    }

    EcalRecHit newRecHit;
    if ( flagS == true)
      newRecHit=*it;
    else
      continue;
    if(rhEt > 3)
      std::cout<<"ADDING RECHIT - time: " << it->time() << " swiss: " << swissCrx << std::endl;
    //std::cout<<"ADDING CLEAN RECHIT"<<std::endl;
    corRecHitsEB->push_back(newRecHit);
  }

  // Put collection of corrected SuperClusters into the event
  iEvent.put(corRecHitsEB, ebOutputCollection_);
}

// ------------ method called once each job just before starting event loop  ------------
void
HiRecHitSpikeCleaner::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
HiRecHitSpikeCleaner::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiRecHitSpikeCleaner);
