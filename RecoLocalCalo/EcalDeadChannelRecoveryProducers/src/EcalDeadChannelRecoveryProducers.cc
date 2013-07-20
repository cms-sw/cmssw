// -*- C++ -*-
//
// Package:    EcalDeadChannelRecoveryProducers
// Class:      EcalDeadChannelRecoveryProducers
// 
/**\class EcalDeadChannelRecoveryProducers EcalDeadChannelRecoveryProducers.cc RecoLocalCalo/EcalDeadChannelRecoveryProducers/src/EcalDeadChannelRecoveryProducers.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Georgios Daskalakis
//         Created:  Thu Apr 12 17:01:03 CEST 2007
// $Id: EcalDeadChannelRecoveryProducers.cc,v 1.7 2011/05/22 23:08:19 eulisse Exp $
//
//


// system include files
#include <memory>



// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryProducers/interface/EcalDeadChannelRecoveryProducers.h"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryAlgos.h"


#include <string>
#include <cstdio>

using namespace cms;
using namespace std;



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalDeadChannelRecoveryProducers::EcalDeadChannelRecoveryProducers(const edm::ParameterSet& ps)
{

   //now do what ever other initialization is needed
  CorrectDeadCells_     = ps.getParameter<bool>("CorrectDeadCells");
  CorrectionMethod_     = ps.getParameter<std::string>("CorrectionMethod");
  hitProducer_          = ps.getParameter<std::string>("hitProducer");
  hitCollection_        = ps.getParameter<std::string>("hitCollection");
  reducedHitCollection_ = ps.getParameter<std::string>("reducedHitCollection");
  DeadChannelFileName_ = ps.getParameter<std::string>("DeadChannelsFile");
  Sum8GeVThreshold_= ps.getParameter<double>("Sum8GeVThreshold");

   produces< EcalRecHitCollection >(reducedHitCollection_);
}


EcalDeadChannelRecoveryProducers::~EcalDeadChannelRecoveryProducers()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalDeadChannelRecoveryProducers::produce(edm::Event& evt, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  
  // get the hit collection from the event:
  edm::Handle<EcalRecHitCollection> rhcHandle;
  evt.getByLabel(hitProducer_, hitCollection_, rhcHandle);
  if (!(rhcHandle.isValid())) 
    {
      std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
      return;
    }
  const EcalRecHitCollection* hit_collection = rhcHandle.product();
  
  // create an auto_ptr to a EcalRecHitCollection, copy the RecHits into it and put it in the Event:
  std::auto_ptr< EcalRecHitCollection > redCollection(new EcalRecHitCollection);
  
  EcalDeadChannelRecoveryAlgos *DeadChannelCorrector = new EcalDeadChannelRecoveryAlgos(theCaloTopology.product());
  
  //Dead Cells are read from a text file
  std::vector<EBDetId>::const_iterator DeadCell;

  //
  //This should work only if we REMOVE the DC RecHit from the reduced RecHit collection 
  //
  for(EcalRecHitCollection::const_iterator it = hit_collection->begin(); it != hit_collection->end(); ++it) {     
    std::vector<EBDetId>::const_iterator CheckDead = ChannelsDeadID.begin();
    bool OverADeadRecHit=false;
    while(CheckDead<ChannelsDeadID.end()){
      if(it->detid()==*CheckDead){OverADeadRecHit=true;break;}
      CheckDead++;
    }
    if(!OverADeadRecHit)redCollection->push_back( *it );
  }
  for(DeadCell=ChannelsDeadID.begin();DeadCell<ChannelsDeadID.end();DeadCell++){
    EcalRecHit NewRecHit = DeadChannelCorrector->Correct(*DeadCell,hit_collection,CorrectionMethod_,Sum8GeVThreshold_);
    redCollection->push_back( NewRecHit );
  }



  
  delete DeadChannelCorrector ;
  
  evt.put(redCollection, reducedHitCollection_);
  
}



// ------------ method called once each job just before starting event loop  ------------
void 
EcalDeadChannelRecoveryProducers::beginJob()
{
    FILE* DeadCha;
    printf("Dead Channels FILE: %s\n",DeadChannelFileName_.c_str());
    DeadCha = fopen(DeadChannelFileName_.c_str(),"r");

    int fileStatus=0;
    int ieta=-10000;
    int iphi=-10000;
    while(fileStatus != EOF) {
    fileStatus = fscanf(DeadCha,"%d %d\n",&ieta,&iphi);
    //    std::cout<<" ieta "<<ieta<<" iphi "<<iphi<<std::endl;
    if(ieta==-10000||iphi==-10000){std::cout << "Problem reading Dead Channels file "<<std::endl;break;}
    EBDetId cell(ieta,iphi);
    ChannelsDeadID.push_back(cell);
    } //end while	    
    fclose(DeadCha);
    //  std::cout<<" Read "<<ChannelsDeadID.size()<<" dead channels "<<std::endl;
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalDeadChannelRecoveryProducers::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalDeadChannelRecoveryProducers);
