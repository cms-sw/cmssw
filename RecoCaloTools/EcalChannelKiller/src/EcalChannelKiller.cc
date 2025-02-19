// -*- C++ -*-
//
// Package:    EcalChannelKiller
// Class:      EcalChannelKiller
// 
/**\class EcalChannelKiller EcalChannelKiller.cc RecoCaloTools/EcalChannelKiller/src/EcalChannelKiller.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Georgios Daskalakis
//         Created:  Tue Apr 24 17:21:31 CEST 2007
// $Id: EcalChannelKiller.cc,v 1.8 2012/01/10 18:22:10 eulisse Exp $
//
//




// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"


#include "RecoCaloTools/EcalChannelKiller/interface/EcalChannelKiller.h"

#include <string>
#include <cstdio>
using namespace cms;
using namespace std;


//
// constructors and destructor
//
EcalChannelKiller::EcalChannelKiller(const edm::ParameterSet& ps)
{

   hitProducer_          = ps.getParameter<std::string>("hitProducer");
   hitCollection_        = ps.getParameter<std::string>("hitCollection");
   reducedHitCollection_ = ps.getParameter<std::string>("reducedHitCollection");
   DeadChannelFileName_  = ps.getParameter<std::string>("DeadChannelsFile");

   produces< EcalRecHitCollection >(reducedHitCollection_);


}


EcalChannelKiller::~EcalChannelKiller()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalChannelKiller::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get the hit collection from the event:
   edm::Handle<EcalRecHitCollection> rhcHandle;
   iEvent.getByLabel(hitProducer_, hitCollection_, rhcHandle);
   if (!(rhcHandle.isValid())) 
     {
       //       std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
       return;
     }
   const EcalRecHitCollection* hit_collection = rhcHandle.product();
 
   int nRed = 0;
   
   // create an auto_ptr to a EcalRecHitCollection, copy the RecHits into it and put in the Event:
   std::auto_ptr< EcalRecHitCollection > redCollection(new EcalRecHitCollection);
   
   
   for(EcalRecHitCollection::const_iterator it = hit_collection->begin(); it != hit_collection->end(); ++it) {
     
     double NewEnergy =it->energy();
     bool ItIsDead=false;
     //Dead Cells are read from text files
     std::vector<EBDetId>::const_iterator DeadCell;
     for(DeadCell=ChannelsDeadID.begin();DeadCell<ChannelsDeadID.end();DeadCell++){
       if(it->detid()==*DeadCell){
	 ItIsDead=true;
	 NewEnergy =0.;
	 nRed++;
	 
       }
     }//End looping on vector of Dead Cells

     // Make a new RecHit
     //
     // TODO what will be the it->time() for D.C. ?
     // Could we use it for "correction" identification?
     //
     if(!ItIsDead){
       EcalRecHit NewHit(it->id(),NewEnergy,it->time());
       redCollection->push_back( NewHit );
     }
   }
   //   std::cout << "total # hits: " << nTot << "  #hits with E = " << 0 << " GeV : " << nRed << std::endl;
   
   iEvent.put(redCollection, reducedHitCollection_);
   
}




// ------------ method called once each job just before starting event loop  ------------
void 
EcalChannelKiller::beginJob()
{

  //Open the DeadChannel file, read it.
  FILE* DeadCha;
  printf("Dead Channels FILE: %s\n",DeadChannelFileName_.c_str());
  DeadCha = fopen(DeadChannelFileName_.c_str(),"r");

  int fileStatus=0;
  int ieta=-10000;
  int iphi=-10000;
  while(fileStatus != EOF) {
    fileStatus = fscanf(DeadCha,"%d %d\n",&ieta,&iphi);
    //    std::cout<<" ieta "<<ieta<<" iphi "<<iphi<<std::endl;
    if(ieta==-10000||iphi==-10000){/*std::cout << "Problem reading Dead Channels file "<<std::endl;*/break;}
    EBDetId cell(ieta,iphi);
    ChannelsDeadID.push_back(cell);
  } //end while	    
  fclose(DeadCha);
  //  std::cout<<" Read "<<ChannelsDeadID.size()<<" dead channels "<<std::endl;
  
}



// ------------ method called once each job just after ending the event loop  ------------
void 
EcalChannelKiller::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalChannelKiller);
