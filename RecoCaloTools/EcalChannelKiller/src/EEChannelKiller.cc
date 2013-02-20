// -*- C++ -*-
//
// Package:    EcalChannelKiller
// Class:      EEChannelKiller
//
/**\class EEChannelKiller EEChannelKiller.cc RecoCaloTools/EcalChannelKiller/src/EEChannelKiller.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
// 
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Wed Nov 21 11:24:39 EET 2012
// 
//      Nov 21 2012:   First version of the code. Based on the old "EcalChannelKiller.cc" code
//


// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapHardcodedTopology.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "RecoCaloTools/EcalChannelKiller/interface/EEChannelKiller.h"

#include <string>
#include <cstdio>
using namespace cms;
using namespace std;

//
// constructors and destructor
//
EEChannelKiller::EEChannelKiller(const edm::ParameterSet& ps)
{
    hitProducer_          = ps.getParameter<std::string>("hitProducer");
    hitCollection_        = ps.getParameter<std::string>("hitCollection");
    reducedHitCollection_ = ps.getParameter<std::string>("reducedHitCollection");
    DeadChannelFileName_  = ps.getParameter<std::string>("DeadChannelsFile");

    produces< EcalRecHitCollection >(reducedHitCollection_);
}


EEChannelKiller::~EEChannelKiller()
{
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EEChannelKiller::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;

    // get the hit collection from the event:
    edm::Handle<EcalRecHitCollection> rhcHandle;
    iEvent.getByLabel(hitProducer_, hitCollection_, rhcHandle);
    if (!(rhcHandle.isValid()))
    {
        //  std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
        return;
    }
    const EcalRecHitCollection* hit_collection = rhcHandle.product();

    int nRed = 0;

    // create an auto_ptr to a EcalRecHitCollection, copy the RecHits into it and put in the Event:
    std::auto_ptr< EcalRecHitCollection > redCollection(new EcalRecHitCollection);

    for (EcalRecHitCollection::const_iterator it = hit_collection->begin(); it != hit_collection->end(); ++it) {

        double NewEnergy = it->energy();
        bool   ItIsDead  = false;
        
        //Dead Cells are read from text files
        std::vector<EEDetId>::const_iterator DeadCell;
        for (DeadCell = ChannelsDeadID.begin(); DeadCell != ChannelsDeadID.end(); ++DeadCell) {
            if ( it->detid() == *DeadCell ) {
                ItIsDead=true;
                NewEnergy =0.;
                nRed++;
                //  If a "dead" cell is detected add a corresponding recHit with zero energy.
                //  It perserves the total number of recHits and simulates the true "dead" cell situation.
                EcalRecHit NewDeadHit(it->id(),NewEnergy,it->time());
                redCollection->push_back( NewDeadHit );
            }
        }//End looping on vector of Dead Cells

        // Make a new RecHit
        //
        // TODO what will be the it->time() for D.C. ?
        // Could we use it for "correction" identification?
        //
        if ( !ItIsDead ) { redCollection->push_back( *it ); }
    }

    iEvent.put(redCollection, reducedHitCollection_);

}




// ------------ method called once each job just before starting event loop  ------------
void
EEChannelKiller::beginJob()
{
    //Open the DeadChannel file, read it.
    FILE* DeadCha;
    printf("Dead Channels FILE: %s\n",DeadChannelFileName_.c_str());
    DeadCha = fopen(DeadChannelFileName_.c_str(),"r");

    int fileStatus=0;
    int ix=-10000;
    int iy=-10000;
    int iz=-10000;
    while (fileStatus != EOF) {

        fileStatus = fscanf(DeadCha,"%d %d %d\n",&ix,&iy,&iz);

        //  Problem reading Dead Channels file
        if (ix==-10000||iy==-10000||iz==-10000) { break; }

        if( EEDetId::validDetId(ix,iy,iz) ) {
            EEDetId cell(ix,iy,iz);
            ChannelsDeadID.push_back(cell);
        }
        
    } //end while
    
    fclose(DeadCha);
}



// ------------ method called once each job just after ending the event loop  ------------
void
EEChannelKiller::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EEChannelKiller);
