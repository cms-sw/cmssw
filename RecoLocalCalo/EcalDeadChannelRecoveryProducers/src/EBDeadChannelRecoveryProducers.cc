// -*- C++ -*-
//
// Package:    EcalDeadChannelRecoveryProducers
// Class:      EBDeadChannelRecoveryProducers
//
/**\class EBDeadChannelRecoveryProducers EBDeadChannelRecoveryProducers.cc RecoLocalCalo/EcalDeadChannelRecoveryProducers/src/EBDeadChannelRecoveryProducers.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
// 
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Wed Nov 21 11:24:39 EET 2012
// 
//      Nov 21 2012:   First version of the code. Based on the old "EcalDeadChannelRecoveryProducers.cc" code
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

#include "RecoLocalCalo/EcalDeadChannelRecoveryProducers/interface/EBDeadChannelRecoveryProducers.h"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EBDeadChannelRecoveryAlgos.h"

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
EBDeadChannelRecoveryProducers::EBDeadChannelRecoveryProducers(const edm::ParameterSet& ps)
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


EBDeadChannelRecoveryProducers::~EBDeadChannelRecoveryProducers()
{
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EBDeadChannelRecoveryProducers::produce(edm::Event& evt, const edm::EventSetup& iSetup)
{
    using namespace edm;

    edm::ESHandle<CaloTopology> theCaloTopology;
    iSetup.get<CaloTopologyRecord>().get(theCaloTopology);

    // get the hit collection from the event:
    edm::Handle<EcalRecHitCollection> rhcHandle;
    evt.getByLabel(hitProducer_, hitCollection_, rhcHandle);
    if (!(rhcHandle.isValid()))
    {
        //  std::cout << "could not get a handle on the EcalRecHitCollection!" << std::endl;
        return;
    }
    const EcalRecHitCollection* hit_collection = rhcHandle.product();

    // create an auto_ptr to a EcalRecHitCollection, copy the RecHits into it and put it in the Event:
    std::auto_ptr< EcalRecHitCollection > redCollection(new EcalRecHitCollection);

    EBDeadChannelRecoveryAlgos *DeadChannelCorrector = new EBDeadChannelRecoveryAlgos(theCaloTopology.product());

    //
    //  Double loop over EcalRecHit collection and "dead" cell RecHits.
    //  If we step into a "dead" cell call "DeadChannelCorrector::correct()"
    //
    for (EcalRecHitCollection::const_iterator it = hit_collection->begin(); it != hit_collection->end(); ++it) {
        std::vector<EBDetId>::const_iterator CheckDead = ChannelsDeadID.begin();
        bool OverADeadRecHit=false;
        while ( CheckDead != ChannelsDeadID.end() ) {
            if (it->detid()==*CheckDead) {
                OverADeadRecHit=true;
                bool AcceptRecHit=true;
                EcalRecHit NewRecHit = DeadChannelCorrector->correct(it->detid(),hit_collection,CorrectionMethod_,Sum8GeVThreshold_, &AcceptRecHit);
                //  Accept the new rec hit if the flag is true.
                if( AcceptRecHit ) { redCollection->push_back( NewRecHit ); }
                else               { redCollection->push_back( *it );}
	            break;
	        }
	        CheckDead++;
        }
        if (!OverADeadRecHit) { redCollection->push_back( *it ) ; }
    }
    
    delete DeadChannelCorrector ;

    evt.put(redCollection, reducedHitCollection_);
}


// ------------ method called once each job just before starting event loop  ------------
void
EBDeadChannelRecoveryProducers::beginJob()
{
    //Open the DeadChannel file, read it.
    FILE* DeadCha;
    printf("Dead Channels FILE: %s\n",DeadChannelFileName_.c_str());
    DeadCha = fopen(DeadChannelFileName_.c_str(),"r");

    int fileStatus=0;
    int ieta=-10000;
    int iphi=-10000;
    
    while (fileStatus != EOF) {
    
        fileStatus = fscanf(DeadCha,"%d %d\n",&ieta,&iphi);

        //  Problem reading Dead Channels file
        if (ieta==-10000||iphi==-10000) { break; }
        
        if( EBDetId::validDetId(ieta,iphi) ) {
            EBDetId cell(ieta,iphi);
            ChannelsDeadID.push_back(cell);
        }
        
    } //end while
    
    fclose(DeadCha);
}

// ------------ method called once each job just after ending the event loop  ------------
void
EBDeadChannelRecoveryProducers::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(EBDeadChannelRecoveryProducers);
