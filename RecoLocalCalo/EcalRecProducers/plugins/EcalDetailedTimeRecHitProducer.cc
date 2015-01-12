/** \class EcalDetailedTimeRecHitProducer
 *   produce ECAL detailed time Rechits
 *  $Id:  $
 *  $Date:  $
 *  $Revision:  $
 *  \author Paolo Meridiani
 *
 **/

#include "RecoLocalCalo/EcalRecProducers/plugins/EcalDetailedTimeRecHitProducer.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cmath>
#include <vector>

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"


//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h" 

EcalDetailedTimeRecHitProducer::EcalDetailedTimeRecHitProducer(const edm::ParameterSet& ps) :
  m_geometry(0)
{
   EBRecHitCollection_ = ps.getParameter<edm::InputTag>("EBRecHitCollection");
   EERecHitCollection_ = ps.getParameter<edm::InputTag>("EERecHitCollection");
   EKRecHitCollection_ = ps.getParameter<edm::InputTag>("EKRecHitCollection");

   ebTimeDigiCollection_ = ps.getParameter<edm::InputTag>("EBTimeDigiCollection");
   eeTimeDigiCollection_ = ps.getParameter<edm::InputTag>("EETimeDigiCollection");
   ekTimeDigiCollection_ = ps.getParameter<edm::InputTag>("EKTimeDigiCollection");

   EBDetailedTimeRecHitCollection_        = ps.getParameter<std::string>("EBDetailedTimeRecHitCollection");
   EEDetailedTimeRecHitCollection_        = ps.getParameter<std::string>("EEDetailedTimeRecHitCollection");
   EKDetailedTimeRecHitCollection_        = ps.getParameter<std::string>("EKDetailedTimeRecHitCollection");
   
   correctForVertexZPosition_ = ps.getParameter<bool>("correctForVertexZPosition");
   useMCTruthVertex_ = ps.getParameter<bool>("useMCTruthVertex");
   recoVertex_       = ps.getParameter<edm::InputTag>("recoVertex");
   simVertex_       = ps.getParameter<edm::InputTag>("simVertex");

   ebTimeLayer_ = ps.getParameter<int>("EBTimeLayer");
   eeTimeLayer_ = ps.getParameter<int>("EETimeLayer");
   ekTimeLayer_ = ps.getParameter<int>("EKTimeLayer");

   produces< EBRecHitCollection >(EBDetailedTimeRecHitCollection_);
   produces< EERecHitCollection >(EEDetailedTimeRecHitCollection_);
   produces< EKRecHitCollection >(EKDetailedTimeRecHitCollection_);
}

EcalDetailedTimeRecHitProducer::~EcalDetailedTimeRecHitProducer() {
}


void EcalDetailedTimeRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
        using namespace edm;
        using namespace reco;

	edm::ESHandle<CaloGeometry>               hGeometry   ;
	es.get<CaloGeometryRecord>().get( hGeometry ) ;
	
	m_geometry = &*hGeometry;

        Handle< EBRecHitCollection > pEBRecHits;
        Handle< EERecHitCollection > pEERecHits;
        Handle< EKRecHitCollection > pEKRecHits;

        const EBRecHitCollection*  EBRecHits = 0;
        const EERecHitCollection*  EERecHits = 0; 
        const EKRecHitCollection*  EKRecHits = 0; 

	//        if ( EBRecHitCollection_.label() != "" && EBRecHitCollection_.instance() != "" ) {
        if ( EBRecHitCollection_.label() != "" ) {
                evt.getByLabel( EBRecHitCollection_, pEBRecHits);
                if ( pEBRecHits.isValid() ) {
                        EBRecHits = pEBRecHits.product(); // get a ptr to the product
#ifdef DEBUG
                        LogDebug("EcalRecHitDebug") << "total # EB rechits to be re-calibrated: " << EBRecHits->size();
#endif
                } else {
                        edm::LogError("EcalRecHitError") << "Error! can't get the product " << EBRecHitCollection_.label() ;
                }
        }

	//        if ( EERecHitCollection_.label() != "" && EERecHitCollection_.instance() != "" ) {
        if ( EERecHitCollection_.label() != ""  ) {
                evt.getByLabel( EERecHitCollection_, pEERecHits);
                if ( pEERecHits.isValid() ) {
                        EERecHits = pEERecHits.product(); // get a ptr to the product
#ifdef DEBUG
                        LogDebug("EcalRecHitDebug") << "total # EE uncalibrated rechits to be re-calibrated: " << EERecHits->size();
#endif
                } else {
                        edm::LogError("EcalRecHitError") << "Error! can't get the product " << EERecHitCollection_.label() ;
                }
        }

        if ( EKRecHitCollection_.label() != ""  ) {
                evt.getByLabel( EKRecHitCollection_, pEKRecHits);
                if ( pEKRecHits.isValid() ) {
                        EKRecHits = pEKRecHits.product(); // get a ptr to the product
#ifdef DEBUG
                        LogDebug("EcalRecHitDebug") << "total # EK uncalibrated rechits to be re-calibrated: " << EKRecHits->size() << std::endl;
#endif
                } else {
                        edm::LogError("EcalRecHitError") << "Error! can't get the product " << EKRecHitCollection_.label() ;
                }
        }

        Handle< EcalTimeDigiCollection > pEBTimeDigis;
        Handle< EcalTimeDigiCollection > pEETimeDigis;
        Handle< EcalTimeDigiCollection > pEKTimeDigis;

        const EcalTimeDigiCollection* ebTimeDigis =0;
        const EcalTimeDigiCollection* eeTimeDigis =0;
        const EcalTimeDigiCollection* ekTimeDigis =0;

        if ( ebTimeDigiCollection_.label() != "" && ebTimeDigiCollection_.instance() != "" ) {
                evt.getByLabel( ebTimeDigiCollection_, pEBTimeDigis);
                //evt.getByLabel( digiProducer_, pEBTimeDigis);
                if ( pEBTimeDigis.isValid() ) {
                        ebTimeDigis = pEBTimeDigis.product(); // get a ptr to the produc
                        edm::LogInfo("EcalDetailedTimeRecHitInfo") << "total # ebTimeDigis: " << ebTimeDigis->size() ;
                } else {
                        edm::LogError("EcalDetailedTimeRecHitError") << "Error! can't get the product " << ebTimeDigiCollection_;
                }
        }

        if ( eeTimeDigiCollection_.label() != "" && eeTimeDigiCollection_.instance() != "" ) {
                evt.getByLabel( eeTimeDigiCollection_, pEETimeDigis);
                //evt.getByLabel( digiProducer_, pEETimeDigis);
                if ( pEETimeDigis.isValid() ) {
                        eeTimeDigis = pEETimeDigis.product(); // get a ptr to the product
                        edm::LogInfo("EcalDetailedTimeRecHitInfo") << "total # eeTimeDigis: " << eeTimeDigis->size() ;
                } else {
                        edm::LogError("EcalDetailedTimeRecHitError") << "Error! can't get the product " << eeTimeDigiCollection_;
                }
        }

        if ( ekTimeDigiCollection_.label() != "" && ekTimeDigiCollection_.instance() != "" ) {
                evt.getByLabel( ekTimeDigiCollection_, pEKTimeDigis);
                //evt.getByLabel( digiProducer_, pEKTimeDigis);
                if ( pEKTimeDigis.isValid() ) {
                        ekTimeDigis = pEKTimeDigis.product(); // get a ptr to the product
//                        std::cout << "[EcalDetailedTimeRecHitInfo]"
                        edm::LogInfo("EcalDetailedTimeRecHitInfo") << "total # ekTimeDigis: " << ekTimeDigis->size() << std::endl;
                } else {
                        edm::LogError("EcalDetailedTimeRecHitError") << "Error! can't get the product " << ekTimeDigiCollection_;
                }
        }

        // collection of rechits to put in the event
        std::auto_ptr< EBRecHitCollection > EBDetailedTimeRecHits( new EBRecHitCollection );
        std::auto_ptr< EERecHitCollection > EEDetailedTimeRecHits( new EERecHitCollection );
        std::auto_ptr< EKRecHitCollection > EKDetailedTimeRecHits( new EKRecHitCollection );

	GlobalPoint* vertex=0;

	if (correctForVertexZPosition_)
	  {
	    if (!useMCTruthVertex_)
	      {
	      //Get the first reco vertex
	      // get primary vertices
		
		edm::Handle<VertexCollection> VertexHandle;
		evt.getByLabel(recoVertex_, VertexHandle);
		
		if ( VertexHandle.isValid() )
		  {
		    if ((*VertexHandle).size()>0) //at least 1 vertex
		      {
			const reco::Vertex* myVertex= &(*VertexHandle)[0];
			vertex=new GlobalPoint(myVertex->x(),myVertex->y(),myVertex->z());
		      }
		  }
		else {
		  edm::LogError("EcalDetailedTimeRecHitError") << "Error! can't get the product " << recoVertex_;
		}
	      }
	    else
	      {
		edm::Handle<SimVertexContainer> VertexHandle;
		evt.getByLabel(simVertex_, VertexHandle);
		
		if ( VertexHandle.isValid() )
		  {
		    if ((*VertexHandle).size()>0) //at least 1 vertex
		      {
			assert ((*VertexHandle)[0].vertexId() == 0);
			const SimVertex* myVertex= &(*VertexHandle)[0];
			vertex=new GlobalPoint(myVertex->position().x(),myVertex->position().y(),myVertex->position().z());
		      }
		  }
		else {
		  edm::LogError("EcalDetailedTimeRecHitError") << "Error! can't get the product " << simVertex_;
		}
	      }
	  }

        if (EBRecHits && ebTimeDigis) {
                // loop over uncalibrated rechits to make calibrated ones
                for(EBRecHitCollection::const_iterator it  = EBRecHits->begin(); it != EBRecHits->end(); ++it) {
		  EcalRecHit aHit( (*it) );
		  EcalTimeDigiCollection::const_iterator timeDigi=ebTimeDigis->find((*it).id());
		  if (timeDigi!=ebTimeDigis->end())
		    {
		      if (timeDigi->sampleOfInterest()>=0)
			{
			  float myTime=(*timeDigi)[timeDigi->sampleOfInterest()];
			  //Vertex corrected ToF
			  if (vertex)
			    {
			      aHit.setTime(myTime+deltaTimeOfFlight(*vertex,(*it).id(),ebTimeLayer_));
			    }
			  else
			    //Uncorrected ToF
			    aHit.setTime(myTime);
			}
		    }
		    // leave standard time if no timeDigi is associated (e.g. noise recHits)
		  EBDetailedTimeRecHits->push_back( aHit );
                }
        }

        if (EERecHits && eeTimeDigis)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(EERecHitCollection::const_iterator it  = EERecHits->begin();
                                it != EERecHits->end(); ++it) {
			
		  EcalRecHit aHit( *it );
		  EcalTimeDigiCollection::const_iterator timeDigi=eeTimeDigis->find((*it).id());
		  if (timeDigi!=eeTimeDigis->end())
		    {
		      if (timeDigi->sampleOfInterest()>=0)
			{
			  float myTime=(*timeDigi)[timeDigi->sampleOfInterest()];
			  //Vertex corrected ToF
			  if (vertex)
			    {
			      aHit.setTime(myTime+deltaTimeOfFlight(*vertex,(*it).id(),eeTimeLayer_));
			    }
			  else
			    //Uncorrected ToF
			    aHit.setTime(myTime);
			}
		    }
		  EEDetailedTimeRecHits->push_back( aHit );
                }
        }

        if (EKRecHits && ekTimeDigis)
        {
                // loop over uncalibrated rechits to make calibrated ones
                for(EKRecHitCollection::const_iterator it  = EKRecHits->begin();
                                it != EKRecHits->end(); ++it) {
			
		  EcalRecHit aHit( *it );
		  EcalTimeDigiCollection::const_iterator timeDigi=ekTimeDigis->find((*it).id());
		  if (timeDigi!=ekTimeDigis->end())
		    {
		      if (timeDigi->sampleOfInterest()>=0)
			{
			  float myTime=(*timeDigi)[timeDigi->sampleOfInterest()];
			  //Vertex corrected ToF
			  if (vertex)
			    {
			      aHit.setTime(myTime+deltaTimeOfFlight(*vertex,(*it).id(),ekTimeLayer_));
//			      std::cout << " uncorr time: " << myTime << "  corrected: " << myTime+deltaTimeOfFlight(*vertex,(*it).id(),ekTimeLayer_)
//			                << "  old rechit: " << (*it).time() << std::endl;
			    }
			  else
			    //Uncorrected ToF
			    aHit.setTime(myTime);
			}
		    }
		  EKDetailedTimeRecHits->push_back( aHit );
                }
        }


        // put the collection of recunstructed hits in the event   
        LogInfo("EcalDetailedTimeRecHitInfo") << "total # EB rechits: " << EBDetailedTimeRecHits->size();
        LogInfo("EcalDetailedTimeRecHitInfo") << "total # EE rechits: " << EEDetailedTimeRecHits->size();
//        std::cout << "[EcalDetailedTimeRecHitInfo]" 
        LogInfo("EcalDetailedTimeRecHitInfo") << "Storing total # EK rechits: " << EKDetailedTimeRecHits->size() << std::endl;

        evt.put( EBDetailedTimeRecHits, EBDetailedTimeRecHitCollection_ );
        evt.put( EEDetailedTimeRecHits, EEDetailedTimeRecHitCollection_ );
        evt.put( EKDetailedTimeRecHits, EKDetailedTimeRecHitCollection_ );

	if (vertex)
	  delete vertex;
}

double EcalDetailedTimeRecHitProducer::deltaTimeOfFlight( GlobalPoint& vertex, const DetId& detId , int layer) const 
{
  const CaloCellGeometry* cellGeometry ( m_geometry->getGeometry( detId ) ) ;
  assert( 0 != cellGeometry ) ;
  GlobalPoint layerPos = (dynamic_cast<const TruncatedPyramid*>(cellGeometry))->getPosition( double(layer)+0.5 ); //depth in mm in the middle of the layer position
  GlobalVector tofVector = layerPos-vertex;
  return (layerPos.mag()*cm-tofVector.mag()*cm)/(float)c_light ;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( EcalDetailedTimeRecHitProducer );
