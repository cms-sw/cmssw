//*****************************************************************************
// File:      EgammaRecHitIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, hacked by Sam Harper (ie the ugly stuff is mine)
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace std;

EgammaRecHitIsolation::EgammaRecHitIsolation (double extRadius,
					  double intRadius,
					  double etLow,
					  edm::ESHandle<CaloGeometry> theCaloGeom,
					  CaloRecHitMetaCollectionV* caloHits,
					  DetId::Detector detector) :
  extRadius_(extRadius),
  intRadius_(intRadius),
  etLow_(etLow),
  theCaloGeom_(theCaloGeom) ,  
  caloHits_(caloHits)
{
  //set up the geometry and selector
  const CaloGeometry* caloGeom = theCaloGeom_.product();
  doubleConeSel_ = new CaloDualConeSelector (intRadius_ ,extRadius_, caloGeom, detector);
}

EgammaRecHitIsolation::~EgammaRecHitIsolation ()
{
  delete doubleConeSel_;
}

double EgammaRecHitIsolation::getSum_(const reco::Candidate* emObject,bool returnEt) const
{

  double energySum = 0.;
  if (caloHits_) 
   {
      //Take the SC position
     reco::SuperClusterRef sc = emObject->get<reco::SuperClusterRef>();
     math::XYZPoint theCaloPosition = sc.get()->position();

     GlobalPoint pclu (theCaloPosition.x () ,
		       theCaloPosition.y () ,
		       theCaloPosition.z () );
     //Compute the energy in a cone of 0.4 radius
     std::auto_ptr<CaloRecHitMetaCollectionV> chosen = doubleConeSel_->select(pclu,*caloHits_);
     for (CaloRecHitMetaCollectionV::const_iterator i = chosen->begin () ; 
	  i!= chosen->end () ; 
	  ++i) 
       {
	 
	 double eta = theCaloGeom_.product()->getPosition(i->detid()).eta();
	 double et = i->energy()*sin(2*atan(exp(-eta)));
	 if ( et > etLow_){
	   if(returnEt) energySum+=et;
	   else energySum+=i->energy();
	 }
       }
     
   } 

  return energySum;
}

