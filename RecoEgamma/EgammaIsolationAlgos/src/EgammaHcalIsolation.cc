//*****************************************************************************
// File:      EgammaHcalIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************
//C++ includes
#include <vector>
#include <functional>

//ROOT includes
#include <Math/VectorUtil.h>

//CMSSW includes
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

using namespace std;

EgammaHcalIsolation::EgammaHcalIsolation (double extRadius,
			    double intRadius,
			    double etLow,
			    edm::ESHandle<CaloGeometry> theCaloGeom ,
			    HBHERecHitMetaCollection*  mhbhe) :
  extRadius_(extRadius),
  intRadius_(intRadius),
  etLow_(etLow),
  theCaloGeom_(theCaloGeom) ,  
  mhbhe_(mhbhe)
{
  //set up the geometry and selector
  const CaloGeometry* caloGeom = theCaloGeom_.product();
  doubleConeSel_ = new CaloDualConeSelector (intRadius_ ,extRadius_, caloGeom, DetId::Hcal);
}

EgammaHcalIsolation::~EgammaHcalIsolation ()
{
  delete doubleConeSel_;
}

double EgammaHcalIsolation::getHcalEtSum (const reco::Candidate* emObject) const
{

  double hcalEt = 0.;
  if (mhbhe_) 
   {
      //Take the SC position
     reco::SuperClusterRef sc = emObject->get<reco::SuperClusterRef>();
     math::XYZPoint theCaloPosition = sc.get()->position();
     //      math::XYZPoint theCaloPosition = (emObject->get<reco::SuperClusterRef>())->position() ;
      GlobalPoint pclu (theCaloPosition.x () ,
                	theCaloPosition.y () ,
			theCaloPosition.z () );
      //Compute the HCAL energy behind ECAL
      std::auto_ptr<CaloRecHitMetaCollectionV> chosen = doubleConeSel_->select(pclu,*mhbhe_);
      for (CaloRecHitMetaCollectionV::const_iterator i = chosen->begin () ; 
                                                     i!= chosen->end () ; 
						     ++i) 
       {
	 double hcalHit_eta = theCaloGeom_.product()->getPosition(i->detid()).eta();
	 double hcalHit_Et = i->energy()*sin(2*atan(exp(-hcalHit_eta)));
	 if ( hcalHit_Et > etLow_)
	      hcalEt += hcalHit_Et;
       }
    } 
  return hcalEt ;
}
