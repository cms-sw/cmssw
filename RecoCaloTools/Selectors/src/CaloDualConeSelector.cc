#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionFast.h" 
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include <algorithm>

CaloDualConeSelector::CaloDualConeSelector(double dRmin, double dRmax, const CaloGeometry* geom) :
  geom_(geom),deltaRmin_(dRmin),deltaRmax_(dRmax),detector_(DetId::Detector(0)),subdet_(0) {
}

CaloDualConeSelector::CaloDualConeSelector(double dRmin, double dRmax, const CaloGeometry* geom, DetId::Detector detector, int subdet) : 
  geom_(geom),deltaRmin_(dRmin),deltaRmax_(dRmax),detector_(detector),subdet_(subdet) {
}

std::auto_ptr<CaloRecHitMetaCollectionV> CaloDualConeSelector::select(double eta, double phi, const CaloRecHitMetaCollectionV& inputCollection) {
  GlobalPoint p(GlobalPoint::Cylindrical(1,phi,tanh(eta)));
  return select(p,inputCollection);
}

std::auto_ptr<CaloRecHitMetaCollectionV> CaloDualConeSelector::select(const GlobalPoint& p, const CaloRecHitMetaCollectionV& inputCollection) {
  CaloRecHitMetaCollectionFast* c=new CaloRecHitMetaCollectionFast();

  // TODO: handle default setting of detector_ (loops over subdet)
  // TODO: heuristics of when it is better to loop over inputCollection instead (small # hits)
  for (int subdet=subdet_; subdet<=7 && (subdet_==0 || subdet_==subdet); subdet++) {
    const CaloSubdetectorGeometry* sdg=geom_->getSubdetectorGeometry(detector_,subdet);
    if (sdg!=0) {
      // get the list of detids within range (from geometry)
      CaloSubdetectorGeometry::DetIdSet dis_excl=sdg->getCells(p,deltaRmin_);
      CaloSubdetectorGeometry::DetIdSet dis_all=sdg->getCells(p,deltaRmax_);
      // use set operations to determine detids in annulus
      CaloSubdetectorGeometry::DetIdSet dis;
      std::set_difference(dis_all.begin(),dis_all.end(),
			  dis_excl.begin(),dis_excl.end(),
			  std::inserter(dis,dis.begin()));
      // loop over detids...
      CaloRecHitMetaCollectionV::const_iterator j,je=inputCollection.end();      

      for (CaloSubdetectorGeometry::DetIdSet::iterator i=dis.begin(); i!=dis.end(); i++) {
	if (i->subdetId()!=subdet) continue; // possible for HCAL where the same geometry object handles all the detectors
	j=inputCollection.find(*i);
	if (j!=je) c->add(&(*j));
      }
    }    
  }

  return std::auto_ptr<CaloRecHitMetaCollectionV>(c);
}
