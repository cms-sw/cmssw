#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionFast.h" 
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

CaloConeSelector::CaloConeSelector(double dR, const CaloGeometry* geom) :
  geom_(geom),deltaR_(dR),detector_(DetId::Detector(0)),subdet_(0) {
}

CaloConeSelector::CaloConeSelector(double dR, const CaloGeometry* geom, DetId::Detector detector, int subdet) : 
  geom_(geom),deltaR_(dR),detector_(detector),subdet_(subdet) {
}

std::auto_ptr<CaloRecHitMetaCollectionV> CaloConeSelector::select(double eta, double phi, const CaloRecHitMetaCollectionV& inputCollection) {
  GlobalPoint p(GlobalPoint::Cylindrical(1,phi,tanh(eta)));
  return select(p,inputCollection);
}

std::auto_ptr<CaloRecHitMetaCollectionV> CaloConeSelector::select(const GlobalPoint& p, const CaloRecHitMetaCollectionV& inputCollection) {
  CaloRecHitMetaCollectionFast* c=new CaloRecHitMetaCollectionFast();

  // TODO: handle default setting of detector_ (loops over subdet)
  // TODO: heuristics of when it is better to loop over inputCollection instead (small # hits)
  for (int subdet=subdet_; subdet<=7 && (subdet_==0 || subdet_==subdet); subdet++) {
    const CaloSubdetectorGeometry* sdg=geom_->getSubdetectorGeometry(detector_,subdet);
    if (sdg!=0) {
      // get the list of detids within range (from geometry)
      CaloSubdetectorGeometry::DetIdSet dis=sdg->getCells(p,deltaR_);
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
