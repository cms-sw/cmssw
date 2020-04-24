#ifndef HTrackAssociator_HCaloDetIdAssociator_h
#define HTrackAssociator_HCaloDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    HTrackAssociator
// Class:      HCaloDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
// Modified for ECAL+HCAL by Michal Szleper
//

#include "Calibration/Tools/interface/DetIdAssociator.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

class HCaloDetIdAssociator: public HDetIdAssociator{
 public:
   HCaloDetIdAssociator():HDetIdAssociator(72, 70 ,0.087),geometry_(nullptr){};
   HCaloDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :HDetIdAssociator(nPhi, nEta, etaBinSize),geometry_(nullptr){};
   
   virtual void setGeometry(const CaloGeometry* ptr){ geometry_ = ptr; };
   
 protected:
   void check_setup() override
     {
	HDetIdAssociator::check_setup();
	if (geometry_==nullptr) throw cms::Exception("CaloGeometry is not set");
     };
   
   GlobalPoint getPosition(const DetId& id) override{
      GlobalPoint point = (id.det() == DetId::Hcal) ? 
	((HcalGeometry*)(geometry_->getSubdetectorGeometry(id)))->getPosition(id) : 
	geometry_->getPosition(id);
      return point;
   };
   
   std::set<DetId> getASetOfValidDetIds() override{
      std::set<DetId> setOfValidIds;
      const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Calo, 1);
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);

      return setOfValidIds;
   };
   
   std::vector<GlobalPoint> getDetIdPoints(const DetId& id) override{
      std::vector<GlobalPoint> points;
      if(! geometry_->getSubdetectorGeometry(id)){
	 LogDebug("CaloDetIdAssociator") << "Cannot find sub-detector geometry for " << id.rawId() <<"\n";
      } else {
	 if(! geometry_->getSubdetectorGeometry(id)->getGeometry(id)) {
	    LogDebug("CaloDetIdAssociator") << "Cannot find CaloCell geometry for " << id.rawId() <<"\n";
	 } else {
	    const CaloCellGeometry::CornersVec& cor ( geometry_->getSubdetectorGeometry(id)->getGeometry(id)->getCorners() );
	    points.assign( cor.begin(), cor.end() ) ;
	    points.push_back(getPosition(id));
	 }
      }
      
      return  points;
   };

   bool insideElement(const GlobalPoint& point, const DetId& id) override{
      return  geometry_->getSubdetectorGeometry(id)->getGeometry(id)->inside(point);
   };

   const CaloGeometry* geometry_;
};
#endif
