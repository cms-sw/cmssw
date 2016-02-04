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

class HCaloDetIdAssociator: public HDetIdAssociator{
 public:
   HCaloDetIdAssociator():HDetIdAssociator(72, 70 ,0.087),geometry_(0){};
   HCaloDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :HDetIdAssociator(nPhi, nEta, etaBinSize),geometry_(0){};
   
   virtual void setGeometry(const CaloGeometry* ptr){ geometry_ = ptr; };
   
 protected:
   virtual void check_setup()
     {
	HDetIdAssociator::check_setup();
	if (geometry_==0) throw cms::Exception("CaloGeometry is not set");
     };
   
   virtual GlobalPoint getPosition(const DetId& id){
      return geometry_->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
   };
   
   virtual std::set<DetId> getASetOfValidDetIds(){
      std::set<DetId> setOfValidIds;
      const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Calo, 1);
      for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
         setOfValidIds.insert(*it);

      return setOfValidIds;
   };
   
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId& id){
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

   virtual bool insideElement(const GlobalPoint& point, const DetId& id){
      return  geometry_->getSubdetectorGeometry(id)->getGeometry(id)->inside(point);
   };

   const CaloGeometry* geometry_;
};
#endif
