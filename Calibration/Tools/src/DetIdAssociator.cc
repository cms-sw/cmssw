// -*- C++ -*-
//
// Package:    HTrackAssociator
// Class:      HDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
// Modified for ECAL+HCAL by:  Michal Szleper
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: DetIdAssociator.cc,v 1.6 2009/04/08 12:34:29 argiro Exp $
//
//


#include "Calibration/Tools/interface/DetIdAssociator.h"


// surfaces is a vector of GlobalPoint representing outermost point on a cylinder
std::vector<GlobalPoint> HDetIdAssociator::getTrajectory( const FreeTrajectoryState& ftsStart,
						    const std::vector<GlobalPoint>& surfaces)
{
   check_setup();
   std::vector<GlobalPoint> trajectory;
   TrajectoryStateOnSurface tSOSDest;
   FreeTrajectoryState ftsCurrent = ftsStart;

   for(std::vector<GlobalPoint>::const_iterator surface_iter = surfaces.begin(); 
       surface_iter != surfaces.end(); surface_iter++) {
      // this stuff is some weird pointer, which destroy itself
      Cylinder *cylinder = new Cylinder(surface_iter->perp(), 
                                        Surface::PositionType(0,0,0),
					Surface::RotationType() ); 
      Plane *forwardEndcap = new Plane(Surface::PositionType(0,0,surface_iter->z()),
				       Surface::RotationType());
      Plane *backwardEndcap = new Plane(Surface::PositionType(0,0,-surface_iter->z()),
					Surface::RotationType());

      
      LogTrace("StartingPoint")<< "Propagate from "<< "\n"
	<< "\tx: " << ftsStart.position().x()<< "\n"
	<< "\ty: " << ftsStart.position().y()<< "\n"
	<< "\tz: " << ftsStart.position().z()<< "\n"
	<< "\tmomentum eta: " << ftsStart.momentum().eta()<< "\n"
	<< "\tmomentum phi: " << ftsStart.momentum().phi()<< "\n"
	<< "\tmomentum: " << ftsStart.momentum().mag()<< "\n";
     
         float tanTheta = ftsCurrent.momentum().perp()/ftsCurrent.momentum().z();
         float corner = surface_iter->perp()/surface_iter->z();
/*
      std::cout<<"Propagate from "<< "\n"
        << "\tx: " << ftsCurrent.position().x()<< "\n"
        << "\ty: " << ftsCurrent.position().y()<< "\n"
        << "\tz: " << ftsCurrent.position().z()<< "\n"
        << "\tz: " << ftsCurrent.position().perp()<< "\n"
        << "\tz: " << tanTheta<<" "<< corner <<"\n"
        << "\tmomentum eta: " << ftsCurrent.momentum().eta()<< "\n"
        << "\tmomentum phi: " << ftsCurrent.momentum().phi()<< "\n"
        << "\tmomentum: " << ftsCurrent.momentum().mag()<<std::endl;
*/ 
      // First propage the track to the cylinder if |eta|<1, othewise to the encap
      // and correct depending on the result
      int ibar = 0;
      if (fabs(tanTheta) > corner)
	{
                   tSOSDest = ivProp_->propagate(ftsCurrent, *cylinder); 
 //                  std::cout<<" Propagate to cylinder "<<std::endl;
        }
      else if(tanTheta > 0.)
	{tSOSDest = ivProp_->propagate(ftsCurrent, *forwardEndcap); ibar=1; }
      else
	{tSOSDest = ivProp_->propagate(ftsCurrent, *backwardEndcap); ibar=-1; }

//       std::cout<<" Trajectory valid? "<<tSOSDest.isValid()<<" First propagation in "<<ibar<<std::endl;

      if(! tSOSDest.isValid() )
      {
// barrel
       if(ibar == 0){ 
           if (tanTheta < 0 ) tSOSDest = ivProp_->propagate( ftsCurrent,*forwardEndcap);
           if (tanTheta >= 0 ) tSOSDest = ivProp_->propagate( ftsCurrent,*backwardEndcap);
       }
         else
         {
                tSOSDest = ivProp_->propagate(ftsCurrent, *cylinder);
         }
      } else
          {
// missed target
            if(abs(ibar) > 0)
            {
              if(tSOSDest.globalPosition().perp() > surface_iter->perp())
              {
                tSOSDest = ivProp_->propagate(ftsCurrent, *cylinder);
              }           
            }
              else
              {
                if (tanTheta < 0 ) tSOSDest = ivProp_->propagate( ftsCurrent,*forwardEndcap);
                if (tanTheta >= 0 ) tSOSDest = ivProp_->propagate( ftsCurrent,*backwardEndcap);
              }
          }


      // If missed the target, propagate to again
//      if ((!tSOSDest.isValid()) && point.perp() > surface_iter->perp())
//	{tSOSDest = ivProp_->propagate(ftsCurrent, *cylinder);std::cout<<" Propagate again 1 "<<std::endl;}
//         std::cout<<" Track is ok after repropagation to cylinder or not? "<<tSOSDest.isValid()<<std::endl;
//      if ((!tSOSDest.isValid()) && ftsStart.momentum().eta()>0. && fabs(ftsStart.momentum().eta())>1.) 
//	{tSOSDest = ivProp_->propagate(ftsStart, *forwardEndcap);std::cout<<" Propagate again 2 "<<std::endl;}
//       std::cout<<" Track is ok after repropagation forward or not? "<<tSOSDest.isValid()<<std::endl;
//      if ((!tSOSDest.isValid()) && ftsStart.momentum().eta()<0.&&fabs(ftsStart.momentum().eta())>1.) 
//	{tSOSDest = ivProp_->propagate(ftsStart, *backwardEndcap);std::cout<<" Propagate again 3 "<<std::endl;}
//       std::cout<<" Track is after repropagation backward ok or not? "<<tSOSDest.isValid()<<std::endl; 


      if (! tSOSDest.isValid()) return trajectory;
      
//      std::cout<<" Propagate reach something"<<std::endl; 
      LogTrace("SuccessfullPropagation") << "Great, I reached something." << "\n"
	<< "\tx: " << tSOSDest.freeState()->position().x() << "\n"
	<< "\ty: " << tSOSDest.freeState()->position().y() << "\n"
	<< "\tz: " << tSOSDest.freeState()->position().z() << "\n"
	<< "\teta: " << tSOSDest.freeState()->position().eta() << "\n"
	<< "\tphi: " << tSOSDest.freeState()->position().phi() << "\n";

//      std::cout<<" The position of trajectory "<<tSOSDest.freeState()->position().perp()<<" "<<tSOSDest.freeState()->position().z()<<std::endl;  

      GlobalPoint point = tSOSDest.freeState()->position(); 
      point = tSOSDest.freeState()->position();
      ftsCurrent = *tSOSDest.freeState();
      trajectory.push_back(point);
   }
   return trajectory;
}

//------------------------------------------------------------------------------
std::set<DetId> HDetIdAssociator::getDetIdsCloseToAPoint(const GlobalPoint& direction,
						   const int idR)
{
   std::set<DetId> set;
   check_setup();
   int nDets=0;
   if (! theMap_) buildMap();
   LogTrace("MatchPoint") << "point (eta,phi): " << direction.eta() << "," << direction.phi() << "\n";
   int ieta = iEta(direction);
   int iphi = iPhi(direction);

   LogTrace("MatchPoint") << "(ieta,iphi): " << ieta << "," << iphi << "\n";
   
   if (ieta>=0 && ieta<nEta_ && iphi>=0 && iphi<nPhi_){
      set = (*theMap_)[ieta][iphi];
      nDets++;
      if (idR>0){
	  LogTrace("MatchPoint") << "Add neighbors (ieta,iphi): " << ieta << "," << iphi << "\n";
	 //add neighbors
	 int maxIEta = ieta+idR;
	 int minIEta = ieta-idR;
	 if(maxIEta>=nEta_) maxIEta = nEta_-1;
	 if(minIEta<0) minIEta = 0;
	 int maxIPhi = iphi+idR;
	 int minIPhi = iphi-idR;
	 if(minIPhi<0) {
	    minIPhi+=nPhi_;
	    maxIPhi+=nPhi_;
	 }
	 LogTrace("MatchPoint") << "\tieta (min,max): " << minIEta << "," << maxIEta<< "\n";
	 LogTrace("MatchPoint") << "\tiphi (min,max): " << minIPhi << "," << maxIPhi<< "\n";
	 for (int i=minIEta;i<=maxIEta;i++)
	   for (int j=minIPhi;j<=maxIPhi;j++) {
	      if( i==ieta && j==iphi) continue; // already in the set
	      set.insert((*theMap_)[i][j%nPhi_].begin(),(*theMap_)[i][j%nPhi_].end());
              nDets++;
	   }
      }
   }
//     if(set.size() > 0) {
//   if (ieta+idR<55 && ieta-idR>14 && set.size() != (2*idR+1)*(2*idR+1)){
//       std::cout<<" RRRA: "<<set.size()<<" DetIds in region "<<ieta<<" "<<iphi<<std::endl;
//       for( std::set<DetId>::const_iterator itr=set.begin(); itr!=set.end(); itr++) {
//         GlobalPoint point = getPosition(*itr);
//         std::cout << "DetId: " <<itr->rawId() <<" (eta,phi): " << point.eta() << "," << point.phi()<<" "<<iEta(point)<<" "<<iPhi(point)<<std::endl;
//       }
//     }
//     else {
//       std::cout <<" HDetIdAssociator::getDetIdsCloseToAPoint::There are strange days "<<std::endl; 
//     }  
   return set;
}

//------------------------------------------------------------------------------
int HDetIdAssociator::iEta (const GlobalPoint& point)
{
// unequal bin sizes for endcap, following HCAL geometry
   int iEta1 = int(point.eta()/etaBinSize_ + nEta_/2);
   if (point.eta()>1.827 && point.eta()<=1.830) return iEta1-1;
   else if (point.eta()>1.914 && point.eta()<=1.930) return iEta1-1;
   else if (point.eta()>2.001 && point.eta()<=2.043) return iEta1-1;
   else if (point.eta()>2.088 && point.eta()<=2.172) return iEta1-1;
   else if (point.eta()>2.175 && point.eta()<=2.262) return iEta1-1;
   else if (point.eta()>2.262 && point.eta()<=2.332) return iEta1-2;
   else if (point.eta()>2.332 && point.eta()<=2.349) return iEta1-1;
   else if (point.eta()>2.349 && point.eta()<=2.436) return iEta1-2;
   else if (point.eta()>2.436 && point.eta()<=2.500) return iEta1-3;
   else if (point.eta()>2.500 && point.eta()<=2.523) return iEta1-2;
   else if (point.eta()>2.523 && point.eta()<=2.610) return iEta1-3;
   else if (point.eta()>2.610 && point.eta()<=2.650) return iEta1-4;
   else if (point.eta()>2.650 && point.eta()<=2.697) return iEta1-3;
   else if (point.eta()>2.697 && point.eta()<=2.784) return iEta1-4;
   else if (point.eta()>2.784 && point.eta()<=2.868) return iEta1-5;
   else if (point.eta()>2.868 && point.eta()<=2.871) return iEta1-4;
   else if (point.eta()>2.871 && point.eta()<=2.958) return iEta1-5;
   else if (point.eta()>2.958) return iEta1-6;
   else if (point.eta()<-1.827 && point.eta()>=-1.830) return iEta1+1;
   else if (point.eta()<-1.914 && point.eta()>=-1.930) return iEta1+1;
   else if (point.eta()<-2.001 && point.eta()>=-2.043) return iEta1+1;
   else if (point.eta()<-2.088 && point.eta()>=-2.172) return iEta1+1;
   else if (point.eta()<-2.175 && point.eta()>=-2.262) return iEta1+1;
   else if (point.eta()<-2.262 && point.eta()>=-2.332) return iEta1+2;
   else if (point.eta()<-2.332 && point.eta()>=-2.349) return iEta1+1;
   else if (point.eta()<-2.349 && point.eta()>=-2.436) return iEta1+2;
   else if (point.eta()<-2.436 && point.eta()>=-2.500) return iEta1+3;
   else if (point.eta()<-2.500 && point.eta()>=-2.523) return iEta1+2;
   else if (point.eta()<-2.523 && point.eta()>=-2.610) return iEta1+3;
   else if (point.eta()<-2.610 && point.eta()>=-2.650) return iEta1+4;
   else if (point.eta()<-2.650 && point.eta()>=-2.697) return iEta1+3;
   else if (point.eta()<-2.697 && point.eta()>=-2.784) return iEta1+4;
   else if (point.eta()<-2.784 && point.eta()>=-2.868) return iEta1+5;
   else if (point.eta()<-2.868 && point.eta()>=-2.871) return iEta1+4;
   else if (point.eta()<-2.871 && point.eta()>=-2.958) return iEta1+5;
   else if (point.eta()<-2.349) return iEta1+6;
   else return iEta1;
}

//------------------------------------------------------------------------------
int HDetIdAssociator::iPhi (const GlobalPoint& point)
{
   double pi=4*atan(1.);
   int iPhi1 = int((double(point.phi())+pi)/(2*pi)*nPhi_);
   return iPhi1;
}

//------------------------------------------------------------------------------
void HDetIdAssociator::buildMap()
{
// modified version: take only detector central position
   check_setup();
   LogTrace("HDetIdAssociator")<<"building map" << "\n";
   if(theMap_) delete theMap_;
   theMap_ = new std::vector<std::vector<std::set<DetId> > >(nEta_,std::vector<std::set<DetId> >(nPhi_));
   int numberOfDetIdsOutsideEtaRange = 0;
   int numberOfDetIdsActive = 0;
   std::set<DetId> validIds = getASetOfValidDetIds();
   for (std::set<DetId>::const_iterator id_itr = validIds.begin(); id_itr!=validIds.end(); id_itr++) {	 
//      std::vector<GlobalPoint> points = getDetIdPoints(*id_itr);
      GlobalPoint point = getPosition(*id_itr);
// reject fake DetIds (eta=0 - what are they anyway???)
      if(point.eta()==0)continue;

      int ieta = iEta(point);
      int iphi = iPhi(point);
      int etaMax(-1);
      int etaMin(nEta_);
      int phiMax(-1);
      int phiMin(nPhi_);
      if ( iphi >= nPhi_ ) iphi = iphi % nPhi_;
      assert (iphi>=0);
      if ( etaMin > ieta) etaMin = ieta;
      if ( etaMax < ieta) etaMax = ieta;
      if ( phiMin > iphi) phiMin = iphi;
      if ( phiMax < iphi) phiMax = iphi;
// for abs(eta)>1.8 one tower covers two phi segments
      if ((ieta>54||ieta<15) && iphi%2==0) phiMax++;
      if ((ieta>54||ieta<15) && iphi%2==1) phiMin--;

      if (etaMax<0||phiMax<0||etaMin>=nEta_||phiMin>=nPhi_) {
	 LogTrace("HDetIdAssociator")<<"Out of range: DetId:" << id_itr->rawId() <<
	   "\n\teta (min,max): " << etaMin << "," << etaMax <<
	   "\n\tphi (min,max): " << phiMin << "," << phiMax <<
	   "\nTower id: " << id_itr->rawId() << "\n";
	 numberOfDetIdsOutsideEtaRange++;
	 continue;
      }
	  
      if (phiMax-phiMin > phiMin+nPhi_-phiMax){
	 phiMin += nPhi_;
	 std::swap(phiMin,phiMax);
      }
      for(int ieta = etaMin; ieta <= etaMax; ieta++)
	for(int iphi = phiMin; iphi <= phiMax; iphi++)
	  (*theMap_)[ieta][iphi%nPhi_].insert(*id_itr);
      numberOfDetIdsActive++;
   }
   LogTrace("HDetIdAssociator") << "Number of elements outside the allowed range ( |eta|>"<<
     nEta_/2*etaBinSize_ << "): " << numberOfDetIdsOutsideEtaRange << "\n";
   LogTrace("HDetIdAssociator") << "Number of active DetId's mapped: " << 
     numberOfDetIdsActive << "\n";
}

//------------------------------------------------------------------------------
std::set<DetId> HDetIdAssociator::getDetIdsInACone(const std::set<DetId>& inset, 
         			     const std::vector<GlobalPoint>& trajectory,
	        		     const double dR)
{
// modified version: if dR<0, returns 3x3 towers around the input one (Michal)
   check_setup();
   std::set<DetId> outset;
  
   if(dR>=0) {
     for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++)
       for(std::vector<GlobalPoint>::const_iterator point_iter = trajectory.begin(); point_iter != trajectory.end(); point_iter++)
         if (nearElement(*point_iter,*id_iter,dR)) outset.insert(*id_iter);
   }
   else {
     if (inset.size()!=1) return outset;
     std::set<DetId>::const_iterator id_inp = inset.begin();
     int ieta;
     int iphi;
     GlobalPoint point = getPosition(*id_inp);
     ieta = iEta(point);
     iphi = iPhi(point);
     for (int i=ieta-1;i<=ieta+1;i++) {
       for (int j=iphi-1;j<=iphi+1;j++) {
//         if( i==ieta && j==iphi) continue;
         if( i<0 || i>=nEta_) continue;
         int j2fill = j%nPhi_;
         if(j2fill<0) j2fill+=nPhi_;
         if((*theMap_)[i][j2fill].size()==0)continue;
         outset.insert((*theMap_)[i][j2fill].begin(),(*theMap_)[i][j2fill].end());
       }
     }
   }

//   if (outset.size() > 0) {
//     std::cout<<" RRRA: DetIds in cone:"<<std::endl;
//     for( std::set<DetId>::const_iterator itr=outset.begin(); itr!=outset.end(); itr++) {
//       GlobalPoint point = getPosition(*itr);
//       std::cout << "DetId: " <<itr->rawId() <<" (eta,phi): " << point.eta() << "," << point.phi()<<std::endl;
//     }
//   }

   return outset;
}

//------------------------------------------------------------------------------
std::set<DetId> HDetIdAssociator::getCrossedDetIds(const std::set<DetId>& inset,
					     const std::vector<GlobalPoint>& trajectory)
{
   check_setup();
   std::set<DetId> outset;
   for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++)
     for(std::vector<GlobalPoint>::const_iterator point_iter = trajectory.begin(); point_iter != trajectory.end(); point_iter++)
       if (insideElement(*point_iter, *id_iter))  outset.insert(*id_iter);
   return outset;
}

//------------------------------------------------------------------------------
std::set<DetId> HDetIdAssociator::getMaxEDetId(const std::set<DetId>& inset,
                                           edm::Handle<CaloTowerCollection> caloTowers)
{
// returns the most energetic tower in the NxN box (Michal)
   check_setup();
   std::set<DetId> outset;
   std::set<DetId>::const_iterator id_max = inset.begin();
   double Ehadmax=0;

   for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++) {
     DetId id(*id_iter);
//     GlobalPoint point = getPosition(*id_iter);
//     int ieta = iEta(point);
//     int iphi = iPhi(point);
     CaloTowerCollection::const_iterator tower = (*caloTowers).find(id);
     if(tower != (*caloTowers).end() && tower->hadEnergy()>Ehadmax) {
       id_max = id_iter;
       Ehadmax = tower->hadEnergy();
     }
   }

   if (Ehadmax > 0) outset.insert(*id_max);

//   if (outset.size() > 0) {
//     std::cout<<" RRRA: Most energetic DetId:"<<std::endl;
//     for( std::set<DetId>::const_iterator itr=outset.begin(); itr!=outset.end(); itr++) {
//       GlobalPoint point = getPosition(*itr);
//       std::cout << "DetId: " <<itr->rawId() <<" (eta,phi): " << point.eta() << "," << point.phi()<<std::endl;
//     }
//   }

   return outset;
}

//------------------------------------------------------------------------------
std::set<DetId> HDetIdAssociator::getMaxEDetId(const std::set<DetId>& inset,
                                           edm::Handle<HBHERecHitCollection> recHits)
{
// returns the most energetic tower in the NxN box - from RecHits (Michal)
   check_setup();
   std::set<DetId> outset;
   std::set<DetId>::const_iterator id_max = inset.begin();
   double Ehadmax=0;

   for(std::set<DetId>::const_iterator id_iter = inset.begin(); id_iter != inset.end(); id_iter++) {
     DetId id(*id_iter);
//     GlobalPoint point = getPosition(*id_iter);
//     int ieta = iEta(point);
//     int iphi = iPhi(point);
     HBHERecHitCollection::const_iterator hit = (*recHits).find(id);
     if(hit != (*recHits).end() && hit->energy()>Ehadmax) {
       id_max = id_iter;
       Ehadmax = hit->energy();
     }
   }

   if (Ehadmax > 0) outset.insert(*id_max);

//   if (outset.size() > 0) {
//     std::cout<<" RRRA: Most energetic DetId:"<<std::endl;
//     for( std::set<DetId>::const_iterator itr=outset.begin(); itr!=outset.end(); itr++) {
//       GlobalPoint point = getPosition(*itr);
//       std::cout << "DetId: " <<itr->rawId() <<" (eta,phi): " << point.eta() << "," << point.phi()<<std::endl;
//     }
//   }

   return outset;
}
