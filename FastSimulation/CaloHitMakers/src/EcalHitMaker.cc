#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/EcalBarrelAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalEndcapAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/CaloDirection.h"

#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CaloGeometryTools/interface/CrystalWindowMap.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloDirectionOperations.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"

#include "FastSimulation/Calorimetry/interface/FamosDebug.h"

#include <algorithm>

EcalHitMaker::EcalHitMaker(CaloGeometryHelper * theCalo,
			   const HepPoint3D& ecalentrance, 
			   const DetId& cell, int onEcal,
			   unsigned size, unsigned showertype):
  CaloHitMaker(theCalo,DetId::Ecal,((onEcal==1)?EcalBarrel:EcalEndcap),onEcal,showertype),
  EcalEntrance_(ecalentrance),onEcal_(onEcal),myTrack_(NULL)
{
#ifdef FAMOSDEBUG
  myHistos = Histos::instance();
#endif
  //  myHistos->debug("Constructeur EcalHitMaker");
  X0depthoffset_ = 0. ;
  X0PS1_ = 0.;
  X0PS2_ = 0.; 
  X0ECAL_ = 0.;
  X0EHGAP_ = 0.;
  X0HCAL_ = 0.;
  L0PS1_ = 0.;
  L0PS2_ = 0.;
  L0ECAL_ = 0.;
  L0EHGAP_ = 0.;
  L0HCAL_ = 0.;
  maxX0_ = 0.;
  totalX0_ = 0;
  totalL0_ = 0. ;
  if(onEcal) 
    myCalorimeter->buildCrystal(cell,pivot_);
  else
    pivot_=Crystal();
  central_=onEcal==1;
  ecalFirstSegment_=-1;
  
  myCrystalWindowMap_ = 0; 
  // In some cases, a "dummy" grid, not based on a cell, can be built. The previous variables
  // should however be initialized. In such a case onEcal=0
  if(!onEcal) return;

  // Same size in eta-phi
  etasize_ = size;
  phisize_ = size;



#ifdef FAMOSDEBUG
  myHistos = Histos::instance();
#endif


  // Build the grid
  // The result is put in CellsWindow and is ordered by distance to the pivot
  myCalorimeter->getWindow(pivot_.getDetId(),size,size,CellsWindow_);

  buildGeometry();
  //  std::cout << " Geometry built " << regionOfInterest_.size() << std::endl;

  truncatedGrid_ = CellsWindow_.size()!=(etasize_*phisize_);
  
#ifdef DEBUGGW
  myHistos->fill("h10",EcalEntrance_.eta(),CellsWindow_.size());
  if(onEcal==2) 
    {
      myHistos->fill("h20",EcalEntrance_.perp(),CellsWindow_.size());
      if(EcalEntrance_.perp()>70&&EcalEntrance_.perp()<80&&CellsWindow_.size()<35)
	{
	  std::cout << " Truncated grid " << CellsWindow_.size() << " " << EcalEntrance_.perp() << std::endl;
	  std::cout << " Pivot " << myCalorimeter->getEcalEndcapGeometry()->getGeometry(pivot_.getDetId())->getPosition().perp();
	  std::cout << EEDetId(pivot_.getDetId()) << std::endl;

	  std::cout << " Test getClosestCell " << EcalEntrance_ << std::endl;
	  DetId testcell = myCalorimeter->getClosestCell(EcalEntrance_, true, false);
	  std::cout << " Result "<< EEDetId(testcell) << std::endl;
	  std::cout << " Position " << myCalorimeter->getEcalEndcapGeometry()->getGeometry(testcell)->getPosition() << std::endl;
	}
    }

#endif

}

EcalHitMaker::~EcalHitMaker()
{
  if (myCrystalWindowMap_ != 0)
    {
      delete myCrystalWindowMap_;
    }
}

bool EcalHitMaker::addHitDepth(double r,double phi,double depth)
{
  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.getDetId().rawId());
  if(itcheck==hitMap_.end())
    {
      hitMap_.insert(std::pair<uint32_t,float>(pivot_.getDetId().rawId(),spotEnergy));
    }
  else
    {
      itcheck->second+=spotEnergy;
    }
  return true;
}


bool EcalHitMaker::addHit(double r,double phi,unsigned layer)
{
  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.getDetId().rawId());
  if(itcheck==hitMap_.end())
    {
      hitMap_.insert(std::pair<uint32_t,float>(pivot_.getDetId().rawId(),spotEnergy));
    }
  else
    {
      itcheck->second+=spotEnergy;
    }
  return true;
}


void EcalHitMaker::setTrackParameters(const HepNormal3D& normal,
				   double X0depthoffset,
				   const FSimTrack& theTrack)
{
  //  myHistos->debug("setTrackParameters");
  intersections_.clear();
  myTrack_=&theTrack;
  normal_=normal.unit();
  X0depthoffset_=X0depthoffset;
  cellLine(intersections_);
  buildSegments(intersections_);
//  std::cout << " Segments " << segments_.size() << std::endl;
//  for(unsigned ii=0; ii<segments_.size() ; ++ii)
//    {
//      std::cout << segments_[ii] << std::endl;
//    }
}


void EcalHitMaker::cellLine(std::vector<CaloPoint>& cp) 
{
  cp.clear();
  //  if(myTrack->onVFcal()!=2)
  //    {
  if(!central_&&onEcal_) preshowerCellLine(cp);
  if(onEcal_)ecalCellLine(EcalEntrance_,EcalEntrance_+normal_,cp);
  //    }
  
  hcalCellLine(cp);
  //sort the points by distance (in the ECAL they are not necessarily ordered)
  HepPoint3D vertex(myTrack_->vertex().position());
  CaloPoint::DistanceToVertex myDistance(vertex);
  sort(cp.begin(),cp.end(),myDistance);
}


void EcalHitMaker::preshowerCellLine(std::vector<CaloPoint>& cp) const
{
  //  FSimEvent& mySimEvent = myEventMgr->simSignal();
  //  FSimTrack myTrack = mySimEvent.track(fsimtrack_);
  //  std::cout << "FsimTrack " << fsimtrack_<< std::endl;
  //  std::cout << " On layer 1 " << myTrack.onLayer1() << std::endl;
  // std::cout << " preshowerCellLine " << std::endl; 
  if(myTrack_->onLayer1())
    {
      HepPoint3D point1=(myTrack_->layer1Entrance().vertex()).vect();
      double phys_eta=myTrack_->layer1Entrance().eta();
      double cmthickness = 
	myCalorimeter->layer1Properties(1)->thickness(phys_eta);

      if(cmthickness>0)
	{
	  HepVector3D dir=myTrack_->layer1Entrance().vect().unit();
	  HepPoint3D point2=point1+dir*cmthickness;
	  
	  CaloPoint cp1(DetId::Ecal,EcalPreshower,1,point1);
	  CaloPoint cp2(DetId::Ecal,EcalPreshower,1,point2);
	  cp.push_back(cp1);
	  cp.push_back(cp2);
	}
      else
	{
	  //      std::cout << "Track on ECAL " << myTrack.EcalEntrance_().vertex()*0.1<< std::endl;
	}
    }

  //  std::cout << " On layer 2 " << myTrack.onLayer2() << std::endl;
  if(myTrack_->onLayer2())
    {
      HepPoint3D point1=(myTrack_->layer2Entrance().vertex()).vect();
      double phys_eta=myTrack_->layer2Entrance().eta();
      double cmthickness = 
	myCalorimeter->layer2Properties(1)->thickness(phys_eta);
      if(cmthickness>0)
	{
	  HepVector3D dir=myTrack_->layer2Entrance().vect().unit();
	  HepPoint3D point2=point1+dir*cmthickness;
	  

	  CaloPoint cp1(DetId::Ecal,EcalPreshower,2,point1);
	  CaloPoint cp2(DetId::Ecal,EcalPreshower,2,point2);

	  cp.push_back(cp1);
	  cp.push_back(cp2);
	}
      else
	{
	  //      std::cout << "Track on ECAL " << myTrack.EcalEntrance_().vertex()*0.1 << std::endl;
	}
    }
  //  std::cout << " Exit preshower CellLine " << std::endl;
}

void EcalHitMaker::hcalCellLine(std::vector<CaloPoint>& cp) const
{
  //  FSimEvent& mySimEvent = myEventMgr->simSignal();
  //  FSimTrack myTrack = mySimEvent.track(fsimtrack_);
  int onHcal=myTrack_->onHcal();
  if(onHcal<=2&&onHcal>0)
    {
      HepPoint3D point1=(myTrack_->hcalEntrance().vertex()).vect();

      double eta=point1.eta();
      // HCAL thickness in cm (assuming that the particle is coming from 000)
      double thickness= myCalorimeter->hcalProperties(onHcal)->thickness(eta);
      cp.push_back(CaloPoint(DetId::Hcal,point1));
      HepVector3D dir=myTrack_->hcalEntrance().vect().unit();
      HepPoint3D point2=point1+dir*thickness;

      cp.push_back(CaloPoint(DetId::Hcal,point2));
    }
  int onVFcal=myTrack_->onVFcal();
  if(onVFcal==2)
    {
      HepPoint3D point1=(myTrack_->vfcalEntrance().vertex()).vect();
      double eta=point1.eta();
      // HCAL thickness in cm (assuming that the particle is coming from 000)
      double thickness= myCalorimeter->hcalProperties(3)->thickness(eta);
      cp.push_back(CaloPoint(DetId::Hcal,point1));
      HepVector3D dir=myTrack_->vfcalEntrance().vect().unit();
      if(thickness>0)
	{
	  HepPoint3D point2=point1+dir*thickness;
	  cp.push_back(CaloPoint(DetId::Hcal,point2));
	}
    }
}

void EcalHitMaker::ecalCellLine(const HepPoint3D& a,const HepPoint3D& b,std::vector<CaloPoint>& cp) const
{
  std::vector<HepPoint3D> corners;
  corners.resize(4);
  unsigned ic=0;
  double t;
  HepPoint3D xp;
  DetId c_entrance,c_exit;
  bool entrancefound(false),exitfound(false);
  //  std::cout << " Look for intersections " << ncrystals_ << std::endl;
  //  std::cout << " regionOfInterest_ " << truncatedGrid_ << " " << regionOfInterest_.size() << std::endl;
  // try to determine the number of crystals to test
  // First determine the incident angle
  double angle=acos(normal_.dot(regionOfInterest_[0].getAxis().unit()));

  //  std::cout << " Normal " << normal_<< " Axis " << regionOfInterest_[0].getAxis().unit() << std::endl;
  double backdistance=regionOfInterest_[0].getAxis().mag()*tan(angle);
  // 1/2.2cm = 0.45
//   std::cout << " Angle " << angle << std::endl;
//   std::cout << " Back distance " << backdistance << std::endl;
  unsigned ncrystals=(unsigned)(backdistance*0.45);
  unsigned highlim=(ncrystals+4);
  highlim*=highlim;
  if(highlim>ncrystals_) highlim=ncrystals_;
  //  unsigned lowlim=(ncrystals>2)? (ncrystals-2):0;
  //  std::cout << " Ncrys " << ncrystals << std::endl;
  
  
  while(ic<ncrystals_&&(ic<highlim||!exitfound))
    {
      // Check front side
      //      if(!entrancefound)
	{
	  HepPlane3D plan=regionOfInterest_[ic].getFrontPlane();
	  HepVector3D axis1=(plan.normal());
	  HepVector3D axis2=regionOfInterest_[ic].getFirstEdge();
	  xp=intersect(plan,a,b,t,false);
	  regionOfInterest_[ic].getFrontSide(corners);
	  CrystalPad pad(9999,onEcal_,corners,regionOfInterest_[ic].getCorner(0),axis1,axis2);
	  if(pad.globalinside(xp)) 
	    {
	      cp.push_back(CaloPoint(regionOfInterest_[ic].getDetId(),UP,xp));
	      entrancefound=true;
	      c_entrance=regionOfInterest_[ic].getDetId();
	      //      myHistos->fill("j12",highlim,ic);
	    }
	}
      
      // check rear side
	//	if(!exitfound)
	{
	  HepPlane3D plan=regionOfInterest_[ic].getBackPlane();
	  HepVector3D axis1=(plan.normal());
	  HepVector3D axis2=regionOfInterest_[ic].getFifthEdge();
	  xp=intersect(plan,a,b,t,false);
	  regionOfInterest_[ic].getBackSide(corners);
	  CrystalPad pad(9999,onEcal_,corners,regionOfInterest_[ic].getCorner(4),axis1,axis2);
	  if(pad.globalinside(xp)) 
	    {
	      cp.push_back(CaloPoint(regionOfInterest_[ic].getDetId(),DOWN,xp));
	      exitfound=true;
	      c_exit=regionOfInterest_[ic].getDetId();
	      //	      std::cout << " Crystal : " << ic << std::endl;
	      //	      myHistos->fill("j10",highlim,ic);
	    }
	}
      
      if(entrancefound&&exitfound&&c_entrance==c_exit) return;
      // check lateral sides 
      for(unsigned iside=0;iside<4;++iside)
	{
	  HepPlane3D plan=regionOfInterest_[ic].getLateralPlane(iside);
	  xp=intersect(plan,a,b,t,false);
	  HepVector3D axis1=(plan.normal());
	  HepVector3D axis2=regionOfInterest_[ic].getLateralEdge(iside);
	  regionOfInterest_[ic].getLateralSide(iside,corners);
	  CrystalPad pad(9999,onEcal_,corners,regionOfInterest_[ic].getCorner(iside),axis1,axis2);
	  if(pad.globalinside(xp)) 
	    {
	      cp.push_back(CaloPoint(regionOfInterest_[ic].getDetId(),CaloDirectionOperations::Side(iside),xp)); 
	      //	      std::cout << cp[cp.size()-1] << std::endl;
	    }	  
	}
      // Go to next crystal 
      ++ic;
    }    
}


void EcalHitMaker::buildSegments(const std::vector<CaloPoint>& cp)
{
  //  myHistos->debug();
  //  TimeMe theT("FamosGrid::buildSegments");
  unsigned size=cp.size();
  //  std::cout << " Starting building segment " << size << std::endl;
  if(size%2!=0) 
    {
      //      std::cout << " There is a problem " << std::endl;
      return;
    }
  //  myHistos->debug();
  unsigned nsegments=size/2;
  if (size==0) return;
  // curv abs
  double s=0.;
  double sX0=0.;
  double sL0=0.;
  
  unsigned ncrossedxtals = 0;
  unsigned is=0;
  while(is<nsegments)
    {

      if(cp[2*is].getDetId()!=cp[2*is+1].getDetId()&&
	 cp[2*is].whichDetector()!=DetId::Hcal&&
	 cp[2*is+1].whichDetector()!=DetId::Hcal) 
	{
//	  std::cout << " Problem with the segments " << std::endl;
//	  std::cout << cp[2*is].whichDetector() << " " << cp[2*is+1].whichDetector() << std::endl;
//	  std::cout << is << " " <<cp[2*is].getDetId() << std::endl; 
//	  std::cout << (2*is+1) << " " <<cp[2*is+1].getDetId() << std::endl; 
	  ++is;
	  continue;
	}

      // Check if it is a Preshower segment - Layer 1 
      // One segment per layer, nothing between
      //      myHistos->debug("Just avant Preshower"); 
      if(cp[2*is].whichDetector()==DetId::Ecal && cp[2*is].whichSubDetector()==EcalPreshower && cp[2*is].whichLayer()==1)
	{
	  if(cp[2*is+1].whichDetector()==DetId::Ecal && cp[2*is+1].whichSubDetector()==EcalPreshower && cp[2*is+1].whichLayer()==1)
	    {
	      CaloSegment preshsegment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::PS,myCalorimeter);
	      segments_.push_back(preshsegment);
	      //	      std::cout << " Added (1-1)" << preshsegment << std::endl;
              s+=preshsegment.length();
	      sX0+=preshsegment.X0length();
	      sL0+=preshsegment.L0length();
	      X0PS1_+=preshsegment.X0length();
	      L0PS1_+=preshsegment.L0length();
	    }
	  else
	    {
	      std::cout << " Strange segment between Preshower1 and " << cp[2*is+1].whichDetector();
	      std::cout << std::endl;
	    }
	  ++is;
	  continue;
	}

      // Check if it is a Preshower segment - Layer 2 
      // One segment per layer, nothing between
      if(cp[2*is].whichDetector()==DetId::Ecal && cp[2*is].whichSubDetector()==EcalPreshower && cp[2*is].whichLayer()==2)
	{
	  if(cp[2*is+1].whichDetector()==DetId::Ecal && cp[2*is+1].whichSubDetector()==EcalPreshower && cp[2*is+1].whichLayer()==2)
	    {
	      CaloSegment preshsegment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::PS,myCalorimeter);
	      segments_.push_back(preshsegment);
	      //	      std::cout << " Added (1-2)" << preshsegment << std::endl;
              s+=preshsegment.length();
	      sX0+=preshsegment.X0length();
	      sL0+=preshsegment.L0length();
	      X0PS2_+=preshsegment.X0length();
	      L0PS2_+=preshsegment.L0length();
	    }
	  else
	    {
	      std::cout << " Strange segment between Preshower2 and " << cp[2*is+1].whichDetector();
	      std::cout << std::endl;
	    }
	  ++is;
	  continue;
	}

      // Now deal with the ECAL
      // One segment in each crystal. Segment corresponding to cracks/gaps are added
      //      myHistos->debug("Just avant ECAL"); 
      if(cp[2*is].whichDetector()==DetId::Ecal && (cp[2*is].whichSubDetector()==EcalBarrel || cp[2*is].whichSubDetector()==EcalEndcap))
	{
	  if(cp[2*is+1].whichDetector()==DetId::Ecal && (cp[2*is+1].whichSubDetector()==EcalBarrel || cp[2*is+1].whichSubDetector()==EcalEndcap) )
	    {
	      DetId cell2=cp[2*is+1].getDetId();
	      // set the real entrance
	      if (ecalFirstSegment_<0) ecalFirstSegment_=segments_.size();

	      // !! Approximatiom : the first segment is always in a crystal
	      if(cp[2*is].getDetId()==cell2)
		{
		  CaloSegment segment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::PbWO4,myCalorimeter);
		 segments_.push_back(segment);
		 // std::cout << " Added (2)" << segment << std::endl;
		 s+=segment.length();
		 sX0+=segment.X0length();
		 sL0+=segment.L0length(); 
		 X0ECAL_+=segment.X0length();
		 L0ECAL_+=segment.L0length();
		 ++ncrossedxtals;
		 ++is; 
		}
	      else
		{
		  std::cout << " One more bug in the segment " <<std::endl;
		  ++is;
		}
	      // Now check if a gap or crack should be added
	      if(is<nsegments)
		{		  
		  //		  DetId cell3=cp[2*is].getDetId();
		  if(cp[2*is].whichDetector()!=DetId::Hcal) 
		    {
		      // Crack inside the ECAL
		      //		      bool bordercrossing=myCalorimeter->borderCrossing(cell2,cell3);
		      bool bordercrossing=false;
		      CaloSegment cracksegment(cp[2*is-1],cp[2*is],s,sX0,sL0,(bordercrossing)?CaloSegment::CRACK:CaloSegment::GAP,myCalorimeter);
		      segments_.push_back(cracksegment);
		      s+=cracksegment.length();
		      sX0+=cracksegment.X0length();
		      sL0+=cracksegment.L0length();
		      X0ECAL_+=cracksegment.X0length();
		      L0ECAL_+=cracksegment.L0length();
		      //   std::cout <<" Added(3) "<< cracksegment << std::endl;
		    }
		  else
		    {
		      // a segment corresponding to ECAL/HCAL transition should be
		      // added here
		      CaloSegment cracksegment(cp[2*is-1],cp[2*is],s,sX0,sL0,CaloSegment::ECALHCALGAP,myCalorimeter);
		      segments_.push_back(cracksegment);
		      s+=cracksegment.length();
		      sX0+=cracksegment.X0length();
		      sL0+=cracksegment.L0length();
		      X0EHGAP_+=cracksegment.X0length();
		      L0EHGAP_+=cracksegment.L0length();
		    }
		}
	      continue;
	    }
	  else
	    {
	      std::cout << " Strange segment between " << cp[2*is].whichDetector();
	      std::cout << " and " << cp[2*is+1].whichDetector() << std::endl;
	      ++is;
	      continue;
	    }	  
	}
      //      myHistos->debug("Just avant HCAL"); 
      // HCAL
      if(cp[2*is].whichDetector()==DetId::Hcal&&cp[2*is+1].whichDetector()==DetId::Hcal)
	{
	  CaloSegment segment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::HCAL,myCalorimeter);
	  segments_.push_back(segment);
	  s+=segment.length();
	  sX0+=segment.X0length();
	  sL0+=segment.L0length(); 
	  X0HCAL_+=segment.X0length();
	  L0HCAL_+=segment.L0length();
	  //	  std::cout <<" Added(4) "<< segment << std::endl;
	  ++is; 
	}
    }
//  std::cout << " PS1 " << X0PS1_ << " " << L0PS1_ << std::endl;
//  std::cout << " PS2 " << X0PS2_ << " " << L0PS2_ << std::endl;
//  std::cout << " ECAL " << X0ECAL_ << " " << L0ECAL_ << std::endl;
//  std::cout << " HCAL " << X0HCAL_ << " " << L0HCAL_ << std::endl;
  
  totalX0_ = X0PS1_+X0PS2_+X0ECAL_+X0EHGAP_+X0HCAL_;
  totalL0_ = L0PS1_+L0PS2_+L0ECAL_+L0EHGAP_+L0HCAL_;
  //  myHistos->debug("Just avant le fill"); 

  #ifdef DEBUGCELLLINE
  myHistos->fill("h200",fabs(EcalEntrance_.eta()),X0ECAL_);
  myHistos->fill("h210",EcalEntrance_.phi(),X0ECAL_);
  if(X0ECAL_<20)
    myHistos->fill("h212",EcalEntrance_.phi(),X0ECAL_);
//  if(X0ECAL_<1.) 
//    {
//      for(unsigned ii=0; ii<segments_.size() ; ++ii)
//	{
//	  std::cout << segments_[ii] << std::endl;
//	}
//    }
  myHistos->fillByNumber("h30",ncrossedxtals,fabs(EcalEntrance_.eta()),X0ECAL_);
  myHistos->fill("h310",fabs(EcalEntrance_.eta()),X0ECAL_);
  
  #endif
  //  std::cout << " Finished the segments " << std::endl;
    
}

void EcalHitMaker::buildGeometry()
{
  configuredGeometry_ = false;
  ncrystals_ = CellsWindow_.size();
  // create the vector with of pads with the appropriate size
  padsatdepth_.resize(ncrystals_);
  
  // This is fully correct in the barrel. 
  ny_= phisize_;
  nx_=ncrystals_/ny_;
  std::vector<unsigned> empty;
  empty.resize(ny_,0);

  for(unsigned inx=0;inx<(unsigned)nx_;++inx)
    {      
      myCrystalNumberArray_.push_back(empty);
    }

  hits_.resize(ncrystals_,0.);
  regionOfInterest_.clear();
  regionOfInterest_.resize(ncrystals_);
  validPads_.resize(ncrystals_);
  for(unsigned ic=0;ic<ncrystals_;++ic)
    {
      myCalorimeter->buildCrystal(CellsWindow_[ic],regionOfInterest_[ic]);
      regionOfInterest_[ic].setNumber(ic);
      DetIdMap_.insert(std::pair<DetId,unsigned>(CellsWindow_[ic],ic));     
    }
  
  // Computes the map of the neighbours
  myCrystalWindowMap_ = new CrystalWindowMap(myCalorimeter,regionOfInterest_);
}



// depth is in X0 
bool EcalHitMaker::getPads(double depth) 
{
  //std::cout << " New depth " << depth << std::endl;
  // The first time, the relationship between crystals must be calculated
  // but only in the case of EM showers

  if(EMSHOWER && !configuredGeometry_) configureGeometry();


  radiusFactor_ = (EMSHOWER) ? moliereRadius*radiusCorrectionFactor_:interactionLength;
  detailedShowerTail_ = false;
  currentdepth_ = depth+X0depthoffset_;
  
//  if(currentdepth_>maxX0_+ps1TotalX0()+ps2TotalX0()) 
//    {
//      currentdepth_=maxX0_+ps1TotalX0()+ps2TotalX0()-1.; // the -1 is for safety
//      detailedShowerTail_=true;
//    }
    
//  std::cout << " FamosGrid::getQuads " << currentdepth_ << " " << maxX0_ << std::endl;

  ncrackpadsatdepth_=0;
  
  xmin_=ymin_=999;
  xmax_=ymax_=-999;
  double locxmin,locxmax,locymin,locymax;

  // Get the depth of the pivot
  std::vector<CaloSegment>::const_iterator segiterator;
  // First identify the correct segment
  // EM shower 
  if(EMSHOWER)
    segiterator = find_if(segments_.begin(),segments_.end(),CaloSegment::inX0Segment(currentdepth_));
  
  //Hadron shower 
  if(HADSHOWER)
    segiterator = find_if(segments_.begin(),segments_.end(),CaloSegment::inL0Segment(currentdepth_));
  
  if(segiterator==segments_.end()) 
    {
      std::cout << " FamosGrid: Could not go at such depth " << depth << std::endl;
      std::cout << " EMSHOWER " << EMSHOWER << std::endl;
      std::cout << " Track " << *myTrack_ << std::endl;
      return false;
    }
  //  std::cout << *segiterator << std::endl;
  
  if(segiterator->whichDetector()!=DetId::Ecal)
    {
      std::cout << " In  " << segiterator->whichDetector() << std::endl;
      //      buildPreshower();
      return false;
    }
  
  //  std::cout << *segiterator << std::endl;
  // get the position of the origin
  
  HepPoint3D origin;
  if(EMSHOWER)
    origin=segiterator->positionAtDepthinX0(currentdepth_);
  if(HADSHOWER)
    origin=segiterator->positionAtDepthinL0(currentdepth_);

  //  std::cout << " currentdepth_ " << currentdepth_ << " " << origin << std::endl;
  HepVector3D newaxis=pivot_.getFirstEdge().cross(normal_);

//  std::cout  << "Normal " << normal_ << std::endl;
//  std::cout << " New axis " << newaxis << std::endl;

//  std::cout << " ncrystals  " << ncrystals << std::endl;
  plan_ = HepPlane3D(normal_,origin);

  unsigned nquads=0;
  double sign=(central_) ? -1.: 1.;
  HepTransform3D trans(origin,origin+normal_,origin+newaxis,HepPoint3D(0,0,0),HepPoint3D(0.,0.,sign),HepPoint3D(0.,1.,0.));
  for(unsigned ic=0;ic<ncrystals_;++ic)
    {
      //      std::cout << " Building geometry for " << regionOfInterest_[ic].getCellID() << std::endl;
      HepPoint3D a,b;

      //      std::cout << " Origin " << origin << std::endl;

      std::vector<HepPoint3D> corners;
      double dummyt;
      bool hasbeenpulled=false;
      bool behindback=false;
      for(unsigned il=0;il<4;++il)
	{
	  // a is the il-th front corner of the crystal. b is the corresponding rear corner
	  regionOfInterest_[ic].getLateralEdges(il,a,b);
	  
	  // pull the surface if necessary (only in the front of the crystals) 
	  HepPoint3D aprime=a;
	  if(pulled(origin,normal_,a)) 
	    {
	      b=aprime;
	      hasbeenpulled=true;
	    }
	  
	  // compute the intersection. 
	  // Check that the intersection is in the [a,b] segment  if HADSHOWER
	  // if EMSHOWER the intersection is calculated as if the crystals were infinite
	  HepPoint3D xx=(EMSHOWER)?intersect(plan_,a,b,dummyt,false):intersect(plan_,a,b,dummyt,true);
	  
	  if(dummyt>1) behindback=true;
	  //	  std::cout << " Intersect " << il << " " << a << " " << b << " " << plan_ << " " << xx << std::endl;
	  // check that the intersection actually exists 
	  if(xx.mag()!=0)
	    {
	      corners.push_back(xx);
	    }	  
	}    
      //      std::cout << " ncorners " << corners.size() << std::endl;
      if(behindback&&EMSHOWER) detailedShowerTail_=true;
      // If the quad is completly defined. Store it ! 
      if(corners.size()==4)
	{
	  padsatdepth_[ic]=CrystalPad(ic,corners,trans,bfactor_);
	  // Parameter to be tuned
	  if(hasbeenpulled) padsatdepth_[ic].setSurvivalProbability(pulledPadProbability_);
	  validPads_[ic]=true;
	  ++nquads;	
	  // In principle, this should be done after the quads reorganization. But it would cost one more loop
	  //  quadsatdepth_[ic].extrems(locxmin,locxmax,locymin,locymax);
	  //  if(locxmin<xmin_) xmin_=locxmin;
	  //  if(locymin<ymin_) ymin_=locymin;
	  //  if(locxmax>xmax_) xmax_=locxmax;
	  //  if(locymax>ymax_) ymax_=locymax;
	}
      else
	{
	  padsatdepth_[ic]=CrystalPad();
	  validPads_[ic]=false;
	}
    }
  //  std::cout << " Number of quads " << quadsatdepth_.size() << std::endl; 
  //  if(doreorg_)reorganizeQuads();
  //  std::cout << "Finished to reorganize " << std::endl;
  npadsatdepth_=nquads;
  //  std::cout << " prepareCellIDMap " << std::endl;
  

  // Resize the Quads to allow for some numerical inaccuracy 
  // in the "inside" function
  for(unsigned ic=0;ic<ncrystals_;++ic) 
    {
    
      if (!validPads_[ic]) continue;

      if(EMSHOWER) padsatdepth_[ic].resetCorners();

      padsatdepth_[ic].extrems(locxmin,locxmax,locymin,locymax);
      if(locxmin<xmin_) xmin_=locxmin;
      if(locymin<ymin_) ymin_=locymin;
      if(locxmax>xmax_) xmax_=locxmax;
      if(locymax>ymax_) ymax_=locymax;
    }
  
  sizex_=(xmax_-xmin_)/nx_;
  sizey_=(ymax_-ymin_)/ny_;


  // Make sure that sizex_ and sizey_ are set before running prepareCellIDMap
  prepareCrystalNumberArray();

  //  std::cout << " Finished  prepareCellIDMap " << std::endl;
  ncrackpadsatdepth_=crackpadsatdepth_.size();

  return true;
}




void EcalHitMaker::configureGeometry()
{ 
  configuredGeometry_=true;
  for(unsigned ic=0;ic<ncrystals_;++ic)
    {
      //      std::cout << " Building " << cellids_[ic] << std::endl;
      for(unsigned idir=0;idir<8;++idir)
	{
	  //	  std::cout << " Direction " << FamosCrystal::neighbourDirection(idir) << std::endl;
	  unsigned oppdir=CaloDirectionOperations::oppositeDirection(idir);
	  // Is there something else to do ? 
	  // The relationship with the neighbour may have been set previously.
          if(regionOfInterest_[ic].crystalNeighbour(idir).status()>=0) 
	    {
	      //	      std::cout << " Nothing to do " << std::endl;
	      continue ;
	    }

	  const DetId & oldcell(regionOfInterest_[ic].getDetId());
	  CaloDirection dir=CaloDirectionOperations::neighbourDirection(idir);
	  DetId newcell(oldcell);
	  if(!myCalorimeter->move(newcell,dir))
	    {
	      // no neighbour in this direction
	      regionOfInterest_[ic].crystalNeighbour(idir).setStatus(-1);
	      continue;
	    }
	  // Determine the number of this neighbour
	  //	  std::cout << " The neighbour is " << newcell << std::endl;
	  std::map<DetId,unsigned>::const_iterator niter(DetIdMap_.find(newcell));
	  if(niter==DetIdMap_.end())
	    {
	      //	      std::cout << " The neighbour is not in the map " << std::endl;
	      regionOfInterest_[ic].crystalNeighbour(idir).setStatus(-1);
	      continue;
	    }
	  // Now there is a neighbour	  
	  //	  std::cout << " The neighbour is " << niter->second << " " << cellids_[niter->second] << std::endl;
	  regionOfInterest_[ic].crystalNeighbour(idir).setNumber(niter->second);
	  //	  std::cout << " Managed to set crystalNeighbour " << ic << " " << idir << std::endl;
	  //	  std::cout << " Trying  " << niter->second << " " << oppdir << std::endl;
	  regionOfInterest_[niter->second].crystalNeighbour(oppdir).setNumber(ic);
	  //	  std::cout << " Crack/gap " << std::endl;
	  if(myCalorimeter->borderCrossing(oldcell,newcell))
	    {
	      regionOfInterest_[ic].crystalNeighbour(idir).setStatus(1);
	      regionOfInterest_[niter->second].crystalNeighbour(oppdir).setStatus(1);
	      //	      std::cout << " Crack ! " << std::endl;
	    }
	  else
	    {
	      regionOfInterest_[ic].crystalNeighbour(idir).setStatus(0);
	      regionOfInterest_[niter->second].crystalNeighbour(oppdir).setStatus(0);
	      //	      std::cout << " Gap" << std::endl;
	    }	    
	}
    }
    // Magnetic field a la Charlot
  double theta=EcalEntrance_.theta();
  if(theta>M_PI_2) theta=M_PI-theta;
  bfactor_=1./(1.+0.133*theta);
  // the effect of the magnetic field in the EC is currently ignored 
  if(myCalorimeter->magneticField()==0. || !central_) bfactor_=1.;
}

// project fPoint on the plane (original,normal)
bool EcalHitMaker::pulled(const HepPoint3D & origin, const HepNormal3D& normal, HepPoint3D & fPoint) const 
{
  HepVector3D vec1=fPoint-origin;
  // check if fPoint is behind the origin
  double dotproduct=normal*vec1;
  if(dotproduct<=0.) 
    {
      //      std::cout << " Pulled : nothing done " << std::endl;
      return false;
    }
  //norm of normal is 1 
  HepVector3D vec2=vec1-dotproduct*normal;
  //  std::cout << " Pulled : " << fPoint << " " << origin+vec2 -normal<< std::endl;  
  fPoint = origin+vec2-normal;
  //                  ^^^^^^^ security margin when the intersection is computed
  return true;
}


void EcalHitMaker::prepareCrystalNumberArray()
{
  for(unsigned iq=0;iq<npadsatdepth_;++iq)
    {
      if(!validPads_[iq]) continue;
      unsigned d1,d2;
      convertIntegerCoordinates(padsatdepth_[iq].center().x(),padsatdepth_[iq].center().y(),d1,d2);
      myCrystalNumberArray_[d1][d2]=iq;
    }
}

void EcalHitMaker::convertIntegerCoordinates(double x, double y,unsigned &ix,unsigned &iy) const 
{
  int tix=(int)((x-xmin_)/sizex_);
  int tiy=(int)((y-ymin_)/sizey_);
  ix=iy=9999;
  if(tix>=0) ix=(unsigned)tix;
  if(tiy>=0) iy=(unsigned)tiy;
}

