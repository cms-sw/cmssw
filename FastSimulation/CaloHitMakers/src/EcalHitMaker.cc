#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"

#include <algorithm>

EcalHitMaker::EcalHitMaker(Calorimeter * theCalo,
			   const HepPoint3D& ecalentrance, 
			   const DetId& cell, int onEcal,
			   unsigned size, unsigned showertype):
  CaloHitMaker(theCalo,DetId::Ecal,((onEcal==1)?EcalBarrel:EcalEndcap),onEcal,showertype),
  EcalEntrance_(ecalentrance),myTrack_(NULL)
{
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
  pivot_ = cell;
  central_=onEcal==1;
  ecalFirstSegment_=-1;
}

EcalHitMaker::~EcalHitMaker()
{
  ;
}

bool EcalHitMaker::addHitDepth(double r,double phi,double depth)
{
  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.rawId());
  if(itcheck==hitMap_.end())
    {
      hitMap_.insert(std::pair<uint32_t,float>(pivot_.rawId(),spotEnergy));
    }
  else
    {
      itcheck->second+=spotEnergy;
    }
  return true;
}


bool EcalHitMaker::addHit(double r,double phi,unsigned layer)
{
  std::map<unsigned,float>::iterator itcheck=hitMap_.find(pivot_.rawId());
  if(itcheck==hitMap_.end())
    {
      hitMap_.insert(std::pair<uint32_t,float>(pivot_.rawId(),spotEnergy));
    }
  else
    {
      itcheck->second+=spotEnergy;
    }
  return true;
}


bool EcalHitMaker::getQuads(double depth)
{
  return true;
}

void EcalHitMaker::setTrackParameters(const HepNormal3D& normal,
				   double X0depthoffset,
				   const FSimTrack& theTrack)
{
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
  //  std::cout << " ECALCellLine " << std::endl;      
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
	  
	  CaloPoint cp1(myCalorimeter,"Preshower1",point1);
	  CaloPoint cp2(myCalorimeter,"Preshower1",point2);
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
	  

	  CaloPoint cp1(myCalorimeter,"Preshower2",point1);
	  CaloPoint cp2(myCalorimeter,"Preshower2",point2);

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
      cp.push_back(CaloPoint(myCalorimeter,"HCAL",point1));
      HepVector3D dir=myTrack_->hcalEntrance().vect().unit();
      HepPoint3D point2=point1+dir*thickness;

      cp.push_back(CaloPoint(myCalorimeter,"HCAL",point2));
    }
  int onVFcal=myTrack_->onVFcal();
  if(onVFcal==2)
    {
      HepPoint3D point1=(myTrack_->vfcalEntrance().vertex()).vect();
      double eta=point1.eta();
      // HCAL thickness in cm (assuming that the particle is coming from 000)
      double thickness= myCalorimeter->hcalProperties(3)->thickness(eta);
      cp.push_back(CaloPoint(myCalorimeter,"HCAL",point1));
      HepVector3D dir=myTrack_->vfcalEntrance().vect().unit();
      if(thickness>0)
	{
	  HepPoint3D point2=point1+dir*thickness;
	  cp.push_back(CaloPoint(myCalorimeter,"HCAL",point2));
	}
    }
}

void EcalHitMaker::ecalCellLine(const HepPoint3D& a,const HepPoint3D& b,std::vector<CaloPoint>& cp) const
{
  if(myTrack_->onEcal())
    {
      HepPoint3D point1=(myTrack_->ecalEntrance().vertex()).vect();
      double phys_eta=myTrack_->ecalEntrance().eta();
      double cmthickness = 
	myCalorimeter->ecalProperties(onEcal_)->thickness(phys_eta);
      HepVector3D dir=myTrack_->ecalEntrance().vect().unit();
      HepPoint3D point2=point1+dir*cmthickness;
      cp.push_back(CaloPoint(myCalorimeter,"ECAL",point1));
      cp.push_back(CaloPoint(myCalorimeter,"ECAL",point2));
    }
}


void EcalHitMaker::buildSegments(const std::vector<CaloPoint>& cp)
{
  //  TimeMe theT("FamosGrid::buildSegments");
  unsigned size=cp.size();
  //  std::cout << " Starting building segment " << size << std::endl;
  if(size%2!=0) 
    {
      //      std::cout << " There is a problem " << std::endl;
      return;
    }

  unsigned nsegments=size/2;
  if (size==0) return;
  // curv abs
  double s=0.;
  double sX0=0.;
  double sL0=0.;
  
  unsigned is=0;
  while(is<nsegments)
    {

      if(cp[2*is].getDetId()!=cp[2*is+1].getDetId()&&
	 cp[2*is].whichDetector()!="HCAL"&&
	 cp[2*is+1].whichDetector()!="HCAL") 
	{
//	  std::cout << " Problem with the segments " << std::endl;
//	  std::cout << cp[2*is].whichDetector() << " " << cp[2*is+1].whichDetector() << std::endl;
//	  std::cout << is << " " <<cp[2*is].getCellID() << std::endl; 
//	  std::cout << (2*is+1) << " " <<cp[2*is+1].getCellID() << std::endl; 
	  ++is;
	  continue;
	}

      // Check if it is a Preshower segment - Layer 1 
      // One segment per layer, nothing between
      if(cp[2*is].whichDetector()=="Preshower1")
	{
	  if(cp[2*is+1].whichDetector()=="Preshower1")
	    {
	      CaloSegment preshsegment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::PS);
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
      if(cp[2*is].whichDetector()=="Preshower2")
	{
	  if(cp[2*is+1].whichDetector()=="Preshower2")
	    {
	      CaloSegment preshsegment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::PS);
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
      if(cp[2*is].whichDetector()=="ECAL")
	{
	  if(cp[2*is+1].whichDetector()=="ECAL")
	    {
	      DetId cell2=cp[2*is+1].getDetId();
	      // set the real entrance
	      if (ecalFirstSegment_<0) ecalFirstSegment_=segments_.size();

	      // !! Approximatiom : the first segment is always in a crystal
	      if(cp[2*is].getDetId()==cell2)
		{
		 CaloSegment segment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::PbWO4);
		 segments_.push_back(segment);
		 // std::cout << " Added (2)" << segment << std::endl;
		 s+=segment.length();
		 sX0+=segment.X0length();
		 sL0+=segment.L0length(); 
		 X0ECAL_+=segment.X0length();
		 L0ECAL_+=segment.L0length();
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
		  if(cp[2*is].whichDetector()!="HCAL") 
		    {
		      // Crack inside the ECAL
		      //		      bool bordercrossing=myCalorimeter->borderCrossing(cell2,cell3);
		      bool bordercrossing=false;
		      CaloSegment cracksegment(cp[2*is-1],cp[2*is],s,sX0,sL0,(bordercrossing)?CaloSegment::CRACK:CaloSegment::GAP);
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
		      CaloSegment cracksegment(cp[2*is-1],cp[2*is],s,sX0,sL0,CaloSegment::ECALHCALGAP);
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

      // HCAL
      if(cp[2*is].whichDetector()=="HCAL"&&cp[2*is+1].whichDetector()=="HCAL")
	{
	  CaloSegment segment(cp[2*is],cp[2*is+1],s,sX0,sL0,CaloSegment::HCAL);
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
  //  std::cout << " Finished the segments " << std::endl;
    
}
