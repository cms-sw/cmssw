#include "FastSimulation/CaloGeometryTools/interface/CrystalPad.h"

#include <iostream> 



CrystalPad::CrystalPad(unsigned number, const std::vector<Hep2Vector>& corners):survivalProbability_(1.),epsilon_(0.001)
{
  number_=number;
  //  std::cout << " Hello " << std::endl;
  if(corners.size()!=4)
    {
      std::cout << " Try to construct a quadrilateral with " << corners.size() << " points ! " << std::endl;     
      dummy_=true;
    }
  else
    {
      dummy_=false;
      // Set explicity the z to 0 !
      center_=Hep2Vector(0.,0.);
      for(unsigned ic=0; ic<4;++ic)
	{
	  corners_.push_back(corners[ic]);
	  Hep2Vector tmp((corners[(ic+1)%4]-corners[ic]).unit());
	  //	  std::cout << " First constructor " << corners_[ic] << std::endl;
	  dir_.push_back(tmp);	  
	  center_+=corners_[ic];
	}
      center_*=0.25;
    }
//  std::cout << " End of 1 constructor " << std::endl;
//  std::cout << " Ncorners " << corners_.size() << std::endl;
//  std::cout << " Ndirs " << dir_.size() << std::endl;
}

CrystalPad::CrystalPad(unsigned number, int onEcal, const std::vector<HepPoint3D>& corners,HepPoint3D origin,HepVector3D vec1,HepVector3D vec2):number_(number),survivalProbability_(1.),epsilon_(0.001)
{
  //  std::cout << " We are in the 2nd constructor " << std::endl;
  if(corners.size()!=4)
    {
      std::cout << " Try to construct a quadrilateral with " << corners.size() << " points ! " << std::endl;     
      dummy_=true;
    }
  else
    {
      dummy_=false;
      double sign=(onEcal==1) ? -1.: 1.;
      center_=Hep2Vector(0.,0.);
      // the good one in the central
      // trans_=HepTransform3D(origin,origin+vec1,origin+vec2,HepPoint3D(0,0,0),HepPoint3D(0.,0.,-1.),HepPoint3D(0.,1.,0.));
      trans_=HepTransform3D(origin,origin+vec1,origin+vec2,HepPoint3D(0,0,0),HepPoint3D(0.,0.,sign),HepPoint3D(0.,1.,0.));
      //      std::cout << " Constructor 2; input corners "  << std::endl;
      for(unsigned ic=0;ic<4;++ic)
	{	
	  HepPoint3D corner=corners[ic];
	  //	  std::cout << corner << " " ;
	  corner.transform(trans_);
	  //	  std::cout << corner << std::endl ;
	  corners_.push_back(Hep2Vector(corner.x(),corner.y()));
	  center_+=corners_[ic];
	}
      for(unsigned ic=0;ic<4;++ic)
	{
	  Hep2Vector tmp((corners_[(ic+1)%4]-corners_[ic]).unit());
	  dir_.push_back(tmp);
	}
      center_*=0.25;
    }  
//  std::cout << " End of 2 constructor " << std::endl;
//  std::cout << " Corners(constructor) " ;
//  std::cout << corners_[0] << std::endl;
//  std::cout << corners_[1] << std::endl;
//  std::cout << corners_[2] << std::endl;
//  std::cout << corners_[3] << std::endl;
}
CrystalPad::CrystalPad(unsigned number, const std::vector<HepPoint3D>& corners,const HepTransform3D & trans,double scaf):number_(number),survivalProbability_(1.),epsilon_(0.001),yscalefactor_(scaf)
{
  //  std::cout << " We are in the 2nd constructor " << std::endl;
  if(corners.size()!=4)
    {
      std::cout << " Try to construct a quadrilateral with " << corners.size() << " points ! " << std::endl;     
      dummy_=true;
    }
  else
    {
      dummy_=false;
      center_=Hep2Vector(0.,0.);
      // the good one in the central
      // trans_=HepTransform3D(origin,origin+vec1,origin+vec2,HepPoint3D(0,0,0),HepPoint3D(0.,0.,-1.),HepPoint3D(0.,1.,0.));
      trans_=trans;
      //      std::cout << " Constructor 2; input corners "  << std::endl;
      for(unsigned ic=0;ic<4;++ic)
	{	
	  HepPoint3D corner=corners[ic];
	  //	  std::cout << corner << " " ;
	  corner.transform(trans_);
	  //	  std::cout << corner << std::endl ;
	  corners_.push_back(Hep2Vector(corner.x(),corner.y()*yscalefactor_));
	  center_+=corners_[ic];
	}
      for(unsigned ic=0;ic<4;++ic)
	{
	  Hep2Vector tmp((corners_[(ic+1)%4]-corners_[ic]).unit());
	  dir_.push_back(tmp);
	}
      center_*=0.25;
    }  
}

bool CrystalPad::inside(const Hep2Vector & ppoint,bool debug) const
{
//  std::cout << "Inside " << ppoint <<std::endl;
//  std::cout << "Corners " << corners_.size() << std::endl;
//  std::cout << corners_[0] << std::endl;
//  std::cout << corners_[1] << std::endl;
//  std::cout << corners_[2] << std::endl;
//  std::cout << corners_[3] << std::endl;
//  std::cout << " Got the 2D point " << std::endl;
  Hep2Vector pv0(ppoint-corners_[0]);
  Hep2Vector pv2(ppoint-corners_[2]);
  Hep2Vector n1(pv0-(pv0*dir_[0])*dir_[0]);
  Hep2Vector n2(pv2-(pv2*dir_[2])*dir_[2]);

  //  double N1(n1.mag());
  //  double N2(n2.mag());
  double r1(n1*n2);
  bool inside1(r1<=0.);

  if (!inside1) return false;

//  if(debug) 
//    {
//      std::cout << n1 << std::endl;
//      std::cout << n2 << std::endl;
//      std::cout << r1 << std::endl;
//      std::cout << inside1 << std::endl;
//    }

//  bool close1=(N1<epsilon_||N2<epsilon_);
//  
//  if(!close1&&!inside1) return false;
  //  std::cout << " First calculation " << std::endl;
  Hep2Vector pv1(ppoint-corners_[1]);
  Hep2Vector pv3(ppoint-corners_[3]);
  Hep2Vector n3(pv1-(pv1*dir_[1])*dir_[1]);
  Hep2Vector n4(pv3-(pv3*dir_[3])*dir_[3]);
  //  double N3(n3.mag());
  //  double N4(n4.mag());
  //  bool close2=(N3<epsilon_||N4<epsilon_);
  double r2(n3*n4);
  bool inside2(r2<=0.);
//  //  std::cout << " pv1 & pv3 " << pv1.mag() << " " << pv3.mag() << std::endl;
//  //  double tmp=(pv1-(pv1*dir_[1])*dir_[1])*(pv3-(pv3*dir_[3])*dir_[3]);
//  //  std::cout << " Computed tmp " << tmp << std::endl;
//  if(debug) 
//    {
//      std::cout << n3 << std::endl;
//      std::cout << n4 << std::endl;
//      std::cout << r2 << std::endl;
//      std::cout << inside2 << std::endl;
//    }
  //  if(!close2&&!inside2) return false;
//  std::cout << " Second calculation " << std::endl;
//  std::cout << " True " << std::endl;
  //    return (!close1&&!close2||(close2&&inside1||close1&&inside2));

  return inside2;
}

bool CrystalPad::globalinside(HepPoint3D point) const
{
  //  std::cout << " Global inside " << std::endl;
  //  std::cout << point << " " ;
  point.transform(trans_);
  //  std::cout << point << std::endl;
  //  print();
  Hep2Vector ppoint(point.x(),point.y());
  bool result=inside(ppoint);
  //  std::cout << " Result " << result << std::endl;
  return result;
}

void CrystalPad::print() const
{
  std::cout << " Corners " << std::endl;
  std::cout << corners_[0] << std::endl;
  std::cout << corners_[1] << std::endl;
  std::cout << corners_[2] << std::endl;
  std::cout << corners_[3] << std::endl;
}

Hep2Vector CrystalPad::localPoint(HepPoint3D point) const
{
  point.transform(trans_);
  return Hep2Vector(point);
}

Hep2Vector& CrystalPad::edge(unsigned iside,int n) 
{
  return corners_[(iside+n)%4];
}

Hep2Vector & CrystalPad::edge(CaloDirection dir)
{
  switch(dir)
    {
    case NORTHWEST:
      return corners_[0];
      break;
    case NORTHEAST:
      return corners_[1];
      break;
    case SOUTHEAST:
      return corners_[2];
      break;
    case SOUTHWEST:
      return corners_[3];
      break;
    default:
      {
	std::cout << " Serious problem in CrystalPad ! " << dir << std::endl;
	return corners_[0];
      }
    }
  return corners_[0];
}


void CrystalPad::extrems(double &xmin,double& xmax,double &ymin, double& ymax) const
{
  xmin=ymin=999;
  xmax=ymax=-999;
  for(unsigned ic=0;ic<4;++ic)
    {
      if(corners_[ic].x()<xmin) xmin=corners_[ic].x();
      if(corners_[ic].x()>xmax) xmax=corners_[ic].x();
      if(corners_[ic].y()<ymin) ymin=corners_[ic].y();
      if(corners_[ic].y()>ymax) ymax=corners_[ic].y();
    }
}

void CrystalPad::resetCorners() {

  // Find the centre-of-gravity of the Quad (after re-organization)
  center_ = Hep2Vector(0.,0.);
  for(unsigned ic=0;ic<4;++ic) center_ += corners_[ic];
  center_ *= 0.25;

  // Rescale the corners to allow for some inaccuracies in 
  // in the inside test
  for(unsigned ic=0;ic<4;++ic) 
    corners_[ic] += 0.001 * (corners_[ic] - center_) ;

}

std::ostream & operator << (std::ostream& ost,  CrystalPad & quad)
{
  ost << " Number " << quad.getNumber() << std::endl ;
  ost << NORTHWEST << quad.edge(NORTHWEST) << std::endl;
  ost << NORTHEAST << quad.edge(NORTHEAST) << std::endl;
  ost << SOUTHEAST << quad.edge(SOUTHEAST) << std::endl;
  ost << SOUTHWEST << quad.edge(SOUTHWEST) << std::endl;
  
  return ost;
}

