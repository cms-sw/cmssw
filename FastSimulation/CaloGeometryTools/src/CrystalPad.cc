#include "FastSimulation/CaloGeometryTools/interface/CrystalPad.h"

#include <iostream> 

std::vector<CLHEP::Hep2Vector> CrystalPad::aVector(4);


CrystalPad::CrystalPad(const CrystalPad& right) 
{
  corners_ = right.corners_;
  dir_ = right.dir_;
  number_ = right.number_;
  survivalProbability_ = right.survivalProbability_;
  center_ = right.center_;
  epsilon_ = right.epsilon_;
  dummy_ = right.dummy_;
}

CrystalPad&  
CrystalPad::operator = (const CrystalPad& right ) {
  if (this != &right) { // don't copy into yourself
    corners_ = right.corners_;
    dir_ = right.dir_;
    number_ = right.number_;
    survivalProbability_ = right.survivalProbability_;
    center_ = right.center_;
    epsilon_ = right.epsilon_;
    dummy_ = right.dummy_;
  }
  return *this;
}

CrystalPad::CrystalPad(unsigned number, 
		       const std::vector<CLHEP::Hep2Vector>& corners) 
  :
  corners_(corners),
  dir_(aVector),
  number_(number),
  survivalProbability_(1.),
  center_(0.,0.),
  epsilon_(0.001)
{

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
      for(unsigned ic=0; ic<4;++ic)
	{
	  dir_[ic] = (corners[(ic+1)%4]-corners[ic]).unit();
	  center_+=corners_[ic];
	}
      center_*=0.25;
    }
//  std::cout << " End of 1 constructor " << std::endl;
//  std::cout << " Ncorners " << corners_.size() << std::endl;
//  std::cout << " Ndirs " << dir_.size() << std::endl;
}

CrystalPad::CrystalPad(unsigned number, int onEcal, 
		       const std::vector<XYZPoint>& corners,
		       const XYZPoint& origin, 
		       const XYZVector& vec1,
		       const XYZVector& vec2) 
  : 
  corners_(aVector),
  dir_(aVector),
  number_(number),
  survivalProbability_(1.),
  center_(0.,0.),
  epsilon_(0.001)
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

      // the good one in the central
      trans_=Transform3D((Point)origin,
			 (Point)(origin+vec1),
			 (Point)(origin+vec2),
			 Point(0.,0.,0.),
			 Point(0.,0.,sign),
			 Point(0.,1.,0.));
      trans_.GetDecomposition(rotation_,translation_);
      //      std::cout << " Constructor 2; input corners "  << std::endl;
      for(unsigned ic=0;ic<4;++ic)
	{	
	  //	  std::cout << corners[ic]<< " " ;
	  XYZPoint corner = rotation_(corners[ic])+translation_;
	  //	  std::cout << corner << std::endl ;
	  corners_[ic] = CLHEP::Hep2Vector(corner.X(),corner.Y());
	  center_+=corners_[ic];
	}
      for(unsigned ic=0;ic<4;++ic)
	{
	  dir_[ic] = (corners_[(ic+1)%4]-corners_[ic]).unit();
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
CrystalPad::CrystalPad(unsigned number, 
		       const std::vector<XYZPoint>& corners,
		       const Transform3D & trans,double scaf,bool bothdirections) 
  : 
  corners_(aVector),
  dir_(aVector),
  number_(number),
  survivalProbability_(1.),
  center_(0.,0.),
  epsilon_(0.001),
  yscalefactor_(scaf)
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

      // the good one in the central
      trans_=trans;
      //      std::cout << " Constructor 2; input corners "  << std::endl;
      trans_.GetDecomposition(rotation_,translation_);
      for(unsigned ic=0;ic<4;++ic)
	{	

	  XYZPoint corner=rotation_(corners[ic])+translation_;
	  //	  std::cout << corner << std::endl ;
	  double xscalefactor=(bothdirections) ? yscalefactor_:1.;
	  corners_[ic] = CLHEP::Hep2Vector(corner.X()*xscalefactor,corner.Y()*yscalefactor_);
	  center_+=corners_[ic];
	}
      for(unsigned ic=0;ic<4;++ic)
	{
	  dir_[ic] = (corners_[(ic+1)%4]-corners_[ic]).unit();
	}
      center_*=0.25;
    }  
}

bool 
CrystalPad::inside(const CLHEP::Hep2Vector & ppoint,bool debug) const
{
//  std::cout << "Inside " << ppoint <<std::endl;
//  std::cout << "Corners " << corners_.size() << std::endl;
//  std::cout << corners_[0] << std::endl;
//  std::cout << corners_[1] << std::endl;
//  std::cout << corners_[2] << std::endl;
//  std::cout << corners_[3] << std::endl;
//  std::cout << " Got the 2D point " << std::endl;
  CLHEP::Hep2Vector pv0(ppoint-corners_[0]);
  CLHEP::Hep2Vector pv2(ppoint-corners_[2]);
  CLHEP::Hep2Vector n1(pv0-(pv0*dir_[0])*dir_[0]);
  CLHEP::Hep2Vector n2(pv2-(pv2*dir_[2])*dir_[2]);

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
  CLHEP::Hep2Vector pv1(ppoint-corners_[1]);
  CLHEP::Hep2Vector pv3(ppoint-corners_[3]);
  CLHEP::Hep2Vector n3(pv1-(pv1*dir_[1])*dir_[1]);
  CLHEP::Hep2Vector n4(pv3-(pv3*dir_[3])*dir_[3]);
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

/*
bool 
CrystalPad::globalinside(XYZPoint point) const
{
  //  std::cout << " Global inside " << std::endl;
  //  std::cout << point << " " ;
  ROOT::Math::Rotation3D r;
  XYZVector t;
  point = rotation_(point)+translation_;
  //  std::cout << point << std::endl;
  //  print();
  CLHEP::Hep2Vector ppoint(point.X(),point.Y());
  bool result=inside(ppoint);
  //  std::cout << " Result " << result << std::endl;
  return result;
}
*/


void CrystalPad::print() const
{
  std::cout << " Corners " << std::endl;
  std::cout << corners_[0] << std::endl;
  std::cout << corners_[1] << std::endl;
  std::cout << corners_[2] << std::endl;
  std::cout << corners_[3] << std::endl;
}

/*
CLHEP::Hep2Vector 
CrystalPad::localPoint(XYZPoint point) const
{
  point = rotation_(point)+translation_;
  return CLHEP::Hep2Vector(point.X(),point.Y());
}
*/

CLHEP::Hep2Vector& CrystalPad::edge(unsigned iside,int n) 
{
  return corners_[(iside+n)%4];
}

CLHEP::Hep2Vector & CrystalPad::edge(CaloDirection dir)
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


void 
CrystalPad::extrems(double &xmin,double& xmax,double &ymin, double& ymax) const
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

void 
CrystalPad::resetCorners() {

  // Find the centre-of-gravity of the Quad (after re-organization)
  center_ = CLHEP::Hep2Vector(0.,0.);
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

void 
CrystalPad::getDrawingCoordinates(std::vector<float> &x, std::vector<float>&y) const
{
  x.clear();
  y.clear();
  x.push_back(corners_[0].x());
  x.push_back(corners_[1].x());
  x.push_back(corners_[2].x());
  x.push_back(corners_[3].x());
  x.push_back(corners_[0].x());
  y.push_back(corners_[0].y());
  y.push_back(corners_[1].y());
  y.push_back(corners_[2].y());
  y.push_back(corners_[3].y());
  y.push_back(corners_[0].y());
}
