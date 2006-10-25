//CMSSW headers
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/Crystal.h"
#include "FastSimulation/CaloGeometryTools/interface/CrystalPad.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloDirectionOperations.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


Crystal::Crystal(const DetId&  cell):cellid_(cell),dummy_(true)
{
  subdetn_ = cellid_.subdetId();
}

void Crystal::setCorners(const std::vector<GlobalPoint>& vec,const GlobalPoint& pos)
{
  unsigned ncorners= vec.size();
  if(ncorners!=8) return;

  // This is really a pity to have to make the conversion GlobalPoint to CLHEP , but the latter have many
  // useful properties (transformations, conversions....) that aren't implemented
  // for the GlobalPoints and GlobalVectors. 

  if(subdetn_==EcalBarrel)
    {
      if(pos.z()>0.)
	{
	  corners_.clear();
	  corners_.resize(8);
	  for(unsigned ic=0;ic<8;++ic)
	    {
	      corners_[ic]=HepPoint3D(vec[ic].x(),vec[ic].y(),vec[ic].z());
	    }
	}
      else
	{
	  corners_.clear();
	  corners_.resize(8);
	  corners_[0]=HepPoint3D(vec[2].x(),vec[2].y(),vec[2].z());
	  corners_[1]=HepPoint3D(vec[3].x(),vec[3].y(),vec[3].z());
	  corners_[2]=HepPoint3D(vec[0].x(),vec[0].y(),vec[0].z());
	  corners_[3]=HepPoint3D(vec[1].x(),vec[1].y(),vec[1].z());
	  corners_[4]=HepPoint3D(vec[6].x(),vec[6].y(),vec[6].z());
	  corners_[5]=HepPoint3D(vec[7].x(),vec[7].y(),vec[7].z());
	  corners_[6]=HepPoint3D(vec[4].x(),vec[4].y(),vec[4].z());
	  corners_[7]=HepPoint3D(vec[5].x(),vec[5].y(),vec[5].z());
	}
    }
  else if(subdetn_==EcalEndcap)
    {
      double x=center_.x();
      double y=center_.y();
      double z=center_.z();
      unsigned offset=0;
      int zsign=1;
      if(z>0) 
	{
	  if(x>0&&y>0) 
	    offset=1;
	  else  if(x<0&&y>0) 
	    offset=2;
	  else if(x>0&&y<0) 
	    offset=0;
	  else if (x<0&&y<0) 
	    offset=3;
	  zsign=1;
	}
      else
	{
	  if(x>0&&y>0) 
	    offset=3;
	  else if(x<0&&y>0) 
	    offset=2;
	  else if(x>0&&y<0) 
	    offset=0;
	  else if(x<0&&y<0) 
	    offset=1;
	  zsign=-1;
	}
      
      corners_.clear();
      corners_.resize(8);
      for(unsigned ic=0;ic<4;++ic)
	{
	  unsigned i1=(unsigned)((zsign*ic+offset)%4);
	  unsigned i2=i1+4;
	  corners_[ic]=HepPoint3D(vec[i1].x(),vec[i1].y(),vec[i1].z());
	  corners_[4+ic]=HepPoint3D(vec[i2].x(),vec[i2].y(),vec[i2].z());
	}
    }
  
  center_=HepPoint3D(0.,0.,0.);
  
  for(unsigned ic=0;ic<8;++ic)
    {
      center_+=corners_[ic];
    }

  center_*=0.125;

  //  std::cout << " Ncorners ? " << corners_.size() << std::endl;
  frontcenter_ = 0.25*(corners_[0]+corners_[1]+corners_[2]+corners_[3]);
  backcenter_ = 0.25*(corners_[4]+corners_[5]+corners_[6]+corners_[7]);
  crystalaxis_ = backcenter_-frontcenter_;
  firstedgedirection_=-1.*((HepVector3D)(corners_[1]-corners_[0])).unit();
  //  firstedgedirection_=((HepVector3D)(corners_[0]-corners_[3])).unit();
  fifthedgedirection_=-1.*((HepVector3D)(corners_[5]-corners_[4])).unit();
  lateraldirection_.resize(4);
  //  std::cout << " Direction laterales " << std::endl;
  for(unsigned il=0;il<4;++il)
    {
      lateraldirection_[il]=-((HepVector3D)(corners_[(il+1)%4]-corners_[il])).unit();
    }
  neighbours_.resize(8);  
  dummy_ = false; 
}

void Crystal::getLateralEdges(unsigned i,HepPoint3D& a,HepPoint3D& b)const
{
  if(i>=0&&i<4)
    {
      a=corners_[i];
      b=corners_[i+4]; 
    }
}

void Crystal::getFrontSide(HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const
{
  a=corners_[0];
  b=corners_[1];
  c=corners_[2];
  d=corners_[3];
}

void Crystal::getFrontSide(std::vector<HepPoint3D>& corners) const
{
  if(corners.size()==4)
    {
      corners[0]=corners_[0];
      corners[1]=corners_[1];
      corners[2]=corners_[2];
      corners[3]=corners_[3];
    }
}

void Crystal::getBackSide(HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const
{
  a=corners_[4];
  b=corners_[5];
  c=corners_[6];
  d=corners_[7];
}

void Crystal::getBackSide(std::vector<HepPoint3D>& corners) const
{
  if(corners.size()==4)
    {
      corners[0]=corners_[4];
      corners[1]=corners_[5];
      corners[2]=corners_[6];
      corners[3]=corners_[7];
    }
}

void Crystal::getLateralSide(unsigned i,HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const
{
  if(i>=0&&i<4)
    {
      getLateralEdges(i,a,b);
      getLateralEdges((i+1)%4,c,d);
    }
}

void Crystal::getLateralSide(unsigned i,std::vector<HepPoint3D>& corners) const
{
  if(corners.size()==4&&i>=0&&i<4)
    {
      corners[0]=corners_[i];
      corners[1]=corners_[i+4];
      corners[2]=corners_[4+(i+1)%4];
      corners[3]=corners_[(i+1)%4];
    }
}

HepPlane3D Crystal::getFrontPlane() const
{
  return HepPlane3D(corners_[0],corners_[1],corners_[2]);
}

HepPlane3D Crystal::getBackPlane() const
{
  return HepPlane3D(corners_[4],corners_[5],corners_[6]);
}

HepPlane3D Crystal::getLateralPlane(unsigned i) const
{
  if(i>=0&&i<4)
    {
      HepPlane3D tmp(corners_[i],corners_[(i+1)%4],corners_[i+4]);
//      if(tmp.normal().mag()==0)
//	{
//	  std::cout << " Crystal::getLateralPlane() " << std::endl;
//	  std::cout << corners_[i] << " " << corners_[(i+1)%4] << " " << corners_[i+4] << std::endl;
//	}
      return HepPlane3D(corners_[i],corners_[(i+1)%4],corners_[i+4]);
      
    }
 return HepPlane3D();
}


void Crystal::getDrawingCoordinates(std::vector<float> &x,std::vector<float> &y,std::vector<float> &z) const
{
  x.clear();
  y.clear();
  z.clear();

  x.push_back(corners_[0].x());
  x.push_back(corners_[3].x());
  x.push_back(corners_[2].x());
  x.push_back(corners_[1].x());
  x.push_back(corners_[5].x());
  x.push_back(corners_[6].x());
  x.push_back(corners_[7].x());
  x.push_back(corners_[4].x());
  x.push_back(corners_[0].x());
  x.push_back(corners_[1].x());
  x.push_back(corners_[2].x());
  x.push_back(corners_[6].x());
  x.push_back(corners_[5].x());
  x.push_back(corners_[4].x());
  x.push_back(corners_[7].x());
  x.push_back(corners_[3].x());

  y.push_back(corners_[0].y());
  y.push_back(corners_[3].y());
  y.push_back(corners_[2].y());
  y.push_back(corners_[1].y());
  y.push_back(corners_[5].y());
  y.push_back(corners_[6].y());
  y.push_back(corners_[7].y());
  y.push_back(corners_[4].y());
  y.push_back(corners_[0].y());
  y.push_back(corners_[1].y());
  y.push_back(corners_[2].y());
  y.push_back(corners_[6].y());
  y.push_back(corners_[5].y());
  y.push_back(corners_[4].y());
  y.push_back(corners_[7].y());
  y.push_back(corners_[3].y());

  z.push_back(corners_[0].z());
  z.push_back(corners_[3].z());
  z.push_back(corners_[2].z());
  z.push_back(corners_[1].z());
  z.push_back(corners_[5].z());
  z.push_back(corners_[6].z());
  z.push_back(corners_[7].z());
  z.push_back(corners_[4].z());
  z.push_back(corners_[0].z());
  z.push_back(corners_[1].z());
  z.push_back(corners_[2].z());
  z.push_back(corners_[6].z());
  z.push_back(corners_[5].z());
  z.push_back(corners_[4].z());
  z.push_back(corners_[7].z());
  z.push_back(corners_[3].z());
}




void Crystal::getSide(const CaloDirection& side, HepPoint3D &a,HepPoint3D &b,HepPoint3D &c,HepPoint3D &d) const
{
  switch (side)
    {
    case UP:
      getFrontSide(a,b,c,d);
      break;
    case DOWN:
      getBackSide(a,b,c,d);
      break;
    default:
      getLateralSide(CaloDirectionOperations::Side(side),a,b,c,d);
    }
}

void Crystal::getSide(const CaloDirection& side, std::vector<HepPoint3D>& corners) const
{
  switch (side)
    {
    case UP:
      getFrontSide(corners);
      break;
    case DOWN:
      getBackSide(corners);
      break;
    default:
      getLateralSide(CaloDirectionOperations::Side(side),corners);
    }
}


HepPlane3D Crystal::getPlane(const CaloDirection& side) const
{
  switch (side)
    {
    case UP:
      return getFrontPlane();
      break;
    case DOWN:
      return getBackPlane();
      break;
    default:
      return getLateralPlane(CaloDirectionOperations::Side(side));
    }
  return getFrontPlane();
}

HepVector3D Crystal::exitingNormal(const CaloDirection& side) const
{
  HepPlane3D plan=getPlane(side);
  return (plan.distance(center_)<0.)? plan.normal().unit() : -plan.normal().unit() ;
}

