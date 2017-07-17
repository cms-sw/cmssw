//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/BaseCrystal.h"

// Data Formats
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

typedef ROOT::Math::Plane3D::Point Point;

BaseCrystal::BaseCrystal(const DetId&  cell):cellid_(cell)
{
  subdetn_ = cellid_.subdetId();
}

void BaseCrystal::setCorners(const CaloCellGeometry::CornersVec& vec,const GlobalPoint& pos)
{
  unsigned ncorners= vec.size();
  if(ncorners!=8) return;

  // This is really a pity to have to make the conversion GlobalPoint to XYZPoint, but the latter have many
  // useful properties (transformations, conversions....) that aren't implemented
  // for the GlobalPoints and GlobalVectors. 

  if(subdetn_==EcalBarrel)
    {
      if(pos.z()>0.)
	{
	  for(unsigned ic=0;ic<8;++ic)
	    {
	      corners_[ic]=XYZPoint(vec[ic].x(),vec[ic].y(),vec[ic].z());
	    }
	}
      else
	{
	  corners_[0]=XYZPoint(vec[2].x(),vec[2].y(),vec[2].z());
	  corners_[1]=XYZPoint(vec[3].x(),vec[3].y(),vec[3].z());
	  corners_[2]=XYZPoint(vec[0].x(),vec[0].y(),vec[0].z());
	  corners_[3]=XYZPoint(vec[1].x(),vec[1].y(),vec[1].z());
	  corners_[4]=XYZPoint(vec[6].x(),vec[6].y(),vec[6].z());
	  corners_[5]=XYZPoint(vec[7].x(),vec[7].y(),vec[7].z());
	  corners_[6]=XYZPoint(vec[4].x(),vec[4].y(),vec[4].z());
	  corners_[7]=XYZPoint(vec[5].x(),vec[5].y(),vec[5].z());
	}
    }
  else if(subdetn_==EcalEndcap)
    {
      double x=pos.x();
      double y=pos.y();
      double z=pos.z();
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
      for(unsigned ic=0;ic<4;++ic)
	{
	  unsigned i1=(unsigned)((zsign*ic+offset)%4);
	  unsigned i2=i1+4;
	  corners_[ic]=XYZPoint(vec[i1].x(),vec[i1].y(),vec[i1].z());
	  corners_[4+ic]=XYZPoint(vec[i2].x(),vec[i2].y(),vec[i2].z());
	}
    }
  computeBasicProperties();
}

void BaseCrystal::computeBasicProperties()
{
  //if(corners_.size()==0) return;
  center_=XYZPoint(0.,0.,0.);  
  for(unsigned ic=0;ic<8;++ic)
    {
      center_+=corners_[ic];
    }
  
  center_*=0.125;

  //  std::cout << " Ncorners ? " << corners_.size() << std::endl;
  frontcenter_ = 0.25*(corners_[0]+corners_[1]+corners_[2]+corners_[3]);
  backcenter_ = 0.25*(corners_[4]+corners_[5]+corners_[6]+corners_[7]);
  crystalaxis_ = backcenter_-frontcenter_;
  firstedgedirection_=-(corners_[1]-corners_[0]).Unit();
  fifthedgedirection_=-(corners_[5]-corners_[4]).Unit();
  //  std::cout << " Direction laterales " << std::endl;
  for(unsigned il=0;il<4;++il)
    {
      lateraldirection_[il]=-(corners_[(il+1)%4]-corners_[il]).Unit();
    }
  
  Plane3D frontPlane((Point)corners_[0],(Point)corners_[1],(Point)corners_[2]);
  Plane3D backPlane ((Point)corners_[4],(Point)corners_[5],(Point)corners_[6]);
  for(unsigned i=0;i<4;++i)
    {
      lateralPlane_[i]=
	Plane3D((Point)corners_[i],(Point)corners_[(i+1)%4],(Point)corners_[i+4]);
    }
  // Front plane i=4 (UP)
  lateralPlane_[4] = frontPlane;
  // Back plane i =5 (DOWN)
  lateralPlane_[5] = backPlane;

  for(unsigned i=0;i<6;++i)
    {
      exitingNormal_[i] = 
	(lateralPlane_[i].Distance(Point(center_.X(),center_.Y(),center_.Z())) < 0.) ? 
	lateralPlane_[i].Normal().Unit() : -lateralPlane_[i].Normal().Unit();
    }
}

void BaseCrystal::getLateralEdges(unsigned i,XYZPoint& a,XYZPoint& b)const
{
  if(i<4U) // i >= 0, since i is unsigned
    {
      a=corners_[i];
      b=corners_[i+4]; 
    }
}

void BaseCrystal::getFrontSide(XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const
{
  a=corners_[0];
  b=corners_[1];
  c=corners_[2];
  d=corners_[3];
}

void BaseCrystal::getFrontSide(std::vector<XYZPoint>& corners) const
{
  if(corners.size()==4)
    {
      corners[0]=corners_[0];
      corners[1]=corners_[1];
      corners[2]=corners_[2];
      corners[3]=corners_[3];
    }
}

void BaseCrystal::getBackSide(XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const
{
  a=corners_[4];
  b=corners_[5];
  c=corners_[6];
  d=corners_[7];
}

void BaseCrystal::getBackSide(std::vector<XYZPoint>& corners) const
{
  if(corners.size()==4)
    {
      corners[0]=corners_[4];
      corners[1]=corners_[5];
      corners[2]=corners_[6];
      corners[3]=corners_[7];
    }
}

void BaseCrystal::getLateralSide(unsigned i,XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const
{
  if(i<4U) // i >= 0, since i is unsigned
    {
      getLateralEdges(i,a,b);
      getLateralEdges((i+1)%4,c,d);
    }
}

void BaseCrystal::getLateralSide(unsigned i,std::vector<XYZPoint>& corners) const
{
  if(corners.size()==4&&i<4U) // i >= 0, since i is unsigned
    {
      corners[0]=corners_[i];
      corners[1]=corners_[i+4];
      corners[2]=corners_[4+(i+1)%4];
      corners[3]=corners_[(i+1)%4];
    }
}

void BaseCrystal::getDrawingCoordinates(std::vector<float> &x,std::vector<float> &y,std::vector<float> &z) const
{
  x.clear();
  y.clear();
  z.clear();

  x.push_back(corners_[0].X());
  x.push_back(corners_[3].X());
  x.push_back(corners_[2].X());
  x.push_back(corners_[1].X());
  x.push_back(corners_[5].X());
  x.push_back(corners_[6].X());
  x.push_back(corners_[7].X());
  x.push_back(corners_[4].X());
  x.push_back(corners_[0].X());
  x.push_back(corners_[1].X());
  x.push_back(corners_[2].X());
  x.push_back(corners_[6].X());
  x.push_back(corners_[5].X());
  x.push_back(corners_[4].X());
  x.push_back(corners_[7].X());
  x.push_back(corners_[3].X());

  y.push_back(corners_[0].Y());
  y.push_back(corners_[3].Y());
  y.push_back(corners_[2].Y());
  y.push_back(corners_[1].Y());
  y.push_back(corners_[5].Y());
  y.push_back(corners_[6].Y());
  y.push_back(corners_[7].Y());
  y.push_back(corners_[4].Y());
  y.push_back(corners_[0].Y());
  y.push_back(corners_[1].Y());
  y.push_back(corners_[2].Y());
  y.push_back(corners_[6].Y());
  y.push_back(corners_[5].Y());
  y.push_back(corners_[4].Y());
  y.push_back(corners_[7].Y());
  y.push_back(corners_[3].Y());

  z.push_back(corners_[0].Z());
  z.push_back(corners_[3].Z());
  z.push_back(corners_[2].Z());
  z.push_back(corners_[1].Z());
  z.push_back(corners_[5].Z());
  z.push_back(corners_[6].Z());
  z.push_back(corners_[7].Z());
  z.push_back(corners_[4].Z());
  z.push_back(corners_[0].Z());
  z.push_back(corners_[1].Z());
  z.push_back(corners_[2].Z());
  z.push_back(corners_[6].Z());
  z.push_back(corners_[5].Z());
  z.push_back(corners_[4].Z());
  z.push_back(corners_[7].Z());
  z.push_back(corners_[3].Z());
}




void BaseCrystal::getSide(const CaloDirection& side, XYZPoint &a,XYZPoint &b,XYZPoint &c,XYZPoint &d) const
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

void BaseCrystal::print() const{
  std::cout << "CellID " << cellid_.rawId() << std::endl;
  std::cout << " Corners " << std::endl;
  for(unsigned ic=0;ic<8;++ic)
    std::cout << corners_[ic] << std::endl;
  std::cout << " Center " << center_ << std::endl;
  std::cout << " Front Center " << frontcenter_ << std::endl;
  std::cout << " Back Center " << backcenter_ << std::endl;
  std::cout << " Normales sortantes " << std::endl;
  for(unsigned id=0;id<6;++id)
    std::cout << exitingNormal_[id] << std::endl;
}

void BaseCrystal::getSide(const CaloDirection& side, std::vector<XYZPoint>& corners) const
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



