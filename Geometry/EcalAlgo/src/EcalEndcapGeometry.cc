#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>

float EcalEndcapGeometry::grid::dx=0.;
float EcalEndcapGeometry::grid::dy=0.;

EcalEndcapGeometry::EcalEndcapGeometry() :
  _nnmods(0 ),_nncrys(0)
{}

EcalEndcapGeometry::~EcalEndcapGeometry() {}

void 
EcalEndcapGeometry::makeGridMap()
{
   if (cellGeometries().empty())
      return;
   calculateGridSizeFromCells( EcalEndcapGeometry::grid::dx ,
			       EcalEndcapGeometry::grid::dy  ) ;

   CaloSubdetectorGeometry::CellCont::const_iterator i;

  zeP=0.;
  zeN=0.;
  unsigned nP=0;
  unsigned nN=0;
  for (i=cellGeometries().begin(); i!=cellGeometries().end(); i++)
    {
      addCrystalToZGridmap(i->first,dynamic_cast<const TruncatedPyramid*>(i->second));
      float z=dynamic_cast<const TruncatedPyramid*>(i->second)->getPosition(0.).z();
      if(z>0.)
	{
	  zeP+=z;
	  ++nP;
	}
      else
	{
	  zeN+=z;
	  ++nN;
	}

    }
  zeP/=(float)nP;
  zeN/=(float)nN;
  //  std::cout << " Z ECAL " << zeP << " " << zeN << std::endl;
}

// Get closest cell, etc...
DetId 
EcalEndcapGeometry::getClosestCell(const GlobalPoint& r) const 
{
  grid key(r.x() * fabs(zeP/r.z()),r.y() * fabs(zeP/r.z()),(r.z()>0.)?zeP:zeN,0,0);
  const gridmap *pos2cell;
  if(key.side())
    pos2cell=&PZpos2cell;
  else
    pos2cell=&NZpos2cell;
  
  gridmap::const_iterator g_it_low=pos2cell->lower_bound(key.gridX());
  gridmap::const_iterator g_it_up=pos2cell->upper_bound(key.gridX());
  gridmap::const_iterator g_it;

  if(g_it_low==pos2cell->end())
    {
      if(g_it_up!=pos2cell->end())
	g_it=g_it_up;
      else
	{
	  gridmap::const_iterator stupid_loop=pos2cell->begin();
	  do
	    {
	      g_it=stupid_loop;
	      stupid_loop++;
	    } 
	  while(stupid_loop!=pos2cell->end());
	}
    }
  else
    {
      g_it=g_it_low;
    }

  onedmap::const_iterator o_it_low=(*g_it).second.lower_bound(key.gridY());
  onedmap::const_iterator o_it_up=(*g_it).second.upper_bound(key.gridY());
  onedmap::const_iterator o_it;
  if(o_it_low==(*g_it).second.end())
    {
      if(o_it_up!=(*g_it).second.end())
	o_it=o_it_up;
      else
      {
	 return (*(*g_it).second.rbegin()).second;
      }
    }
  else
    {
      o_it=o_it_low;
    }

    try
      {
	EEDetId mycellID((*o_it).second);

	if (!present(mycellID))
	  return DetId(0);

	HepPoint3D  A;
	HepPoint3D  B;
	HepPoint3D  C;
	HepPoint3D  point(r.x(),r.y(),r.z());
	// D.K. : equation of plane : AA*x+BB*y+CC*z+DD=0;
	// finding equation for each edge
      
	// ================================================================
	double x,y,z;
	unsigned offset=0;
	int zsign=1;
	//================================================================
	std::vector<double> SS;
      
	// compute the distance of the point with respect of the 4 crystal lateral planes
	const GlobalPoint& myPosition=getGeometry(mycellID)->getPosition();

	x=myPosition.x();
	y=myPosition.y();
	z=myPosition.z();
      
	offset=0;
	// This will disappear when Andre has applied his fix
	zsign=1;
      
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
	std::vector<GlobalPoint> corners;
	corners.clear();
	corners.resize(8);
	for(unsigned ic=0;ic<4;++ic)
	  {
	    corners[ic]=getGeometry(mycellID)->getCorners()[(unsigned)((zsign*ic+offset)%4)];
	    corners[4+ic]=getGeometry(mycellID)->getCorners()[(unsigned)(4+(zsign*ic+offset)%4)];
	  }
  
	for (short i=0; i < 4 ; ++i)
        {
	  A = HepPoint3D(corners[i%4].x(),corners[i%4].y(),corners[i%4].z());
	  B = HepPoint3D(corners[(i+1)%4].x(),corners[(i+1)%4].y(),corners[(i+1)%4].z());
	  C = HepPoint3D(corners[4+(i+1)%4].x(),corners[4+(i+1)%4].y(),corners[4+(i+1)%4].z());
	  HepPlane3D plane(A,B,C);
	  plane.normalize();
	  double distance = plane.distance(point);
	  if (corners[0].z()<0.) distance=-distance;
	  SS.push_back(distance);
	}
  
	// Only one move in necessary direction

	const bool yout ( 0 > SS[0]*SS[2] ) ;
	const bool xout ( 0 > SS[1]*SS[3] ) ;

	if( yout || xout )
	{
	   try
	   {
	     const int ydel ( !yout ? 0 :  ( 0 < SS[0] ? -1 : 1 ) ) ;
	     const int xdel ( !xout ? 0 :  ( 0 < SS[1] ? -1 : 1 ) ) ;
	     EEDetId nextPoint;
	     nextPoint=EEDetId( mycellID.ix() + xdel,
				mycellID.iy() + ydel, mycellID.zside());
	      if (present(nextPoint))
		 mycellID = nextPoint;
	   }
	   catch ( cms::Exception &e ) 
	   {
	   }
	}

  
	return mycellID;
      }
    catch ( cms::Exception &e ) 
      { 
	return DetId(0);
      } 
}

void 
EcalEndcapGeometry::calculateGridSizeFromCrystalPositions(double xzero, 
							  double yzero, 
							  double xprime,
							  double yprime,
							  float &dx, 
							  float &dy) 
{
  dx=fabs(xprime-xzero)*0.5*1.1;
  dy=fabs(yprime-yzero)*0.5*1.1;
  assert(dx > 0);
  assert(dy > 0);
}

//----------------------------------------------------------------------

void 
EcalEndcapGeometry::calculateGridSizeFromCells(float &dx, float &dy)
{
  // calculate the position of the first crystal in the first module

  const TruncatedPyramid *ref_crystal, *crystal_right, *crystal_up;

  // get some fixed crystals (which should exist) to determine the
  // grid size
  const int z_index = 1;
  int ix = 51; // (crystal 1 does not exist in module 1) 
  int iy = 71;
  
  {
    EEDetId cell_id(ix,iy,z_index,EEDetId::XYMODE);
    assert(!cell_id.null()); // make sure we've found the cell
    ref_crystal = dynamic_cast<const TruncatedPyramid*>(getGeometry(cell_id));
    assert(ref_crystal != NULL);
  }

  // calculate the position of the neighbour crystal at higher x 
  // (assumes that crystal numbering increases by one in the y
  // direction and by module_edge_length in the x direction)
  {
    EEDetId cell_id(ix+1,iy,z_index,EEDetId::XYMODE);
    crystal_right= dynamic_cast<const TruncatedPyramid*>(getGeometry(cell_id));
    assert(crystal_right != NULL);
  }

  // calculate the position of the neighbour crystal at higher y
  {
    EEDetId cell_id(ix,iy+1,z_index,EEDetId::XYMODE);
    crystal_up = dynamic_cast<const TruncatedPyramid*>(getGeometry(cell_id));
    assert(crystal_up != NULL);
  }

  calculateGridSizeFromCrystalPositions
    (ref_crystal->getPosition(0.).x(),
     ref_crystal->getPosition(0.).y(),
     crystal_right->getPosition(0.).x(),
     crystal_up->getPosition(0.).y(),
     dx,dy);

}


//----------------------------------------------------------------------

void 
EcalEndcapGeometry::addCrystalToZGridmap( const DetId &id, 
					  const TruncatedPyramid* crystal)
{
  grid g;
  g = grid(crystal->getPosition(0.).x(),
           crystal->getPosition(0.).y(),
           crystal->getPosition(0.).z(),
           crystal->getThetaAxis(),
           crystal->getPhiAxis()
           );
  
  if(g.side())
    PZpos2cell[g.gridX()][g.gridY()]=id;
  else
    NZpos2cell[g.gridX()][g.gridY()]=id;
}
