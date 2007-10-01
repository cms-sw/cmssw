#ifndef EcalEndcapGeometry_h
#define EcalEndcapGeometry_h

#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <vector>
#include <map>

class TruncatedPyramid;

class EcalEndcapGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef EcalEndcapNumberingScheme NumberingScheme ;

      enum CornersCount { k_NumberOfCellsForCorners = 14648 } ;

      EcalEndcapGeometry() ;
  
      virtual ~EcalEndcapGeometry();

      int getNumberOfModules()          const { return _nnmods ; }

      int getNumberOfCrystalPerModule() const { return _nncrys ; }

      void setNumberOfModules(          const int nnmods ) { _nnmods=nnmods ; }

      void setNumberOfCrystalPerModule( const int nncrys ) { _nncrys=nncrys ; }

      // Get closest cell, etc...
      virtual DetId getClosestCell( const GlobalPoint& r ) const ;

      void makeGridMap();

      static std::string hitString() { return "EcalHitsEE" ; }

      static std::string producerName() { return "EcalEndcap" ; }

   private:

      /// number of modules
      int _nnmods;
  
      /// number of crystals per module
      int _nncrys; 

      float zeP,zeN;

   protected:
      /** Internal grid class for EcalEndcapBase Xtals calculations */
      class grid
      {
	 public:
	    
	    /**
	       \param a x coordinate (e.g. of crystal front face)
	       \param b y coordinate (e.g. of crystal front face)
	       \param c z coordinate (e.g. of crystal front face)
	       \param t theta angle (e.g. of crystal axis)
	       \param p phi angle (e.g. of crystal axis)
	    */
	    grid(double a,double b,double c,double t,double p):
	       x(a),y(b),z(c),thetaAxis(t),phiAxis(p){}

	    /** default constructor (useful for STL containers etc.) */
	    grid() :
	       x(0),y(0),z(0),thetaAxis(0),phiAxis(0){}

	    /** copy constructor */
	    grid(const grid&other):
	       x(other.x),y(other.y),z(other.z),thetaAxis(other.thetaAxis),
	       phiAxis(other.phiAxis){}
      

	    grid& operator=(const grid&other)
	    {
	       x=other.x;
	       y=other.y;
	       z=other.z;
	       thetaAxis=other.thetaAxis;
	       phiAxis=other.phiAxis;
	       return *this;
	    }

	    bool side() const {return z>0.0;}
	    
	    float gridX() const {return x;}
	    float gridY() const {return y;}
	    float gridZ() const {return z;}

	    float phiaxis() const{return phiAxis;}
	    float thetaaxis() const{return thetaAxis;}

	    //    HepPoint3D Point() const {return HepPoint3D(x,y,z);}
	    static float dx;
	    static float dy;
	    bool operator()(const grid&a,const grid&b) const
	    {
	       if(a.side()<b.side()) return true;
	       if(a.x<b.x && b.x-a.x>dx) return true;
	       if(a.y<b.y && b.y-a.y>dy) return true;
	       return false;
	    }
	 private:
	    float x;
	    float y;
	    float z;

	    // theta and phi are needed to go an HCAL cell
	    float thetaAxis;
	    float phiAxis;
      };

   private:
      /** Internal xcomp EcalEndcapBase class for x computations */
      class xcomp
      {
	 public:
	    bool operator()(const float& a,const float& b) const
	    {
	       return (a<b && b-a>grid::dx);
	    }
      };

      /** Internal ycomp EcalEndcapBase class for x computations */
      class ycomp
      {
	 public:
	    bool operator()(const float& a,const float& b) const
	    {
	       return (a<b && b-a>grid::dy);
	    }
      };

      typedef std::map<float,DetId,ycomp > onedmap;
      typedef std::map<float,onedmap,xcomp > gridmap;

      /** cells with positive z coordinate */
      gridmap PZpos2cell;

      /** cells with negative z coordinate */
      gridmap NZpos2cell;

      /** helper function for calculateGridSizeFromCells(..)
	  and calculateGridSizeFromG3(..): Given the position of the
	  reference crystal and the x position of the crystal on its right and the y
	  position of the crystal above it, this initializes the grid size

	  \param xzero the x position of the 'reference' crystal
	  \param yzero the y position of the 'reference' crystal
	  \param xprime the x position of the crystal right of the 'reference' crystal
	  \param yprime the y position of the crystal above the the 'reference' crystal
      */
      void calculateGridSizeFromCrystalPositions( double xzero,
						  double yzero, 
						  double xprime,
						  double yprime,
						  float &dx,
						  float &dy      );
  
      /** calculates the grid size (needed for navigation) from the 
	  geometry of the (already initialized !) cells. 
      */
      void calculateGridSizeFromCells( float &dx, 
				       float &dy  ) ;

      /** helper function which adds the CellID id to either PZpos2cell or
	  NZpos2cell (Depending on it's z coordinate) and to the cell2pos
	  map 

      */
      void addCrystalToZGridmap( const DetId&            id      ,
				 const TruncatedPyramid* crystal   ) ;
} ;


#endif

