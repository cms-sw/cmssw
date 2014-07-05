#ifndef GeometryFCalGeometryShashlikGeometry_h
#define GeometryFCalGeometryShashlikGeometry_h


/*
 * Geometry for Shashlik ECAL
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 * Fedor Ratnikov, Apr. 8 2014
 */

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/ShashlikGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include <vector>

class TruncatedPyramid;

class ShashlikGeometry GCC11_FINAL: public CaloSubdetectorGeometry 
{
 public:
  
  typedef std::vector<TruncatedPyramid> CellVec ;
  
  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

  enum { k_NumberOfParametersPerShape = 11 } ; // TruncatedPyramid
  enum { k_NumberOfShapes = 1 } ; 
 
  ShashlikGeometry(const ShashlikTopology& topology) ;
  
  virtual ~ShashlikGeometry();
  
  void addValidID(const DetId& id);

  // avoid sorting set in base class  
  virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det = DetId::Detector(0), int subdet = 0) const {return m_validIds;}
  
  virtual void newCell( const GlobalPoint& f1 ,
			const GlobalPoint& f2 ,
			const GlobalPoint& f3 ,
			const CCGFloat*    parm ,
			const DetId&       detId );
  
  /// Get the cell geometry of a given detector id.  Should return false if not found.
  virtual const CaloCellGeometry* getGeometry( const DetId& id ) const ;
  
  
  virtual void fillNamedParams (DDFilteredView fv);
  virtual void initializeParms() ;
  
  static std::string producerTag() { return "Shashlik" ; }
  static const char* cellElement() { return "ShashlikModule" ; }
  
  const ShashlikTopology& topology () const {return mTopology;}
  
 protected:
  
  virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const ;
  virtual unsigned int indexFor(const DetId& id) const { 
    return  mTopology.cell2denseId(id); 
  }
  virtual unsigned int sizeForDenseIndex(const DetId& id) const { 
    return mTopology.cellHashSize(); 
  }

  
 private:
  const ShashlikTopology& mTopology;
  struct SideConstants {
    double zMean;
    double xMin;
    double xMax;
    double yMin;
    double yMax;
    int ixMin;
    int ixMax;
    int iyMin;
    int iyMax;
    SideConstants () {
      zMean = 0;
      xMin = yMin = 9999;
      xMax = yMax = -9999;
      ixMin = iyMin = 9999;
      ixMax = iyMax = -9999;
    }
  };
  SideConstants mSide[2];
  int xindex( CCGFloat x, CCGFloat z ) const ; 
  int yindex( CCGFloat y, CCGFloat z ) const ; 
  
  EKDetId gId( float x, float y, float z ) const ; 
  
  CellVec m_cellVec ; 
};


#endif

