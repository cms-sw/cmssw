#ifndef GeometryHGCalGeometryHGCalGeometry_h
#define GeometryHGCalGeometryHGCalGeometry_h

/*
 * Geometry for High Granularity Calorimeter
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 */

#include "DataFormats/Common/interface/AtomicPtrCache.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include <vector>

class FlatTrd;

class HGCalGeometry GCC11_FINAL: public CaloSubdetectorGeometry {

public:
  
  typedef std::vector<FlatTrd> CellVec ;
  
  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

  typedef std::set<DetId>            DetIdSet;
  typedef std::vector<GlobalPoint>   CornersVec ;

  enum { k_NumberOfParametersPerShape = 12 } ; // FlatTrd
  enum { k_NumberOfShapes = 50 } ; 
 
  HGCalGeometry(const HGCalTopology& topology) ;
  
  virtual ~HGCalGeometry();

  virtual void newCell( const GlobalPoint& f1 ,
			const GlobalPoint& f2 ,
			const GlobalPoint& f3 ,
			const CCGFloat*    parm ,
			const DetId&       detId );
  
  /// Get the cell geometry of a given detector id.  Should return false if not found.
  virtual const CaloCellGeometry* getGeometry( const DetId& id ) const ;
  
  GlobalPoint getPosition( const DetId& id ) const;
      
  /// Returns the corner points of this cell's volume.
  CornersVec getCorners( const DetId& id ) const; 

  // avoid sorting set in base class  
  virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det = DetId::Detector(0), int subdet = 0) const {return m_validIds;}
  
  // Get closest cell, etc...
  virtual DetId getClosestCell( const GlobalPoint& r ) const ;
  
  /** \brief Get a list of all cells within a dR of the given cell
      
      The default implementation makes a loop over all cell geometries.
      Cleverer implementations are suggested to use rough conversions between
      eta/phi and ieta/iphi and test on the boundaries.
  */
  virtual DetIdSet getCells( const GlobalPoint& r, double dR ) const ;
  
  virtual void fillNamedParams (DDFilteredView fv);
  virtual void initializeParms() ;
  
  static std::string producerTag() { return "HGCal" ; }
  std::string cellElement() const;
  
  const HGCalTopology& topology () const {return mTopology;}
  
protected:

  virtual unsigned int indexFor(const DetId& id) const ;
  virtual unsigned int sizeForDenseIndex() const;
  
  virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const ;
  
  void addValidID(const DetId& id);
  unsigned int getClosestCellIndex ( const GlobalPoint& r ) const;
  
private:
  const HGCalTopology&    mTopology;
  
  CellVec                 m_cellVec ; 
  std::vector<DetId>      m_validGeomIds;
  bool                    m_halfType;
  ForwardSubdetector      m_subdet;
};

#endif
