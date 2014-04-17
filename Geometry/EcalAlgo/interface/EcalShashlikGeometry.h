#ifndef EcalShashlikGeometry_h
#define EcalShashlikGeometry_h

/*
 * Geometry for Shashlik ECAL
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 * Fedor Ratnikov, Apr. 8 2014
 */

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h" 
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/EcalShashlikGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h" 
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "Geometry/HGCalCommonData/interface/EcalShashlikNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/PEcalShashlikRcd.h"
#include <vector>
#include <map>

class TruncatedPyramid;

class EcalShashlikGeometry GCC11_FINAL: public CaloSubdetectorGeometry 
{
 public:
  
  typedef std::vector<TruncatedPyramid> CellVec ;
  
  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
  
  typedef IdealGeometryRecord      IdealRecord   ;
  typedef EcalShashlikGeometryRecord AlignedRecord ;
  typedef EEAlignmentRcd           AlignmentRecord ; 
  typedef PEcalShashlikRcd           PGeometryRecord ; 
  
  typedef EZArrayFL<EBDetId> OrderedListOfEBDetId ; // like an stl vector: begin(), end(), [i] 
  
  typedef std::vector<OrderedListOfEBDetId*>  VecOrdListEBDetIdPtr ; 
  
  typedef EcalShashlikNumberingScheme NumberingScheme ;
  
  typedef EKDetId DetIdType ;
  
  //enum { k_NumberOfCellsForCorners =  EKDetId::kSizeForDenseIndexing } ; 
  
  enum { k_NumberOfShapes = 1 } ; 
  
  enum { k_NumberOfParametersPerShape = 11 } ; 
  
  
  static std::string dbString() { return "PEcalShashlikRcd" ; }
  
  EcalShashlikGeometry() ;
  EcalShashlikGeometry(const ShashlikTopology& topology) ;
  
  virtual ~EcalShashlikGeometry();
  
  void allocateCorners( CaloCellGeometry::CornersVec::size_type n ) ;

  void addValidID(const DetId& id);

  // avoid sorting set in base class  
  virtual const std::vector<DetId>& getValidDetIds( DetId::Detector det, int subdet) const {return m_validIds;}
  
  virtual void newCell( const GlobalPoint& f1 ,
			const GlobalPoint& f2 ,
			const GlobalPoint& f3 ,
			const CCGFloat*    parm ,
			const DetId&       detId );
  
  /// Get the cell geometry of a given detector id.  Should return false if not found.
  virtual const CaloCellGeometry* getGeometry( const DetId& id ) const ;
  
  
  const OrderedListOfEBDetId* getClosestBarrelCells( EKDetId id ) const ;
  
  // Get closest cell
  virtual DetId getClosestCell( const GlobalPoint& r ) const ;
  
  virtual CaloSubdetectorGeometry::DetIdSet getCells( const GlobalPoint& r,
						      double             dR ) const ;
  virtual void fillNamedParams (DDFilteredView fv);
  virtual void initializeParms() ;
  
  CCGFloat avgAbsZFrontFaceCenter() const ; // average over both endcaps. Positive!
  
  static std::string producerTag() { return "EcalShashlik" ; }
  static const char* cellElement() { return "ShashlikModule" ; }
  
  
  static unsigned int numberOfAlignments() { return 4 ; }

  unsigned int alignmentTransformIndexLocal( const DetId& id ) const;
  
  static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;
  
  DetId detIdFromLocalAlignmentIndex( unsigned int iLoc ) const;
  
  static void localCorners( Pt3DVec&        lc  ,
			    const CCGFloat* pv  ,
			    unsigned int    i   ,
			    Pt3D&           ref   ) ;
  
  const ShashlikTopology& topology () const;
  
 protected:
  
  virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const ;
  virtual unsigned int indexFor(const DetId& id) const { 
    return  mTopology.cell2denseId(id); 
  }
  virtual unsigned int sizeForDenseIndex(const DetId& id) const { 
    return mTopology.cellHashSize(); 
  }

  
 private:
  ShashlikTopology mTopology;
  bool initializedTopology;
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

