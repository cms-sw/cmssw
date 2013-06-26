#ifndef EcalEndcapGeometry_h
#define EcalEndcapGeometry_h

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/EcalEndcapGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/PEcalEndcapRcd.h"
#include <vector>
#include <map>

class TruncatedPyramid;

class EcalEndcapGeometry GCC11_FINAL: public CaloSubdetectorGeometry 
{
   public:

      typedef std::vector<TruncatedPyramid> CellVec ;

      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

      typedef IdealGeometryRecord      IdealRecord   ;
      typedef EcalEndcapGeometryRecord AlignedRecord ;
      typedef EEAlignmentRcd           AlignmentRecord ;
      typedef PEcalEndcapRcd           PGeometryRecord ;

      typedef EZArrayFL<EBDetId> OrderedListOfEBDetId ; // like an stl vector: begin(), end(), [i]

      typedef std::vector<OrderedListOfEBDetId*>  VecOrdListEBDetIdPtr ;

      typedef EcalEndcapNumberingScheme NumberingScheme ;

      typedef EEDetId DetIdType ;

      enum { k_NumberOfCellsForCorners = EEDetId::kSizeForDenseIndexing } ;

      enum { k_NumberOfShapes = 1 } ;

      enum { k_NumberOfParametersPerShape = 11 } ;


      static std::string dbString() { return "PEcalEndcapRcd" ; }

      virtual unsigned int numberOfShapes() const { return k_NumberOfShapes ; }
      virtual unsigned int numberOfParametersPerShape() const { return k_NumberOfParametersPerShape ; }

      EcalEndcapGeometry() ;
  
      virtual ~EcalEndcapGeometry();

      int getNumberOfModules()          const { return _nnmods ; }

      int getNumberOfCrystalPerModule() const { return _nncrys ; }

      void setNumberOfModules(          const int nnmods ) { _nnmods=nnmods ; }

      void setNumberOfCrystalPerModule( const int nncrys ) { _nncrys=nncrys ; }

      const OrderedListOfEBDetId* getClosestBarrelCells( EEDetId id ) const ;
      // Get closest cell, etc...
      virtual DetId getClosestCell( const GlobalPoint& r ) const ;

      virtual CaloSubdetectorGeometry::DetIdSet getCells( const GlobalPoint& r,
							  double             dR ) const ;

      virtual void initializeParms() ;

      CCGFloat avgAbsZFrontFaceCenter() const ; // average over both endcaps. Positive!

      static std::string hitString() { return "EcalHitsEE" ; }

      static std::string producerTag() { return "EcalEndcap" ; }

      static unsigned int numberOfAlignments() { return 4 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static DetId detIdFromLocalAlignmentIndex( unsigned int iLoc ) ;

      static void localCorners( Pt3DVec&        lc  ,
				const CCGFloat* pv  ,
				unsigned int    i   ,
				Pt3D&           ref   ) ;

      virtual void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm ,
			    const DetId&       detId   ) ;

   protected:

      virtual const CaloCellGeometry* cellGeomPtr( uint32_t index ) const ;

   private:

      static int myPhi( int i ) { i+=720; return ( 1 + (i-1)%360 ) ; }

      /// number of modules
      int _nnmods;
  
      /// number of crystals per module
      int _nncrys; 

      CCGFloat zeP, zeN;

      CCGFloat m_wref, m_xlo[2], m_xhi[2], m_ylo[2], m_yhi[2], m_xoff[2], m_yoff[2], m_del ;

      unsigned int m_nref ;

      unsigned int xindex( CCGFloat x, CCGFloat z ) const ;
      unsigned int yindex( CCGFloat y, CCGFloat z ) const ;

      EEDetId gId( float x, float y, float z ) const ;

      mutable EZMgrFL<EBDetId>*     m_borderMgr ;

      mutable VecOrdListEBDetIdPtr* m_borderPtrVec ;

      mutable CCGFloat m_avgZ ;

      CellVec m_cellVec ;
} ;


#endif

