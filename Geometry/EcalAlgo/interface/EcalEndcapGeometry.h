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

class EcalEndcapGeometry : public CaloSubdetectorGeometry 
{
   public:

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

      static std::string hitString() { return "EcalHitsEE" ; }

      static std::string producerName() { return "EcalEndcap" ; }

      static unsigned int numberOfAlignments() { return 4 ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static std::vector<HepPoint3D> localCorners( const double* pv,
						   unsigned int  i,
						   HepPoint3D&   ref ) ;

      static CaloCellGeometry* newCell( const GlobalPoint& f1 ,
					const GlobalPoint& f2 ,
					const GlobalPoint& f3 ,
					CaloCellGeometry::CornersMgr* mgr,
					const double*      parm ) ;

   private:

      static int myPhi( int i ) { i+=720; return ( 1 + (i-1)%360 ) ; }

      /// number of modules
      int _nnmods;
  
      /// number of crystals per module
      int _nncrys; 

      float zeP, zeN;

      float m_wref, m_href ;

      unsigned int m_nref ;

      unsigned int index( float x ) const ;
      EEDetId gId( float x, float y, float z ) const ;

      mutable EZMgrFL<EBDetId>*     m_borderMgr ;

      mutable VecOrdListEBDetIdPtr* m_borderPtrVec ;
} ;


#endif

