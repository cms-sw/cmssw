#ifndef HcalGeometry_h
#define HcalGeometry_h

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentRcd.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"

class HcalGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef HcalAlignmentRcd   AlignmentRecord ;
      typedef HcalGeometryRecord AlignedRecord   ;
      typedef PHcalRcd           PGeometryRecord ;
      typedef HcalDetId          DetIdType       ;

      enum { k_NumberOfCellsForCorners = HcalDetId::kSizeForDenseIndexing } ;

      enum { k_NumberOfShapes = 500 } ;

      enum { k_NumberOfParametersPerShape = 5 } ;

      static std::string dbString() { return "PHcalRcd" ; }

      virtual unsigned int numberOfShapes() const { return k_NumberOfShapes ; }
      virtual unsigned int numberOfParametersPerShape() const { return k_NumberOfParametersPerShape ; }


      HcalGeometry();

      HcalGeometry(const HcalTopology * topology);

      /// The HcalGeometry will delete all its cell geometries at destruction time
      virtual ~HcalGeometry();
  
      virtual const std::vector<DetId>& getValidDetIds(
	 DetId::Detector det    = DetId::Detector ( 0 ), 
	 int             subdet = 0 ) const;

      virtual DetId getClosestCell(const GlobalPoint& r) const ;
      
      virtual CaloSubdetectorGeometry::DetIdSet getCells( const GlobalPoint& r,
							  double             dR ) const ;


      static std::string producerTag() { return "HCAL" ; }

      static unsigned int numberOfBarrelAlignments() { return 36 ; }

      static unsigned int numberOfEndcapAlignments() { return 36 ; }

      static unsigned int numberOfOuterAlignments() { return 36 ; }

      static unsigned int numberOfForwardAlignments() { return 60 ; }

      static unsigned int numberOfAlignments() 
      { return ( numberOfBarrelAlignments() +
		 numberOfEndcapAlignments() +
		 numberOfOuterAlignments() +
		 numberOfForwardAlignments() ) ; }

      static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static std::vector<HepGeom::Point3D<double> > localCorners( const double* pv, 
						   unsigned int  i,
						   HepGeom::Point3D<double> &   ref ) ;

      static CaloCellGeometry* newCell( const GlobalPoint& f1 ,
					const GlobalPoint& f2 ,
					const GlobalPoint& f3 ,
					CaloCellGeometry::CornersMgr* mgr,
					const double*      parm,
					const DetId&       detId     ) ;
					

   private:

      void fillDetIds() const ;

      /// helper methods for getClosestCell
      int etaRing(HcalSubdetector bc, double abseta) const;
      int phiBin(double phi, int etaring) const;


      const HcalTopology * theTopology;

      mutable std::vector<DetId> m_hbIds ;
      mutable std::vector<DetId> m_heIds ;
      mutable std::vector<DetId> m_hoIds ;
      mutable std::vector<DetId> m_hfIds ;
      mutable std::vector<DetId> m_emptyIds ;
      bool m_ownsTopology ;
};


#endif

