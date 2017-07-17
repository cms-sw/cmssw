#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERGEOMETRY_H 1

#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/Records/interface/CaloTowerGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"

/** \class CaloTowerGeometry
  *  
  * Only DetId::Calo, subdet=1 DetIds are handled by this class.
  *
  * \author J. Mans - Minnesota
  */
class CaloTowerGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef std::vector<IdealObliquePrism> CellVec ;

      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

      typedef CaloTowerAlignmentRcd    AlignmentRecord ;
      typedef CaloTowerGeometryRecord  AlignedRecord   ;
      typedef PCaloTowerRcd            PGeometryRecord ;
      typedef CaloTowerDetId           DetIdType       ;

      //enum { k_NumberOfCellsForCorners = CaloTowerDetId::kSizeForDenseIndexing } ;

      //enum { k_NumberOfShapes = 41 } ;

      enum { k_NumberOfParametersPerShape = 5 } ;

      static std::string dbString() { return "PCaloTowerRcd" ; }

      unsigned int numberOfShapes() const override { return k_NumberOfShapes ; }
      unsigned int numberOfParametersPerShape() const override { return k_NumberOfParametersPerShape ; }
      virtual unsigned int numberOfCellsForCorners() const { return k_NumberOfCellsForCorners ; }


      CaloTowerGeometry(const CaloTowerTopology *cttopo_);
      ~CaloTowerGeometry() override;  

      static std::string producerTag() { return "TOWER" ; }

      static unsigned int numberOfAlignments() { return 0 ; }

      //static unsigned int alignmentTransformIndexLocal( const DetId& id ) ;
      unsigned int alignmentTransformIndexLocal( const DetId& id ) ;

      //static unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;
      unsigned int alignmentTransformIndexGlobal( const DetId& id ) ;

      static void localCorners( Pt3DVec&        lc  ,
				const CCGFloat* pv  , 
				unsigned int    i   ,
				Pt3D&           ref   ) ;

      void newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm,
			    const DetId&       detId     ) override ;
				
      const CaloCellGeometry* getGeometry( const DetId& id ) const override {
          return cellGeomPtr( cttopo->denseIndex(id) ) ;
      }

  void getSummary( CaloSubdetectorGeometry::TrVec&  trVector,
			   CaloSubdetectorGeometry::IVec&   iVector,
			   CaloSubdetectorGeometry::DimVec& dimVector,
			   CaloSubdetectorGeometry::IVec& dinsVector ) const override ;

   protected:

      const CaloCellGeometry* cellGeomPtr( uint32_t index ) const override ;
      unsigned int indexFor(const DetId& id) const override { return  cttopo->denseIndex(id); }
      unsigned int sizeForDenseIndex(const DetId& id) const override { return cttopo->sizeForDenseIndexing(); }

   private:
      const CaloTowerTopology* cttopo;
      int k_NumberOfCellsForCorners;
	  int k_NumberOfShapes;
      CellVec m_cellVec ;
	  CaloSubdetectorGeometry::IVec m_dins;
};

#endif
