#ifndef EcalBarrelGeometry_h
#define EcalBarrelGeometry_h

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/EcalBarrelGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <vector>

class EcalBarrelGeometry : public CaloSubdetectorGeometry 
{
   public:

      typedef IdealGeometryRecord      IdealRecord   ;
      typedef EcalBarrelGeometryRecord AlignedRecord ;
      typedef EBAlignmentRcd           AlignmentRecord ;

      typedef EZArrayFL<EEDetId> OrderedListOfEEDetId ; // like an stl vector: begin(), end(), [i]

      typedef std::vector<OrderedListOfEEDetId*>  VecOrdListEEDetIdPtr ;

      typedef EcalBarrelNumberingScheme NumberingScheme ;

      enum CornersCount { k_NumberOfCellsForCorners = 61200 } ;

      EcalBarrelGeometry() ;
  
      virtual ~EcalBarrelGeometry();

      int getNumXtalsPhiDirection()           const { return _nnxtalPhi ; }

      int getNumXtalsEtaDirection()           const { return _nnxtalEta ; }

      const std::vector<int>& getEtaBaskets() const { return _EtaBaskets ; }

      int getBasketSizeInPhi()                const { return _PhiBaskets ; }  

      void setNumXtalsPhiDirection( const int& nnxtalPhi )     { _nnxtalPhi=nnxtalPhi ; }

      void setNumXtalsEtaDirection( const int& nnxtalEta )     { _nnxtalEta=nnxtalEta ; }

      void setEtaBaskets( const std::vector<int>& EtaBaskets ) { _EtaBaskets=EtaBaskets ; }

      void setBasketSizeInPhi( const int& PhiBaskets )         { _PhiBaskets=PhiBaskets ; }

      const OrderedListOfEEDetId* getClosestEndcapCells( EBDetId id ) const ;

      // Get closest cell, etc...
      virtual DetId getClosestCell( const GlobalPoint& r ) const ;

      virtual CaloSubdetectorGeometry::DetIdSet getCells( const GlobalPoint& r,
							  double             dR ) const ;

      static std::string hitString() { return "EcalHitsEB" ; }

      static std::string producerName() { return "EcalBarrel" ; }

      static unsigned int numberOfAlignments() { return 36 ; }

   private:
      /** number of crystals in eta direction */
      int _nnxtalEta;
  
      /** number of crystals in phi direction */
      int _nnxtalPhi;
  
      /** size of the baskets in the eta direction. This is needed
	  to find out whether two adjacent crystals lie in the same
	  basked ('module') or not (e.g. this can be used for correcting
	  cluster energies etc.) */
      std::vector<int> _EtaBaskets;
      
      /** size of one basket in phi */
      int _PhiBaskets;

      mutable EZMgrFL<EEDetId>*     m_borderMgr ;

      mutable VecOrdListEEDetIdPtr* m_borderPtrVec ;
};


#endif

