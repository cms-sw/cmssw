#ifndef EcalBarrelGeometry_h
#define EcalBarrelGeometry_h

#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <vector>

class EcalBarrelGeometry : public CaloSubdetectorGeometry 
{
   public:

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

      // Get closest cell, etc...
      virtual DetId getClosestCell( const GlobalPoint& r ) const ;

      virtual CaloSubdetectorGeometry::DetIdSet getCells( const GlobalPoint& r,
							  double             dR ) const ;

      static std::string hitString() { return "EcalHitsEB" ; }

      static std::string producerName() { return "EcalBarrel" ; }

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
};


#endif

