#ifndef EcalEndcapGeometry_h
#define EcalEndcapGeometry_h

#include "DataFormats/EcalDetId/interface/EEDetId.h"
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

      virtual CaloSubdetectorGeometry::DetIdSet getCells( const GlobalPoint& r,
							  double             dR ) const ;

      void initialize();

      static std::string hitString() { return "EcalHitsEE" ; }

      static std::string producerName() { return "EcalEndcap" ; }

   private:

      /// number of modules
      int _nnmods;
  
      /// number of crystals per module
      int _nncrys; 

      float zeP, zeN;

      float m_wref, m_href ;

      unsigned int m_nref ;

      unsigned int index( float x ) const ;
      EEDetId gId( float x, float y, float z ) const ;
} ;


#endif

