#ifndef GEOMETRY_CALOGEOMETRY_CALOCELLCROSSING_H
#define GEOMETRY_CALOGEOMETRY_CALOCELLCROSSING_H 1

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include <vector>
#include <string>

class CaloCellCrossing 
{
   public:

      typedef std::vector< DetId >        DetIds ;
      typedef std::vector< GlobalPoint >  Points;
      typedef std::vector< double >       Lengths ;

      CaloCellCrossing( const GlobalPoint&             gp ,
			const GlobalVector&            gv ,
			const DetIds*                  di ,
			const CaloSubdetectorGeometry* sg ,
			DetId::Detector                det ,
			int                            subdet,
			double                         small  = 1.e-10,
			bool                           onewayonly = false ) ;

      virtual ~CaloCellCrossing() {} ;

      const GlobalPoint&  gp() const { return m_gp ; }
      const GlobalVector& gv() const { return m_gv ; }

      const DetIds&  detIds()    const { return m_detId ; }
      const Points&  centers()   const { return m_ctr   ; }
      const Points&  entrances() const { return m_entr  ; }
      const Points&  exits()     const { return m_exit  ; }
      const Lengths& lengths()   const { return m_len   ; }

      CaloCellCrossing( const CaloCellCrossing& ) = delete;
      CaloCellCrossing operator=( const CaloCellCrossing& ) = delete;

   private:

      GlobalPoint  m_gp ;
      GlobalVector m_gv ;

      DetIds  m_detId ;

      Points  m_ctr  ;

      Points  m_entr ;
      Points  m_exit ;
      Lengths m_len  ;

};

std::ostream& operator<<( std::ostream& s, const CaloCellCrossing& cell ) ;

#endif
