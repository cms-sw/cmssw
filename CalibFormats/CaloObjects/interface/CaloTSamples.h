#ifndef CALOTSAMPLES_H
#define CALOTSAMPLES_H 1

#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"

/** \class CaloTSamples
    
Class which represents the charge/voltage measurements of an event/channel
with the ADC decoding performed.

*/

template <class Ttype, uint32_t Tsize> 
class CaloTSamples : public CaloTSamplesBase<Ttype>
{
   public:

      enum { kCapacity = Tsize } ;

      CaloTSamples<Ttype,Tsize>()  ;
      CaloTSamples<Ttype,Tsize>( const CaloTSamples<Ttype,Tsize>& cs )  ;
      CaloTSamples<Ttype,Tsize>( const DetId& id   , 
				 uint32_t size = 0 ,
				 uint32_t pre  = 0  ) ;
      virtual ~CaloTSamples<Ttype,Tsize>() ;

      CaloTSamples<Ttype,Tsize>& operator=( const CaloTSamples<Ttype,Tsize>& cs ) ;

      virtual uint32_t capacity() const ;

   private:

      virtual       Ttype* data(  uint32_t i ) ;
      virtual const Ttype* cdata( uint32_t i ) const ;

      Ttype m_data[ Tsize ] ;
} ;

#endif
