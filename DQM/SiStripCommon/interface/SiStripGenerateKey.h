#ifndef DQM_SiStripCommon_SiStripGenerateKey_H
#define DQM_SiStripCommon_SiStripGenerateKey_H

#include <boost/cstdint.hpp>

using namespace std;

class SiStripGenerateKey {
  
 public:
  
  SiStripGenerateKey() {;}
  ~SiStripGenerateKey() {;}
  
  /** 32-bit key that uniquely identifies a FED channel. The key is
      built from the FED id and channel number. */
  inline static uint32_t fed( uint32_t fed_id, uint32_t fed_ch ) {
    return ( (fed_id&0xFFFF)<<16 ) | ( (fed_ch&0xFFFF)<<0 );
  }
  
  /** 32-bit key that uniquely identifies a module. The key is built
      from the crate, FEC, ring, CCU and module addresses. */
  inline static uint32_t module( uint32_t crate, 
				 uint32_t fec, 
				 uint32_t ring, 
				 uint32_t ccu, 
				 uint32_t module ) {
    return ( static_cast<uint32_t>( (crate&0x0F)<<28 ) | 
	     static_cast<uint32_t>( (fec&0xFF)<<20 ) | 
	     static_cast<uint32_t>( (ring&0x0F)<<16 ) | 
	     static_cast<uint32_t>( (ccu&0xFF)<<8 ) | 
	     static_cast<uint32_t>( (module&0xFF)<<0 ) );
  }

/*   /\** 32-bit key that uniquely identifies a FED channel. The key is */
/*       built from the FED id and channel number. *\/ */
/*   inline static pair<uint32_t,uint32_t> fed( uint32_t fed_key ) { */
/*     return pair<uint32_t,uint32_t>( (fed_id&0xFFFF)<<16 ) | ( (fed_ch&0xFFFF)<<0 ); */
/*   } */
  
/*   /\** 32-bit key that uniquely identifies a module. The key is built */
/*       from the crate, FEC, ring, CCU and module addresses. *\/ */
/*   inline static uint32_t module( uint32_t crate,  */
/* 				 uint32_t fec,  */
/* 				 uint32_t ring,  */
/* 				 uint32_t ccu,  */
/* 				 uint32_t module ) { */
/*     return ( static_cast<uint32_t>( (crate&0x0F)<<28 ) |  */
/* 	     static_cast<uint32_t>( (fec&0xFF)<<20 ) |  */
/* 	     static_cast<uint32_t>( (ring&0x0F)<<16 ) |  */
/* 	     static_cast<uint32_t>( (ccu&0xFF)<<8 ) |  */
/* 	     static_cast<uint32_t>( (module&0xFF)<<0 ) ); */
/*   } */

};

#endif // DQM_SiStripCommon_SiStripGenerateKey_H
