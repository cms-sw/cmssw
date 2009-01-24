#ifndef HcalCastorDetId_h_included
#define HcalCastorDetId_h_included 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

/** \class HcalCastorDetId
  *  
  *  Contents of the HcalCastorDetId :
  *     [9]   Z position (true for positive)
  *     [8:7] Section (EM/HAD)
  *     [6:4] sector (depth)
  *	[3:0] module

  * NEW:
  * [8]   z position
  * [7:4] sector 
  * [3:0] module

  * \author P. Katsas, T. McCauley
  */

class HcalCastorDetId : public DetId 
{
   public:

      enum Section { Unknown=0, EM=1, HAD=2 };

      // 1 => CaloTower, 2 => ZDC, 3 => Castor
      static const int SubdetectorId = 3;

      /** Create a null cellid*/
      HcalCastorDetId();
  
      /** Create cellid from raw id (0=invalid tower id) */
      HcalCastorDetId(uint32_t rawid);
  
      /** Constructor from section, z-side, sector and module */
      HcalCastorDetId( Section section,
		       bool true_for_positive_eta,
		       int sector,
		       int module);

      // constructor without section
      HcalCastorDetId( bool true_for_positive_eta,
		       int sector,
		       int module );

      /** Constructor from a generic cell id */
      HcalCastorDetId(const DetId& id);
  
      /** Assignment from a generic cell id */
      HcalCastorDetId& operator=(const DetId& id);

      /// get the z-side of the cell (1/-1)
      //int zside() const { return (id_&0x40)?(1):(-1); }
      int zside() const { return 2*( ( id_ >> 8 ) & 0x1 ) - 1 ; }

      /// get the section
      //Section section() const { return (Section)((id_>>7)&0x3); }
      Section section() const ;
    
      /// get the module (1-2 for EM, 1-12 for HAD)
      //int module() const { return id_&0xF; }
      int module() const { return ( id_ & 0xF ) ; }
    
      /// get the sector (1-16)
      //int sector() const { return (id_>>6)&0x3; }
      int sector() const { return ((id_ >> 4) & 0xF) + 1 ; }
   
      // get the individual cell id
      //  int channel() const;

      enum { kNumberModulesPerEnd  = 14 ,
	     kNumberSectorsPerEnd  = 16 ,
	     kNumberCellsPerEnd    = kNumberModulesPerEnd*kNumberSectorsPerEnd ,
	     kSizeForDenseIndexing = kNumberCellsPerEnd } ;

      uint32_t denseIndex() const ;

      static bool validDetId( Section iSection,
			      bool    posEta,
			      int     iSector,
			      int     iMod     ) ;

      static bool validDenseIndex( uint32_t din ) 
      { return ( din < kSizeForDenseIndexing ) ; }

      static HcalCastorDetId detIdFromDenseIndex( uint32_t di ) ;

   private:

      void buildMe( Section section,
		    bool    true_for_positive_eta,
		    int     sector,
		    int     module ) ;
};

std::ostream& operator<<(std::ostream&,const HcalCastorDetId& id);


#endif // HcalCastorDetId_h_included

