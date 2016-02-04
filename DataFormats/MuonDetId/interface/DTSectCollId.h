//-------------------------------------------------
//
/**  \class DTSectCollId  
 *   Definition of a Sector Coollector
 *
 *
 *   $Date: 2007/04/27 07:35:17 $
 *
 *   \authors 
 *           D. Bonacorsi, 
 *           S. Marcellini
 *   
 */
//
//--------------------------------------------------
#ifndef DT_SECT_COLL_ID_H
#define DT_SECT_COLL_ID_H

class DTSectCollId {

 public:
  //  Constructor
  DTSectCollId():
    _wheel(  0),
    _sector( 0) {}

  DTSectCollId(int wheel_id,  
	      int sector_id): 
    _wheel(wheel_id),
    _sector(sector_id) {}
  

   DTSectCollId(const  DTSectCollId& statId) :
    _wheel(statId._wheel),
    _sector(statId._sector) {}


  // Destructor

  // Operations 
  inline int wheel()   const { return _wheel; }
  inline int sector()  const { return _sector; }

  inline bool operator == ( const DTSectCollId & ) const;
  inline bool operator != ( const DTSectCollId & ) const;
  inline bool operator < ( const  DTSectCollId& ) const;
  
  inline  DTSectCollId & operator = ( const  DTSectCollId& );

 private:
  int _wheel;
  int _sector;

};

#include <iosfwd>
std::ostream& operator<<(std::ostream &os, const  DTSectCollId& id)  ;
#include "DataFormats/MuonDetId/interface/DTSectCollId.icc"

#endif









