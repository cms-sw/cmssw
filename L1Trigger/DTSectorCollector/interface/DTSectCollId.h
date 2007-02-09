//-------------------------------------------------
//
/**  \class DTSectCollId  
 *   Definition of a Sector Coollector
 *
 *
 *   $Date: 2006/07/19 10:44:41 $
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


// SM typedef unsigned char myint8;

//typedef unsigned short myint16;


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
#include "L1Trigger/DTSectorCollector/interface/DTSectCollId.icc"

#endif









