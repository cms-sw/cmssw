//-------------------------------------------------
//
/**  \class DTSectCollId  
 *   Definition of a Sector Coollector
 *
 *
 *   $Date: 2004/03/18 09:43:24 $
 *
 *   \author D. Bonacorsi, S. Marcellini
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
    _station(0),
    _sector( 0) {}

/*  DTSectCollId(myint8 wheel_id,  */
/* 	      myint8 station_id,  */
/* 	      myint8 sector_id):  */
/*     _wheel(wheel_id), */
/*     _station(station_id), */
/*     _sector(sector_id) {} */
 DTSectCollId(int wheel_id, 
	      int station_id, 
	      int sector_id): 
    _wheel(wheel_id),
    _station(station_id),
    _sector(sector_id) {}
  

   DTSectCollId(const  DTSectCollId& statId) :
    _wheel(statId._wheel),
    _station(statId._station),
    _sector(statId._sector) {}


  // Destructor

  // Operations 
  inline int wheel()   const { return _wheel; }
  inline int station() const { return _station; }
  inline int sector()  const { return _sector; }

  inline bool operator == ( const DTSectCollId & ) const;
  inline bool operator < ( const  DTSectCollId& ) const;
  
  inline  DTSectCollId & operator = ( const  DTSectCollId& );

 private:
/*   myint8 _wheel; */
/*   myint8 _station; */
/*   myint8 _sector; */
  int _wheel;
  int _station;
  int _sector;

};

#include <iosfwd>
std::ostream& operator<<(std::ostream &os, const  DTSectCollId& id)  ;
#include "L1Trigger/DTSectorCollector/interface/DTSectCollId.icc"

#endif









