//-------------------------------------------------
//
/**  \class DTTracoId
 *    TRACO Identifier
 *
 *   \author C.Grandi
 */
//
//--------------------------------------------------
#ifndef DT_TRACO_ID_H_
#define DT_TRACO_ID_H_

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

//---------------
// C++ Headers --
//---------------

//---------------------------------------------------
//                      DTTracoChip
//---------------------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTracoId {

 public:

  ///  Constructor
  DTTracoId() : _traco(0) {}

  ///  Constructor
  DTTracoId(const DTChamberId& mu_stat_id, 
                    const int traco_id) : _statId(mu_stat_id),
                                             _traco(traco_id) {}

  ///  Constructor
  DTTracoId(const int wheel_id, 
	        const int station_id, 
                const int sector_id, 
                const int traco_id) :
                  _statId(wheel_id,station_id,sector_id),
                  _traco(traco_id) {}
 
  ///  Constructor
  DTTracoId(const DTTracoId& tracoId) :
                  _statId(tracoId._statId), 
                  _traco(tracoId._traco) {}
 
  /// Destructor
  virtual ~DTTracoId() {}

  /// Returns wheel number
  inline int wheel()   const { return _statId.wheel(); }
  /// Returns station number
  inline int station() const { return _statId.station(); }
  /// Returns sector number
  inline int sector()  const { return _statId.sector(); }
  /// Returns the traco
  inline int traco()   const { return _traco; }
  /// Returns the chamber id
  inline DTChamberId ChamberId() const { return _statId; }

  bool operator == ( const DTTracoId& id) const { 
    if ( wheel()!=id.wheel()) return false;
    if ( sector()!=id.sector())return false;
    if ( station()!=id.station())return false;
    if ( _traco!=id.traco())return false;
    return true;
  }

  bool operator <  ( const DTTracoId& id) const { 
    if ( wheel()       < id.wheel()      ) return true;
    if ( wheel()       > id.wheel()      ) return false;
  
    if ( station()     < id.station()    ) return true;
    if ( station()     > id.station()    ) return false;
    
    if ( sector()      < id.sector()     ) return true;
    if ( sector()      > id.sector()     ) return false;

    if ( traco()         < id.traco()    ) return true;
    if ( traco()         > id.traco()    ) return false;

    return false;
  
  }


 private:
  DTChamberId _statId; // this is 3 bytes
  int _traco;

};

#endif
