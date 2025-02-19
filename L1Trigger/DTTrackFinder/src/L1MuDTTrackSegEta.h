//-------------------------------------------------
//
/**  \class L1MuDTTrackSegEta
 *
 *   ETA Track Segment
 *
 *
 *   $Date: 2007/02/27 11:44:00 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_TRACK_SEG_ETA_H
#define L1MUDT_TRACK_SEG_ETA_H

//---------------
// C++ Headers --
//---------------

#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTTrackSegEta {

  public:
 
    /// default constructor
    L1MuDTTrackSegEta();

    /// constructor
    L1MuDTTrackSegEta(int wheel_id, int sector_id, int station_id, 
                      int position = 0, int quality = 0, int bx = 17);

    /// constructor
    L1MuDTTrackSegEta(const L1MuDTTrackSegLoc&, 
                      int position = 0, int quality = 0, int bx = 17); 
  
    /// copy constructor
    L1MuDTTrackSegEta(const L1MuDTTrackSegEta&);

    /// destructor
    virtual ~L1MuDTTrackSegEta();

    /// reset eta track segment 
    void reset();
    
    /// return wheel
    inline int wheel() const { return m_location.wheel(); }
    
    /// return sector
    inline int sector() const { return m_location.sector(); }
    
    /// return station
    inline int station() const { return m_location.station(); }
    
    /// return location of eta track segment
    inline const L1MuDTTrackSegLoc& where() const{ return m_location; }
    
    /// return position
    inline unsigned int position() const { return m_position; }
    
    /// return quality code
    inline unsigned int quality() const { return m_quality; }
    
    /// return bunch crossing
    inline int bx() const { return m_bx; }

    /// is it an empty eta track segment?
    inline bool empty() const { return m_position == 0; } 
    
    /// assignment operator
    L1MuDTTrackSegEta& operator=(const L1MuDTTrackSegEta&);

    /// equal operator
    bool operator==(const L1MuDTTrackSegEta&) const;
    
    /// unequal operator
    bool operator!=(const L1MuDTTrackSegEta&) const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuDTTrackSegEta&);

  private:

    L1MuDTTrackSegLoc  m_location;   // logical location of a TS
    unsigned int       m_position;   // 7 bits
    unsigned int       m_quality;    // 7 bits
    int                m_bx;         // bunch crossing identifier

};
  
#endif
