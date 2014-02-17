//-------------------------------------------------
//
/**  \class L1MuDTTrackSegLoc
 *
 *   Logical location of a Track Segment:
 *
 *   The location of a track segment is
 *   given by a triple (wheel, sector, station)
 *   with wheel: -3, -2, -1, 0, +1, +2, +3
 *   ( -3, +3 : are forward- and backward-endcaps),<BR>
 *   sector: 0-11 (30 deg sectors!)<BR>
 *   station: 1-5 (station 5=ME13)
 *
 *
 *   $Date: 2007/02/27 11:44:00 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP 
 */
//
//--------------------------------------------------
#ifndef L1MUDT_TRACK_SEG_LOC_H
#define L1MUDT_TRACK_SEG_LOC_H

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

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTTrackSegLoc {

  public:
 
    /// default constructor
    L1MuDTTrackSegLoc();

    /// constructor
    L1MuDTTrackSegLoc(int wheel_id, 
                      int sector_id,
                      int station_id); 
  
    /// copy constructor
    L1MuDTTrackSegLoc(const L1MuDTTrackSegLoc&);

    /// destructor
    virtual ~L1MuDTTrackSegLoc();

    /// return wheel 
    inline int wheel() const { return m_wheel; }
    
    /// return sector (30 deg)
    inline int sector() const { return m_sector; }
    
    /// return station
    inline int station() const { return m_station; }

    /// assignment operator
    L1MuDTTrackSegLoc& operator=(const L1MuDTTrackSegLoc&);

    /// equal operator
    bool operator==(const L1MuDTTrackSegLoc&) const;
    
    /// unequal operator
    bool operator!=(const L1MuDTTrackSegLoc&) const;
    
    /// less operator
    bool operator<(const L1MuDTTrackSegLoc&) const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuDTTrackSegLoc&);

  private:
 
    int m_wheel;
    int m_sector;
    int m_station;
  
};
  
#endif
