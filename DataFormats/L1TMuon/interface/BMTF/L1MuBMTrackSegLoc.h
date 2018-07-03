//-------------------------------------------------
//
/**  \class L1MuBMTrackSegLoc
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
 *
 *   N. Neumeister            CERN EP 
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TRACK_SEG_LOC_H
#define L1MUBM_TRACK_SEG_LOC_H

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

class L1MuBMTrackSegLoc {

  public:
 
    /// default constructor
    L1MuBMTrackSegLoc();

    /// constructor
    L1MuBMTrackSegLoc(int wheel_id, 
                      int sector_id,
                      int station_id); 
  
    /// copy constructor
    L1MuBMTrackSegLoc(const L1MuBMTrackSegLoc&);

    /// destructor
    virtual ~L1MuBMTrackSegLoc();

    /// return wheel 
    inline int wheel() const { return m_wheel; }
    
    /// return sector (30 deg)
    inline int sector() const { return m_sector; }
    
    /// return station
    inline int station() const { return m_station; }

    /// assignment operator
    L1MuBMTrackSegLoc& operator=(const L1MuBMTrackSegLoc&);

    /// equal operator
    bool operator==(const L1MuBMTrackSegLoc&) const;
    
    /// unequal operator
    bool operator!=(const L1MuBMTrackSegLoc&) const;
    
    /// less operator
    bool operator<(const L1MuBMTrackSegLoc&) const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuBMTrackSegLoc&);

  private:
 
    int m_wheel;
    int m_sector;
    int m_station;
  
};
  
#endif
