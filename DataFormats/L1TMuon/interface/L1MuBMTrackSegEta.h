//-------------------------------------------------
//
/**  \class L1MuBMTrackSegEta
 *
 *   ETA Track Segment
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TRACK_SEG_ETA_H
#define L1MUBM_TRACK_SEG_ETA_H

//---------------
// C++ Headers --
//---------------

#include <iosfwd>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackSegLoc.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMTrackSegEta;

typedef std::vector<L1MuBMTrackSegEta> L1MuBMTrackSegEtaCollection;

class L1MuBMTrackSegEta {

  public:

    /// default constructor
    L1MuBMTrackSegEta();

    /// constructor
    L1MuBMTrackSegEta(int wheel_id, int sector_id, int station_id,
                      int position = 0, int quality = 0, int bx = 17);

    /// constructor
    L1MuBMTrackSegEta(const L1MuBMTrackSegLoc&,
                      int position = 0, int quality = 0, int bx = 17);

    /// copy constructor
    L1MuBMTrackSegEta(const L1MuBMTrackSegEta&);

    /// destructor
    virtual ~L1MuBMTrackSegEta();

    /// reset eta track segment
    void reset();

    /// return wheel
    inline int wheel() const { return m_location.wheel(); }

    /// return sector
    inline int sector() const { return m_location.sector(); }

    /// return station
    inline int station() const { return m_location.station(); }

    /// return location of eta track segment
    inline const L1MuBMTrackSegLoc& where() const{ return m_location; }

    /// return position
    inline unsigned int position() const { return m_position; }

    /// return quality code
    inline unsigned int quality() const { return m_quality; }

    /// return bunch crossing
    inline int bx() const { return m_bx; }

    /// is it an empty eta track segment?
    inline bool empty() const { return m_position == 0; }

    /// assignment operator
    L1MuBMTrackSegEta& operator=(const L1MuBMTrackSegEta&);

    /// equal operator
    bool operator==(const L1MuBMTrackSegEta&) const;

    /// unequal operator
    bool operator!=(const L1MuBMTrackSegEta&) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuBMTrackSegEta&);

  private:

    L1MuBMTrackSegLoc  m_location;   // logical location of a TS
    unsigned int       m_position;   // 7 bits
    unsigned int       m_quality;    // 7 bits
    int                m_bx;         // bunch crossing identifier

};

#endif
