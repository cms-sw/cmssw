//-------------------------------------------------
//
/**  \class L1MuBMTrackSegPhi
 *
 *   PHI Track Segment
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TRACK_SEG_PHI_H
#define L1MUBM_TRACK_SEG_PHI_H

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

class L1MuBMTrackSegPhi;

typedef std::vector<L1MuBMTrackSegPhi> L1MuBMTrackSegPhiCollection;

class L1MuBMTrackSegPhi {

  public:

    /// quality code of BBMX phi track segments
    enum TSQuality { Li, Lo, Hi, Ho, LL, HL, HH, Null };

    /// default constructor
    L1MuBMTrackSegPhi();

    /// constructor
    L1MuBMTrackSegPhi(int wheel_id, int sector_id, int station_id,
                      int phi = 0, int phib = 0,
                      TSQuality quality = Null, bool tag = false, int bx = 17,
                      bool etaFlag = false);

    /// constructor
    L1MuBMTrackSegPhi(const L1MuBMTrackSegLoc&,
                      int phi = 0, int phib = 0,
                      TSQuality quality = Null, bool tag = false, int bx = 17,
                      bool etaFlag = false);

    /// copy constructor
    L1MuBMTrackSegPhi(const L1MuBMTrackSegPhi&);

    /// destructor
    virtual ~L1MuBMTrackSegPhi();

    /// reset phi track segment
    void reset();

    /// return phi-value in global coordinates [0,2pi]
    double phiValue() const;

    /// return phib-value in global coordinates [0,2pi]
    double phibValue() const;

    /// return wheel
    inline int wheel() const { return m_location.wheel(); }

    /// return sector
    inline int sector() const { return m_location.sector(); }

    /// return station
    inline int station() const { return m_location.station(); }

    /// return location of phi track segment
    inline const L1MuBMTrackSegLoc& where() const{ return m_location; }

    /// return phi
    inline int phi() const { return m_phi; }

    /// return phib
    inline int phib() const { return m_phib; }

    /// return quality code
    inline int quality() const { return m_quality; }

    /// return tag (second TS tag)
    inline int tag() const { return m_tag; }

    /// return bunch crossing
    inline int bx() const { return m_bx; }

    /// return eta flag
    inline bool etaFlag() const { return m_etaFlag; }

    /// is it an empty phi track segment?
    inline bool empty() const { return m_quality == Null; }

    /// set eta flag
    inline void setEtaFlag(bool flag) { m_etaFlag = flag; }

    /// assignment operator
    L1MuBMTrackSegPhi& operator=(const L1MuBMTrackSegPhi&);

    /// equal operator
    bool operator==(const L1MuBMTrackSegPhi&) const;

    /// unequal operator
    bool operator!=(const L1MuBMTrackSegPhi&) const;

    /// overload output stream operator for phi track segment quality
    friend std::ostream& operator<<(std::ostream&, const TSQuality&);

    /// overload output stream operator for phi track segments
    friend std::ostream& operator<<(std::ostream&, const L1MuBMTrackSegPhi&);

  private:

    L1MuBMTrackSegLoc m_location;   // logical location of TS
    int               m_phi;        // 12 bits
    int               m_phib;       // 10 bits
    TSQuality         m_quality;    // 3 bits
    bool              m_tag;        // tag for second TS (of chamber)
    int               m_bx;         // bunch crossing identifier
    bool              m_etaFlag;    // eta flag (for overlap region)

};

#endif
