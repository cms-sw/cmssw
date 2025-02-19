//-------------------------------------------------
//
/**  \class L1MuDTEtaPattern
 *
 *   Pattern for Eta Track Finder:
 *
 *   An eta pattern consists of:
 *     - Pattern ID, 
 *     - quality code : [1,26],
 *     - eta code : [-32, +32],
 *     - position and wheel of hits in stations 1, 2, 3<BR>
 *       (wheel: -2, -1, 0, +1, +2, position : [1,7])
 * 
 *
 *   $Date: 2010/01/19 18:39:54 $
 *   $Revision: 1.3 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_ETA_PATTERN_H
#define L1MUDT_ETA_PATTERN_H

//---------------
// C++ Headers --
//---------------

#include <iosfwd>
#include <string>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTEtaPattern {

  public:
 
    /// default constructor
    L1MuDTEtaPattern();

    /// constructor
    L1MuDTEtaPattern(int id, int w1, int w2, int w3, int p1, int p2, int p3, 
                     int eta, int qual); 

    L1MuDTEtaPattern(int id, const std::string& pat, int eta, int qual);
                       
    /// copy constructor
    L1MuDTEtaPattern(const L1MuDTEtaPattern&);

    /// destructor
    virtual ~L1MuDTEtaPattern();

    /// return id 
    inline int id() const { return m_id; }
    
    /// return eta
    inline int eta() const { return m_eta; }
    
    /// return quality
    inline int quality() const { return m_qual; }

    /// return wheel number in station [1,3]
    inline int wheel(int station) const { return m_wheel[station-1]; }
    
    /// return position in station [1,3]
    inline int position(int station) const { return m_position[station-1]; }

    /// assignment operator
    L1MuDTEtaPattern& operator=(const L1MuDTEtaPattern&);

    /// equal operator
    bool operator==(const L1MuDTEtaPattern&) const;
    
    /// unequal operator
    bool operator!=(const L1MuDTEtaPattern&) const;
  
    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1MuDTEtaPattern&);

    /// input stream operator
    friend std::istream& operator>>(std::istream&, L1MuDTEtaPattern&);
    
  private:
 
    short m_id;
    short m_wheel[3]; 		// -2, -1, 0, +1, +2
    short m_position[3];          // position in wheel [1,7], 0 = empty
    short m_eta;                  // eta code: [-32, +32]
    short m_qual;                 // quality code: [0,26] 
  
};
  
#endif
