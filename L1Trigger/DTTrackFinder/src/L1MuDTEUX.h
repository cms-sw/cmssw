//-------------------------------------------------
//
/**  \class L1MuDTEUX
 *
 *   Extrapolator:
 *
 *   The Extrapolator receives a start and 
 *   a target track segment and performs the 
 *   actual extrapolation
 *
 *
 *   $Date: 2008/11/28 10:30:51 $
 *   $Revision: 1.5 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_EUX_H
#define L1MUDT_EUX_H

//---------------
// C++ Headers --
//---------------

#include <utility>
#include <map>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/DTTrackFinder/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/Framework/interface/ESHandle.h>
class L1MuDTTrackSegPhi;
class L1MuDTSectorProcessor;
class L1MuDTSEU;
class L1MuDTExtLut;
class L1MuDTTFParameters;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTEUX : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuDTEUX(const L1MuDTSectorProcessor& sp, const L1MuDTSEU& seu, int id );

    /// destructor
    virtual ~L1MuDTEUX();

    /// equal operator
    bool operator==(const L1MuDTEUX&) const;
    
    /// run Extrapolator
    virtual void run(const edm::EventSetup& c);
    
    /// reset Extrapolator
    virtual void reset();
    
    /// load data into EUX 
    void load(const L1MuDTTrackSegPhi* start_ts, const L1MuDTTrackSegPhi* target_ts);

    /// return pointer to start and target track segment
    std::pair<const L1MuDTTrackSegPhi*, const L1MuDTTrackSegPhi*> ts() const;

    /// set precision of phi and phib
    static void setPrecision();
    
    /// helper class for finding the best and second best extrapolation
    class EUX_Comp : std::binary_function< L1MuDTEUX*, L1MuDTEUX*, bool> {
      public :
        EUX_Comp( const L1MuDTEUX* k = 0 ) : _not(k) {}
        bool operator()( const L1MuDTEUX* first, const L1MuDTEUX* second ) const {
          if ( !second->result() ) return false;
          if ( _not != 0 && *first  == *_not ) return true; 
          if ( _not != 0 && *second == *_not ) return false;
          return ( first->quality() < second->quality() );
        }
      private:
        const L1MuDTEUX* _not;
    };

    /// return Extrapolation identifier
    inline int id() const { return m_id; }
    
    /// return extrapolation result
    inline bool result() const { return m_result; }
    
    /// return extrapolation quality
    inline unsigned int quality() const { return m_quality; }
    
    /// return extrapolation address (0-11) (address = 15 indicates negative ext. result)
    inline unsigned short int address() const { return m_address; }

  private:

    /// output sector numbers in the range -6 to +5
    int sec_mod(int) const;

  private:

    const L1MuDTSectorProcessor& m_sp;
    const L1MuDTSEU& m_seu;           // reference to Single Extrapolation Unit 
    int              m_id;            // index of start TS

    bool               m_result;      //@@ 1 bit
    unsigned short int m_quality;     //@@ 1 bit
    unsigned short int m_address;     //@@ 4 bits

    const L1MuDTTrackSegPhi* m_start;       // start track segment
    const L1MuDTTrackSegPhi* m_target;      // target track segment
      
    edm::ESHandle< L1MuDTExtLut > theExtLUTs;  // extrapolation look-up tables
    static int               theExtFilter;     // extrapolation quality filter
    static unsigned short    nbit_phi;         // number of bits used for phi
    static unsigned short    nbit_phib;        // number of bits used for phib

    edm::ESHandle< L1MuDTTFParameters > pars;

};

#endif
