//-------------------------------------------------
//
/**  \class L1MuBMEUX
 *
 *   Extrapolator:
 *
 *   The Extrapolator receives a start and
 *   a target track segment and performs the
 *   actual extrapolation
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_EUX_H
#define L1MUBM_EUX_H

//---------------
// C++ Headers --
//---------------

#include <utility>
#include <map>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1AbstractProcessor.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMLUTHandler.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/Framework/interface/ESHandle.h>
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"

class L1MuBMTrackSegPhi;
class L1MuBMSectorProcessor;
class L1MuBMSEU;
class L1MuDTExtLut;
class L1MuDTTFParameters;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMEUX : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuBMEUX(const L1MuBMSectorProcessor& sp, const L1MuBMSEU& seu, int id );

    /// destructor
    ~L1MuBMEUX() override;

    /// equal operator
    bool operator==(const L1MuBMEUX&) const;

    /// run Extrapolator
    void run(const edm::EventSetup& c) override;

    /// reset Extrapolator
    void reset() override;

    /// load data into EUX
    void load(const L1MuBMTrackSegPhi* start_ts, const L1MuBMTrackSegPhi* target_ts);

    /// return pointer to start and target track segment
    std::pair<const L1MuBMTrackSegPhi*, const L1MuBMTrackSegPhi*> ts() const;

    /// helper class for finding the best and second best extrapolation
    class EUX_Comp : std::binary_function< L1MuBMEUX*, L1MuBMEUX*, bool> {
      public :
        EUX_Comp( const L1MuBMEUX* k = nullptr ) : _not(k) {}
        bool operator()( const L1MuBMEUX* first, const L1MuBMEUX* second ) const {
          if ( !second->result() ) return false;
          if ( _not != nullptr && *first  == *_not ) return true;
          if ( _not != nullptr && *second == *_not ) return false;
          return ( first->quality() < second->quality() );
        }
      private:
        const L1MuBMEUX* _not;
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

    const L1MuBMSectorProcessor& m_sp;
    const L1MuBMSEU& m_seu;           // reference to Single Extrapolation Unit
    int              m_id;            // index of start TS

    bool               m_result;      //@@ 1 bit
    unsigned short int m_quality;     //@@ 1 bit
    unsigned short int m_address;     //@@ 4 bits

    const L1MuBMTrackSegPhi* m_start;       // start track segment
    const L1MuBMTrackSegPhi* m_target;      // target track segment

    edm::ESHandle< L1TMuonBarrelParams > bmtfParamsHandle;

    //edm::ESHandle< L1MuDTExtLut > theExtLUTs;  // extrapolation look-up tables
    L1MuBMLUTHandler* theExtLUTs;
    int            const   theExtFilter = 1;        // extrapolation quality filter
    unsigned short const   nbit_phi = 12;           // number of bits used for phi
    unsigned short const   nbit_phib = 10;          // number of bits used for phib

    //edm::ESHandle< L1MuDTTFParameters > pars;
    L1MuDTTFParameters pars;

};

#endif
