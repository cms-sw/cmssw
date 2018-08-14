//-------------------------------------------------
//
/**  \class L1MuBMExtrapolationUnit
 *
 *   Extrapolation Unit:
 *
 *   The Extrapolation Unit attempts to join
 *   track segment pairs of different stations.
 *   it contains 12 Single Extrapolation Units
 *   to perform all extrapolations in its
 *   own wheel and 6 Single Extrapolation Units
 *   to perform all extrapolations
 *   in the adjacent wheel (next wheel neighbour)
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_EXTRAPOLATION_UNIT_H
#define L1MUBM_EXTRAPOLATION_UNIT_H

//---------------
// C++ Headers --
//---------------

#include <utility>
#include <map>
#include <bitset>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTExtParam.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include <FWCore/Framework/interface/ESHandle.h>
class L1MuBMSectorProcessor;
class L1MuBMSEU;
class L1MuDTTFParameters;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMExtrapolationUnit : public L1AbstractProcessor {

  public:

    typedef std::pair<Extrapolation, unsigned int> SEUId;
    typedef std::map<SEUId, L1MuBMSEU*, std::less<SEUId> > SEUmap;

    /// constructor
    L1MuBMExtrapolationUnit(const L1MuBMSectorProcessor& );

    /// destructor
    ~L1MuBMExtrapolationUnit() override;

    /// run Extrapolation Unit
    void run(const edm::EventSetup& c) override;

    /// reset Extrapolation Unit
    void reset() override;

     /// reset a single extrapolation
    void reset(Extrapolation ext, unsigned int startAdr, unsigned int relAdr );

    /// get extrapolation address from a given ERS
    unsigned short int getAddress(Extrapolation ext, unsigned int startAdr, int id) const;

    /// get extrapolation quality from a given ERS
    unsigned short int getQuality(Extrapolation ext, unsigned int startAdr, int id) const;

    /// get Extrapolator table for a given SEU
    const std::bitset<12>& getEXTable(Extrapolation ext, unsigned int startAdr) const;

    /// get Quality Sorter table for a given SEU
    const std::bitset<12>& getQSTable(Extrapolation ext, unsigned int startAdr) const;

    /// return number of successful extrapolations
    int numberOfExt() const;

    /// print all successful extrapolations
    void print(int level=0) const;

    /// return station of start and target track segment for a given extrapolation
    static std::pair<int,int> which_ext(Extrapolation ext);

  private:

    const L1MuBMSectorProcessor& m_sp;   // reference to Sector Processor

    mutable SEUmap m_SEUs;               // Single Extrapolation Units

    //edm::ESHandle< L1MuDTTFParameters > pars;
    edm::ESHandle< L1TMuonBarrelParams > bmtfParamsHandle;
    L1MuDTTFParameters  pars;

};

#endif
