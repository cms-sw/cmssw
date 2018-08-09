//-------------------------------------------------
//
/**  \class L1MuKBMCombinedStub
 *
 *  Class that combines TwinMux Phi and eta segments to a common segment to be used
 *  by the Kalman MuonTrack Finder
 *
 *
 *   M.Bachtis (UCLA)
 */
//
//--------------------------------------------------
#ifndef L1MUKBM_COMBINED_STUB_H
#define L1MUKBM_COMBINED_STUB_H

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
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/Common/interface/Ref.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuKBMTCombinedStub;

typedef std::vector<L1MuKBMTCombinedStub> L1MuKBMTCombinedStubCollection;
typedef edm::Ref<L1MuKBMTCombinedStubCollection > L1MuKBMTCombinedStubRef;
typedef std::vector<edm::Ref<L1MuKBMTCombinedStubCollection > > L1MuKBMTCombinedStubRefVector;

class L1MuKBMTCombinedStub {

  public:

    /// default constructor
    L1MuKBMTCombinedStub();

    /// constructor
    L1MuKBMTCombinedStub(int wheel,int sector,int station,int phi,int phiB,bool tag,int bx,int quality,int eta1=0,int eta2=0, int qeta1=-1,int qeta2=-1);
    ~L1MuKBMTCombinedStub();
    /// return wheel
    inline int whNum() const { return whNum_; }
    /// return sector
    inline int scNum() const { return scNum_; }
    /// return station
    inline int stNum() const { return stNum_; }
    /// return phi
    inline int phi() const { return phi_; }
    /// return phib
    inline int phiB() const { return phiB_; }
    /// return quality code
    inline int quality() const { return quality_; }
    /// return tag (second TS tag)
    inline int tag() const { return tag_; }
    /// return bunch crossing
    inline int bxNum() const { return bxNum_; }

    /// return first eta
    inline int eta1() const { return eta1_; }
    /// return second eta
    inline int eta2() const { return eta2_; }
    /// return first eta quality
    inline int qeta1() const { return qeta1_; }
    /// return second eta quality
    inline int qeta2() const { return qeta2_; }

    /// assignment operator
    L1MuKBMTCombinedStub& operator=(const L1MuKBMTCombinedStub&);
    /// equal operator
    bool operator==(const L1MuKBMTCombinedStub&) const;
    /// unequal operator
    bool operator!=(const L1MuKBMTCombinedStub&) const;

    /// overload output stream operator for phi track segments
    friend std::ostream& operator<<(std::ostream&, const L1MuKBMTCombinedStub&);

  private:

    int               whNum_;      
    int               scNum_;
    int               stNum_;
    int               phi_;        // 12 bits
    int               phiB_;       // 10 bits
    bool              tag_;        // tag for second TS (of chamber)
    int               quality_;    // 3 bits
    int               bxNum_;      // bunch crossing identifier
    int               eta1_;       //fine eta 1
    int               eta2_;       //fine eta 2
    int               qeta1_;      //fine eta quality 1
    int               qeta2_;      //fine eta quality 2 
};

#endif
