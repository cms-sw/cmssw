//-------------------------------------------------
//
/**  \class MuonStub
 *
 *  Class that creates a super-primitive for all chambers
 *
 *
 *   M.Bachtis (UCLA)
 */
//
//--------------------------------------------------
#ifndef L1TMUPHASE2GMTSTUB_H
#define L1TMUPHASE2GMTSTUB_H
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

//#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackSegLoc.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace l1t {

  class MuonStub;

  typedef std::vector<MuonStub> MuonStubCollection;
  typedef edm::Ref<MuonStubCollection> MuonStubRef;
  typedef std::vector<edm::Ref<MuonStubCollection> > MuonStubRefVector;

  class MuonStub {
  public:
    /// default constructor
    MuonStub();

    /// constructor
    MuonStub(int etaRegion,
             int phiRegion,
             int depthRegion,
             uint tfLayer,
             int coord1,
             int coord2,
             int id,
             int bx,
             int quality,
             int eta1 = 0,
             int eta2 = 0,
             int etaQuality = -1,
             int type = 0);
    ~MuonStub();
    /// return wheel
    inline int etaRegion() const { return etaRegion_; }
    /// return sector
    inline int phiRegion() const { return phiRegion_; }
    /// return station
    inline int depthRegion() const { return depthRegion_; }
    /// return track finder layer
    inline uint tfLayer() const { return tfLayer_; }
    /// return phi
    inline int coord1() const { return coord1_; }
    /// return phib
    inline int coord2() const { return coord2_; }
    /// return quality code
    inline int quality() const { return quality_; }
    /// return tag (second TS tag)
    inline int id() const { return id_; }
    /// return bunch crossing
    inline int bxNum() const { return bxNum_; }

    /// return eta
    inline int eta1() const { return eta1_; }
    inline int eta2() const { return eta2_; }
    /// return first eta quality
    inline int etaQuality() const { return etaQuality_; }
    //return type
    inline int type() const { return type_; }

    inline bool isBarrel() const { return (type_ == 1); }
    inline bool isEndcap() const { return (type_ == 0); }

    inline double offline_coord1() const { return offline_coord1_; }
    inline double offline_coord2() const { return offline_coord2_; }
    inline double offline_eta1() const { return offline_eta1_; }
    inline double offline_eta2() const { return offline_eta2_; }

    void setOfflineQuantities(double coord1, double coord2, double eta1, double eta2) {
      offline_coord1_ = coord1;
      offline_coord2_ = coord2;
      offline_eta1_ = eta1;
      offline_eta2_ = eta2;
    }
    void setEta(int eta1, int eta2, int etaQ) {
      eta1_ = eta1;
      eta2_ = eta2;
      etaQuality_ = etaQ;
    }

    void setID(int id) { id_ = id; }
    /// equal operator
    bool operator==(const MuonStub&) const;
    /// unequal operator
    bool operator!=(const MuonStub&) const;

    void print() const;

  private:
    int etaRegion_;    //In the barrel this is wheel. In the endcap it is 6-ring
    int phiRegion_;    //In the barrel it is sector. In the endcap it is chamber
    int depthRegion_;  //Station
    uint tfLayer_;     //TF Layer
    int coord1_;       // global position angle in units of 30 degrees/2048
    int coord2_;       // bending angle  only in barrel for now
    int id_;           // stub id in case of more stubs per chamber
    int quality_;      //
    int bxNum_;        // bunch crossing identifier
    int eta1_;         // eta coordinate - in units of 3.0/512.
    int eta2_;         // eta coordinate - in units of 3.0/512.
    int etaQuality_;   // quality of the eta information
    int type_;         //Type: 0 TwinMux or DT, 1 RPC Barrel, 2 CSC, 3 RPC endcap
    /////////members that are not hardware but used for offline studies///////////////////////////////
    double offline_coord1_;  //offline coordinate 1
    double offline_coord2_;  //offline coordinate two
    double offline_eta1_;    //offline eta1
    double offline_eta2_;    //offline eta2
  };

}  // namespace l1t
#endif
