/**
 * \class TrackStub
 * \author L. Gray
 *
 * A transient data class used to wrap a Correlated LCT
 * and give access to its eta and phi coordinates.
 * This is essentially the merging of a CSCDetId and a CorrelatedLCT
 * into one class.
 *
 * \remark Takes the place of both L1MuCSCCorrelatedLCT and L1MuTrackStub
 *        
 */
#ifndef L1CSCTrackFinder_TrackStub_h
#define L1CSCTrackFinder_TrackStub_h

#include <DataFormats/DetId/interface/DetId.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>

namespace csctf {

  class TrackStub : public CSCCorrelatedLCTDigi {
  public:
    TrackStub() {}
    TrackStub(const CSCCorrelatedLCTDigi &, const DetId &);
    TrackStub(const CSCCorrelatedLCTDigi &, const DetId &, const unsigned &phi, const unsigned &eta);
    TrackStub(const TrackStub &);

    /// set Eta and Phi from integer values.
    void setEtaPacked(const unsigned &eta_) { theEta_ = eta_; }
    void setPhiPacked(const unsigned &phi_) { thePhi_ = phi_; }

    /// return the Eta Value of this stub's position.
    double etaValue() const { return (theEta_ * theEtaBinning + CSCTFConstants::minEta); }
    /// return the Phi Value of this stub's position in local coordinates.
    double phiValue() const { return (thePhi_ * thePhiBinning); }

    /// Return the binned eta for this stub.
    unsigned etaPacked() const { return theEta_; }
    /// Return the binned phi for this stub.

    unsigned phiPacked() const { return thePhi_; }

    /// Get the digi this stub was made from.
    const CSCCorrelatedLCTDigi *getDigi() const { return dynamic_cast<const CSCCorrelatedLCTDigi *>(this); }
    DetId getDetId() const { return DetId(theDetId_); }

    /// Time / Space identifiers
    /// See CSCTransientDataType.h for more details.
    unsigned endcap() const;
    unsigned station() const;
    unsigned sector() const;
    unsigned subsector() const;
    unsigned cscid() const;
    unsigned cscidSeparateME1a() const;
    int BX() const { return getBX(); }

    /// Comparision Operators, used for MPC sorting
    bool operator>(const TrackStub &) const;
    bool operator<(const TrackStub &) const;
    bool operator>=(const TrackStub &rhs) const { return !(this->operator<(rhs)); }
    bool operator<=(const TrackStub &rhs) const { return !(this->operator>(rhs)); }
    bool operator==(const TrackStub &rhs) const {
      return ((theDetId_ == rhs.theDetId_) && (*(getDigi()) == *(rhs.getDigi())));
    }
    bool operator!=(const TrackStub &rhs) const { return !(this->operator==(rhs)); }

  private:
    uint32_t theDetId_;
    unsigned thePhi_, theEta_, link_;
    static const double theEtaBinning, thePhiBinning;
  };
}  // namespace csctf

#endif
