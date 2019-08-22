#ifndef DataFormats_CaloRecHit_CaloID_h
#define DataFormats_CaloRecHit_CaloID_h

/** \class reco::CaloID 
 *  
 * ID information for all calorimeters. 
 *
 * \author Colin Bernet, LLR
 *
 *
 */

#include <iosfwd>

namespace reco {

  class CaloID {
  public:
    enum Detectors {
      DET_ECAL_BARREL = 0,
      DET_ECAL_ENDCAP,
      DET_PS1,
      DET_PS2,
      DET_HCAL_BARREL,
      DET_HCAL_ENDCAP,
      DET_HF,
      DET_HF_EM,
      DET_HF_HAD,
      DET_HO,
      DET_HGCAL_ENDCAP,
      DET_NONE
    };

    /// default constructor. Sets energy and position to zero
    CaloID() : detectors_(0) {}

    CaloID(Detectors det) : detectors_(0) { setDetector(det, true); }

    /// abstract class
    virtual ~CaloID() {}

    /// tells the CaloID that it describes a given detector
    void setDetector(CaloID::Detectors theDetector, bool value);

    /// \return packed detector information
    unsigned detectors() const { return detectors_; }

    /// \return true if this CaloID is in a given detector
    bool detector(CaloID::Detectors theDetector) const;

    /// \return true if this CaloID describes a single detector
    bool isSingleDetector() const {
      // check that detectors_ is a power of 2
      return static_cast<bool>(detectors_ && !((detectors_ - 1) & detectors_));
    }

    /// \return the described detector if isSingleDetector(),
    /// and DET_NONE otherwise.
    Detectors detector() const;

    CaloID& operator=(const CaloID& rhs) {
      detectors_ = rhs.detectors_;
      return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const CaloID& id);

  private:
    /// \return lsb position in an integer
    int leastSignificantBitPosition(unsigned n) const;

    /// packs the detector information into a bitmask.
    /// a CaloID can describe several detectors (bit or)
    unsigned detectors_;
  };

  std::ostream& operator<<(std::ostream& out, const CaloID& id);

}  // namespace reco

#endif
