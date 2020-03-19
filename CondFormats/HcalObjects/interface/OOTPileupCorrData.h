#ifndef CondFormats_HcalObjects_OOTPileupCorrData_h_
#define CondFormats_HcalObjects_OOTPileupCorrData_h_

#include <cmath>
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/serialization/vector.hpp"
#include "boost/serialization/version.hpp"

#include "CondFormats/HcalObjects/interface/AbsOOTPileupCorrection.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrDataFcn.h"
#include <cstdint>

class OOTPileupCorrData : public AbsOOTPileupCorrection {
public:
  //
  // Constructor arguments are as follows:
  //
  // corrs          -- Correction function to apply in different ranges
  //                   of abs(iEta).
  //
  // iEtaLimits     -- Definition of ranges for abs(iEta). This is just
  //                   an increasing sequence of positive iEta values.
  //                   The size of this vector must be equal to the
  //                   size of "corrs" vector minus one. The correction
  //                   corrs[k] will normally be applied in case
  //                   iEtaLimits[k-1] <= abs(iEta) < iEtaLimits[k].
  //                   In case abs(iEta) < iEtaLimits[0], corrs[0] is
  //                   applied, and in case
  //                   iEtaLimits[iEtaLimits.size()-1] <= abs(iEta),
  //                   corrs[corrs.size()-1] is applied.
  //
  // chargeLimit    -- Minimum charge in the triggered time slice.
  //                   The observed charge must be larger than this minimum
  //                   in order for corrections to be applied.
  //
  // requireFirstTS -- The code will check that the corrections were
  //                   derived for the right triggered TS. Corrections
  //                   will be applied if either requireFirstTS < 0 (check
  //                   disabled) or if firstTimeSlice == requireFirstTS.
  //
  // requireNTS     -- The code will check that the corrections were derived
  //                   for the right number of time slices to integrate.
  //                   Corrections will be applied if either requireNTS < 0
  //                   (check disabled) or if requireNTS == nTimeSlices.
  //
  // readjustTiming -- Set "true" if one should use OOT pileup corrected
  //                   energies to derive hit time. To use the original,
  //                   uncorrected energies set this to "false".
  //
  OOTPileupCorrData(const std::vector<OOTPileupCorrDataFcn>& corrs,
                    const std::vector<uint32_t>& iEtaLimits,
                    double chargeLimit,
                    int requireFirstTS,
                    int requireNTS,
                    bool readjustTiming);

  inline ~OOTPileupCorrData() override {}

  // Main correction function
  void apply(const HcalDetId& id,
             const double* inputCharge,
             unsigned lenInputCharge,
             const BunchXParameter* bcParams,
             unsigned lenBcParams,
             unsigned firstTimeSlice,
             unsigned nTimeSlices,
             double* correctedCharge,
             unsigned lenCorrectedCharge,
             bool* pulseShapeCorrApplied,
             bool* leakCorrApplied,
             bool* readjustTiming) const override;

  // Are we using charge or energy?
  inline bool inputIsEnergy() const override { return false; }

  // Simplified correction function which does actual work
  inline void apply(const HcalDetId& id, double* ts, const int tsTrig) const {
    if (ts[tsTrig] > chargeLimit_) {
      const unsigned nLimits(iEtaLimits_.size());
      if (nLimits) {
        const uint32_t uEta = std::abs(id.ieta());
        const uint32_t* limits(&iEtaLimits_[0]);
        unsigned which(0U);
        for (; which < nLimits; ++which)
          if (uEta < limits[which])
            break;
        corrs_[which].pucorrection(ts, tsTrig);
      }
    }
  }

  inline const OOTPileupCorrDataFcn& getCorrectionByID(const HcalDetId& id) const {
    const unsigned nLimits = iEtaLimits_.size();
    unsigned which(0U);
    if (nLimits) {
      const uint32_t uEta = std::abs(id.ieta());
      const uint32_t* limits(&iEtaLimits_[0]);
      for (; which < nLimits; ++which)
        if (uEta < limits[which])
          break;
    }
    return corrs_.at(which);
  }

protected:
  // Comparison function must be implemented
  inline bool isEqual(const AbsOOTPileupCorrection& otherBase) const override {
    const OOTPileupCorrData& r = static_cast<const OOTPileupCorrData&>(otherBase);
    return corrs_ == r.corrs_ && iEtaLimits_ == r.iEtaLimits_ && chargeLimit_ == r.chargeLimit_ &&
           requireFirstTS_ == r.requireFirstTS_ && requireNTS_ == r.requireNTS_ && readjustTiming_ == r.readjustTiming_;
  }

public:
  // Default constructor needed for serialization.
  // Do not use in application code.
  inline OOTPileupCorrData() {}

private:
  bool validate() const;

  std::vector<OOTPileupCorrDataFcn> corrs_;
  std::vector<uint32_t> iEtaLimits_;
  double chargeLimit_;
  int32_t requireFirstTS_;
  int32_t requireNTS_;
  uint8_t readjustTiming_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void save(Archive& ar, const unsigned /* version */) const {
    if (!validate())
      throw cms::Exception("In OOTPileupCorrData::save: invalid data");
    boost::serialization::base_object<AbsOOTPileupCorrection>(*this);
    ar& corrs_& iEtaLimits_& chargeLimit_& requireFirstTS_& requireNTS_& readjustTiming_;
  }

  template <class Archive>
  inline void load(Archive& ar, const unsigned /* version */) {
    boost::serialization::base_object<AbsOOTPileupCorrection>(*this);
    ar& corrs_& iEtaLimits_& chargeLimit_& requireFirstTS_& requireNTS_& readjustTiming_;
    if (!validate())
      throw cms::Exception("In OOTPileupCorrData::load: invalid data");
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

BOOST_CLASS_VERSION(OOTPileupCorrData, 1)
BOOST_CLASS_EXPORT_KEY(OOTPileupCorrData)

#endif  // CondFormats_HcalObjects_OOTPileupCorrData_h_
