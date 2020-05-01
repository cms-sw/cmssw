#ifndef MagneticField_Engine_localMagneticField_h
#define MagneticField_Engine_localMagneticField_h

#include "MagneticField/Engine/interface/MagneticField.h"

namespace local {
  /**
   * A helper class for efficient thread-safe internal caching in
   * classes deriving from ::MagneticField (where such caching is
   * useful). This class is intended to provide the interface of
   * MagneticField while passing the cache internally. It is intended
   * as a straightforward replacement of ::MagneticField in the user
   * code.
   *
   * Note that one local::MagneticField object should not be shared
   * between threads. In case of doubt, copy the local::MagneticField
   * object (e.g. for tbb::parallel_for);
   */
  class MagneticField {
  public:
    MagneticField() : field_(nullptr) {}
    explicit MagneticField(::MagneticField const* field) : field_(field) {}

    /// Field value ad specified global point, in Tesla
    GlobalVector inTesla(const GlobalPoint& gp) { return field_->inTesla(gp, cache_); }

    /// Field value ad specified global point, in KGauss
    GlobalVector inKGauss(const GlobalPoint& gp) { return inTesla(gp) * 10.F; }

    /// Field value ad specified global point, in 1/Gev
    GlobalVector inInverseGeV(const GlobalPoint& gp) { return inTesla(gp) * 2.99792458e-3F; }

    /// True if the point is within the region where the concrete field
    // engine is defined.
    bool isDefined(const GlobalPoint& gp) const { return field_->isDefined(gp); }

    /// Optional implementation that derived classes can implement to provide faster query
    /// by skipping the check to isDefined.
    GlobalVector inTeslaUnchecked(const GlobalPoint& gp) { return field_->inTeslaUnchecked(gp, cache_); }

    /// The nominal field value for this map in kGauss
    int nominalValue() const { return field_->nominalValue(); }

    void reset(::MagneticField const* iField) {
      field_ = iField;
      cache_.reset();
    }

  private:
    ::MagneticField const* field_;
    MagneticFieldCache cache_;
  };
}  // namespace local

#endif
