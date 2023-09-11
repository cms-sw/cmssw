#ifndef ECALEBPHASE2TRIGGERPRIMITIVEDIGI_H
#define ECALEBPHASE2TRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalEBPhase2TriggerPrimitiveSample.h"

/** \class EcalEBPhase2TriggerPrimitiveDigi
\author N. Marinelli - Univ. of Notre Dame


*/

class EcalEBPhase2TriggerPrimitiveDigi {
public:
  typedef EBDetId key_type;  ///< For the sorted collection

  EcalEBPhase2TriggerPrimitiveDigi();  // for persistence
  EcalEBPhase2TriggerPrimitiveDigi(const EBDetId& id);

  void swap(EcalEBPhase2TriggerPrimitiveDigi& rh) {
    std::swap(id_, rh.id_);
    std::swap(size_, rh.size_);
    std::swap(data_, rh.data_);
  }

  const EBDetId& id() const { return id_; }
  int size() const { return size_; }

  const EcalEBPhase2TriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
  const EcalEBPhase2TriggerPrimitiveSample& sample(int i) const { return data_[i]; }

  void setSize(int size);
  void setSample(int i, const EcalEBPhase2TriggerPrimitiveSample& sam);
  void setSampleValue(int i, uint16_t value) { data_[i].setValue(value); }

  static const int MAXSAMPLES = 20;

  /// get the 10 bits Et of interesting sample
  int encodedEt() const;

  /// Spike flag
  bool l1aSpike() const;

  /// Time info
  int time() const;

  /// True if debug mode (# of samples > 1)
  bool isDebug() const;

  /// Gets the interesting sample
  int sampleOfInterest() const;

private:
  EBDetId id_;
  int size_;
  std::vector<EcalEBPhase2TriggerPrimitiveSample> data_;
};

inline void swap(EcalEBPhase2TriggerPrimitiveDigi& lh, EcalEBPhase2TriggerPrimitiveDigi& rh) { lh.swap(rh); }

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveDigi& digi);

#endif
