#ifndef ECALPSEUDOSTRIPINPUTDIGI_H
#define ECALPSEUDOSTRIPINPUTDIGI_H

#include <ostream>
#include <vector>
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "DataFormats/EcalDigi/interface/EcalPseudoStripInputSample.h"

/** \class EcalPseudoStripInputDigi
      
*/

class EcalPseudoStripInputDigi {
public:
  typedef EcalTriggerElectronicsId key_type;  ///< For the sorted collection

  EcalPseudoStripInputDigi();  // for persistence
  explicit EcalPseudoStripInputDigi(const EcalTriggerElectronicsId& id);

  const EcalTriggerElectronicsId& id() const { return id_; }
  int size() const { return size_; }

  const EcalPseudoStripInputSample& operator[](int i) const { return data_[i]; }
  const EcalPseudoStripInputSample& sample(int i) const { return data_[i]; }

  void setSize(int size);
  void setSample(int i, const EcalPseudoStripInputSample& sam) { data_[i] = sam; }
  void setSampleValue(int i, uint16_t value) { data_[i].setValue(value); }

  static const int MAXSAMPLES = 20;

  /// get the encoded/compressed Et of interesting sample
  int pseudoStripInput() const;

  /// get the fine-grain bit of interesting sample
  bool fineGrain() const;

  /// True if debug mode (# of samples > 1)
  bool isDebug() const;

  /// Gets the interesting sample
  int sampleOfInterest() const;

private:
  EcalTriggerElectronicsId id_;
  int size_;
  std::vector<EcalPseudoStripInputSample> data_;
};

std::ostream& operator<<(std::ostream& s, const EcalPseudoStripInputDigi& digi);

#endif
