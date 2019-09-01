#ifndef CALOSAMPLES_H
#define CALOSAMPLES_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include <ostream>
#include <vector>

/** \class CaloSamples

Class which represents the charge/voltage measurements of an event/channel
with the ADC decoding performed.

*/
class CaloSamples {
public:
  CaloSamples();
  explicit CaloSamples(const DetId &id, int size);
  explicit CaloSamples(const DetId &id, int size, int preciseSize);

  /// get the (generic) id
  DetId id() const { return id_; }

  /// get the size
  int size() const { return size_; }
  /// mutable operator to access samples
  double &operator[](int i) { return data_[i]; }
  /// const operator to access samples
  double operator[](int i) const { return data_[i]; }

  /// mutable function to access precise samples
  float &preciseAtMod(int i) { return preciseData_[i]; }
  /// const function to access precise samples
  float preciseAt(int i) const { return preciseData_[i]; }

  /// access presample information
  int presamples() const { return presamples_; }
  /// set presample information
  void setPresamples(int pre);

  /// multiply each item by this value
  CaloSamples &scale(double value);
  /// scale all samples
  CaloSamples &operator*=(double value) { return scale(value); }

  /// add a value to all samples
  CaloSamples &operator+=(double value);
  CaloSamples &operator+=(const CaloSamples &other);

  /// shift all the samples by a time, in ns, interpolating
  // between values
  CaloSamples &offsetTime(double offset);

  void setDetId(DetId detId) { id_ = detId; }

  void setSize(unsigned int size) {
    size_ = size;
    data_.resize(size, 0.);
  }

  void setPreciseSize(unsigned int size) {
    preciseSize_ = size;
    preciseData_.resize(preciseSize_, 0.);
  }

  bool isBlank() const;  // are the samples blank (zero?)

  void setBlank();  // keep id, presamples, size but zero out data

  /// get the size
  int preciseSize() const {
    if (preciseData_.empty())
      return 0;
    return preciseSize_;
  }
  int precisePresamples() const { return precisePresamples_; }
  float preciseDeltaT() const { return deltaTprecise_; }

  void setPrecise(int precisePresamples, float deltaT) {
    precisePresamples_ = precisePresamples;
    deltaTprecise_ = deltaT;
  }

  void resetPrecise();

  // preserved for use in other packages - no longer used in this class
  static const int MAXSAMPLES = 10;

private:
  DetId id_;
  int size_, presamples_;
  std::vector<double> data_;  //
  float deltaTprecise_;
  std::vector<float> preciseData_;
  int preciseSize_, precisePresamples_;
};

std::ostream &operator<<(std::ostream &s, const CaloSamples &samps);

typedef std::vector<CaloSamples> CaloSamplesCollection;

#endif
