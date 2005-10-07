#ifndef CALOSAMPLES_H
#define CALOSAMPLES_H 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

/** \class CaloSamples
    
Class which represents the charge/voltage measurements of an event/channel
with the ADC decoding performed.

$Date: 2005/07/27 19:44:02 $
$Revision: 1.1 $
*/
class CaloSamples {
public:
  CaloSamples();
  explicit CaloSamples(const DetId& id, int size);
  
  /// get the (generic) id
  DetId id() const { return id_; }

  /// get the size
  int size() const { return size_; }
  /// mutable operator to access samples
  double& operator[](int i) { return data_[i]; }
  /// const operator to access samples
  double operator[](int i) const { return data_[i]; }

  /// access presample information
  int presamples() const { return presamples_; }
  /// set presample information
  void setPresamples(int pre);

  static const int MAXSAMPLES=10;
private:
  DetId id_;
  double data_[MAXSAMPLES]; // 
  int size_, presamples_;
};

std::ostream& operator<<(std::ostream& s, const CaloSamples& samps);

#endif
