#ifndef INTEGERCALOSAMPLES_H
#define INTEGERCALOSAMPLES_H 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"

/** \class IntegerCaloSamples
    
Class which represents the linear charge/voltage measurements of an
event/channel, but with defined resolution.  

This class uses 32-bit bins, so users should be careful if their
calculation implies fewer bins.

$Date: 2006/03/23 18:24:51 $
$Revision: 1.1 $
*/
class IntegerCaloSamples {
public:
  IntegerCaloSamples();
  explicit IntegerCaloSamples(const DetId& id, int size);
  
  /// get the (generic) id
  DetId id() const { return id_; }

  /// get the size
  int size() const { return size_; }
  /// mutable operator to access samples
  uint32_t& operator[](int i) { return data_[i]; }
  /// const operator to access samples
  uint32_t operator[](int i) const { return data_[i]; }

  /// access presample information
  int presamples() const { return presamples_; }
  /// set presample information
  void setPresamples(int pre);

  static const int MAXSAMPLES=10;
private:
  DetId id_;
  uint32_t data_[MAXSAMPLES]; // 
  int size_, presamples_;
};

std::ostream& operator<<(std::ostream& s, const IntegerCaloSamples& samps);

#endif
