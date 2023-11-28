#ifndef CALOTSAMPLES_H
#define CALOTSAMPLES_H 1

#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"

/** \class CaloTSamples

Class which represents the charge/voltage measurements of an event/channel
with the ADC decoding performed.

*/

template <class Ttype, uint32_t Tsize>
class CaloTSamples : public CaloTSamplesBase<Ttype> {
public:
  enum { kCapacity = Tsize };

  CaloTSamples();
  CaloTSamples(const CaloTSamples<Ttype, Tsize> &cs);
  CaloTSamples(const DetId &id, uint32_t size = 0, uint32_t pre = 0);
  ~CaloTSamples() override;

  CaloTSamples<Ttype, Tsize> &operator=(const CaloTSamples<Ttype, Tsize> &cs);

  uint32_t capacity() const override;

private:
  Ttype *data(uint32_t i) override;
  const Ttype *cdata(uint32_t i) const override;

  Ttype m_data[Tsize];
};

#endif
