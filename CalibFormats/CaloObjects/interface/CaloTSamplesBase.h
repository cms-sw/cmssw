#ifndef CALOTSAMPLESBASE_H
#define CALOTSAMPLESBASE_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include <cassert>
#include <ostream>

template <class Ttype>
class CaloTSamplesBase {
public:
  CaloTSamplesBase(Ttype *mydata, uint32_t size);

  CaloTSamplesBase(const CaloTSamplesBase<Ttype> &cs);

  CaloTSamplesBase(Ttype *mydata, uint32_t length, const DetId &id, uint32_t size, uint32_t pre);

  virtual ~CaloTSamplesBase();

  void setZero();

  DetId id() const;
  uint32_t size() const;
  uint32_t pre() const;
  bool zero() const;

  Ttype &operator[](uint32_t i);

  const Ttype &operator[](uint32_t i) const;

  CaloTSamplesBase<Ttype> &operator=(const CaloTSamplesBase<Ttype> &cs);

  CaloTSamplesBase<Ttype> &operator*=(Ttype value);

  CaloTSamplesBase<Ttype> &operator+=(Ttype value);

  CaloTSamplesBase<Ttype> &operator+=(const CaloTSamplesBase<Ttype> &cs);

  virtual uint32_t capacity() const = 0;

private:
  virtual Ttype *data(uint32_t i) = 0;
  virtual const Ttype *cdata(uint32_t i) const = 0;

  DetId m_id;
  uint32_t m_size;
  uint32_t m_pre;
};

template <class Ttype>
std::ostream &operator<<(std::ostream &s, const CaloTSamplesBase<Ttype> &sam);

#endif
