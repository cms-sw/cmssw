#ifndef DIGIECAL_ESDATAFRAME_H
#define DIGIECAL_ESDATAFRAME_H

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "DataFormats/Common/interface/DataFrame.h"
#include <vector>
#include <ostream>

class ESDataFrame {
public:
  typedef ESDetId key_type;  ///< For the sorted collection

  ESDataFrame();
  explicit ESDataFrame(const ESDetId& id);

  ESDataFrame(const edm::DataFrame& df);

  const ESDetId& id() const { return id_; }

  int size() const { return size_; }

  const ESSample& operator[](int i) const { return data_[i]; }
  const ESSample& sample(int i) const { return data_[i]; }

  void setSize(int size);

  void setSample(int i, const ESSample& sam) { data_[i] = sam; }

  static const int MAXSAMPLES = 3;

private:
  ESDetId id_;
  int size_;

  ESSample data_[MAXSAMPLES];
};

std::ostream& operator<<(std::ostream&, const ESDataFrame&);

#endif
