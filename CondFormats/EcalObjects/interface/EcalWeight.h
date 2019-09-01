#ifndef CondFormats_EcalObjects_EcalWeight_H
#define CondFormats_EcalObjects_EcalWeight_H

/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * This is workaround in order to be able to use vector<double>
 * for ECAL weights. because of a conflict I need to define this trivial class
 * so that I can use POOL to store vector<EcalWeight>
 *
 **/

#include <iostream>

class EcalWeight {
public:
  EcalWeight();
  EcalWeight(const double& awgt);
  EcalWeight(const EcalWeight& awgt);
  EcalWeight& operator=(const EcalWeight& rhs);

  double value() const { return wgt_; }
  double operator()() const { return wgt_; }

  void setValue(const double& awgt) { wgt_ = awgt; }
  bool operator==(const EcalWeight& rhs) const { return (wgt_ == rhs.wgt_); }
  bool operator!=(const EcalWeight& rhs) const { return (wgt_ != rhs.wgt_); }
  bool operator<(const EcalWeight& rhs) const { return (wgt_ < rhs.wgt_); }
  bool operator>(const EcalWeight& rhs) const { return (wgt_ > rhs.wgt_); }
  bool operator<=(const EcalWeight& rhs) const { return (wgt_ <= rhs.wgt_); }
  bool operator>=(const EcalWeight& rhs) const { return (wgt_ >= rhs.wgt_); }

private:
  double wgt_;
};

//std::ostream& operator<<(std::ostream& os, const EcalWeight& wg) {
//   return os << wg.value();
//}

#endif
