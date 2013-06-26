#ifndef CondFormats_ESObjects_ESWeight_H
#define CondFormats_ESObjects_ESWeight_H

/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * This is workaround in order to be able to use vector<double>
 * for ECAL weights. because of a conflict I need to define this trivial class
 * so that I can use POOL to store vector<ESWeight>
 *
 **/

#include <iostream>

class ESWeight {
 public:
  ESWeight();
  ESWeight(const double& awgt);
  ESWeight(const ESWeight& awgt);
  ESWeight& operator=(const ESWeight&rhs);

  double value() const { return wgt_; }
  double operator()() const { return wgt_; }

  void setValue(const double& awgt) { wgt_ = awgt; }
  bool operator ==(const ESWeight&rhs) const { return (wgt_ == rhs.wgt_); }
  bool operator !=(const ESWeight&rhs) const { return (wgt_ != rhs.wgt_); }
  bool operator <(const ESWeight&rhs) const { return (wgt_ < rhs.wgt_); }
  bool operator >(const ESWeight&rhs) const { return (wgt_ > rhs.wgt_); }
  bool operator <=(const ESWeight&rhs) const { return (wgt_ <= rhs.wgt_); }
  bool operator >=(const ESWeight&rhs) const { return (wgt_ >= rhs.wgt_); }

 private:
  double wgt_;
};

//std::ostream& operator<<(std::ostream& os, const ESWeight& wg) {
//   return os << wg.value();
//}

#endif
