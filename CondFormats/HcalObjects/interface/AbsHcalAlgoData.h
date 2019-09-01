#ifndef CondFormats_HcalObjects_AbsHcalAlgoData_h_
#define CondFormats_HcalObjects_AbsHcalAlgoData_h_

#include <typeinfo>

//
// Classes inheriting from this one are supposed to configure
// various Hcal reco algorithms
//
class AbsHcalAlgoData {
public:
  inline virtual ~AbsHcalAlgoData() {}

  // Comparison operators. Note that they are not virtual and should
  // not be overriden by derived classes. These operators are very
  // useful for I/O testing.
  inline bool operator==(const AbsHcalAlgoData& r) const { return (typeid(*this) == typeid(r)) && this->isEqual(r); }
  inline bool operator!=(const AbsHcalAlgoData& r) const { return !(*this == r); }

protected:
  // Method needed to compare objects for equality.
  // Must be implemented by derived classes.
  virtual bool isEqual(const AbsHcalAlgoData&) const = 0;
};

#endif  // CondFormats_HcalObjects_AbsHcalAlgoData_h_
