#ifndef RBCEmulator_RBCOutputSignalContainer_h
#define RBCEmulator_RBCOutputSignalContainer_h
#include <set>

class RBCOutputSignal;

class RBCOutputSignalContainer{
 public:
  typedef std::set<RBCOutputSignal>::iterator iterator;
  RBCOutputSignalContainer();
  virtual ~RBCOutputSignalContainer();
  void insert(RBCOutputSignal s);
  // iterators
  iterator begin();
  iterator end();

 private:
  std::set<RBCOutputSignal> signals;

};

#endif
