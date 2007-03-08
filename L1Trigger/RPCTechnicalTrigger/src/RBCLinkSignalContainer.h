#ifndef RPCTechnicalTrigger_RBCLinkSignalContainer_h
#define RPCTechnicalTrigger_RBCLinkSignalContainer_h
#include <map>
class RBCLinkSignal;
class RBCId;
class RBCLinkSignalContainer{
 public:
  typedef std::map< std::pair<RBCId, int>, RBCLinlSignal>::iterator iterator;
  RBCLinkSignalContainer();
  void insert(RBCLinkSignal link);
  iterator begin();
  iterator end();

 private:
  std::map< std::pair<RBCId, int>, RBCLinkSignal> c;
  std::map< RBCid, std::set<RBCLinkSignal> > s;
#endif
