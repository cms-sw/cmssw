#ifndef CSCCommonTrigger_CSCTriggerContainer_h
#define CSCCommonTrigger_CSCTriggerContainer_h

/**
 * \class CSCTriggerContainer
 * \author L. Gray
 * 
 * A container class to make things more manageable for a Trigger Processor.
 * DigiCollections make per-BX processing complicated, this class makes it easier.
 * 
 * Any class T must have the following functions: // inherit from base class!
 * T(const T&)
 * operator=
 * station()
 * sector()
 * subsector()
 * BX()
 */

#include <vector>

template<class T>
class CSCTriggerContainer
{
 public:

  CSCTriggerContainer() {}
  CSCTriggerContainer(const CSCTriggerContainer& cpy) { _objs = cpy._objs; }
  CSCTriggerContainer(const std::vector<T>&);

  CSCTriggerContainer& operator=(const CSCTriggerContainer&);
  CSCTriggerContainer& operator=(const std::vector<T>&);

  std::vector<T> get() const;
  std::vector<T> get(int station, int tsector, int tsubsector, int BX = 0) const; /// for a specific station in a sector
  std::vector<T> get(int sector, int BX = 0) const; /// for objects which span multiple stations

  void push_back(const T&);

 private:

  std::vector<T> _objs;
};

#endif
