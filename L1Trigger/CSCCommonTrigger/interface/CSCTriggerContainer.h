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
  std::vector<T> get(const unsigned& endcap, const unsigned& station, const unsigned& tsector, 
		     const unsigned& tsubsector, const unsigned& cscid, const int& BX = 0) const; /// For a specific chamber 
                                                                                                  /// in a station.
  std::vector<T> get(const unsigned& endcap, const unsigned& station, const unsigned& tsector,    /// For a specific station in a sector.
		     const unsigned& tsubsector, const int& BX = 0) const;
  std::vector<T> get(const unsigned& endcap, const unsigned& sector, const int& BX = 0) const;    /// For objects which span multiple stations.

  void push_back(const T data) { _objs.push_back(data); }
  void clear() { _objs.clear(); } 

 private:
  
  std::vector<T> _objs;
};

#endif
