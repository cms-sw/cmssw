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
		     const unsigned& tsubsector, const unsigned& cscid, const int& BX) const; /// For a specific chamber 
                                                                                                  /// in a station.
  std::vector<T> get(const unsigned& endcap, const unsigned& station, const unsigned& tsector,    /// For a specific station in a sector.
		     const unsigned& tsubsector, const int& BX) const;
  std::vector<T> get(const unsigned& endcap, const unsigned& sector, const int& BX) const; /// For objects which span multiple stations.
  std::vector<T> get(const unsigned& endcap, const unsigned& sector) const;
  std::vector<T> get(const int& BX) const;

  void push_back(const T data) { _objs.push_back(data); }
  void push_many(const std::vector<T>& data) { _objs.insert(_objs.end(), data.begin(), data.end()); }
  void push_many(const CSCTriggerContainer<T> data) { std::vector<T> vec = data.get(); _objs.insert(_objs.end(), vec.begin(), vec.end()); }
  void clear() { _objs.clear(); } 

 private:
  
  std::vector<T> _objs;
};

template<class T>
CSCTriggerContainer<T>::CSCTriggerContainer(const std::vector<T>& parent)
{
  _objs = parent;
}

template<class T>
CSCTriggerContainer<T>& CSCTriggerContainer<T>::operator=(const CSCTriggerContainer& rhs)
{
  if(this != &rhs)
    {
      _objs = rhs._objs;
    }
  return *this;
}

template<class T>
CSCTriggerContainer<T>& CSCTriggerContainer<T>::operator=(const std::vector<T>& rhs)
{
  _objs = rhs;
  return *this;
}

template<class T>
std::vector<T> CSCTriggerContainer<T>::get() const
{
  return _objs;
}

template<class T>
std::vector<T> CSCTriggerContainer<T>::get(const unsigned& endcap, const unsigned& station, 
					   const unsigned& tsector,const unsigned& tsubsector, 
					   const unsigned& cscid, const int& BX) const
{
  std::vector<T> result;  

  for(unsigned i = 0; i < _objs.size(); i++)
    if(_objs[i].endcap() == endcap && _objs[i].station() == station && 
       _objs[i].sector() == tsector && (station != 1 || _objs[i].subsector() == tsubsector) && 
       _objs[i].cscid() == cscid && _objs[i].BX() == BX)
      result.push_back(_objs[i]);
  
  return result;
}

template<class T>
std::vector<T> CSCTriggerContainer<T>::get(const unsigned& endcap, const unsigned& station, 
					   const unsigned& tsector,const unsigned& tsubsector, 
					   const int& BX) const
{
  std::vector<T> result;  

  for(unsigned i = 0; i < _objs.size(); ++i)
    if(_objs[i].endcap() == endcap && _objs[i].station() == station && 
       _objs[i].sector() == tsector && (station != 1 || _objs[i].subsector() == tsubsector) 
       && _objs[i].BX() == BX)
      result.push_back(_objs[i]);
  
  return result;
}

template<class T>
std::vector<T> CSCTriggerContainer<T>::get(const unsigned& endcap, const unsigned& sector, 
					   const int& BX) const
{
  std::vector<T> result;

  for(unsigned i = 0; i < _objs.size(); ++i)
    if(_objs[i].endcap() == endcap && _objs[i].sector() == sector && _objs[i].BX() == BX)
      result.push_back(_objs[i]);

  return result;
}

template<class T>
std::vector<T> CSCTriggerContainer<T>::get(const unsigned& endcap, const unsigned& sector) const
{
  std::vector<T> result;

  for(unsigned i = 0; i < _objs.size(); ++i)
    if(_objs[i].endcap() == endcap && _objs[i].sector() == sector)
      result.push_back(_objs[i]);

  return result;
}

template<class T>
std::vector<T> CSCTriggerContainer<T>::get(const int& BX) const
{
  std::vector<T> result;

  for(unsigned i = 0; i < _objs.size(); ++i)
    if(_objs[i].BX() == BX) result.push_back(_objs[i]);

  return result;
}

#endif
