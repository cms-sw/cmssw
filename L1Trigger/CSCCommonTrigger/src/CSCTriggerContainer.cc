#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerContainer.h>

template<class T>
CSCTriggerContainer<T>::CSCTriggerContainer(const std::vector<T>& parent)
{
  _objs.copy(parent.begin(), parent.end());
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
std::vector<T> CSCTriggerContainer<T>::get(int station, int tsector, int tsubsector, int BX) const
{
  std::vector<T> result;  

  for(unsigned i = 0; i < _objs.size(); i++)
    if(_objs[i].station() == station && _objs[i].sector() == tsector &&
       (station != 1 || _objs[i].subsector() == tsubsector) && _objs[i].BX() == BX)
      result.push_back(_objs[i]);
  
  return result;
}

template<class T>
std::vector<T> CSCTriggerContainer<T>::get(int sector, int BX) const
{
  std::vector<T> result;

  for(unsigned i = 0; curr < _objs.size(); i++)
    if(_objs[i].sector() == sector && _objs[i].BX() == BX)
      result.push_back(_objs[i]);

  return result;
}
