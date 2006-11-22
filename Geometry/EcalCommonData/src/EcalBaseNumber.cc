#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"

void EcalBaseNumber::addLevel(const std::string& name,int copyNumber)
{
  _sortedBaseNumber.push_back(std::pair<std::string,int>(name,copyNumber));
}

int EcalBaseNumber::getLevels() const
{
  return _sortedBaseNumber.size();
}

int EcalBaseNumber::getCopyNumber(int level) const
{
  return _sortedBaseNumber[level].second;
}

int EcalBaseNumber::getCopyNumber(const std::string& levelName) const
{
  basenumber_type::const_iterator cur=_sortedBaseNumber.begin();
  basenumber_type::const_iterator end=_sortedBaseNumber.end();
  while (cur!=end) {
    if ((*cur).first==levelName) {
      return (*cur).second;
    }
    cur++;
  }
  return 0;
}

std::string EcalBaseNumber::getLevelName(int level) const
{
  return _sortedBaseNumber[level].first;
}
