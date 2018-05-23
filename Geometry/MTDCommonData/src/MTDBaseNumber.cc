#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"

MTDBaseNumber::MTDBaseNumber():_theLevels(0) { } 

MTDBaseNumber::MTDBaseNumber(const MTDBaseNumber & aBaseNumber):
  _sortedName(aBaseNumber._sortedName),
  _sortedCopyNumber(aBaseNumber._sortedCopyNumber),  
  _theLevels(aBaseNumber._theLevels) { }  

void MTDBaseNumber::setSize(const int & size) {
  if ( size < MAXLEVEL ) {
    _sortedName.resize(size);
    _sortedCopyNumber.resize(size);
  }
  else {
    _sortedName.resize(MAXLEVEL);
    _sortedCopyNumber.resize(MAXLEVEL);
    edm::LogWarning("MTDGeom") << "Required base number size exceeding maximum";
  }
}

void MTDBaseNumber::addLevel(const std::string& name, const int & copyNumber)
{
  if ( _theLevels == MAXLEVEL-1 ) {
    throw cms::Exception("WrongMTDGeom") << "MTDBaseNumber required to add more levels than maximum allowed";
  }
  _sortedName[_theLevels] = name;
  _sortedCopyNumber[_theLevels] = copyNumber;
  _theLevels++;
}

int MTDBaseNumber::getLevels() const
{
  return _theLevels;
}

int MTDBaseNumber::getCopyNumber(int level) const
{
  return _sortedCopyNumber[level];
}

int MTDBaseNumber::getCopyNumber(const std::string& levelName) const
{
  for ( int iLevel = 0; iLevel < _theLevels; iLevel++ ) {
    if ( _sortedName[iLevel] == levelName ) { return _sortedCopyNumber[iLevel]; }  
  }
  return 0;
}

std::string const & MTDBaseNumber::getLevelName(int level) const
{
  return _sortedName[level];
}

int MTDBaseNumber::getCapacity() 
{
  return  _sortedName.capacity();  
}

void MTDBaseNumber::reset()
{
  _theLevels = 0;
}

