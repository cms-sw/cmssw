#include "FWCore/PythonParameterSet/interface/Python11ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

Python11ParameterSet::Python11ParameterSet()
:  theParameterSet()
{
}


void Python11ParameterSet::addVPSet(bool tracked, std::string const& name,
              pybind11::list  value)
{
  std::vector<Python11ParameterSet> v
    = edm::toVector<Python11ParameterSet>(value);
  std::vector<edm::ParameterSet> v2;
  v2.reserve(v.size());
  for(std::vector<Python11ParameterSet>::iterator ppsetItr = v.begin(), ppsetItrEnd = v.end();
      ppsetItr != ppsetItrEnd; ++ppsetItr)
  {
    v2.push_back(ppsetItr->theParameterSet);
  }
  addParameter(tracked, name, v2);
}


pybind11::list Python11ParameterSet::getVPSet(bool tracked, std::string const& name)
{
  std::vector<edm::ParameterSet> const& v =
    (tracked ? theParameterSet.getParameterSetVector(name) : theParameterSet.getUntrackedParameterSetVector(name));

  // convert to Python11ParameterSets
  pybind11::list l;
  for(std::vector<edm::ParameterSet>::const_iterator psetItr = v.begin(), psetItrEnd = v.end();
      psetItr != psetItrEnd; ++psetItr)
  {
    l.append(Python11ParameterSet(*psetItr));
  }

  return l;
}


void  Python11ParameterSet::addNewFileInPath(bool tracked, std::string const & name, std::string const & value)
{
  addParameter(tracked, name, edm::FileInPath(value));
}
