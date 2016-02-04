#include "FWCore/PythonParameterSet/interface/PythonParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

PythonParameterSet::PythonParameterSet()
:  theParameterSet()
{
}


void PythonParameterSet::addVPSet(bool tracked, std::string const& name,
              boost::python::list  value)
{
  std::vector<PythonParameterSet> v
    = edm::toVector<PythonParameterSet>(value);
  std::vector<edm::ParameterSet> v2;
  v2.reserve(v.size());
  for(std::vector<PythonParameterSet>::iterator ppsetItr = v.begin(), ppsetItrEnd = v.end();
      ppsetItr != ppsetItrEnd; ++ppsetItr)
  {
    v2.push_back(ppsetItr->theParameterSet);
  }
  addParameter(tracked, name, v2);
}


boost::python::list PythonParameterSet::getVPSet(bool tracked, std::string const& name)
{
  std::vector<edm::ParameterSet> const& v =
    (tracked ? theParameterSet.getParameterSetVector(name) : theParameterSet.getUntrackedParameterSetVector(name));

  // convert to PythonParameterSets
  boost::python::list l;
  for(std::vector<edm::ParameterSet>::const_iterator psetItr = v.begin(), psetItrEnd = v.end();
      psetItr != psetItrEnd; ++psetItr)
  {
    l.append(PythonParameterSet(*psetItr));
  }

  return l;
}


void  PythonParameterSet::addNewFileInPath(bool tracked, std::string const & name, std::string const & value)
{
  addParameter(tracked, name, edm::FileInPath(value));
}
