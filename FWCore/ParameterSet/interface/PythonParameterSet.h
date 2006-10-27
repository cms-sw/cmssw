#ifndef PythonParameterSet_h
#define PythonParameterSet_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/python.hpp>
#include "FWCore/ParameterSet/src/PythonWrapper.h"

class PythonParameterSet
{
public:
  PythonParameterSet();

  PythonParameterSet(const edm::ParameterSet & p)
  : theParameterSet(p) {}

  template <class T>
  T
  getParameter(bool tracked, std::string const& name) const 
  {
    T result;
    if(tracked)
    {
      result = theParameterSet.template getParameter<T>(name);
    }
    else 
    {
      result = theParameterSet.template getUntrackedParameter<T>(name, result);
    }
    return result;
  }


  template <class T>
  void
  addParameter(bool tracked, std::string const& name, T value)
  {
   if(tracked)
   {
     theParameterSet.template addParameter<T>(name, value);
   }
   else 
   {
     theParameterSet.template addUntrackedParameter<T>(name, value);
   }
  }


  /// templated on the type of the contained object
  template <class T>
  boost::python::list
  getParameters(bool tracked, const std::string & name) const
  {
    std::vector<T> v = getParameter<std::vector<T> >(tracked, name);
    return edm::toPythonList(v);
  }

  /// unfortunate side effect: destroys the original list!
  template <class T>
  void
  addParameters(bool tracked, std::string const& name, 
                boost::python::list  value)
  {
    std::vector<T> v = edm::toVector<T>(value);
    addParameter(tracked, name, v);
  }


  void addPSet(bool tracked, std::string const& name,
               const PythonParameterSet & ppset)
  {
    addParameter(tracked, name, ppset.theParameterSet);
  }


  PythonParameterSet getPSet(bool tracked, std::string const& name) const
  {
    return PythonParameterSet(getParameter<edm::ParameterSet>(tracked, name));
  }


  void addVPSet(bool tracked, std::string const& name,
                boost::python::list  value)
  {
    std::vector<PythonParameterSet> v 
      = edm::toVector<PythonParameterSet>(value);
    std::vector<edm::ParameterSet> v2;
    v2.reserve(v.size());
    for(std::vector<PythonParameterSet>::iterator ppsetItr = v.begin();
        ppsetItr != v.end(); ++ppsetItr)
    {
      v2.push_back(ppsetItr->theParameterSet);
    }
    addParameter(tracked, name, v2);
  }

private:
  edm::ParameterSet theParameterSet;
};

#endif

