#ifndef FWCore_PythonParameterSet_PythonParameterSet_h
#define FWCore_PythonParameterSet_PythonParameterSet_h

#include "FWCore/PythonParameterSet/interface/BoostPython.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/src/PythonWrapper.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"

#include <string>
#include <vector>

class PythonParameterSet
{
public:
  PythonParameterSet();

  PythonParameterSet(edm::ParameterSet const& p)
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
  getParameters(bool tracked, std::string const& name) const
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


  /// these custom classes do seem to be a hassle
  /// to wrap, compared to, say, InputTag
  /// maybe we will need to template these someday
  void addPSet(bool tracked, std::string const& name,
               PythonParameterSet const& ppset)
  {
    addParameter(tracked, name, ppset.theParameterSet);
  }


  PythonParameterSet getPSet(bool tracked, std::string const& name) const
  {
    return PythonParameterSet(getParameter<edm::ParameterSet>(tracked, name));
  }


  void addVPSet(bool tracked, std::string const& name,
                boost::python::list  value);

  boost::python::list getVPSet(bool tracked, std::string const& name);

  // no way to interface straight into the other python InputTag
  edm::InputTag newInputTag(std::string const& label,
                            std::string const& instance,
                            std::string const& process)
  {
    return edm::InputTag(label, instance, process);
  }

   edm::ESInputTag newESInputTag(std::string const& module,
                             std::string const& data)
   {
      return edm::ESInputTag(module, data);
   }
   
   edm::EventID newEventID(unsigned int run, unsigned int event)
  {
    return edm::EventID(run, event);
  }

  edm::LuminosityBlockID newLuminosityBlockID(unsigned int run, unsigned int lumi)
  {
    return edm::LuminosityBlockID(run, lumi);
  }

  edm::LuminosityBlockRange newLuminosityBlockRange(unsigned int start, unsigned int startSub,
                                                    unsigned int end,   unsigned int endSub)
  {
    return edm::LuminosityBlockRange(start, startSub, end, endSub);
  }

  edm::EventRange newEventRange(unsigned int start, unsigned int startSub,
                                unsigned int end,   unsigned int endSub)
  {
    return edm::EventRange(start, startSub, end, endSub);
  }

  void addNewFileInPath(bool tracked, std::string const & name, std::string const & value);

  PythonParameterSet newPSet() const {return PythonParameterSet();}

  edm::ParameterSet& pset() {return theParameterSet;}

  edm::ParameterSet const& pset() const {return theParameterSet;}

  std::string dump() const {return theParameterSet.dump();}

private:
  edm::ParameterSet theParameterSet;
};

#endif
