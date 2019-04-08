#ifndef FWCore_PyBind11ParameterSet_Python11ParameterSet_h
#define FWCore_PyBind11ParameterSet_Python11ParameterSet_h
#include <pybind11/pybind11.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/PyBind11Wrapper.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

#include <string>
#include <vector>

class Python11ParameterSet {
public:
  Python11ParameterSet();

  Python11ParameterSet(edm::ParameterSet const& p)
  : theParameterSet(p) {}

  template<typename T>
  T
  getParameter(bool tracked, std::string const& name) const {
    T result;
    if(tracked) {
      result = theParameterSet.template getParameter<T>(name);
    } else {
      result = theParameterSet.template getUntrackedParameter<T>(name);
    }
    return result;
  }


  template<typename T>
  void
  addParameter(bool tracked, std::string const& name, T const& value) {
   if(tracked) {
     theParameterSet.template addParameter<T>(name, value);
   } else {
     theParameterSet.template addUntrackedParameter<T>(name, value);
   }
  }


  /// templated on the type of the contained object
  template<typename T>
  pybind11::list
  getParameters(bool tracked, std::string const& name) const {
    std::vector<T> v = getParameter<std::vector<T> >(tracked, name);
    return edm::toPython11List(v);
  }

  /// unfortunate side effect: destroys the original list!
  template<typename T>
  void
  addParameters(bool tracked, std::string const& name,
                pybind11::list  value) {
    std::vector<T> v = edm::toVector<T>(value);
    addParameter(tracked, name, v);
  }


  /// these custom classes do seem to be a hassle
  /// to wrap, compared to, say, InputTag
  /// maybe we will need to template these someday
  void addPSet(bool tracked, std::string const& name,
               Python11ParameterSet const& ppset) {
    addParameter(tracked, name, ppset.theParameterSet);
  }


  Python11ParameterSet getPSet(bool tracked, std::string const& name) const {
    return Python11ParameterSet(getParameter<edm::ParameterSet>(tracked, name));
  }


  void addVPSet(bool tracked, std::string const& name,
                pybind11::list  value);

  pybind11::list getVPSet(bool tracked, std::string const& name);

  // no way to interface straight into the other python InputTag
  edm::InputTag newInputTag(std::string const& label,
                            std::string const& instance,
                            std::string const& process) {
    return edm::InputTag(label, instance, process);
  }

   edm::ESInputTag newESInputTag(std::string const& module,
                             std::string const& data) {
      return edm::ESInputTag(module, data);
   }

   edm::EventID newEventID(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event) {
    return edm::EventID(run, lumi, event);
  }

  edm::LuminosityBlockID newLuminosityBlockID(unsigned int run, unsigned int lumi) {
    return edm::LuminosityBlockID(run, lumi);
  }

  edm::LuminosityBlockRange newLuminosityBlockRange(unsigned int start, unsigned int startSub,
                                                    unsigned int end,   unsigned int endSub) {
    return edm::LuminosityBlockRange(start, startSub, end, endSub);
  }

  edm::EventRange newEventRange(edm::RunNumber_t start, edm::LuminosityBlockNumber_t startLumi, edm::EventNumber_t startSub,
                                edm::RunNumber_t end,   edm::LuminosityBlockNumber_t endLumi, edm::EventNumber_t endSub) {
    return edm::EventRange(start, startLumi, startSub, end, endLumi, endSub);
  }

  void addNewFileInPath(bool tracked, std::string const& name, std::string const& value);

  Python11ParameterSet newPSet() const {return Python11ParameterSet();}

  edm::ParameterSet& pset() {return theParameterSet;}

  edm::ParameterSet const& pset() const {return theParameterSet;}

  std::string dump() const {return theParameterSet.dump();}

private:
  edm::ParameterSet theParameterSet;
};

#endif
