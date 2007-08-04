#ifndef PythonProcessDesc_h
#define PythonProcessDesc_h

#include "FWCore/ParameterSet/interface/PythonParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"

class PythonProcessDesc
{
public:
  PythonProcessDesc();
  /** This constructor will parse the given file,
      and create two objects in python-land:
    * a PythonProcessDesc named 'processDesc'
    * a PythonParameterSet named 'processPSet'
  */
  PythonProcessDesc(const std::string & filename);

  void addService(const PythonParameterSet & pset) {theServices.push_back(pset);}

  PythonParameterSet newPSet() const {return PythonParameterSet();}

  std::string dump() const;

  // makes a new (copy) of the ProcessDesc
  boost::shared_ptr<edm::ProcessDesc> processDesc() const;

private:

  PythonParameterSet theProcessPSet;
  std::vector<PythonParameterSet> theServices;

};

#endif

