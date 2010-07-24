// -*- C++ -*-
//
// Package:    PythonFilter
// Class:      PythonFilter
// 
/**\class PythonFilter PythonFilter.cc FWCore/PythonFilter/src/PythonFilter.cc

 Description: an EDFilter which uses python code to do the work

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris D Jones & Benedikt Hegner
//         Created:  Sun Jul 22 10:50:17 EST 2006
//
#ifndef FWCore_Python_PythonFilter_h
#define FWCore_Python_PythonFilter_h

#include "FWCore/PythonParameterSet/interface/BoostPython.h"
#include "FWCore/Python/src/PythonManager.h"
#include "FWCore/Framework/interface/EDFilter.h"

class PythonFilter : public edm::EDFilter {
   public:
      explicit PythonFilter(const edm::ParameterSet&);
      ~PythonFilter();

      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      std::string command_; 
      //have this first to guarantee that Py_Finalize not called
      // until the object's destructor is called
      PythonManagerHandle handle_;
      boost::python::object filter_;
};

#endif // FWCore_Python_PythonFilter_h
