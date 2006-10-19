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
// $Id: PythonFilter.h,v 1.2 2006/07/22 13:47:13 hegner Exp $
//
//
#ifndef PythonFilter_h
#define PythonFilter_h

#include "boost/python.hpp"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

#endif // PythonFilter_h
