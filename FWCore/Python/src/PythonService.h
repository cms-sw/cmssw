#ifndef PythonService_h
#define PythonService_h
// -*- C++ -*-
//
// Package:     Python
// Class  :     PythonService
// 
/**\class PythonService PythonService.h FWCore/Python/src/PythonService.h

 Description: interface to write EDMService in Python

 Usage:
    <usage>

*/
//
// Original Author:  Benedikt Hegner
//         Created:  Sun Jul 23 11:31:33 CEST 2006
// $Id: PythonService.h,v 1.1 2006/07/23 15:41:54 hegner Exp $
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "boost/python.hpp"

// subpackage specific includes
#include "FWCore/Python/src/EventWrapper.h"
#include "FWCore/Python/src/PythonManager.h"




class PythonService {
    public:
        PythonService(const edm::ParameterSet&,edm::ActivityRegistry&);
        ~PythonService();
        void postBeginJob();
        void postEndJob();
        void preProcessEvent(const edm::EventID&, const edm::Timestamp&);
        void postProcessEvent(const edm::Event&, const edm::EventSetup&);
		
	private:
	    PythonManagerHandle handle_;
		boost::python::object service_;
		std::string fileName_;
		std::string command_;
		

};

#endif // PythonService_h
