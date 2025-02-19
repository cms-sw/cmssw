#ifndef FWCore_ParameterSet_PythonService_h
#define FWCore_ParameterSet_PythonService_h
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
//

#include "FWCore/PythonParameterSet/interface/BoostPython.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

// subpackage specific includes
#include "FWCore/Python/src/EventWrapper.h"
#include "FWCore/Python/src/PythonManager.h"

namespace edm {
   class ConfigurationDescriptions;
}

class PythonService {
    public:
        PythonService(const edm::ParameterSet&,edm::ActivityRegistry&);
        ~PythonService();
        void postBeginJob();
        void postEndJob();
        void preProcessEvent(const edm::EventID&, const edm::Timestamp&);
        void postProcessEvent(const edm::Event&, const edm::EventSetup&);

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

	private:
	    PythonManagerHandle handle_;
		boost::python::object service_;
		std::string fileName_;
		std::string command_;
		

};

#endif // FWCore_ParameterSet_PythonService_h
