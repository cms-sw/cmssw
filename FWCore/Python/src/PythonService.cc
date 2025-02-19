// -*- C++ -*-
//
// Package:     Python
// Class  :     PythonService
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Benedikt Hegner
//         Created:  Sun Jul 23 11:31:33 CEST 2006
//

// system include files

// user include files
#include "FWCore/Python/src/PythonService.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// constructor
//
PythonService::PythonService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry):
    handle_(PythonManager::handle()) {

    std::cout << "Start preparing PythonService" << std::endl;
    std::string const fileName = iConfig.getParameter<std::string>("fileName");

    using namespace boost::python;

    command_ = "from " + fileName + " import *\n";
    object main_module((boost::python::handle<>(borrowed(PyImport_AddModule(const_cast<char*>("__main__"))))));
    object main_namespace = main_module.attr("__dict__");
    try {
      object result((boost::python::handle<>(PyRun_String(command_.c_str(),
                                             Py_file_input,
                                             main_namespace.ptr(),
                                             main_namespace.ptr()))));
      service_ = main_namespace["service"];
    } catch(...) {
      throw cms::Exception("Configuration") << "No 'service' python variable defined in given fileName parameter.\n Please create an instance of the python class you want to use and pass that instance to the variable named 'service'.";
    }

    // connect methods and signals
    // later on here will be a check what python methods are present
    // for now we expect postBeginJob, postEndJob and postProcessEvent

    iRegistry.watchPostBeginJob(this,&PythonService::postBeginJob);
    iRegistry.watchPostEndJob(this,&PythonService::postEndJob);
    iRegistry.watchPostProcessEvent(this,&PythonService::postProcessEvent);
}

//
// destructor
//
PythonService::~PythonService() {
}

//
// member functions
//
void PythonService::postBeginJob() {
        using namespace boost::python;
    object main_module((boost::python::handle<>(borrowed(PyImport_AddModule(const_cast<char *>("__main__"))))));
    object main_namespace = main_module.attr("__dict__");
    main_namespace["tempService"] = service_;

    try {
      object result((boost::python::handle<>(PyRun_String("tempService.postBeginJob()",
                                             Py_eval_input,
                                             main_namespace.ptr(),
                                             main_namespace.ptr()))));
    } catch(error_already_set) {
      pythonToCppException("RuntimeError");
    }
}

void PythonService::postEndJob() {
    using namespace boost::python;
    object main_module((boost::python::handle<>(borrowed(PyImport_AddModule(const_cast<char *>("__main__"))))));
    object main_namespace = main_module.attr("__dict__");
    main_namespace["tempService"] = service_;

    try {
       object result((boost::python::handle<>(PyRun_String("tempService.postEndJob()",
                                                           Py_eval_input,
                                                           main_namespace.ptr(),
                                                           main_namespace.ptr()))));
    } catch(error_already_set) {
       pythonToCppException("RuntimeError");
    }
}

void PythonService::preProcessEvent(edm::EventID const&, edm::Timestamp const&) {
}

void PythonService::postProcessEvent(edm::Event const&, edm::EventSetup const&) {
    using namespace boost::python;
    object main_module((boost::python::handle<>(borrowed(PyImport_AddModule(const_cast<char *>("__main__"))))));
    object main_namespace = main_module.attr("__dict__");
    main_namespace["tempService"] = service_;

    try {
       object result((boost::python::handle<>(PyRun_String("tempService.postProcessEvent()",
                                                           Py_eval_input,
                                                           main_namespace.ptr(),
                                                           main_namespace.ptr()))));
    } catch(error_already_set) {
       pythonToCppException("RuntimeError");
    }

}

void PythonService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("fileName");
    descriptions.addDefault(desc);
}
