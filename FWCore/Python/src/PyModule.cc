/*
 *  PyModule.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 3/22/06.
 *
 */
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/exception_translator.hpp>

#include "FWCore/Framework/interface/GenericHandle.h"

#include "FWCore/Python/src/EventWrapper.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "Rtypes.h"
#include "TClass.h"

class TClass;
namespace PyROOT {
  PyObject* BindRootObject( void* object, TClass* klass, Bool_t isRef = kFALSE );
}

namespace {
   void (edm::python::ConstEventWrapper::*getByLabel1)(std::string const& , edm::GenericHandle& ) const = &edm::python::ConstEventWrapper::getByLabel;
   void (edm::python::ConstEventWrapper::*getByLabel2)(std::string const& , std::string const&, edm::GenericHandle& ) const = &edm::python::ConstEventWrapper::getByLabel;
  
  PyObject* getFromHandle(const edm::GenericHandle& iHandle)
  {
     if(0 == iHandle.product() ) {
	boost::python::object null;
	return null.ptr();
     }
    return PyROOT::BindRootObject( iHandle.product()->address(), TClass::GetClass(iHandle.type().typeInfo()));
  }
}

//this function is used to translate C++ exceptions to python exceptions
static 
void
exceptionTranslation(cms::Exception const& iException) {
   PyErr_SetString(PyExc_RuntimeError, iException.what());
}


using namespace boost::python;
BOOST_PYTHON_MODULE(libFWCorePython)
{
  register_exception_translator< cms::Exception >(exceptionTranslation);

  class_<edm::python::ConstEventWrapper>("ConstEvent",no_init)
  .def("getByLabel",getByLabel1)
  .def("getByLabel",getByLabel2);
  
  class_<edm::GenericHandle>("Handle",init<std::string>())
  .def("get",getFromHandle);
}
