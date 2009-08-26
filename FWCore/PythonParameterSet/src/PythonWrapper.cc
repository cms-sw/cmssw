#include "FWCore/PythonParameterSet/src/PythonWrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
//#include <iostream>
namespace edm {

void pythonToCppException(const std::string& iType)
 {
  using namespace boost::python;
  PyObject *exc=NULL, *val=NULL, *trace=NULL;
  PyErr_Fetch(&exc,&val,&trace);
  PyErr_NormalizeException(&exc,&val,&trace);
  handle<> hExc(allow_null(exc));
  handle<> hVal(allow_null(val));
  handle<> hTrace(allow_null(trace));
 
  if(hTrace) {
    object oTrace(hTrace);
    handle<> hStringTr(PyObject_Str(oTrace.ptr()));
    object stringTr(hStringTr);
//std::cout << "PR TR " << stringTr <<  " DONE "<<  std::endl;
  }

  if(hVal && hExc) {
    object oExc(hExc);
    object oVal(hVal);
    handle<> hStringVal(PyObject_Str(oVal.ptr()));
    object stringVal( hStringVal );

    handle<> hStringExc(PyObject_Str(oExc.ptr()));
    object stringExc( hStringExc);

    //PyErr_Print();
    throw cms::Exception(iType) <<"python encountered the error: "
                                << PyString_AsString(stringExc.ptr())<<" "
                                << PyString_AsString(stringVal.ptr())<<"\n";
  } else {
    throw cms::Exception(iType)<<" unknown python problem occurred.\n";
  }
}

}


