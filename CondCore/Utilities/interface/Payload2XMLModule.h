#ifndef CondCore_Utilities_Payload2XMLModule_h
#define CondCore_Utilities_Payload2XMLModule_h

#include <string>
#include <memory>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "CondFormats/Serialization/interface/Archive.h"

#define XML_CONVERTER_NAME(CLASS_NAME) (std::string(#CLASS_NAME) + "2xml").c_str()

#define PAYLOAD_2XML_MODULE(MODULE_NAME) PYBIND11_MODULE(MODULE_NAME, m)

#define PAYLOAD_2XML_CLASS(CLASS_NAME)                                    \
  py::class_<Payload2xml<CLASS_NAME> >(m, XML_CONVERTER_NAME(CLASS_NAME)) \
      .def(py::init<>())                                                  \
      .def("write", &Payload2xml<CLASS_NAME>::write);

namespace {  // Avoid cluttering the global namespace.

  template <typename PayloadType>
  class Payload2xml {
  public:
    Payload2xml() {}
    //
    std::string write(const std::string &payloadData) {
      // now to convert
      std::unique_ptr<PayloadType> payload;
      std::stringbuf sdataBuf;
      sdataBuf.pubsetbuf(const_cast<char *>(payloadData.c_str()), payloadData.size());

      std::istream inBuffer(&sdataBuf);
      cond::serialization::InputArchive ia(inBuffer);
      payload.reset(new PayloadType);
      ia >> (*payload);

      // now we have the object in memory, convert it to xml in a string and return it
      std::ostringstream outBuffer;
      {
        boost::archive::xml_oarchive xmlResult(outBuffer);
        xmlResult << boost::serialization::make_nvp("cmsCondPayload", *payload);
      }
      return outBuffer.str();
    }
  };

}  // end namespace

#endif
