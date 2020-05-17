
#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include <stdexcept>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/Serialization/interface/Archive.h"
#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/RunInfo/src/headers.h"

// The compiler knows our default-constructed objects' members
// may not be initialized when we serialize them.
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

template <typename T>
void toXML(std::string filename = "") {
  if (filename.empty()) {
    filename = std::string(typeid(T).name()) + ".xml";
  }

  // C++ does not allow to construct const objects
  // of non-POD types without user-provided default constructor
  // (since it would be uninitialized), so we always create
  // a non-const object.
  T originalObject;
  const T& originalObjectRef = originalObject;
  {
    std::ofstream ofs(filename);
    //cond::serialization::OutputArchiveXML oa(ofs);
    boost::archive::xml_oarchive oa(ofs);
    std::cout << "Serializing " << typeid(T).name() << " ..." << std::endl;
    oa& boost::serialization::make_nvp("cmsCondPayload", originalObjectRef);
  }
}

template <typename T>
T fromXML(std::string filename = "") {
  if (filename.empty()) {
    filename = std::string(typeid(T).name()) + ".xml";
  }

  // C++ does not allow to construct const objects
  // of non-POD types without user-provided default constructor
  // (since it would be uninitialized), so we always create
  // a non-const object.
  T originalObject;
  {
    std::cout << "going to read from " << filename << std::endl;
    std::ifstream ifs(filename);
    try {
      // cond::serialization::InputArchiveXML ia( ifs );
      boost::archive::xml_iarchive ia(ifs);
      std::cout << "Deserializing " << typeid(T).name() << " ..." << std::endl;
      ia& boost::serialization::make_nvp("cmsCondPayload", originalObject);
    } catch (const boost::archive::archive_exception& ae) {
      std::cout << "** boost::archive load exception"
                << ": " << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << ae.what() << "\n"
                << std::endl;
    }
  }
  return originalObject;
}

class Simple {
public:
  Simple() : my_i(42), my_f(42.) { /* nop */
  }

  // COND_SERIALIZABLE;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version);

private:
  int my_i;
  float my_f;
};

template <class Archive>
void Simple::serialize(Archive& ar, const unsigned int version) {
  ar& BOOST_SERIALIZATION_NVP(my_i);
  ar& BOOST_SERIALIZATION_NVP(my_f);
}

int main(int, char**) {
  toXML<Simple>();
  fromXML<Simple>();

  std::cout << "Simple done" << std::endl;

  RunInfo ri = fromXML<RunInfo>("src/CondFormats/RunInfo/test/RunInfo-000fe8.xml");
  std::cout << "Found info for run " << ri.m_run << std::endl;
  std::cout << "        started at " << ri.m_start_time_str << std::endl;

  return 0;
}
