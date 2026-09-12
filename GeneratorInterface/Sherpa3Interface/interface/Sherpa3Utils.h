#ifndef GeneratorInterface_Sherpa3Interface_Sherpa3Utils_h
#define GeneratorInterface_Sherpa3Interface_Sherpa3Utils_h

#include <iostream>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <openssl/evp.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Utilities/OpenSSL/interface/openssl_init.h"

namespace sh3utils {

  class Sherpa3Utils {
  public:
    Sherpa3Utils(edm::ParameterSet const &);
    ~Sherpa3Utils();
    int Fetch();
    const char *classname() const { return "Sherpa3Utils"; }

  private:
    int CopyFile(const std::string &pathstring);
    // function for calculating the MD5 checksum of a file
    void MD5File(std::string, char*);
    std::string Sherpa3Process;
    std::string GridpackLocation;
    std::string GridpackChecksum;
    std::string EvtGenDirectory;
  };

}  // namespace sh3utils

#endif
