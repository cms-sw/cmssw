#include <iostream>
#include <TSystem.h>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"

#include "Utilities/OpenSSL/interface/openssl_init.h"

using namespace std;

int main(int argc, char** argv) {
  // load framework libraries
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();

  if (argc < 3) {
    std::cout << "Usage : " << argv[0] << " [your_cutflow_only.py] "
              << "[cutflow_name] " << std::endl;
    return 0;
  }

  if (!edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>(argv[2])) {
    std::cout << " ERROR: ParametersSet '" << argv[2] << "' is missing in your configuration file" << std::endl;
    exit(0);
  }

  edm::ParameterSet conf = edm::readPSetsFrom(argv[1])->getParameterSet(argv[2]);
  edm::ParameterSet trackedconf = conf.trackedPart();

  std::string tracked(trackedconf.dump()), untracked(conf.dump());

  if (tracked != untracked) {
    throw cms::Exception("InvalidConfiguration") << "IDs are not allowed to have untracked parameters"
                                                 << " in the configuration ParameterSet!";
  }

  // now setup the md5 and cute accessor functions
  cms::openssl_init();
  EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
  const EVP_MD* md = EVP_get_digestbyname("SHA1");
  unsigned int md_len = 0;
  unsigned char id_md5_[EVP_MAX_MD_SIZE];

  EVP_DigestInit_ex(mdctx, md, nullptr);
  EVP_DigestUpdate(mdctx, (const unsigned char*)tracked.c_str(), tracked.size());
  EVP_DigestFinal_ex(mdctx, id_md5_, &md_len);
  EVP_MD_CTX_free(mdctx);

  printf("%s : ", argv[2]);
  for (unsigned i = 0; i < md_len; ++i) {
    printf("%02x", id_md5_[i]);
  }
  printf("\n");

  return 0;
}
