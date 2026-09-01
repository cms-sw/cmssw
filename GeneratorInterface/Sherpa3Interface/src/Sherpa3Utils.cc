#include "GeneratorInterface/Sherpa3Interface/interface/Sherpa3Utils.h"

#include <cstdlib>
#include <fstream>

namespace sh3utils {

  Sherpa3Utils::Sherpa3Utils(edm::ParameterSet const &pset) {
    if (!pset.exists("Sherpa3Process"))
      Sherpa3Process = "";
    else
      Sherpa3Process = pset.getParameter<std::string>("Sherpa3Process");
    if (!pset.exists("GridpackLocation"))
      GridpackLocation = "";
    else
      GridpackLocation = pset.getParameter<std::string>("GridpackLocation");
    if (!pset.exists("GridpackChecksum"))
      GridpackChecksum = "";
    else
      GridpackChecksum = pset.getParameter<std::string>("GridpackChecksum");
    if (!pset.exists("EvtGenDirectory"))
      EvtGenDirectory = "./SHERPA3GEN";
    else
      EvtGenDirectory = pset.getParameter<std::string>("EvtGenDirectory");
  }

  int Sherpa3Utils::Fetch() {
    // GridpackLocation is the full path to a
    // Sherpa3_<process>_<SCRAM_ARCH>_<CMSSW_VERSION>_tarball.tar.xz file
    std::string gridpackPath = GridpackLocation;
    size_t lastSlash = gridpackPath.find_last_of("/\\");
    std::string gridpack = (lastSlash == std::string::npos) ? gridpackPath : gridpackPath.substr(lastSlash + 1);

    std::cout << "Sherpa3Utils: Trying to fetch the Gridpack from " << gridpackPath << std::endl;
    int res = CopyFile(gridpackPath);
    if (res != 1) {
    throw cms::Exception("Sherpa3Interface")
        << "Sherpa3Utils: Fetching of Gridpack did not succeed, terminating" << std::endl;
    return -1;
    }
    std::cout << "Sherpa3Utils: Fetching successful" << std::endl;

    std::ifstream my_file(gridpack.c_str());
    if (!my_file.good()) {
      throw cms::Exception("Sherpa3Interface") << "Sherpa3Utils: No Gridpack found at " << gridpack
                                               << std::endl;
      return -2;
    }
    my_file.close();
    std::cout << "Sherpa3Utils: Gridpack found" << std::endl;

    if (!GridpackChecksum.empty()) {
      char md5checksum[33];
      MD5File(gridpack, md5checksum);
      for (int k = 0; k < 33; k++) {
        if (md5checksum[k] != GridpackChecksum[k]) {
          throw cms::Exception("Sherpa3Interface")
              << "Sherpa3Utils: failure, calculated and specified checksums differ!" << std::endl;
          return -3;
        }
      }
      std::cout << "Sherpa3Utils: Calculated checksum of the Gridpack is " << md5checksum << " and matches"
                << std::endl;
    } else {
      std::cout << "Sherpa3Utils: Ignoring Checksum" << std::endl;
    }

    const char *envCMSSWVersion = std::getenv("CMSSW_VERSION");
    const char *envSCRAMArch = std::getenv("SCRAM_ARCH");
    std::string cmsswVersion = envCMSSWVersion ? envCMSSWVersion : "";
    std::string scramArch = envSCRAMArch ? envSCRAMArch : "";
    if (!cmsswVersion.empty() && !scramArch.empty()) {
      bool versionMatch = (gridpack.find(cmsswVersion) != std::string::npos);
      bool archMatch = (gridpack.find(scramArch) != std::string::npos);
      if (versionMatch && archMatch) {
        std::cout << "Sherpa3Utils: CMSSW_VERSION (" << cmsswVersion << ") and SCRAM_ARCH (" << scramArch
                  << ") match the Gridpack" << std::endl;
      } else {
        if (!versionMatch)
          std::cout << "Sherpa3Utils: WARNING - CMSSW_VERSION mismatch: environment has " << cmsswVersion
                    << " but not found in Gridpack " << gridpack << std::endl;
        if (!archMatch)
          std::cout << "Sherpa3Utils: WARNING - SCRAM_ARCha CH mismatch: environment has " << scramArch
                    << " but not found in Gridpack " << gridpack << std::endl;
      }
    } else {
      std::cout << "Sherpa3Utils: CMSSW_VERSION or SCRAM_ARCH not set in environment, skipping compatibility "
                   "check"
                << std::endl;
    }

    // (re)create the event generation directory: delete it if it exists,
    // then create it fresh
    std::cout << "Sherpa3Utils: Preparing event generation directory " << EvtGenDirectory << std::endl;
    std::string rmCmd = "rm -rf " + EvtGenDirectory;
    if (system(rmCmd.c_str()) != 0) {
      throw cms::Exception("Sherpa3Interface")
          << "Sherpa3Utils: Could not remove existing directory " << EvtGenDirectory << std::endl;
      return -4;
    }
    std::string mkdirCmd = "mkdir -p " + EvtGenDirectory;
    if (system(mkdirCmd.c_str()) != 0) {
      throw cms::Exception("Sherpa3Interface")
          << "Sherpa3Utils: Could not create directory " << EvtGenDirectory << std::endl;
      return -4;
    }

    std::cout << "Sherpa3Utils: Trying to decompress the Gridpack (tar.xz): " << gridpack << std::endl;
    std::string tarCmd = "tar -xJf " + gridpack + " -C " + EvtGenDirectory;
    res = system(tarCmd.c_str());
    if (res != 0) {
      throw cms::Exception("Sherpa3Interface") << "Sherpa3Utils: Decompressing failed " << std::endl;
      return -4;
    }
    std::cout << "Sherpa3Utils: Decompressing successful " << std::endl;
    return 0;
  }

  int Sherpa3Utils::CopyFile(const std::string &pathstring) {
    // if the file is already present in the current directory, no need to copy
    size_t lastSlash = pathstring.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? pathstring : pathstring.substr(lastSlash + 1);
    std::ifstream existing(filename.c_str());
    if (existing.good()) {
      existing.close();
      std::cout << "\t File " << filename << " already present, no need to copy" << std::endl;
      return 1;
    }
    existing.close();
    std::cout << "\t Trying to copy file " << pathstring << std::endl;
    std::string command = "cp " + pathstring + " .";
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
      throw cms::Exception("Sherpa3Interface") << "failed to copy Gridpack ";
    pclose(pipe);
    return 1;
  }

  // function for calculating the MD5 checksum of a file
  void Sherpa3Utils::MD5File(std::string filename, char *result) {
    char buffer[4096];
    cms::openssl_init();
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    const EVP_MD *md = EVP_get_digestbyname("MD5");
    EVP_DigestInit_ex(mdctx, md, nullptr);

    //Open File
    int fd = open(filename.c_str(), O_RDONLY);
    int nb_read;
    while ((nb_read = read(fd, buffer, 4096 - 1))) {
      EVP_DigestUpdate(mdctx, buffer, nb_read);
      memset(buffer, 0, 4096);
    }
    close(fd);

    unsigned int md_len = 0;
    unsigned char tmp[EVP_MAX_MD_SIZE];
    EVP_DigestFinal_ex(mdctx, tmp, &md_len);
    EVP_MD_CTX_free(mdctx);

    assert(result);
    //Convert the result
    for (unsigned int k = 0; k < md_len; ++k) {
      sprintf(result + k * 2, "%02x", tmp[k]);
    }
  }

  Sherpa3Utils::~Sherpa3Utils() {}

}  // namespace sh3utils
