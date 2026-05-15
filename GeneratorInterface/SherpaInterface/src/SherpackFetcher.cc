#include "GeneratorInterface/SherpaInterface/interface/SherpackFetcher.h"

namespace spf {

  SherpackFetcher::SherpackFetcher(edm::ParameterSet const &pset) {
    if (!pset.exists("SherpaProcess"))
      SherpaProcess = "";
    else
      SherpaProcess = pset.getParameter<std::string>("SherpaProcess");
    if (!pset.exists("SherpackLocation"))
      SherpackLocation = "";
    else
      SherpackLocation = pset.getParameter<std::string>("SherpackLocation");
    if (!pset.exists("SherpackChecksum"))
      SherpackChecksum = "";
    else
      SherpackChecksum = pset.getParameter<std::string>("SherpackChecksum");
    if (!pset.exists("FetchSherpack"))
      FetchSherpack = false;
    else
      FetchSherpack = pset.getParameter<bool>("FetchSherpack");
    if (!pset.exists("SherpaPath"))
      SherpaPath = "";
    else
      SherpaPath = pset.getParameter<std::string>("SherpaPath");
  }

  int SherpackFetcher::Fetch() {
    // Extract filename from SherpackLocation (which is a full path)
    std::string sherpackPath = SherpackLocation;
    size_t lastSlash = sherpackPath.find_last_of("/\\");
    std::string sherpack = (lastSlash == std::string::npos) ? sherpackPath : sherpackPath.substr(lastSlash + 1);

    if (FetchSherpack == true) {
      std::cout << "SherpackFetcher: Trying to fetch the Sherpack from " << sherpackPath << std::endl;
      int res = CopyFile(sherpackPath);

      if (res != 1) {
        throw cms::Exception("SherpaInterface")
            << "SherpackFetcher: Fetching of Sherpack did not succeed, terminating" << std::endl;
        return -1;
      }
      std::cout << "SherpackFetcher: Fetching successful" << std::endl;
    }

    std::ifstream my_file(sherpack.c_str());
    if (!my_file.good()) {
      throw cms::Exception("SherpaInterface") << "SherpackFetcher: No Sherpack found at " << sherpack << std::endl;
      return -2;
    }
    my_file.close();
    std::cout << "SherpackFetcher: Sherpack found" << std::endl;

    if (!SherpackChecksum.empty()) {
      char md5checksum[33];
      spu::md5_File(sherpack, md5checksum);
      for (int k = 0; k < 33; k++) {
        if (md5checksum[k] != SherpackChecksum[k]) {
          throw cms::Exception("SherpaInterface")
              << "SherpackFetcher: failure, calculated and specified checksums differ!" << std::endl;
          return -3;
        }
      }
      std::cout << "SherpackFetcher: Calculated checksum of the Sherpack is " << md5checksum << " and matches"
                << std::endl;
    } else {
      std::cout << "SherpackFetcher: Ignoring Checksum" << std::endl;
    }

    const char *envCMSSWVersion = std::getenv("CMSSW_VERSION");
    const char *envSCRAMArch = std::getenv("SCRAM_ARCH");
    std::string cmsswVersion = envCMSSWVersion ? envCMSSWVersion : "";
    std::string scramArch = envSCRAMArch ? envSCRAMArch : "";
    if (!cmsswVersion.empty() && !scramArch.empty()) {
      bool versionMatch = (sherpack.find(cmsswVersion) != std::string::npos);
      bool archMatch = (sherpack.find(scramArch) != std::string::npos);
      if (versionMatch && archMatch) {
        std::cout << "SherpackFetcher: CMSSW_VERSION (" << cmsswVersion << ") and SCRAM_ARCH (" << scramArch
                  << ") match the Sherpack" << std::endl;
      } else {
        if (!versionMatch)
          std::cout << "SherpackFetcher: WARNING - CMSSW_VERSION mismatch: environment has " << cmsswVersion
                    << " but not found in Sherpack " << sherpack << std::endl;
        if (!archMatch)
          std::cout << "SherpackFetcher: WARNING - SCRAM_ARCH mismatch: environment has " << scramArch
                    << " but not found in Sherpack " << sherpack << std::endl;
      }
    } else {
      std::cout << "SherpackFetcher: CMSSW_VERSION or SCRAM_ARCH not set in environment, skipping compatibility check"
                << std::endl;
    }

    std::cout << "SherpackFetcher: Trying to decompress the Sherpack (tar.xz): " << sherpack << std::endl;
    std::string tarCmd = "tar -xJf " + sherpack;
    int res = system(tarCmd.c_str());
    if (res != 0) {
      throw cms::Exception("SherpaInterface") << "SherpackFetcher: Decompressing failed " << std::endl;
      return -4;
    }
    std::cout << "SherpackFetcher: Decompressing successful " << std::endl;
    return 0;
  }

  int SherpackFetcher::CopyFile(std::string pathstring) {
    //No need to backwards compatibility with the FnFileGet method, throw exception if only the relative path is given
    if ((pathstring.find("slc6_amd64_gcc") == 0) || (pathstring.find("slc5_amd64_gcc") == 0)) {
      throw cms::Exception("SherpaInterface") << "Old method of sherpack retrieving used, please use /cvmfs to store "
                                                 "files and specify the full path to the sherpack directory";
    }
    std::cout << "Trying to copy file " << pathstring << std::endl;
    std::string command = "cp " + pathstring + " .";
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
      throw cms::Exception("SherpaInterface") << "failed to copy Sherpack ";
    pclose(pipe);
    return 1;
  }

  SherpackFetcher::~SherpackFetcher() {}

}  // namespace spf

//~ using spf::SherpackFetcher;
//~ DEFINE_FWK_MODULE(SherpackFetcher);
