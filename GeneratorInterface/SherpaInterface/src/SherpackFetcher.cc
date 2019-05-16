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
    std::string option = "-c";
    std::string constr = "`cmsGetFnConnect frontier://smallfiles`";
    std::string sherpack = "sherpa_" + SherpaProcess + "_MASTER.tgz";
    std::string sherpackunzip = "sherpa_" + SherpaProcess + "_MASTER.tar";
    std::string path = SherpackLocation + "/" + sherpack;

    if (FetchSherpack == true) {
      std::cout << "SherpackFetcher: Trying to fetch the Sherpack " << sherpack << std::endl;
      int res = -1;

      res = CopyFile(path);

      if (res != 1) {
        throw cms::Exception("SherpaInterface")
            << "SherpackFetcher: Fetching of Sherpack did not succeed, terminating" << std::endl;
        return -1;
      }
      std::cout << "SherpackFetcher: Fetching successful" << std::endl;
    }

    std::ifstream my_file(sherpack.c_str());
    if (!my_file.good()) {
      throw cms::Exception("SherpaInterface") << "SherpackFetcher: No Sherpack found" << std::endl;
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

    std::cout << "SherpackFetcher: Trying to unzip the Sherpack" << std::endl;
    int res = spu::Unzip(sherpack, sherpackunzip);
    if (res != 0) {
      throw cms::Exception("SherpaInterface") << "SherpackFetcher: Decompressing failed " << std::endl;
      return -4;
    }
    std::cout << "SherpackFetcher: Decompressing successful " << std::endl;

    FILE *file = fopen(const_cast<char *>(sherpackunzip.c_str()), "r");
    if (file) {
      std::cout << "SherpackFetcher: Decompressed Sherpack exists with name " << sherpackunzip
                << " starting to untar it" << std::endl;
      spu::Untar(file, SherpaPath.c_str());
    } else {
      throw cms::Exception("SherpaInterface") << "SherpackFetcher: Could not open decompressed Sherpack" << std::endl;
      return -5;
    }
    fclose(file);
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
