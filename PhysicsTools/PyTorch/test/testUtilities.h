#ifndef PhysicsTools_PyTorch_test_testUtilities_h
#define PhysicsTools_PyTorch_test_testUtilities_h

#include <filesystem>
#include <string>

namespace torchtest {

  std::string cmsswPath(std::string path) {
    if (path.size() > 0 && path.substr(0, 1) != "/")
      path = "/" + path;

    std::string base = std::string(std::getenv("CMSSW_BASE"));
    std::string rel = std::string(std::getenv("CMSSW_RELEASE_BASE"));
    return (std::filesystem::exists(base.c_str()) ? base : rel) + path;
  }

}  // namespace torchtest

#endif  // PhysicsTools_PyTorch_test_testUtilities_h