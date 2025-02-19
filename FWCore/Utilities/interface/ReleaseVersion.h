#ifndef FWCore_Utilities_ReleaseVersion_h
#define FWCore_Utilities_ReleaseVersion_h

#include <string>

namespace edm {
  namespace releaseversion {

    class DecomposedReleaseVersion {
    public:
      explicit DecomposedReleaseVersion(std::string releaseVersion);
      bool operator<(DecomposedReleaseVersion const& other) const;
    private:
      bool irregular_;
      unsigned int major_;
      unsigned int minor_;
//      unsigned int point_;
//      unsigned int patch_;
//      unsigned int pre_;
    };

    bool isEarlierRelease(std::string const& a, std::string const& b);
    bool isEarlierRelease(DecomposedReleaseVersion const& a, std::string const& b);
    bool isEarlierRelease(std::string const& a, DecomposedReleaseVersion const& b);
    bool isEarlierRelease(DecomposedReleaseVersion const& a, DecomposedReleaseVersion const& b);
  }
}
#endif
