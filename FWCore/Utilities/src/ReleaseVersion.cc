#include "FWCore/Utilities/interface/ReleaseVersion.h"

#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <vector>

namespace edm {
  namespace releaseversion {
    
    struct IsNotDigit {
      bool operator()(char const c) const {
        return !isdigit(c);
      } 
    };

    struct IsEmpty {
      bool operator()(std::string const& s) const {
        return s.empty();
      } 
    };

    DecomposedReleaseVersion::DecomposedReleaseVersion(std::string releaseVersion) : irregular_(true), major_(0), minor_(0)/*, point_(0), patch_(0), pre_(0)*/ {
      try {
        std::vector<std::string> parts;
        parts.reserve(releaseVersion.size());
        boost::algorithm::split(parts, releaseVersion, IsNotDigit(), boost::algorithm::token_compress_on);
        parts.erase(remove_if(parts.begin(), parts.end(), IsEmpty()), parts.end());

        if(parts.size() < 3) {
	  return;
        }
/*
        if(parts.size() == 4) {
          if(releaseVersion.find("patch") != std::string::npos) {
            patch_ = boost::lexical_cast<unsigned int>(parts[3]);
          } else if(releaseVersion.find("pre") != std::string::npos) {
            pre_ = boost::lexical_cast<unsigned int>(parts[3]);
          } else {
            return;
          }
        }
*/
        major_ = boost::lexical_cast<unsigned int>(parts[0]);
        minor_ = boost::lexical_cast<unsigned int>(parts[1]);
//        point_ = boost::lexical_cast<unsigned int>(parts[2]);
        irregular_ = false;
      } catch(std::exception const&) {
      }
    }


    bool DecomposedReleaseVersion::operator<(DecomposedReleaseVersion const& other) const {
      if(irregular_ || other.irregular_) return false;
      if(major_ < other.major_) return true;
      if(major_ > other.major_) return false;
      if(minor_ < other.minor_) return true;
//      if(minor_ > other.minor_) return false;
//      if(point_ < other.point_) return true;
//      if(point_ > other.point_) return false;
//      if(patch_ < other.patch_) return true;
//      if(patch_ > other.patch_) return false;
//      if(pre_ < other.pre_) return true;
      return false;
    }

    bool
    isEarlierRelease(std::string const& a, std::string const& b) {
      return(DecomposedReleaseVersion(a) < DecomposedReleaseVersion(b));
    }

    bool
    isEarlierRelease(DecomposedReleaseVersion const& a, std::string const& b) {
      return(a < DecomposedReleaseVersion(b));
    }

    bool
    isEarlierRelease(std::string const& a, DecomposedReleaseVersion const& b) {
      return(DecomposedReleaseVersion(a) < b);
    }

    bool
    isEarlierRelease(DecomposedReleaseVersion const& a, DecomposedReleaseVersion const& b) {
      return(a < b);
    }

  }
}
