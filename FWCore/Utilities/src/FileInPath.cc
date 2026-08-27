// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <ranges>
#include <syncstream>
#include <vector>

#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Parse.h"
#include "FWCore/Utilities/interface/resolveSymbolicLinks.h"

namespace {

  std::atomic<bool> s_fileLookupDisabled{false};

  /// These are the names of the environment variables which control
  /// the behavior  of the FileInPath  class.  They are local to  this
  /// class; other code should not even know about them!

  const std::string PathVariableName("CMSSW_SEARCH_PATH");
  // Environment variables for local and release areas:
  const std::string LOCALTOP("CMSSW_BASE");
  const std::string RELEASETOP("CMSSW_RELEASE_BASE");
  const std::string DATATOP("CMSSW_DATA_PATH");

#if 1
  // Needed for backward compatibility prior to CMSSW_1_5_0_pre3.
  // String to serve as placeholder for release top.
  // Do not change this value.
  const std::string BASE("BASE");
#endif
  const std::string version("V001");

  // Remove symlinks from path
  std::string removeSymLinks(std::string const& envName) {
    char const* const var = std::getenv(envName.c_str());
    if (var == nullptr) {
      return std::string();
    }
    std::string path = var;
    edm::resolveSymbolicLinks(path);
    return path;
  }

  std::string removeSymLinksSrc(std::string const& envName) {
    char const* const var = std::getenv(envName.c_str());
    if (var == nullptr) {
      return std::string();
    }
    std::string const src = "/src";
    std::string path = var + src;
    edm::resolveSymbolicLinks(path);
    size_t actualSize = path.size() - src.size();
    assert(path.substr(actualSize, src.size()) == src);
    return path.substr(0, actualSize);
  }

  std::vector<std::filesystem::path> removeSymLinksTokens(std::string const& envName) {
    char const* const var = std::getenv(envName.c_str());
    if (var == nullptr) {
      return {};
    }
    auto pathElements = edm::tokenize(std::string(var), ":");
    std::vector<std::filesystem::path> ret;
    ret.reserve(pathElements.size());
    for (auto& element : pathElements) {
      edm::resolveSymbolicLinks(element);
      ret.emplace_back(element);
    }
    return ret;
  }

  // Check for existence of a file for the given relative path and
  // 'prefix'.
  // Return true if a file (not directory or symbolic link) is found
  // Return false is *nothing* is found
  // Throw an exception if either a directory or symbolic link is found.
  // If true is returned, then put the
  bool locateFile(std::filesystem::path p, std::string const& relative) {
    p /= relative;

    if (!std::filesystem::exists(p))
      return false;

    if (std::filesystem::is_directory(p)) {
      throw edm::Exception(edm::errors::FileInPathError) << "Path " << p.string() << " is a directory, not a file\n";
    }

    if (std::filesystem::is_symlink(std::filesystem::symlink_status(p))) {
      throw edm::Exception(edm::errors::FileInPathError)
          << "Path " << p.string() << " is a symbolic link, not a file\n";
    }
    return true;
  }

  // Return true if 'path' begins with 'prefix'
  bool pathBeginsWith(std::filesystem::path const& path, std::filesystem::path const& prefix) {
    // lexically_relative() prepends ".." components whenever 'path' has to go up and out of 'prefix'.
    auto const rel = path.lexically_relative(prefix);
    return !rel.empty() && rel.begin()->string() != "..";
  }
}  // namespace

namespace edm {

  FileInPath::FileInPath() : relativePath_(), canonicalFilename_(), location_(Unknown) {
    if (s_fileLookupDisabled) {
      return;
    }
    getEnvironment();
  }

  FileInPath::FileInPath(const std::string& r) : relativePath_(r), canonicalFilename_(), location_(Unknown) {
    if (s_fileLookupDisabled) {
      return;
    }
    getEnvironment();
    initialize_();
  }

  FileInPath::FileInPath(char const* r) : relativePath_(r ? r : ""), canonicalFilename_(), location_(Unknown) {
    if (s_fileLookupDisabled) {
      return;
    }
    if (r == nullptr) {
      throw edm::Exception(edm::errors::FileInPathError) << "Relative path must not be null\n";
    }
    getEnvironment();
    initialize_();
  }

  void FileInPath::swap(FileInPath& other) {
    relativePath_.swap(other.relativePath_);
    canonicalFilename_.swap(other.canonicalFilename_);
    std::swap(location_, other.location_);
    localTop_.swap(other.localTop_);
    releaseTop_.swap(other.releaseTop_);
    dataTop_.swap(other.dataTop_);
  }

  const std::string& FileInPath::relativePath() const { return relativePath_; }

  FileInPath::LocationCode FileInPath::location() const { return location_; }

  const std::string& FileInPath::fullPath() const { return canonicalFilename_; }

  void FileInPath::write(std::ostream& os) const {
    if (location_ == Unknown) {
      if (relativePath_.empty()) {
        os << version << " @ " << location_;
      } else {
        os << version << ' ' << relativePath_ << ' ' << location_;
      }
    } else if (location_ == Local) {
      // Guarantee a site independent value by stripping $LOCALTOP.
      if (localTop_.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << "Environment Variable " << LOCALTOP << " is not set.\n";
      }
      std::string::size_type pos = canonicalFilename_.find(localTop_);
      if (pos != 0) {
        throw edm::Exception(edm::errors::FileInPathError)
            << "Path " << canonicalFilename_ << " is not in the local release area " << localTop_ << "\n";
      }
      os << version << ' ' << relativePath_ << ' ' << location_ << ' ' << canonicalFilename_.substr(localTop_.size());
    } else if (location_ == Release) {
      // Guarantee a site independent value by stripping $RELEASETOP.
      if (releaseTop_.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << "Environment Variable " << RELEASETOP << " is not set.\n";
      }
      std::string::size_type pos = canonicalFilename_.find(releaseTop_);
      if (pos != 0) {
        throw edm::Exception(edm::errors::FileInPathError)
            << "Path " << canonicalFilename_ << " is not in the base release area " << releaseTop_ << "\n";
      }
      os << version << ' ' << relativePath_ << ' ' << location_ << ' ' << canonicalFilename_.substr(releaseTop_.size());
    } else if (location_ == Data) {
      // Guarantee a site independent value by stripping $DATATOP.
      if (dataTop_.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << "Environment Variable " << DATATOP << " is not set.\n";
      }
      std::string::size_type pos = canonicalFilename_.find(dataTop_);
      if (pos != 0) {
        throw edm::Exception(edm::errors::FileInPathError)
            << "Path " << canonicalFilename_ << " is not in the data area " << dataTop_ << "\n";
      }
      os << version << ' ' << relativePath_ << ' ' << location_ << ' ' << canonicalFilename_.substr(dataTop_.size());
    }
  }

  void FileInPath::read(std::istream& is) {
    std::string vsn;
    std::string relname;
    std::string canFilename;
#if 1
    // This #if needed for backward compatibility
    // for files written before CMSSW_1_5_0_pre3.
    is >> vsn;
    if (!is)
      return;
    bool oldFormat = (version != vsn);
    if (oldFormat) {
      relname = vsn;
      bool local;
      is >> local;
      location_ = (local ? Local : Release);
      is >> canFilename;
    } else {
      // Current format
      int loc;
      is >> relname >> loc;
      location_ = static_cast<FileInPath::LocationCode>(loc);
      if (location_ != Unknown) {
        is >> canFilename;
      } else if (relname == "@") {
        relname = "";
      }
    }
#else
    is >> vsn >> relname >> loc >> canFilename;
#endif
    if (!is)
      return;
    relativePath_ = relname;
    if (location_ == Local) {
      if (localTop_.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << "Environment Variable " << LOCALTOP << " is not set.\n"
                                                           << "Trying to read Local file: " << canFilename << ".\n";
      }
#if 1
      // This #if needed for backward compatibility
      // for files written before CMSSW_1_5_0_pre3.
      if (oldFormat) {
        canonicalFilename_ = canFilename;
      } else
#endif
        canonicalFilename_ = localTop_ + canFilename;
    } else if (location_ == Release) {
      if (releaseTop_.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << "Environment Variable " << RELEASETOP << " is not set.\n";
      }
#if 1
      // This #if needed for backward compatibility
      // for files written before CMSSW_1_5_0_pre3.
      if (oldFormat) {
        std::string::size_type pos = canFilename.find(BASE);
        if (pos == 0) {
          // Replace the placehoder with the path to the base release (site dependent).
          canonicalFilename_ = releaseTop_ + canFilename.substr(BASE.size());
        } else {
          // Needed for files written before CMSSW_1_2_0_pre2.
          canonicalFilename_ = canFilename;
        }
      } else
#endif
        canonicalFilename_ = releaseTop_ + canFilename;
    } else if (location_ == Data) {
      if (dataTop_.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << "Environment Variable " << DATATOP << " is not set.\n";
      }
      canonicalFilename_ = dataTop_ + canFilename;
    }
  }

  void FileInPath::readFromParameterSetBlob(std::istream& is) {
    std::string vsn;
    std::string relname;
    std::string canFilename;
    is >> vsn;
    if (!is)
      return;
    bool oldFormat = (version != vsn);
    if (oldFormat) {
      relname = vsn;
      bool local;
      is >> local;
      location_ = (local ? Local : Release);
      is >> canFilename;
    } else {
      // Current format
      int loc;
      is >> relname >> loc;
      location_ = static_cast<FileInPath::LocationCode>(loc);
      if (location_ != Unknown) {
        is >> canFilename;
      } else if (relname == "@") {
        relname = "";
      }
    }
    if (!is)
      return;
    relativePath_ = relname;
    if (location_ == Local) {
      if (localTop_.empty()) {
        localTop_ = "@LOCAL";
      }
      if (oldFormat) {
        canonicalFilename_ = canFilename;
      } else
        canonicalFilename_ = localTop_ + canFilename;
    } else if (location_ == Release) {
      if (releaseTop_.empty()) {
        releaseTop_ = "@RELEASE";
      }
      if (oldFormat) {
        std::string::size_type pos = canFilename.find(BASE);
        if (pos == 0) {
          // Replace the placehoder with the path to the base release (site dependent).
          canonicalFilename_ = releaseTop_ + canFilename.substr(BASE.size());
        } else {
          // Needed for files written before CMSSW_1_2_0_pre2.
          canonicalFilename_ = canFilename;
        }
      } else
        canonicalFilename_ = releaseTop_ + canFilename;
    } else if (location_ == Data) {
      if (dataTop_.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << "Environment Variable " << DATATOP << " is not set.\n";
      }
      canonicalFilename_ = dataTop_ + canFilename;
    }
  }

  //------------------------------------------------------------
  std::vector<std::filesystem::path> const& FileInPath::searchPath() {
    static std::vector<std::filesystem::path> const s_searchPath = removeSymLinksTokens(PathVariableName);
    return s_searchPath;
  }
  //------------------------------------------------------------

  void FileInPath::getEnvironment() {
    static std::string const releaseTop = removeSymLinksSrc(RELEASETOP);
    releaseTop_ = releaseTop;

    static std::string const localTop = removeSymLinksSrc(LOCALTOP);
    localTop_ = localTop;

    static std::string const dataTop = removeSymLinks(DATATOP);
    dataTop_ = dataTop;

    static std::once_flag s_onceFlag;
    std::call_once(s_onceFlag, [this]() {
      auto const& searchPathElements = searchPath();
      if (searchPathElements.empty()) {
        throw edm::Exception(edm::errors::FileInPathError) << PathVariableName << " must be defined\n";
      }
      auto filtered = searchPathElements | std::views::filter([this](std::filesystem::path const& s) {
                        return !s.empty() && !pathBeginsWith(s, std::filesystem::path(releaseTop_)) &&
                               !pathBeginsWith(s, std::filesystem::path(localTop_)) &&
                               !pathBeginsWith(s, std::filesystem::path(dataTop_));
                      }) |
                      std::views::transform([](std::filesystem::path const& s) { return s.string(); });
      std::vector<std::string> const notFound(filtered.begin(), filtered.end());
      if (!notFound.empty()) {
        std::osyncstream ss(std::cerr);
        ss << "Warning: The following elements of $" << PathVariableName << " are not in any of the $" << LOCALTOP
           << ", $" << RELEASETOP << ", or $" << DATATOP << " areas:\n";
        for (const auto& element : notFound) {
          ss << " " << element << "\n";
        }
      }
    });

    if (releaseTop_.empty()) {
      // RELEASETOP was not set.  This means that the environment is set
      // for the base release itself.  So LOCALTOP actually contains the
      // location of the base release.
      releaseTop_ = localTop_;
      localTop_.clear();
    }
    if (releaseTop_ == localTop_) {
      // RELEASETOP is the same as LOCALTOP.  This means that the environment is set
      // for the base release itself.  So LOCALTOP actually contains the
      // location of the base release.
      localTop_.clear();
    }
  }

  void FileInPath::initialize_() {
    if (relativePath_.empty()) {
      throw edm::Exception(edm::errors::FileInPathError) << "Relative path must not be empty";
    }
    if (std::filesystem::path(relativePath_).is_absolute()) {
      throw edm::Exception(edm::errors::FileInPathError)
          << "The path must be relative, not absolute: " << relativePath_;
    }

    // Find the file, based on the value of searchPath.
    // Iterate over every element of CMSSW_SEARCH_PATH
    for (auto const& pathPrefix : searchPath()) {
      // Does the a file exist? locateFile throws is it finds
      // something goofy.
      if (locateFile(pathPrefix, relativePath_)) {
        // Convert relative path to canonical form, and save it.
        relativePath_ = std::filesystem::path(relativePath_).lexically_normal().string();

        // Save the absolute path.
        canonicalFilename_ = std::filesystem::absolute(pathPrefix / relativePath_).string();
        if (canonicalFilename_.empty()) {
          throw edm::Exception(edm::errors::FileInPathError)
              << "fullPath is empty"
              << "\nrelativePath() is: " << relativePath_ << "\npath prefix is: " << pathPrefix.string() << '\n';
        }

        // Determine which search area the current path element belongs to:
        if (!localTop_.empty() && pathBeginsWith(pathPrefix, std::filesystem::path(localTop_))) {
          location_ = Local;
          return;
        }

        if (!releaseTop_.empty() && pathBeginsWith(pathPrefix, std::filesystem::path(releaseTop_))) {
          location_ = Release;
          return;
        }

        if (!dataTop_.empty() && pathBeginsWith(pathPrefix, std::filesystem::path(dataTop_))) {
          location_ = Data;
          return;
        }

        throw edm::Exception(edm::errors::FileInPathError)
            << "edm::FileInPath found file " << relativePath_ << " in search path element " << pathPrefix.string()
            << ", but that element is not in any of the known search areas.\n"
            << "Known search areas are:\n"
            << "  Local area:   " << localTop_ << "\n"
            << "  Release area: " << releaseTop_ << "\n"
            << "  Data area:    " << dataTop_ << "\n";
      }
    }

    // If we got here, we ran out of path elements without finding
    // what we're looking found.
    throw edm::Exception(edm::errors::FileInPathError)
        << "edm::FileInPath unable to find file " << relativePath_ << " anywhere in the search path."
        << "\nThe search path is defined by: " << PathVariableName << "\n${" << PathVariableName
        << "} is: " << std::getenv(PathVariableName.c_str())
        << "\nCurrent directory is: " << std::filesystem::current_path().string() << "\n";
  }

  void FileInPath::disableFileLookup() { s_fileLookupDisabled = true; }

  std::string FileInPath::findFile(const std::string& iFileName) {
    // Find the file, based on the value of path variable.
    for (auto const& pathPrefix : searchPath()) {
      // Does the a file exist? locateFile throws is it finds
      // something goofy.
      if (locateFile(pathPrefix, iFileName)) {
        // Convert relative path to canonical form, and save it.
        return std::filesystem::absolute(pathPrefix / iFileName).string();
      }
    }
    return {};
  }

}  // namespace edm
