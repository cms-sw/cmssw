#include "FWCore/Utilities/interface/resolveSymbolicLinks.h"
#include "FWCore/Utilities/interface/Parse.h"

#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

#include <vector>

namespace edm {

  namespace {
    namespace bf = boost::filesystem;
    bool resolveOneSymbolicLink(std::string& fullPath) {
      if(fullPath.empty()) return false;
      if(fullPath[0] != '/') return false;
      std::string pathToResolve;
      std::vector<std::string> pathElements = edm::tokenize(fullPath, "/");
      for(auto const& path : pathElements) {
        if(!path.empty()) {
          pathToResolve += "/";
          pathToResolve += path;
          bf::path symLinkPath(pathToResolve);
          if (bf::is_symlink(bf::symlink_status(symLinkPath))) {
            bf::path resolved = bf::read_symlink(symLinkPath);
            // This check is needed because in weird filesystems
            // (e.g. AFS), the resolved link may not be accessible.
            if(!bf::exists(resolved)) {
              continue;
            }
            std::string resolvedPath = resolved.string();
            auto begin = fullPath.begin();
            auto end = begin + pathToResolve.size();
            // resolvedPath may or may not contain the leading "/".
            if(resolvedPath[0] == '/') {
              fullPath.replace(begin, end, resolvedPath);
            } else {
              fullPath.replace(begin + 1, end, resolvedPath);
            }
            return true;
          }
        }
      }
      return false;
    }
  }

  // Resolves symlinks recursively from anywhere in fullPath.
  void resolveSymbolicLinks(std::string& fullPath) {
    bool found = resolveOneSymbolicLink(fullPath);
    if(found) {
      resolveSymbolicLinks(fullPath);
    }
  }
}
