#ifndef FWCore_Utilities_src_pathBeginsWith_h
#define FWCore_Utilities_src_pathBeginsWith_h

#include <filesystem>

namespace edm::detail {
  // This is a helper function used in FileInPath. It is defined here for unit tests. It is not part of the public interface.
  // Return true if 'path' begins with 'prefix'
  inline bool pathBeginsWith(std::filesystem::path const& path, std::filesystem::path const& prefix) {
    // lexically_relative() prepends ".." components whenever 'path' has to go up and out of 'prefix'.
    // Normalize first so that e.g. "/a/./b" and "/a/d/../b" are treated as "/a/b".
    auto const rel = path.lexically_normal().lexically_relative(prefix.lexically_normal());
    // 'rel' is empty e.g. if 'path' and 'prefix' have no common beginning
    // if 'path' begins withg 'prefix', then 'rel' does not start with '..'
    return !rel.empty() && rel.begin()->string() != "..";
  }
}  // namespace edm::detail

#endif
