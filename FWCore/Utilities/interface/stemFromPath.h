#ifndef FWCore_Utilities_stemFromPath_h
#define FWCore_Utilities_stemFromPath_h

#include <string>

namespace edm {
  // This functions extracts the stem of a file from the path (= file
  // name without the extension).
  //
  // The reason to have our own function instead of
  // std/boost::filesystem is that tehcnically these paths are not
  // filesystem paths, but paths in CMS LFN/PFN space that (may) have
  // different rules.
  std::string stemFromPath(const std::string& path);
}  // namespace edm

#endif
