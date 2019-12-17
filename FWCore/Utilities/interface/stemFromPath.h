#ifndef FWCore_Utilities_stemFromPath_h
#define FWCore_Utilities_stemFromPath_h

#include <string_view>

namespace edm {
  // This functions extracts the stem of a file from the path (= file
  // name without the extension). The returned value is a string_view
  // to the input string. Caller should ensure that the input string
  // object lives long enough.
  //
  // The reason to have our own function instead of
  // std/boost::filesystem is that tehcnically these paths are not
  // filesystem paths, but paths in CMS LFN/PFN space that (may) have
  // different rules.
  std::string_view stemFromPath(std::string_view path);
}  // namespace edm

#endif
