#ifndef __DataFormats_PatCandidates_throwMissingLabel_h__
#define __DataFormats_PatCandidates_throwMissingLabel_h__

#include <string>
#include <vector>

namespace pat {
  void throwMissingLabel(const std::string& what, const std::string& bad_label, 
                         const std::vector<std::string>& available);
}

#endif
