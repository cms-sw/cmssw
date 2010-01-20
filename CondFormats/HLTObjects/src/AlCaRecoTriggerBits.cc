#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"


AlCaRecoTriggerBits::AlCaRecoTriggerBits(){}
AlCaRecoTriggerBits::~AlCaRecoTriggerBits(){}

const std::string::value_type AlCaRecoTriggerBits::delimeter_ = ';'; // separator

//_____________________________________________________________________
std::string AlCaRecoTriggerBits::compose(const std::vector<std::string> &paths) const
{
  std::string mergedPaths;
  for (std::vector<std::string>::const_iterator iPath = paths.begin();
       iPath != paths.end(); ++iPath) {
    if (iPath != paths.begin()) mergedPaths += delimeter_;
    mergedPaths += *iPath;
  }
  
  // Special case: DB cannot store empty strings, so choose a single space for that,
  // see e.g. https://hypernews.cern.ch/HyperNews/CMS/get/database/674.html .
  // But of course that means that a single space cannot be specified...
  if (mergedPaths == " ") {
    // cms::Exception? std::cerr? edm::LogError? What to do in CondFormats?
  } else if (mergedPaths.empty()) {
    mergedPaths = " ";
  }
  return mergedPaths;
}

//_____________________________________________________________________
std::vector<std::string> AlCaRecoTriggerBits::decompose(const std::string &s) const
{
  // decompose 's' into its parts that are separated by ';'
  // (similar as in 
  //  Alignment/CommonAlignmentAlgorithm/src/AlignmentParameterSelector.cc)

  std::vector<std::string> result;
  if (s != " ") { // single space indicates an empty list as DB cannot store empty strings
    std::string::size_type previousPos = 0;
    while (true) {
      const std::string::size_type delimiterPos = s.find(delimeter_, previousPos);
      if (delimiterPos == std::string::npos) {
        result.push_back(s.substr(previousPos)); // until end
        break;
      }
      result.push_back(s.substr(previousPos, delimiterPos - previousPos));
      previousPos = delimiterPos + 1; // +1: skip delim
    }
  }

  return result;
}
