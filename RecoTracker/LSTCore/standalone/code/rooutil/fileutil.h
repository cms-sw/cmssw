//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef fileutil_h
#define fileutil_h

#include <vector>
#include <map>
#include <dirent.h>
#include <glob.h>    // glob(), globfree()
#include <string.h>  // memset()
#include <stdexcept>
#include <string>
#include <sstream>
#include <unistd.h>

#include "TChain.h"
#include "TDirectory.h"
#include "TH1.h"
#include "stringutil.h"
#include "printutil.h"

namespace RooUtil {
  namespace FileUtil {
    TChain* createTChain(TString, TString);
    TH1* get(TString);
    std::map<TString, TH1*> getAllHistograms(TFile*);
    void saveAllHistograms(std::map<TString, TH1*>, TFile*);
    std::vector<TString> getFilePathsInDirectory(TString dirpath);
    std::vector<TString> glob(const std::string& pattern);
  }  // namespace FileUtil
}  // namespace RooUtil

#endif
