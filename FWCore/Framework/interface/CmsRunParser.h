#ifndef FWCore_Framework_CmsRunParser_h
#define FWCore_Framework_CmsRunParser_h

#include "boost/program_options.hpp"

#include <variant>

namespace edm {
  class CmsRunParser {
  public:
    using MapOrExit = std::variant<boost::program_options::variables_map, int>;
    CmsRunParser(const char* name);
    MapOrExit parse(int argc, char* argv[]) const;

  private:
    boost::program_options::options_description desc_;
    boost::program_options::options_description all_options_;
    boost::program_options::positional_options_description pos_options_;

  public:
    //Command line parameters
    static inline const char* const kParameterSetOpt = "parameter-set";
    static inline const char* const kPythonOpt = "pythonOptions";
    static inline const char* const kPythonOptDefault = "CMSRUN_PYTHONOPT_DEFAULT";
    static inline const char* const kCmdCommandOpt = "command,c";
    static inline const char* const kCmdOpt = "command";
    static inline const char* const kJobreportCommandOpt = "jobreport,j";
    static inline const char* const kJobreportOpt = "jobreport";
    static inline const char* const kEnableJobreportCommandOpt = "enablejobreport,e";
    static inline const char* const kEnableJobreportOpt = "enablejobreport";
    static inline const char* const kJobModeCommandOpt = "mode,m";
    static inline const char* const kJobModeOpt = "mode";
    static inline const char* const kNumberOfThreadsCommandOpt = "numThreads,n";
    static inline const char* const kNumberOfThreadsOpt = "numThreads";
    static inline const char* const kSizeOfStackForThreadCommandOpt = "sizeOfStackForThreadsInKB,s";
    static inline const char* const kSizeOfStackForThreadOpt = "sizeOfStackForThreadsInKB";
    static inline const char* const kHelpOpt = "help";
    static inline const char* const kHelpCommandOpt = "help,h";
    static inline const char* const kStrictOpt = "strict";    
  };
}

#endif
