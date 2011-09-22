#ifndef IOPool_Input_InputFile_h
#define IOPool_Input_InputFile_h

/*----------------------------------------------------------------------

Holder for an input TFile.
----------------------------------------------------------------------*/
#include "FWCore/MessageLogger/interface/JobReport.h"

#include "TFile.h"

#include "boost/scoped_ptr.hpp"
#include "boost/utility.hpp"

#include <string>
#include <vector>

class TObject;

namespace edm {
  class InputFile : private boost::noncopyable {
  public:  
    explicit InputFile(char const* fileName, char const* msg);
    ~InputFile();
    void Close();
    void inputFileOpened(std::string const& logicalFileName,
                         std::string const& inputType,
                         std::string const& moduleName,
                         std::string const& label,
                         std::string const& fid,
                         std::vector<std::string> const& branchNames);    
    void eventReadFromFile(unsigned int run, unsigned int event) const;
    void reportInputRunNumber(unsigned int run) const;
    void reportInputLumiSection(unsigned int run, unsigned int lumi) const;
    static void reportSkippedFile(std::string const& fileName, std::string const& logicalFileName);

    TObject* Get(char const* name) {return file_->Get(name);}
    TFileCacheRead* GetCacheRead() const {return file_->GetCacheRead();}
    void SetCacheRead(TFileCacheRead* tfcr) {file_->SetCacheRead(tfcr);}
    void logFileAction(char const* msg, char const* fileName) const;
  private:
    boost::scoped_ptr<TFile> file_;
    std::string fileName_;
    JobReport::Token reportToken_;
  }; 
}
#endif
