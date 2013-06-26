#ifndef FWCore_Sources_FromFiles_h
#define FWCore_Sources_FromFiles_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <vector>
#include <string>

#include "FWCore/Catalog/interface/InputFileCatalog.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;

  class FromFiles {
  public:
    FromFiles(ParameterSet const& pset);
    ~FromFiles();

    std::vector<std::string> const& logicalFileNames() const {return catalog_.logicalFileNames();}
    std::vector<std::string> const& fileNames() const {return catalog_.fileNames();}
    InputFileCatalog& catalog() {return catalog_;}
    
    static void fillDescription(ParameterSetDescription& desc);

    void incrementFileIndex() {++fileIndex_;}

    size_t fileIndex() const;

  private:

    InputFileCatalog catalog_;
    size_t fileIndex_;
  };
}
#endif
