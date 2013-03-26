#ifndef FWCore_Sources_RawInputSourceFromFiles_h
#define FWCore_Sources_RawInputSourceFromFiles_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <string>
#include <vector>

#include "FWCore/Sources/interface/FromFiles.h"
#include "FWCore/Sources/interface/RawInputSource.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;

  class RawInputSourceFromFiles : public RawInputSource, private FromFiles {
  public:
    RawInputSourceFromFiles(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~RawInputSourceFromFiles();

    using FromFiles::logicalFileNames;
    using FromFiles::fileNames;
    using FromFiles::catalog;
    
    static void fillDescription(ParameterSetDescription& desc);

  protected:
    using FromFiles::incrementFileIndex;

  private:
    using FromFiles::noFiles;
    using FromFiles::fileIndex;
  };
}
#endif
