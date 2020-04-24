#ifndef FWCore_Sources_ProducerSourceFromFiles_h
#define FWCore_Sources_ProducerSourceFromFiles_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <string>
#include <vector>

#include "FWCore/Sources/interface/FromFiles.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;

  class ProducerSourceFromFiles : public ProducerSourceBase, private FromFiles {
  public:
    ProducerSourceFromFiles(ParameterSet const& pset, InputSourceDescription const& desc, bool realData);
    ~ProducerSourceFromFiles() override;

    using FromFiles::logicalFileNames;
    using FromFiles::fileNames;
    using FromFiles::catalog;

    bool noFiles() const override {
      return fileNames().empty();
    }
    
    static void fillDescription(ParameterSetDescription& desc);

  protected:
    using FromFiles::incrementFileIndex;

  private:
    using FromFiles::fileIndex;
  };
}
#endif
