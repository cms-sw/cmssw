#ifndef IOPool_Output_PoolOutputModule_h
#define IOPool_Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// $Id: PoolOutputModule.h,v 1.46 2008/03/04 00:05:03 paterno Exp $
//
// Class PoolOutputModule. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <iosfwd>
#include "boost/scoped_ptr.hpp"
#include "boost/utility.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "IOPool/Output/src/RootOutputFile.h"

namespace edm {
  class ParameterSet;

  class PoolOutputModule : public OutputModule {
  public:
    friend class RootOutputFile;
    explicit PoolOutputModule(ParameterSet const& ps);
    virtual ~PoolOutputModule();
    std::string const& fileName() const {return fileName_;}
    std::string const& logicalFileName() const {return logicalFileName_;}
    int const& compressionLevel() const {return compressionLevel_;}
    int const& basketSize() const {return basketSize_;}
    int const& splitLevel() const {return splitLevel_;}
    int const& treeMaxVirtualSize() const {return treeMaxVirtualSize_;}
    bool const& fastCloning() const {return fastCloning_;}
    bool const& fastMetaCloning() const {return fastMetaCloning_;}

  private:
    virtual void openFile(FileBlock const& fb);
    virtual void respondToOpenInputFile(FileBlock const& fb);
    virtual void respondToCloseInputFile(FileBlock const& fb);
    virtual void write(EventPrincipal const& e);
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const& lb);
    virtual void writeRun(RunPrincipal const& r);

    virtual bool isFileOpen() const;
    virtual bool isFileFull() const;
    virtual void doOpenFile();


    virtual void startEndFile();
    virtual void writeFileFormatVersion();
    virtual void writeFileIdentifier();
    virtual void writeFileIndex();
    virtual void writeEventHistory();
    virtual void writeProcessConfigurationRegistry();
    virtual void writeProcessHistoryRegistry();
    virtual void writeModuleDescriptionRegistry();
    virtual void writeParameterSetRegistry();
    virtual void writeProductDescriptionRegistry();
    virtual void writeEntryDescriptions();
    virtual void finishEndFile();

    std::string const fileName_;
    std::string const logicalFileName_;
    std::string const catalog_;
    unsigned int const maxFileSize_;
    int const compressionLevel_;
    int const basketSize_;
    int const splitLevel_;
    int const treeMaxVirtualSize_;
    bool fastCloning_;
    bool fastMetaCloning_;
    FileBlock *fileBlock_;
    std::string const moduleLabel_;
    int fileCount_;
    boost::scoped_ptr<RootOutputFile> rootOutputFile_;
  };
}

#endif
