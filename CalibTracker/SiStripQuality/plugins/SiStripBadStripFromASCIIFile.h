#ifndef SiStripBadStripFromASCIIFile_H
#define SiStripBadStripFromASCIIFile_H


#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"


class SiStripBadStripFromASCIIFile:public ConditionDBWriter<SiStripBadStrip> {

 public:

  explicit SiStripBadStripFromASCIIFile( const edm::ParameterSet& iConfig);

  ~SiStripBadStripFromASCIIFile(){};



 private:
  virtual SiStripBadStrip * getNewObject();
  edm::FileInPath fp_;
  bool printdebug_;
};

#endif
