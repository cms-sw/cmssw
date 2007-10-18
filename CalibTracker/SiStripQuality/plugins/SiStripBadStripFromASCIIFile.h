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


 protected:
  // Leave possibility of inheritance
  virtual void algoBeginJob(const edm::EventSetup&);

 private:
  virtual SiStripBadStrip * getNewObject();

  typedef std::pair<short,short> p_channelflag;
  typedef std::vector<p_channelflag> v_channelflag;
  typedef std::pair<uint32_t, v_channelflag> p_detidchannelflag;
  typedef std::vector<p_detidchannelflag> v_detidallbadstrips;

  
  edm::FileInPath fp_;
  bool printdebug_;
  v_channelflag badstripsandflags;
  std::vector< std::pair<uint32_t, unsigned short> > detid_strips;
  std::vector< std::pair<uint32_t, v_channelflag> > constdb_strips;
  v_detidallbadstrips v_allbadstrips;




};

#endif
