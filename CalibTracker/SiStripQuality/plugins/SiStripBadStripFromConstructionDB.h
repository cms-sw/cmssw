#ifndef SiStripBadStripFromConstructionDB_H
#define SiStripBadStripFromConstructionDB_H


#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"



class SiStripBadStripFromConstructionDB:public ConditionDBWriter<SiStripBadStrip> {

 public:

  explicit SiStripBadStripFromConstructionDB( const edm::ParameterSet& iConfig);

  ~SiStripBadStripFromConstructionDB(){};


 protected:
  // Leave possibility of inheritance
  virtual void algoBeginJob(const edm::EventSetup&);
  virtual void algoAnalyze(const edm::Event& , const edm::EventSetup& );

 private:
  virtual SiStripBadStrip * getNewObject();
  

  bool printdebug_;
  std::vector<uint32_t> ext_bad_detids;
  std::vector<short> badstrips;
  std::vector< std::pair<uint32_t, unsigned short> > detid_strips;
  std::vector< std::pair<uint32_t, std::vector<short> > > constdb_strips;

};

#endif
