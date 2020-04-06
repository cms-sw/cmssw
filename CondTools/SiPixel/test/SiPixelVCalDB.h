#ifndef CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#define CalibTracker_SiPixelVCalDB_SiPixelVCalDB_h
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "CondTools/SiPixel/test/SiPixelVCalDB.h"

class SiPixelVCalDB : public edm::EDAnalyzer {
public:

  // "PixelID"
  // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
  // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
  enum PixelID {
    L1=1100, L2=1200, L3=1300, L4=1400, // BPix
    Rm1l=2111, Rm1u=2112, Rm2l=2121, Rm2u=2122, Rm3l=2131, Rm3u=2132, // FPix minus
    Rp1l=2211, Rp1u=2212, Rp2l=2221, Rp2u=2222, Rp3l=2231, Rp3u=2232, // FPix plus
  };

  explicit SiPixelVCalDB(const edm::ParameterSet& conf);
  virtual ~SiPixelVCalDB();
  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  //const PixelID detIdToPixelID(const unsigned int detid); //const uint32_t&
  static const PixelID calculateBPixID(const unsigned int layer);
  static const PixelID calculateFPixID(const unsigned int side, const unsigned int disk, const unsigned int ring);
  static const int getPixelSubDetector(const unsigned int pixid);
  
private:
  
  std::vector<std::pair<uint32_t, float> > m_slope;  // detId -> slope
  std::vector<std::pair<uint32_t, float> > m_offset; // detId -> offset
  edm::ParameterSet conf_;
  std::string recordName_;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BPixParameters_;
  Parameters FPixParameters_;
  
};

#endif
