#ifndef __LUMIFILEWRITER_H__
#define __LUMIFILEWRITER_H__

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>

namespace HCAL_HLX{
  class TCPReceiver;
  class ROOTSchema;
  struct LUMI_SECTION;
}

class LumiFileWriter : public edm::EDAnalyzer {
public:
  explicit LumiFileWriter(const edm::ParameterSet&);
  ~LumiFileWriter();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  std::string DistribIP1;
  std::string DistribIP2;
  
  HCAL_HLX::TCPReceiver*      HLXTCP_;  
  HCAL_HLX::LUMI_SECTION*     lumiSection_;
  HCAL_HLX::ROOTSchema*       LumiSchema_;
  
  unsigned int reconTime;
  
  bool bMerge_;
  bool bWBM_;
  bool bTransfer_;
  bool bTest_;

};

#endif
