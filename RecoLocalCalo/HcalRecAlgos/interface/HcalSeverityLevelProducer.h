#ifndef HCALSEVERITYLEVELPRODUCER_H
#define HCALSEVERITYLEVELPRODUCER_H

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"


class HcalSeverityLevelProducer
{

 public:
  HcalSeverityLevelProducer( const edm::ParameterSet& );
  ~HcalSeverityLevelProducer();
  
  // gives back severity level based on evaluation of the RecHit flag and cell's channel status
  int getSeverityLevel(const DetId myid, const uint32_t myflag, const uint32_t mystatus);
  
 private:
  class HcalSeverityDefinition
  {
  public:
    int sevLevel;
    uint32_t chStatusMask;
    uint32_t HBHEFlagMask, HOFlagMask, HFFlagMask, ZDCFlagMask, CalibFlagMask;
  HcalSeverityDefinition():
    sevLevel(0), chStatusMask(0),
      HBHEFlagMask(0), HOFlagMask(0), HFFlagMask(0), ZDCFlagMask(0), CalibFlagMask(0)
      {}
    
  };

  std::vector<HcalSeverityDefinition> SevDef;

  void setBit (const unsigned bitnumber, uint32_t& where);

  friend std::ostream& operator<<(std::ostream& s, const HcalSeverityLevelProducer::HcalSeverityDefinition& def);


};


#endif
