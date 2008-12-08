#ifndef CalibTracker_SiStripESProducers_SiStripBadModuleGenerator_H
#define CalibTracker_SiStripESProducers_SiStripBadModuleGenerator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondTools/SiStrip/interface/SiStripCondObjBuilderBase.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include <string>

class SiStripBadModuleGenerator : public SiStripCondObjBuilderBase<SiStripBadStrip> {
 public:

  explicit SiStripBadModuleGenerator(const edm::ParameterSet&,const edm::ActivityRegistry&);
  ~SiStripBadModuleGenerator();
  
  void getObj(SiStripBadStrip* & obj){createObject(); obj=obj_;}

 private:
  
  void createObject();

  void selectDetectors(const std::vector<uint32_t>& , std::vector<uint32_t>& );

  bool isTIBDetector(const uint32_t & therawid,
		     uint32_t requested_layer,
		     uint32_t requested_bkw_frw,
		     uint32_t requested_int_ext,
		     uint32_t requested_string,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;
  
  bool isTOBDetector(const uint32_t & therawid,
		     uint32_t requested_layer,
		     uint32_t requested_bkw_frw,
		     uint32_t requested_rod,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;

  bool isTIDDetector(const uint32_t & therawid,
		     uint32_t requested_side,
		     uint32_t requested_wheel,
		     uint32_t requested_ring,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;

  bool isTECDetector(const uint32_t & therawid,
		     uint32_t requested_side,
		     uint32_t requested_wheel,
		     uint32_t requested_petal_bkw_frw,
		     uint32_t requested_petal,			
		     uint32_t requested_ring,
		     uint32_t requested_ster,
		     uint32_t requested_detid) const;

  bool printdebug_;
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters BadComponentList_;          
  
};

#endif 
