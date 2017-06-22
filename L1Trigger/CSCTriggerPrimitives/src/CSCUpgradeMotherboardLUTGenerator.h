#ifndef L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUTGenerator_h
#define L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUTGenerator_h

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include <vector>

namespace
{
  // function only makes sense for endcap!
  RPCDetId getRPCfromCSC(const CSCDetId& csc_id)
  {
    const int region(csc_id.zendcap());
    const int csc_trig_sect(CSCTriggerNumbering::triggerSectorFromLabels(csc_id));
    const int csc_trig_id( CSCTriggerNumbering::triggerCscIdFromLabels(csc_id));
    const int csc_trig_chid((3*(csc_trig_sect-1)+csc_trig_id)%18 +1);
    const int rpc_trig_sect((csc_trig_chid-1)/3+1);
    const int rpc_trig_subsect((csc_trig_chid-1)%3+1);
    return RPCDetId(region,1,csc_id.station(),rpc_trig_sect,1,rpc_trig_subsect,0);
  }
}

class CSCUpgradeMotherboardLUTGenerator
{
public:

  CSCUpgradeMotherboardLUTGenerator() {}
  ~CSCUpgradeMotherboardLUTGenerator() {}
  
  /// set CSC and GEM geometries for the matching needs
  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }
  void setRPCGeometry(const RPCGeometry *g) { rpc_g = g; }
  
  /// generate and print LUT
  void generateLUTs(unsigned e, unsigned s, unsigned se, unsigned sb, unsigned c) const;
  void generateLUTsME11(unsigned e, unsigned se, unsigned sb, unsigned c) const;
  void generateLUTsME21(unsigned e, unsigned se, unsigned sb, unsigned c) const;
  void generateLUTsME3141(unsigned e, unsigned s, unsigned se, unsigned sb, unsigned c) const;
  int assignRoll(const std::vector<std::pair<double,double> >&, double eta) const;
  
 private:
  // create LUT: roll->(etaMin,etaMax)
  void gemRollToEtaLimitsLUT(const GEMChamber* c, std::vector<std::pair<double,double> >& ) const;
  
  // create LUT: roll->(etaMin,etaMax)
  void rpcRollToEtaLimitsLUT(const RPCChamber* c, std::vector<std::pair<double,double> >&) const;

  // create LUT: WG->(rollMin,rollMax)
  void cscWgToRollLUT(const std::vector<std::pair<double,double> >&,
		 const std::vector<std::pair<double,double> >&,
		 std::vector<std::pair<int,int> >&) const;
  
  // create LUT: WG->(etaMin,etaMax)
  void cscWgToEtaLimitsLUT(const CSCLayer*, std::vector<std::pair<double,double> >&) const;
  
  // create LUT: HS->pad
  void cscHsToGemPadLUT(const CSCLayer*, const GEMEtaPartition*, int minH, int maxH, std::vector<std::pair<int,int> >&) const;

  // create LUT: pad->HS
  void gemPadToCscHsLUT(const CSCLayer*, const GEMEtaPartition*, std::vector<int>&) const;

  // create LUT: HS->strip
  void cscHsToRpcStripLUT(const CSCLayer*, const RPCRoll*, int minH, int maxH, std::vector<std::pair<int,int> >&) const;

  // create LUT: strip->HS
  void rpcStripToCscHsLUT(const CSCLayer*, const RPCRoll*, std::vector<int>&) const;

  const CSCGeometry* csc_g;
  const GEMGeometry* gem_g;
  const RPCGeometry* rpc_g;
};

template<typename T>
std::ostream &operator <<(std::ostream &os, const std::vector<std::pair<T,T> >&v) 
{
  int i = 0;
  os << "{" << std::endl;
  for(const auto& p : v) {
    os << " {" << p.first << ", " << p.second << "}, ";
    if (i%8==0) os << std::endl;
    i++;
  }
  os << "}" << std::endl;
  os << std::endl;

  return os;
}

template<typename T>
std::ostream &operator <<(std::ostream &os, const std::vector<T>&v) 
{
  int i = 0;
  os << "{" << std::endl;
  for(const auto& p : v) {
    os << " " << p << ",";
    if (i%10==0) os << std::endl;
    i++;
  }
  os << "}" << std::endl;
  os << std::endl;

  return os;
}

#endif
