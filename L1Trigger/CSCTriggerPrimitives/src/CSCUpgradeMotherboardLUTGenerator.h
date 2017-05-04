#ifndef L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUTGenerator_h
#define L1Trigger_CSCTriggerPrimitives_CSCUpgradeMotherboardLUTGenerator_h

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include <vector>

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
  std::vector<std::pair<double,double> > 
  gemRollToEtaLimitsLUT(const GEMChamber* c) const;

  // create LUT: roll->(etaMin,etaMax)
  std::vector<std::pair<double,double> > 
  rpcRollToEtaLimitsLUT(const RPCChamber* c) const;

  // create LUT: WG->(rollMin,rollMax)
  std::vector<std::pair<int,int> > 
  cscWgToRollLUT(const std::vector<std::pair<double,double> >&,
		 const std::vector<std::pair<double,double> >&) const;
  
  // create LUT: WG->(etaMin,etaMax)
  std::vector<std::pair<double,double> > 
  cscWgToEtaLimitsLUT(const CSCLayer*) const;
  
  // create LUT: HS->pad
  std::vector<std::pair<int,int> >
  cscHsToGemPadLUT(const CSCLayer*, const GEMEtaPartition*, int minH, int maxH) const;

  // create LUT: pad->HS
  std::vector<int>
  gemPadToCscHsLUT(const CSCLayer*, const GEMEtaPartition*) const;

  // create LUT: HS->strip
  std::vector<std::pair<int,int> >
  cscHsToRpcStripLUT(const CSCLayer*, const RPCRoll*, int minH, int maxH) const;

  // create LUT: strip->HS
  std::vector<int>
  rpcStripToCscHsLUT(const CSCLayer*, const RPCRoll*) const;

  const CSCGeometry* csc_g;
  const GEMGeometry* gem_g;
  const RPCGeometry* rpc_g;
};

template<typename T>
std::ostream &operator <<(std::ostream &os, const std::vector<std::pair<T,T> >&v) 
{
  int i = 0;
  os << "{" << std::endl;
  for(auto p : v) {
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
  for(auto p : v) {
    os << " " << p << ",";
    if (i%10==0) os << std::endl;
    i++;
  }
  os << "}" << std::endl;
  os << std::endl;

  return os;
}

#endif
