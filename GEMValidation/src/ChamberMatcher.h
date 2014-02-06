#ifndef GEMValidation_ChamberMatcher_h
#define GEMValidation_ChamberMatcher_h

/**\class ChamberMatcher

 Description: Matching of SimTrack to GEM,CSC,RPC and ME0 chambers

 Original Author:  "Sven Dildick"
*/

#include "GEMCode/GEMValidation/src/BaseMatcher.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <vector>
#include <map>
#include <set>

class ChamberMatcher : public BaseMatcher
{
public:
  
  ChamberMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es);
  
  ~ChamberMatcher();

  /// GEM partitions' detIds 
  std::set<unsigned int> detIdsGEM(int gem_type = GEM_ME11 ) const;
  /// RPC partitions' detIds 
  std::set<unsigned int> detIdsRPC(int rpc_type = RPC_RE12) const;
  /// ME0 partitions' detIds 
  std::set<unsigned int> detIdsME0() const;
  /// CSC layers' detIds 
  std::set<unsigned int> detIdsCSC(int csc_type = CSC_ME1b) const;

private:

  void init();

  std::string simInputLabel_;

  const CSCGeometry* csc_geo_;
  const GEMGeometry* gem_geo_;

  bool verboseGEM_;
  bool verboseCSC_;
  bool verboseRPC_;
  bool verboseME0_;
};

#endif

