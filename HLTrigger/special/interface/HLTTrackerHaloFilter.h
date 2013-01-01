#ifndef HLTTrackerHaloFilter_h
#define HLTTrackerHaloFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

///////////////////////////////////////////////////////
//
// HLTrackerHaloFilter
//
// Filter selecting beam halo track candidates by looking at 
// TEC clusters accumulations
//
// This filter is working with events seeded by L1_BeamHalo
// (BPTX_Xor && (36 || 37 || 38 || 39))
//
// More details:
// http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.MIB
//
// S.Viret: 27/01/2011 (viret@in2p3.fr)
//
///////////////////////////////////////////////////////


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/RefGetter.h"




class HLTTrackerHaloFilter : public HLTFilter {
public:
  explicit HLTTrackerHaloFilter(const edm::ParameterSet&);
  ~HLTTrackerHaloFilter();

private:
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

  edm::InputTag inputTag_; // input tag identifying product containing pixel clusters
  int max_clusTp_; // Maximum number of TEC+ clusters
  int max_clusTm_; // Maximum number of TEC- clusters
  int sign_accu_;  // Minimal size for a signal accumulation
  int max_clusT_;  // Maximum number of TEC clusters
  int max_back_;   // Max number of accumulations per side
  int fastproc_;   // fast unpacking of cluster info, based on DetIds 
 
  int SST_clus_MAP_m[5][8][9];
  int SST_clus_MAP_p[5][8][9];
  int SST_clus_PROJ_m[5][8];
  int SST_clus_PROJ_p[5][8];

  static const int m_TEC_cells[];


};

#endif
