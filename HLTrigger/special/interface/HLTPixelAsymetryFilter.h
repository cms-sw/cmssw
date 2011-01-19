#ifndef HLTPixelAsymetryFilter_h
#define HLTPixelAsymetryFilter_h


///////////////////////////////////////////////////////
//
// HLTPixelAsymetryFilter
//
// Filter definition
//
// We perform a selection on PIXEL cluster repartition 
//
// This filter is primarily used to select Beamgas (aka PKAM) events
// 
// An asymetry parameter, based on the pixel clusters, is computed as follows
// 
//  asym1 = fpix-/(fpix- + fpix+) for beam1
//  asym2 = fpix+/(fpix- + fpix+) for beam2 
//
// with:
//
//  fpix- = mean cluster charge in FPIX-
//  fpix+ = mean cluster charge in FPIX+
//  bpix  = mean cluster charge in BarrelPIX
//
//  Usually for PKAM events, cluster repartition is quite uniform and asymetry is around 0.5 
//
//
// More details:
// http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.MIB
//
// S.Viret: 12/01/2011 (viret@in2p3.fr)
//
///////////////////////////////////////////////////////

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

class HLTPixelAsymetryFilter : public HLTFilter {
 public:
  explicit HLTPixelAsymetryFilter(const edm::ParameterSet&);
  ~HLTPixelAsymetryFilter();

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);

  edm::InputTag inputTag_; // input tag identifying product containing pixel clusters
  bool          saveTag_;  // whether to save this tag
  double  min_asym_;       // minimum asymetry 
  double  max_asym_;       // maximum asymetry
  double  clus_thresh_;    // minimum charge for a cluster to be selected (in e-)
  double  bmincharge_;     // minimum average charge in the barrel (bpix, in e-)

};

#endif

