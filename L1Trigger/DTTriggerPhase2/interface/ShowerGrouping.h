#ifndef Phase2L1Trigger_DTTrigger_ShowerGrouping_h
#define Phase2L1Trigger_DTTrigger_ShowerGrouping_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuffer.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

// ===============================================================================
// Class declarations
// ===============================================================================

bool hitWireSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2)
{
  int wi1 = hit1.channelId();
  int wi2 = hit2.channelId();

  if (wi1 < wi2) return true;
  else return false;
}

bool hitLayerSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2)
{
  int lay1 = hit1.layerId();
  int lay2 = hit2.layerId();

  if (lay1 < lay2) return true;
  else if (lay1 > lay2) return false;
  else return hitWireSort_shower(hit1, hit2);
}

bool hitTimeSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2)
{
  int tdc1 = hit1.tdcTimeStamp();
  int tdc2 = hit2.tdcTimeStamp();

  if (tdc1 < tdc2) return true;
  else if (tdc1 > tdc2) return false;
  else return hitLayerSort_shower(hit1, hit2);
}

class ShowerGrouping {

public:
  // Constructors and destructor
  ShowerGrouping(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  virtual ~ShowerGrouping();

  // Main methods
  virtual void initialise(const edm::EventSetup& iEventSetup);
  virtual void run(edm::Event& iEvent,
                   const edm::EventSetup& iEventSetup,
                   const DTDigiCollection& digis,
                   ShowerBufferPtr &showerBuffer);
  virtual void finish();

  // Other public methods

  // Public attributes

private:
  // Private methods
  void setInChannels(const DTDigiCollection* digi);
  void clearChannels();
  void sortHits();
  bool hitWireSort(const DTPrimitive& hit1, const DTPrimitive& hit2);
  bool hitLayerSort(const DTPrimitive& hit1, const DTPrimitive& hit2);
  bool hitTimeSort(const DTPrimitive& hit1, const DTPrimitive& hit2);
  bool triggerShower(const ShowerBufferPtr& showerBuf);

  // Private attributes
  const bool debug_;

  const int nHits_per_bx;
  const int threshold_for_shower;
  DTPrimitives channelIn_[cmsdt::NUM_LAYERS_2SL][cmsdt::NUM_CH_PER_LAYER];
  DTPrimitives chInDummy_;
  DTPrimitives all_hits;
  std::map<int, DTPrimitives> all_hits_perBx;
  int showerTaggingAlgo_;
  int currentBaseChannel_;

};

#endif
