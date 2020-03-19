#ifndef GEMRecHit_GEMSegmentBuilderPluginFactory_h
#define GEMRecHit_GEMSegmentBuilderPluginFactory_h

/** \class GEMSegmentBuilderPluginFactory derived from CSC
 *  Plugin factory for concrete GEMSegmentBuilder algorithms
 *
 * \author Piet Verwilligen
 * 
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithmBase.h"

typedef edmplugin::PluginFactory<GEMSegmentAlgorithmBase *(const edm::ParameterSet &)> GEMSegmentBuilderPluginFactory;

#endif
