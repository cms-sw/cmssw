#ifndef GEMRecHit_GEMSegmentBuilderPluginFactory_h
#define GEMRecHit_GEMSegmentBuilderPluginFactory_h

/** \class GEMSegmentBuilderPluginFactory derived from CSC
 *  Plugin factory for concrete GEMSegmentBuilder algorithms
 *
 * \author Piet Verwilligen
 * 
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithm.h>

typedef edmplugin::PluginFactory<GEMSegmentAlgorithm *(const edm::ParameterSet&)> GEMSegmentBuilderPluginFactory;

#endif
