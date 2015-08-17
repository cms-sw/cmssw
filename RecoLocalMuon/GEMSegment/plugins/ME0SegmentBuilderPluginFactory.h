#ifndef GEMRecHit_ME0SegmentBuilderPluginFactory_h
#define GEMRecHit_ME0SegmentBuilderPluginFactory_h

/** \class ME0SegmentBuilderPluginFactory derived from CSC
 *  Plugin factory for concrete ME0SegmentBuilder algorithms
 *
 * \author Marcello Maggi
 * 
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentAlgorithm.h>

typedef edmplugin::PluginFactory<ME0SegmentAlgorithm *(const edm::ParameterSet&)> ME0SegmentBuilderPluginFactory;

#endif
