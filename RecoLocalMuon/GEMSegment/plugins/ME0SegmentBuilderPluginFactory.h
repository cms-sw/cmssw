#ifndef GEMRecHit_ME0SegmentBuilderPluginFactory_h
#define GEMRecHit_ME0SegmentBuilderPluginFactory_h

/** \class ME0SegmentBuilderPluginFactory derived from CSC
 *  Plugin factory for concrete ME0SegmentBuilder algorithms
 *
 * \author Marcello Maggi
 * 
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLocalMuon/GEMSegment/plugins/ME0SegmentAlgorithmBase.h"

typedef edmplugin::PluginFactory<ME0SegmentAlgorithmBase *(const edm::ParameterSet &)> ME0SegmentBuilderPluginFactory;

#endif
