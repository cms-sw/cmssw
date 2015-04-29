#ifndef GEMRecHit_ME0SegmentBuilderPluginFactory_h
#define GEMRecHit_ME0SegmentBuilderPluginFactory_h

/** \class ME0SegmentBuilderPluginFactory derived from CSC
 *  Plugin factory for concrete ME0SegmentBuilder algorithms
 *
 * $Date: 2014/02/04 12:01:11 $
 * $Revision: 1.1 $
 * \author Marcello Maggi
 * 
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <RecoLocalMuon/GEMRecHit/src/ME0SegmentAlgorithm.h>

typedef edmplugin::PluginFactory<ME0SegmentAlgorithm *(const edm::ParameterSet&)> ME0SegmentBuilderPluginFactory;

#endif
