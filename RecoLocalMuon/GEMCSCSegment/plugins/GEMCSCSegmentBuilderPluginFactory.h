#ifndef GEMRecHit_GEMCSCSegmentBuilderPluginFactory_h
#define GEMRecHit_GEMCSCSegmentBuilderPluginFactory_h

/** \class GEMCSCSegmentBuilderPluginFactory
 *  Plugin factory for concrete GEMCSCSegmentBuilder algorithms
 *
 * $Date:  $
 * $Revision: 1.6 $
 * \author Raffaella Radogna
 * 
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentAlgorithm.h>

typedef edmplugin::PluginFactory<GEMCSCSegmentAlgorithm *(const edm::ParameterSet&)> GEMCSCSegmentBuilderPluginFactory;

#endif
