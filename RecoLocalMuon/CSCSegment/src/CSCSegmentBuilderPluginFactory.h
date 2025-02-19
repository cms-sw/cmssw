#ifndef CSCSegment_CSCSegmentBuilderPluginFactory_h
#define CSCSegment_CSCSegmentBuilderPluginFactory_h

/** \class CSCSegmentBuilderPluginFactory
 *  Plugin factory for concrete CSCSegmentBuilder algorithms
 *
 * $Date: 2009/12/16 02:01:11 $
 * $Revision: 1.6 $
 * \author M. Sani
 * 
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>

typedef edmplugin::PluginFactory<CSCSegmentAlgorithm *(const edm::ParameterSet&)> CSCSegmentBuilderPluginFactory;

#endif
