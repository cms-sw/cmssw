#ifndef CSCSegment_CSCSegmentBuilderPluginFactory_h
#define CSCSegment_CSCSegmentBuilderPluginFactory_h

/** \class CSCSegmentBuilderPluginFactory
 *  Plugin factory for concrete CSCSegmentBuilder algorithms
 *
 * $Date: 2007/04/18 23:32:56 $
 * $Revision: 1.5 $
 * \author M. Sani
 * 
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>

typedef edmplugin::PluginFactory<CSCSegmentAlgorithm *(const edm::ParameterSet&)> CSCSegmentBuilderPluginFactory;

#endif
