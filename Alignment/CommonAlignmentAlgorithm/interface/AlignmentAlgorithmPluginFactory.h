#ifndef Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmPluginFactory_h
#define Alignment_CommonAlignmentAlgorithm_AlignmentAlgorithmPluginFactory_h

/// \class AlignmentAlgorithmPluginFactory
///  Plugin factory for alignment algorithm
///
///  \author F. Ronga - CERN
///

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

typedef edmplugin::PluginFactory<AlignmentAlgorithmBase*(const edm::ParameterSet&, edm::ConsumesCollector&)>
    AlignmentAlgorithmPluginFactory;

#endif
