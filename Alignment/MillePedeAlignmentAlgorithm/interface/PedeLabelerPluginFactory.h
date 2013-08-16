#ifndef Alignment_MillePedeAlignmentAlgorithm_PedeLabelerPluginFactory_h
#define Alignment_MillePedeAlignmentAlgorithm_PedeLabelerPluginFactory_h

/** \class PedeLabelerPluginFactory
 *
 * A plugin factory for pede labelers
 *
 *  Original author: Andreas Mussgiller, January 2011
 *
 *  $Date: 2010/09/10 10:26:20 $
 *  $Revision: 1.1 $
 *  (last update by $Author: mussgill $)
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"

typedef edmplugin::PluginFactory<PedeLabelerBase* (const PedeLabelerBase::TopLevelAlignables&,
						   const edm::ParameterSet&)> PedeLabelerPluginFactory;

#endif
