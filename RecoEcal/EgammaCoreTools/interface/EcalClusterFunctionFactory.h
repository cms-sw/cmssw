#ifndef RecoEcal_EgammaCoreTools_hh
#define RecoEcal_EgammaCoreTools_hh

/** \class EcalClusterFunctionFactory
  *  Manage all the cluster functions
  *
  *  $Id: EcalClusterFunctionFactory.h
  *  $Date:
  *  $Revision:
  *  \author Federico Ferri, CEA Saclay, November 2008
  */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
typedef edmplugin::PluginFactory< EcalClusterFunctionBaseClass*(const edm::ParameterSet&) > EcalClusterFunctionFactory;

#endif
