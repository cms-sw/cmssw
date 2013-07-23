#ifndef Fireworks_Electrons_makeSuperCluster_h
#define Fireworks_Electrons_makeSuperCluster_h
// -*- C++ -*-
//
// Package:     Electrons
// Class  :     makeSuperCluster
//
/**\class makeSuperCluster makeSuperCluster.h Fireworks/Electrons/interface/makeSuperCluster.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Dec  5 15:32:20 EST 2008
// $Id: makeSuperCluster.h,v 1.4 2010/04/20 20:49:42 amraktad Exp $
//

// system include files
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

// user include files

// forward declarations
class FWEventItem;
class TEveElement;
class FWProxyBuilderBase;

namespace fireworks {
   bool makeRhoPhiSuperCluster(FWProxyBuilderBase*,
                               const reco::SuperClusterRef& iCluster,
                               float iPhi,
                               TEveElement& oItemHolder);
   bool makeRhoZSuperCluster(FWProxyBuilderBase*,
                             const reco::SuperClusterRef& iCluster,
                             float iPhi,
                             TEveElement& oItemHolder);
}
#endif
