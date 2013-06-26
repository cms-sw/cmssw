//
// $Id: GenJetSelector.h,v 1.3 2009/03/27 22:30:27 hegner Exp $
//
 
#ifndef PhysicsTools_JetMCAlgos_GenJetSelector_h
#define PhysicsTools_JetMCAlgos_GenJetSelector_h
 
#include "DataFormats/Common/interface/RefVector.h"
 
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
 
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

typedef SingleObjectSelector<
            reco::GenJetCollection,
            StringCutObjectSelector<reco::GenJet>
        > GenJetSelector;

#endif
