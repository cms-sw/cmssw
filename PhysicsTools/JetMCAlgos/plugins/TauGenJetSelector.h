//
// $Id: TauGenJetSelector.h,v 1.3 2008/06/19 13:22:12 veelken Exp $
//
 
#ifndef PhysicsTools_JetMCAlgos_TauGenJetSelector_h
#define PhysicsTools_JetMCAlgos_TauGenJetSelector_h
 
#include "DataFormats/Common/interface/RefVector.h"
 
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
 
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

typedef SingleObjectSelector<
            reco::GenJetCollection,
            StringCutObjectSelector<reco::GenJet>
        > TauGenJetSelector;

typedef SingleObjectSelector<
               reco::GenJetCollection,
               StringCutObjectSelector<reco::GenJet>,
               edm::RefVector<reco::GenJetCollection>
        > TauGenJetRefSelector;

#endif
