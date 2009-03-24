//
// $Id: TauGenJetSelector.h,v 1.1 2009/01/21 15:31:19 veelken Exp $
//
 
#ifndef PhysicsTools_JetMCAlgos_TauGenJetSelector_h
#define PhysicsTools_JetMCAlgos_TauGenJetSelector_h
 
#include "DataFormats/Common/interface/RefVector.h"
 
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
 
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
