//
// $Id: GenJetSelector.h,v 1.1 2009/01/21 15:31:19 veelken Exp $
//
 
#ifndef PhysicsTools_JetMCAlgos_GenJetSelector_h
#define PhysicsTools_JetMCAlgos_GenJetSelector_h
 
#include "DataFormats/Common/interface/RefVector.h"
 
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
 
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

typedef SingleObjectSelector<
            reco::GenJetCollection,
            StringCutObjectSelector<reco::GenJet>
        > GenJetSelector;

typedef SingleObjectSelector<
               reco::GenJetCollection,
               StringCutObjectSelector<reco::GenJet>,
               edm::RefVector<reco::GenJetCollection>
        > GenJetRefSelector;

#endif
