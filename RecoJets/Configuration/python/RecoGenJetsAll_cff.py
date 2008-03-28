import FWCore.ParameterSet.Config as cms

# $Id: RecoGenJetsAll.cff,v 1.1 2007/08/02 21:47:55 fedor Exp $
#
# ShR 27 Mar 07: move modules producing candidates for Jets into separate cff file due to scheduling problem
#
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.JetProducers.kt6GenJets_cff import *
from RecoJets.JetProducers.kt10GenJets_cff import *
from RecoJets.JetProducers.iterativeCone5GenJets_cff import *
from RecoJets.JetProducers.iterativeCone7GenJets_cff import *
from RecoJets.JetProducers.midPointCone5GenJets_cff import *
from RecoJets.JetProducers.midPointCone7GenJets_cff import *
from RecoJets.JetProducers.sisCone5GenJets_cff import *
from RecoJets.JetProducers.sisCone7GenJets_cff import *
from RecoJets.JetProducers.cdfMidpointCone5GenJets_cff import *
from RecoJets.JetProducers.kt6GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.kt10GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.iterativeCone5GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.iterativeCone7GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.midPointCone5GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.midPointCone7GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.sisCone5GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.sisCone7GenJetsNoNuBSM_cff import *
from RecoJets.JetProducers.cdfMidpointCone5GenJetsNoNuBSM_cff import *
jetsWithAllAll = cms.Sequence(kt6GenJets+kt10GenJets+iterativeCone5GenJets+iterativeCone7GenJets+midPointCone5GenJets+midPointCone7GenJets+sisCone5GenJets+sisCone7GenJets+cdfMidpointCone5GenJets)
jetsNoNuBSMAll = cms.Sequence(kt6GenJetsNoNuBSM+kt10GenJetsNoNuBSM+iterativeCone5GenJetsNoNuBSM+iterativeCone7GenJetsNoNuBSM+midPointCone5GenJetsNoNuBSM+midPointCone7GenJetsNoNuBSM+sisCone5GenJetsNoNuBSM+sisCone7GenJetsNoNuBSM+cdfMidpointCone5GenJetsNoNuBSM)
jetsWithAllPt10All = cms.Sequence(kt6GenJetsPt10Seq+iterativeCone5GenJetsPt10Seq+iterativeCone7GenJetsPt10Seq+midPointCone5GenJetsPt10Seq+midPointCone7GenJetsPt10Seq+sisCone5GenJetsPt10Seq+sisCone7GenJetsPt10Seq+cdfMidpointCone5GenJetsPt10Seq)
# ShR 27 Mar 2007
# NB: genJetParticles sequence MUST be executed before this one but we cannot add it here
# since causes scheduling conflicts with modules from MET
recoGenJetsAll = cms.Sequence(jetsWithAllAll+jetsNoNuBSMAll+jetsWithAllPt10All)

