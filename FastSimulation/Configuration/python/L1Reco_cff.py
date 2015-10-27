import FWCore.ParameterSet.Config as cms

# the only thing FastSim runs from L1Reco is l1extraParticles
from L1Trigger.Configuration.L1Reco_cff import l1extraParticles

# If the Stage 1 trigger is running, there is also some different configuration.
# Note that this next file does nothing if the stage1L1Trigger era is not active, so
# it is safe to import even if the Stage 1 trigger is not required. It *MUST* be
# imported into this namespace, i.e. "from <module> import *".
from L1Trigger.Configuration.ConditionalStage1Configuration_cff import *

# some collections have different labels
def _changeLabelForFastSim( object ) :
    """
    Takes an InputTag, changes the first letter of the module label to a capital
    and adds "sim" in front, e.g. "gctDigid" -> "simGctDigis".
    This works for both Run 1 and Run 2 collections.
    """
    object.moduleLabel="sim"+object.moduleLabel[0].upper()+object.moduleLabel[1:]

_changeLabelForFastSim( l1extraParticles.isolatedEmSource )
_changeLabelForFastSim( l1extraParticles.nonIsolatedEmSource )

_changeLabelForFastSim( l1extraParticles.centralJetSource )
_changeLabelForFastSim( l1extraParticles.tauJetSource )
_changeLabelForFastSim( l1extraParticles.isoTauJetSource )
_changeLabelForFastSim( l1extraParticles.forwardJetSource )

_changeLabelForFastSim( l1extraParticles.etTotalSource )
_changeLabelForFastSim( l1extraParticles.etHadSource )
_changeLabelForFastSim( l1extraParticles.htMissSource )
_changeLabelForFastSim( l1extraParticles.etMissSource )

_changeLabelForFastSim( l1extraParticles.hfRingEtSumsSource )
_changeLabelForFastSim( l1extraParticles.hfRingBitCountsSource )

# This one is subtly different, but is the same for Run 1 and Run 2 FastSim
l1extraParticles.muonSource = cms.InputTag('simGmtDigis')


# must be set to true when used in HLT, as is the case for FastSim
l1extraParticles.centralBxOnly = True

L1Reco = cms.Sequence(l1extraParticles)
