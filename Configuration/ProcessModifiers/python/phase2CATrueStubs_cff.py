import FWCore.ParameterSet.Config as cms

# Overlay on phase2CAStubs, swapping in TrueStubProducer for studies with truth-matched stubs:
# --procModifiers phase2CAStubs,phase2CATrueStubs
phase2CATrueStubs = cms.Modifier()
