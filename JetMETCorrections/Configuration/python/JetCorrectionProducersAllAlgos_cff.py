import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
from JetMETCorrections.Configuration.JetCorrectionProducers_cff import *

# L1 Correction Producers
ak7CaloJetsL1 = ak5CaloJetsL1.clone( src = 'ak7CaloJets' )
kt4CaloJetsL1 = ak5CaloJetsL1.clone( src = 'kt4CaloJets' )
kt6CaloJetsL1 = ak5CaloJetsL1.clone( src = 'kt6CaloJets' )
ic5CaloJetsL1 = ak5CaloJetsL1.clone( src = 'ic5CaloJets' )

ak7PFJetsL1 = ak5PFJetsL1.clone( src = 'ak7PFJets' )
ak7PFCHSJetsL1 = ak5PFCHSJetsL1.clone( src = 'ak7PFCHSJets' )
kt4PFJetsL1 = ak5PFJetsL1.clone( src = 'kt4PFJets' )
kt6PFJetsL1 = ak5PFJetsL1.clone( src = 'kt6PFJets' )
ic5PFJetsL1 = ak5PFJetsL1.clone( src = 'ic5PFJets' )


# L2L3 Correction Producers
ak7CaloJetsL2 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL2Relative'])
kt4CaloJetsL2 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2Relative'])
kt6CaloJetsL2 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2Relative'])
ic5CaloJetsL2 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL2Relative'])

ak7PFJetsL2 = ak7PFJetsL1.clone(correctors = ['ak7PFL2Relative'])
ak7PFCHSJetsL2 = ak7PFCHSJetsL1.clone(correctors = ['ak7PFCHSL2Relative'])
kt4PFJetsL2 = kt4PFJetsL1.clone(correctors = ['kt4PFL2Relative'])
kt6PFJetsL2 = kt6PFJetsL1.clone(correctors = ['kt6PFL2Relative'])
ic5PFJetsL2 = ic5PFJetsL1.clone(correctors = ['ic5PFL2Relative'])

ak5JPTJetsL2 = ak5JPTJetsL1.clone(correctors = ['ak5JPTL2Relative'])
ak5TrackJetsL2 = ak5TrackJetsL1.clone(correctors = ['ak5TRKL2Relative'])

# L2L3 Correction Producers
ak7CaloJetsL2L3 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL2L3'])
kt4CaloJetsL2L3 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2L3'])
kt6CaloJetsL2L3 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2L3'])
ic5CaloJetsL2L3 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL2L3'])

ak7PFJetsL2L3 = ak7PFJetsL1.clone(correctors = ['ak7PFL2L3'])
ak7PFCHSJetsL2L3 = ak7PFCHSJetsL1.clone(correctors = ['ak7PFCHSL2L3'])
kt4PFJetsL2L3 = kt4PFJetsL1.clone(correctors = ['kt4PFL2L3'])
kt6PFJetsL2L3 = kt6PFJetsL1.clone(correctors = ['kt6PFL2L3'])
ic5PFJetsL2L3 = ic5PFJetsL1.clone(correctors = ['ic5PFL2L3'])

ak5JPTJetsL2L3 = ak5JPTJetsL1.clone(correctors = ['ak5JPTL2L3'])
ak5TrackJetsL2L3 = ak5TrackJetsL1.clone(correctors = ['ak5TRKL2L3'])

# L1L2L3 Correction Producers
ak7CaloJetsL1L2L3 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL1L2L3'])
kt4CaloJetsL1L2L3 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL1L2L3'])
kt6CaloJetsL1L2L3 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL1L2L3'])
ic5CaloJetsL1L2L3 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL1L2L3'])

ak7PFJetsL1L2L3 = ak7PFJetsL1.clone(correctors = ['ak7PFL1L2L3'])
ak7PFCHSJetsL1L2L3 = ak7PFCHSJetsL1.clone(correctors = ['ak7PFCHSL1L2L3'])
kt4PFJetsL1L2L3 = kt4PFJetsL1.clone(correctors = ['kt4PFL1L2L3'])
kt6PFJetsL1L2L3 = kt6PFJetsL1.clone(correctors = ['kt6PFL1L2L3'])
ic5PFJetsL1L2L3 = ic5PFJetsL1.clone(correctors = ['ic5PFL1L2L3'])

ak5JPTJetsL1L2L3 = ak5JPTJetsL1.clone(correctors = ['ak5JPTL1L2L3'])
ak5TrackJetsL1L2L3 = ak5TrackJetsL1.clone(correctors = ['ak5TRKL1L2L3'])

# L2L3L6 CORRECTION PRODUCERS
ak7CaloJetsL2L3L6 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL2L3L6'])
kt4CaloJetsL2L3L6 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2L3L6'])
kt6CaloJetsL2L3L6 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2L3L6'])
ic5CaloJetsL2L3L6 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL2L3L6'])

ak7PFJetsL2L3L6 = ak7PFJetsL1.clone(correctors = ['ak7PFL2L3L6'])
kt4PFJetsL2L3L6 = kt4PFJetsL1.clone(correctors = ['kt4PFL2L3L6'])
kt6PFJetsL2L3L6 = kt6PFJetsL1.clone(correctors = ['kt6PFL2L3L6'])
ic5PFJetsL2L3L6 = ic5PFJetsL1.clone(correctors = ['ic5PFL2L3L6'])


# L1L2L3L6 CORRECTION PRODUCERS
ak7CaloJetsL1L2L3L6 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL1L2L3L6'])
kt4CaloJetsL1L2L3L6 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL1L2L3L6'])
kt6CaloJetsL1L2L3L6 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL1L2L3L6'])
ic5CaloJetsL1L2L3L6 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL1L2L3L6'])

ak7PFJetsL1L2L3L6 = ak7PFJetsL1.clone(correctors = ['ak7PFL1L2L3L6'])
kt4PFJetsL1L2L3L6 = kt4PFJetsL1.clone(correctors = ['kt4PFL1L2L3L6'])
kt6PFJetsL1L2L3L6 = kt6PFJetsL1.clone(correctors = ['kt6PFL1L2L3L6'])
ic5PFJetsL1L2L3L6 = ic5PFJetsL1.clone(correctors = ['ic5PFL1L2L3L6'])
