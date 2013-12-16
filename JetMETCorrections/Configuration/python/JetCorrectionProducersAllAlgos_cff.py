import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
from JetMETCorrections.Configuration.JetCorrectionProducers_cff import *

# L1 Correction Producers
ak8CaloJetsL1 = ak4CaloJetsL1.clone( src = 'ak8CaloJets' )
kt4CaloJetsL1 = ak4CaloJetsL1.clone( src = 'kt4CaloJets' )
kt6CaloJetsL1 = ak4CaloJetsL1.clone( src = 'kt6CaloJets' )
ak4CaloJetsL1 = ak4CaloJetsL1.clone( src = 'ak4CaloJets' )

ak8PFJetsL1 = ak4PFJetsL1.clone( src = 'ak8PFJets' )
kt4PFJetsL1 = ak4PFJetsL1.clone( src = 'kt4PFJets' )
kt6PFJetsL1 = ak4PFJetsL1.clone( src = 'kt6PFJets' )
ak4PFJetsL1 = ak4PFJetsL1.clone( src = 'ak4PFJets' )


# L2L3 Correction Producers
ak8CaloJetsL2 = ak8CaloJetsL1.clone(correctors = ['ak8CaloL2Relative'])
kt4CaloJetsL2 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2Relative'])
kt6CaloJetsL2 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2Relative'])
ak4CaloJetsL2 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2Relative'])

ak8PFJetsL2 = ak8PFJetsL1.clone(correctors = ['ak8PFL2Relative'])
kt4PFJetsL2 = kt4PFJetsL1.clone(correctors = ['kt4PFL2Relative'])
kt6PFJetsL2 = kt6PFJetsL1.clone(correctors = ['kt6PFL2Relative'])
ak4PFJetsL2 = ak4PFJetsL1.clone(correctors = ['ak4PFL2Relative'])

ak4JPTJetsL2 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL2Relative'])
ak4TrackJetsL2 = ak4TrackJetsL1.clone(correctors = ['ak4TRKL2Relative'])

# L2L3 Correction Producers
ak8CaloJetsL2L3 = ak8CaloJetsL1.clone(correctors = ['ak8CaloL2L3'])
kt4CaloJetsL2L3 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2L3'])
kt6CaloJetsL2L3 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2L3'])
ak4CaloJetsL2L3 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2L3'])

ak8PFJetsL2L3 = ak8PFJetsL1.clone(correctors = ['ak8PFL2L3'])
kt4PFJetsL2L3 = kt4PFJetsL1.clone(correctors = ['kt4PFL2L3'])
kt6PFJetsL2L3 = kt6PFJetsL1.clone(correctors = ['kt6PFL2L3'])
ak4PFJetsL2L3 = ak4PFJetsL1.clone(correctors = ['ak4PFL2L3'])

ak4JPTJetsL2L3 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL2L3'])
ak4TrackJetsL2L3 = ak4TrackJetsL1.clone(correctors = ['ak4TRKL2L3'])

# L1L2L3 Correction Producers
ak8CaloJetsL1L2L3 = ak8CaloJetsL1.clone(correctors = ['ak8CaloL1L2L3'])
kt4CaloJetsL1L2L3 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL1L2L3'])
kt6CaloJetsL1L2L3 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL1L2L3'])
ak4CaloJetsL1L2L3 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1L2L3'])

ak8PFJetsL1L2L3 = ak8PFJetsL1.clone(correctors = ['ak8PFL1L2L3'])
kt4PFJetsL1L2L3 = kt4PFJetsL1.clone(correctors = ['kt4PFL1L2L3'])
kt6PFJetsL1L2L3 = kt6PFJetsL1.clone(correctors = ['kt6PFL1L2L3'])
ak4PFJetsL1L2L3 = ak4PFJetsL1.clone(correctors = ['ak4PFL1L2L3'])

ak4JPTJetsL1L2L3 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL1L2L3'])
ak4TrackJetsL1L2L3 = ak4TrackJetsL1.clone(correctors = ['ak4TRKL1L2L3'])

# L2L3L6 CORRECTION PRODUCERS
ak8CaloJetsL2L3L6 = ak8CaloJetsL1.clone(correctors = ['ak8CaloL2L3L6'])
kt4CaloJetsL2L3L6 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2L3L6'])
kt6CaloJetsL2L3L6 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2L3L6'])
ak4CaloJetsL2L3L6 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2L3L6'])

ak8PFJetsL2L3L6 = ak8PFJetsL1.clone(correctors = ['ak8PFL2L3L6'])
kt4PFJetsL2L3L6 = kt4PFJetsL1.clone(correctors = ['kt4PFL2L3L6'])
kt6PFJetsL2L3L6 = kt6PFJetsL1.clone(correctors = ['kt6PFL2L3L6'])
ak4PFJetsL2L3L6 = ak4PFJetsL1.clone(correctors = ['ak4PFL2L3L6'])


# L1L2L3L6 CORRECTION PRODUCERS
ak8CaloJetsL1L2L3L6 = ak8CaloJetsL1.clone(correctors = ['ak8CaloL1L2L3L6'])
kt4CaloJetsL1L2L3L6 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL1L2L3L6'])
kt6CaloJetsL1L2L3L6 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL1L2L3L6'])
ak4CaloJetsL1L2L3L6 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1L2L3L6'])

ak8PFJetsL1L2L3L6 = ak8PFJetsL1.clone(correctors = ['ak8PFL1L2L3L6'])
kt4PFJetsL1L2L3L6 = kt4PFJetsL1.clone(correctors = ['kt4PFL1L2L3L6'])
kt6PFJetsL1L2L3L6 = kt6PFJetsL1.clone(correctors = ['kt6PFL1L2L3L6'])
ak4PFJetsL1L2L3L6 = ak4PFJetsL1.clone(correctors = ['ak4PFL1L2L3L6'])
