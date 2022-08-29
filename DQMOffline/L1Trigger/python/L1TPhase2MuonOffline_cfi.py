from builtins import range
import FWCore.ParameterSet.Config as cms

# define binning for efficiency plots
# pt
import itertools
effVsPtBins=list(itertools.chain(range(0, 30, 1), range(30, 50, 2), 
                                 range(50, 70, 5), range(70, 100, 10), 
                                 range(100, 200, 25), range(200, 300, 50), 
                                 range(300, 500, 100), range(500, 700, 200), 
                                 range(700, 1000, 300)))
effVsPtBins.append(1000)

# phi
nPhiBins = 34
phiMin = -3.4
phiMax = 3.4
effVsPhiBins = [i*(phiMax-phiMin)/nPhiBins + phiMin for i in range(nPhiBins+1)]

# eta
nEtaBins = 50
etaMin = -2.5
etaMax = 2.5
effVsEtaBins = [i*(etaMax-etaMin)/nEtaBins + etaMin for i in range(nEtaBins+1)]

# vtx
effVsVtxBins = range(0, 101)

# A list of pt cut + quality cut pairs for which efficiency plots should be made
ptQualCuts = [[22, 12], [15, 8], [3, 4]]
cutsPSets = []
for ptQualCut in ptQualCuts:
    cutsPSets.append(cms.untracked.PSet(ptCut = cms.untracked.int32(ptQualCut[0]),
                                        qualCut = cms.untracked.int32(ptQualCut[1])))
    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tPhase2MuonOffline = DQMEDAnalyzer('L1TPhase2MuonOffline',
    histFolder = cms.untracked.string('L1T/L1TPhase2/Muons/'),
    cuts = cms.untracked.VPSet(cutsPSets),
    useL1AtVtxCoord = cms.untracked.bool(False),
    
    genParticlesInputTag = cms.untracked.InputTag("genParticles"),
    gmtMuonToken  = cms.InputTag("l1tSAMuonsGmt", "promptSAMuons"),
    gmtTkMuonToken  = cms.InputTag("l1tTkMuonsGmt",""),

    efficiencyVsPtBins = cms.untracked.vdouble(effVsPtBins),
    efficiencyVsPhiBins = cms.untracked.vdouble(effVsPhiBins),
    efficiencyVsEtaBins = cms.untracked.vdouble(effVsEtaBins),
    efficiencyVsVtxBins = cms.untracked.vdouble(effVsVtxBins),
   
    maxDR = cms.untracked.double(0.3),
)


