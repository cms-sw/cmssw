import FWCore.ParameterSet.Config as cms

# generate the efficiency strings for the DQMGenericClient from the pt and quality cuts
def generateEfficiencyStrings(ptQualCuts):
    numDenDir = "numerators_and_denominators/"
    varStrings = ['pt', 'eta', 'phi', 'vtx']
    etaStrings = ['etaMin0_etaMax0p83', 'etaMin0p83_etaMax1p24', 'etaMin1p24_etaMax2p4', 'etaMin0_etaMax2p4']
    qualStrings = {0:'qualAll', 4:'qualOpen', 8:'qualDouble', 12:'qualSingle'}

    efficiencyStrings = []
    for ptQualCut in ptQualCuts:
        effDenNamePrefix = numDenDir+"effDen_"
        effNumNamePrefix = numDenDir+"effNum_"
        effNamePrefix = "eff_"
        for varString in varStrings:
            effDenNameVar = effDenNamePrefix+varString
            effNumNameVar = effNumNamePrefix+varString+"_"+str(ptQualCut[0])
            effNameVar = effNamePrefix+varString+"_"+str(ptQualCut[0])
            if varString != "pt":
                effDenNameVar += "_"+str(ptQualCut[0])
            effDenNameEta = ''
            effNumNameEta = ''
            effNameEta = ''
            if varString != "eta":
                for etaString in etaStrings:
                    effDenName = effDenNameVar+"_"+etaString
                    effNumName = effNumNameVar+"_"+etaString+"_"+qualStrings[ptQualCut[1]]
                    effName = effNameVar+"_"+etaString+"_"+qualStrings[ptQualCut[1]]
                    efficiencyStrings.append(effName+" '"+effName+";;L1 muon efficiency' "+effNumName+" "+effDenName)
            else:
                effDenName = effDenNameVar
                effNumName = effNumNameVar+"_"+qualStrings[ptQualCut[1]]
                effName = effNameVar+"_"+qualStrings[ptQualCut[1]]
                efficiencyStrings.append(effName+" '"+effName+";;L1 muon efficiency' "+effNumName+" "+effDenName)
    return efficiencyStrings

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.L1Trigger.L1TMuonDQMOffline_cfi import ptQualCuts, ptQualCuts_HI

l1tMuonDQMEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("L1T/L1TObjects/L1TMuon/L1TriggerVsReco/"),
    efficiency = cms.vstring(),
    efficiencyProfile = cms.untracked.vstring(generateEfficiencyStrings(ptQualCuts)),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(0)
)

# emulator efficiency
l1tMuonDQMEmuEfficiency = l1tMuonDQMEfficiency.clone(
    subDirs = cms.untracked.vstring("L1TEMU/L1TObjects/L1TMuon/L1TriggerVsReco/")
)

# modifications for the pp reference run
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tMuonDQMEfficiency,
    efficiencyProfile = cms.untracked.vstring(generateEfficiencyStrings(ptQualCuts_HI))
)
ppRef_2017.toModify(l1tMuonDQMEmuEfficiency,
    efficiencyProfile = cms.untracked.vstring(generateEfficiencyStrings(ptQualCuts_HI))
)

