import FWCore.ParameterSet.Config as cms

# generate the efficiency strings for the DQMGenericClient from the pt and quality cuts
def generateEfficiencyStrings(ptQualCuts):
    numDenDir = "nums_and_dens/"
    varStrings = ['Pt', 'Eta', 'Phi']
    etaStrings = ['etaMin0_etaMax0p83', 'etaMin0p83_etaMax1p24', 'etaMin1p24_etaMax2p4', 'etaMin0_etaMax2p4']
    qualStrings = ['qualOpen', 'qualDouble', 'qualSingle']
    muonStrings = ['SAMuon','TkMuon'] 

    efficiencyStrings = []

    for muonString in muonStrings:
        for qualString in qualStrings:
            for etaString in etaStrings: 
                effNumDenPrefix = numDenDir+"Eff_"+muonString+"_"+etaString+"_"+qualString+"_"
                effNamePrefix = "efficiencies/eff_"+muonString+"_"+etaString+"_"+qualString+"_"
                
                for varString in varStrings:
                    effDenName = effNumDenPrefix+varString+"_Den"
                    effNumName = effNumDenPrefix+varString+"_Num"
                    effName = effNamePrefix+varString
                    
                    efficiencyStrings.append(effName+" '"+effName+";;L1 muon efficiency' "+effNumName+" "+effDenName)
    return efficiencyStrings

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.L1Trigger.L1TPhase2MuonOffline_cfi import ptQualCuts

l1tPhase2MuonEfficiency = DQMEDHarvester("DQMGenericClient",
                                         subDirs = cms.untracked.vstring(["L1T/L1TPhase2/Muons/SAMuon","L1T/L1TPhase2/Muons/TkMuon"]),
    efficiency = cms.vstring(),
    efficiencyProfile = cms.untracked.vstring(generateEfficiencyStrings(ptQualCuts)),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(4)
)
