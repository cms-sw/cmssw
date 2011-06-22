import FWCore.ParameterSet.Config as cms

'''

Dummy code to 'select' MC jets

Authors: Christian Veelken, Evan Friis (UC Davis)

'''

# Have to put all this in a pset so it gets imported.
filterConfig = cms.PSet(
    name = cms.string("MC"),
    hltPaths = cms.vstring('HLT_Mu9'),
    #hltPaths = cms.vstring('HLT_Mu9'),
    # Flag to specify whether we want to use (unbiased) jets that are matched to our
    # trigger.  Not for this case.
    useUnbiasedHLTMatchedJets = cms.bool(False),
)

#--------------------------------------------------------------------------------

selectEnrichedEvents = cms.Sequence()

#  dbs search --query="find file where primds=RelValWM and release=CMSSW_3_11_1 and tier=GEN-SIM-RECO"
filterConfig.testFiles = cms.vstring([
    "file:/data2/friis/skims/final_events_AHtoMuTau_ZtautauPU156bx_Run52plainskim_30_46ba.root",
    #"/store/relval/CMSSW_4_1_3/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/START311_V2-v1/0038/AC686E74-5B52-E011-A4EE-003048679070.root",
    #"/store/relval/CMSSW_4_1_3/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/START311_V2-v1/0037/CEF63B8A-C351-E011-8A49-001A92971B04.root",
    #"/store/relval/CMSSW_4_1_3/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/START311_V2-v1/0037/7CA5D0AE-D051-E011-8F76-003048678E2A.root",
    #"/store/relval/CMSSW_4_1_3/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/START311_V2-v1/0037/5CA6FBAA-C851-E011-92E5-00304867929E.root",

])
