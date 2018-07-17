import os, sys
import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTExoticaOfflineAnalysis")

process.load("HLTriggerOffline.Exotica.ExoticaValidation_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# Decide input data
myinput   = ""
myfileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/0A69FE2A-DAFD-E311-9FA2-00261894391C.root',
    #'/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/4ACF05B1-ABFD-E311-BC8C-0026189438BC.root'
    'file:/results_exo_nobptx.root'
    )

for i in range(0,len(sys.argv)):
    if str(sys.argv[i])=="_input" and len(sys.argv)>i+1:
        myinput = str(sys.argv[i+1])
        
print "Using myinput="+myinput

if   myinput=="ZEE" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/0A69FE2A-DAFD-E311-9FA2-00261894391C.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/4ACF05B1-ABFD-E311-BC8C-0026189438BC.root')

elif myinput=="ZMM" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZMM_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/12E10345-F9FD-E311-9694-0026189438AE.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZMM_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/B25222BF-F9FD-E311-A217-003048FFCBA4.root')

elif myinput=="ZpEE" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZpEE_2250_13TeV_Tauola/GEN-SIM-RECO/POSTLS172_V1-v1/00000/343EDE59-F9FD-E311-898D-0025905A612A.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZpEE_2250_13TeV_Tauola/GEN-SIM-RECO/POSTLS172_V1-v1/00000/E2A8B37B-FDFD-E311-8018-002618943849.root')

elif myinput=="ZpMM" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZpMM_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/80CB6E2E-0BFE-E311-B644-0025905A4964.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZpMM_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/E0547611-00FE-E311-88AB-0025905A48F0.root')

elif myinput=="PhotonJets" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/6AD79C3F-F9FD-E311-AD0C-0025905A610A.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/F8CBE44F-F6FD-E311-8F8C-0025905964BC.root')

elif myinput=="QCD":
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/1EB9CDDA-C8FE-E311-9082-0025905A6066.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/40620B64-C8FE-E311-86CB-002354EF3BDE.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/6065EBCA-C2FE-E311-9CD4-003048FFCB8C.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/68B120F8-CAFE-E311-AAA0-0025905A60FE.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/72A73941-C8FE-E311-8093-0025905A612C.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/7E99AB18-CCFE-E311-9ECD-0025905A607E.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/A0A799B8-C7FE-E311-A820-0025905A6122.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/AA107A6D-C9FE-E311-B3E4-003048FFCBA4.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/B4A2DA31-C7FE-E311-B197-0025905A60A8.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/B67E922A-CAFE-E311-86F9-002618943918.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1-v1/00000/E695A31A-C3FE-E311-B86B-002618943865.root')
    
elif myinput=="ZEE_f" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/0C68F3E2-D300-E411-A2E1-003048FFCC0A.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/6A11D18F-D800-E411-A121-0026189438C4.root')

elif myinput=="ZMM_f" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZMM_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/2200CE0D-C600-E411-BF5B-0025905A60B4.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZMM_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/8860B293-DB00-E411-A736-0025905A60D6.root')

elif myinput=="ZpEE_f" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpMM_f" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZpMM_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/88067900-C600-E411-B0D4-003048FFD76E.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZpMM_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/BA282D59-C900-E411-B969-0025905A6064.root')

elif myinput=="PhotonJets_f" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/90422F57-D900-E411-944B-002590593872.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/E0DD9115-BE00-E411-8C70-00261894382D.root')

elif myinput=="QCD_f":
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/0AE4F4AE-BE00-E411-8E38-00259059649C.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/0CCB8433-C600-E411-AFA1-0025905B85B2.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/24043AB8-BD00-E411-B4A6-003048FFD730.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/28514BA6-BD00-E411-9B9A-002618943860.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/4C0435CC-BD00-E411-8766-002354EF3BD2.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/52BBEE04-DC00-E411-9BDC-003048FFD7D4.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/5E349B87-BE00-E411-B2CA-0025905A609E.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/74B1258C-BF00-E411-91AC-0025905AA9F0.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/8668C7C0-DE00-E411-863D-003048FFD752.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/9A6C429F-BD00-E411-86FE-002618943857.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/ACA31613-BE00-E411-B352-002590596484.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/C65023E5-BD00-E411-963E-0025905A60E4.root',
       '/store/relval/CMSSW_7_2_0_pre1/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS172_V1_frozenHLT-v1/00000/E8DB8D9A-BE00-E411-BA12-003048FFD7C2.root')

# PU40 25ns
elif   myinput=="ZEE_PU25ns" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1-v1/00000/00E2C10F-E6FD-E311-ACE3-0025905A60DA.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1-v1/00000/62F42186-91FD-E311-92E8-0025905A60EE.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1-v1/00000/78D08417-8DFD-E311-B022-0026189438D9.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1-v1/00000/CE1B9922-8BFD-E311-82F5-00259059642E.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1-v1/00000/E468EFE3-E7FD-E311-A9FF-0025905A6088.root')

elif myinput=="ZMM_PU25ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpEE_PU25ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpMM_PU25ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZEE_f_PU25ns" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1_frozenHLT-v1/00000/0C0AFEDA-B600-E411-B890-0025905A6094.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1_frozenHLT-v1/00000/684B8352-CC00-E411-B019-0026189438B3.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1_frozenHLT-v1/00000/A872BF63-B800-E411-AC92-003048FFCBA8.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1_frozenHLT-v1/00000/EC7D73CF-D500-E411-926D-002618943886.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_POSTLS172_V1_frozenHLT-v1/00000/F2B7DB47-B500-E411-88B3-0025905A613C.root')
    
elif myinput=="ZMM_f_PU25ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpEE_f_PU25ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpMM_f_PU25ns" :
    myfileNames = cms.untracked.vstring()

## PU40 50ns
elif myinput=="ZEE_PU50ns" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/0213DA84-8AFD-E311-A2FB-0026189438FF.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/10BE77F0-86FD-E311-A7F6-0026189438DF.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/3410BEEC-E6FD-E311-9D13-0025905A60F4.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/5471FED0-8CFD-E311-8064-003048FFCB74.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/C2BECB0F-89FD-E311-83DE-0025905B8610.root')

elif myinput=="ZMM_PU50ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpEE_PU50ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpMM_PU50ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZEE_f_PU50ns" :
    myfileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2_frozenHLT-v1/00000/02D5E5C6-BD00-E411-B421-00259059391E.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2_frozenHLT-v1/00000/2817C10B-D600-E411-B17C-002618943939.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2_frozenHLT-v1/00000/547D27A4-C700-E411-8686-0025905A60EE.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2_frozenHLT-v1/00000/58E959E1-BA00-E411-89F8-0025905938A4.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2_frozenHLT-v1/00000/D0F38943-D800-E411-8517-002590593920.root',
        '/store/relval/CMSSW_7_2_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2_frozenHLT-v1/00000/FABFF406-C300-E411-AE96-002618943800.root')

elif myinput=="ZMM_f_PU50ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpEE_f_PU50ns" :
    myfileNames = cms.untracked.vstring()

elif myinput=="ZpMM_f_PU50ns" :
    myfileNames = cms.untracked.vstring()

print "### Files : "
print myfileNames


##############################################################################
##### Templates to change parameters in hltMuonValidator #####################
# process.hltMuonValidator.hltPathsToCheck = ["HLT_IsoMu3"]
# process.hltMuonValidator.genMuonCut = "abs(mother.pdgId) == 24"
# process.hltMuonValidator.recMuonCut = "isGlobalMuon && eta < 1.2"
##############################################################################

hltProcessName = "HLT"
process.hltExoticaValidator.hltProcessName = hltProcessName

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string(autoCond['startup'])

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource", fileNames=myfileNames)

process.DQMStore = cms.Service("DQMStore")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.destinations += ['ExoticaValidationMessages']
process.MessageLogger.categories   += ['ExoticaValidation']
#process.MessageLogger.debugModules += ['HLTExoticaValidator','HLTExoticaSubAnalysis','HLTExoticaPlotter']
process.MessageLogger.debugModules += ['*']
process.MessageLogger.ExoticaValidationMessages = cms.untracked.PSet(
    threshold       = cms.untracked.string('DEBUG'),
    default         = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    ExoticaValidation = cms.untracked.PSet(limit = cms.untracked.int32(1000))
    )

process.MessageLogger.categories.extend(["GetManyWithoutRegistration","GetByLabelWithoutRegistration"])

_messageSettings = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1),
    optionalPSet = cms.untracked.bool(True),
    limit = cms.untracked.int32(10000000)
    )

process.MessageLogger.cerr.GetManyWithoutRegistration = _messageSettings
process.MessageLogger.cerr.GetByLabelWithoutRegistration = _messageSettings

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_MEtoEDMConverter_*_HLTExoticaOfflineAnalysis'),
    fileName = cms.untracked.string('hltExoticaValidator'+myinput+'.root')
)


process.analyzerpath = cms.Path(
    process.ExoticaValidationProdSeq +
    process.ExoticaValidationSequence +
    process.MEtoEDMConverter
)


process.outpath = cms.EndPath(process.out)
