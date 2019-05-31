import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing ('analysis')
# add a list of strings for events to process
# options.register ('eventsToProcess',
#                                   '',
#                                   VarParsing.multiplicity.list,
#                                   VarParsing.varType.string,
#                                   "Events to process")
# options.parseArguments()


Source_Files = cms.untracked.vstring(
#        "/store/relval/CMSSW_10_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/94X_upgrade2023_realistic_v2_2023D17noPU-v2/10000/06C888F3-CFCE-E711-8928-0CC47A4D764C.root"
         #"/store/relval/CMSSW_9_3_2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v2_2023D17noPU-v1/10000/0681719F-AFA6-E711-87C9-0CC47A4C8E14.root"
         #"file:///eos/user/k/kbunkow/cms_data/0681719F-AFA6-E711-87C9-0CC47A4C8E14.root"
         #"file:///eos/cms/store/group/upgrade/sandhya/SMP-PhaseIIFall17D-00001.root"
         #'file:///afs/cern.ch/work/k/kbunkow/private/omtf_data/SingleMu_15_p_1_1_qtl.root' no high eta in tis file
         #'file:///eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/SingleMu_12_p_10_1_mro.root' ,
         #'file:///eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/SingleMu_20_p_118_1_sTk.root' ,
         #'file:///eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/SingleMu_5_p_81_1_Ql3.root',
         #'file:///eos/user/k/kbunkow/cms_data/SingleMuFullEta/721_FullEta_v4/SingleMu_31_p_89_2_MJS.root',
         #"/store/mc/PhaseIIFall17D/SingleMu_FlatPt-2to100/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/00000/F4EEAE55-C937-E811-8C29-48FD8EE739D1.root"
        #"/store/mc/PhaseIIFall17D/HSCPppstau_M_871_TuneCUETP8M1_14TeV_pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/120000/18156A80-66EC-E811-AE02-0CC47AFCC62A.root"
        #"/store/mc/PhaseIIFall17D/HSCPppstau_M_200_TuneCUETP8M1_14TeV_pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/120000/FE3D8AD6-B6D0-E811-8FBD-141877412793.root"
        #'/store/mc/RunIISummer16DR80Premix/HSCPppstau_M-651_TuneZ2star_13TeV_pythia6/AODSIM/PUMoriond17_HSCP_customise_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/0E0D542C-A9C8-E611-981C-A0000420FE80.root'
        #'/store/mc/RunIISummer16DR80Premix/HSCPppstau_M-308_TuneZ2star_13TeV-pythia6/AODSIM/PUMoriond17_HSCP_customise_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/3EAEE028-AED2-E611-83E8-002590E7DFEE.root'
        #'/store/mc/RunIISummer16DR80Premix/HSCPppstau_M-308_TuneZ2star_13TeV-pythia6/AODSIM/PUMoriond17_HSCP_customise_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/80000/9A3F1E44-AED2-E611-9AD6-1CC1DE18CFDE.root'
        #'/store/mc/PhaseIIFall17D/ZMM_14TeV_TuneCUETP8M1_Pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/60000/EE29AF8E-51AF-E811-A2BD-484D7E8DF0D3.root'
        #'/store/mc/PhaseIITDRSpring17GS/HSCPppstau_M_1599_TuneCUETP8M1_14TeV_pythia8_Customised/GEN-SIM/Customised_91X_upgrade2023_realistic_v3-v2/00000/CC5458A2-5BA9-E711-86E1-0025905D1D7A.root'
        '/store/mc/PhaseIIFall17D/SingleMu_FlatPt-2to100/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v1/00000/32DF01CC-A342-E811-9FE7-48D539F3863E.root'
)

process = cms.Process("PickEvent")
process.source = cms.Source ("PoolSource",
          fileNames = Source_Files, #cms.untracked.vstring (options.inputFiles),
          #eventsToProcess = cms.untracked.VEventRange (options.eventsToProcess),
          dropDescendantsOfDroppedBranches=cms.untracked.bool(False),
          inputCommands=cms.untracked.vstring(
          'keep *',
          'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
          'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
          'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
          'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
          'drop l1tEMTFTrack2016s_simEmtfDigis__HLT',
          'drop l1tHGCalClusterBXVector_hgcalTriggerPrimitiveDigiProducer_cluster2D_HLT',
          'drop *HGCal*_*_*_*',
          'drop *hgcal*_*_*_*',
          'drop *Ecal*_*_*_*',
          'drop *Hcal*_*_*_*',
          'drop *Calo*_*_*_*',
          
          'drop *_*HGCal*_*_*',
          'drop *_*hgcal*_*_*',
          'drop *_*Ecal*_*_*',
          'drop *_*Hcal*_*_*',
          'drop *_*Calo*_*_*',
          
          'drop *_*_*HGCal*_*',
          'drop *_*_*hgcal*_*',
          'drop *_*_*Ecal*_*',
          'drop *_*_*Hcal*_*',
          'drop *_*_*Calo*_*',
          )                               
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(500))

#outputFileNme = '/eos/user/k/kbunkow/cms_data/mc/PhaseIIFall17D/SingleMu_FlatPt-2to100/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/00000/F4EEAE55-C937-E811-8C29-48FD8EE739D1_dump1000Events.root'
#outputFileNme = 'HSCPppstau_M-651_TuneZ2star_13TeV_0E0D542C-A9C8-E611-981C-A0000420FE80_dump100Events.root'
outputFileNme = 'SingleMu_PU200_32DF01CC-A342-E811-9FE7-48D539F3863E_dump500Events.root'

process.Out = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string (outputFileNme)
)

process.end = cms.EndPath(process.Out)
