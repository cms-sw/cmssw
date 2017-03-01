import FWCore.ParameterSet.Config as cms

process = cms.Process('TOPDQM')

## imports of standard configurations
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## --------------------------------------------------------------------
## Frontier Conditions: (adjust accordingly!!!)
##
## For CMSSW_3_8_X MC use             ---> 'START38_V12::All'
## For Data (38X re-processing) use   ---> 'GR_R_38X_V13::All'
## For Data (38X prompt reco) use     ---> 'GR10_P_V10::All'
##
## For more details have a look at: WGuideFrontierConditions
## --------------------------------------------------------------------
##process.GlobalTag.globaltag = 'GR_R_42_V14::All' 
process.GlobalTag.globaltag = 'MCRUN2_74_V9'


#dbs search --query 'find file where site=srm-eoscms.cern.ch and dataset=/RelValTTbar/CMSSW_7_0_0_pre3-PRE_ST62_V8-v1/GEN-SIM-RECO'
#dbs search --query 'find dataset where dataset=/RelValTTbar/CMSSW_7_0_0_pre6*/GEN-SIM-RECO'

## input file(s) for testing
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring("file:input.root',")
    fileNames = cms.untracked.vstring(
    #"/store/relval/CMSSW_6_2_0_pre1-START61_V8/RelValTTbarLepton/GEN-SIM-RECO/v1/00000/C6CC53CC-6E6D-E211-8EAB-003048D3756A.root',"
    
    #/RelValTTbar/CMSSW_7_0_0_pre6-PRE_ST62_V8-v1/GEN-SIM-RECO
    #    '/store/relval/CMSSW_7_0_0_pre6/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/B627D32C-0B3C-E311-BBE6-0026189438E6.root',
    #    '/store/relval/CMSSW_7_0_0_pre6/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/72477A84-F93B-E311-BF63-003048FFD720.root',
    #    '/store/relval/CMSSW_7_0_0_pre6/RelValTTbar/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/12A06D7A-F93B-E311-AA64-003048678BEA.root'
        #        '/store/relval/CMSSW_7_1_0_pre4/RelValTTbarLepton_13/GEN-SIM-RECO/POSTLS171_V1-v2/00000/48ED95A2-66AA-E311-9865-02163E00E5AE.root'
#        '/store/relval/CMSSW_7_1_0_pre9/RelValTTbarLepton_13/GEN-SIM-RECO/POSTLS171_V11-v1/00000/90D92DC6-67F0-E311-8D2E-0025905A613C.root'

         '/store/mc/RunIISpring15MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2_ext3-v1/10000/003964D7-D06E-E511-A8DA-001517F7F524.root',
         '/store/mc/RunIISpring15MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2_ext3-v1/10000/0041D4C0-D86E-E511-8D6B-001E67A3E8F9.root',
         '/store/mc/RunIISpring15MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2_ext3-v1/10000/00962466-E86E-E511-9105-0026189438D7.root',
         '/store/mc/RunIISpring15MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2_ext3-v1/10000/00A7C4B3-766E-E511-973F-002618943904.root',
         '/store/mc/RunIISpring15MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2_ext3-v1/10000/020B5100-426E-E511-888A-0026189437F9.root',
         '/store/mc/RunIISpring15MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2_ext3-v1/10000/024E3A40-706F-E511-AE4B-68B5996BD98E.root',
         '/store/mc/RunIISpring15MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/74X_mcRun2_asymptotic_v2_ext3-v1/10000/025CB88B-706F-E511-9F87-6CC2173D46A0.root'

#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/00000/B4414D37-CF97-E511-B2C4-002590747DE2.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/00000/B4414D37-CF97-E511-B2C4-002590747DE2.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/006945CA-3088-E511-96C4-0025905A497A.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/04B296F0-2A88-E511-A760-002618943856.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/04F9A001-2888-E511-ABEC-0025905A60B6.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/0A37E1C7-668B-E511-81DC-FACADE0000AF.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/0A401EEE-2A88-E511-AFBC-0026189438AD.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/0A5583EC-2A88-E511-A990-0CC47A4D768C.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/0E9802F4-2A88-E511-B4EE-002618943970.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/1609D954-5D87-E511-AB94-0025907B4F64.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/1646260C-189A-E511-9854-0025904C66E4.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/16DF38A8-2388-E511-B447-0CC47A4C8E56.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/182952B2-6C89-E511-9CF7-0CC47A4D76C0.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/1A473DEA-2A88-E511-8DC0-00261894393E.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/1A5653A9-2388-E511-BE59-0025905A60AA.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/2002E1DA-2A88-E511-B93C-0CC47A78A4B8.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/2031DEEF-2A88-E511-ADF1-0CC47A4C8EEA.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/20ED3D7F-2588-E511-AC35-00505602077D.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/22B66702-2188-E511-AB24-20CF305616E2.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/24B786E8-A087-E511-80C3-002618943950.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/2A2AF453-2188-E511-9918-0025907B503A.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/2AEC7850-1F88-E511-832C-0025907B503A.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/2C573255-009A-E511-A1DB-0025905C2CD2.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/30F9373F-3188-E511-9769-0CC47A4C8ECA.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/34BFD4ED-2A88-E511-8609-0CC47A4D7634.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/366DF93F-3188-E511-BF1D-0CC47A4C8EB6.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/3ACD13F0-2A88-E511-9FD6-0CC47A4C8F12.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/4467632E-2188-E511-A5F1-002590D0B044.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/508CF007-189A-E511-AB08-0025905C3D6A.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/5204AAEB-9F87-E511-AA0B-0026189438CC.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/642A7DD7-2A88-E511-BB53-0CC47A4D767E.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/64395FEB-2A88-E511-929E-0026189437FD.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/689CC1E6-3888-E511-8D89-0CC47A4C8E0E.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/6A1E3BFF-2788-E511-B16D-0CC47A4D767E.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/6E848416-7D8B-E511-82A9-047D7BD6DF34.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/74096A0B-189A-E511-AC3C-0025905C9742.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/74788752-1F88-E511-9F39-20CF305616E2.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/748ED45E-3C88-E511-8E27-0025905964B6.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/7C7F85C8-668B-E511-A3D0-FACADE0000AF.root',
#    '/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/7CACA6D7-2A88-E511-83D9-0CC47A78A472.root',



    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/7E65204F-1F88-E511-8D2C-002590D0B044.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/8072C45A-948B-E511-9106-00266CFFA1AC.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/82F384FF-2788-E511-B894-0CC47A78A4B8.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/8414FF0E-7D8B-E511-8007-047D7B881D26.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/848000A9-2388-E511-B9D2-0025905A60FE.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/868B6D02-2888-E511-B667-0CC47A78A472.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/86F0A513-0A89-E511-9D31-A01D48EE8318.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/8C081F49-AD91-E511-A1FA-0025904CDDFA.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/8C9B920B-189A-E511-8523-0025904C63FA.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/8EB36200-2888-E511-8DC3-0025905A6136.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/90D9B150-1F88-E511-BD7A-002590D0B058.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/924B4D60-3C88-E511-BEA7-0025905A60BE.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/949C8F64-FF99-E511-B8DE-0025905BA734.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/94FD91A8-2388-E511-9F33-0CC47A78A42E.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/96BFF2B0-2888-E511-8916-002590D0B0D4.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/9A33A5D1-668B-E511-93F1-002590AC4BF8.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/A06767EC-2A88-E511-BE75-00261894398C.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/AA123970-FF99-E511-A864-0025904C641C.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/AA8972DE-5787-E511-9DCB-002590D0AFC6.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/AE9D38EE-5A87-E511-A5B2-002590D0AFB6.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/B66A08F1-2A88-E511-A98F-0025905A6094.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/B80D1283-5E87-E511-B089-002590D0B046.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/BAB37002-2888-E511-BBE0-0CC47A78A472.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/C02D41A7-2388-E511-8C02-0CC47A4D761A.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/C041AD83-5E87-E511-8258-002590D0B052.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/C05800A4-2388-E511-ADE5-0026189438FD.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/C84167BF-9C87-E511-A3C2-002618943940.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/D221EED8-2588-E511-9A68-002590D0AFF4.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/D63C75F1-2A88-E511-8B15-0025905B860E.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/DA0B72BF-9C87-E511-8734-002618943916.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/DE080BE8-3888-E511-B2BD-0CC47A4C8E0E.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/E240F1C1-9C87-E511-9374-0026189438CC.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/EE697B2B-3788-E511-857E-0CC47A7452D0.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/F231403E-3E88-E511-A3E6-0025905B855E.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/F4FD5AF0-2A88-E511-8C28-0CC47A4C8F10.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/F675A30B-189A-E511-B35B-0025904C63FA.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/FA529E5B-5987-E511-8F30-005056020786.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/FAA42B44-A087-E511-9D9E-00248C0BE005.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/20000/FAC5E021-3788-E511-8953-0CC47A4D7618.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/0028B733-4C87-E511-BBEE-002590D0B052.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/12DF8FB8-5C87-E511-88A1-0025907B5002.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/1607DD10-5E87-E511-BEC7-0025905A60D2.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/2821C96A-5087-E511-86F9-002590D0B086.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/2A04A99D-7187-E511-A3B5-0025905A6088.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/2EA9647A-5887-E511-8BF8-002590D0B09A.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/2EEA626F-D18E-E511-9FDE-02163E00F449.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/368618C7-2F8E-E511-BCB2-6CC2173C3DD0.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/48DCFD10-5E87-E511-8193-0025905A6138.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/4A2F3EE2-6087-E511-8D3F-0025905A612E.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/524B846A-5087-E511-81AE-002590D0B0CC.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/76FAB9E2-6087-E511-B1E0-0025905A6080.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/A625DC4C-4C87-E511-A0C6-002590D0B052.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/A69D21E3-5C87-E511-BCF5-0CC47A4DEEF8.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/B8AFEA2D-B786-E511-964E-0025905AA9CC.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/F828D2E2-6087-E511-BDBD-0025905A6138.root',
    #'/store/mc/RunIIFall15DR76/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/25nsFlat10to25TSG_76X_mcRun2_asymptotic_v11_ext3-v1/30000/FEEE3F10-5E87-E511-B346-0025905B85D6.root',


    )
)

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

## apply VBTF electronID (needed for the current implementation
## of topSingleElectronDQMLoose and topSingleElectronDQMMedium)
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("DQM.Physics.topElectronID_cff")
process.load('Configuration/StandardSequences/Reconstruction_cff')


## output
process.output = cms.OutputModule("PoolOutputModule",
  fileName       = cms.untracked.string('topDQM_production.root'),
  outputCommands = cms.untracked.vstring(
    'drop *_*_*_*',
    'keep *_*_*_TOPDQM',
    'drop *_TriggerResults_*_TOPDQM',
    'drop *_simpleEleId70cIso_*_TOPDQM',
  ),
  splitLevel     = cms.untracked.int32(0),
  dataset = cms.untracked.PSet(
    dataTier   = cms.untracked.string(''),
    filterName = cms.untracked.string('')
  )
)

## load jet corrections
#process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
#process.prefer("ak4PFL2L3")

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TopSingleLeptonDQM'   )
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('TopDiLeptonOfflineDQM')
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('SingleTopTChannelLeptonDQM'   )
process.MessageLogger.cerr.SingleTopTChannelLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MEtoEDMConverter.deleteAfterCopy = cms.untracked.bool(False)  ## line added to avoid crash when changing run number


process.load("DQM.Physics.topSingleLeptonDQM_miniAOD_cfi")
process.load("DQM.Physics.singleTopDQM_miniAOD_cfi")


## path definitions
process.p      = cms.Path(
#    process.simpleEleId70cIso          *
#    process.DiMuonDQM                  +
#    process.DiElectronDQM              +
#    process.ElecMuonDQM                +
    #process.topSingleMuonLooseDQM      +
    process.topSingleMuonMediumDQM_miniAOD +
    #process.topSingleElectronLooseDQM  +
    process.topSingleElectronMediumDQM_miniAOD +
     process.singleTopMuonMediumDQM_miniAOD     +
    process.singleTopElectronMediumDQM_miniAOD
)
process.endjob = cms.Path(
    process.endOfProcess
)
process.fanout = cms.EndPath(
    process.output
)

## schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.endjob,
    process.fanout
)
