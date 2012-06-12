#
# Last update: new version for python
#
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("cluTest")
                   
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
# accept if 'path_1' succeeds
process.hltfilter = hlt.hltHighLevel.clone(
# Min-Bias	
#    HLTPaths = ['HLT_Physics_v*'],
#    HLTPaths = ['HLT_Random_v*'],
    HLTPaths = ['HLT_ZeroBias_v*'],
#    HLTPaths = ['HLT_JetE50_NoBPTX3BX_NoHalo_v*','HLT_JetE30_NoBPTX3BX_NoHalo_v*','HLT_JetE30_NoBPTX_v*','HLT_JetE30_NoBPTX_NoHalo_v*'],
#    HLTPaths = ['HLT_PixelTracks_Multiplicity100_v*','HLT_PixelTracks_Multiplicity80_v*'],
#    HLTPaths = ['HLT_JetE50_NoBPTX3BX_NoHalo_v*'],
#    HLTPaths = ['HLT_JetE30_NoBPTX3BX_NoHalo_v*'],
#    HLTPaths = ['HLT_JetE30_NoBPTX_v*'],
#    HLTPaths = ['HLT_JetE30_NoBPTX_NoHalo_v*'],
#     HLTPaths = ['HLT_L1Tech54_ZeroBias*'],
#     HLTPaths = ['HLT_L1Tech53_MB*'],
# Commissioning:
#    HLTPaths = ['HLT_BeamGas_HF_Beam1_v*'],
#    HLTPaths = ['HLT_BeamGas_HF_Beam2_v*'],
#    HLTPaths = ['HLT_BeamGas_HF_Beam1_v*','HLT_BeamGas_HF_Beam2_v*'],
#
#    HLTPaths = ['p*'],
#    HLTPaths = ['path_?'],
    andOr = True,  # False = and, True=or
    throw = False
    )

# to select PhysicsBit
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

# i do not know what is this doing?
triggerSelection = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring(
    'HLT_ZeroBias / 1' ),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag( "gtDigis" ),
    l1tIgnoreMask = cms.bool( True ),
    l1techIgnorePrescales = cms.bool( True ),
    daqPartitions = cms.uint32( 1 ),
    throw = cms.bool( True )
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelClusters'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(    
#  "file:/afs/cern.ch/work/d/dkotlins/public/data/digis.root"
  
## 2012, cosmics
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/791/CEA46376-7069-E111-B395-001D09F24D67.root",
# "/store/data/Commissioning12/Commissioning/RECO/PromptReco-v1/000/186/791/6EC3470C-6F69-E111-93CA-001D09F241B9.root",
# "/store/data/Commissioning12/Cosmics/RECO/PromptReco-v1/000/186/791/6A54D2A0-6D69-E111-ABA8-001D09F2441B.root",
# R186822
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/2C4E0F91-C569-E111-B751-003048D2C01A.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/38A8E118-C969-E111-B30B-003048F117EC.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/3AF5B2FF-C669-E111-8930-003048F024FA.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/48F7CB3D-C469-E111-AB5D-BCAEC53296F8.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/664D8A17-C769-E111-81CA-003048F11114.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/6C772594-C569-E111-ADAE-BCAEC5364C93.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/B29DAEDB-C469-E111-A9DF-0025901D6268.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/BC3CB891-C569-E111-A8A8-BCAEC518FF89.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/BEC6EFFE-C669-E111-BE09-003048F0258C.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/E8CDFC34-CB69-E111-A466-001D09F2AF1E.root",
# "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/186/822/FEC1AA17-C769-E111-BDAE-003048CF94A6.root",
# R 187446
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/FE7B607F-D76D-E111-993E-003048D37538.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/F4E94D8C-D36D-E111-8B8E-0025B3203898.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/F45BE48A-D16D-E111-8873-001D09F25041.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/F2A06371-D36D-E111-89CB-0025901D5D90.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/F01C4674-D36D-E111-B595-5404A63886EB.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/EEC93216-DB6D-E111-A734-BCAEC5329709.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/E447DE30-D86D-E111-928E-5404A63886C7.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/E2193A73-D36D-E111-B41A-E0CB4E4408E7.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/E09580FD-D86D-E111-96FC-003048F1C420.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/E08B0C31-D86D-E111-B44B-E0CB4E55365D.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/D8E4334B-D26D-E111-856A-5404A63886AB.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/CC38AC13-DB6D-E111-8A1B-E0CB4E4408E3.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/CA77838E-D36D-E111-BF84-003048F0258C.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/C8B07B4E-DA6D-E111-A95D-003048F24A04.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/BE506EC4-D66D-E111-B3F5-003048673374.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/BE4F2977-D36D-E111-9884-BCAEC53296F4.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/BC1FA198-D56D-E111-A364-001D09F252E9.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/BC0A2B97-D56D-E111-9878-001D09F24399.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/B8EE564F-DA6D-E111-A3C6-0025B32036D2.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/B07B8B7F-D76D-E111-BB09-003048D2BDD8.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/ACC25274-D36D-E111-862C-BCAEC518FF54.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/AA900C51-DA6D-E111-A4DF-00215AEDFCCC.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/9E5A6448-D26D-E111-B2EC-BCAEC518FF74.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/9C679D51-DA6D-E111-B8FB-002481E0D646.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/98D2E27F-D76D-E111-9DFD-0015C5FDE067.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/8ED12356-DA6D-E111-85F7-001D09F25267.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/8E3DDA51-DA6D-E111-8066-002481E0D958.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/866BDE7F-D76D-E111-B95E-0025B32035BC.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/8666AD8C-D36D-E111-986C-003048F1C424.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/801FD68D-D36D-E111-A36D-BCAEC5329717.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/74E7F77E-D76D-E111-9572-003048D2C16E.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/72C42773-D36D-E111-915A-BCAEC518FF41.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/72AF0133-D66D-E111-837F-003048D2BC38.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/7280F888-D16D-E111-B684-001D09F295FB.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/702B534F-DA6D-E111-BD9A-001D09F29619.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/5AB938FD-D86D-E111-BC54-003048F024FA.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/5A8B532F-D86D-E111-8F81-5404A63886C4.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/58D3E574-D36D-E111-97FA-BCAEC5329705.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/56941D79-D36D-E111-AB00-BCAEC5329719.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/563B1597-D56D-E111-94D5-002481E0D73C.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/54EF4674-D36D-E111-9019-BCAEC5329713.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/5427C280-D76D-E111-8948-001D09F242EF.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/4EF9C201-D56D-E111-825E-00215AEDFD74.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/3EA8C331-D86D-E111-A427-001D09F2A690.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/3E4BCC7C-D76D-E111-BCCF-003048F11114.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/3C6C6C51-DA6D-E111-99CA-00237DDBE49C.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/3AE663DF-D26D-E111-964E-BCAEC518FF44.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/32F29045-D26D-E111-98C6-BCAEC5364C4C.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/2EA3407F-D76D-E111-9847-002481E0DEC6.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/2E137EFD-D86D-E111-9CBC-0025B32035A2.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/28BE4756-DA6D-E111-8899-003048F11DE2.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/2010C213-DB6D-E111-99CC-5404A63886D6.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/1E1641BD-DB6D-E111-B1EB-003048F1C832.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/1C15184F-DA6D-E111-BA10-003048F118C4.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/1A5F5A56-DA6D-E111-A8DD-003048F11942.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/06681347-D26D-E111-8005-E0CB4E4408E3.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/04E47C26-DD6D-E111-8D1C-003048F024F6.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/0233A574-D36D-E111-91C3-BCAEC5364C42.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/0228C49A-D56D-E111-A48B-001D09F24EE3.root",

# 190389 (ran OK, no zb)
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/009B5147-9F80-E111-90B5-001D09F2424A.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/06DD07A7-A080-E111-AA52-0015C5FDE067.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/06FF4150-AA80-E111-B82A-5404A63886B0.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/0E8628A8-A080-E111-AA55-001D09F292D1.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/0EE8D651-AA80-E111-9B0C-5404A63886EF.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/10976EFE-9180-E111-9390-5404A63886B6.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/1A0F5A0C-AD80-E111-BA2A-BCAEC5364C93.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/1A806611-AD80-E111-B702-001D09F2A690.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/220A939E-B180-E111-806C-003048CF94A6.root",
## "/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/190/389/260242F4-B080-E111-AE76-00215AEDFD74.root",

# "/store/express/Commissioning12/ExpressPhysics/FEVT/Express-v1/000/190/411/0280693C-F87E-E111-9911-BCAEC532970F.root",

# run 191271
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/0C745F0F-BE88-E111-9978-485B3977172C.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/0EC6C2CC-B288-E111-93DB-001D09F24353.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/7A86F1C4-B988-E111-BF91-00215AEDFCCC.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/7C042DFF-B488-E111-8171-5404A640A642.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/8CC241CE-B288-E111-8320-001D09F29321.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/AAD273EC-BB88-E111-891A-BCAEC5329713.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/B20AAA76-B688-E111-A69E-BCAEC5364C42.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/B43F5CB7-C388-E111-BBB3-5404A63886EF.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/B60FA7FF-C288-E111-B5D7-5404A6388699.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/C84D5FFB-B488-E111-B0BE-5404A6388694.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/D0E4D5F9-B488-E111-9E82-BCAEC518FF41.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/271/D6F911A0-BC88-E111-A1C9-BCAEC518FF7A.root",

# 191411
#  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/411/66FD8F4B-088A-E111-A96C-001D09F252E9.root",

# 191718
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/2AB8ED3A-B88B-E111-A086-5404A6388692.root",
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/3CCF1232-D28B-E111-8E14-001D09F2424A.root",
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/46F6126B-D68B-E111-8895-003048F11114.root",
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/52075473-C58B-E111-95CD-0015C5FDE067.root",
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/AC4A5932-D28B-E111-BD5F-001D09F24FEC.root",
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/B42DACB6-CE8B-E111-A6DC-001D09F23C73.root",
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/C0241E8A-CC8B-E111-B8A4-001D09F24303.root",
# "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/191/718/F2E89EB6-CE8B-E111-8F33-0019B9F4A1D7.root",

# "/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/718/44DC8D5C-DB8B-E111-B786-5404A63886B2.root",
# "/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/718/46C834F6-D98B-E111-A1D7-5404A638869E.root",
# "/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/718/4E7B5EAE-CE8B-E111-9E02-001D09F29114.root",
# "/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/718/8057AD53-E08B-E111-B0B3-485B39897227.root",
# "/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/718/8801A26B-D68B-E111-A0FC-002481E0DEC6.root",
# "/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/718/A028490E-D08B-E111-8D6D-0025901D627C.root",
# "/store/data/Run2012A/Commissioning/RECO/PromptReco-v1/000/191/718/DE0D2BA4-DA8B-E111-A55A-0025B320384C.root",

# fill 2576
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/10CC3327-9B95-E111-A670-001D09F2A465.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/1AA80E34-8C95-E111-A50D-001D09F24FBA.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/801B1A24-8F95-E111-B7F8-003048D2C108.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/86223BA8-8795-E111-9FEB-BCAEC5329713.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/903C8452-9395-E111-A3EC-001D09F2905B.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/9E6FE753-8995-E111-AE3B-003048F118E0.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/A6D3BB38-8795-E111-ADF6-BCAEC518FF6E.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/B48B8874-9095-E111-9A6A-003048F11C5C.root",
#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_ZeroBias/RECO/PromptReco-v1/000/193/092/DA2D45C9-8995-E111-88A1-0025901D631E.root",

#   "rfio:/castor/cern.ch/cms/store/data/Run2012A/LP_MinBias1/RECO/PromptReco-v1/000/193/092/002CD9DE-8995-E111-869C-001D09F2AF1E.root",

# fill 2596, 
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/1E63A782-8B9A-E111-97CF-001D09F2B30B.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/20257BCD-8F9A-E111-9EE4-001D09F25460.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/3CDAD5EC-949A-E111-9B86-003048D2BC42.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/4C13AD2F-919A-E111-A327-001D09F2B30B.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/5E47D39E-929A-E111-89A8-001D09F25267.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/72295CE6-949A-E111-B7DB-003048F1C836.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/7AC6A094-989A-E111-9F81-001D09F2512C.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/82FD1049-999A-E111-B882-003048F024FE.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/94441B27-949A-E111-93D1-003048F1BF68.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/A0EAD5B6-9C9A-E111-89BB-BCAEC518FF6B.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/A8869045-8C9A-E111-8D13-001D09F27003.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/AAF776F5-8C9A-E111-B091-001D09F25460.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/B05A6127-949A-E111-A824-003048CF9B28.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/B6ED0C24-8A9A-E111-AA86-0019B9F72D71.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/BA234F8A-939A-E111-9F84-001D09F241B9.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/CA8D40AC-8D9A-E111-A572-001D09F244DE.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/CC0FE027-949A-E111-9390-003048F118DE.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/CE296C9F-929A-E111-8748-001D09F244DE.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/D8D77840-8C9A-E111-9332-001D09F2447F.root",
##  "/store/data/Run2012A/MinimumBias/RECO/PromptReco-v1/000/193/621/FEF2BAC8-999A-E111-A3DC-485B3977172C.root",

##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/161575BF-339D-E111-B009-485B3977172C.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/5464CD17-329D-E111-8062-5404A63886BE.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/8271BC4A-2F9D-E111-891F-003048F11C28.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/84AE016A-319D-E111-9EB2-5404A63886C0.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/C67F52B9-419D-E111-98E9-003048D37560.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/D2897664-2A9D-E111-AE88-003048D2BC52.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/DCBD02EA-399D-E111-966F-BCAEC518FF68.root",
##    "rfio:/castor/cern.ch/cms/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/193/998/E8BA354A-2F9D-E111-BEC6-0015C5FDE067.root",

# fill 2621
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/0E3C3137-179E-E111-9A06-5404A63886C5.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/0E727127-159E-E111-AC08-00215AEDFD74.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/102488EE-3C9E-E111-B275-003048D3733E.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/16F23FD3-269E-E111-90F3-0025B320384C.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/20D96233-179E-E111-9BCD-BCAEC5329702.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/261AE52F-429E-E111-8BAE-0025901D5DF4.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/2638B30A-309E-E111-8973-BCAEC518FF6E.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/268D0C20-269E-E111-9561-003048D2BD66.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/44301897-0F9E-E111-82A4-003048D374F2.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/6213E48B-119E-E111-BA26-0030486733B4.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/68790229-1C9E-E111-84F2-003048D2BC4C.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/6C1ECA8D-0F9E-E111-93BC-002481E0D7C0.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/6C301D9C-0F9E-E111-B74C-BCAEC5364C93.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/743C1AC6-1A9E-E111-9B5D-00215AEDFCCC.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/762082AC-139E-E111-816F-0025901D631E.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/7A1FF1BF-1A9E-E111-AFBA-002481E0D7EC.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/880CCD28-1C9E-E111-9664-003048D2BC38.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/92236757-149E-E111-8389-002481E0D73C.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/92E3A46D-0F9E-E111-8A45-E0CB4E553651.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/A03F677E-229E-E111-8D0C-003048D2BC4C.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/A0DEB0E1-179E-E111-A3AF-003048F1C82A.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/A6ABBAF3-0F9E-E111-90A2-001D09F2437B.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/B0C7F697-189E-E111-A9E0-003048D37538.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/B6460AFE-199E-E111-A3A2-003048D374CA.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/B8F5592F-429E-E111-BDF6-5404A63886AB.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/BE90BB55-149E-E111-95B9-003048F118DE.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/C4225BD2-0F9E-E111-A17A-BCAEC5364C42.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/CEBEC4A4-0F9E-E111-A9B1-003048D37538.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/D28804BE-1A9E-E111-90F9-0025B32035BC.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/D2B5D280-119E-E111-915D-5404A63886D2.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/DE912530-139E-E111-AAE6-0030486780B4.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/F6587C8E-519E-E111-A914-5404A63886AD.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/FEDC688B-279E-E111-9EF7-0025901D62A0.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/16A57AB6-649E-E111-9D6D-001D09F28F25.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/5457E6EE-489E-E111-81DA-5404A63886E6.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/9ADAB38B-519E-E111-B3A4-BCAEC5329717.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/A44197F4-579E-E111-8264-003048D2BE06.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/BA6CD7F5-579E-E111-A79D-003048D2BB90.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/C4A51D8C-519E-E111-A84B-5404A638868F.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/0E3ABC62-499E-E111-9BD7-485B3962633D.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/050/887E0357-5E9E-E111-99DC-0025901D5D80.root",

# fill 2663
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/00471167-82A8-E111-B467-BCAEC53296F3.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/0E6700C0-25A8-E111-93C0-003048D2C108.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/0ECA6CAF-89A8-E111-895D-5404A63886B1.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/101E6F20-5CA8-E111-8806-BCAEC5329709.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/103AD8C4-64A8-E111-8049-003048D2BC42.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/1470FC66-3CA8-E111-ABED-003048D2C01A.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/16DDF92A-66A8-E111-9C6D-001D09F23174.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/1E2AEDB0-60A8-E111-B148-5404A63886EC.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/1EFEA2DF-1BA8-E111-97F0-001D09F24353.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/22826145-2EA8-E111-8ABF-001D09F292D1.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/24E517CB-64A8-E111-8521-0030486780B8.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/264A3786-91A8-E111-A791-BCAEC518FF80.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/28C2B478-5CA8-E111-88C0-BCAEC532971E.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/2A6C9B7B-80A8-E111-8EF0-E0CB4E5536AE.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/2E4C455F-9BA8-E111-B1B0-003048D3751E.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/2EBC8E96-5FA8-E111-BF27-BCAEC532970F.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/328F7379-A2A8-E111-B1BC-001D09F297EF.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/367D5007-D2A8-E111-B369-0025901D5C88.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/3805738A-9DA8-E111-889D-003048D2C0F4.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/384D77E3-1BA8-E111-988D-0025901D5DB2.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/544C491F-5CA8-E111-A7F0-BCAEC532971D.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/54AF243E-72A8-E111-9FBA-003048F118C6.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/6418506B-41A8-E111-A313-001D09F24DA8.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/68C64FBF-36A8-E111-A588-0025901D5D7E.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/6AE7927D-80A8-E111-98D3-E0CB4E4408E7.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/722C7109-5BA8-E111-BA78-0025901D62A0.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/741E027E-3EA8-E111-B23E-BCAEC5329716.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/74E1E304-4FA8-E111-BF2D-003048D2C01E.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/76E4A9E1-97A8-E111-AD5E-003048D2C020.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/7CE53D59-35A8-E111-BE61-003048D37580.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/8E480FB0-60A8-E111-9AC3-BCAEC518FF54.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/8E98112E-36A8-E111-BB0E-5404A63886CF.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/90725E0D-5BA8-E111-A7BF-5404A6388694.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/96AC63B1-53A8-E111-B916-5404A6388698.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/98D4701A-7CA8-E111-8281-BCAEC518FF74.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/A6187646-94A8-E111-8444-001D09F29533.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/A8DAE671-61A8-E111-AAEA-0025B3203898.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/AA15F3EF-99A8-E111-AF0E-003048D2BC52.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/B035818D-98A8-E111-B805-003048D373AE.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/B0E3C7D0-3DA8-E111-B0F1-5404A63886B2.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/B20E582B-A8A8-E111-8BFE-001D09F2527B.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/B427EA9A-A9A8-E111-9E21-001D09F25460.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/B43B7388-1FA8-E111-866E-001D09F24DA8.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/BC322A0E-70A8-E111-90F8-003048D2BD66.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/C4421C8B-9DA8-E111-8FE5-00237DDBE41A.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/C4695770-65A8-E111-96DB-BCAEC53296F8.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/CADF58A6-34A8-E111-B24D-0025B32445E0.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/CC08A75D-6DA8-E111-A049-001D09F2447F.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/D08E43A4-87A8-E111-AB98-485B3962633D.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/D439CAEF-33A8-E111-B362-BCAEC518FF52.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/D6097E1A-A6A8-E111-BEEB-5404A63886A0.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/E8F99B4E-52A8-E111-905B-BCAEC53296F8.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/F2C035BD-69A8-E111-B021-003048F1BF66.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/F8A3A8A1-4CA8-E111-9481-BCAEC518FF8D.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/FC394E53-24A8-E111-A17D-003048F118C6.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/FCCD33CD-B7A8-E111-B7F1-003048D2C01E.root",
    "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/194/912/FEC82608-A1A8-E111-B4CD-BCAEC518FF50.root",

# fill 2670

## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/EA8428A6-6BAA-E111-90B6-485B3962633D.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/E20CB792-5CAA-E111-9CBC-003048D37666.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/A2DB76A2-44AA-E111-872A-5404A638868F.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/78E58267-66AA-E111-9505-BCAEC5364C4C.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/4475310E-65AA-E111-A450-BCAEC518FF52.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/40AB4A06-72AA-E111-83E5-0025901D5DB2.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/32B9AA8E-61AA-E111-B6F5-003048F118AA.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/2AAD3A6B-59AA-E111-8ED7-0025901D629C.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/1AC3A20F-6FAA-E111-9A64-BCAEC53296FB.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/06BB53B2-73AA-E111-B98A-003048F117B4.root",
## "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/099/024EFA12-60AA-E111-BA44-E0CB4E553673.root",

    
# fill 2671
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/26682BC3-EAAA-E111-8D66-002481E0D7EC.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/2627AD36-F0AA-E111-93B7-003048D2C16E.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/30AA5C32-E0AA-E111-A2E5-5404A63886B4.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/5E44AA1A-ECAA-E111-ADC1-BCAEC518FF52.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/5E7E23B0-F8AA-E111-A69F-5404A63886B9.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/7A2DAF17-E5AA-E111-ADD1-0025901D6288.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/8EEDAF3D-0DAB-E111-8D24-003048D2BE08.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/B22F7F59-E4AA-E111-B424-BCAEC518FF89.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/B2D4DCA8-F6AA-E111-BFAC-5404A638869E.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/BC8ACBB7-FCAA-E111-A853-0025901D5DB2.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/C4ABF907-00AB-E111-B9A1-0025901D626C.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/C800A10C-00AB-E111-9077-003048F11CF0.root",
##   "/store/data/Run2012B/MinimumBias/RECO/PromptReco-v1/000/195/109/EA16EF88-02AB-E111-88D7-002481E0D790.root",

  )
    
)

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124230:26-124230:9999','124030:2-124030:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('190389:40-190389:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('191271:55-191271:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('191718:30-191718:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('193621:58-193621:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('193998:63-193998:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('194050:52-194050:9999')
process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('194912:52-194912:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('194912:52-194912:330')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('195099:61-195099:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('195109:85-195109:9999')

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('h.root')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# what is this?
# process.load("Configuration.StandardSequences.Services_cff")

# what is this?
#process.load("SimTracker.Configuration.SimTracker_cff")

# needed for global transformation
# process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
process.GlobalTag.globaltag = "GR_P_V28::All"
# 2011
# process.GlobalTag.globaltag = "GR_P_V20::All"
#  process.GlobalTag.globaltag = "GR_R_311_V2::All"
# 2010
# process.GlobalTag.globaltag = 'GR10_P_V5::All'
# process.GlobalTag.globaltag = 'GR10_P_V4::All'
# OK for 2009 LHC data
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'

process.d = cms.EDAnalyzer("TestClusters",
    Verbosity = cms.untracked.bool(False),
    src = cms.InputTag("siPixelClusters"),
    Select1 = cms.untracked.int32(1),  # cut on the num of dets <4 skip, 0 means 4 default 
    Select2 = cms.untracked.int32(0),  # 6 no bptx, 0 no selection                               
)

process.p = cms.Path(process.hltPhysicsDeclared*process.hltfilter*process.d)
#process.p = cms.Path(process.hltPhysicsDeclared*process.d)
#process.p = cms.Path(process.hltfilter*process.d)
#process.p = cms.Path(process.d)


# define an EndPath to analyze all other path results
#process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
#    HLTriggerResults = cms.InputTag( 'TriggerResults','','' )
#)
#process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
