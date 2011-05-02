import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkDQM")
process.load("DQM.Physics.ewkMuDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/Physics/EWK/Muon')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
# Run 163255 (2011) example for testing
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/E019E9CD-306E-E011-96EC-003048F117EA.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/E28DA1AB-336E-E011-9D73-003048F118D4.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/E2F5804A-396E-E011-90D4-003048F118C2.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/E822E237-266E-E011-882F-003048D2BC30.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/EAB11F46-216E-E011-803E-0019B9F7312C.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/EC1D9839-926E-E011-A034-003048D2C16E.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/EE60DCAA-2E6E-E011-858A-003048D2BB58.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/EEEFC64F-5B6E-E011-9A41-001D09F2AD7F.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/F05E5915-246E-E011-8EF3-000423D9A212.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/F27EFA3C-7F6E-E011-9F24-0030487C90EE.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/F6D03B2D-526E-E011-AD5E-001617C3B5D8.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/FC65C71E-716E-E011-8CD5-001D09F254CE.root",
#"rfio:/castor/cern.ch/cms/store/data/Run2011A/SingleMu/AOD/PromptReco-v2/000/163/255/FCFCA383-206E-E011-9E9D-003048F1110E.root"

# 4_3_X RelVal for testing
# Reprocessing of 2010B data (preselected Z & W "golden" events)
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F0098BD7-A971-E011-BD67-001A9281173E.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/E0AA1CD3-A971-E011-B496-002354EF3BDF.root",   
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/E429E0D2-A971-E011-93FE-001A9281170C.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/E6217071-A971-E011-99CE-0030486790B8.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/E65670D7-A971-E011-AC5D-00304867920C.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/EA3E3040-A971-E011-8463-001A9281170C.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F0098BD7-A971-E011-BD67-001A9281173E.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F046C3EE-A971-E011-96DA-003048678A6C.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F437CCD7-A971-E011-90CE-00304867924E.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F4B963D7-A971-E011-A1E9-0018F3D09676.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F6C202D3-A971-E011-9F61-002618943870.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F82893DA-A971-E011-B652-0026189437E8.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F862194D-A971-E011-9186-003048678B3C.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/F8DD8CEF-A971-E011-91EA-0018F3D096E6.root",
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/Mu/RECO/GR_R_43_V1_RelVal_wzMu2010B-v3/0048/FCCE47E0-A971-E011-B7E5-001A928116AE.root"

# W RelVal:
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre1/RelValWM/GEN-SIM-RECO/MC_42_V7-v1/0048/6AA13071-0859-E011-A86A-0018F3D09688.root"
"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_3_0_pre3/RelValTTbar/GEN-SIM-DIGI-RECO/MC_43_V1_FastSim-v3/0043/F2E7E93E-F570-E011-8E10-001A92971B0E.root"

)
)
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            threshold = cms.untracked.string('DEBUG')
           #threshold = cms.untracked.string('ERROR')
    )
)
#process.ana = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.ewkMuDQM+process.dqmSaver)

