import FWCore.ParameterSet.Config as cms

process = cms.Process("JETCOMP")
process.MessageLogger = cms.Service("MessageLogger")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/061C7BFF-E633-DD11-B91B-001617E30F56.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/0AE84C7B-E733-DD11-9DFE-001617DC1F70.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/0E440D46-E833-DD11-BD5C-001617E30CD4.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/0E8ACF4D-E733-DD11-94CD-001617C3B6C6.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/1E4C2A80-E733-DD11-91D3-001617DBD224.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/2CB2CDB8-E733-DD11-8EAB-000423D985B0.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/32C45457-EB33-DD11-ABC5-001617E30D0A.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/34F0C04C-E833-DD11-930F-000423D9939C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/424438F9-E933-DD11-9063-001617DBD5AC.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/4E9F0076-E833-DD11-A47F-001617E30E2C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/5876057C-E833-DD11-BE9C-000423D6CA6E.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/5CE49533-E833-DD11-9D98-000423D6B48C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/5E6C4293-E833-DD11-B9B2-000423D94700.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/5EE759BB-E733-DD11-8C22-000423D9863C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/621A19E8-E733-DD11-8653-001617DBD5B2.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/62E9F139-E933-DD11-B309-000423D9863C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/62F658B8-E733-DD11-BC9B-000423D9989E.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/648418E3-EF33-DD11-BC34-000423D986A8.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/6E41EEA2-E833-DD11-9DA5-001617E30F4C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/78E8FF0A-F033-DD11-9786-000423D6B48C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/7C0F9E4E-E833-DD11-BDDD-000423D6B48C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/7C9976C4-E733-DD11-8042-001617DBCF1E.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/84CD22F0-E733-DD11-9D16-001617E30CE8.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/8858529A-E833-DD11-B778-001617DBD230.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/8A5A2E9C-E833-DD11-B78B-001617C3B77C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/909DA5B4-E733-DD11-BF0C-000423D98E54.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/90A232C9-EB33-DD11-9F41-000423D9863C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/9A4424B7-E733-DD11-8C79-001617E30F4C.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/9CCD8B4F-E933-DD11-989A-000423D98AF0.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/9E74E877-E833-DD11-B80B-001617DBCF1E.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/A4E561B5-E733-DD11-822A-001617DBD224.root', 
        '/store/relval/2008/6/6/RelVal-RelValQCD_Pt_80_120-1212531852-IDEAL_V1-2nd-02/0000/A8FCCD3B-E933-DD11-ADC4-000423D986A8.root')
)

process.jetComp = cms.EDAnalyzer("JetComparison",
    MinEnergy = cms.double(50.0),
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('fastjet50-120_full.root')
)

process.p = cms.Path(process.jetComp)


