import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigPhase2Prod")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("L1Trigger.DTPhase2Trigger.dtTriggerPhase2PrimitiveDigis_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
#process.GlobalTag.globaltag = "90X_dataRun2_Express_v2"
process.GlobalTag.globaltag = "80X_dataRun2_2016SeptRepro_v7"

#Calibrate Digis
process.load("Phase2L1Trigger.CalibratedDigis.CalibratedDigishlt_cfi")
#process.CalibratedDigis.flat_calib = 0 #turn to 0 to use the DB  , 325 for JM and Jorge benchmark

#DTTriggerPhase2
process.load("L1Trigger.DTPhase2Trigger.dtTriggerPhase2PrimitiveDigishlt_cfi")
process.dtTriggerPhase2PrimitiveDigis.pinta = True

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/00420561-F145-E711-9694-02163E01A23B.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/008F5CE5-F045-E711-99C2-02163E01A1F0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/00CD2DAE-A045-E711-9FE6-02163E013611.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/0268ACB3-9845-E711-855E-02163E019CA9.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/02C331DE-9245-E711-8126-02163E01A2BB.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/04D4FB88-8845-E711-932D-02163E014626.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/0805908A-8B45-E711-A5F8-02163E019CC0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/08ACE77B-8745-E711-BC01-02163E011A70.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/0CB45EF2-8845-E711-B06A-02163E011B22.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/0ECDEACD-1946-E711-AC7C-02163E01A2F0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/149B2E0B-9845-E711-96F6-02163E01A5C0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/1613329C-F245-E711-AFC2-02163E013522.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/16BA879D-8B45-E711-8C90-02163E01A4FD.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/1C2A7911-F145-E711-ADD2-02163E01A777.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/1C6C75C8-9245-E711-A7F8-02163E0138F9.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/22A6F66D-9045-E711-9144-02163E01A790.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/247F04F7-F045-E711-BECB-02163E01A593.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/264870FB-9245-E711-AE3C-02163E014735.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/264D8070-9045-E711-BB01-02163E01A621.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/2A9B2909-9345-E711-9375-02163E01A777.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/2C201C9D-9945-E711-AEF4-02163E01A6D4.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/2EC23D5F-F345-E711-85B5-02163E0145C5.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/30654C9D-F245-E711-A876-02163E014337.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/32C68A5E-9045-E711-B0D9-02163E0143FC.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/36A838EC-F045-E711-84C3-02163E01287D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/38CE468B-8A45-E711-AF16-02163E01A25D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/3A42CDB8-9245-E711-96FA-02163E011C45.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/3C3F83FE-9745-E711-87BD-02163E01A52D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/3E52D6EB-9245-E711-8301-02163E011F2A.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/421C7068-8645-E711-90C4-02163E019D31.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/42671281-8A45-E711-8DD2-02163E011F94.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/44ABB70D-8B45-E711-8118-02163E013492.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/46D807B3-F245-E711-AF4F-02163E0129DC.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/484A9143-9745-E711-877E-02163E0142F4.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/48631851-F345-E711-9C68-02163E011F14.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/4C2D7773-9045-E711-B489-02163E019E1C.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/4C5514D4-9245-E711-A467-02163E0136F8.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/54595BC7-8345-E711-9CBF-02163E01356C.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/54FED25C-9A45-E711-A853-02163E01A722.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/58B02EF4-F045-E711-AA1A-02163E019BAF.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/5A047F45-9545-E711-AFB0-02163E01A3B8.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/5CD6A44B-9545-E711-984B-02163E01A4E8.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/5EC54124-9345-E711-9204-02163E0144C6.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/6052277B-8645-E711-AE08-02163E0139CF.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/60893844-9345-E711-9071-02163E0134A0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/60AEFC2C-F345-E711-A449-02163E011DD0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/622327E3-8C45-E711-9020-02163E0142E5.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/624E5E69-8645-E711-86D5-02163E01A54D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/62F5EF58-9045-E711-8906-02163E01396B.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/64337DD5-F045-E711-A368-02163E01A1CC.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/64A36DC9-9745-E711-92FC-02163E014697.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/64C26447-9045-E711-80CC-02163E011D0D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/66F5DB6C-9B45-E711-9D7C-02163E0125AA.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/6A52EA76-8645-E711-9C8D-02163E014487.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/6AD10C06-B045-E711-AEEE-02163E0144C8.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/6CA3480E-9345-E711-82C1-02163E014468.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/6EF25BF0-9245-E711-9E1D-02163E01A212.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/72CA08B9-8745-E711-8EDC-02163E014554.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/72F405A2-8B45-E711-9BA0-02163E01289A.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/72FB7408-EB45-E711-87EA-02163E019CB3.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/7AE9D1A3-9845-E711-ABF1-02163E019DAE.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/7C99B269-8645-E711-A4FC-02163E0143FC.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/7ECBD832-9A45-E711-8CE9-02163E01A671.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/7ECC660A-9345-E711-9254-02163E014441.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/80111D94-F245-E711-9BA3-02163E011C94.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/80C58183-9045-E711-86AF-02163E011C14.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/82569C1C-9545-E711-8BE3-02163E01A3C0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/82574F77-9045-E711-926D-02163E01A4B0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/82B235BD-9A45-E711-97A2-02163E014443.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/82DCBF80-9645-E711-B5F2-02163E01A31C.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/842D4F2B-9145-E711-A656-02163E01262D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/84F73D42-F345-E711-A715-02163E01A3A6.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/8C914937-F145-E711-98BA-02163E01A654.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/90DF9E9D-F245-E711-9DFD-02163E01185E.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/924A6455-9A45-E711-B61F-02163E0142FD.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/98917E89-8B45-E711-92BF-02163E019C72.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/9E14A79B-8B45-E711-A18F-02163E019C78.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/9E7A2B9D-8B45-E711-A60A-02163E01A57B.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A011E17E-F145-E711-A6A3-02163E01A366.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A2878D5C-9045-E711-8281-02163E01A56F.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A2E77817-8F45-E711-87BC-02163E01A3D0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A2EEE768-9045-E711-8F14-02163E01414E.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A2F78E45-9545-E711-AFE9-02163E019C94.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A43D908F-A245-E711-8A44-02163E0122A1.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A4AA3EF2-8845-E711-8AA8-02163E01A1FA.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A6AF0A9A-8B45-E711-B662-02163E01A5A0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A6B967EF-9245-E711-B138-02163E014529.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/A8E24EDD-8D45-E711-AAE3-02163E0128C8.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/AA4A365A-8845-E711-AB5A-02163E01A488.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/AA4C6878-8645-E711-9054-02163E019D26.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/AE1C5D03-9C45-E711-976A-02163E01A57D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/AEB1E499-F245-E711-BE8F-02163E01A570.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B061BF21-8945-E711-B2A8-02163E01A79D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B07C163F-9545-E711-A600-02163E01A403.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B0919986-8845-E711-90B6-02163E011A0C.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B246A0C3-8345-E711-8F54-02163E014160.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B2C5ABCF-9245-E711-B341-02163E0145E3.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B48BFA97-8B45-E711-A314-02163E019BE1.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B4F6911F-9D45-E711-9913-02163E01A2B9.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B6F4B3F3-9245-E711-A639-02163E013393.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/B8A5D954-9545-E711-9EDE-02163E0126EA.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/BA453A17-9945-E711-908E-02163E019D0D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/BABD6CB6-AE45-E711-80F8-02163E0140E4.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/BC8F4CB2-8B45-E711-97D1-02163E019B27.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/BCE79033-0946-E711-A068-02163E01A34F.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/BE3C6E6A-F345-E711-A4AE-02163E019BF2.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/C0E5FFDF-9145-E711-BA36-02163E01A52D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/CCC58404-8945-E711-9496-02163E01A52B.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/CE1B0ED3-9245-E711-B42C-02163E0135FA.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/CE875571-9B45-E711-8BF9-02163E019C33.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/CE8A8144-F345-E711-BE57-02163E01A3E3.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/CE9DF36D-9045-E711-A49F-02163E011899.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/CECC244F-9045-E711-8D74-02163E011B8A.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/D02C2F11-9945-E711-A248-02163E019D9C.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/D0831B9F-8B45-E711-AAAC-02163E0133AD.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/D236BF82-8645-E711-8418-02163E019C0C.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/D49A5158-9045-E711-98A6-02163E011E48.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/D6EA1115-8F45-E711-96E3-02163E019B33.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/D83A4DE3-8845-E711-9167-02163E01A2C0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/DA0886D6-F045-E711-8187-02163E0139B6.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/DA08A847-9045-E711-BEC0-02163E019BCE.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/DC33E799-8B45-E711-8D35-02163E011E08.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/DC498541-F345-E711-8046-02163E01A2DF.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/DE64771C-9145-E711-850E-02163E011870.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/E25203CB-F045-E711-8613-02163E01A6C6.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/E4B69960-9645-E711-BE94-02163E01410C.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/E6BCC809-9345-E711-9D9E-02163E019D8D.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/E83934DA-9145-E711-A253-02163E01423E.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/E8D4CC92-9645-E711-9473-02163E01202F.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/EA10491B-9545-E711-88F2-02163E0145BD.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/EA1FD173-9645-E711-B8CB-02163E019E21.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/EA8C3332-8B45-E711-AF00-02163E0146D8.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/EE4A5B5C-8645-E711-841E-02163E019B66.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/F0063FE1-EB45-E711-BB9E-02163E01A50B.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/F01F5A55-F345-E711-B696-02163E01351B.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/F0E230BA-9A45-E711-B4AC-02163E011C87.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/F4AACC18-8945-E711-A263-02163E0139D0.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/F4C2A86D-8645-E711-B9A0-02163E012147.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/F8B12CAE-EC45-E711-A114-02163E014522.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/F8BC99EB-F045-E711-BD92-02163E01A2E3.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/FABE9AB7-F245-E711-8DF3-02163E019DAD.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/FCBE1392-9845-E711-8F7A-02163E01A72A.root',
        'file:/eos/cms/store/data/Run2017A/RPCMonitor/RAW/v1/000/295/655/00000/FCDFCBC3-9245-E711-93D0-02163E01A2CF.root'
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
#    input = cms.untracked.int32(100000)
)

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *'),
                               fileName = cms.untracked.string('/tmp/carrillo/output.root')
                               )




process.p = cms.Path(process.CalibratedDigis*process.dtTriggerPhase2PrimitiveDigis)
#process.this_is_the_end = cms.EndPath(process.out)






