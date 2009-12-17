import FWCore.ParameterSet.Config as cms
process = cms.Process("eventntuplizer")
process.load("FWCore.MessageService.MessageLogger_cfi")

from RecoJets.Configuration.GenJetParticles_cff import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

process.chargeParticles = cms.EDFilter("GenParticleSelector",
    filter = cms.bool(False),
    src = cms.InputTag("genParticles"),
    cut = cms.string('charge != 0 & pt > 0.500 & status = 1')
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.source = cms.Source("PoolSource",
    #inputCommands = cms.untracked.vstring("keep *", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap_*_HLT"),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    fileNames = cms.untracked.vstring(
#run 123596 only reRecoed BSCNOBEAMHALO




        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/F633052B-71EA-DE11-9FC4-0024E8768BFC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/D85C661E-71EA-DE11-8262-0024E8768C98.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/B8FFF344-71EA-DE11-AF62-00151796C12C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/A6B4A077-71EA-DE11-A8EA-00151796C12C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/8CE16D3D-71EA-DE11-A469-0024E8768C98.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/7E18573F-71EA-DE11-9360-0024E876A7FA.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/729AD036-71EA-DE11-8D6C-0024E8768CB2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/643BDC4C-71EA-DE11-B02C-00151796D7F4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/46CBED55-71EA-DE11-9B7A-0024E8768446.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/44F3581F-71EA-DE11-9CA5-0024E876A7FA.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/242B6569-71EA-DE11-9BF2-0024E876994B.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/2274401A-71EA-DE11-B10F-0024E8769B60.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/0654E33A-71EA-DE11-B9F7-0024E876841F.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0102/04CED921-71EA-DE11-9430-0024E8768446.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/FC22D500-31EA-DE11-A37E-0024E8768258.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/FAD9AA20-40EA-DE11-A44E-0024E8768C3D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/FA6FF0A0-20EA-DE11-996D-001D0967DA17.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F8B09789-1FEA-DE11-B2DA-00151796C138.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F87700B8-25EA-DE11-A88A-0024E8767D11.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F8635F0A-38EA-DE11-AD56-00151796D508.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F6F75E4F-34EA-DE11-BFD2-0024E8767DA0.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F6AC319A-25EA-DE11-B008-0015178C0198.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F6A6B6F9-2DEA-DE11-B4AD-001D0967DE45.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F65A7D01-2EEA-DE11-AE31-00151796C114.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F4DDF0E0-31EA-DE11-9001-001D0967DF53.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F4AAACC2-2BEA-DE11-8721-00151796D6D4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F454A2F8-2DEA-DE11-86FD-001D0967D37D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F0F6010A-2EEA-DE11-9D11-00151796D824.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F08C5F84-29EA-DE11-B20B-0024E8766393.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/F043B2F8-2DEA-DE11-B01B-0024E8768C23.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/EEB93BDA-26EA-DE11-AD50-00151796D80C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/EEB252F1-31EA-DE11-9937-001D0967DC42.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/EE1129F6-2EEA-DE11-8609-001D0967CE50.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/EACA0203-2FEA-DE11-9A3A-00151796C0E8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/EA6D5105-31EA-DE11-B427-001D0967DF12.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E83D8951-52EA-DE11-B859-00151796D9A8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E6978D2D-38EA-DE11-82A1-001D0967D9E5.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E690F99E-20EA-DE11-AD72-001D0967DF0D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E6338ADE-39EA-DE11-80A9-00151796D45C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E4CD2AEE-3CEA-DE11-B635-0015179EDC2C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E4B486BC-2BEA-DE11-A394-00151796C1C8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E2F161D9-39EA-DE11-91C5-0015178C4B80.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E264642D-34EA-DE11-860A-001D096B0DB4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E251FCC1-3DEA-DE11-ACC3-00151796D870.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E24E3E2D-34EA-DE11-80DB-001D096792CA.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E07114FC-27EA-DE11-AC63-00151796C184.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/E05F86F1-27EA-DE11-9BEC-001D0967DF53.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DEC34D14-31EA-DE11-BAFF-0015178C4A68.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DE88DFD3-24EA-DE11-A550-001D0968F337.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DE659A0C-36EA-DE11-A60D-0015178C4D14.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DC950BAE-45EA-DE11-878D-00151796D910.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DC7BB6EE-41EA-DE11-B3CD-0024E8766415.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DC2E4506-39EA-DE11-8C2E-001D0966E23E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DAC9E11E-36EA-DE11-A118-00151796D680.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DAAE70F9-2DEA-DE11-9FC6-001D0967DA49.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DA6F63C2-3DEA-DE11-885B-00151796D45C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/DA042E38-4FEA-DE11-955F-0024E876803E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D8B347DC-2FEA-DE11-91E3-0015178C4D14.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D8945708-31EA-DE11-8B24-0024E8768C98.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D8088B38-31EA-DE11-8547-001D0967DEA4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D672CC82-29EA-DE11-84F6-001D0967C9A0.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D4A50CD5-1FEA-DE11-B264-00151796C1E8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D4824D2A-34EA-DE11-A8F9-001D0967C0CC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D44D12F6-2DEA-DE11-B09B-00151796D6D4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D412BDF0-4FEA-DE11-86A0-00151796D6D4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D28A0C0C-29EA-DE11-9897-0024E8768C7E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/D284FBCD-32EA-DE11-B36C-001D0967DF12.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/CE9DF461-59EA-DE11-AB2E-0015178C48B8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/CC2F4E14-36EA-DE11-9902-0015178C6B04.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/CA7FC981-1DEA-DE11-A124-001D0967DF53.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/CA73063A-34EA-DE11-B5C1-00151796D5C4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/CA125BCF-32EA-DE11-9B44-00151796D534.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C8EEF1C2-32EA-DE11-BB87-0015178C48E4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C86500FD-27EA-DE11-95F2-00151796C138.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C6ED1393-25EA-DE11-8C44-00151796D79C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C6EA944E-23EA-DE11-A96C-0015178C4B94.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C6E2A981-29EA-DE11-AAF2-0024E8766415.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C6B977FF-2EEA-DE11-8DFF-00151796C1B0.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C635DDD3-41EA-DE11-889C-00151796D508.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C6043EEF-3CEA-DE11-884B-0015178C65F4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C4EA91F7-2EEA-DE11-99CB-0024E8766393.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C49CA215-35EA-DE11-A3B8-0015178C6C24.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C2D47A00-32EA-DE11-8FC9-0024E8767D11.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C22740C1-21EA-DE11-B56D-001D0967D49F.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C2152DBE-2BEA-DE11-BE21-0024E876A82E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C0B9DC40-34EA-DE11-B404-00151796C1E8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/C05D0C0F-2EEA-DE11-992C-00151796C45C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/BEEC15D1-2FEA-DE11-BEFD-001D0968F337.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/BE782F46-38EA-DE11-95EA-0024E8768C30.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/BC85A501-2EEA-DE11-9CA8-0024E876635F.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/BC796905-31EA-DE11-AA59-0015178C4BE4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/BAE0173F-34EA-DE11-B7A4-0024E86E8D8D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/BAB5DE98-25EA-DE11-A778-001D0967BC3E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/BA0DF202-28EA-DE11-A354-00151796C1CC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B8D993CE-40EA-DE11-BCB8-00151796D760.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B8612AD7-32EA-DE11-B322-00151785FF78.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B84E76D0-32EA-DE11-9852-00151796D45C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B6FA1DE9-31EA-DE11-9163-0024E8768826.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B60AD5FD-2DEA-DE11-A702-001D0967DB7A.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B2769395-1FEA-DE11-852F-0015178C4B94.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B2714FDD-39EA-DE11-A138-0024E86E8DA7.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B223EBCD-28EA-DE11-8FDA-001D096B0F99.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B0EA27EA-3CEA-DE11-992E-001D0966E1E9.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/B054A5FD-46EA-DE11-94A1-001D0967D341.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/AE96A28D-29EA-DE11-9349-0024E8768101.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/ACED9E91-1FEA-DE11-A456-0024E8769ADE.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/ACED5336-34EA-DE11-8528-00151796D80C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/ACE1C3B4-24EA-DE11-B156-00151796C12C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/AC620A93-29EA-DE11-A334-0015178C6644.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/AC09EAA3-4BEA-DE11-A88A-0015178C65F4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/AAAD5669-22EA-DE11-AD8B-00151796D544.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A8740DC6-2BEA-DE11-8CC5-00151796C478.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A83782B3-24EA-DE11-9728-001D0967D49F.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A82032B1-24EA-DE11-BEEB-001D0967D5A3.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A813BB2F-34EA-DE11-8F9B-001D0967D909.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A6DEADE0-35EA-DE11-95BF-0024E8769958.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A6D8153A-34EA-DE11-84AF-0024E876A7ED.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A6AC66C3-21EA-DE11-A659-00151796C444.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A67BB590-1DEA-DE11-8D8F-00151796C118.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A422C106-2EEA-DE11-A37B-0015178C6A88.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/A2F12830-34EA-DE11-A32A-00151796D508.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/9E968909-31EA-DE11-A7E9-001D096760DE.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/9C59B102-31EA-DE11-9FF2-001D0967D972.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/9A4665BE-21EA-DE11-BA22-00151796C1C8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/9A1F1C00-28EA-DE11-939F-001D0966E1E9.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/985614C7-24EA-DE11-AC49-0015178C6B78.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/96E466DF-28EA-DE11-BF3F-00151796D910.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/96C2B0E6-26EA-DE11-90AE-001D0967CE50.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/94690D8D-29EA-DE11-B6CE-0015178C6BD4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/9466FE30-38EA-DE11-A45E-0024E87683B7.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/94241612-36EA-DE11-8C80-001D0967D5A3.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/92AAD6CE-39EA-DE11-B432-001D0967DA6C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/924390FC-27EA-DE11-A38E-001D096792CA.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/90799708-31EA-DE11-AF5E-00151796D79C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/8C7E06B7-2BEA-DE11-970D-001D0967E061.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/8C2C600E-31EA-DE11-AFB9-00151796C1CC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/8A77CE99-25EA-DE11-83AF-0024E8769ADE.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/8A2486D0-39EA-DE11-83DD-001D0967D0DF.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/88D6AA69-35EA-DE11-89F7-0024E876A82E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/88CB140B-35EA-DE11-B79E-0015178C696C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/86EFC188-29EA-DE11-B3C6-00151796C1B4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/86C2E98E-1EEA-DE11-A9DF-001D0967CFA9.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/865863CC-21EA-DE11-9E87-0015178C0218.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/8654D661-22EA-DE11-9A64-00151796D9EC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/843E95CD-28EA-DE11-9166-001D0967D9E5.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/82BC24D1-32EA-DE11-BF82-0024E8766415.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/82AF10D9-28EA-DE11-AA1A-0015178C646C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/80FFF27F-29EA-DE11-A939-001D0967DA17.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7EFEEBE5-31EA-DE11-B039-0024E876A7D3.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7CFEF1B5-2BEA-DE11-B0F9-0015178C6860.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7C6DEA07-40EA-DE11-BACD-001D0967BC3E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7C34EEBF-2BEA-DE11-A630-001D0967D91D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7AD8D1B7-2BEA-DE11-B082-00151796D45C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7A965E1C-36EA-DE11-A669-00151796C1A8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7A2C7312-36EA-DE11-BE2E-00151796D4B4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/7A0C929F-45EA-DE11-9B6C-0024E86E8D18.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0101/78F3ADB0-24EA-DE11-A165-0024E8768272.root',
        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-Dec14thSkim_v1/0100/046BAE13-18EA-DE11-9CDD-001D0967DA26.root'



#    '/store/relval/CMSSW_3_3_3/RelValProdMinBias/GEN-SIM-RECO/MC_31X_V9-v1/0003/F89A408C-1FD2-DE11-9617-001D09F241B9.root'

##~~> MC 10XAPE
#'file:/data2/lucaroni/Tracks/CMSSW_3_3_4/src/RecoTracker/TrackProducer/test/MCwith10APE/MCwithLargeAPEVertex.root'
#~~>test newReReco
#'/store/data/BeamCommissioning09/MinimumBias/RECO/rereco_GR09_P_V7_v1/0099/AC65FE60-D5E2-DE11-857A-002618943954.root'
#MC a la Andrea
#'rfio:/castor/cern.ch/user/l/lucaroni/PerLuca/MCwithVertexDATA.root'

##~~> from 123592 to 123615:
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/FCEF8EA6-22E5-DE11-82BE-0026189438F8.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/F66B5C9A-22E5-DE11-88D1-0026189438B5.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/ECE9B13A-22E5-DE11-B670-00261894386C.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/C4F58276-22E5-DE11-96BC-00261894386D.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/745F3357-22E5-DE11-A666-0026189438B5.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/6C216467-22E5-DE11-9CCB-0026189438B5.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/68F3B885-22E5-DE11-88B7-00261894386C.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/687B33A1-22E5-DE11-9674-002618943896.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/60463596-22E5-DE11-A9F4-002618943920.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/5EEE0F8E-22E5-DE11-9759-002618943920.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/5C1FE043-22E5-DE11-8ACB-002618943920.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/5A8A4A57-22E5-DE11-81B0-0026189438B5.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/4EF5D17A-22E5-DE11-8405-00261894386D.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/289FCB66-22E5-DE11-89DC-002618FDA265.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/1A202DCE-1DE5-DE11-A176-002618943985.root'

#run 123592
##~~> vinceChiochia
#'rfio:/castor/cern.ch/user/c/chiochia/09_beam_commissioning/BSCskim_123592_Express_bit40-41.root'

#run 123596
##~~> gPetrucciani
#  'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-123596-stdReco-fitBS123592v2-lumi1-68.root',
#  'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-123596-stdReco-fitBS123592v2-lumi130-143.root',
#  'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-123596-stdReco-fitBS123592v2-lumi69-129.root'

  #  'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-run123596-lumi_68_129.root',
  #  'rfio:/castor/cern.ch/user/g/gpetrucc/900GeV/DATA/bit40-run123596-lumi130_143.root'

##~~> MC no Bfield TOB-only
    #     '/store/caf/user/gpetrucc/MinBias900GeV_NoField_v2/ReReco_TOB_Only/36a1eece1b40b34e467e1583425a6ef2/MinBias900GeV_NoField_v2_TOBONLY_1.root'    

##~~> MC startup
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/E4B3A7BE-3AD7-DE11-9230-002618943939.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/E4590360-4CD7-DE11-8CB4-002618943896.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/AAE638A2-3BD7-DE11-B6A8-002618943944.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/A0AFB73F-38D7-DE11-82F3-0026189438B4.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/702846CC-38D7-DE11-8C11-0026189438E2.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/5A2ACAFC-47D7-DE11-9A9F-002618943935.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0004/BEDB1206-36D7-DE11-9B65-002354EF3BE4.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0004/72A7CB53-23D7-DE11-A681-00261894389E.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0004/2A33F2AE-17D7-DE11-9668-003048679030.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0003/FE95B543-06D7-DE11-B3A1-00261894385A.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0003/F2C002D3-0FD7-DE11-A0D8-003048678D78.root',
#       '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0003/AC9866AB-0FD7-DE11-9446-0018F3D0966C.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0003/9AF119CE-0BD7-DE11-BC1D-003048678B86.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0003/8A220D47-06D7-DE11-A3A2-002618FDA279.root',
#        '/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0003/80995B54-06D7-DE11-A444-002354EF3BE2.root'

##DESIGN
   #  '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/FEC95427-3BD7-DE11-815F-0026189438B4.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/FCBE4955-39D7-DE11-9BC1-0026189438EB.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/FA34002A-3AD7-DE11-A9AF-002354EF3BCE.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F8D72361-39D7-DE11-9F81-002618FDA216.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F868EF3C-38D7-DE11-ACE0-0026189438E7.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F6B96629-3AD7-DE11-ABE3-0026189438EB.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F69590BB-38D7-DE11-97D3-002618943865.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F6493DB5-38D7-DE11-B68A-0026189437FA.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F62032BA-3AD7-DE11-828C-002618943901.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F4A702B3-37D7-DE11-846C-002354EF3BCE.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/F04973B4-3BD7-DE11-B601-00248C0BE005.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EE3CB7C3-38D7-DE11-8576-0026189438B4.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EE2351B7-37D7-DE11-90F9-002618943832.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EE0B8228-3AD7-DE11-8A90-002618943963.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EC7B5E53-38D7-DE11-87E6-00261894391D.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EC5E5B3C-38D7-DE11-ABA6-0026189438E7.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EC14B15C-39D7-DE11-8684-002354EF3BE1.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EA555039-38D7-DE11-B156-002354EF3BDE.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EA3D703C-37D7-DE11-891C-002354EF3BDD.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/EA21A524-3BD7-DE11-A4BB-002354EF3BDE.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/E4DEE8BD-38D7-DE11-A716-0026189437FA.root',
   #     '/store/mc/Summer09/MinBias/GEN-SIM-RECO/DESIGN_3X_V8A_900GeV-v1/0005/E2E67BC0-38D7-DE11-863A-0026189437FA.root'
#BeamHalo
 #   '/store/mc/Summer09/TkBeamHalo/GEN-SIM-RECO/STARTUP31X_V4-v1/0008/66A44C19-4EC5-DE11-BE13-001F29C9E4D0.root',
 # '/store/mc/Summer09/TkBeamHalo/GEN-SIM-RECO/STARTUP31X_V4-v1/0000/78BCFC4A-5EB3-DE11-80EB-001F29C6E900.root'
    )
)

process.TFileService = cms.Service(
    "TFileService",
    ##fileName = cms.string("ntuplizedEvent.root")
    ##fileName = cms.string("ntuplized_ReRecoBSCNOBEAMHALO.root")
    #fileName = cms.string("ntuplized_AllRuns.root")
    fileName = cms.string("ntuplized_thebigone.root")
 )

process.MyAnalyzer = cms.EDAnalyzer("Event_Ntuplizer",
    particleCollection = cms.InputTag("generalTracks","","RECO"),
    vertexCollection   = cms.InputTag("offlinePrimaryVertices","","RECO"),
    pixelClusterInput=cms.InputTag("siPixelClusters"),
 
      ##particleCollection = cms.InputTag("TrackRefitter1"),
      ##vertexCollection   = cms.InputTag("offlinePrimaryVertices","","Refitting"),
      ##vertexCollection   = cms.InputTag("offlinePrimaryVertices","","REVERTEX"),
      #particleCollection = cms.InputTag("generalTracks","","EXPRESS"),
      #vertexCollection   = cms.InputTag("offlinePrimaryVertices","","EXPRESS"),
      #particleCollection = cms.InputTag("generalTracks","","TOBONLY"),
      #vertexCollection   = cms.InputTag("offlinePrimaryVertices","","TOBONLY"),
    
    #BEAMHALO
    #particleCollection = cms.InputTag("ctfWithMaterialTracksBeamHaloMuon","","RECO"),
    #vertexCollection   = cms.InputTag("","","RECO"),

    #CaloJetCollectionName = cms.InputTag("sisCone5CaloJets"),
    #L1TechPaths_byBit = cms.vint32(40,41),
    #L1TechComb_byBit = cms.string("OR"),#must be -> "OR","AND"
    genEventScale = cms.InputTag("generator"),
    OnlyRECO = cms.bool(True)                               

)


process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

process.L1T1=process.hltLevel1GTSeed.clone()
process.L1T1.L1TechTriggerSeeding = cms.bool(True)
process.L1T1.L1SeedsLogicalExpression = cms.string('(0) AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)')






process.p = cms.Path(process.L1T1*process.MyAnalyzer)
