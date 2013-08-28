import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQM/L1TMonitor/L1TCSCTF_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

#off-line
process.GlobalTag.globaltag = 'GR10_P_V2::All'

#ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GLT 
process.l1tcsctf.disableROOToutput = False
process.l1tcsctf.outputFile = 'cscTest.root'
process.l1tcsctf.verbose = True
process.l1tcsctf.statusProducer = 'csctfDigis'
process.l1tcsctf.trackProducer = 'csctfDigis'
process.l1tcsctf.lctProducer = 'csctfDigis'
process.l1tcsctf.mbProducer = 'null'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring(
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/147D57F8-8224-DF11-8BBC-000423D98EC4.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/14C18A56-7624-DF11-8A38-000423D8FA38.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/1691940B-7724-DF11-B897-000423D98834.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/1835E376-7624-DF11-976E-000423D98FBC.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/184168DB-8024-DF11-A778-0030487A3232.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/18503A25-7924-DF11-9FA4-000423D996B4.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/189D7270-7124-DF11-9FA2-000423D94990.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/1A241990-8124-DF11-81EB-000423D99EEE.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/1A577E56-7D24-DF11-BAAA-0030487CD7C6.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/1A882C0D-7E24-DF11-8E69-0030487C6090.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/1C2FC121-7224-DF11-8E1E-000423D99160.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/1E9AA2F9-8224-DF11-91E2-0030487CD14E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/2253AB06-7E24-DF11-B6CF-0030487CAEAC.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/22CA416F-7124-DF11-83F9-000423D98C20.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/24063722-7224-DF11-A508-000423D94E70.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/240B7438-7424-DF11-8D17-000423D99614.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/240FCF84-7324-DF11-9688-000423D98844.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/244F50D3-7224-DF11-A924-0030487C635A.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/262C8EA2-7C24-DF11-96C0-0030487CD6D2.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/26489A24-8724-DF11-9B99-000423D99F3E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/26745AA4-7524-DF11-B164-0030487CD700.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/280FFDF1-7424-DF11-9F6B-000423D99160.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/282C1A0B-7724-DF11-BE4F-000423D98B6C.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/2A2C670A-7E24-DF11-8723-000423D999CA.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/2A5B4FD9-7924-DF11-9CB6-0030487CD812.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/2ACA0D27-7224-DF11-9A37-0030487A322E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/2C7C0378-8624-DF11-8E7E-0030487CD17C.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/2CCC3A6F-7124-DF11-940D-000423D9989E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/2E5BC13C-8224-DF11-A836-0030487C90D4.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/327DFA41-8224-DF11-8EC1-0030487CD14E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/32AE54A5-7524-DF11-8201-0030487C8E02.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/32DC41A3-7524-DF11-A8A5-00304879FC6C.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/348250DC-8024-DF11-93DD-0030487A3DE0.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/363C93F9-8224-DF11-8232-0030487D05B0.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3660941C-7224-DF11-8113-000423D98E54.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/384A0A24-7224-DF11-9B98-0030487C635A.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/38571BB7-7024-DF11-A8B3-000423D99614.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3A169823-8024-DF11-82AE-000423D94C68.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3A611AC1-7E24-DF11-AEEB-0030487A3232.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3A883D70-7124-DF11-9566-0030487A322E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3AFB1D25-8724-DF11-A32E-000423D60FF6.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3C4AAC23-7224-DF11-BCC0-0030487CD14E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3CBA4B3B-8224-DF11-B658-000423D99264.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3CD22F22-7224-DF11-B8CD-000423D98920.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3E12CAA1-7524-DF11-8D0C-000423D99896.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3E430DED-7424-DF11-98B4-000423D99BF2.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3E68B60B-7E24-DF11-8B39-000423D94E1C.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/3E9685E3-8024-DF11-836F-0030487CD906.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/40A7228A-7A24-DF11-8662-0030487C8CBE.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/40AB1B0A-8524-DF11-9ABB-000423D98834.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/40B01021-7924-DF11-B799-000423D98B6C.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/40DC1EA2-7C24-DF11-AFD6-0030487C6A66.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/42D695DA-8024-DF11-A4A0-000423D98AF0.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/42E97D74-7824-DF11-8EED-0030487A18D8.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/4453B90A-7724-DF11-A1B8-000423D98844.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/44673740-8224-DF11-A429-000423D95030.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/44BDD384-7324-DF11-9A4F-000423D98EC8.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/46B91B71-7824-DF11-BB38-0030487CD184.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/487F1456-7D24-DF11-873C-0030487CD840.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/4A44FF23-8024-DF11-858A-00304879FC6C.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/4CA510F2-7424-DF11-92D0-000423D94E70.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/4E4E831D-7224-DF11-8567-000423D33970.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/4E6C10BD-7724-DF11-8D04-000423D9989E.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/4E9178DB-8024-DF11-939F-0030487CD13A.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/4EF1A2C1-7E24-DF11-A70C-0030487CD7EA.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/50623342-8224-DF11-B072-0030487C8CB6.root',
		'/store/express/Commissioning10/ExpressPhysics/FEVT/v3/000/129/468/50E405D8-7924-DF11-A93B-000423D952C0.root'
	)
)

process.p = cms.Path(process.RawToDigi*process.l1tcsctf)
#process.outputEvents = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('testdemon.root')
#)
#process.ep = cms.EndPath(process.outputEvents)
#process.s = cms.Schedule(process.p,process.ep)
