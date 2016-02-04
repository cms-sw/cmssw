import FWCore.ParameterSet.Config as cms

process = cms.Process("zptscale")
process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_1.root',
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_2.root',
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_3.root'
                                   
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/086FD387-F97F-DF11-89A1-002618943957.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E2BB3FAB-F27F-DF11-98A9-003048678AE4.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/0CC6A6C5-3C80-DF11-B35E-003048678B92.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E4D7DBB0-EC7F-DF11-B0EE-00248C0BE016.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/1896BBB8-F97F-DF11-9F2F-001A92971B48.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E8576675-EC7F-DF11-91DA-002618943856.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/64DB4BB0-F97F-DF11-98C7-002618943926.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/EA9489E6-EB7F-DF11-9875-002354EF3BCE.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/70A33058-2F80-DF11-8EAB-002618943983.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/EC9C1993-EB7F-DF11-9206-00261894391C.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/802A2887-F97F-DF11-855C-002618943886.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/ECD82E7B-F37F-DF11-B775-002618943800.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/AC8B97B2-F97F-DF11-BD20-002618FDA263.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/EE04C8B3-F37F-DF11-9BD1-003048679044.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/BC8AF7A7-F97F-DF11-9F75-003048678FA6.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/EED4B178-F27F-DF11-BE1A-002618943899.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/CEC0C35C-2180-DF11-809E-0018F3D095FA.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/F2A31B46-F27F-DF11-A9BD-002618943860.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/D4A4F191-F27F-DF11-BE80-00261894380B.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/FA0ACF88-EC7F-DF11-85EE-003048678DD6.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/DE2AED38-F27F-DF11-B867-002618943901.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/FE2F8537-9A80-DF11-B80E-0026189438C9.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E05B3CBB-F27F-DF11-8E2E-002618FDA263.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/FEC277B8-F37F-DF11-A644-00261894380D.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E0AECCD7-F27F-DF11-9601-002618FDA262.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/FEEFF5AC-F97F-DF11-A5B6-0018F3D09626.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E0E68C3A-ED7F-DF11-8D04-003048678BE6.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E224964B-ED7F-DF11-B36B-00261894387E.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E2549BBA-F37F-DF11-BEA1-00261894382D.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E289C8C2-F27F-DF11-A41B-0018F3D0965A.root',
'file:/scratch2/users/fabozzi/summer10/zmmPowheg_cteq66/E28AE239-F27F-DF11-9AA3-00261894398A.root',


  
                                   )
 )


process.evtInfo = cms.OutputModule("AsciiOutputModule")


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('ZptScale_100K_1xcent.root')
)



process.zHistos = cms.EDAnalyzer("ZMuPtScaleAnalyzer",
    genParticles = cms.InputTag("genParticles"),
    nbinsMass=cms.untracked.uint32(200),
    nbinsPt=cms.untracked.uint32(200),
    nbinsAng=cms.untracked.uint32(200),
    massMax =  cms.untracked.double(200.),
    ptMax=  cms.untracked.double(200.),
    angMax = cms.untracked.double(6.),
    #parameter for the geometric acceptance
    accPtMin = cms.untracked.double(20.0),
    accMassMin = cms.untracked.double(60.0),
    accMassMax = cms.untracked.double(120.0),
    accMassMinDen = cms.untracked.double(60.0),
    accMassMaxDen = cms.untracked.double(120.0),                             
    accEtaMin = cms.untracked.double(0.0),
    accEtaMax = cms.untracked.double(2.1),
    muPdgStatus = cms.untracked.int32(1),                                 
    # scaling of 1%                             
    ptScale = cms.untracked.double(0.006)
    
)






process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)
                                                  

                                                  
process.maxEvents = cms.untracked.PSet(
  input =cms.untracked.int32(-1)
)

process.path=cms.Path(process.zHistos)

process.end = cms.EndPath(process.evtInfo )



