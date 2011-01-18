import FWCore.ParameterSet.Config as cms

process = cms.Process("zptscale")
process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_1.root',
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_2.root',
##    'rfio:/castor/cern.ch/user/d/degrutto/MCatNLOzToMuMum20/dimuons_3.root'
                                   
 'file:../../WReco/test/MSTW2008nnlo68clpdfAnalyzer_100KEvts.root',
  
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
    accEtaMin = cms.untracked.double(0.0),
    accEtaMax = cms.untracked.double(2.0),
    # scaling of 1%                             
    ptScale = cms.untracked.double(0.02)
    
)






process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)
                                                  

                                                  
process.maxEvents = cms.untracked.PSet(
  input =cms.untracked.int32(100000)
)

process.path=cms.Path(process.zHistos)

process.end = cms.EndPath(process.evtInfo )



