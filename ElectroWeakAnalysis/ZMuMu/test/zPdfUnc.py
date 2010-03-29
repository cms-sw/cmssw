import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("zpdfsys")

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(-1)
    input = cms.untracked.int32(10000)
)


## process.source = cms.Source("PoolSource",
##     debugVerbosity = cms.untracked.uint32(0),
##     debugFlag = cms.untracked.bool(False),
##     fileNames = cms.untracked.vstring()
## )
## import os
## dirname = "/scratch1/cms/data/summer08/Zmumu_M20/"
## dirlist = os.listdir(dirname)
## basenamelist = os.listdir(dirname + "/")
## for basename in basenamelist:
##             process.source.fileNames.append("file:" + dirname + "/" + basename)
## print "Number of files to process is %s" % (len(process.source.fileNames))

 
process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
 'file:../../WReco/test/cteq65pdfAnalyzer_Events_100K.root',                                  
)
)
process.evtInfo = cms.OutputModule("AsciiOutputModule")


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('cteq_test.root')
)


#for i in range(41):
#  proc = "process.zpdf" + str(i)
 # print "proc", proc
process.zpf = cms.EDAnalyzer("zPdfUnc",
    genParticles = cms.InputTag("genParticles"),
    pdfweights = cms.InputTag("cteq65ewkPdfWeights"),                              pdfmember = cms.untracked.uint32(0),
    nbinsMass=cms.untracked.uint32(200),
    nbinsPt=cms.untracked.uint32(200),
    nbinsAng=cms.untracked.uint32(200),
    massMax =  cms.untracked.double(200.),
    ptMax=  cms.untracked.double(200.),
    angMax = cms.untracked.double(6.),
    #parameter for the geometric acceptance
    accPtMin = cms.untracked.double(20.0),
    accMassMin = cms.untracked.double(40.0),
    accMassMax = cms.untracked.double(12000.0),                             
    accEtaMin = cms.untracked.double(0.0),
    accEtaMax = cms.untracked.double(2.0),
    isMCatNLO= cms.untracked.bool(False),
    outfilename= cms.untracked.string("cteq65_10K.txt")
  )

for i in range(41):
  module = copy.deepcopy(process.zpf)
  setattr(module, "pdfmember", i)
  moduleLabel = module.label() + str(i)
  setattr(process, moduleLabel, module)
  if i == 0:
    seq = module
  else:
    seq = seq + module

print "seq", seq
    
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)

print seq
                  
process.path=cms.Path(seq)
process.end = cms.EndPath(process.evtInfo )



