import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ewkSystUnc")

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(-1)
    input = cms.untracked.int32(-1)
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
 'file:genParticlePlusISRANDFSRWeights.root',
)
)
process.evtInfo = cms.OutputModule("AsciiOutputModule")


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('EWKWeights.root')
)


#for i in range(41):
#  proc = "process.zpdf" + str(i)
 # print "proc", proc
process.ewkSyst = cms.EDAnalyzer("EWKSystUnc",
    genParticles = cms.InputTag("genParticles"),
    weights = cms.InputTag("xxxxx"),
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
    accEtaMax = cms.untracked.double(2.1),
    isMCatNLO= cms.untracked.bool(False),
    outfilename= cms.untracked.string("xxxxx.txt")
  )

w_1 = "isrWeight"
w_2 = "fsrWeight"
w_3= "isrGammaWeight"



### w1  members ###  
module_1 = copy.deepcopy(process.ewkSyst)
setattr(module_1, "weights", w_1)
setattr(module_1, "outfilename", w_1 + ".txt")
moduleLabel_1 = module_1.label()  + w_1
setattr(process, moduleLabel_1, module_1)


### w2  members ###  
module_2 = copy.deepcopy(process.ewkSyst)
setattr(module_2, "weights", w_2)
setattr(module_2, "outfilename", w_2 + ".txt")
moduleLabel_2 = module_2.label()  + w_2
setattr(process, moduleLabel_2, module_2)

### w2  members ###  
module_3 = copy.deepcopy(process.ewkSyst)
setattr(module_3, "weights", w_3)
setattr(module_3, "outfilename", w_3 + ".txt")
moduleLabel_3 = module_3.label()  + w_3
setattr(process, moduleLabel_3, module_3)

seq= module_1 + module_2 + module_3



print "sequence", seq
    
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)



                  
process.path=cms.Path(seq)
process.end = cms.EndPath(process.evtInfo )



