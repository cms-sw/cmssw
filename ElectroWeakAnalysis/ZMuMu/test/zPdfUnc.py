import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("zpdfsys")

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
 'file:genParticlePlusCteq65AndMRST06NNLOAndMSTW2007LOmodWeigths.root',
)
)
process.evtInfo = cms.OutputModule("AsciiOutputModule")


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('cteq65AndMST06NLOABDMSTW2007lomod.root')
)


#for i in range(41):
#  proc = "process.zpdf" + str(i)
 # print "proc", proc
process.zpf = cms.EDAnalyzer("zPdfUnc",
    genParticles = cms.InputTag("genParticles"),
    pdfweights = cms.InputTag("pdfWeights:xxxxx"),
    pdfmember = cms.untracked.uint32(0),
    nbinsMass=cms.untracked.uint32(200),
    nbinsPt=cms.untracked.uint32(200),
    nbinsAng=cms.untracked.uint32(200),
    massMax =  cms.untracked.double(200.),
    ptMax=  cms.untracked.double(200.),
    angMax = cms.untracked.double(6.),
    #parameter for the geometric acceptance (numerator)
    accPtMin = cms.untracked.double(20.0),
    accMassMin = cms.untracked.double(60.0),
    accMassMax = cms.untracked.double(120.0),                             
    accEtaMin = cms.untracked.double(0.0),
    accEtaMax = cms.untracked.double(2.1),
    # for denominator 
    accMassMinDenominator=cms.untracked.double(40.0),
    isMCatNLO= cms.untracked.bool(False),
    outfilename= cms.untracked.string("xxxxx.txt")
  )

pdf_1 = "cteq65"
pdf_2 = "MRST2006nnlo"
pdf_3= "MRST2007lomod"



### cteq65 has 1 + 2*20 members ###
for i in range(41):  
  module = copy.deepcopy(process.zpf)
  setattr(module, "pdfweights", "pdfWeights:cteq65")
  setattr(module, "pdfmember", i)
  setattr(module, "outfilename", "cteq65.txt")
  moduleLabel = module.label()  + pdf_1+ "_" + str(i)
  setattr(process, moduleLabel, module)
  if i == 0:
    seq = module
  else:
    seq = seq + module

###  MRST2006nnlo has 1 + 2*30 members ###
for j in range(61):
  module = copy.deepcopy(process.zpf)
  setattr(module, "pdfweights", "pdfWeights:MRST2006nnlo")
  setattr(module, "pdfmember", j)
  setattr(module, "outfilename", "MRST2006nnlo.txt")
  moduleLabel = module.label()  + pdf_2+ "_" + str(j)
  setattr(process, moduleLabel, module)
 #  needed only if the sequence is filled for the first time
 # if j == 0:
 #   seq_2 = module
 # else:
  seq = seq + module

###  MRST2007lomod has 1 member ###
for k in range(1):
  module = copy.deepcopy(process.zpf)
  setattr(module, "pdfweights", "pdfWeights:MRST2007lomod")
  setattr(module, "pdfmember", k)
  setattr(module, "outfilename", "MRST2007lomod.txt")
  moduleLabel = module.label()  + pdf_3+ "_" + str(k)
  setattr(process, moduleLabel, module)
 # needed only if the sequence is filled for the first time
 # if k == 0:
 #   seq_3 = module
 # else:
  seq = seq + module



print "sequence", seq
    
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)



                  
process.path=cms.Path(seq)
process.end = cms.EndPath(process.evtInfo )



