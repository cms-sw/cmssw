import FWCore.ParameterSet.Config as cms

isData=True
outputRAW=False
maxNrEvents=1000
outputSummary=True
newL1Menu=False
hltProcName="HLT3"
runOpen=False #ignore all filter decisions, true for testing
runProducers=False #run the producers or not, 
if isData:
    from hlt import *
else:
    from muPathsMC import *

process.load("setup_cff")

if runProducers==False:
    hltProcName=hltProcName+"PB"

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(maxNrEvents)
)

# enable the TrigReport and TimeReport
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( outputSummary ),
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)


import sys
filePrefex="file:"
if(sys.argv[2].find("/pnfs/")==0):
    filePrefex="dcap://heplnx209.pp.rl.ac.uk:22125"

if(sys.argv[2].find("/store/")==0):
    filePrefex=""

if(sys.argv[2].find("/castor/")==0):
    filePrefex="rfio:"
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(),
                            eventsToProcess =cms.untracked.VEventRange()
                            )
for i in range(2,len(sys.argv)-1):
    print filePrefex+sys.argv[i]
    process.source.fileNames.extend([filePrefex+sys.argv[i],])


process.load('Configuration/EventContent/EventContent_cff')
process.output = cms.OutputModule("PoolOutputModule",
                                  splitLevel = cms.untracked.int32(0),
                                  outputCommands =cms.untracked.vstring("drop *",
                                                                        "keep *_TriggerResults_*_*",
                                                                        "keep *_hltTriggerSummaryAOD_*_*"),
                                  
                                  fileName = cms.untracked.string(sys.argv[len(sys.argv)-1]),
                                  dataset = cms.untracked.PSet(dataTier = cms.untracked.string('HLTDEBUG'),)
                                  )
if outputRAW:
    process.output.outputCommands=cms.untracked.vstring("drop *","keep *_rawDataCollector_*_*","keep *_addPileupInfo_*_*","keep *_TriggerResults_*_*","keep *_hltTriggerSummaryAOD_*_*")
                                                                
process.HLTOutput_sam = cms.EndPath(process.output)

isCrabJob=False
#if 1, its a crab job...
if isCrabJob:
    print "using crab specified filename"
    process.output.fileName= "OUTPUTFILE"
  
else:
    print "using user specified filename"
    process.output.fileName= sys.argv[len(sys.argv)-1]

#hlt stuff
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)

# override the process name
process.setName_(hltProcName)

# En-able HF Noise filters in GRun menu
if 'hltHfreco' in process.__dict__:
    process.hltHfreco.setNoiseFlags = cms.bool( True )

# override the L1 menu from an Xml file
if newL1Menu:
    process.l1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",
                                                TriggerMenuLuminosity = cms.string('startup'),
                                                DefXmlFile = cms.string('L1Menu_Collisions2012_v0_L1T_Scales_20101224_Imp0_0x1027.xml'),
                                                VmeXmlFile = cms.string('')
                                                )
    
    process.L1GtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
                                                    recordName = cms.string('L1GtTriggerMenuRcd'),
                                                    iovIsRunNotTime = cms.bool(True),
                                                    firstValid = cms.vuint32(1)
                                                    )
    
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml') 


# adapt HLT modules to the correct process name
if 'hltTrigReport' in process.__dict__:
    process.hltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreExpressCosmicsOutputSmart' in process.__dict__:
    process.hltPreExpressCosmicsOutputSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreExpressOutputSmart' in process.__dict__:
    process.hltPreExpressOutputSmart.TriggerResultsTag        = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreDQMForHIOutputSmart' in process.__dict__:
    process.hltPreDQMForHIOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreDQMForPPOutputSmart' in process.__dict__:
    process.hltPreDQMForPPOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreHLTDQMResultsOutputSmart' in process.__dict__:
    process.hltPreHLTDQMResultsOutputSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreHLTDQMOutputSmart' in process.__dict__:
    process.hltPreHLTDQMOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreHLTMONOutputSmart' in process.__dict__:
    process.hltPreHLTMONOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltDQMHLTScalers' in process.__dict__:
    process.hltDQMHLTScalers.triggerResults                   = cms.InputTag( 'TriggerResults', '', hltProcName )
    process.hltDQMHLTScalers.processname                      = hltProcName

if 'hltDQML1SeedLogicScalers' in process.__dict__:
    process.hltDQML1SeedLogicScalers.processname              = hltProcName

# remove the HLT prescales
if 'PrescaleService' in process.__dict__:
    process.PrescaleService.lvl1DefaultLabel = cms.string( '0' )
    process.PrescaleService.lvl1Labels       = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    process.PrescaleService.prescaleTable    = cms.VPSet( )


# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
    from Configuration.AlCa.autoCond import autoCond
    if isData:
        process.GlobalTag.globaltag = autoCond['hltonline'].split(',')[0]
       # process.GlobalTag.globaltag = 'GR_H_V29::All'
    else:
        process.GlobalTag.globaltag = autoCond['startup']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')
    process.MessageLogger.suppressInfo = cms.untracked.vstring('ElectronSeedProducer',"hltL1NonIsoStartUpElectronPixelSeeds","hltL1IsoStartUpElectronPixelSeeds","BasicTrajectoryState")

def uniq(input):
    output = []
    for x in input:
        if x not in output:
            output.append(x)
    #print output
    return output

prodWhiteList=[]


# This small loop is adding the producers that
# needs to be re-run for Btagging and Tau paths

for moduleName in process.producerNames().split():
    prod = getattr(process,moduleName)
    if moduleName.endswith("Discriminator"):
        #print moduleName
        prodWhiteList.append(moduleName)
    for paraName in prod.parameters_():
        para = prod.getParameter(paraName)
        if type(para).__name__=="VInputTag":
            if paraName == "tagInfos":
                #print moduleName
                paranew = para.value()[0]
                #print paranew
                prodWhiteList.append(paranew)
                prodWhiteList.append(moduleName)
        if type(para).__name__=="InputTag":
            if paraName == "jetTracks":
                #print para.getModuleLabel()
                paranew2 = para.getModuleLabel()
                prodWhiteList.append(paranew2)
                #print moduleName
                prodWhiteList.append(moduleName)

            
prodWhiteList.append("hltFastPVJetTracksAssociator")
prodWhiteList.append("hltCombinedSecondaryVertex")
prodWhiteList.append("hltSecondaryVertexL25TagInfosHbbVBF") #This is because VInput has 2 arguments
prodWhiteList.append("hltSecondaryVertexL3TagInfosHbbVBF") #This is because VInput has 2 arguments
prodWhiteList.append("hltL3SecondaryVertexTagInfos") #This is because VInput has 2 arguments


prodWhiteList = uniq(prodWhiteList)


prodTypeWhiteList=[]


pathBlackList=[]
# We don't really care about emulating these triggers..
# these version numbers need to be updated
pathBlackList.append("HLT_BeamHalo_v13")
pathBlackList.append("HLT_IsoTrackHE_v15")
pathBlackList.append("HLT_IsoTrackHB_v14")
pathBlackList.append("DQM_FEDIntegrity_v11")
pathBlackList.append("AlCa_EcalEtaEBonly_v6")
pathBlackList.append("AlCa_EcalEtaEEonly_v6")
pathBlackList.append("AlCa_EcalPi0EBonly_v6")
pathBlackList.append("AlCa_EcalPi0EEonly_v6")


filterBlackList=[]

if runProducers==False:
    for pathName in process.pathNames().split():
        path = getattr(process,pathName)
        for moduleName in path.moduleNames():
            if moduleName in filterBlackList:
                notAllCopiesRemoved=True
                while notAllCopiesRemoved:
                    notAllCopiesRemoved = path.remove(getattr(process,moduleName))

for pathName in process.pathNames().split():
    if pathName in pathBlackList:
        path = getattr(process,pathName)
        for moduleName in path.moduleNames():
            notAllCopiesRemoved=True
            while notAllCopiesRemoved:
                notAllCopiesRemoved = path.remove(getattr(process,moduleName))

if runProducers==False:
    for path in process.pathNames().split():
       # print path
        for producer in process.producerNames().split():
            if producer not in prodWhiteList:
                if getattr(process,producer).type_() not in prodTypeWhiteList:
                    notAllCopiesRemoved=True
                    #print producer
                    while notAllCopiesRemoved:
                        notAllCopiesRemoved = getattr(process,path).remove(getattr(process,producer))


#okay this is horrible, we just need a list of ignored filters
#however I dont know how to get a filter to tell me its ignored
#so we have to dump the python config of the path and look for cms.ignore
#this doesnt expand sequences so we need to also check in sequences
#and I've now found the better way and this is no longer used...
def findFiltersAlreadyIgnored(path,process): #there has got to be a better way...
    filtersAlreadyIgnored=[]
    pathSeq= path.dumpPython(options=cms.Options())
    for module in pathSeq.split("+"):
       # print "mod one ",module,"test"        
        if module.startswith("cms.ignore"):
            module=module.lstrip("cms.ignore")
            module=module.lstrip("(")
            module=module.rstrip(")")
            module=module.lstrip("process.")
            filtersAlreadyIgnored.append(module)
        else:
            module=module.lstrip("cms.Path(")
            module=module.lstrip("(")
            module=module.rstrip("\n")
            module=module.rstrip(")")
            if module.startswith("process."):
                module=module.lstrip("process.")
        #        print module, type(getattr(process,module))
                if type(getattr(process,module)).__name__=="Sequence":
                    #print "sequence"
                    #print module
                    filtersIgnoredInSequence = findFiltersAlreadyIgnored(getattr(process,module),process)
                    for filter in filtersIgnoredInSequence:
                        filtersAlreadyIgnored.append(filter)
    return filtersAlreadyIgnored

#this removes opperators such as not and ignore from filters
#this is because you cant ignore twice or not ignore
def rmOperatorsFromFilters(process,path):
    if path._seq!=None:
        for obj in path._seq._collection:
            if obj.isOperation():
                moduleName = obj.dumpSequencePython()
                if moduleName.startswith("~"):
                    moduleName = moduleName.lstrip("~process.")
                    module =getattr(process,moduleName)
                    path.replace(obj,module)
                elif moduleName.startswith("cms.ignore"):
                    moduleName = moduleName.lstrip("cms.ignore(process.")
                    moduleName = moduleName.rstrip(")")
                    module = getattr(process,moduleName)
                    path.replace(obj,module)
            if type(obj).__name__=="Sequence":
                rmOperatorsFromFilters(process,obj)

if runOpen:
    for pathName in process.pathNames().split():
        path = getattr(process,pathName)
        rmOperatorsFromFilters(process,path)
        for filterName in path.moduleNames():
            filt = getattr(process,filterName)
            if type(filt).__name__=="EDFilter":
                path.replace(filt,cms.ignore(filt))
            

def cleanList(input,blacklist):
    output = []
    for x in input:
        if x not in blacklist:
            output.append(x)
    #print output
    return output

productsToKeep = []
for pathName in process.pathNames().split():
    path = getattr(process,pathName)
    for filterName in path.moduleNames():
        #print filterName
        filt = getattr(process,filterName)
        #print filt
        #print filt.type_()
        if type(filt).__name__=="EDFilter":
            #print filterName
            for paraName in filt.parameters_():
                para = filt.getParameter(paraName)
                if type(para).__name__=="InputTag":
                    if para.getModuleLabel()!="":
                        productsToKeep.append(para.getModuleLabel())
                   # print paraName,type(para).__name__,para.getModuleLabel()
                if type(para).__name__=="VInputTag":
                    #print paraName,type(para).__name__,para.getModuleLabel()
                    for tag in para:
                        if tag!="":
                            productsToKeep.append(tag)

# This adds all the producers to be kept, just to be safe
# Later on it should be optimized so that only the
# producers we run on should be saved.
for moduleName in process.producerNames().split():
    if moduleName.startswith("hlt"):
        productsToKeep.append(moduleName)
    
               
productsToKeep = uniq(productsToKeep)
productsToKeep = cleanList(productsToKeep,process.filterNames().split())

process.output.outputCommands=cms.untracked.vstring("drop *","keep *_TriggerResults_*_*",
                                                    "keep *_hltTriggerSummaryAOD_*_*")

for product in productsToKeep:
    process.output.outputCommands.append("keep *_"+product+"_*_*")

# version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

# ---- dump ----
#dump = open('dump.py', 'w')
#dump.write( process.dumpPython() )
#dump.close()

