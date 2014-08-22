import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTHARVEST")

process.load("HLTriggerOffline.Common.HLTValidationHarvest_cff")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load("DQMServices.Components.EDMtoMEConverter_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)
#process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
process.DQMStore.collateHistograms = False

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = ""
process.DQMStore.verbose=3

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE'),
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Other statements

#Adding DQMFileSaver to the message logger configuration
process.MessageLogger.categories.append('DQMFileSaver')
process.MessageLogger.cout.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )
process.MessageLogger.cerr.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )

process.load("HLTriggerOffline.Common.HLTValidation_cff")

# settings 
from PhysicsTools.PatAlgos.tools.coreTools import *	

# read them from my.ini
from HLTriggerOffline.Btag.Validation.helper import *
l25JetTag="hltBLifetimeL25BJetTagsbbPhiL1FastJetFastPV"
l3TagInfo="hltBLifetimeL3TagInfosbbPhiL1FastJetFastPV"
l3JetTag="hltBLifetimeL3BJetTagsbbPhiL1FastJetFastPV"
HLTPathNames="HLT_DiJet40Eta2p6_BTagIP3DFastPV"
jets="hltCaloJetL1FastJetCorrected"
processname="HLT"
CMSSWVER="CMSSW_X_Y_Z"
files="file:output.root"
genParticlesProcess="HLT"
BTagAlgorithms="TCHE"
maxEvents=-1
wops="3.41"
# set up variables
try:
 Config.read("my.ini")
 l25JetTag=ConfigSectionMap("l25")["jettag"]
 l3TagInfo=ConfigSectionMap("l3")["taginfo"]
 l3JetTag=ConfigSectionMap("l3")["jettag"]
 HLTPathNames=ConfigSectionMap("l25")["hltpathnames"]
 processname=ConfigSectionMap("l25")["processname"]
 CMSSWVER=ConfigSectionMap("l25")["cmsswver"]
 jets=ConfigSectionMap("l25")["hltjets"]
 files=ConfigSectionMap("l25")["files"]
 genParticlesProcess=ConfigSectionMap("additional")["genparticles"]
 BTagAlgorithms=ConfigSectionMap("l25")["btagalgorithms"]
 maxEvents=ConfigSectionMap("l25")["maxevents"]
 wops=ConfigSectionMap("l25")["wops"]
except:
 print "Something wrong with ini"

files=files.splitlines()
files=filter(lambda x: len(x)>0,files)

HLTPathNames=HLTPathNames.splitlines()
HLTPathNames=filter(lambda x: len(x)>0,HLTPathNames)

BTagAlgorithms=BTagAlgorithms.splitlines()
BTagAlgorithms=filter(lambda x: len(x)>0,BTagAlgorithms)

mintags=cms.vdouble()
wops=wops.splitlines()
wops=filter(lambda x: len(x)>0,wops)
wops=[ float(i) for i in wops]
mintags.extend(wops)



# fix all InputTags of L25 and L3 collections:



l25JetTag=l25JetTag.splitlines()
l25JetTag=filter(lambda x: len(x)>0,l25JetTag)
l25JetTag=[cms.InputTag(i,'',processname) for i in l25JetTag]
l25JetTags=cms.VInputTag()
l25JetTags.extend(l25JetTag)

l3TagInfo=l3TagInfo.splitlines()
l3TagInfo=filter(lambda x: len(x)>0,l3TagInfo)
l3TagInfo=[cms.InputTag(i,'',processname) for i in l3TagInfo]
l3TagInfos=cms.VInputTag()
l3TagInfos.extend(l3TagInfo)


l3JetTag=l3JetTag.splitlines()
l3JetTag=filter(lambda x: len(x)>0,l3JetTag)
l3JetTag=[cms.InputTag(i,'',processname) for i in l3JetTag]
l3JetTags=cms.VInputTag()
l3JetTags.extend(l3JetTag)



massSearchReplaceParam(process.hltvalidation,"L3IPTagInfo",cms.VInputTag(cms.InputTag("hltBLifetimeL3TagInfosbbPhiL1FastJetFastPV")),l3TagInfos)
massSearchReplaceParam(process.hltvalidation,"L25JetTag",cms.VInputTag(cms.InputTag("hltBLifetimeL25BJetTagsbbPhiL1FastJetFastPV")),l25JetTags)
massSearchReplaceParam(process.hltvalidation,"L3JetTag",cms.VInputTag(cms.InputTag("hltBLifetimeL3BJetTagsbbPhiL1FastJetFastPV")),l3JetTags)



massSearchReplaceAnyInputTag(process.hltvalidation,cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),cms.InputTag(jets,'',processname),verbose=True)
massSearchReplaceAnyInputTag(process.hltpostvalidation,cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),cms.InputTag(jets,'',processname),verbose=True)


# fix path name in hltvalidation and hltpostvalidation

massSearchReplaceParam(process.hltvalidation,"HLTPathNames",cms.vstring('HLT_DiJet40Eta2p6_BTagIP3DFastPV'),HLTPathNames)
massSearchReplaceParam(process.hltpostvalidation,"HLTPathNames",cms.vstring('HLT_DiJet40Eta2p6_BTagIP3DFastPV'),HLTPathNames)


# fix Trigger results
massSearchReplaceAnyInputTag(process.hltvalidation,cms.InputTag("TriggerResults"),cms.InputTag("TriggerResults",'',processname),verbose=True)

#fix BTagAlgorithms
massSearchReplaceParam(process.hltvalidation,"BTagAlgorithms",cms.vstring('TCHE'),BTagAlgorithms,verbose=True)

# fix minTags
massSearchReplaceParam(process.hltpostvalidation,"minTags",cms.vdouble(3.41),mintags)



print  "l3TagInfos=",  l3TagInfos
print  "l25JetTags=", l25JetTags
print  "l3JetTags=", l3JetTags
print  "minTags=",mintags



print HLTPathNames
print  CMSSWVER
print processname
print files
print jets
print BTagAlgorithms
print "genParticlesProcess=", genParticlesProcess

process.dqmSaver.workflow = "/" + CMSSWVER + "/RelVal/TrigVal"
process.DQMStore.verbose=0
process.maxEvents.input = int(maxEvents)

process.source.fileNames.extend(files)
#process.source.fileNames = cms.untracked.vstring(
#'file:output.root'
#'/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2_09Jul2013-v1/00000/040F41E6-0BE9-E211-8C6A-002618943923.root',
#'/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2_09Jul2013-v1/00000/022CD2E9-0BE9-E211-A45E-002354EF3BE0.root',
#'/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2_09Jul2013-v1/00000/36945551-16E9-E211-9387-0025905938AA.root',
#'/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2_09Jul2013-v1/00000/8EE30C05-18E9-E211-A9A7-002618943810.root',
#'/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2_09Jul2013-v1/00000/9AC7EF32-0DE9-E211-8ECB-002618943948.root',
#'/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2_09Jul2013-v1/00000/9CD0287D-0BE9-E211-BEF3-003048678FDE.root',
#'/store/relval/CMSSW_5_3_11_patch3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2_09Jul2013-v1/00000/E47EE1C6-0DE9-E211-A92F-00261894389D.root'
#'file:step2_RAW2DIGI_RECO_VALIDATION.root'
#'file:MyOutputFile.root'
#)


print process.source.fileNames



#extra config needed in standalone
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
#process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load("HLTriggerOffline.Btag.Validation.hltJetMCTools_cff")
process.validation = cms.Path(
    process.hltvalidation
   # process.HLTMuonVal
   #+process.HLTTauVal
   #+process.EcalPi0Mon
   #+process.EcalPhiSymMon
   #+process.egammaValidationSequence
   #+process.HLTTopVal
   #+process.HLTSusyExoVal
   #+process.HLTFourVector
   #+process.heavyFlavorValidationSequence
    )

process.post_validation = cms.Path(
    process.hltpostvalidation
    )

# fix hltCaloJets and genParticles
process.extra_jetmctools  = cms.Path( process.hltJetMCTools )
massSearchReplaceAnyInputTag(process.extra_jetmctools,cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),cms.InputTag(jets,'',processname),verbose=True)
massSearchReplaceAnyInputTag(process.extra_jetmctools,cms.InputTag("genParticles","","HLT"),cms.InputTag("genParticles",'',genParticlesProcess),verbose=True)


process.EDMtoMEconv_and_saver= cms.Path(process.EDMtoMEConverter*process.dqmSaver)



#print process.dumpPython()


process.schedule = cms.Schedule(
	process.extra_jetmctools,
    process.validation,
    process.post_validation,
    process.EDMtoMEconv_and_saver
    )

for filter in (getattr(process,f) for f in process.filters_()):
    if hasattr(filter,"outputFile"):
        filter.outputFile=""


