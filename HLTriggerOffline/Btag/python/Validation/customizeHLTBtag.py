import FWCore.ParameterSet.Config as cms

# tool to change settings 
from PhysicsTools.PatAlgos.tools.coreTools import *
from HLTriggerOffline.Btag.Validation.helper import *

def customizeHLTBtag(process):

 process.load("DQMServices.Components.EDMtoMEConverter_cff")
# read settings from my.ini
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

# massSearchReplaceAnyInputTag(process.hltpostvalidation,cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),cms.InputTag(jets,'',processname),verbose=True)


# fix path name in hltvalidation and hltpostvalidation
 massSearchReplaceParam(process.hltvalidation,"HLTPathNames",cms.vstring('HLT_DiJet40Eta2p6_BTagIP3DFastPV'),HLTPathNames)

# fix path name in  hltpostvalidation

 #massSearchReplaceParam(process.hltpostvalidation,"HLTPathNames",cms.vstring('HLT_DiJet40Eta2p6_BTagIP3DFastPV'),HLTPathNames)


# fix Trigger results
 massSearchReplaceAnyInputTag(process.hltvalidation,cms.InputTag("TriggerResults"),cms.InputTag("TriggerResults",'',processname),verbose=True)

#fix BTagAlgorithms
 massSearchReplaceParam(process.hltvalidation,"BTagAlgorithms",cms.vstring('TCHE'),BTagAlgorithms,verbose=True)

# fix minTags
 #massSearchReplaceParam(process.hltpostvalidation,"minTags",cms.vdouble(3.41),mintags)


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


# process.dqmSaver.workflow = "/" + CMSSWVER + "/RelVal/TrigVal"
# process.DQMStore.verbose=0
 process.maxEvents.input = int(maxEvents)
 process.source.fileNames.extend(files)

 print process.source.fileNames


 # fix hltCaloJets and genParticles
 massSearchReplaceAnyInputTag(process.hltassociation,cms.InputTag("hltCaloJetL1FastJetCorrected","","HLT"),cms.InputTag(jets,'',processname),verbose=True)
 massSearchReplaceAnyInputTag(process.hltassociation,cms.InputTag("genParticles","","HLT"),cms.InputTag("genParticles",'',genParticlesProcess),verbose=True)

# remove sequences which require missed objects in input files
# removeIfInSequence(process,"globalPrevalidation","prevalidation")

# removeIfInSequence(process,"basicGenTest_seq","validation")
# removeIfInSequence(process,"globaldigisanalyze","validation")
# removeIfInSequence(process,"globalhitsanalyze","validation")
# removeIfInSequence(process,"globalrechitsanalyze","validation")
# removeIfInSequence(process,"globalValidation","validation")

#  process.prevalidation.remove(globalPrevalidation)
 if 'validation' in process.__dict__:
  process.validation=cms.Sequence(process.hltvalidation)

 if 'globalPrevalidation' in process.__dict__:
  process.globalPrevalidation=cms.Sequence()

 if 'RandomNumberGeneratorService' in process.__dict__:
  del process.RandomNumberGeneratorService


 return process
