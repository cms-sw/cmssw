# set up variables
#def readConfig(fileName)
import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.Validation.helper import *

class fileXML:
	def __init__(self, fileName):
		self.fileName=fileName

	def read(self):
		try:
		 Config.read("my.ini")
		 l3TagInfo=ConfigSectionMap("l3")["taginfo"]
		 l3JetTag=ConfigSectionMap("l3")["jettag"]
		 l25JetTag=ConfigSectionMap("l25")["jettag"]
		 HLTPathNames=ConfigSectionMap("l25")["hltpathnames"]
		 self.processname=ConfigSectionMap("l25")["processname"]
		 self.CMSSWVER=ConfigSectionMap("l25")["cmsswver"]
		 self.jets=ConfigSectionMap("l25")["hltjets"]
		 files=ConfigSectionMap("l25")["files"]
		 self.genParticlesProcess=ConfigSectionMap("additional")["genparticles"]
		 BTagAlgorithms=ConfigSectionMap("l25")["btagalgorithms"]
		 self.maxEvents=ConfigSectionMap("l25")["maxevents"]
		 wops=ConfigSectionMap("l25")["wops"]
		except:
		 print "Something wrong with ini"

		files=files.splitlines()
		self.files=filter(lambda x: len(x)>0,files)

		HLTPathNames=HLTPathNames.splitlines()
		self.HLTPathNames=filter(lambda x: len(x)>0,HLTPathNames)

		BTagAlgorithms=BTagAlgorithms.splitlines()
		self.BTagAlgorithms=filter(lambda x: len(x)>0,BTagAlgorithms)

		self.mintags=cms.vdouble()
		wops=wops.splitlines()
		wops=filter(lambda x: len(x)>0,wops)
		wops=[ float(i) for i in wops]
		self.mintags.extend(wops)

		# fix all InputTags of L25 and L3 collections:
		l25JetTag=l25JetTag.splitlines()
		l25JetTag=filter(lambda x: len(x)>0,l25JetTag)
		l25JetTag=[cms.InputTag(i,'',self.processname) for i in l25JetTag]
		self.l25JetTags=cms.VInputTag()
		self.l25JetTags.extend(l25JetTag)

		l3TagInfo=l3TagInfo.splitlines()
		l3TagInfo=filter(lambda x: len(x)>0,l3TagInfo)
		l3TagInfo=[cms.InputTag(i,'',self.processname) for i in l3TagInfo]
		self.l3TagInfos=cms.VInputTag()
		self.l3TagInfos.extend(l3TagInfo)


		l3JetTag=l3JetTag.splitlines()
		l3JetTag=filter(lambda x: len(x)>0,l3JetTag)
		l3JetTag=[cms.InputTag(i,'',self.processname) for i in l3JetTag]
		self.l3JetTags=cms.VInputTag()
		self.l3JetTags.extend(l3JetTag)
