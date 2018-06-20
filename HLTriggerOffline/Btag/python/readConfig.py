# set up variables
#def readConfig(fileName)
import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Btag.helper import *

class fileINI:
	def __init__(self, fileName):
		self.fileName=fileName

	def read(self):
		 Config.optionxform = str
		 Config.read(self.fileName)
		 self.processname=ConfigSectionMap("config")["processname"]
		 self.CMSSWVER=ConfigSectionMap("config")["cmsswver"]
		 self.jets=ConfigSectionMap("config")["hltjets"]
		 files=ConfigSectionMap("config")["files"]
		 self.maxEvents=ConfigSectionMap("config")["maxevents"]
#		 self.denominatorTriggerPath=ConfigSectionMap("config")["denominatorTriggerPath"]

		 files=files.splitlines()
		 self.files=[x for x in files if len(x)>0]

		 self.btag_modules=cms.VInputTag()
		 self.btag_pathes=cms.vstring()
		 self.btag_modules_string=cms.vstring()
		 for path in Config.options("btag"):
		 	print path
		 	modules=Config.get("btag",path)
		 	modules=modules.splitlines()
		 	for module in modules:
		 		if(module!="" and path!=""):
				 	self.btag_modules.extend([cms.InputTag(module)])
				 	self.btag_modules_string.extend([module])
				 	self.btag_pathes.extend([path])

		 self.vertex_modules=cms.VInputTag()
		 self.vertex_pathes=cms.vstring()
		 for path in Config.options("vertex"):
		 	print path
		 	modules=Config.get("vertex",path)
		 	modules=modules.splitlines()
		 	for module in modules:
		 		if(module!="" and path!=""):
				 	self.vertex_modules.extend([cms.InputTag(module)])
				 	self.vertex_pathes.extend([path])
				 	
def printMe(self):
	print
	print  "Reading ", self.fileName
	print
	print  "denominatorTriggerPath		=	",self.denominatorTriggerPath
	print  "maxEvents		=	",self.maxEvents
	print  "CMSSWVER		=	",self.CMSSWVER
	print  "processname		=	",self.processname
	print  "jets (for matching)	=	",self.jets
	print  "files			=	",self.files
	print  "btag_modules		",self.btag_modules
	print  "btag_pathes		",self.btag_pathes
	print  "vertex_modules		",self.vertex_modules
	print  "vertex_pathes		",self.vertex_pathes
	print
