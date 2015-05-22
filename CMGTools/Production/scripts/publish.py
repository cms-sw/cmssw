#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## September 27th 2011

import os, getpass, sys, re, optparse, copy
from datetime import *
from CMGTools.Production.nameOps import *
from CMGTools.Production.publish import publish
from CMGTools.Production.publishTask import PublishTask
from optparse import *

if __name__ == '__main__':
	parser = optparse.OptionParser()
	def isComment(word):
		if word[0]=='"'or word[0]=="'":
			return word
		else: return None
	
	def separateOutput(line):
		line = re.sub("\s+", " ", line)
		line = line.lstrip().rstrip()
		line = line.split(" ")
		if len(line) > 3:
			print "too many args in line"
			return False
		for word in line: word = word.lstrip().rstrip()
		comment = None
		fileowner = None
		sampleName = None
		if isCMGDBName(line[0]):
			if len(line)>2:
				print "too many args in line"
				return False
			fileowner = getFileOwner(line[0])
			sampleName = getSampleName(line[0])
			if len(line) == 2: comment = isComment(line[1])
		elif re.search("%", line[0]):
			fileowner = line[0].split("%")[0]
			sampleName = line[0].split("%")[1]
			if len(line)==2:
				comment = isComment(line[1])
			if len(line)>2:
				print "too many args in line"
				return False
		elif isSampleName(line[0]):
			sampleName = line[0]
			if len(line)>1:
				comment = isComment(line[1])
				if comment is None:
					fileowner = line[1]
					if len(line)==3:
						comment = isComment(line[2])
		return sampleName, str(fileowner), comment
      	
	parser.usage = """
	%prog [options] <sampleName>
		
	Use this script to publish dataset details to CmgDB.
	Example:
	publish.py -F cbern /VBF_HToTauTau_M-120_7TeV-powheg-pythia6-tauola/Summer11-PU_S4_START42_V11-v1/AODSIM/V2/PAT_CMG_V2_5_0_Test_v2
	"""
	
	group = OptionGroup(parser, "Publish Options", """These options affect the way you publish to Savannah and CMGDB""")
	genGroup = OptionGroup(parser, "Login Options", """These options apply to your login credentials""")
	PublishTask.addOptionStatic(group)
	group.add_option("--min-run", dest="min_run", default=-1, type=int, help='When querying DBS, require runs >= than this run')
	group.add_option("--max-run", dest="max_run", default=-1, type=int, help='When querying DBS, require runs <= than this run')


	# If specified is used to log in to DBS (only required if user that created the dataset,
	# is different to user publishing it)
	genGroup.add_option("-u", "--username",
						action = "store",
						dest="username",
						help="""Specify the username to access the DBS servers. 
						Default is $USER.""",
						default=os.environ['USER'] )
	# If specified is used as password to DBS
	# If ommited the secure password prompt will appear
	genGroup.add_option("-p", "--password",
						action = "store",
						dest="password",
						help="""Specify the password to access the DBS servers.
	                                        If not entered, secure password prompt will appear.""",
	                  default=None )
    
    	genGroup.add_option("-d", "--dev",
						action = "store_true",
						dest="development",
						help="""Publish on official or development database.""",
	                 			default=False )
	# If user wants to add multiple datasets from file
	group.add_option("-M", "--multi",
						action = "store_true",
						dest="multi",
						help="""Argument is now LFN to location of .txt file
				Entries in the file should be on independant lines in the form: DatasetName Fileowner 'comment'
				Comment is not compulsory, and if fileowner is not entered, $USER will be used as default.
				Comment MUST be enclosed in speech marks
				E.g.
				/MuHad/Run2011A-05Aug2011-v1/AOD/V2 cmgtools 'comment'
				Single or double speech marks are accepted""",
				default = False)
	parser.add_option_group(genGroup)
	parser.add_option_group(group)
	
	(options, args) = parser.parse_args()
	
	# Allow no more than one argument
	if len(args)!=1:
		parser.print_help()
		sys.exit(1)
	
	#if options.password is None:
	#	options.password = PublishTask.getPassword(options.username)
	#if options.password is None:
        #	print "fail"
	#	print "Authentication Failed, exiting\n\n"
	#	sys.exit(1)
		
	
	# For multiple file input
	if options.multi:
		file = open(args[0], 'r')
		lines = file.readlines()
		for line in lines:
			sampleName, fileowner, comment = separateOutput(line)
			if fileowner is not None:
				options.fileown = fileowner
			if comment is not None:
				options.commented = comment
			pub = PublishTask(sampleName,options.fileown,copy.deepcopy(options) )
			pub.password = options.password
			pub.run({})
	
	# For singular file input
	else:
		sampleName = args[0].rstrip("/")
		pub = PublishTask(sampleName,options.fileown, copy.deepcopy(options) )
		pub.password = options.password
		pub.run({})

