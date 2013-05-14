#!/usr/bin/env python
import sys
import os
import re
import subprocess
import getopt
import random
from xml.dom.minidom import parse, parseString
import CondCore.RegressionTest.common_db    as common_db
import CondCore.RegressionTest.results_db   as results_db
import CondCore.RegressionTest.reference_db as reference_db

def ParseXML(filename, label):
	initSeq = []
	mainSeq = []
	finalSeq = []
	dom = parse(filename)
	xml = dom.getElementsByTagName("xml")
	foundLabel = False
	if xml != None:
		for xm in xml:
			testData = xm.getElementsByTagName("test")
			for it in testData:
				#print it.toxml()
				if "name" in  it.attributes.keys():
					tLabel = str(it.attributes["name"].value)
					if  tLabel == label:
						foundLabel = True
				if foundLabel == True:
					inits = it.getElementsByTagName("init")
					finals = it.getElementsByTagName("final")
					seqs = it.getElementsByTagName("sequence")
					for init in inits:
						commands = init.getElementsByTagName("command")
						for command in commands:
							commPars0 = None
							commPars1 = None
							if "exec" in  command.attributes.keys():
								commPars0 = str(command.attributes["exec"].value)
							if "env" in  command.attributes.keys():
								commPars1 = str(command.attributes["env"].value)
							initSeq.append((commPars0,commPars1))
					for seq in seqs:
						commands = seq.getElementsByTagName("command")
						for command in commands:
							commPars0 =  None
							commPars1 =  None
							commPars2 =  None
							if "exec" in  command.attributes.keys():
								commPars0 = str(command.attributes["exec"].value)
							if "result" in  command.attributes.keys():
								commPars2 = str(command.attributes["result"].value)
							if "env" in  command.attributes.keys():
								commPars1 = str(command.attributes["env"].value)
							mainSeq.append( (commPars0,commPars1,commPars2) )
					for final in finals:
						commands = final.getElementsByTagName("command")
						for command in commands:
							commPars0 =  None
							commPars1 =  None
							if "exec" in  command.attributes.keys():
								commPars0 = str(command.attributes["exec"].value)
							if "env" in  command.attributes.keys():
								commPars1 = str(command.attributes["env"].value)
							finalSeq.append( (commPars0,commPars1) )
				foundLabel = False
	return initSeq,mainSeq,finalSeq

def SetEnv( release, arch, path):
	CONNECTION_STRING,USERNAME,PASSWORD,AUTH_PATH = common_db.getDBConnectionParams()
        random.seed()
	srcPath = os.path.join(path, release,"src")
	cmds = """
if [ $RETVAL = 0 ]; then
echo "Setting environment variables for """+release+""" """+arch+""" "
echo "path : """+path+""""
	eval pushd """+srcPath+"""
	RETVAL=$?
	if [ $RETVAL = 0 ]; then
		export SCRAM_ARCH="""+arch+"""
		RETVAL=$?
		if [ $RETVAL = 0 ]; then
			eval `scram runtime -sh`
			RETVAL=$?
			if [ $RETVAL = 0 ]; then
				export TNS_ADMIN=/afs/cern.ch/project/oracle/admin
				RETVAL=$?
				if [ $RETVAL = 0 ]; then
					eval popd
					RETVAL=$?
				fi
			fi
		fi
	TRELEASE="""+release+"""
	TARCH="""+arch+"""
	TPATH="""+path+"""
	TMAPNAME="""+release+"""_"""+arch+"""
	TMAINDB="""+CONNECTION_STRING+"""
        TAUXDB="""+"oracle://cms_orcoff_prep/CMS_COND_WEB"+"""
	TUSERNAME="""+USERNAME+"""
	TPASSWORD="""+PASSWORD+"""
	TTEST=$LOCALRT/test/$TARCH
	TBIN=$LOCALRT/bin/$TARCH
	TAUTH="""+AUTH_PATH+"""
        TSEED="""+str(random.randrange(1, 10))+"""
echo "Environment variables set successfully"
	else
echo "Setting environment failed on """+release+""" """+arch+""" return code :  $RETVAL"
	fi
fi
echo "----------------------------------------------" 
"""
	return cmds

def Command(runStr):
	cmds = """
if [ $RETVAL = 0 ]; then
echo "Executing Command """+runStr+""" "
echo "with $TRELEASE $TARCH :"
	"""+runStr+"""
	RETVAL=$?
	if [ $RETVAL != 0 ]; then
echo "Task failed on $TRELEASE $TARCH return code :  $RETVAL"
	else
echo "Task performed successfully"
	fi
fi
"""
	return cmds

def RunTest(label,testSeq, release, arch, path, refRelease, refArch, refPath ):
	cmds ="""
	RETVAL=0
echo "*****************************************************************************************"
echo "Reference release: """+refRelease+""" "
echo "Arch: """+refArch+""" "
echo "Path: """+refPath+""" "
echo "*****************************************************************************************"
"""
	nr = 0
	currEnv = 0
	print "-> init"
        initSeq = testSeq[0]
	mainSeq = testSeq[1]
	finalSeq = testSeq[2]
	for step in initSeq:
                cmds += """
echo "==============================================="
"""
		print step[0]
		if step[1] == "cand" and currEnv != 1:
			cmds += SetEnv(release,arch,path)
			currEnv = 1
		elif step[1] == "ref" and currEnv != 2:
			cmds += SetEnv(refRelease, refArch,refPath)
			currEnv = 2
		cmds +=Command(step[0])
	print "-> test sequence"
	for step in mainSeq:
                cmds += """
echo "==============================================="
"""
		print step[0]
		if step[1] == "cand" and currEnv != 1:
			cmds += SetEnv(release,arch,path)
			currEnv = 1
		elif step[1] == "ref" and currEnv != 2:
			cmds += SetEnv(refRelease, refArch,refPath)
			currEnv = 2
		cmds +=Command(step[0])
		cmds += 'RCODE['+str(nr)+']=$RETVAL'
		nr +=1
	print "-> final"
	cmds += """
	RETVAL=0
	"""
	for step in finalSeq:
                cmds += """
echo "==============================================="
"""
		print step[0]
		if step[1] == "cand" and currEnv != 1:
			cmds += SetEnv(release,arch,path)
			currEnv = 1
		elif step[1] == "ref" and currEnv != 2:
			cmds += SetEnv(refRelease, refArch,refPath)
			currEnv = 2
		cmds +=Command(step[0])
	cmds += """
echo "==============================================="
        echo "Script return code : ${RCODE[1]}"
echo "!L!"""+label+"""!TR!"""+release+"""!TA!"""+arch+"""!RR!"""+refRelease+"""!RA!"""+refArch
	for i in range (0, nr):
		cmds+= "!C"+str(i)+"!${RCODE["+str(i)+"]}"
	cmds += "\""
	return cmds

def ExecuteCommand( cmdList ):
	stdout_value = None
	if cmdList != "":
		cmdList+="""
	echo "End of test"
        echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" """
		pipe = subprocess.Popen(cmdList, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		stdout_value = pipe.communicate()[0]
	return stdout_value

class RegressionTest:
	
    def __init__( self, conn ):
        self.resDb = results_db.ResultsDB( conn )
	self.refDb = reference_db.ReferenceDB( conn )
	self.label = None
	self.n_res = 0
	self.resTags = None
	self.out_value = None

    def runOnDb(self, label, release, arch, path):
	relID = reference_db.ExtractID(release)
	print 'Running test "'+label+'" on rel='+release+' arch='+arch+' from path= '+path
	cmds = """
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Candidate release: """+release+""" "
echo "Arch: """+arch+""" "
echo "Path: """+path+""" "
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"""
        testSeq = ParseXML("sequences.xml", label)
	self.n_res = len(testSeq[1])
        if(self.n_res == 0):
            print "Error : no test sequence found for label %s"%label
            return 0
	self.resTags = []
	for step in testSeq[1]:
		self.resTags.append( step[2] )
	releases = self.refDb.listReleases( relID )
	for rel in releases:
		cmd  = RunTest( label,testSeq, release, arch, path, rel[0], rel[1], rel[2])
		cmds += cmd
	self.out_value =  ExecuteCommand( cmds )
	self.label = label
	return len(releases)

    def runOnReference(self, label, release, arch, path, refRelease, refArch, refPath):
	relID = reference_db.ExtractID(release)
	print "Testing "+release+" "+arch+" from path: "+path
	curs = conn.cursor()
	cmds = """
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Candidate release: """+release+""" "
echo "Arch: """+arch+""" "
echo "Path: """+path+""" "
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"""
        testSeq = ParseXML("sequences.xml", label)
	self.n_res = len( testSeq[1] )
	if(self.n_res == 0):
		print "Error : no test sequence found for label %s"%label
		return 0
        self.resTags = []
	for step in testSeq[1]:
		self.resTags.append( step[2] )
	cmd = RunTest(label,testSeq, release, arch, path, refRelease, refArch, refPath)
	cmds += cmd
	self.out_value =  ExecuteCommand( cmds )
	self.label = label
	return 1

    def finalize( self, writeFlag ):

	reStr = "\!L\!([^!]+)\!TR\!([^!]+)\!TA\!([^!]+)\!RR\!([^!]+)\!RA\!([^!]+)"
	for i in range (0, self.n_res):
		reStr +=  "\!C"+str(i)+"\!(\d+)"
	#print 'restr1=',reStr

	pattern = re.compile(reStr)
	matching = pattern.findall(self.out_value)
	stdoutMod = pattern.sub("", self.out_value)
	timeStamp = self.resDb.getDate()
        stat = "SUCCESS"
	runID = 0
	if( writeFlag ):
		runID = self.resDb.getNewRunId()
	for match in matching:
		#print match
		if( writeFlag):
			self.resDb.writeResult(runID, timeStamp, match,self.resTags)
		for i in range(5, len(match)):
			if( match[i] != str(0) ):
				stat = "FAILURE"
	print stdoutMod
	if( writeFlag ):
		self.resDb.addResultLog(runID, stdoutMod)
        print "Test '%s' runID=%d" %(self.label, runID)
        print "Exit status=%s" %stat
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

def CmdUsage():
	print "Command line arguments :"
	print "-F (--full) -t [test_label] -r [release] -a [arch] -p [path]: runs the full test. " 
	print "-S (--self) -t [test_label] -r [release] -a [arch] -p [path]: runs the self test. " 
	print "-R [ref_release] -A [ref_arch] -P [ref_path]  -t [test_label] -r [release] -a [arch] -p [path]: runs the test against the specified ref release. "
	print "   optional flag -w: write the test result in the database. "
	
def CheckPath (release, arch, path):
	if(os.path.exists(path)):
		if(os.path.exists(os.path.join(path, release))):
			if(os.path.exists(os.path.join(path, release, "test", arch))):
				return True
			else:
				print "Architecture not found"
				return False
		else:
			print "Release not found"
			return False
	else:
		print "Path not found"
		return False
	
try:
	opts, args = getopt.getopt(sys.argv[1:], "FSR:A:P:t:r:a:p:hw", ['full', 'self', 'help'])
except getopt.GetoptError, err:
	# print help information and exit:
	print str(err) # will print something like "option -a not recognized"
       	CmdUsage()
       	sys.exit(2)
RELEASE = None
ARCH = None
PATH = None
REF_RELEASE = None
REF_ARCH = None
REF_PATH = None
LABEL = None
fflag = False
sflag = False
wflag = False
for o, a in opts:
	if o in ("-F", "--full"):
		fflag = True
	elif o in ("-S", "--self"):
		sflag = True
	elif o == "-R":
		REF_RELEASE = a
	elif o == "-A":
		REF_ARCH = a
	elif o == "-P":
		REF_PATH = a
	elif o == "-t":
		LABEL = a
	elif o == "-r":
		RELEASE = a
	elif o == "-a":
		ARCH = a
	elif o == "-p":
		PATH = a
	elif o == "-w":
		wflag = True
	elif o in ("-h", "--help"):
		CmdUsage()
		sys.exit(2)
	else:
		assert False, "unhandled option"
if( fflag == False and sflag == False and REF_RELEASE == None ):
	print "Error: missing main run option."
else:
	okPar = True
	if ( LABEL == None ):
                okPar = False
                print "Error: missing -l (label) parameter"
	if(RELEASE == None ):
                okPar = False
                print "Error: missing -r (release) parameter"
	if( ARCH == None ):
                okPar = False
                print "Error: missing -a (architecture) parameter"
	if( PATH == None):
                okPar = False
                print "Error: missing -p (path) parameter"
	if(CheckPath(RELEASE, ARCH, PATH) == False):
                okPar = False
                print "Error: bad path specified for the release"
	if( okPar == True ):
	        conn = common_db.createDBConnection()
		test = RegressionTest( conn )
                done = False
		ret = 0
                if( fflag == True and done==False ):
                    ret = test.runOnDb(LABEL, RELEASE, ARCH, PATH)
                    done = True
                if( sflag == True and done==False ):
                    ret = test.runOnReference(LABEL, RELEASE, ARCH, PATH, RELEASE, ARCH, PATH)
                    done = True
                if( REF_RELEASE != None and done==False ):
                    if( REF_ARCH == None ):
                        okPar = False
                        print "Error: missing -A (ref architecture) parameter"
                    if( REF_PATH == None):
                        okPar = False
                        print "Error: missing -P (ref path) parameter"
                    if(CheckPath(REF_RELEASE, REF_ARCH, REF_PATH) == False):
                        okPar = False
                        print "Error: bad path specified for the ref release"
                    if( okPar == True ):
                        ret = test.runOnReference(LABEL, RELEASE, ARCH, PATH, REF_RELEASE, REF_ARCH, REF_PATH)
			done = True
		if ( done == True and ret>0  ):
			test.finalize( wflag )
		conn.close()	
