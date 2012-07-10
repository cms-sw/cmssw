#!/usr/bin/env python
import sys
import re
import os
import shlex, subprocess
import getopt
import random
import time
from xml.dom.minidom import parse, parseString
import string
try:
    import cx_Oracle
except ImportError, e:
    print "Cannot import cx_Oracle:", e
	
#global varables
resCount = 0
resNames = []
mLabel = ''
DATABASE = "cms_orcoff_prep"
#USERNAME = "CMS_COND_REGRESSION"
USERNAME = "CMS_COND_WEB"
AUTH_PATH = "/afs/cern.ch/cms/DB/conddb/test/authentication.xml"
#end

def getText(nodelist):
    rc = []
    for node in nodelist:
		if node.nodeType == node.TEXT_NODE:
			rc.append(node.data)
    return ''.join(rc)
def ParseXML(filename, label):
	results = []
	initResults = []
	finalResults = []
	dom = parse(filename)
	xml = dom.getElementsByTagName("xml")
	execTest = False
	env = []
	if xml != None:
		for xm in xml:
			test = xm.getElementsByTagName("test")
			for it in test:
				#print it.toxml()
				if "name" in  it.attributes.keys():
					tLabel = str(it.attributes["name"].value)
					if  tLabel == label:
						execTest = True
						global mLabel
						mLabel = label
				if execTest == True:
					inits = it.getElementsByTagName("init")
					finals = it.getElementsByTagName("final")
					seqs = it.getElementsByTagName("sequence")
					for init in inits:
						commands = init.getElementsByTagName("command")
						for command in commands:
							if "exec" in  command.attributes.keys():
								initResults.append(str(command.attributes["exec"].value))
							if "env" in  command.attributes.keys():
									env.append(str(command.attributes["env"].value))
					for seq in seqs:
						commands = seq.getElementsByTagName("command")
						for command in commands:
							if "exec" in  command.attributes.keys():
								results.append(str(command.attributes["exec"].value))
							if "result" in  command.attributes.keys():
                                                                global resNames
                                                                resNames.append(str(command.attributes["result"].value))
                                                        else: 
                                                                resNames.append("%NONE")
							if "env" in  command.attributes.keys():
                                                                env.append(str(command.attributes["env"].value))
					for final in finals:
						commands = final.getElementsByTagName("command")
						for command in commands:
							if "exec" in  command.attributes.keys():
								finalResults.append(str(command.attributes["exec"].value))
							if "env" in  command.attributes.keys():
									env.append(str(command.attributes["env"].value))
				execTest = False
	return initResults,results, finalResults,  env
def extractLogin(login):
	pattern = re.compile(r'value="([^"]+)')
	matching = pattern.search(login)
	version = 0
	if matching:
		g = matching.groups()
		return g[0]
def getLogin(auth, connStr):
	pfile = open(auth, "r")
	plist = pfile.readlines()
	for i in range (0, len(plist)):
		if string.find(plist[i], '<connection name="'+connStr+'">') != -1:
			PASSWORD = extractLogin(plist[i+2])
	return (PASSWORD)
	
def setConn():
	os.environ['TNS_ADMIN'] = "/afs/cern.ch/project/oracle/admin"
	coralConnStr = "oracle://"+DATABASE+"/"+USERNAME+""
	PASSWORD = getLogin(AUTH_PATH, coralConnStr)
	conn_string = str(USERNAME+"/"+PASSWORD+"@"+DATABASE)	
	conn = cx_Oracle.connect(conn_string)	
	return {'r0':conn, 'r1':coralConnStr, 'r2':USERNAME, 'r3':PASSWORD, 'r4':AUTH_PATH}
def WebStatusRunID(label):
	result = setConn()
	con = result.get('r0')
	curs = con.cursor()
	sqlstr = "SELECT MAX(RUNID) FROM TEST_STATUS WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	max = 0
	for row in curs:
		return row[0]
def WebStatusHeaders(label):
	result = setConn()
	con = result.get('r0')
	curs = con.cursor()
	sqlstr = "SELECT DISTINCT RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH "
	sqlstr +="FROM TEST_STATUS WHERE LABEL = :labl ORDER BY RUNID DESC"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl = label)
	result = curs.fetchall()
	return result
		
def WebLabels():
	result = setConn()
	con = result.get('r0')
	curs = con.cursor()
	sqlstr = "SELECT DISTINCT LABEL FROM TEST_STATUS"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	return curs

def WebStatus(id, label):
	result = setConn()
	con = result.get('r0')
	curs = con.cursor()
	sqlstr = "SELECT ID, R_RELEASE, R_ARCH FROM TEST_STATUS WHERE RUNID = :ids AND LABEL = :labl ORDER BY R_RELEASE" 
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids=id, labl=label)
	result = curs.fetchall()
	return result
	
def WebResultsList(runID, label):
	result = setConn()
	con = result.get('r0')
	curs = con.cursor()
	sqlstr = "SELECT ID FROM TEST_STATUS WHERE RUNID = :rid AND LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = runID, labl = label)
	idList = []
	for row in curs:
		idList.append(row[0])
	sqlstr = "SELECT ID, STATUS, NAME FROM TEST_RESULTS WHERE ID >= :mi AND ID <= :ma AND LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, mi=min(idList), ma=max(idList), labl = label)
	result = curs.fetchall()
	count = len(result)/len(idList)
	print count
	return result, count
def WebReleasesHeaders(label, release="", arch=""):
	result = setConn()
	con = result.get('r0')
	curs = con.cursor()
	if(release != "" and arch != ""):
		sqlstr = "SELECT DISTINCT RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH FROM TEST_STATUS WHERE T_RELEASE = :rel AND T_ARCH = :arc AND LABEL = :labl ORDER BY RUNID DESC"
		curs.prepare(sqlstr)
		curs.execute(sqlstr, rel=release, arc=arch, labl=label)
	elif(release != ""):
		sqlstr = "SELECT DISTINCT RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH FROM TEST_STATUS WHERE T_RELEASE = :rel AND LABEL = :labl ORDER BY RUNID DESC"
		curs.prepare(sqlstr)
		curs.execute(sqlstr, rel=release, labl=label)
	elif(arch != ""):
		sqlstr = "SELECT DISTINCT RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH FROM TEST_STATUS WHERE T_ARCH = :arc AND LABEL = :labl ORDER BY RUNID DESC"
		curs.prepare(sqlstr)
		curs.execute(sqlstr, arc=arch, labl=label)
	result = curs.fetchall()
	return result
def WebReadLogStatusDB(label, runID):
	result = setConn()
	con = result.get('r0')
	curs = con.cursor()
	sqlstr = "SELECT LOG FROM TEST_STATUS WHERE RUNID = :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = runID)
	for row in curs:
		return row[0].read()
def BuildLogfile(label, runID):
	logFile = open("logs/TestLog"+str(runID)+".txt", "w")
	logFile.write('\Output:'+stdout_value)
	

def extractID(release):
	pattern = re.compile("^CMSSW_(\d+)_(\d+)_(\d+|\D)(_pre(\d+)|_patch(\d+))?")
	matching = pattern.match(release)
	version = 0
	if matching:
		g = matching.groups()
		if(g[2].isdigit()):
			if(g[4] is not None and g[4].isdigit()):
					version = int(g[0]) * 1000000 + int(g[1]) * 10000 + int(g[2]) * 100 + int(g[4])
			else:
				version = int(g[0]) * 1000000 + int(g[1]) * 10000 + int(g[2]) * 100
		else:
			version = int(g[0]) * 1000000 + int(g[1]) * 10000 +9999
		if(version is not None):
			return version
def extractErr(runID, stdoutStr):
	reStr = "\!L\!([^!]+)\!TR\!([^!]+)\!TA\!([^!]+)\!RR\!([^!]+)\!RA\!([^!]+)"
	for i in range (0, resCount):
		reStr +=  "\!C"+str(i)+"\!(\d+)"

	pattern = re.compile(reStr)
	matching = pattern.findall(stdoutStr)
	stdoutMod = pattern.sub("", stdoutStr)
	for match in matching:
		WriteStatusDB(runID, match)
	return stdoutMod

def CreateReleaseDB():
	curs = conn.cursor()
	sqlstr = "CREATE TABLE VERSION_TABLE (ID NUMBER, RELEASE VARCHAR2(50), ARCH VARCHAR2(30), PATH VARCHAR(255), CONSTRAINT PK_ID PRIMARY KEY(RELEASE, ARCH) )"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	print 'VERSION TABLE CREATED'
def DropReleaseDB():
	curs = conn.cursor()
	sqlstr = "DROP TABLE VERSION_TABLE"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	print 'VERSION TABLE DROPPED'
def DeleteReleaseDB(release, arch):
	curs = conn.cursor()
	sqlstr = "DELETE FROM VERSION_TABLE WHERE RELEASE = :rel AND ARCH = :arc"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rel = release, arc = arch)
	conn.commit()
	print 'VALUE DELETED'
def ReadReleaseDB():
	curs = conn.cursor()
	sqlstr = "SELECT ID, RELEASE, ARCH, PATH FROM VERSION_TABLE ORDER BY ID, RELEASE, ARCH"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	print 'VERSION TABLE READ'
        print 'ID     RELEASE     ARCH     PATH'
	for row in curs:
		print row
		
def CreateStatusDB():
	curs = conn.cursor()
	sqlstr = "CREATE TABLE TEST_STATUS (ID NUMBER, RUNID NUMBER, RDATE DATE, LABEL VARCHAR2(20), "
	sqlstr += "T_RELEASE VARCHAR2(50), T_ARCH VARCHAR2(30), R_RELEASE VARCHAR(50), R_ARCH VARCHAR2(30), LOG CLOB, "
	sqlstr += "CONSTRAINT PK_ID2 PRIMARY KEY(ID, RUNID, LABEL))"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	CreateResultsDB()
	CreateSequenceDB()
	conn.commit()
	print 'STATUS TABLE CREATED'
def DropStatusDB():
	curs = conn.cursor()
	sqlstr = "DROP TABLE TEST_STATUS"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	DropResultsDB()
	DropSequenceDB()
	print 'STATUS TABLE DROPPED'
def ReadStatusDB():
	curs = conn.cursor()
	sqlstr = "SELECT ID, RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH, R_RELEASE, R_ARCH, LOG "
	sqlstr +="FROM TEST_STATUS ORDER BY ID"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	print 'STATUS TABLE READ'
	print 'ID  RUNID RDATE   LABEL  T_RELEASE     T_ARCH      R_RELEASE     R_ARCH'
	for row in curs:
                print row[0:8]
                print ReadResultsDB(row[0])
                print row[8].read() 
def GetMaxRunIDStatusDB(label):
	curs = conn.cursor()
	sqlstr = "SELECT MAX(RUNID) FROM TEST_STATUS WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		if(row[0] is not None):
			return row[0]
		else:
			return 0
def ReadCurrentStatusDB(runID):
	curs = conn.cursor()
	sqlstr = "SELECT ID, RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH, R_RELEASE, R_ARCH "
	sqlstr +="FROM TEST_STATUS WHERE RUNID = :rid ORDER BY ID"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid= runID)
	print 'STATUS TABLE READ'
	print 'ID  RUNID RDATE  LABEL   T_RELEASE     T_ARCH      R_RELEASE     R_ARCH'
	for row in curs:
		print row
		print ReadResultsDB(row[0])
def WriteStatusDB(runID, match):
	curs = conn.cursor()
	id = NextIDVal(match[0])
	sqlstr = "INSERT INTO TEST_STATUS(ID, RUNID, RDATE, LABEL, T_RELEASE, T_ARCH, R_RELEASE, R_ARCH)"
	sqlstr +="VALUES(:ids, :rid, :ts, :labl, "
	sqlstr +=":trel, :tarc, :rrel, :rarc)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids = id , rid = runID, ts=timeStamp, labl=match[0], trel = match[1], tarc = match[2], rrel = match[3], rarc = match[4])
	conn.commit()
	for i in range(5, len(match)):
                if resNames[i-5] != "%NONE":
                        WriteResultsDB(id, label, resNames[i-5], match[i])
def AddLogStatusDB(label, runID, logStr):
	curs = conn.cursor()
	sqlstr = "UPDATE TEST_STATUS SET LOG = :lstr WHERE LABEL = :labl AND RUNID = :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, lstr = logStr, labl=label, rid=runID)
	conn.commit()
def CheckIfOkStatusDB(runID, label):
	curs = conn.cursor()
	sqlstr = "SELECT ID FROM TEST_STATUS WHERE RUNID = :rid AND LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = runID, labl = label)
	idList = []
	for row in curs:
		idList.append(row[0])
	curs = conn.cursor()
	sqlstr = "SELECT ID, STATUS FROM TEST_RESULTS WHERE ID >= :mi AND ID <= :ma AND LABEL = :labl"
	curs.prepare(sqlstr)
        #print "ID = %d label=%s minid=%d maxid=%d" %(runID,label, min(idList),max(idList))
	curs.execute(sqlstr, mi=min(idList), ma=max(idList), labl = label)
	for row in curs:
		if row[1] != 0:
			return False
	return True

def CreateResultsDB():	
	curs = conn.cursor()
	sqlstr = "CREATE TABLE TEST_RESULTS (RID NUMBER, ID NUMBER, LABEL VARCHAR(20), NAME VARCHAR(100), STATUS NUMBER)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "CREATE SEQUENCE AUTO_INC INCREMENT BY 1 START WITH 1"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	conn.commit()
def DropResultsDB():
	curs = conn.cursor()
	sqlstr = "DROP TABLE TEST_RESULTS"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "DROP SEQUENCE AUTO_INC"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	conn.commit()
def WriteResultsDB(id, label, name, status):
	curs = conn.cursor()
	sqlstr = "INSERT INTO TEST_RESULTS(RID, ID, LABEL, NAME, STATUS)"
	sqlstr +="VALUES(:rid, :ids, :labl, :nam, :stat)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = GetAutoIncResultsDB(), ids = id, labl = label, nam = name, stat = status)
	conn.commit()
def ReadResultsDB(id):
	curs = conn.cursor()
	sqlstr = "SELECT LABEL, NAME, STATUS FROM TEST_RESULTS WHERE ID = :ids"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids = id)
	conn.commit()
	for row in curs:
		print row
def GetAutoIncResultsDB():
	curs = conn.cursor()
	sqlstr = "SELECT AUTO_INC.NextVal FROM DUAL"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	for row in curs:
		return row[0]
	
def CreateSequenceDB():
	curs = conn.cursor()
	sqlstr = "CREATE TABLE SEQUENCES (LABEL VARCHAR(20), ID NUMBER, RUNID NUMBER)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	conn.commit()
def WriteSequenceDB(label):
	curs = conn.cursor()
	sqlstr = "INSERT INTO SEQUENCES (LABEL, ID, RUNID) VALUES (:labl, 0, 0)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
def ReadSequenceDB(label):
	curs = conn.cursor()
	sqlstr = "SELECT * FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		print row
def CheckLabelSequenceDB(label):
	curs = conn.cursor()
	sqlstr = "SELECT * FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		return True
	return False
def DropSequenceDB():
	curs = conn.cursor()
	sqlstr = "DROP TABLE SEQUENCES"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	conn.commit()
def NextIDVal(label):
	curs = conn.cursor()
	sqlstr = "SELECT ID FROM SEQUENCES WHERE LABEL = :labl FOR UPDATE"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	maxID = 0
	for row in curs:
		maxID = row[0]
	maxID += 1
	sqlstr = "UPDATE SEQUENCES SET ID = :maxid WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, maxid = maxID, labl=label)
	conn.commit()
	curs = conn.cursor()
	sqlstr = "SELECT ID FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		return row[0]
def NextRunIDVal(label):
	curs = conn.cursor()
	sqlstr = "SELECT RUNID FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	maxID = 0
	for row in curs:
		maxID = row[0]
	maxID += 1
	sqlstr = "UPDATE SEQUENCES SET RUNID = :maxid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, maxid = maxID)
	sqlstr = "SELECT RUNID FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		return row[0]
		
def GetDate():
	curs = conn.cursor()
	sqlstr = "SELECT SYSTIMESTAMP AS \"NOW\" FROM DUAL"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	for row in curs:
		return row[0]	
def TestCompat(label, release, arch, path):
	relID = extractID(release)
	print "Testing exec "+label+" "+release+" "+arch+" from "
	print path+" : "
	curs = conn.cursor()
	cmds = """
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Candidate release: """+release+""" "
echo "Arch: """+arch+""" "
echo "Path: """+path+""" "
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"""
	sqlstr = "SELECT RELEASE, ARCH, PATH FROM VERSION_TABLE WHERE ID < :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = relID)
	for row in curs:
		cmds += RunTest(label, release, arch, path, row[0], row[1], row[2])
	return cmds
def TestCompatRef(label, release, arch, path, refRelease, refArch, refPath):
	relID = extractID(release)
	print "Testing "+release+" "+arch+" from "
	print path+" : "
	curs = conn.cursor()
	cmds = """
echo ""
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Candidate release: """+release+""" "
echo "Arch: """+arch+""" "
echo "Path: """+path+""" "
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
"""
	cmds += RunTest(label, release, arch, path, refRelease, refArch, refPath)
	return cmds
def WriteReleaseDB(release, arch, path):
	curs = conn.cursor()
	relID = extractID(release)
	print "relID "+str(relID)
	sqlstr = "INSERT INTO VERSION_TABLE(ID, RELEASE, ARCH, PATH) VALUES(:rid, :rel, :arc, :pat)"
	curs.execute(sqlstr, rid = relID, rel = release, arc = arch, pat = path)
	conn.commit()
	print 'TABLE WRITTEN'	
def SetEnv(release, arch, path):
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
	TMAINDB="""+coralConnStr+"""
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
def RunTest(label, release, arch, path, refRelease, refArch, refPath ):
	cmds ="""
	RETVAL=0
echo "*****************************************************************************************"
echo "Reference release: """+refRelease+""" "
echo "Arch: """+refArch+""" "
echo "Path: """+refPath+""" "
echo "*****************************************************************************************"
"""
	nr = 0
	enr = 0
	envType = 0
	print "init"
	for res in iResults:
                cmds += """
echo "==============================================="
"""
		print res
		if setEnvs[enr] == "ref" and envType != 1:
			cmds += SetEnv(refRelease, refArch,refPath)
			envType = 1
		elif setEnvs[enr] == "cand" and envType != 2:
			cmds += SetEnv(release, arch, path)
			envType = 2
		cmds +=Command(res)
		enr +=1
	print "cmd"
	for res in pResults:
                cmds += """
echo "==============================================="
"""
		print res
		if setEnvs[enr] == "ref" and envType != 1:
			cmds += SetEnv(refRelease, refArch,refPath)
			envType = 1
		elif setEnvs[enr] == "cand" and envType != 2:
			cmds += SetEnv(release, arch, path)
			envType = 2
		cmds +=Command(res)
		cmds += 'RCODE['+str(nr)+']=$RETVAL'
		nr +=1
		enr +=1
	print "final"
	cmds += """
	RETVAL=0
	"""
	for res in fResults:
                cmds += """
echo "==============================================="
"""
		print res
		if setEnvs[enr] == "ref" and envType != 1:
			cmds += SetEnv(refRelease, refArch,refPath)
			envType = 1
		elif setEnvs[enr] == "cand" and envType != 2:
			cmds += SetEnv(release, arch, path)
			envType = 2
		cmds +=Command(res)
		enr +=1
	global resCount
	resCount = nr
	cmds += """
echo "==============================================="
        echo "Script return code : ${RCODE[1]}"
echo "!L!"""+label+"""!TR!"""+release+"""!TA!"""+arch+"""!RR!"""+refRelease+"""!RA!"""+refArch
	for i in range (0, nr):
		cmds+= "!C"+str(i)+"!${RCODE["+str(i)+"]}"
	cmds += "\""
	return cmds
def CmdUsage():
	print "Command line arguments :"
	print "-c (-s) creates descriptor(status) db schema"
	print "-d drops descriptor(status) db schema. Optional : -R [release] -A [arch] to drop single entry (for descriptor db)"
	print "-w writes data to db. Goes only with -R [release] -A [arch] -P [path]"
	print "-r (-s) reads contents of descriptor(status) db"
	print "-t (-o) (-i) runs test. Goes only with -L [label] -R [release] -A [arch] -P [path] " 
	print "(-o) specifies reference release. supply additional parameters --R [refRelease] --A [refArch] --P [refPath]"
	print "(-i) argument forces candidate to test with itself as a reference"
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
def ReadArgs():
	try:
		opts, args = getopt.getopt(sys.argv[1:], "cdwrthsoiL:R:A:P:", ['R=', 'A=', 'P='])
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
	wflag = False
	tflag = False
	dflag = False
	sflag = False
	cflag = False
	rflag = False
	oflag = False
	iflag = False
	cmds = ""
	for o, a in opts:
		if o == "-c":
			cflag = True
		elif o == "-d":
			dflag = True
		elif o == "-w":
			wflag = True
		elif o == "-r":
			rflag = True
		elif o == "-t":
			tflag = True
		elif o == "-s":
			sflag = True
		elif o == "-o":
			oflag = True
		elif o == "-i":
			iflag = True
		elif o == "-L":
			LABEL = a
		elif o == "-R":
			RELEASE = a
		elif o == "-A":
			ARCH = a
		elif o == "-P":
			PATH = a
		elif o in ("-K", "--R"):
			REF_RELEASE = a
		elif o in ("-L", "--A"):
			REF_ARCH = a
		elif o in ("-M", "--P"):
			REF_PATH = a
		elif o == "-h":
			CmdUsage()
		else:
			assert False, "unhandled option"
	if(cflag == True):
		if(sflag == True):
			CreateStatusDB()
		else:
			CreateReleaseDB()
	if(rflag == True):
		if(sflag == True):
			ReadStatusDB()
		else:
			ReadReleaseDB()
	if(wflag == True):
		if(RELEASE != None and ARCH != None and PATH != None):
				WriteReleaseDB(RELEASE, ARCH, PATH)
		else:
			print "Bad arguments for -w"
	if(dflag == True):
		if(sflag == True):
			DropStatusDB();
		else:
			if(RELEASE != None and ARCH != None):
				DeleteReleaseDB(RELEASE, ARCH)
			else:
				DropReleaseDB();
	if(tflag == True): 
		if LABEL != None :
			global pResults, iResults, fResults, setEnvs
			iResults,pResults, fResults , setEnvs = ParseXML("sequences.xml", LABEL)
			if(iResults == None or pResults == None or fResults == None):
				print "Error : sequences.xml reading failed!"
			if(RELEASE != None and ARCH != None and PATH != None):
				if(oflag == True):
					if(REF_RELEASE != None and REF_ARCH != None and REF_PATH != None):
						if(CheckPath(RELEASE, ARCH, PATH) == True):
							if(CheckPath(REF_RELEASE, REF_ARCH, REF_PATH) == True):
								cmds=TestCompatRef(LABEL, RELEASE, ARCH, PATH, REF_RELEASE, REF_ARCH, REF_PATH)
							else :
								print "Bad reference release arguments"
						else :
							print "Bad test release arguments"
					else:
						print "Bad arguments for -t"
				else:
					if(CheckPath(RELEASE, ARCH, PATH) == True):
						cmds = ''
						if iflag == True:
							cmds += TestCompatRef(LABEL, RELEASE, ARCH, PATH, RELEASE, ARCH, PATH)
						cmds += TestCompat(LABEL, RELEASE, ARCH, PATH)
					else :
						print "Bad test release arguments"
		else:
			print "Bad arguments for -t"
	return (cmds, LABEL)

result = setConn()
conn, coralConnStr, USERNAME, PASSWORD,AUTH_PATH = result.get('r0'), result.get('r1'), result.get('r2'), result.get('r3'), result.get('r4')

cmdList, label = ReadArgs()
#print cmdList
timeStamp = GetDate()
runID =0

if cmdList != "":
	cmdList+="""
	echo "End of test"
        echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" """
	pipe = subprocess.Popen(cmdList, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	stdout_value = pipe.communicate()[0]
	
	if CheckLabelSequenceDB(label) == False:
		WriteSequenceDB(label)
	runID = NextRunIDVal(label)
	stdout_value = extractErr(runID, stdout_value)	
	print '\Output:', stdout_value
	AddLogStatusDB(label, runID, stdout_value)
        print "Test '%s' runID=%d" %(label, runID)
        stat = "SUCCESS"
	if(CheckIfOkStatusDB(runID, label) == False):
                stat = "FAILURE"
        print "Exit status=%s" %stat
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
conn.close()
