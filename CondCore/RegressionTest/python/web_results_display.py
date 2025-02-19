import re
import os
import string
try:
    import cx_Oracle
except ImportError, e:
    print "Cannot import cx_Oracle:", e

#global varables
DATABASE = "cms_orcoff_prep"
USERNAME = "CMS_COND_REGRESSION"
#USERNAME = "CMS_COND_WEB"
AUTH_PATH = "/afs/cern.ch/cms/DB/conddb/test/authentication.xml"
#end

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

def createDBConnection():
    coralConnStr, USERNAME, PASSWORD, AUTH_PATH = getDBConnectionParams()
    conn_string = str(USERNAME+"/"+PASSWORD+"@"+DATABASE)
    conn = cx_Oracle.connect(conn_string)
    return conn
	
def getDBConnectionParams():
    os.environ['TNS_ADMIN'] = "/afs/cern.ch/project/oracle/admin"
    coralConnStr = "oracle://"+DATABASE+"/"+USERNAME+""
    PASSWORD = getLogin(AUTH_PATH, coralConnStr)
    return (coralConnStr,USERNAME,PASSWORD,AUTH_PATH)

class WebResultsDisplay:
    def __init__(self, connect):
        self.conn = connect
            
    def resultHeaders(self, label):
	curs = self.conn.cursor()
	sqlstr = "SELECT RID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH "
	sqlstr +="FROM RUN_HEADER WHERE LABEL = :labl ORDER BY RID DESC"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl = label)
	result = curs.fetchall()
	return result
		
    def labels( self ):
	curs = self.conn.cursor()
	sqlstr = "SELECT DISTINCT LABEL FROM RUN_HEADER"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	return curs

    def runResults( self, rid ):
	curs = self.conn.cursor()
	sqlstr = "SELECT ID, R_RELEASE, R_ARCH FROM RUN_RESULT WHERE RID = :ids ORDER BY R_RELEASE" 
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids=rid)
	result = curs.fetchall()
	return result
	
    def stepResults( self, id ):
	curs = self.conn.cursor()
	sqlstr = "SELECT ID, STATUS, STEP_LABEL FROM RUN_STEP_RESULT WHERE ID= :ids"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids=id)
	result = curs.fetchall()
	count = len(result)
	return result, count
    
    def releasesHeaders( self, label, release="", arch=""):
	curs = self.conn.cursor()
	if(release != "" and arch != ""):
		sqlstr = "SELECT RID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH FROM RUN_HEADER WHERE T_RELEASE = :rel AND T_ARCH = :arc AND LABEL = :labl ORDER BY RID DESC"
		curs.prepare(sqlstr)
		curs.execute(sqlstr, rel=release, arc=arch, labl=label)
	elif(release != ""):
		sqlstr = "SELECT RID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH FROM RUN_HEADER WHERE T_RELEASE = :rel AND LABEL = :labl ORDER BY RID DESC"
		curs.prepare(sqlstr)
		curs.execute(sqlstr, rel=release, labl=label)
	elif(arch != ""):
		sqlstr = "SELECT RID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH FROM RUN_HEADER WHERE T_ARCH = :arc AND LABEL = :labl ORDER BY RID DESC"
		curs.prepare(sqlstr)
		curs.execute(sqlstr, arc=arch, labl=label)
	result = curs.fetchall()
	return result
    
    def readLogStatus( self, label, runID):
	curs = self.conn.cursor()
	sqlstr = "SELECT LOG FROM RUN_HEADER WHERE RID = :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = runID)
	for row in curs:
		return row[0].read()

def GetRunResults( rid ):
    conn = createDBConnection()
    resDb = WebResultsDisplay( conn )
    stat = resDb.runResults( rid )
    conn.close
    return stat

def GetResultsList( runID ):
    conn = createDBConnection()
    resDb = WebResultsDisplay( conn )
    resList = resDb.stepResults( runID )
    conn.close
    return resList

def GetLabels():
    conn = createDBConnection()
    resDb = WebResultsDisplay( conn )
    webLabels = resDb.labels()
    conn.close
    return webLabels
    
def GetReleasesHeaders( label, release="", arch="" ):
    conn = createDBConnection()
    resDb = WebResultsDisplay( conn )
    relHeaders = resDb.releasesHeaders( label, release, arch)
    conn.close
    return relHeaders

def GetResultHeaders( label ):
    conn = createDBConnection()
    resDb = WebResultsDisplay( conn )
    statusHeaders = resDb.resultHeaders( label )
    conn.close
    return statusHeaders

def GetReadLogStatus( label, runId ):
    conn = createDBConnection()
    resDb = WebResultsDisplay( conn )
    logStatus = resDb.readLogStatus( label, runId )
    conn.close
    return logStatus
