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

