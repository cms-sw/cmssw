try:
    import cx_Oracle
except ImportError, e:
    print "Cannot import cx_Oracle:", e
import common_db

class ResultsDB:
    def __init__(self, connect):
        self.conn = connect

    def create( self ):
	curs = self.conn.cursor()
	sqlstr = "CREATE TABLE TEST_STATUS (ID NUMBER, RUNID NUMBER, RDATE DATE, LABEL VARCHAR2(20), "
	sqlstr += "T_RELEASE VARCHAR2(50), T_ARCH VARCHAR2(30), R_RELEASE VARCHAR(50), R_ARCH VARCHAR2(30), LOG CLOB, "
	sqlstr += "CONSTRAINT PK_ID2 PRIMARY KEY(ID, RUNID, LABEL))"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "CREATE TABLE TEST_RESULTS (RID NUMBER, ID NUMBER, LABEL VARCHAR(20), NAME VARCHAR(100), STATUS NUMBER)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "CREATE SEQUENCE AUTO_INC INCREMENT BY 1 START WITH 1"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	curs = self.conn.cursor()
	sqlstr = "CREATE TABLE SEQUENCES (LABEL VARCHAR(20), ID NUMBER, RUNID NUMBER)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	self.conn.commit()
	print 'RESULTS DATABASE CREATED'
        
    def drop( self ):
	curs = self.conn.cursor()
	sqlstr = "DROP TABLE TEST_STATUS"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "DROP TABLE TEST_RESULTS"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "DROP SEQUENCE AUTO_INC"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "DROP TABLE SEQUENCES"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	self.conn.commit()
	print 'RESULT DATABASE DROPPED'
        
    def read( self ):
	curs = self.conn.cursor()
	sqlstr = "SELECT ID, RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH, R_RELEASE, R_ARCH, LOG "
	sqlstr +="FROM TEST_STATUS ORDER BY ID"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	print 'STATUS TABLE READ'
	print 'ID  RUNID RDATE   LABEL  T_RELEASE     T_ARCH      R_RELEASE     R_ARCH'
	for row in curs:
                print row[0:8]
                print self.readResults(row[0])
                if (not row[8]==None ): 
                    print row[8].read()
                    
    def readResults(self, id):
	curs = self.conn.cursor()
	sqlstr = "SELECT LABEL, NAME, STATUS FROM TEST_RESULTS WHERE ID = :ids"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids = id)
	self.conn.commit()
	for row in curs:
		print row

    def webStatusRunID(self, label):
	curs = self.conn.cursor()
	sqlstr = "SELECT MAX(RUNID) FROM TEST_STATUS WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	max = 0
	for row in curs:
		return row[0]
            
    def webStatusHeaders(self, label):
	curs = self.conn.cursor()
	sqlstr = "SELECT DISTINCT RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH "
	sqlstr +="FROM TEST_STATUS WHERE LABEL = :labl ORDER BY RUNID DESC"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl = label)
	result = curs.fetchall()
	return result
		
    def webLabels( self ):
	curs = self.conn.cursor()
	sqlstr = "SELECT DISTINCT LABEL FROM TEST_STATUS"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	return curs

    def webStatus( self, id, label):
	curs = self.conn.cursor()
	sqlstr = "SELECT ID, R_RELEASE, R_ARCH FROM TEST_STATUS WHERE RUNID = :ids AND LABEL = :labl ORDER BY R_RELEASE" 
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids=id, labl=label)
	result = curs.fetchall()
	return result
	
    def webResultsList( self, runID, label):
	curs = self.conn.cursor()
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
    
    def webReleasesHeaders( self, label, release="", arch=""):
	curs = self.conn.cursor()
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
    
    def webReadLogStatus( self, label, runID):
	curs = self.conn.cursor()
	sqlstr = "SELECT LOG FROM TEST_STATUS WHERE RUNID = :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = runID)
	for row in curs:
		return row[0].read()

    def getMaxRunIDStatus( self, label):
	curs = self.conn.cursor()
	sqlstr = "SELECT MAX(RUNID) FROM TEST_STATUS WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		if(row[0] is not None):
			return row[0]
		else:
			return 0
    def readCurrentStatus( self, runID):
	curs = self.conn.cursor()
	sqlstr = "SELECT ID, RUNID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH, R_RELEASE, R_ARCH "
	sqlstr +="FROM TEST_STATUS WHERE RUNID = :rid ORDER BY ID"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid= runID)
	print 'STATUS TABLE READ'
	print 'ID  RUNID RDATE  LABEL   T_RELEASE     T_ARCH      R_RELEASE     R_ARCH'
	for row in curs:
		print row
		print self.readResults(row[0])
                
    def writeStatus(self, runID, timeStamp, match, resTags):
	curs = self.conn.cursor()
	id = self.nextIDVal(match[0])
	sqlstr = "INSERT INTO TEST_STATUS(ID, RUNID, RDATE, LABEL, T_RELEASE, T_ARCH, R_RELEASE, R_ARCH)"
	sqlstr +="VALUES(:ids, :rid, :ts, :labl, "
	sqlstr +=":trel, :tarc, :rrel, :rarc)"
	curs.prepare(sqlstr)
        print 'lab=',match[0]
	curs.execute(sqlstr, ids = id , rid = runID, ts=timeStamp, labl=match[0], trel = match[1], tarc = match[2], rrel = match[3], rarc = match[4])
	self.conn.commit()
	for i in range(5, len(match)):
            if resTags[i-5] != "%NONE":
                self.writeResults(id, match[0], resTags[i-5], match[i])

    def writeResults(self, id, label, name, status):
	curs = self.conn.cursor()
	sqlstr = "INSERT INTO TEST_RESULTS(RID, ID, LABEL, NAME, STATUS)"
	sqlstr +="VALUES(:rid, :ids, :labl, :nam, :stat)"
	curs.prepare(sqlstr)
        rrid = self.getAutoIncResults()
	curs.execute(sqlstr, rid= rrid, ids = id, labl = label, nam = name, stat = status)
	self.conn.commit()

    def addLogStatus(self,label, runID, logStr):
	curs = self.conn.cursor()
	sqlstr = "UPDATE TEST_STATUS SET LOG = :lstr WHERE LABEL = :labl AND RUNID = :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, lstr = logStr, labl=label, rid=runID)
	self.conn.commit()

    def checkIfOkStatus( self, runID, label):
        curs = self.conn.cursor()
        sqlstr = "SELECT ID FROM TEST_STATUS WHERE RUNID = :rid AND LABEL = :labl"
        curs.prepare(sqlstr)
        curs.execute(sqlstr, rid = runID, labl = label)
        idList = []
        for row in curs:
            idList.append(row[0])
        curs = self.conn.cursor()
        sqlstr = "SELECT ID, STATUS FROM TEST_RESULTS WHERE ID >= :mi AND ID <= :ma AND LABEL = :labl"
        curs.prepare(sqlstr)
        #print "ID = %d label=%s minid=%d maxid=%d" %(runID,label, min(idList),max(idList))
	curs.execute(sqlstr, mi=min(idList), ma=max(idList), labl = label)
	for row in curs:
            if row[1] != 0:
                return False
        return True

    def getAutoIncResults( self ):
        curs = self.conn.cursor()
        sqlstr = "SELECT AUTO_INC.NextVal FROM DUAL"
        curs.prepare(sqlstr)
        curs.execute(sqlstr)
        for row in curs:
            return row[0]

    def createSequenceDB( self ):
	curs = self.conn.cursor()
	sqlstr = "CREATE TABLE SEQUENCES (LABEL VARCHAR(20), ID NUMBER, RUNID NUMBER)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	self.conn.commit()
        
    def writeSequence( self, label):
	curs = self.conn.cursor()
	sqlstr = "INSERT INTO SEQUENCES (LABEL, ID, RUNID) VALUES (:labl, 0, 0)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
        
    def readSequence(self, label):
	curs = self.conn.cursor()
	sqlstr = "SELECT * FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		print row
                
    def checkLabelSequence(self, label):
	curs = self.conn.cursor()
	sqlstr = "SELECT * FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		return True
	return False
    
    def dropSequenceDB( self ):
	curs = self.conn.cursor()
	sqlstr = "DROP TABLE SEQUENCES"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	self.conn.commit()
        
    def nextIDVal( self, label):
	curs = self.conn.cursor()
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
	self.conn.commit()
	curs = self.conn.cursor()
	sqlstr = "SELECT ID FROM SEQUENCES WHERE LABEL = :labl"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, labl=label)
	for row in curs:
		return row[0]
            
    def nextRunIDVal(self, label):
	curs = self.conn.cursor()
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

    def getDate( self ):
	curs = self.conn.cursor()
	sqlstr = "SELECT SYSTIMESTAMP AS \"NOW\" FROM DUAL"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	for row in curs:
		return row[0]

def GetWebStatus( id, label ):
    conn = common_db.createDBConnection()
    resDb = results_db.ResultsDB( conn )
    stat = resDb.webStatus( id, label )
    conn.close
    return stat

def GetWebResultList( runID, label ):
    conn = common_db.createDBConnection()
    resDb = results_db.ResultsDB( conn )
    resList = resDb.webResultList( runID, label )
    conn.close
    return resList

def GetWebLabels():
    conn = common_db.createDBConnection()
    resDb = results_db.ResultsDB( conn )
    webLabels = resDb.webLabels()
    conn.close
    return webLabels
    
def GetWebReleasesHeaders( label, release="", arch="" ):
    conn = common_db.createDBConnection()
    resDb = results_db.ResultsDB( conn )
    relHeaders = resDb.webReleasesHeaders( label, release, arch)
    conn.close
    return relHeaders

def GetWebStatusHeaders( label ):
    conn = common_db.createDBConnection()
    resDb = results_db.ResultsDB( conn )
    statusHeaders = resDb.webStatusHeaders( label )
    conn.close
    return statusHeaders

def GetWebReadLogStatus( label, runId ):
    conn = common_db.createDBConnection()
    resDb = results_db.ResultsDB( conn )
    logStatus = resDb.webReadLogStatus( label, runId )
    conn.close
    return logStatus
