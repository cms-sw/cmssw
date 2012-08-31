try:
    import cx_Oracle
except ImportError, e:
    print "Cannot import cx_Oracle:", e
import common_db

def printRun( row ):
    print '============================================================='
    print 'run=%s  date=[%s]  test=%s'%(row[0],row[1],row[2])
    print 'cand=[%s on %s]'%(row[3],row[4])

def printResult( row ):
    print '-------------------------------------------------------------'
    print 'ref=[%s on %s]'%(row[1],row[2])

def printStepResult( row ):
    step=row[0]
    code = row[1]
    check = 'v'
    marker = '         '
    if (code!=0):
        marker = ' <----ERROR'
        check = 'x'
    if (step==None):
        step='setting up'
    print '%s [%s]=%d %s'%(check,step,code,marker) 


class ResultsDB:
    def __init__(self, connect):
        self.conn = connect

    def create( self ):
	curs = self.conn.cursor()
	sqlstr = "CREATE TABLE RUN_HEADER (RID NUMBER, RDATE DATE, LABEL VARCHAR2(20), "
	sqlstr += "T_RELEASE VARCHAR2(50), T_ARCH VARCHAR2(30), LOG CLOB "
	sqlstr += "CONSTRAINT PK_ID0 PRIMARY KEY(RID))"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "CREATE TABLE RUN_RESULT (ID NUMBER, RID NUMBER, R_RELEASE VARCHAR(50), R_ARCH VARCHAR2(30) "
	sqlstr += "CONSTRAINT PK_ID PRIMARY KEY(ID))"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "CREATE TABLE RUN_STEP_RESULT (ID NUMBER, STEP_LABEL VARCHAR(100), STATUS NUMBER)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "CREATE SEQUENCE RES_ID_SEQ INCREMENT BY 1 START WITH 1"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	self.conn.commit()
	print 'RESULTS DATABASE CREATED'
        
    def drop( self ):
	curs = self.conn.cursor()
	sqlstr = "DROP TABLE RUN_HEADER"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "DROP TABLE RUN_RESULT"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "DROP TABLE RUN_STEP_RESULT"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	sqlstr = "DROP SEQUENCE RES_ID_SEQ"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	self.conn.commit()
	print 'RESULT DATABASE DROPPED'
        
    def read( self ):
	curs = self.conn.cursor()
	sqlstr = "SELECT RID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH "
	sqlstr +="FROM RUN_HEADER ORDER BY RID"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
        for row in curs:
            printRun( row )
            self.readRun( row[0] )

    def readRun( self, rid ):
	curs = self.conn.cursor()
        sqlstr = "SELECT ID, R_RELEASE, R_ARCH "
        sqlstr +="FROM RUN_RESULT WHERE RID=:rids ORDER BY ID"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rids=rid)
	for row in curs:
                printResult( row )
                self.readResults(row[0])
                    
    def readResults(self, id):
	curs = self.conn.cursor()
	sqlstr = "SELECT STEP_LABEL, STATUS FROM RUN_STEP_RESULT WHERE ID = :ids"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids = id)
	self.conn.commit()
	for row in curs:
            printStepResult( row )

    def readSelection( self, runId, label, trel, tarch, full  ):
	curs = self.conn.cursor()
	sqlstr = "SELECT RID, TO_CHAR(RDATE, 'DD.MM.YYYY HH24:MI:SS'), LABEL, T_RELEASE, T_ARCH, LOG "
	sqlstr +="FROM RUN_HEADER "
        putAnd = False
        if( runId != None ):
            sqlstr += "WHERE RID="+runId
            putAnd = True
        if( label != None ):
            if( putAnd == True ):
                sqlstr += " AND "
            else:
                sqlstr += "WHERE "
            sqlstr += "LABEL='"+label+"'"
            putAnd = True
        if( trel != None ):
            if( putAnd == True ):
                sqlstr += " AND "
            else:
                sqlstr += "WHERE "
            sqlstr += "T_RELEASE='"+trel+"'"
            putAnd = True
        if( tarch != None ):
            if( putAnd == True ):
                sqlstr += " AND "
            else:
                sqlstr += "WHERE "
            sqlstr += "T_ARCH='"+tarch+"'"
            putAnd = True
            sqlstr += " ORDER BY RID"
        print sqlstr
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	for row in curs:
            print '###########'
            print row[0]
            printRun( row )
            self.readRun( row[0] )
            if( full == True ):
                if ( row[8]!=None ): 
                    print row[8].read()

    def deleteRun( self,rid ):
        curs = self.conn.cursor()
        sqlstr = "SELECT ID FROM RUN_RESULT WHERE RID = :rids"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rids=rid)
        for row in curs:
            self.deleteResultSteps( row[0] )
        sqlstr = "DELETE FROM RUN_RESULT WHERE RID = :rids"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rids=rid)
        sqlstr = "DELETE FROM RUN_HEADER WHERE RID = :rids"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rids=rid)
	self.conn.commit()         

    def deleteResultSteps( self, id ):
        curs = self.conn.cursor()
        sqlstr = "DELETE FROM RUN_STEP_RESULT WHERE ID=:ids"
        curs.prepare(sqlstr)
        curs.execute(sqlstr, ids=id)
                
    def writeResult(self, runID, timeStamp, match, resTags):
	curs = self.conn.cursor()
        sqlstr = "SELECT RID FROM RUN_HEADER WHERE RID = :rid"
	curs.prepare(sqlstr)
        curs.execute(sqlstr, rid = runID)
        foundRun = False
        for row in curs:
            foundRun = True
        if( foundRun == False ):
            sqlstr = "INSERT INTO RUN_HEADER(RID, RDATE, LABEL, T_RELEASE, T_ARCH) VALUES (:rid, :ts, :labl, :trel, :tarc)"
            curs.prepare(sqlstr)
            curs.execute(sqlstr, rid = runID, ts=timeStamp, labl=match[0], trel = match[1], tarc = match[2])    
	id = self.getNewResId()  
	sqlstr = "INSERT INTO RUN_RESULT(ID, RID, R_RELEASE, R_ARCH)"
	sqlstr +="VALUES(:ids, :rid, :rrel, :rarc)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids=id, rid = runID, rrel = match[3], rarc = match[4])
	self.conn.commit()
	for i in range(5, len(match)):
            if resTags[i-5] != "%NONE":
                self.writeStepResult(id, resTags[i-5], match[i])

    def writeStepResult(self, id, step_label, status):
	curs = self.conn.cursor()
	sqlstr = "INSERT INTO RUN_STEP_RESULT(ID, STEP_LABEL, STATUS)"
	sqlstr +="VALUES(:ids, :labl, :stat)"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, ids=id, labl=step_label, stat = status)
	self.conn.commit()

    def addResultLog(self, runID, logStr):
	curs = self.conn.cursor()
	sqlstr = "UPDATE RUN_HEADER SET LOG = :lstr WHERE RID = :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, lstr = logStr, rid=runID)
	self.conn.commit()

    def checkResult( self, runID ):
        curs = self.conn.cursor()
        sqlstr = "SELECT ID FROM RUN_RESULT WHERE RID=:rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid=runID)
        for row in curs:
            innercur = self.conn.cursor()
            isqlstr = "SELECT STATUS FROM RUN_STEP_RESULT WHERE ID= :ids"
            innercur.prepare(isqlstr)
            innercur.execute(isqlstr, ids=row[0])
            for irow in innercur:
                if irow[0] != 0:
                    return False
        return True

    def getNewResId( self ):
        curs = self.conn.cursor()
        sqlstr = "SELECT RES_ID_SEQ.NextVal FROM DUAL"
        curs.prepare(sqlstr)
        curs.execute(sqlstr)
        for row in curs:
            return row[0]

    def getNewRunId( self ):
        curs = self.conn.cursor()
        sqlstr = "SELECT RUN_ID_SEQ.NextVal FROM DUAL"
        curs.prepare(sqlstr)
        curs.execute(sqlstr)
        for row in curs:
            return row[0]

    def getDate( self ):
	curs = self.conn.cursor()
	sqlstr = "SELECT SYSTIMESTAMP AS \"NOW\" FROM DUAL"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	for row in curs:
		return row[0]

