import re
try:
    import cx_Oracle
except ImportError, e:
    print "Cannot import cx_Oracle:", e

def ExtractID(release):
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

class ReferenceDB:
    def __init__(self, connect):
        self.conn = connect

    def create( self ):
	curs = self.conn.cursor()
	sqlstr = "CREATE TABLE VERSION_TABLE (ID NUMBER, RELEASE VARCHAR2(50), ARCH VARCHAR2(30), PATH VARCHAR(255), CONSTRAINT PK_ID PRIMARY KEY(RELEASE, ARCH) )"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	print 'REFERENCE RELEASE TABLE CREATED'
    def drop( self ):
	curs = self.conn.cursor()
	sqlstr = "DROP TABLE VERSION_TABLE"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
	print 'REFERENCE RELEASE TABLE DROPPED'
    def deleteRelease( self, release, arch):
	curs = self.conn.cursor()
	sqlstr = "DELETE FROM VERSION_TABLE WHERE RELEASE = :rel AND ARCH = :arc"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rel = release, arc = arch)
	self.conn.commit()
	print 'RELEASE ENTRY DELETED'
    def read( self ):
	curs = self.conn.cursor()
	sqlstr = "SELECT ID, RELEASE, ARCH, PATH FROM VERSION_TABLE ORDER BY ID, RELEASE, ARCH"
	curs.prepare(sqlstr)
	curs.execute(sqlstr)
        print 'ID     RELEASE     ARCH     PATH'
	for row in curs:
		print row
    def addRelease( self, release, arch, path):
	curs = self.conn.cursor()
	relID = ExtractID(release)
	print "relID "+str(relID)
	sqlstr = "INSERT INTO VERSION_TABLE(ID, RELEASE, ARCH, PATH) VALUES(:rid, :rel, :arc, :pat)"
	curs.execute(sqlstr, rid = relID, rel = release, arc = arch, pat = path)
	self.conn.commit()
	print 'RELEASE ENTRY ADDED.'	

    def listReleases(self, relID):
    	curs = self.conn.cursor()
	sqlstr = "SELECT RELEASE, ARCH, PATH FROM VERSION_TABLE WHERE ID < :rid"
	curs.prepare(sqlstr)
	curs.execute(sqlstr, rid = relID)
        l = []
	for row in curs:
            r = ( row[0], row[1], row[2] )
            l.append( r )
	return l
