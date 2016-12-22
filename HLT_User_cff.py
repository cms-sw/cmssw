hltGetConfiguration: error while retriving the list of paths from the HLT menu

dbURl  = jdbc:oracle:thin:@//cmsr1-s.cern.ch:10121/cms_cond.cern.ch
dbUser = cms_hltdev_reader
dbPwrd = convertme!
=====  Database info =====
DatabaseProductName: Oracle
DatabaseProductVersion: Oracle Database 11g Enterprise Edition Release 11.2.0.4.0 - 64bit Production
With the Partitioning, Real Application Clusters, OLAP, Data Mining
and Real Application Testing options
DatabaseMajorVersion: 11
DatabaseMinorVersion: 2
=====  Driver info =====
DriverName: Oracle JDBC driver
DriverVersion: 12.1.0.1.0
DriverMajorVersion: 12
DriverMinorVersion: 1
=====  JDBC/DB attributes =====
WARNING: Can't retrieve SQL Statement from SQL ResultSet: Closed Resultset: getStatement
ERROR: ConfDB::getConfigId(fullConfigName=/user/krajczar/mathom/CopyOf50nsHLTV67) failed (dirName=/user/krajczar/mathom, configName=CopyOf50nsHLTV67,version=14): Result set after last row
confdb.db.DatabaseException: ConfDB::getConfigId(fullConfigName=/user/krajczar/mathom/CopyOf50nsHLTV67) failed (dirName=/user/krajczar/mathom, configName=CopyOf50nsHLTV67,version=14): Result set after last row
	at confdb.db.ConfDB.getConfigId(ConfDB.java:3126)
	at confdb.converter.BrowserConverter.main(BrowserConverter.java:379)
Caused by: java.sql.SQLException: Result set after last row
	at oracle.jdbc.driver.GeneratedScrollableResultSet.getInt(GeneratedScrollableResultSet.java:519)
	at confdb.db.ConfDB.getConfigId(ConfDB.java:3119)
	... 1 more


