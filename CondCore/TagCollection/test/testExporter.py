#Coral package:CoralTools 
import os
import coral
from CondCore.TagCollection.exporter import exporter

#os.environ["CORAL_AUTH_PATH"]="/afs/cern.ch/sw/lcg/app/pool/db/python"
#os.environ["CORAL_DBLOOKUP_PATH"]="/afs/cern.ch/sw/lcg/app/pool/db/python"

try: 
  #Initialize Connection Service
  svc = coral.ConnectionService()

  #open session proxy for source schema providing logical service name & access mode
  sourceSession = svc.connect( 'sqlite_file:source.db', accessMode = coral.access_Update )

  #open session proxy for destination schema providing logical service name & access mode
  destSession = svc.connect( 'sqlite_file:dest.db', accessMode = coral.access_Update )
  
  rowCachesize=1000
  exp=exporter( sourceSession,destSession,rowCachesize )

  try: 
    print "copyschema() : Copies the schema objects from source to destination, without copying data."
    exp.copyschema() 
    print "Tables created" 
  except Exception, e:
    print "Test Failed" 
    print str(e)

  try: 
    print "copydata(rowCount) : Copies the schema objects from source to destination, including data copy."
    exp.copydata(rowCount=100) 
    print "Data copied"
  except Exception, e:
    print "Test Failed" 
    print str(e)

  try: 
    print "copytableschema(tablename) : Copies the specified table schema without copying data."
    tablename = "T1"
    exp.copytableschema(tablename) 
    print "Table created" 
  except Exception, e:
    print "Test Failed" 
    print str(e)

  try: 
    print "copytabledata(tablename,selectionclause,selectionparameters,rowCount) : Copies the specified table schema, including data copy."
    tablename = "T1"
    selectionclause= "id >= :idmin and id < :idmax "
    selectionparameters = coral.AttributeList()
    selectionparameters.extend( "idmin","int")
    selectionparameters.extend( "idmax","int")
    selectionparameters["idmin"].setData(0)
    selectionparameters["idmax"].setData(2)
    exp.copytabledata(tablename,selectionclause,selectionparameters,rowCount=100) 
    print "Data copied"
  except Exception, e:
    print "Test Failed" 
    print str(e)

  try: 
    print "copytablelistschema(tablelist) : Copies the specified list of tables ordered by hierarchy without copying data."
    tableset = ['T3','T1','T2']
    exp.copytablelistschema(tableset) 
    print "Tables created" 
  except Exception, e:
    print "Test Failed" 
    print str(e)

  try: 
    print "copytablelistdata(tablelist,rowCount) : Copies the specified list of tables ordered by hierarchy, including data copy."
    table1 = "T3"
    selectionclause1= "id >= 4 and id < 8 "

    table2 = "T2"
    selectionclause2= "id >= :idmin and id < :idmax "
    selectionparameters2 = coral.AttributeList()
    selectionparameters2.extend( "idmin","int")
    selectionparameters2.extend( "idmax","int")
    selectionparameters2["idmin"].setData(0)
    selectionparameters2["idmax"].setData(34)

    table3 = "T1"

    tablelist1 = [table1,selectionclause1]
    tablelist2 = [table2,selectionclause2,selectionparameters2]
    tablelist3 = [table3]

    tablelist = [tablelist1,tablelist2,tablelist3]
    exp.copytablelistdata(tablelist,rowCount=100) 
    print "Data copied " 
  except Exception, e:
    print "Test Failed" 
    print str(e)

  del sourceSession
  del destSession 
except Exception, e:
 print "Test FAILED"
 print str(e)


