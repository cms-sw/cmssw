import os
import coral
from CondCore.TagCollection.listobjects import dumpobjectlist

#os.environ["CORAL_AUTH_PATH"]="/afs/cern.ch/sw/lcg/app/pool/db/python"
#os.environ["CORAL_DBLOOKUP_PATH"]="/afs/cern.ch/sw/lcg/app/pool/db/python"

try: 
  #Initialize Connection Service
  svc = coral.ConnectionService()

  #open session proxy using MySQL technology providing logical service name & access mode
  session = svc.connect( 'sqlite_file:my2test.db', accessMode = coral.access_Update )
  transaction = session.transaction()
  transaction.start(True)

  schema = session.nominalSchema()
  result = dumpobjectlist( schema )
  transaction.commit()
  del session

  print "[SUCCESS] Test for dumpobjectlist passed."

except Exception, e:
 transaction.rollback()
 print "Test FAILED"
 print str(e)


