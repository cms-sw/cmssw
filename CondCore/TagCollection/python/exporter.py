'''
exporter(sourceSession,destSession[,rowCachesize])
Input parameter sourceSession : session proxy for source schema providing logical service name & access mode
Input parameter destSession : session proxy for destination schema providing logical service name & access mode
Input parameter rowCachesize :  the number of rows to be cached at the client side, default value =100
Output paramter : the exporter object
'''

import os
import coral
import time
import math
from multivaluedict import mseqdict
from listobjects import listobjects,listschema,listtables,listtableset

class exporter:
 "exporter class for CoralTools"
 m_sourceSession = 0
 m_destSession = 0
 m_rowCachesize = 100

 def __init__( self,sourceSession,destSession,rowCachesize=100 ):
  try:

   self.m_sourceSession = sourceSession
   self.m_destSession = destSession
   self.m_rowCachesize = rowCachesize

   self.m_sourceSession.transaction().start()
   self.m_destSession.transaction().start()

  except Exception, e:
   raise Exception("Error in Initializer: " + str(e)) 

#Copies the schema objects from source to destination, without copying data. 
 def copyschema(self ):
  try:

   listsourceTable=listschema( self.m_sourceSession.nominalSchema() )
   listdestTable=listobjects( self.m_destSession.nominalSchema() )
   self._checktable(listsourceTable,listdestTable)

   for key,value in listsourceTable.items():
    iTable = key
    print iTable
    self._copytablelayout(iTable)

   self.m_destSession.transaction().commit()
   self.m_sourceSession.transaction().commit()
   print "copyschema SUCCESS"

   return True

  except Exception, e:
   self.m_destSession.transaction().rollback()
   self.m_sourceSession.transaction().commit()
   raise Exception ("Error in copyschema method: " + str(e))
   return False

#Copies the schema objects from source to destination, including data. 
 def copydata(self,rowCount=-1 ):
  try:

   self.m_sourceSession.transaction().start()
   self.m_destSession.transaction().start()

   listsourceTable=listschema( self.m_sourceSession.nominalSchema() )
   listdestTable=listobjects( self.m_destSession.nominalSchema() )
   self._checktable(listsourceTable,listdestTable)

   selectionclause=""
   selectionparameters=coral.AttributeList()
   for key,value in listsourceTable.items():
    iTable = key
    print iTable
    currentCount = 0
    self._copytablelayout(iTable)
    self._copydatalayout(iTable,selectionclause,selectionparameters,currentCount,rowCount)

   self.m_destSession.transaction().commit()
   self.m_sourceSession.transaction().commit()
   print "copydata SUCCESS"
   return True

  except Exception, e:
   self.m_destSession.transaction().rollback()
   self.m_sourceSession.transaction().commit()
   raise Exception ("Error in copydata method: " + str(e))
   return False

#Copies the specified table schema without copying data. 
 def copytableschema(self,tablename ):
  try:

   iTable=""
   listsourceTable=listtables( self.m_sourceSession.nominalSchema(),tablename )
   listdestTable=listtables(self.m_destSession.nominalSchema(),tablename )
   self._checktable(listsourceTable,listdestTable)

   for key,value in listsourceTable.items():
    iTable = key
    print iTable
    self._copytablelayout(iTable)

   self.m_destSession.transaction().commit()
   self.m_sourceSession.transaction().commit()
   print "copytableschema SUCCESS"
   return True

  except Exception, e:
   self.m_destSession.transaction().rollback()
   self.m_sourceSession.transaction().commit()
   raise Exception ("Error in copytableschema method: " + str(e)+" : "+iTable)
   return False

#Copies the specified table schema including data.  
 def copytabledata(self,tablename,selectionclause,selectionparameters,rowCount=-1 ):
  try:

   iTable=""
   listsourceTable=listtables( self.m_sourceSession.nominalSchema(),tablename )
   listdestTable=listtables( self.m_destSession.nominalSchema(),tablename )

   currentCount = 0
   for key,value in listsourceTable.items():
    iTable = key
    print iTable
    tableexists = self._checkdata(iTable,listdestTable)

    if not tableexists: 
       self._copytablelayout(iTable)

    self._copydatalayout(iTable,selectionclause,selectionparameters,currentCount,rowCount)

   self.m_destSession.transaction().commit()
   self.m_sourceSession.transaction().commit()
   print "copytabledata SUCCESS"
   return True

  except Exception, e:
   self.m_destSession.transaction().rollback()
   self.m_sourceSession.transaction().commit()
   raise Exception  ("Error in copytabledata method: " + str(e)+" : " + iTable)
   return False

#Copies the specified list of tables ordered by hierarchy without copying data.
 def copytablelistschema(self,tableset ):
  try:

   iTable=""
   listsourceTable=listtableset( self.m_sourceSession.nominalSchema(),tableset )
   listdestTable=listtableset( self.m_destSession.nominalSchema(),tableset )
   self._checktable(listsourceTable,listdestTable)

   for key,value in listsourceTable.items():
       iTable = key
       print iTable
       self._copytablelayout(iTable)

   self.m_destSession.transaction().commit()
   self.m_sourceSession.transaction().commit()
   print "copytablelistschema SUCCESS"
   return True

  except Exception, e:
   self.m_destSession.transaction().rollback()
   self.m_sourceSession.transaction().commit()
   raise Exception ("Error in copytablelistschema method: " + str(e)+" : "+iTable)
   return False

#Copies the specified list of tables ordered by hierarchy including data.
 def copytablelistdata(self,tablelist,rowCount=-1):
  try:

   iTable=""
   tableset=[]
   for table in tablelist:
       i=0
       for parameter in  table:
        if i==0:
           tableset.append(parameter)
        i=i+1

   listsourceTable=listtableset( self.m_sourceSession.nominalSchema(),tableset )
   listdestTable=listtableset( self.m_destSession.nominalSchema(),tableset )
   for key,value in listsourceTable.items():
       iTable = key
       print iTable
       currentCount = 0 
       selectionclause=""
       selectionparameters=coral.AttributeList()
       for table in tablelist:
           i=0
           for parameter in  table:
                 if i==0:
                    table=parameter
                 if table==iTable:
                    if i==1:
                       selectionclause = parameter
                    if i==2:
                       selectionparameters = parameter
                 i=i+1

       tableexists = self._checkdata(iTable,listdestTable)

       if not tableexists:
           self._copytablelayout(iTable)

       self._copydatalayout(iTable,selectionclause,selectionparameters,currentCount,rowCount)

   self.m_destSession.transaction().commit()
   self.m_sourceSession.transaction().commit()
   print "copytablelistdata SUCCESS"
   return True

  except Exception, e:
   self.m_destSession.transaction().rollback()
   self.m_sourceSession.transaction().commit()
   raise Exception ("Error in copytablelistdata method: " + str(e) + " : "+ iTable)
   return False

#Copies the schema objects from source to destination 
 def _copytablelayout(self,iTable ):
  try:

    description = self.m_sourceSession.nominalSchema().tableHandle( iTable ).description()
    table = self.m_destSession.nominalSchema().createTable( description )

    return True

  except Exception, e:
   raise Exception (" " + str(e))
   return False

#Copies the data from source to destination 
 def _copydatalayout(self,iTable,selectionclause,selectionparameters,currentCount,rowCount ):
  try:
    data=coral.AttributeList()
    sourceEditor = self.m_sourceSession.nominalSchema().tableHandle( iTable ).dataEditor()
    destEditor = self.m_destSession.nominalSchema().tableHandle( iTable ).dataEditor()

    sourceEditor.rowBuffer(data)
    sourcequery = self.m_sourceSession.nominalSchema().tableHandle( iTable ).newQuery()
    sourcequery.setCondition(selectionclause,selectionparameters)
    sourcequery.setRowCacheSize(self.m_rowCachesize)
    sourcequery.defineOutput(data)

    bulkOperation = destEditor.bulkInsert(data,self.m_rowCachesize )

    cursor=sourcequery.execute()

    for row in cursor:
        currentCount = currentCount+1
        bulkOperation.processNextIteration()
        if currentCount == rowCount:
           bulkOperation.flush()
           self.m_destSession.transaction().commit()
           self.m_destSession.transaction().start()
           currentCount = 0
    bulkOperation.flush()
    del bulkOperation
    del sourcequery

    return True

  except Exception, e:
   raise Exception (" " + str(e))
   return False

 #Checks if table exists in destination schema
 def _checktable(self,listsourceTable,listdestTable ):
  try:

   for key,value in listsourceTable.items():
     table=key
     for key,value in listdestTable.items():
        if key==table:
          raise Exception( "Table exists in Destination Schema : " )

   return True

  except Exception, e:
   raise Exception (" " + str(e) + table)
   return False

#Checks if data exists in the table in destination schema
 def _checkdata(self,iTable,listdestTable ):
  try:

   foundtable=False
   founddata=False
   for key,value in listdestTable.items():
      if key==iTable:
         counter=0
         # Run a query on the destination table
         query = self.m_destSession.nominalSchema().tableHandle( iTable ).newQuery()
         cursor = query.execute()
         foundtable=True
         for row in cursor:
            counter=counter+1
         del query
         if (counter > 0):
            founddata=True

   return foundtable

  except Exception, e:
   raise Exception (" " + str(e) + iTable)
   return False

 def __del__( self ):
     print ""
