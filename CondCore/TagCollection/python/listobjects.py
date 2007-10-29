import os
import coral
from multivaluedict import mseqdict

'''
dumpobjectlist(schema) 
Dumps the list of tables and views grouped and ordered by hierarchy, specifying the existing constraints and indexes.
Input parameter schema : a schema object, obtained by the sessionproxy object
Output paramter : none
'''
def dumpobjectlist( schema ):
 try:

  dTableInfo=listobjects( schema )

  print "--------------------------------------"
  print "Listing Table Description "
  print "--------------------------------------"
  for key,value in dTableInfo.items():
    tableName= key
    _printTableInfo(schema,tableName)

  print "Listing View Information"
  print "--------------------------------------"
  for viewName in schema.listViews():
    _printViewInfo(schema,viewName)

 except Exception, e:
  raise Exception ("Error in dumpobjectlist method: " + str(e))
  return False

#Returns the list of tables ordered by hierarchy 
def listobjects( schema ):
 try:

  listOfTableNames = schema.listTables()

  #Dictionaries are created for resolving table dependencies 
  dTable=mseqdict( [], {}) 
  dRefTable=mseqdict( [], {}) 
  dCopyTable=mseqdict( [], {}) 

  for tableName in listOfTableNames:

    #Add tablename to dictionary
    dTable.append(tableName,'')
    description = coral.TableDescription()
    description.setName( tableName )
    table = schema.tableHandle(tableName )

    numberOfForeignKeys = table.description().numberOfForeignKeys()
    for i in range(0, numberOfForeignKeys):
     foreignKey = table.description().foreignKey( i )
     columnNames = foreignKey.columnNames()
     #Add referenced tablename to dictionary
     dRefTable.append (tableName, foreignKey.referencedTableName()) 
     columnNamesR = foreignKey.referencedColumnNames()

  #For retrieving the tables in order of dependency 
  r1=mseqdict( [], {})
  r2=mseqdict( [], {})

  for rTable, refTable in dRefTable.items():
      for table in refTable:
        r1.append(table,'')
      r1.append(rTable,'')

  for rTable, refTable in r1.items():
    test=rTable
    for rTable1, refTable1 in dRefTable.items():
      if rTable1==test:
        for table in refTable1:
          if rTable1!=table:
             r2.append(table,'')

  for key,value in r2.items():
      r1.remove(key,'')
      dTable.remove(key,'')

  for key,value in r1.items():
      dTable.remove(key,'')

  for key,value in dTable.items():
    dCopyTable.append(key,'')

  for key,value in r2.items():
    dCopyTable.append(key,'')
  
  for key,value in r1.items():
    dCopyTable.append(key,'')

  return dCopyTable

 except Exception, e:
  raise Exception (" " + str(e))
  return False

#For printing the Table Information
def _printTableInfo( schema,tableName ):
 try:

   description = coral.TableDescription()
   description.setName( tableName )
   table = schema.tableHandle(tableName )

   numberOfColumns = table.description().numberOfColumns()
   print "Table " , tableName
   print "Columns : " , numberOfColumns  
   for  i in range(0, numberOfColumns):
     column = table.description().columnDescription( i )
     print "" , column.name() , " (" , column.type() , ")"
     if ( column.isUnique() ): 
      print "      UNIQUE";
     if ( column.isNotNull() ):
      print "      NOT NULL"

   if ( table.description().hasPrimaryKey() ):
     columnNames = table.description().primaryKey().columnNames()
     print ""
     print "Primary key defined for column :"
     for iColumn in columnNames:
      print "      ",iColumn , " "

   numberOfUniqueConstraints = table.description().numberOfUniqueConstraints()
   print ""
   print "Unique Constraints : " , numberOfUniqueConstraints
   for i in range( 0, numberOfUniqueConstraints ):
     uniqueConstraint = table.description().uniqueConstraint( i )
     print "" , uniqueConstraint.name() , " defined for column"
     columnNames = uniqueConstraint.columnNames()
     for iColumn in columnNames:
       print "      ",iColumn

   numberOfIndices = table.description().numberOfIndices()
   print ""
   print "Index :  " , numberOfIndices
   for i in range(0, numberOfIndices ):
     index = table.description().index( i )
     print "" , index.name()
     if ( index.isUnique() ):
      print " (UNIQUE)"
     print " defined for column"
     columnNames = index.columnNames()
     for iColumn in columnNames:
       print "      ",iColumn

   numberOfForeignKeys = table.description().numberOfForeignKeys()
   print "" 
   print "Foreign Keys : " , numberOfForeignKeys
   for i in range(0, numberOfForeignKeys):
     foreignKey = table.description().foreignKey( i )
     print "" , foreignKey.name() , " defined for column"
     columnNames = foreignKey.columnNames()
     for iColumn in columnNames:
       print "      ",iColumn
     print " references -> " , foreignKey.referencedTableName() , "on Column "; 
     columnNamesR = foreignKey.referencedColumnNames()
     for iColumn in columnNamesR:
       print "      ",iColumn

   print "--------------------------------------"

 except Exception, e:
  raise Exception (" " + str(e))
  return False

#For printing the View Information
def _printViewInfo( schema,viewName ):
 try:

   view = schema.viewHandle(viewName )
   numberOfColumns = view.numberOfColumns()
   print "View " , view.name()
   print "has", " ", numberOfColumns , " columns :"
   for i in range( 0,numberOfColumns ):
    column = view.column( i )
    print "" , column.name(), " (", column.type() , ")"
    if ( column.isUnique() ):
     print "      UNIQUE"
    if ( column.isNotNull() ):
     print "      NOT NULL"

   print " definition string : " , view.definition()

   print "--------------------------------------"

 except Exception, e:
  raise Exception (" " + str(e))
  return False

#Returns the list of tables ordered by hierarchy and checks for circular dependency between tables in source schema 
def listschema( schema ):
 try:
  listOfTableNames = schema.listTables()

  #Dictionaries are created for resolving table dependencies 
  dTable=mseqdict( [], {}) 
  dRefTable=mseqdict( [], {}) 
  dCopyTable=mseqdict( [], {}) 
  dCircTable=mseqdict( [], {}) 

  for tableName in listOfTableNames:

    #Add tablename to dictionary
    dTable.append(tableName,'')
    description = coral.TableDescription()
    description.setName( tableName )
    table = schema.tableHandle(tableName )

    numberOfForeignKeys = table.description().numberOfForeignKeys()
    for i in range(0, numberOfForeignKeys):
     foreignKey = table.description().foreignKey( i )
     columnNames = foreignKey.columnNames()
     #Add referenced tablename to dictionary
     dRefTable.append (tableName, foreignKey.referencedTableName()) 
     dCircTable.append (tableName, foreignKey.referencedTableName()) 
     columnNamesR = foreignKey.referencedColumnNames()

  #For checking circular dependency between the tables 
  d1=mseqdict( [], {})
  d2=mseqdict( [], {})

  for rTable, refTable in dCircTable.items():
    for table in refTable:
           d1.append(rTable,table)

  dCircTable.swap()
  for rTable, refTable in dCircTable.items():
    for table in refTable:
           d2.append(rTable,table)

  for key,value in d1.items():
     firsttable=key
     secondtable=value
     for key,value in d2.items():
        if key==firsttable and value==secondtable:
           raise Exception ("Circular Dependency exists between tables : "+firsttable,secondtable)

  #For retrieving the tables in order of dependency 
  r1=mseqdict( [], {})
  r2=mseqdict( [], {})

  for rTable, refTable in dRefTable.items():
      for table in refTable:
        r1.append(table,'')
      r1.append(rTable,'')

  for rTable, refTable in r1.items():
    test=rTable
    for rTable1, refTable1 in dRefTable.items():
      if rTable1==test:
        for table in refTable1:
          if rTable1!=table:
             r2.append(table,'')

  for key,value in r2.items():
      r1.remove(key,'')
      dTable.remove(key,'')

  for key,value in r1.items():
      dTable.remove(key,'')

  for key,value in dTable.items():
    dCopyTable.append(key,'')

  for key,value in r2.items():
    dCopyTable.append(key,'')
  
  for key,value in r1.items():
    dCopyTable.append(key,'')

  return dCopyTable

 except Exception, e:
  raise Exception (" " + str(e))
  return False

#Returns the tablename  for the specified table schema 
def listtables( schema,tablename ):
 try:
  listOfTableNames = schema.listTables()

  #Dictionaries are created for resolving table dependencies 
  dTable=mseqdict( [], {}) 
  dCopyTable=mseqdict( [], {}) 

  for tableName in listOfTableNames:
    if tablename==tableName: 
     #Add tablename to dictionary
     dTable.append(tableName,'')
     description = coral.TableDescription()
     description.setName( tableName )
     table = schema.tableHandle(tableName )

     numberOfForeignKeys = table.description().numberOfForeignKeys()
     for i in range(0, numberOfForeignKeys):
      foreignKey = table.description().foreignKey( i )
      columnNames = foreignKey.columnNames()
      columnNamesR = foreignKey.referencedColumnNames()

  for key,value in dTable.items():
       dCopyTable.append(key,'')

  return dCopyTable

 except Exception, e:
  raise Exception (" " + str(e))
  return False

#Returns the list of tables ordered by hierarchy for the specified list of tables and also checks for circular dependency between the tables
def listtableset( schema,tableset ):
 try:
  listOfTableNames = schema.listTables()

  #Dictionaries are created for resolving table dependencies 
  dTable=mseqdict( [], {}) 
  dCircTable=mseqdict( [], {}) 
  dCopyTable=mseqdict( [], {}) 
  dTempTable=mseqdict( [], {}) 

  for table in listOfTableNames:
   for tableName in tableset:
    if tableName==table: 
     #Add tablename to dictionary
     dTable.append(tableName,'')
     description = coral.TableDescription()
     description.setName( tableName )
     table = schema.tableHandle(tableName )

     numberOfForeignKeys = table.description().numberOfForeignKeys()
     for i in range(0, numberOfForeignKeys):
      foreignKey = table.description().foreignKey( i )
      columnNames = foreignKey.columnNames()
      #Add referenced tablename to dictionary
      dTable.append (tableName, foreignKey.referencedTableName()) 
      dCircTable.append (tableName, foreignKey.referencedTableName()) 
      columnNamesR = foreignKey.referencedColumnNames()

  #For checking  circular dependency between the tables 
  d1=mseqdict( [], {})
  d2=mseqdict( [], {})

  for rTable, refTable in dCircTable.items():
    for table in refTable:
           d1.append(rTable,table)

  dCircTable.swap()
  for rTable, refTable in dCircTable.items():
    for table in refTable:
           d2.append(rTable,table)

  for key,value in d1.items():
     firsttable=key
     secondtable=value
     for key,value in d2.items():
        if key==firsttable and value==secondtable:
           raise Exception ("Circular Dependency exists between tables : "+firsttable,secondtable)

  #For retrieving the tables in order of dependency 
  r1=mseqdict( [], {})
  r2=mseqdict( [], {})

  for rTable, refTable in dTable.items():
      for table in refTable:
        r1.append(table,'')
      r1.append(rTable,'')

  for rTable, refTable in r1.items():
    test=rTable
    for rTable1, refTable1 in dTable.items():
      if rTable1==test:
        for table in refTable1:
          if rTable1!=table:
             r2.append(table,'')

  for key,value in r2.items():
      r1.remove(key,'')

  for key,value in r2.items():
    dTempTable.append(key,'')
  
  for key,value in r1.items():
    dTempTable.append(key,'')

  for key,value in dTempTable.items():
       iTable= key
       for table in tableset:
           if table==iTable:
              dCopyTable.append(key,'')

  return dCopyTable

 except Exception, e:
  raise Exception (" " + str(e))
  return False

