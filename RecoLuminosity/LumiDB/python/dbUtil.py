import coral
from RecoLuminosity.LumiDB import nameDealer        
class dbUtil(object):
    """Class wrap up all the database operations.\n
    """
    def __init__( self , schema):
        """
        Input: coral schema handle.
        """
        self.__schema = schema
        
    def listIndex(self,tablename):
        mytable=self.__schema.tableHandle(tablename)
        print 'numberofindices ', mytable.description().numberOfIndices()
        for i in range(0,mytable.description().numberOfIndices()):
            index=mytable.description().index(i)
            print ' ', index.name(),' -> '
            for iColumn in index.columnNames():
                print iColumn
            print ' (tablespace : ',index.tableSpaceName(),')'
    def describeSchema(self):
        """
        Print out the overview of the schema
        """
        try:
            tablelist=self.__schema.listTables()
            for t in tablelist:
                table = self.__schema.tableHandle(t)
                print 'table ===',t,'==='
                n=table.description().numberOfColumns()
                for i in range(0,n):
                  columndesp=table.description().columnDescription(i)
                  print '\t',columndesp.name(),columndesp.type()
                if table.description().hasPrimaryKey():
                  print 'Primary Key : '
                  print '\t',table.description().primaryKey().columnNames()
                print 'Indices : '
                self.listIndex(t)
            viewlist=self.__schema.listViews()
            for v in viewlist:
                myview = self.__schema.viewHandle('V0')
                print 'definition : ',myview.definition()
                n=myview.numberOfColumns()
                for i in range(0,n):
                  columndesp=view.column(i)
                  print '\t',columndesp.name(),columndesp.type()
        except Exception, e:
            raise Exception, str(e)
    def existRow( self, tableName, condition, conditionDefDict,conditionDict):
        """
        Return true if one row fulfills the selection criteria
        """
        try:
            tableHandle = self.__schema.tableHandle(tableName)
            query = tableHandle.newQuery()
            queryBind=coral.AttributeList()
            for colname,coltype in conditionDefDict.items():
                queryBind.extend(colname,coltype)
                queryBind[colname].setData(conditionDict[colname])
            query.setCondition(condition,queryBind)
            cursor = query.execute()
            result=False
            while ( cursor.next() ):
                result=True
                cursor.close()
            del query
            return result
        except Exception, e:
            raise Exception, str(e)
    def insertOneRow( self, tableName, tabrowDefDict, tabrowValueDict ):
        """
        Insert row 
        """
        try:
            tableHandle = self.__schema.tableHandle(tableName)
            editor = tableHandle.dataEditor()
            inputData = coral.AttributeList()
            for name,type in tabrowDefDict.items():
               # print name, type
                inputData.extend( name, type )
                inputData[name].setData(tabrowValueDict[name])
            editor.insertRow( inputData )
        except Exception, e:
            raise Exception, 'dbUtil.insertOneRow:'+str(e)
    def updateRows( self,tableName,updateAction,updateCondition,bindvarDef,bulkinput):
        '''
        update rows, note update must be ordered
        input :
           tableName, string
           updateAction,string  e.g. flag=:newflag
           conditionstring, string ,e.g. runnum=:runnum and cmslsnum=:cmslsnum
           bindvarDef,[('newflag':'string'),('runnum','unsigned int'),('cmslsnum','unsigned int')]
           bulkinput,[[('newflag','GOOD'),('runnum',1234),('cmslsnum',1)],[]]
        '''
        try:
            dataEditor=self.__schema.tableHandle(tableName).dataEditor()
            updateData=coral.AttributeList()
            for (columnname,columntype) in bindvarDef:
                updateData.extend(columnname,columntype)
            bulkOperation=dataEditor.bulkUpdateRows(updateAction,updateCondition,updateData,len(bulkinput))
            for valuelist in bulkinput:
                for (columnname,columnvalue) in valuelist:
                    updateData[columnname].setData(columnvalue)
                bulkOperation.processNextIteration()
            bulkOperation.flush()
            del bulkOperation
        except Exception, e:
            raise Exception, 'dbUtil.updateRows:'+str(e)
    def bulkInsert( self, tableName, tabrowDef, bulkinput):
        """
        input:
           tableName, string
           tabrowDef,[('RUNNUM':'unsigned int'),('CMSLSNUM','unsigned int'),('FLAG','string'),('COMMENT','string')]
           bulkinput,[[('RUNNUM',1234),('CMSLSNUM',1234),('FLAG','GOOD'),('COMMENT','coment')],[]]
        """
        try:
            dataEditor=self.__schema.tableHandle(tableName).dataEditor()
            insertdata=coral.AttributeList()
            for (columnname,columntype) in tabrowDef:
                insertdata.extend(columnname,columntype)                
            bulkOperation=dataEditor.bulkInsert(insertdata,len(bulkinput))
            for valuelist in bulkinput:
                for (columnname,columnvalue) in valuelist:
                    insertdata[columnname].setData(columnvalue)
                bulkOperation.processNextIteration()
            bulkOperation.flush()
            del bulkOperation
        except Exception, e:
            raise Exception, 'dbUtil.bulkInsert:'+str(e)
        
    def deleteRows( self, tableName, condition, conditionbindDict ):
        """
        Delete row(s)
        """
        try:
            tableHandle = self.__schema.tableHandle(tableName)
            editor = tableHandle.dataEditor()
            editor.deleteRows( condition, conditionbindDict )
        except Exception, e:
            raise Exception, str(e)
        
    def dropTable( self, tableName ):
        """
        Drop specified table.If associated Id table exists, drop also Id table
        """
        try:
            self.__schema.dropIfExistsTable( tableName )
            self.__schema.dropIfExistsTable( nameDealer.idTableName(tableName) )
        except Exception, e:
            raise Exception, str(e)

    def dropAllTables( self ):
        """
        Drop all tables can be listed by schema.listTables
        """
        try:
            for t in self.__schema.listTables():
                self.__schema.dropTable(t)
        except Exception, e:
            raise Exception, str(e)

    def createTable( self,description,withIdTable=False):
        """
        Create table if non-existing, create Id table if required
        """
        try:
          tableHandle=self.__schema.createTable(description)
          tableHandle.privilegeManager().grantToPublic(coral.privilege_Select)
          if withIdTable is True:
            tableName=tableHandle.description().name()
            self.createIDTable(tableName,True)
        except Exception, e:
          raise Exception, str(e)
        
    def tableExists( self,tableName ):
        """
        Tell whether table exists
        """
        try:
          self.__schema.tableHandle(tableName)
          return True
        except coral.Exception, e:
          return False

    def createIDTable( self, tableName, deleteOld=True ):
        """
        Create ID table  for the given table.\n
        Input: name of the table which needs new associated id table
        Output: name of the id table created
        """
        try:
          idtableName=nameDealer.idTableName(tableName)
          if deleteOld is True:
            self.__schema.dropIfExistsTable(idtableName)
          else:
            if self.__schema.existsTable(idtableName):
               print 'table '+idtableName+' exists, do nothing'
               return
          description = coral.TableDescription()
          description.setName( idtableName )
          description.setPrimaryKey( nameDealer.idTableColumnDefinition()[0] )
          description.insertColumn( nameDealer.idTableColumnDefinition()[0], nameDealer.idTableColumnDefinition()[1])
          idtableHandle=self.__schema.createTable( description )
          idtableHandle.privilegeManager().grantToPublic( coral.privilege_Select )
          inputData = coral.AttributeList()
          editor = idtableHandle.dataEditor()
          editor.rowBuffer( inputData )
          inputData[ nameDealer.idTableColumnDefinition()[0] ].setData(0)
          editor.insertRow( inputData )
        except Exception, e:
          raise Exception, str(e)

if __name__ == "__main__":
    pass
