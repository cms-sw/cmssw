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
    def describeSchema(self):
        """
        Print out the overview of the schema
        """
        try:
            tablelist=self.__schema.listTables()
            for t in tablelist:
                table = self.__schema.tableHandle(t)
                print 'table : ',t
                n=table.description().numberOfColumns()
                for i in range(0,n):
                  columndesp=table.description().columnDescription(i)
                  print columndesp.name(),columndesp.type()
                if table.description().hasPrimaryKey():
                  print 'Primary Key : '
                  print '\t',table.description().primaryKey().columnNames()
                  
            viewlist=self.__schema.listViews()
            for v in viewlist:
                view = self.__schema.viewHandle(v)
                print 'view : ', v
                print 'definition : ',view.definition()
                n=view.numberOfColumns()
                for i in range(0,n):
                  columndesp=view.column(i)
                  print columndesp.name(),columndesp.type()                 
        except Exception, e:
            raise Exception, str(e)
    def existRow( self, tableName, condition, conditionbindDict):
        """
        Return true if one row fulfills the selection criteria
        """
        try:
            tableHandle = self.__schema.tableHandle(tableName)
            query = tableHandle.newQuery()
            query.setCondition(condition,conditionbindDict)
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
            raise Exception, str(e)
    def bulkInsert( self, tableName, tabrowDefDict, bulkinput):
        """
        Bulk insert bulkinput=[{}]
        """
        try:
            dataEditor=self.__schema.tableHandle(tableName).dataEditor()
            insertdata=coral.AttributeList()
            for (columnname,columntype) in tabrowDefDict.items():
                insertdata.extend(columnname,columntype)
                
            bulkOperation=dataEditor.bulkInsert(insertdata,len(bulkinput))
            for valuedict in bulkinput:
                for (columnname,columnvalue) in valuedict.items():
                    insertdata[columnname].setData(columnvalue)
                bulkOperation.processNextIteration()
            bulkOperation.flush()
            del bulkOperation
        except Exception, e:
            raise Exception, str(e)
        
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
          idtableHandle=self.__schema.tableHandle(idtableName)
          if idtableHandle is True:                
            if deleteOld is True:
              self.__schema.dropIfExistsTable(idtableName)
            else:
              return
          else:
            description = coral.TableDescription();
            description.setName( idtableName )
            description.setPrimaryKey( self.__idTableColumnName )
            description.insertColumn(self.__idTableColumnName, self.__idTableColumnType)
            idtableHandle=self.__schema.createTable( description )
            idtableHandle.privilegeManager().grantToPublic( coral.privilege_Select )
            inputData = coral.AttributeList()
            editor = idtableHandle.dataEditor()
            editor.rowBuffer( inputData )
            inputData[self.__idTableColumnName].setData(1)
            editor.insertRow( inputData )
        except Exception, e:
            raise Exception, str(e)

if __name__ == "__main__":
    pass
