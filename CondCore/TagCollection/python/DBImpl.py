import coral
import IdGenerator
        
class DBImpl(object):
    """Class wrap up all the database operations.\n
    """
    def __init__( self , schema):
        """Input: coral schema handle.
        """
        self.__schema = schema
    def existRow( self, tableName, condition, conditionbindDict):
        """Return true if one row fulfills the selection criteria
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
        """Insert row 
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
        """Bulk insert bulkinput=[{}]
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
        """Delete row(s)
        """
        try:
            tableHandle = self.__schema.tableHandle(tableName)
            editor = tableHandle.dataEditor()
            editor.deleteRows( condition, conditionbindDict )
        except Exception, e:
            raise Exception, str(e)
        
    def dropTable( self, tableName ):
        """Drop specified table.
        """
        self.__schema.dropIfExistsTable( tableName )
    def tableExists( self, tableName ):
        """Tell whether table exists
        """
        try:
            self.__schema.tableHandle(tableName)
            return True
        except coral.Exception, e:
            return False

if __name__ == "__main__":
    pass
