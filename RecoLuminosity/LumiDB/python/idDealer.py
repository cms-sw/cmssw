import coral
import nameDealer
class idDealer(object):
    """Manages the autoincremental ID values.\n
    Input: coral.schema object
    """
    def __init__( self , schema  ):
        self.__schema = schema
        self.__idTableColumnName = 'NEXTID'
        self.__idTableColumnType = 'unsigned long long'
        
    def getIDforTable( self, tableName ):
        """
        get the new id value to use for the given table
        """
        try:
            idtableName = nameDealer.idTableName(tableName)
            query = self.__schema.tableHandle(idtableName).newQuery()
            query.addToOutputList(self.__idTableColumnName)
            query.setForUpdate() #lock it
            cursor = query.execute()
            result = 0
            while ( cursor.next() ):
                result = cursor.currentRow()[self.__idTableColumnName].data()
            del query
            return result
        except Exception, e:
            raise Exception, str(e)

    def generateNextID( self, tableName ):
        """
        Set the nextID in the IDTableName to current id value + 1 .\n
        Input: ID table name.
        """
        try:
            idtableName = nameDealer.idTableName(tableName)
            tableHandle = self.__schema.tableHandle(idtableName)
            query = tableHandle.newQuery()
            query.addToOutputList(self.__idTableColumnName)
            query.setForUpdate() #lock it
            cursor = query.execute()
            result = 0
            while ( cursor.next() ):
                result = cursor.currentRow()[0].data()
            del query
            dataEditor = tableHandle.dataEditor()
            inputData = coral.AttributeList()
            inputData.extend( 'newid', self.__idTableColumnType )
            inputData['newid'].setData(result+1)
            dataEditor.updateRows('NEXTID = :newid','',inputData)
        except Exception, e:
            raise Exception, str(e)

    pass
