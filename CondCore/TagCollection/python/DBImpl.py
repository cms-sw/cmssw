import coral
import IdGenerator
        
class DBImpl(object):
    """Class wrap up all the database operations.\n
    """
    def __init__( self , schema):
        """Input: coral schema handle.
        """
        self.__schema = schema        
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
