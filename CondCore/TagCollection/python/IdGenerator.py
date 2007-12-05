import coral
import DBImpl
class IdGenerator(object):
    """Manages the autoincremental ID values.\n
    Input: coral.schema object
    """
    def __init__( self , schema ):
        self.__schema = schema
        self.__idTableColumnName = 'nextID'
        self.__idTableColumnType = 'unsigned long'
    def getNewID( self, IDtableName):
        """Return the ID value in the specified ID table.\n
        Input: ID table name
        """
        try:
            query = self.__schema.tableHandle(IDtableName).newQuery()
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
    def incrementNextID( self, IDtableName ):
        """Set the nextID in the IDTableName to current id value + 1 .\n
        Input: ID table name.
        """
        try:
            tableHandle = self.__schema.tableHandle(IDtableName)
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
            dataEditor.updateRows('nextID = :newid','',inputData)
        except Exception, e:
            raise Exception, str(e)
    #def getIDTableName( self, tableName ):
    #    """Returns the ID table name associated with given table.\n
    #    No check on the existence of the table.\n
    #    Input: data table name
    #    Output: ID table name
    #    """
    #    return tableName+'_IDs'
    def createIDTable( self, idtableName, deleteOld=True ):
        """Create ID table 'tableName_ID' for the given table.\n
        Input: name of the table which needs new associated id table
        Output: name of the id table created
        """
        dbop=DBImpl.DBImpl(self.__schema)
        try:
            if dbop.tableExists(idtableName) is True:
                if deleteOld is True:
                    dbop.dropTable(idtableName)
                else:
                    return
            description = coral.TableDescription();
            description.setName(idtableName)
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
    idtableName = 'TagTreeTable_IDS'
    context = coral.Context()
    context.setVerbosityLevel( 'ERROR' )
    svc = coral.ConnectionService( context )
    session = svc.connect( 'sqlite_file:data.db', accessMode = coral.access_Update )
    transaction = session.transaction()
    try:
        transaction.start()
        schema = session.nominalSchema()
        generator=IdGenerator(schema)
        generator.createIDTable( idtableName )
        transaction.commit()
        transaction.start(True)
        result=generator.getNewID(idtableName)
        print 'new id ',result
        transaction.commit()
        transaction.start(False)
        generator.incrementNextID(idtableName)
        print 'new id ',generator.getNewID(idtableName)
        transaction.commit()
        del session
    except coral.Exception, e:
        transaction.rollback()
        print str(e)
        del session
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del session
