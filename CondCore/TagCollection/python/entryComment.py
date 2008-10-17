import coral
import DBImpl
import CommonUtils

class entryComment(object):
    """Class add optional comment on given entry in a given table\n
    """
    def __init__( self, session ):
        """Input: coral schema handle.
        """
        self.__session = session
        self.__entryCommentTableColumns={'entryid':'unsigned long','tablename':'string','comment':'string'}
        self.__entryCommentTableNotNullColumns=['entryid','tablename']
        self.__entryCommentTablePK=('entryid','tablename')
    def createEntryCommentTable(self):
        """Create entry comment able.Existing table will be deleted.
        """
        try:
           transaction=self.__session.transaction()
           transaction.start()
           schema = self.__session.nominalSchema()
           schema.dropIfExistsTable(CommonUtils.commentTableName())
           description = coral.TableDescription()
           description.setName(CommonUtils.commentTableName())
           for columnName, columnType in self.__entryCommentTableColumns.items():
               description.insertColumn(columnName,columnType)
           for columnName in self.__entryCommentTableNotNullColumns:
               description.setNotNullConstraint(columnName,True)
           description.setPrimaryKey(self.__entryCommentTablePK)
           tablehandle=schema.createTable(description)
           tablehandle.privilegeManager().grantToPublic(coral.privilege_Select)
           transaction.commit()
        except Exception, e:
           transaction.rollback() 
           raise Exception, str(e)
    def insertComment( self, tablename, entryid,comment ):
        """insert comment on the given entry of given table
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(False)
            tabrowValueDict={'entryid':entryid,'tablename':tablename,'comment':comment}
            schema = self.__session.nominalSchema()
            dbop=DBImpl.DBImpl(schema)
            dbop.insertOneRow(CommonUtils.commentTableName(),self.__entryCommentTableColumns,tabrowValueDict)
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
    def bulkinsertComments( self, tableName,bulkinput):
        """bulk insert comments for a given table
        bulkinput [{'entryid':unsigned long, 'tablename':string,'comment':string}]
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(False)
            schema = self.__session.nominalSchema()
            dbop=DBImpl.DBImpl(schema)
            dbop.bulkInsert(CommonUtils.commentTableName(),self.__entryCommentTableColumns,bulkinput)
            transaction.commit()  
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)    
        
    def getCommentForId( self, tableName, entryid ):
        """get comment for given id in given table
        """
        transaction=self.__session.transaction()
        comment=''
        try:
            transaction.start(True)
            schema = self.__session.nominalSchema()
            query = schema.tableHandle(CommonUtils.commentTableName()).newQuery()
            condition='entryid = :entryid AND tablename = :tablename'
            conditionbindDict=coral.AttributeList()
            conditionbindDict.extend('entryid','unsigned long')
            conditionbindDict.extend('tablename','string')
            conditionbindDict['entryid'].setData(entryid)
            conditionbindDict['tablename'].setData(tableName)
            query.addToOutputList('comment')
            query.setCondition(condition,conditionbindDict)
            cursor=query.execute()
            if cursor.next():
                comment=cursor.currentRow()['comment'].data()
                cursor.close()
            transaction.commit()
            del query
            return comment
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
    def getCommentsForTable( self, tableName ):
        """get all comments for given table
        result=[(entryid,comment)]
        """
        transaction=self.__session.transaction()
        result=[]
        
        try:
            transaction.start(True)
            schema = self.__session.nominalSchema()
            query = schema.tableHandle(CommonUtils.commentTableName()).newQuery()
            condition='tablename = :tablename'
            conditionbindDict=coral.AttributeList()
            conditionbindDict.extend('tablename','string')
            conditionbindDict['tablename'].setData(tableName)
            query.addToOutputList('entryid')
            query.addToOutputList('comment')
            query.setCondition(condition,conditionbindDict)
            cursor=query.execute()
            while cursor.next():
                comment=cursor.currentRow()['comment'].data()
                entryid=cursor.currentRow()['entryid'].data()
                result.append((entryid,comment))  
            cursor.close()
            transaction.commit()
            del query
            return result
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
    def modifyCommentForId( self, tableName, entryid, newcomment ):
        """replace comment for given entry for given table
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(False)
            editor = self.__session.nominalSchema().tableHandle(CommonUtils.commentTableName()).dataEditor()
            inputData = coral.AttributeList()
            inputData.extend('newcomment','string')
            inputData.extend('entryid','unsigned long')
            inputData.extend('tablename','string')
            inputData['newcomment'].setData(newcomment)
            inputData['entryid'].setData(entryid)
            inputData['tablename'].setData(tableName)
            editor.updateRows( "comment = :newcomment", "entryid = :entryid AND tablename = :tablename", inputData )
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)

    def replaceId( self, tableName, oldentryid, newentryid ):
        """replace entryid in given table
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(False)
            editor = self.__session.nominalSchema().tableHandle(CommonUtils.commentTableName()).dataEditor()
            inputData = coral.AttributeList()
            inputData.extend('newentryid','unsigned long')
            inputData.extend('oldentryid','unsigned long')
            inputData.extend('tablename','string')
            inputData['newentryid'].setData(newentryid)
            inputData['oldentryid'].setData(oldentryid)
            inputData['tablename'].setData(tableName)
            editor.updateRows( "entryid = :newentryid", "entryid = :oldentryid AND tablename = :tablename", inputData )
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
    def deleteCommentForId( self, tablename, entryid):
        """delete selected comment entry 
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(False)
            dbop=DBImpl.DBImpl(self.__session.nominalSchema())
            condition='tablename = :tablename AND entryid = :entryid'
            conditionbindDict=coral.AttributeList()
            conditionbindDict.extend('tablename','string')
            conditionbindDict.extend('entryid','unsigned long')
            conditionbindDict['tablename'].setData(tablename)
            conditionbindDict['entryid'].setData(entryid)
            dbop.deleteRows(CommonUtils.commentTableName(),condition,conditionbindDict)
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
    def clearAllEntriesForTable( self, tablename ):
        """delete all entries related with given table
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(False)
            dbop=DBImpl.DBImpl(self.__session.nominalSchema())
            condition='tablename = :tablename'
            conditionbindDict=coral.AttributeList()
            conditionbindDict.extend('tablename','string')
            conditionbindDict['tablename'].setData(tablename)
            dbop.deleteRows(CommonUtils.commentTableName(),condition,conditionbindDict)
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
    
if __name__ == "__main__":
    context = coral.Context()
    context.setVerbosityLevel( 'ERROR' )
    svc = coral.ConnectionService( context )
    session = svc.connect( 'sqlite_file:testentryComment.db',
                           accessMode = coral.access_Update )
    try:
        entrycomment=entryComment(session)
        print "test create entrycomment table"
        entrycomment.createEntryCommentTable()
        print "test insert one comment"
        entrycomment.insertComment(CommonUtils.inventoryTableName(),12,'comment1')
        entrycomment.insertComment(CommonUtils.treeTableName('ABCTREE'),12,'comment1')
        print "test bulk insert"
        bulkinput=[]
        bulkinput.append({'entryid':21,'tablename':CommonUtils.inventoryTableName(),'comment':'mycomment'})
        bulkinput.append({'entryid':22,'tablename':CommonUtils.inventoryTableName(),'comment':'mycomment2'})
        bulkinput.append({'entryid':23,'tablename':CommonUtils.inventoryTableName(),'comment':'mycomment3'})
        bulkinput.append({'entryid':24,'tablename':CommonUtils.inventoryTableName(),'comment':'mycomment4'})
        entrycomment.bulkinsertComments(CommonUtils.inventoryTableName(),bulkinput)
        print "test getCommentsForTable ",CommonUtils.inventoryTableName()
        results=entrycomment.getCommentsForTable(CommonUtils.inventoryTableName())
        print results
        result=entrycomment.getCommentForId(CommonUtils.inventoryTableName(),23)
        print result
        entrycomment.modifyCommentForId(CommonUtils.inventoryTableName(),23, 'mynewcomment' )
        print entrycomment.getCommentForId(CommonUtils.inventoryTableName(),23)
        print 'test replaceid'
        entrycomment.replaceId(CommonUtils.inventoryTableName(),23,33 )
        print entrycomment.getCommentForId(CommonUtils.inventoryTableName(),33)
        print 'test deletecomment for id'
        entrycomment.deleteCommentForId(CommonUtils.inventoryTableName(), 24)
        print entrycomment.getCommentsForTable(CommonUtils.inventoryTableName())
        print 'clearAllEntriesForTable'
        entrycomment.clearAllEntriesForTable(CommonUtils.inventoryTableName())
        print entrycomment.getCommentsForTable(CommonUtils.inventoryTableName())
        del session
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del session
        
