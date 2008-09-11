import os
import coral
import IdGenerator, Node, DBImpl
class tagTree(object):
    """Class manages tag tree. Note: tree name is not case sensitive.
    Tree name is always converted to upper case
    """
    def __init__( self, session, treename ):
        self.__session = session
        self.__tagTreeTableName = 'TAGTREE_TABLE_'+str.upper(treename)
        self.__tagTreeIDs = 'TAGTREE_'+str.upper(treename)+'_IDS'
        self.__tagInventoryTableName = 'TAGINVENTORY_TABLE'
        self.__tagTreeTableColumns = {'nodeid':'unsigned long', 'nodelabel':'string', 'lft':'unsigned long', 'rgt':'unsigned long', 'parentid':'unsigned long', 'tagid':'unsigned long', 'globalsince':'unsigned long long', 'globaltill':'unsigned long long'}
        self.__tagTreeTableNotNullColumns = ['nodelabel','lft','rgt','parentid']
        self.__tagTreeTableUniqueColumns = ['nodelabel']
        self.__tagTreeTablePK = ('nodeid')
    def existTagTreeTable( self ):
        """Check if tree table exists
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(True)
            schema = self.__session.nominalSchema()
            result=schema.existsTable(self.__tagTreeTableName)
            transaction.commit()
            #print result
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        return result
    def createTagTreeTable( self ):
        """Create tag tree table. Existing table will be deleted. 
        """
        transaction=self.__session.transaction()
        try:
            transaction.start(False)
            schema = self.__session.nominalSchema()
            schema.dropIfExistsTable( self.__tagTreeTableName )
            description = coral.TableDescription();
            description.setName( self.__tagTreeTableName )
            for columnName, columnType in self.__tagTreeTableColumns.items():
                description.insertColumn(columnName, columnType)
            for columnName in self.__tagTreeTableNotNullColumns :
                description.setNotNullConstraint(columnName,True)
            for columnName in self.__tagTreeTableUniqueColumns :
                description.setUniqueConstraint(columnName)
            description.setPrimaryKey(  self.__tagTreeTablePK )
            #description.createForeignKey('tagid_FK','tagid',self.__tagInventoryTableName,'tagid')
            self.__tagTreeTableHandle = schema.createTable( description )
            self.__tagTreeTableHandle.privilegeManager().grantToPublic( coral.privilege_Select )
            #create also the associated id table
            generator=IdGenerator.IdGenerator(self.__session.nominalSchema())
            generator.createIDTable(self.__tagTreeIDs,True)
            transaction.commit()
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)

    def importFromTree( self, sourcetreename ):
        """fill up this tree by cloning from the given source tree 
        """
        sourcetagTreeTableName = 'TAGTREE_TABLE_'+str.upper(sourcetreename)
        sourcetagTreeIDs = 'TAGTREE_'+str.upper(sourcetreename)+'_IDS'
        
        transaction=self.__session.transaction()
        transaction.start(True)
        schema = self.__session.nominalSchema()
        r1=schema.existsTable(sourcetagTreeTableName)
        r2=schema.existsTable(sourcetagTreeIDs)
        r3=schema.existsTable(self.__tagTreeTableName)
        r4=schema.existsTable(self.__tagTreeIDs)
        transaction.commit()
        if r1 and r2 is False:
            raise "source tag tree doesn't exist "+str(sourcetreename) 
        if r3 and r4 is True:
            transaction.start(False)
            schema.truncateTable(self.__tagTreeTableName)
            schema.truncateTable(self.__tagTreeIDs)
            transaction.commit()
        else:
            self.createTagTreeTable()
            schema.truncateTable(self.__tagTreeIDs)
        nresult=0
        try:
            transaction.start(False)
            insertwtQuery=schema.tableHandle(self.__tagTreeTableName).dataEditor().insertWithQuery()
            insertwtQuery.query().addToTableList(sourcetagTreeTableName)
            nresult=insertwtQuery.execute()
            transaction.commit()
            del insertwtQuery
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        print nresult,' rows copied from ',sourcetagTreeTableName
        
        try:
            transaction.start(False)
            insertwtQuery=schema.tableHandle(self.__tagTreeIDs).dataEditor().insertWithQuery()
            insertwtQuery.query().addToTableList(sourcetagTreeIDs)
            nresult=insertwtQuery.execute()
            transaction.commit()
            del insertwtQuery
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        print nresult,' rows copied from ',sourcetagTreeIDs
        
    def replaceLeafLinks(self, leafnodelinks ):
        """modify the tagid link in leafnodes
        Input: [{leaflabel:newtagid}]
        """
        transaction=self.__session.transaction()
        try:
            updateAction="tagid = :newtagid"
            updateCondition="tagname = :tagname"
            updateData=coral.AttributeList()
            updateData.extend('newtagid','unsigned long')
            updateData.extend('tagname','string')
            transaction.start(False)
            mybulkOperation=schema.tableHandle(self.__tagTreeTableName).dataEditor().bulkUpdateRows("tagid=:newtagid","tagname=:tagname",updateData,1000)
            for leafnodelink in leafnodelinks:
                updateData['newtagid'].setData(leafnodelink[tagname])
                updateData['tagname'].setData(tagname)
                nresult=mybulkOperation.processNextIteration()
                if nresult != 1:
                    raise "updated number of row is not one"
            mybulkOperation.flush()
            transaction.commit()
            del mybulkOperation
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        
    def insertNode( self, node, parentLabel='ROOT' ):
        """Append a new node to specified parent. \n
        Silently ignore duplicate entry \n
        Input: new node. \n
        Input: name of the parent node. \n
        Output: result nodeid  
        """
        nodeid=node.nodeid
        nodelabel=node.nodelabel
        parentid=0
        tagid=node.tagid
        lft=1
        rgt=2
        globalsince=node.globalsince
        globaltill=node.globaltill
        duplicate=False
        transaction=self.__session.transaction()
        try:
            if parentLabel != 'ROOT':
                    parentNode=self.getNode(parentLabel)
                    if parentNode.empty():
                        raise ValueError,"non-existing parent node "+parentLabel
                    parentid=parentNode.nodeid
                    lft=parentNode.rgt
                    rgt=parentNode.rgt+1
            ##start readonly transaction
            transaction.start(False)
            condition='nodelabel=:nodelabel'
            conditionbindDict=coral.AttributeList()
            conditionbindDict.extend('nodelabel','string')
            conditionbindDict['nodelabel'].setData(nodelabel)
            dbop=DBImpl.DBImpl(self.__session.nominalSchema())
            duplicate=dbop.existRow(self.__tagTreeTableName,condition,conditionbindDict)
            if duplicate is False:                
                generator=IdGenerator.IdGenerator(self.__session.nominalSchema())
                nodeid=generator.getNewID(self.__tagTreeIDs)
            if duplicate is False:                
                tabrowValueDict={'nodeid':nodeid, 'nodelabel':nodelabel,
                                 'lft':lft, 'rgt':rgt, 'parentid':parentid,
                                 'tagid':tagid, 'globalsince':globalsince,
                                 'globaltill':globaltill
                                 }
                if parentLabel != 'ROOT':
                    self.__openGap(self.__session.nominalSchema().tableHandle(self.__tagTreeTableName),parentNode.rgt,1 )
                dbop.insertOneRow(self.__tagTreeTableName,
                                  self.__tagTreeTableColumns,
                                  tabrowValueDict)
                generator.incrementNextID(self.__tagTreeIDs)
            transaction.commit()
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        
    def getNodeById( self, nodeid ):
        """return result of query "select * from treetable where nodeid=:nodeid" in Node structure \n
        Input: id of the node to get.\n
        Output: selected node 
        """
        result=Node.Node()
        transaction=self.__session.transaction()
        try:
            transaction.start(True)
            schema = self.__session.nominalSchema()
            query = schema.tableHandle(self.__tagTreeTableName).newQuery()
            condition = 'nodeid =:nodeid'
            conditionData = coral.AttributeList()
            conditionData.extend( 'nodeid','unsigned int' )
            conditionData['nodeid'].setData(nodeid)
            query.setCondition( condition, conditionData)
            cursor = query.execute()
            while ( cursor.next() ):
                result.tagid=cursor.currentRow()['tagid'].data()
                result.nodeid=cursor.currentRow()['nodeid'].data()
                result.nodelabel=cursor.currentRow()['nodelabel'].data()
                result.lft=cursor.currentRow()['lft'].data()
                result.rgt=cursor.currentRow()['rgt'].data()
                result.parentid=cursor.currentRow()['parentid'].data()
                result.globalsince=cursor.currentRow()['globalsince'].data()
                result.globaltill=cursor.currentRow()['globaltill'].data()
            transaction.commit()
            del query
            return result
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        
    def getNode( self, label='ROOT' ):
        """return result of query "select * from treetable where nodelabel=label" in Node structure \n
        Input: name of the node to get. Default to 'ROOT' \n
        Output: selected node 
        """
        result=Node.Node()
        if label=='ROOT':
            return result
        transaction=self.__session.transaction()
        try:
            transaction.start(True)
            query=self.__session.nominalSchema().tableHandle(self.__tagTreeTableName).newQuery()
            condition = 'nodelabel =:nodelabel'
            conditionData = coral.AttributeList()
            conditionData.extend( 'nodelabel','string' )
            query.setCondition( condition, conditionData)
            conditionData['nodelabel'].setData(label)
            cursor = query.execute()
            while ( cursor.next() ):
                result.tagid=cursor.currentRow()['tagid'].data()
                result.nodeid=cursor.currentRow()['nodeid'].data()
                result.nodelabel=cursor.currentRow()['nodelabel'].data()
                result.lft=cursor.currentRow()['lft'].data()
                result.rgt=cursor.currentRow()['rgt'].data()
                result.parentid=cursor.currentRow()['parentid'].data()
                result.globalsince=cursor.currentRow()['globalsince'].data()
                result.globaltill=cursor.currentRow()['globaltill'].data()
            transaction.commit()
            del query
            return result
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
    def getPath( self, label ):
        """Get the path to arrive from ROOT to the given node.\n
        Input: labe of the node
        Output: list of the labels of the nodes in the path
        query "SELECT nodelabel FROM treetable WHERE lft<%s AND rgt>%s ORDER BY lft ASC"
        """
        result=[]
        lft=0
        rgt=0
        try:
            me=self.getNode(label)
            lft=me.lft
            rgt=me.rgt
            transaction=self.__session.transaction()
            transaction.start(True)
            schema = self.__session.nominalSchema()
            query = schema.tableHandle(self.__tagTreeTableName).newQuery()
            query.addToOutputList('nodelabel')
            condition = 'lft <:lft AND rgt>:rgt'
            conditionData = coral.AttributeList()
            conditionData.extend( 'lft','unsigned long' )
            conditionData.extend( 'rgt','unsigned long' )
            conditionData['lft'].setData(lft)
            conditionData['rgt'].setData(rgt)
            query.setCondition( condition, conditionData)
            query.addToOrderList( 'lft' )
            cursor = query.execute()
            while ( cursor.next() ):
                resultNodeLabel = cursor.currentRow()['nodelabel'].data()
                result.append( resultNodeLabel )
            transaction.commit()
            del query
            return result
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)   
    def getAllLeaves( self ):
        """Get all leaf nodes.\n
        Output: list of leaf nodes
        Query "SELECT * FROM treetable WHERE lft=rgt-1"
        """
        result=[]
        try:
            transaction=self.__session.transaction()
            transaction.start(True)
            schema = self.__session.nominalSchema()
            query = schema.tableHandle(self.__tagTreeTableName).newQuery()
            condition = 'lft=rgt-1'
            conditionData = coral.AttributeList()
            query.setCondition( condition, conditionData)
            cursor = query.execute()
            while ( cursor.next() ):
                resultNode=Node.Node()
                resultNode.tagid=cursor.currentRow()['tagid'].data()
                resultNode.nodeid=cursor.currentRow()['nodeid'].data()
                resultNode.nodelabel=cursor.currentRow()['nodelabel'].data()
                resultNode.lft=cursor.currentRow()['lft'].data()
                resultNode.rgt=cursor.currentRow()['rgt'].data()
                resultNode.parentid=cursor.currentRow()['parentid'].data()
                resultNode.globalsince=cursor.currentRow()['globalsince'].data()
                resultNode.globaltill=cursor.currentRow()['globaltill'].data()
                result.append( resultNode )
            transaction.commit()
            del query
            return result
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)   
    def getSubtree( self, label='ROOT' ):
        """Get the tree under node(included) with specified label.
        Input: node label
        Output: subtree in list of Node
        Query "SELECT p1.* FROM treetable AS p1, treetable AS p2 WHERE p1.lft BETWEEN p2.lft AND p2.rgt AND p2.nodelabel=%s ORDER BY p1.lft ASC"
        """
        result=[]
        try:
            if label=='ROOT' :
                transaction=self.__session.transaction()
                transaction.start(True)
                schema = self.__session.nominalSchema()
                query = schema.tableHandle(self.__tagTreeTableName).newQuery()
                for columnName in self.__tagTreeTableColumns:
                    query.addToOutputList(columnName)
                cursor = query.execute()
                while ( cursor.next() ):
                    resultNode=Node.Node()
                    resultNode.tagid=cursor.currentRow()['tagid'].data()
                    resultNode.nodeid=cursor.currentRow()['nodeid'].data()
                    resultNode.nodelabel=cursor.currentRow()['nodelabel'].data()
                    resultNode.lft=cursor.currentRow()['lft'].data()
                    resultNode.rgt=cursor.currentRow()['rgt'].data()
                    resultNode.parentid=cursor.currentRow()['parentid'].data()
                    resultNode.globalsince=cursor.currentRow()['globalsince'].data()
                    resultNode.globaltill=cursor.currentRow()['globaltill'].data()
                    result.append(resultNode)
                transaction.commit()
                del query
                return result
            else:
                me=self.getNode(label)
                parentlft=me.lft
                parentrgt=me.rgt
                transaction=self.__session.transaction()
                transaction.start(True)
                schema = self.__session.nominalSchema()
                query = schema.newQuery()
                query.addToTableList( self.__tagTreeTableName,'p1' )
                query.addToTableList( self.__tagTreeTableName,'p2' )
                for columnname in self.__tagTreeTableColumns.keys():
                    query.addToOutputList( 'p1.'+columnname )
                condition = 'p1.lft BETWEEN p2.lft AND p2.rgt AND p2.nodelabel = :nodelabel'
                query.addToOrderList( "p1.lft" );
                conditionData = coral.AttributeList()
                conditionData.extend( 'nodelabel','string' )
                conditionData['nodelabel'].setData(label)
                query.setCondition( condition, conditionData)
                cursor = query.execute()
                while ( cursor.next() ):
                    resultNode=Node.Node()
                    resultNode.tagid=cursor.currentRow()['p1.tagid'].data()
                    resultNode.nodeid=cursor.currentRow()['p1.nodeid'].data()
                    resultNode.nodelabel=cursor.currentRow()['p1.nodelabel'].data()
                    resultNode.lft=cursor.currentRow()['p1.lft'].data()
                    resultNode.rgt=cursor.currentRow()['p1.rgt'].data()
                    resultNode.parentid=cursor.currentRow()['p1.parentid'].data()
                    resultNode.globalsince=cursor.currentRow()['p1.globalsince'].data()
                    resultNode.globaltill=cursor.currentRow()['p1.globaltill'].data()
                    result.append(resultNode)
                transaction.commit()
                del query
                return result
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
    def nChildren( self, label='ROOT' ):
        """Number of children nodes of the given node
        Input: label of the parent node
        Output: number of children
        """
        if label is 'ROOT' :
            #"select count(*) from tagTreeTable"
            try:
                transaction=self.__session.transaction()
                transaction.start(True)
                schema = self.__session.nominalSchema()
                query = schema.tableHandle(self.__tagTreeTableName).newQuery()
                query.addToOutputList('count(*)', 'ct')
                mycounts=coral.AttributeList()
                mycounts.extend('ct', 'unsigned long');
                query.defineOutput( mycounts );
                cursor = query.execute();
                while ( cursor.next() ):
                    n= cursor.currentRow()['ct'].data()
                transaction.commit()
                del query
                return n
            except coral.Exception, er:
                transaction.rollback()
                raise Exception, str(er)
            except Exception, er:
                transaction.rollback()
                raise Exception, str(er)
        else:
            me=self.getNode(label)
            return int((me.rgt-me.lft)/2)
    def deleteSubtree( self, label='ROOT' ):
        """Delete the subtree under the specified node(included).\n
        Input: label of the top node
        query: DELETE FROM treetable WHERE lft >=%me.lft AND rgt<=%me.rgt
        __closeGap()
        """
        transaction=self.__session.transaction()
        try:
            if label=='ROOT' :
                transaction.start(False)
                self.__session.nominalSchema().dropIfExistsTable(self.__tagTreeTableName)
                self.__session.nominalSchema().dropIfExistsTable(self.__tagTreeIDs)
                #editor = tableHandle.dataEditor()
                #conditionData = coral.AttributeList()
                #editor.deleteRows('',conditionData)
                #idtableHandle = self.__session.nominalSchema().tableHandle(self.__tagTreeIDs)
                #ideditor = idtableHandle.dataEditor()
                #ideditor.deleteRows('',conditionData)
                transaction.commit()
            else :
                me=Node.Node()
                parentlft=me.lft
                parentrgt=me.rgt
                n=self.nChildren(label)
                transaction.start(False)
                tableHandle = self.__session.nominalSchema().tableHandle(self.__tagTreeTableName)
                editor = tableHandle.dataEditor()
                condition = 'lft >= :parentlft AND rgt <= :parentrgt'
                conditionData = coral.AttributeList()
                conditionData.extend('parentlft','unsigned long')
                conditionData.extend('parentrgt','unsigned long')
                conditionData['parentlft'].setData(parentlft)
                conditionData['parentrgt'].setData(parentrgt)
                editor.deleteRows( condition, conditionData )
                self.__closeGap(tableHandle,parentlft,parentrgt,n)
                transaction.commit()
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
    def deleteNode( self, label ):
        """
        DELETE FROM treetable WHERE nodename=label
        """
        assert (label !='ROOT')
        transaction=self.__session.transaction()
        try:
            me=Node.Node()
            parentlft=me.lft
            parentrgt=me.rgt
            transaction.start(False)
            tableHandle = self.__session.nominalSchema().tableHandle(self.__tagTreeTableName)
            editor = tableHandle.dataEditor()
            condition = 'nodelabel = :nodelabel'
            conditionData = coral.AttributeList()
            conditionData.extend('nodelabel','string')
            conditionData['nodelabel'].setData(nodelabel)
            editor.deleteRows( condition, conditionData )
            self.__closeGap(tableHandle,parentlft,parentrgt,1)
            transaction.commit()
        except coral.Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)   
    def __openGap(self,tableHandle,parentrgt,n):
        """Update the parent node after inserting. Must be called inside update transaction.\n
        Input: rootrgt is the rgt of the parent node. \n
        Input: n is the number of positions to add
        """
        delta=2*n
        inputData = coral.AttributeList()
        inputData.extend('parentrgt','unsigned long')
        inputData['parentrgt'].setData(parentrgt)
        editor = tableHandle.dataEditor()
        setClause = 'lft=lft+'+str(delta)
        condition = 'lft>:parentrgt'
        editor.updateRows(setClause, condition, inputData)
        setClause = 'rgt=rgt+'+str(delta)
        condition = 'rgt>=:parentrgt'
        editor.updateRows(setClause, condition, inputData)
    def __closeGap(self, tableHandle,parentlft,parentrgt,n):
        """Update the node lft rgt values after removing. Must be called inside update transaction.\n        
        """
        assert (parentlft!=0 and parentrgt!=0 and n!=0)
        assert (parentrgt>parentlft)
        delta=2*n
        editor = tableHandle.dataEditor()
        setClause1 = 'lft=lft-'+str(delta)
        condition1 = 'lft>'+str(parentrgt)
        inputData =coral.AttributeList()
        editor.updateRows(setClause1,condition1,inputData)
        setClause2 = 'rgt=rgt-'+str(delta)
        condition2 = 'rgt>'+str(parentrgt)
        editor.updateRows(setClause2,condition2,inputData)
if __name__ == "__main__":
    os.putenv( "CORAL_AUTH_PATH", "." )
    context = coral.Context()
    context.setVerbosityLevel( 'ERROR' )
    svc = coral.ConnectionService( context )
    session = svc.connect( 'sqlite_file:testTree.db',
                           accessMode = coral.access_Update )
    #session = svc.connect( 'oracle://devdb10/cms_xiezhen_dev',
    #                       accessMode = coral.access_Update )
    try:
        #create a tree named 'mytest'
        mytree=tagTree(session,'mytest2')
        mytree.createTagTreeTable()
        mynode=Node.Node()
        mynode.nodelabel='A'
        mynode.globalsince=1
        mynode.globaltill=10
        mytree.insertNode(mynode,'ROOT')
        result=mytree.getNode('A')
        print result
        mynode=Node.Node()
        mynode.nodelabel='AC1'
        mynode.globalsince=2
        mynode.globaltill=5
        mytree.insertNode(mynode,'A')
        result=mytree.getNode('A')
        print result
        result=mytree.getNode('AC1')
        print result
        result=mytree.getPath('AC1')
        print result
        result=mytree.getAllLeaves()
        print 'all leafs',result
        mynode=Node.Node()
        mynode.nodelabel='AB2'
        mynode.globalsince=3
        mynode.globaltill=7
        mytree.insertNode(mynode,'A')
        result=mytree.getNode('A')
        print 'Node A ',result
        result=mytree.getNode('AB2')
        print 'Node AB2 ',result
        result=mytree.getPath('AB2')
        print 'Path to AB2 ',result
        result=mytree.getAllLeaves()
        print 'all leaves again',result
        print 'number of children ',mytree.nChildren('A')
        print 'number of children ',mytree.nChildren('ROOT')
        result=mytree.getSubtree('A')
        print 'subtree of A ',result
        newtree=tagTree(session,'mynewtest')
        newtree.importFromTree('mytest2')
        del session
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del session
