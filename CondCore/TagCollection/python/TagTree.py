import coral
import IdGenerator, Node, DBImpl
class tagTree(object):
    """Class manages tag tree
    """
    def __init__( self, session ):
        self.__session = session
        self.__tagTreeTableName = 'TAGTREE_TABLE'
        self.__tagInventoryTableName = 'TAGINVENTORY_TABLE'
        self.__tagTreeTableColumns = {'nodeid':'unsigned long', 'nodelabel':'string', 'lft':'unsigned long', 'rgt':'unsigned long', 'parentid':'unsigned long', 'tagid':'unsigned long', 'globalSince':'unsigned long long', 'globalTill':'unsigned long long','comment':'string'}
        self.__tagTreeTableNotNullColumns = ['nodelabel','lft','rgt','parentid']
        self.__tagTreeTableUniqueColumns = ['nodelabel','lft','rgt']
        self.__tagTreeTablePK = ('nodeid')
    def createTagTreeTable( self ):
        """Create tag tree table. Existing table will be deleted. 
        """
        try:
            transaction=self.__session.transaction()
            transaction.start()
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
            #description.createForeignKey(self.__tagTreeTableFK)
            description.createForeignKey('tagid_FK','tagid',self.__tagInventoryTableName,'tagid')
            self.__tagTreeTableHandle = schema.createTable( description )
            self.__tagTreeTableHandle.privilegeManager().grantToPublic( coral.privilege_Select )
            #create also the associated id table
            generator=IdGenerator.IdGenerator(schema)
            generator.createIDTable(self.__tagTreeTableName,True)
            transaction.commit()
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
    def insertNode( self, node, parentLabel='ROOT' ):
        """Append a new node to specified parent. \n
        Input: new node. \n
        Input: name of the parent node. \n
        Output: result nodeid  
        """
        nodeid=0
        parentid=0
        tagid=0
        transaction=self.__session.transaction()
        try:
            transaction.start(True)
            schema = self.__session.nominalSchema()
            generator=IdGenerator.IdGenerator(schema)
            nodeid=generator.getNewID(generator.getIDTableName(self.__tagTreeTableName))
            transaction.commit()
            if parentLabel == 'ROOT':
                parentid=0
            else:
                parentNode=self.getNode(parentLabel)
                parentid=parentNode.nodeid
            nodelabel=node.nodelabel
            globalSince=node.globalSince
            globalTill=node.globalTill
            if node.isLeaf:
                tagid=node.tagid
            lft=0
            rgt=0
            if parentLabel == 'ROOT':
                lft=1
                rgt=2
            else:
                lft=parentNode.rgt
                rgt=parentNode.rgt+1
            tabrowValueDict={'nodeid':nodeid, 'nodelabel':nodelabel,
                            'lft':lft, 'rgt':rgt, 'parentid':parentid,
                            'tagid':tagid, 'globalSince':globalSince,
                            'globalTill':globalTill,'comment':''
                            }
            transaction.start(False)
            tableHandle = self.__session.nominalSchema().tableHandle(self.__tagTreeTableName)
            if parentLabel != 'ROOT':
                self.__openGap( tableHandle,parentNode.rgt,1 )
            dbop=DBImpl.DBImpl(schema)
            dbop.insertOneRow(self.__tagTreeTableName,
                              self.__tagTreeTableColumns,
                              tabrowValueDict)
            generator.incrementNextID(generator.getIDTableName(self.__tagTreeTableName))
            transaction.commit()
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
            schema = self.__session.nominalSchema()
            query = schema.tableHandle(self.__tagTreeTableName).newQuery()
            condition = 'nodelabel =:nodelabel'
            conditionData = coral.AttributeList()
            conditionData.extend( 'nodelabel','string' )
            conditionData['nodelabel'].setData(label)
            query.setCondition( condition, conditionData)
            cursor = query.execute()
            while ( cursor.next() ):
                result.nodeid=cursor.currentRow()['nodeid'].data()
                result.nodelabel=cursor.currentRow()['nodelabel'].data()
                result.lft=cursor.currentRow()['lft'].data()
                result.rgt=cursor.currentRow()['rgt'].data()
                result.parentid=cursor.currentRow()['parentid'].data()
                result.globalSince=cursor.currentRow()['globalSince'].data()
                result.globalTill=cursor.currentRow()['globalTill'].data()
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
                resultNode.nodeid=cursor.currentRow()['nodeid'].data()
                resultNode.nodelabel=cursor.currentRow()['nodelabel'].data()
                resultNode.lft=cursor.currentRow()['lft'].data()
                resultNode.rgt=cursor.currentRow()['rgt'].data()
                resultNode.parentid=cursor.currentRow()['parentid'].data()
                resultNode.globalSince=cursor.currentRow()['globalSince'].data()
                resultNode.globalTill=cursor.currentRow()['globalTill'].data()
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
                pass
            else:
                me=self.getNode(label)
                parentlft=me.lft
                parentrgt=me.rgt
                transaction=self.__session.transaction()
                transaction.start(True)
                schema = self.__session.nominalSchema()
                query = schema.tableHandle(self.__tagTreeTableName).newQuery()
                condition = 'lft>=:parentlft AND rgt<=:parentrgt'
                query.addToOrderList( "lft" );
                conditionData = coral.AttributeList()
                conditionData.extend( 'parentlft','unsigned long' )
                conditionData.extend( 'parentrgt','unsigned long' )
                conditionData['parentlft'].setData(parentlft)
                conditionData['parentrgt'].setData(parentrgt)
                query.setCondition( condition, conditionData)
                cursor = query.execute()
                while ( cursor.next() ):
                    resultNode=Node.Node()
                    resultNode.nodeid=cursor.currentRow()['nodeid'].data()
                    resultNode.nodelabel=cursor.currentRow()['nodelabel'].data()
                    resultNode.lft=cursor.currentRow()['lft'].data()
                    resultNode.rgt=cursor.currentRow()['rgt'].data()
                    resultNode.parentid=cursor.currentRow()['parentid'].data()
                    resultNode.globalSince=cursor.currentRow()['globalSince'].data()
                    resultNode.globalTill=cursor.currentRow()['globalTill'].data()
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
    def deleteNode( self, label='ROOT' ):
        """
        DELETE FROM treetable WHERE nodename=label
        """
        pass
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
        condition = 'rgt>:parentrgt'
        editor.updateRows(setClause, condition, inputData)
        setClause = 'rgt=rgt+'+str(delta)
        condition = 'rgt>=:parentrgt'
        editor.updateRows(setClause, condition, inputData)
    def __closeGap(self, tableHandle,parentlft,parentrgt,n):
        """Update the node lft rgt values after removing. Must be called inside update transaction.\n
        
        """
        pass
if __name__ == "__main__":
    context = coral.Context()
    context.setVerbosityLevel( 'ERROR' )
    svc = coral.ConnectionService( context )
    session = svc.connect( 'sqlite_file:testTree.db',
                           accessMode = coral.access_Update )
    try:
        mytree=tagTree(session)
        mytree.createTagTreeTable()
        mynode=Node.Node()
        mynode.nodelabel='A'
        mynode.globalSince=1
        mynode.globalTill=10
        mytree.insertNode(mynode,'ROOT')
        result=mytree.getNode('A')
        print result
        mynode=Node.Node()
        mynode.nodelabel='AC1'
        mynode.globalSince=2
        mynode.globalTill=5
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
        mynode.globalSince=3
        mynode.globalTill=7
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
        del session
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del session
