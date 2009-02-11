import coral
import CommonUtils, IdGenerator, Node, DBImpl
class  tagInventory(object):
    """Class manages tag inventory 
    """
    def __init__( self , session ):
        """Input: coral session handle
        """
        self.__session = session
        self.__tagInventoryTableName=CommonUtils.inventoryTableName()
        self.__tagInventoryIDName=CommonUtils.inventoryIDTableName()
        self.__tagInventoryTableColumns = {'tagid':'unsigned long', 'tagname':'string', 'pfn':'string','recordname':'string', 'objectname':'string', 'labelname':'string'}
        self.__tagInventoryTableNotNullColumns = ['tagname','pfn','recordname','objectname']
        #self.__tagInventoryTableUniqueColumns = ['tagname']
        self.__tagInventoryTablePK = ('tagid')
    def existInventoryTable( self ):
        """Check if inventory table exists
        """
        try:
            transaction=self.__session.transaction()
            transaction.start(True)
            schema = self.__session.nominalSchema()
            result=schema.existsTable(self.__tagInventoryTableName)
            transaction.commit()
            #print result
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        return result
    def createInventoryTable( self ):
        """Create tag inventory table. Existing table will be deleted. 
        """
        try:
            transaction=self.__session.transaction()
            transaction.start()
            schema = self.__session.nominalSchema()
            schema.dropIfExistsTable( self.__tagInventoryTableName )
            description = coral.TableDescription();
            description.setName( self.__tagInventoryTableName )
            for columnName, columnType in self.__tagInventoryTableColumns.items():
                description.insertColumn(columnName, columnType)
            for columnName in self.__tagInventoryTableNotNullColumns :
                description.setNotNullConstraint(columnName,True)
            #for columnName in self.__tagInventoryTableUniqueColumns :
                #description.setUniqueConstraint(columnName)
            #combinedunique1=('pfn','recordname','objectname','labelname')
            #description.setUniqueConstraint(combinedunique1)
            #combinedunique2=('tagname','pfn')
            #description.setUniqueConstraint(combinedunique2)
            description.setPrimaryKey(  self.__tagInventoryTablePK )
            self.__tagInventoryTableHandle = schema.createTable( description )
            self.__tagInventoryTableHandle.privilegeManager().grantToPublic( coral.privilege_Select )
            #create also the associated id table
            generator=IdGenerator.IdGenerator(schema)
            generator.createIDTable(self.__tagInventoryIDName,True)
            transaction.commit()
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
    def addEntry( self, leafNode ):
        """Add entry into the inventory.\n
        Input: base tag info. If identical data found already exists, do nothing
        Output: tagid. if tagid=0, there's no new entry added, i.e.no new tagid
        
        """
        tagid=0
        transaction=self.__session.transaction()
        duplicate=False
        try:
            transaction.start(True)
            schema = self.__session.nominalSchema()
            query = schema.tableHandle(self.__tagInventoryTableName).newQuery()
            condition='tagname=:tagname'
            conditionbindDict=coral.AttributeList()
            conditionbindDict.extend('tagname','string')
            conditionbindDict['tagname'].setData(leafNode.tagname)
            if len(leafNode.pfn)!=0:
                condition+=' AND pfn=:pfn'
                conditionbindDict.extend('pfn','string')
                conditionbindDict['pfn'].setData(leafNode.pfn)
            query.setCondition(condition,conditionbindDict)
            #duplicate=dbop.existRow(self.__tagInventoryTableName,condition,conditionbindDict)
            cursor=query.execute()
            while( cursor.next() ):
                duplicate=True
                tagid=cursor.currentRow()['tagid'].data()
                cursor.close()
            transaction.commit()
            del query
            #transaction.commit()
            if duplicate is False:
                transaction.start(False)                
                generator=IdGenerator.IdGenerator(schema)
                tagid=generator.getNewID(self.__tagInventoryIDName)
                tabrowValueDict={'tagid':tagid,'tagname':leafNode.tagname,'objectname':leafNode.objectname,'pfn':leafNode.pfn,'labelname':leafNode.labelname,'recordname':leafNode.recordname}
                dbop=DBImpl.DBImpl(schema)
                dbop.insertOneRow(self.__tagInventoryTableName,
                                  self.__tagInventoryTableColumns,
                                  tabrowValueDict)
                generator.incrementNextID(self.__tagInventoryIDName)           
                transaction.commit()
            return tagid
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        
    def addEntriesReplaceService( self, newservicename ):
        """ clone all existing entries only servicename in pfn are different
        return collection of new (oldtagid,newtagid) pair 
        """
        newtaglinks=[]
        transaction=self.__session.transaction()
        try:
            results=[]
            transaction.start(True)
            query = self.__session.nominalSchema().tableHandle(self.__tagInventoryTableName).newQuery()
            for columnName in self.__tagInventoryTableColumns:
                query.addToOutputList(columnName)
            cursor=query.execute()
            while cursor.next():
                tagid=cursor.currentRow()['tagid'].data()
                tagname=cursor.currentRow()['tagname'].data()
                pfn=cursor.currentRow()['pfn'].data()
                (servicename,schemaname)=pfn.rsplit('/',1)
                newpfn=('/').join([newservicename,schemaname])
                #k=(' ').join([tagname,newpfn])
                objname=cursor.currentRow()['objectname'].data()
                redname=cursor.currentRow()['recordname'].data()
                labname=cursor.currentRow()['labelname'].data()
                #return a tuple
                r=(tagid,tagname,newpfn,objname,redname,labname)
                results.append(r)
            transaction.commit()
            del query
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        
        inv=tagInventory(self.__session)
        try:
            for r in results:
                nd=Node.LeafNode()
                oldtagid=r[0]
                nd.tagname=r[1]
                nd.pfn=r[2]
                nd.objectname=r[3]
                nd.recordname=r[4]
                #if not r.items()[1][2] is None:
                nd.labelname=r[5]
                n=inv.addEntry(nd)
                if n==0:
                    raise "addEntry returns 0"
                newtaglinks.append((oldtagid,n))
            return newtaglinks
        except Exception, e:
            print str(e)
            raise Exception, str(e)
    
    def modifyEntriesReplaceService( self, newservicename ):
        """ replace all existing entries replace service name in pfn
        no change in other parameters
        """
        transaction=self.__session.transaction()
        try:
            allpfns=[]
            transaction.start(True)
            query = self.__session.nominalSchema().tableHandle(self.__tagInventoryTableName).newQuery()
            query.addToOutputList('pfn')
            cursor=query.execute()
            while cursor.next():
                pfn=cursor.currentRow()['pfn'].data()
                allpfns.append(pfn)
            transaction.commit()
            del query
        except Exception, er:
            transaction.rollback()
            del query
            raise Exception, str(er)
        try:
            transaction.start(False)
            editor = self.__session.nominalSchema().tableHandle(self.__tagInventoryTableName).dataEditor()
            inputData = coral.AttributeList()
            inputData.extend('newpfn','string')
            inputData.extend('oldpfn','string')
            for pfn in allpfns:
                (servicename,schemaname)=pfn.rsplit('/',1)
                newpfn=('/').join([newservicename,schemaname])
                inputData['newpfn'].setData(newpfn)
                inputData['oldpfn'].setData(pfn)
                editor.updateRows( "pfn = :newpfn", "pfn = :oldpfn", inputData )
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)

    def cloneEntry( self, sourcetagid, pfn ):
        """ clone an existing entry with different pfn parameter
        Input: sourcetagid, pfn.
        Output: tagid of the new entry. Return 0 in case no new entry created or required entry already exists. 
        """
        newtagid=sourcetagid
        if len(pfn)==0:
            return newtagid
        try:
            nd=self.getEntryById(sourcetagid)
            if nd.tagid==0:
                return newtagid
            oldpfn=nd.pfn
            if oldpfn==pfn:
                return nd.tagid
            transaction=self.__session.transaction()
            transaction.start(False)
            schema = self.__session.nominalSchema()
            generator=IdGenerator.IdGenerator(schema)
            newtagid=generator.getNewID(self.__tagInventoryIDName)
            tabrowValueDict={'tagid':newtagid,'tagname':nd.tagname,'objectname':nd.objectname,'pfn':pfn,'labelname':nd.labelname,'recordname':nd.recordname}
            dbop=DBImpl.DBImpl(schema)
            dbop.insertOneRow(self.__tagInventoryTableName,
                              self.__tagInventoryTableColumns,
                              tabrowValueDict)
            generator.incrementNextID(self.__tagInventoryIDName)
            transaction.commit()
            return newtagid
        except Exception, er:
            transaction.rollback()
            raise Exception, str(er)
        
    def getEntryByName( self, tagName, pfn ):
        """Get basic tag from inventory by tagName+pfn. pfn can be empty\n
        Input: tagname,pfn
        Output: leafNode
        throw if more than one entry is found.
        """
        leafnode = Node.LeafNode()
        transaction=self.__session.transaction()
        try:
            transaction.start(True)
            query = self.__session.nominalSchema().tableHandle(self.__tagInventoryTableName).newQuery()
            for columnName in self.__tagInventoryTableColumns:
                query.addToOutputList(columnName)
            conditionData = coral.AttributeList()
            condition = "tagname=:tagname"
            conditionData.extend( 'tagname','string' )
            conditionData['tagname'].setData(tagName)
            if len(pfn)!=0 :
                condition += " AND pfn=:pfn"
                conditionData.extend( 'pfn','string' )
                conditionData['pfn'].setData(pfn)
            query.setCondition(condition,conditionData)
            cursor = query.execute()
            counter=0
            while ( cursor.next() ):
                if counter > 0 :
                    raise ValueError, "tagName "+tagName+" is not unique, please further specify parameter pfn"
                counter+=1
                leafnode.tagid=cursor.currentRow()['tagid'].data()
                leafnode.tagname=cursor.currentRow()['tagname'].data()
                leafnode.objectname=cursor.currentRow()['objectname'].data()
                leafnode.pfn=cursor.currentRow()['pfn'].data()
                leafnode.labelname=cursor.currentRow()['labelname'].data()
                leafnode.recordname=cursor.currentRow()['recordname'].data()
            transaction.commit()
            del query
            return leafnode
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
    def getEntryById( self, tagId ):
        """Get basic tag from inventory by id.\n
        Input: tagid
        Output: leafNode
        """
        leafnode = Node.LeafNode()
        transaction=self.__session.transaction()
        try:
            transaction.start(True)
            query = self.__session.nominalSchema().tableHandle(self.__tagInventoryTableName).newQuery()
            for columnName in self.__tagInventoryTableColumns:
                query.addToOutputList(columnName)
            condition = "tagid=:tagid"
            conditionData = coral.AttributeList()
            conditionData.extend( 'tagid','unsigned long' )
            conditionData['tagid'].setData(tagId)
            query.setCondition( condition, conditionData)
            cursor = query.execute()
            while ( cursor.next() ):
                #print 'got it'
                leafnode.tagid=cursor.currentRow()['tagid'].data()
                leafnode.tagname=cursor.currentRow()['tagname'].data()
                leafnode.objectname=cursor.currentRow()['objectname'].data()
                leafnode.pfn=cursor.currentRow()['pfn'].data()
                leafnode.labelname=cursor.currentRow()['labelname'].data()
                leafnode.recordname=cursor.currentRow()['recordname'].data()
            transaction.commit()
            del query
            return leafnode
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
    def getAllEntries( self ):
        """Get all entries in the inventory
        Output: list of leafNode objects
        """
        result=[]
        transaction=self.__session.transaction()
        try:
            transaction.start(True)
            query = self.__session.nominalSchema().tableHandle(self.__tagInventoryTableName).newQuery()
            for columnName in self.__tagInventoryTableColumns:
                query.addToOutputList(columnName)    
            cursor = query.execute()
            while ( cursor.next() ):
                leafnode = Node.LeafNode()
                leafnode.tagid=cursor.currentRow()['tagid'].data()
                leafnode.tagname=cursor.currentRow()['tagname'].data()
                leafnode.objectname=cursor.currentRow()['objectname'].data()
                leafnode.pfn=cursor.currentRow()['pfn'].data()
                leafnode.recordname=cursor.currentRow()['recordname'].data()
                leafnode.labelname=cursor.currentRow()['labelname'].data()
                result.append(leafnode)
            transaction.commit()
            del query
            return result
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
    def deleteAllEntries( self ):
        """Delete all entries in the inventory
        """
        try:
            transaction=self.__session.transaction()
            transaction.start(False)
            schema = self.__session.nominalSchema()
            dbop=DBImpl.DBImpl(schema)
            inputData = coral.AttributeList()
            dbop.deleteRows(self.__tagInventoryTableName,
                            '',
                            inputData)
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
    def deleteEntryByName( self, tagname ):
        """Delete entry with given tag name
        """
        try:
            transaction=self.__session.transaction()
            transaction.start(False)
            schema = self.__session.nominalSchema()
            dbop=DBImpl.DBImpl(schema)
            inputData = coral.AttributeList()
            inputData.extend( "tagname","string" )
            inputData[0].setData(tagname)
            dbop.deleteRows(self.__tagInventoryTableName,
                            'tagname=:tagname',
                            inputData)
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
    def replaceTagLabel( self, tagname, label ):
        """Replace the run time label of the given tag
        """
        try:
            transaction=self.__session.transaction()
            transaction.start(False)
            schema = self.__session.nominalSchema()
            inputData = coral.AttributeList()
            inputData.extend( "labelname","string" )
            inputData.extend( "tagname", "string" ) 
            inputData[0].setData(label)
            inputData[1].setData(tagname)
            editor = schema.tableHandle(self.__tagInventoryTableName).dataEditor()
            editor.updateRows( "labelname=:labelname", "tagname=:tagname", inputData )
            transaction.commit()
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)

    def bulkInsertEntries( self, entries ): 
        """insert a chunk of entries.
        Input: entries [{tagid:unsigned long, tagname:string , pfn:string , recordname:string , objectname:string, labelname:string }]
        Output: {oldtagid:newtagid} of the inserted entries. If tag already exists, old tagid is returned
        """
        transaction=self.__session.transaction()
        results={}
        ihad=[]
        try:
            if self.existInventoryTable():
                ihad=self.getAllEntries()
            else:    
                self.createInventoryTable()
            #clean input list removing duplicates
            for e in entries:
                for n in ihad:
                    if n.tagname==e['tagname'] and n.pfn==e['pfn']:
                        results[n.tagid]=n.tagid
                        entries.remove(e)
            transaction.start(False)
            query=self.__session.nominalSchema().tableHandle(self.__tagInventoryIDName).newQuery()
            query.addToOutputList('nextID')
            query.setForUpdate()
            cursor = query.execute()
            nextid=0
            while cursor.next():
                nextid=cursor.currentRow()[0].data()
            idEditor = self.__session.nominalSchema().tableHandle(self.__tagInventoryIDName).dataEditor()
            inputData = coral.AttributeList()
            inputData.extend( 'delta', 'unsigned long' )

            delta=len(entries)
            if nextid==0:
                nextid=1
                delta=1

            inputData['delta'].setData(delta)
            idEditor.updateRows('nextID = nextID + :delta','',inputData)

            dataEditor=self.__session.nominalSchema().tableHandle(self.__tagInventoryTableName).dataEditor()
            insertdata=coral.AttributeList()
            insertdata.extend('tagid','unsigned long')
            insertdata.extend('tagname','string')
            insertdata.extend('pfn','string')
            insertdata.extend('recordname','string')
            insertdata.extend('objectname','string')
            insertdata.extend('labelname','string')
            bulkOperation=dataEditor.bulkInsert(insertdata,delta)
            for entry in entries:
                insertdata['tagid'].setData(nextid)
                insertdata['tagname'].setData(entry['tagname'])
                insertdata['pfn'].setData(entry['pfn'])
                insertdata['recordname'].setData(entry['recordname'])
                insertdata['objectname'].setData(entry['objectname'])
                insertdata['labelname'].setData(entry['labelname'])
                bulkOperation.processNextIteration()
                results[entry['tagid']]=nextid
                nextid=nextid+1
            bulkOperation.flush()
            transaction.commit()
            del bulkOperation
            del query
            return results
        except Exception, e:
            transaction.rollback()
            raise Exception, str(e)
        
if __name__ == "__main__":
    context = coral.Context()
    context.setVerbosityLevel( 'ERROR' )
    svc = coral.ConnectionService( context )
    session = svc.connect( 'sqlite_file:testInventory.db',
                           accessMode = coral.access_Update )
    try:
        inv=tagInventory(session)
        inv.createInventoryTable()
        tagentry=Node.LeafNode()
        tagentry.tagname='ecalpedestalsfromonline'
        tagentry.objectname='EcalPedestals'
        tagentry.pfn='oracle://devdb10/CMS_COND_ECAL'
        tagentry.recordname='EcalPedestalsRcd'
        tagentry.labelname=''
        inv.addEntry(tagentry)
        tagentry.tagname='crap'
        tagentry.objectname='MyPedestals'
        tagentry.pfn='oracle://devdb10/CMS_COND_ME'
        tagentry.recordname='MyPedestalsRcd'
        tagentry.labelname='mylabel'
        inv.addEntry(tagentry)
        result=inv.getAllEntries()
        print 'get all##\t',result
        result=inv.getEntryByName('ecalpedestalsfromonline','oracle://devdb10/CMS_COND_ECAL')
        print 'get ecalpedestalsfromonline##\t',result
        result=inv.getEntryByName('crap','oracle://devdb10/CMS_COND_ME')
        print 'get crap##\t',result
        result=inv.getEntryById(0)
        print 'get by id 0##\t',result
        inv.deleteEntryByName('ecalpedestalsfromonline')
        print 'TESTING getEntryByName'
        result=inv.getEntryByName('ecalpedestalsfromonline','oracle://devdb10/CMS_COND_ECAL')
        print 'get ecalpedestalsfromonline##\t',result
        result=inv.getEntryByName('crap','oracle://devdb10/CMS_COND_ME')
        print 'get crap##\t',result
        print 'TESTING cloneEntry'
        newid=inv.cloneEntry(result.tagid,'fontier://crap/crap')
        print 'newid ',newid
        print 'TESTING addEntriesReplaceService'
        newtagids=inv.addEntriesReplaceService('oracle://cms_orcoff_int')
        print 'new tag ids ',newtagids
        print 'TESTING modifyEntriesReplaceService'
        inv.modifyEntriesReplaceService('oracle://cms_orcoff_int9r')
        print 'TESTING bulkInsertEntries'
        entries=[]
        entries.append({'tagid':10,'tagname':'tag1','pfn':'dbdb','recordname':'myrcd','objectname':'bobo','labelname':''})
        entries.append({'tagid':11,'tagname':'tag2','pfn':'dbdb','recordname':'mdrcd','objectname':'bobo','labelname':''})
        entries.append({'tagid':12,'tagname':'tag3','pfn':'dbdb','recordname':'ndrcd','objectname':'bobo','labelname':''})
        entries.append({'tagid':13,'tagname':'tag4','pfn':'dbdb','recordname':'ndrcd','objectname':'bobo','labelname':''})
        a=inv.bulkInsertEntries(entries)
        print a
        del session
        
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del session
        
