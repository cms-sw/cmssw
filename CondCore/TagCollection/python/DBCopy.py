import coral
import CommonUtils, TagTree, tagInventory

class DBCopy(object):
    
    def __init__( self, sourcesession, destsession, rowcachesize=1024 ):
        self.__sourcesession=sourcesession
	self.__destsession=destsession
	self.__rowcachesize=rowcachesize

    def resetrowcachesize( self, newrowcachesize):
	self.__rowcachesize=newrowcachesize

    def copyInventory( self ):
        """copy entire inventory. The original inventory in the source db will be wiped.
	"""
        inv=tagInventory.tagInventory(self.__destsession)
        inv.createInventoryTable()
        dest_transaction=self.__destsession.transaction()
        source_transaction=self.__sourcesession.transaction()
        try:
	    dest_transaction.start(False)
            #copy inventory table
            data=coral.AttributeList()
            my_editor=self.__destsession.nominalSchema().tableHandle(CommonUtils.inventoryTableName()).dataEditor()
            source_transaction.start(True)
            source_query=self.__sourcesession.nominalSchema().tableHandle(CommonUtils.inventoryTableName()).newQuery()
            conditionData=coral.AttributeList()
            source_query.setCondition('',conditionData)
            source_query.setRowCacheSize(self.__rowcachesize)
            my_editor.rowBuffer(data)
            source_query.defineOutput(data)
            bulkOperation=my_editor.bulkInsert(data,self.__rowcachesize)
            cursor=source_query.execute()
            while (cursor.next() ):
                bulkOperation.processNextIteration()
            bulkOperation.flush()
            del bulkOperation
            del source_query

            #copy inventory id table
            source_query=self.__sourcesession.nominalSchema().tableHandle(CommonUtils.inventoryIDTableName()).newQuery()
            my_ideditor=self.__destsession.nominalSchema().tableHandle(CommonUtils.inventoryIDTableName()).dataEditor()
            iddata=coral.AttributeList()
            source_query.setCondition('',conditionData)
            source_query.setRowCacheSize(self.__rowcachesize)
            my_ideditor.rowBuffer(iddata)
            source_query.defineOutput(iddata)
            bulkOperation=my_ideditor.bulkInsert(iddata,self.__rowcachesize)
            cursor=source_query.execute()
            while cursor.next():
                bulkOperation.processNextIteration()
            bulkOperation.flush()
            del bulkOperation
            del source_query
            
            source_transaction.commit()
            dest_transaction.commit()
        except Exception, e:
            source_transaction.rollback()
            dest_transaction.rollback()
            raise Exception, str(e)

    def copyTrees( self, treenames ):
        """copy tree from an external source.
	Merge inventory if existing in the destination
        """
	allleafs=[]
        for treename in treenames:
            t=TagTree.tagTree(self.__sourcesession,treename)
            allleafs.append(t.getAllLeaves())
	#create a unique tag list
	merged={}
	for s in allleafs:
	  for x in s:
	    merged[x.tagid]=1
        sourceinv=tagInventory.tagInventory(self.__sourcesession)
        sourcetags=sourceinv.getAllEntries()
        entries=[]
        for i in merged.keys():
            for n in sourcetags:
                if n.tagid==i:
                    entry={}
                    entry['tagid']=i
                    entry['tagname']=n.tagname
                    entry['pfn']=n.pfn
                    entry['recordname']=n.recordname
                    entry['objectname']=n.objectname
                    entry['labelname']=n.labelname
                    entries.append(entry)
        inv=tagInventory.tagInventory(self.__destsession)
	tagiddict=inv.bulkInsertEntries(entries)
        dest_transaction=self.__destsession.transaction()
        source_transaction=self.__sourcesession.transaction()
        #copy table contents
	try:
	  for treename in treenames:
              desttree=TagTree.tagTree(self.__destsession,treename)
              desttree.createTagTreeTable()
	      dest_transaction.start(False)
	      source_transaction.start(True)
	      #copy tree tables
	      data=coral.AttributeList()
	      dest_editor=self.__destsession.nominalSchema().tableHandle(CommonUtils.treeTableName(treename)).dataEditor()
	      source_query=self.__sourcesession.nominalSchema().tableHandle(CommonUtils.treeTableName(treename)).newQuery()
	      conditionData=coral.AttributeList()
	      source_query.setCondition('',conditionData)
	      source_query.setRowCacheSize(self.__rowcachesize)
              dest_editor.rowBuffer(data)
	      source_query.defineOutput(data)
	      bulkOperation=dest_editor.bulkInsert(data,self.__rowcachesize)
	      cursor=source_query.execute()
              while cursor.next():
	          bulkOperation.processNextIteration()
	      bulkOperation.flush()
	      del bulkOperation
	      del source_query
	      #copy id tables
              iddata=coral.AttributeList()
	      dest_editor=self.__destsession.nominalSchema().tableHandle(CommonUtils.treeIDTableName(treename)).dataEditor()
	      source_query=self.__sourcesession.nominalSchema().tableHandle(CommonUtils.treeIDTableName(treename)).newQuery()
	      conditionData=coral.AttributeList()
	      source_query.setCondition('',conditionData)
	      source_query.setRowCacheSize(self.__rowcachesize)
              dest_editor.rowBuffer(iddata)
	      source_query.defineOutput(iddata)
	      bulkOperation=dest_editor.bulkInsert(iddata,self.__rowcachesize)
	      cursor=source_query.execute()
              while cursor.next():
	          bulkOperation.processNextIteration()
	      bulkOperation.flush()
	      del bulkOperation
	      del source_query
	      source_transaction.commit()
	      dest_transaction.commit()
	      #fix leaf node links
	      desttree.replaceLeafLinks(tagiddict)
        except Exception, e:
            source_transaction.rollback()
            dest_transaction.rollback()
            raise Exception, str(e)
          		
	
    def copyDB( self ):
        """copy all globaltag related tables from an external source.
        The destination database must be empty. If not so, it will be cleaned implicitly. Inventory are implicitly copied as well.  
        """
        dest_transaction=self.__destsession.transaction()
        source_transaction=self.__sourcesession.transaction()
	tablelist=[]
	alltablelist=[]
	trees=[]
        try:
           source_transaction.start(True)
	   tablelist=list(self.__sourcesession.nominalSchema().listTables())
	   source_transaction.commit()
	except Exception, e:
	   source_transaction.rollback()
	   raise Exception, str(e)
	try:
	   i = tablelist.index(CommonUtils.inventoryTableName())
	   alltablelist.append(CommonUtils.inventoryTableName())
	except ValueError:
	   raise 'Error: '+CommonUtils.inventoryTableName()+' does not exist in the source'
	try:
	   i = tablelist.index(CommonUtils.inventoryIDTableName())
	   alltablelist.append(CommonUtils.inventoryIDTableName())
	except ValueError:
	   raise 'Error: '+CommonUtils.inventoryIDTableName()+' does not exist'
	
	for tablename in tablelist:
	   posbeg=tablename.find('TAGTREE_TABLE_')
	   if posbeg != -1:
	      treename=tablename[posbeg+len('TAGTREE_TABLE_'):]
              trees.append(treename)
        for tree in trees:
            try:
              tablelist.index(CommonUtils.treeIDTableName(tree))
            except ValueError:
              print 'non-existing id table for tree ',tree  
              continue
            alltablelist.append(CommonUtils.treeTableName(tree))
            alltablelist.append(CommonUtils.treeIDTableName(tree))
	#schema preparation
	inv=tagInventory.tagInventory(self.__destsession)
	inv.createInventoryTable()
	for treename in trees:
	   t=TagTree.tagTree(self.__destsession,treename)
	   t.createTagTreeTable()
	#copy table contents
	try:
	  for mytable in alltablelist:
	    dest_transaction.start(False)
	    source_transaction.start(True)
	    data=coral.AttributeList()
	    my_editor=self.__destsession.nominalSchema().tableHandle(mytable).dataEditor()
	    source_query=self.__sourcesession.nominalSchema().tableHandle(mytable).newQuery()
	    conditionData=coral.AttributeList()
	    source_query.setCondition('',conditionData)
	    source_query.setRowCacheSize(self.__rowcachesize)
            my_editor.rowBuffer(data)
	    source_query.defineOutput(data)
	    bulkOperation=my_editor.bulkInsert(data,self.__rowcachesize)
	    cursor=source_query.execute()
            while cursor.next():
	       bulkOperation.processNextIteration()
	    bulkOperation.flush()
	    del bulkOperation
	    del source_query
	    source_transaction.commit()
	    dest_transaction.commit()
        except Exception, e:
            source_transaction.rollback()
            dest_transaction.rollback()
            raise Exception, str(e)

if __name__ == "__main__":
    context = coral.Context()
    context.setVerbosityLevel( 'ERROR' )
    svc = coral.ConnectionService( context )
    
    sourcesession = svc.connect( 'sqlite_file:source.db',
                                 accessMode = coral.access_Update )
    destsession = svc.connect( 'sqlite_file:dest.db',
                                 accessMode = coral.access_Update )
    try:
        dbcp=DBCopy(sourcesession,destsession,1024)
        print "TEST copyInventory"
        dbcp.copyInventory()
        print "TEST copytrees"
        treenames=['CRUZET3_V2H']
        dbcp.copyTrees(treenames)
        del sourcesession
        del destsession
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del sourcesession
        del destsession
     
    sourcesession = svc.connect( 'sqlite_file:source.db',
                                 accessMode = coral.access_Update )
    destsession = svc.connect( 'sqlite_file:dest2.db',
                                 accessMode = coral.access_Update )
    try:
        dbcp=DBCopy(sourcesession,destsession,1024)
        print "TEST full dbCopy"
        dbcp.copyDB()
        del sourcesession
        del destsession
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del sourcesession
        del destsession
