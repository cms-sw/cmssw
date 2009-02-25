
def inventoryTableName():
	return 'TAGINVENTORY_TABLE'
def inventoryIDTableName():
	return 'TAGINVENTORY_IDS'
def treeTableName(treename):
	return 'TAGTREE_TABLE_'+str.upper(treename)
def treeIDTableName(treename):
	return 'TAGTREE_'+str.upper(treename)+'_IDS'
def commentTableName():
	return 'ENTRYCOMMENT_TABLE'

import coral
def dropAllTreeTables(dbsession):
    """drop all tagtree related tables
    """
    try:
        dbsession.transaction().start(False)
	tablelist = dbsession.nominalSchema().listTables()
	for tname in tablelist:
	   if tname.find('TAGTREE_') != -1:
              dbsession.nominalSchema().dropTable(tname)
	dbsession.transaction().commit()
    except Exception, e:
        raise Exception, str(e)
	    
def tagInTrees(dbsession,tagname,pfn=''):
    """returns the tree names which contain the given tag
       select tagid from taginventory_table where tagname=tagname
       select count(*) from tablename where tablename.tagid=tagid
    """
    try:
	dbsession.transaction().start(True)    
	invquery = dbsession.nominalSchema().tableHandle(inventoryTableName()).newQuery()
	conditionbindDict=coral.AttributeList()
	conditionbindDict.extend('tagname','string')
	conditionbindDict['tagname'].setData(tagname)
	condition='tagname = :tagname'
	if len(pfn) !=0 :
	   condition+=' AND pfn = :pfn'
	   conditionbindDict.extend('pfn','string')
	   conditionbindDict['pfn'].setData(pfn)
	invquery.setCondition(condition,conditionbindDict)
	invquery.addToOutputList('tagid')
	invquery.addToOutputList('pfn')
	cursor = invquery.execute()
	tagidmap={}
	while ( cursor.next() ):
	    tagid=cursor.currentRow()['tagid'].data()
	    pfn=cursor.currentRow()['pfn'].data()
	    tagidmap[pfn]=tagid
	cursor.close()
	dbsession.transaction().commit()
	del invquery
	if len(tagidmap)==0:
	   return tagidmap
   
        result={}
        treetablelist=[]
        dbsession.transaction().start(True)    
	tablelist = dbsession.nominalSchema().listTables()
	for t in tablelist:
	   if t.find('TAGTREE_TABLE_')!= -1:
		   treetablelist.append(t)
	for (pfn,tagid) in tagidmap.items():
	   result[pfn]=[]
	   condition='tagid = :tagid'
	   for t in treetablelist:
	      conditionBind=coral.AttributeList()
	      conditionBind.extend('tagid','unsigned long')
	      conditionBind['tagid'].setData(tagid)
	      q=dbsession.nominalSchema().tableHandle(t).newQuery()
	      q.addToOutputList('count(*)','count')
	      myresult=coral.AttributeList() 
	      myresult.extend('count','unsigned long')
	      q.defineOutput(myresult)
	      cr=q.execute()
	      while (cr.next()):
	        if cr.currentRow()['count']!=0:
		  #print cr.currentRow()['count']
		  result[pfn].append(t[len('TAGTREE_TABLE_'):])
	      cr.close()
	      del q
	dbsession.transaction().commit()	
        return result    
    except Exception, e:
        raise Exception, str(e)

if __name__ == "__main__":
    #context = coral.Context()
    #context.setVerbosityLevel( 'ERROR' )
    svc = coral.ConnectionService()
    session = svc.connect( 'sqlite_file:testInventory.db',
                           accessMode = coral.access_Update )
    try:
        print 'TEST 1'
        intrees=tagInTrees(session,'Tracker_Geometry_CRUZET3')
	print intrees
	print 'TEST 2'
	hello=tagInTrees(session,'Tracker_Geometry_CRUZ3')
	print hello
	print 'TEST 3'
	kitty=tagInTrees(session,'Tracker_Geometry_CRUZET3','pfnme')
	print kitty
	print 'TEST 4'
	mikey=tagInTrees(session,'Tracker_Geometry_CRUZET3','frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_20X_ALIGNMENT')
	print mikey
	del session
    except Exception, e:
        print "Failed in unit test"
        print str(e)
        del session
