#
# Revision DML API
#

#def revisionsInBranch(schema,branchid):
# backup implementation without subquery
#    '''
#    returns all revision values in a branch
#    result=[revision_id]
#    select distinct branch_id from revisions where branch_id>:branchid;
#    select revision_id from revisions where branch_id=:branchid and revision_id not in (branch_ids);
#    if the branchid matches and the revisionid is not in the branchid collection, then this revision is in the branch
#    select revision_id from revisions where branch_id=:branchid and revision_id not in (select distinct branch_id from revisions where branch_id>:branchid)
#    require also revisionid>branchid
#    '''
#    result=[]
#    try:
#        nextbranches=[]
#        nextbranchesStr=''
#        qHandle=schema.newQuery()
#        qHandle.addToTableList( nameDealer.revisionTableName() )
#        qHandle.addToOutputList('distinct BRANCH_ID','branch_id')
#        qCondition=coral.AttributeList()
#        qCondition.extend('branchid','unsigned long long')
#        qCondition['branchid'].setData(branchid)
#        qResult=coral.AttributeList()
#        qResult.extend('branch_id','unsigned long long')
#        qHandle.defineOutput(qResult)
#        qHandle.setCondition('BRANCH_ID>:branchid',qCondition)
#        cursor=qHandle.execute()
#        while cursor.next():
#            nextbranches.append(cursor.currentRow()['branch_id'].data())
#        del qHandle
#        print 'nextbranches ',nextbranches
#        conditionStr='BRANCH_ID=:branchid'
#        if len(nextbranches)!=0:
#            nextbranchesStr=','.join([str(x) for x in nextbranches])
#            print nextbranchesStr
#            conditionStr+=' AND REVISION_ID NOT IN ('+nextbranchesStr+')'
#            print conditionStr
#        qHandle=schema.newQuery()
#        qHandle.addToTableList( nameDealer.revisionTableName() )
#        qHandle.addToOutputList('REVISION_ID','revision_id')
#        qCondition=coral.AttributeList()
#        qCondition.extend('branchid','unsigned long long')
#        qCondition['branchid'].setData(branchid)
#        qResult=coral.AttributeList()
#        qResult.extend('revision_id','unsigned long long')
#        qHandle.defineOutput(qResult)
#        qHandle.setCondition(conditionStr,qCondition)
#        cursor=qHandle.execute()
#        while cursor.next():
#            result.append(cursor.currentRow()['revision_id'].data())
#        del qHandle
#        return result
#    except Exception,e :
#        raise RuntimeError(' revisionDML.revisionsInBranch: '+str(e))


#==============================
# SELECT
#==============================
def branchType(schema,name):
    '''
    output: tag,branch
    the difference between tag and branch: tag is an empty branch
    select count(revision_id) from revisions where branch_name=:name
    if >0: is real branch
    else: is tag
    '''
    pass
def revisionsInBranch(schema,branchid):
    '''
    returns all revision values in a branch/tag
    result=[revision_id]
    select revision_id from revisions where branch_id=:branchid and revision_id not in (select distinct branch_id from revisions where branch_id>:branchid)
    '''
    result=[]
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.revisionTableName() )
        qHandle.addToOutputList('REVISION_ID','revision_id')
        qCondition=coral.AttributeList()
        qCondition.extend('branchid','unsigned long long')
        qCondition['branchid'].setData(branchid)
        qResult=coral.AttributeList()
        qResult.extend('revision_id','unsigned long long')
        qHandle.defineOutput(qResult)
        conditionStr='BRANCH_ID=:branchid AND REVISION_ID NOT IN A'
        subquery=qHandle.defineSubQuery('A')
        subquery.addToOutputList('distinct BRANCH_ID')
        subquery.addToTableList( nameDealer.revisionTableName() )
        subqueryCondition=coral.AttributeList()
        subqueryCondition.extend('branchid','unsigned long long')
        subqueryCondition['branchid'].setData(branchid)
        subquery.setCondition('BRANCH_ID>:branchid',subqueryCondition)
        qHandle.addToTableList( 'A')
        qHandle.setCondition(conditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            result.append(cursor.currentRow()['revision_id'].data())
        del qHandle
        return result
    except :
        if qHandle: del qHandle
        raise 
    
def revisionsInBranchName(schema,branchname):
    '''
    returns all revisions in a branch/tag by name
    '''
    result=[]
    try:
        (revision_id,branch_id)=getBranchByName(schema,branchname)
        result=revisionsInBranch(schema,revision_id)
        return result
    except :
        raise 
    
def existsEntryInBranch(schema,datatableName,entryname,branch):
    '''
    whether an entry(by name) already exists in the given branch
    select e.entry_id from entrytable e,revisiontable r where r.revision_id=e.revision_id and e.name=:entryname and r.branch_name=branchname/branch_id
    input:
        if isinstance(branch,str):byname
        else: byid
    output:
        (true/false,entry_id/None)
    '''
    try:
        result=(False,None)
        byname=False
        if isinstance(branch,str):
            byname=True
        result=[false,None]
        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.entryTableName(datatableName),'e' )
        qHandle.addToTableList( nameDealer.revisionTableName(),'r' )
        qHandle.addToOutputList('e.ENTRY_ID','entry_id')
        qCondition=coral.AttributeList()
        qCondition.extend('entryname','string')
        qCondition['entryname'].setData(entryname)
        qConditionStr='r.REVISION_ID=e.REVISION_ID and e.NAME=:entryname and '
        if byname:
            qCondition.extend('branch_name','string')
            qCondition['branch_name'].setData(branch)
            qConditionStr+='r.BRANCH_NAME=:branch_name'
        else:
            qCondition.extend('branch_id','unsigned long long')
            qCondition['branch_id'].setData(branch)
            qConditionStr+='r.BRANCH_ID=:branch_id'
        qResult=coral.AttributeList()
        qResult.extend('entry_id','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            entry_id=cursor.currentRow()['entry_id'].data()
            result=(True,entry_id)
        return result
    except :
        if qHandle: del qHandle
        raise 
    
def dataRevisionsOfEntry(schema,datatableName,entry,revrange):
    '''
    all data version of the given entry whose revision falls in branch revision range
    select d.data_id,r.revision_id from datatable d, datarevmaptable r where d.entry_id(or name )=:entry and d.data_id=r.data_id
    input: if isinstance(entry,str): d.entry_name=:entry ; else d.entry_id=:entry
    '''
    try:
        result=[]
        byname=False
        if isinstance(entry,str):
            byname=True
        qHandle=schema.newQuery()
        qHandle.addToTableList( datatableName,'d' )
        qHandle.addToTableList( nameDealer.revmapTableName(datatableName), 'r')
        qHandle.addToOutputList('d.DATA_ID','data_id')
        qHandle.addToOutputList('r.REVISION_ID','revision_id')
        qCondition=coral.AttributeList()
        qConditionStr='d.DATA_ID=r.DATA_ID and '
        if byname:
            qCondition.extend('entry_name','string')
            qCondition['entry_name'].setData(entry)
            qConditionStr+='d.ENTRY_NAME=:entry_name'
        else:
            qCondition.extend('entry_id','unsigned long long')
            qCondition['entry_id'].setData(entry)
            qConditionStr+='d.ENTRY_ID=:entry_id'
        qResult=coral.AttributeList()
        qResult.extend('data_id','unsigned long long')
        qResult.extend('revision_id','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            data_id=cursor.currentRow()['data_id'].data()
            revision_id=cursor.currentRow()['revision_id'].data()
            if revision_id in branchrevisionFilter:
                result.append(data_id)        
    except :
        if qHandle: del qHandle
        raise

def lastestDataRevisionOfEntry(schema,datatableName,entry,revrange):
    '''
    return max(data_id) of all datarevisionofEntry
    '''
    result=dataRevisionsOfEntry(schema,datatableName,entry,revrange)
    if result and len(result)!=0: return max(result)
    return None

def branchInfoByName(schema,branchName):
    '''
    select (revision_id,branch_id) from revisions where name=:branchName
    '''
    try:
         qHandle=schema.newQuery()
         qHandle.addToTableList( nameDealer.revisionTableName() )
         qHandle.addToOutputList('REVISION_ID','revision_id')
         qHandle.addToOutputList('BRANCH_ID','branch_id')
         qCondition=coral.AttributeList()
         qCondition.extend('name','string')
         qCondition['name'].setData(branchName)
         qResult=coral.AttributeList()
         qResult.extend('revision_id','unsigned long long')
         qResult.extend('branch_id','unsigned long long')
         qHandle.defineOutput(qResult)
         qHandle.setCondition('NAME=:name',qCondition)
         cursor=qHandle.execute()
         while cursor.next():
             revision_id=cursor.currentRow()['revision_id'].data()
             branch_id=cursor.currentRow()['branch_id'].data()
         del qHandle
         return (revision_id,branch_id)
    except Exception,e :
        raise RuntimeError(' revisionDML.getBranchByName: '+str(e))
    

#=======================================================
#
#   INSERT requires in update transaction
#
#=======================================================
def bookNewEntry(schema,datatableName):
    '''
    allocate new revision_id,entry_id,data_id
    '''
    n=newSchemaNames()
    entrytableName=nameDealer.entryTableName(datatableName)
    iddealer=idDealer.idDealer(schema)
    revision_id=iddealer.generateNextIDForTable(n.revisionstable)
    data_id=iddealer.generateNextIDForTable(datatableName)
    entry_id=iddealer.generateNextIDForTable(entrytableName)
    return (revision_id,entry_id,data_id)

def bookNewRevision(schema,datatableName):
    '''
    allocate new revision_id,data_id
    '''
    n=newSchemaNames()
    iddealer=idDealer.idDealer(schema)
    revision_id=iddealer.generateNextIDForTable(n.revisionstable)
    data_id=iddealer.generateNextIDForTable(datatableName)
    return (revision_id,data_id)

def addEntryToBranch(schema,datatableName,revision_id,entry_id,data_id,branchname,name,comment=''):
    (parentrevision_id,parentbranch_id)=getBranchByName(schema,branchname)
    print 'addEntryToBranch ',parentrevision_id,parentbranch_id
    addEntry(schema,datatableName,revision_id,entry_id,data_id,parentrevision_id,name,comment)
     
def addEntry(schema,datatableName,entryinfo,branchinfo):
    '''
    input:
        entryinfo (revision_id(0),entry_id(1),entry_name(2),data_id(3))
        branchinfo (branch_id,branch_name)
    1.allocate and insert a new revision into the revisions table
    2.allocate and insert a new entry into the entry table with the new revision
    3.inset into data_rev table with new data_id ,revision)id mapping
    
    insert into revisions(revision_id,branch_id,branch_name,comment,ctime) values()
    insert into datatablename_entries (entry_id,revision_id) values()
    insert into datatablename_rev(data_id,revision_id) values()
    '''
    revisiontableName=nameDealer.revisionTableName()
    entrytableName=nameDealer.entryTableName(datatableName)
    revtableName=nameDealer.revmapTableName(datatableName)
    
    db=dbUtil.dbUtil(schema)
    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['BRANCH_ID']='unsigned long long'
    tabrowDefDict['BRANCH_NAME']='string'
    tabrowDefDict['CTIME']='time stamp'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=entryinfo[0]
    tabrowValueDict['BRANCH_ID']=branchinfo[0]
    tabrowValueDict['BRANCH_NAME']=branchinfo[1]
    tabrowValueDict['CTIME']=coral.TimeStamp()
    db.insertOneRow(revisiontableName,tabrowDefDict,tabrowValueDict)
    
    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['ENTRY_ID']='unsigned long long'    
    tabrowDefDict['NAME']='string'
    
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=entryinfo[0]
    tabrowValueDict['ENTRY_ID']=entryinfo[1]
    tabrowValueDict['NAME']=entryinfo[2]
    db.insertOneRow(entrytableName,tabrowDefDict,tabrowValueDict)

    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['DATA_ID']='unsigned long long'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=entryinfo[0]
    tabrowValueDict['DATA_ID']=entryinfo[3]
    db.insertOneRow(revtableName,tabrowDefDict,tabrowValueDict)
    
def addDataRevision(schema,datatableName,revision_id,data_id,branch_id,branch_name):
    '''
    1.insert a new revision into the revisions table
    2.insert into data_id, revision_id pair to  datatable_revmap
    insert into revisions(revision_id,branch_id,branch_name,ctime) values()
    insert into datatable_rev(data_id,revision_id) values())
    '''
    revisiontableName=nameDealer.revisionTableName()
    revtableName=nameDealer.revmapTableName(datatableName)
    
    db=dbUtil.dbUtil(schema)
    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['BRANCH_ID']='unsigned long long'
    tabrowDefDict['BRANCH_NAME']='string'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['BRANCH_ID']=branch_id
    tabrowValueDict['BRANCH_NAME']=branch_name
    db.insertOneRow(revisiontableName,tabrowDefDict,tabrowValueDict)

    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['DATA_ID']='unsigned long long'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['DATA_ID']=data_id
    db.insertOneRow(revtableName,tabrowDefDict,tabrowValueDict)

def createNewBranch(schema,revision_id,name,parent_id,parent_name,comment=''):
    '''
    create a new branch/tag under given parentnode
    insert into revisions(revision_id,branch_id,branch_name,name,comment,ctime) values()
    '''
    db=dbUtil.dbUtil(schema)
    tabrowDefDict={}
    tabrowDefDict['REVISION_ID']='unsigned long long'
    tabrowDefDict['BRANCH_ID']='unsigned long long'
    tabrowDefDict['BRANCH_NAME']='string'
    tabrowDefDict['NAME']='string'
    tabrowDefDict['COMMENT']='string'
    tabrowDefDict['CTIME']='time stamp'
    tabrowValueDict={}
    tabrowValueDict['REVISION_ID']=revision_id
    tabrowValueDict['BRANCH_ID']=parent_id
    tabrowValueDict['BRANCH_NAME']=parent_name
    tabrowValueDict['NAME']=name
    tabrowValueDict['COMMENT']=comment
    tabrowValueDict['CTIME']=coral.TimeStamp()
    db.insertOneRow(nameDealer.revisionTableName(),tabrowDefDict, tabrowValueDict )


