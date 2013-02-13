############################################################
# LumiDB Revision and Versioning API
#
# Author:      Zhen Xie
############################################################

import coral
from RecoLuminosity.LumiDB import nameDealer,idDealer,dbUtil
#==============================
# SELECT
#==============================
def revisionsInTag(schema,tagrevisionid,branchid):
    '''
    returns all revisions before tag in selected branch
    select revision_id from revisions where revision_id!=0 and revision_id<tagrevisionid and branch_id=:branchid
    result=[revision_id]
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        nextbranches=[]
        qHandle.addToTableList( nameDealer.revisionTableName() )
        qHandle.addToOutputList('distinct BRANCH_ID','branch_id')
        qCondition=coral.AttributeList()
        qCondition.extend('branchid','unsigned long long')
        qCondition['branchid'].setData(branchid)
        qResult=coral.AttributeList()
        qResult.extend('branch_id','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('BRANCH_ID>:branchid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            nextbranches.append(cursor.currentRow()['branch_id'].data())
        del qHandle
        candidates=[]
        conditionStr='REVISION_ID!=0 and BRANCH_ID=:branchid and REVISION_ID<:tagrevisionid'
        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.revisionTableName() )
        qHandle.addToOutputList('REVISION_ID','revision_id')
        qCondition=coral.AttributeList()
        qCondition.extend('branchid','unsigned long long')
        qCondition.extend('tagrevisionid','unsigned long long')
        qCondition['branchid'].setData(branchid)
        qCondition['tagrevisionid'].setData(tagrevisionid)
        qResult=coral.AttributeList()
        qResult.extend('revision_id','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(conditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            candidates.append(cursor.currentRow()['revision_id'].data())
        del qHandle
        for c in candidates:
            if c in nextbranches:
                continue
            result.append(c)
        return result
    except:
        if qHandle:del qHandle
        raise
def revisionsInBranch(schema,branchid):
    '''
    returns all revision values in a branch
    result=[revision_id]
    select distinct branch_id from revisions where branch_id>:branchid;
    select revision_id from revisions where branch_id=:branchid ;
    if the branchid matches and the revisionid is not in the branchid collection,not 0, then this revision is in the branch
    require also revisionid>branchid
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        nextbranches=[]
        qHandle.addToTableList( nameDealer.revisionTableName() )
        qHandle.addToOutputList('distinct BRANCH_ID','branch_id')
        qCondition=coral.AttributeList()
        qCondition.extend('branchid','unsigned long long')
        qCondition['branchid'].setData(branchid)
        qResult=coral.AttributeList()
        qResult.extend('branch_id','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('BRANCH_ID>:branchid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            nextbranches.append(cursor.currentRow()['branch_id'].data())
        del qHandle
        candidates=[]
        conditionStr='BRANCH_ID=:branchid and REVISION_ID!=0'
        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.revisionTableName() )
        qHandle.addToOutputList('REVISION_ID','revision_id')
        qCondition=coral.AttributeList()
        qCondition.extend('branchid','unsigned long long')
        qCondition['branchid'].setData(branchid)
        qResult=coral.AttributeList()
        qResult.extend('revision_id','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(conditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            candidates.append(cursor.currentRow()['revision_id'].data())
        del qHandle
        for c in candidates:
            if c in nextbranches:
                continue
            result.append(c)
        return result
    except:
        if qHandle: del qHandle
        raise

def branchType(schema,name):
    '''
    output: tag,branch
    the difference between tag and branch: tag is an empty branch
    select count(revision_id) from revisions where branch_name=:name
    if >0: is real branch
    else: is tag
    '''
    result='tag'
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.revisionTableName() )
        qHandle.addToOutputList('count(REVISION_ID)','nchildren')
        qCondition=coral.AttributeList()
        qCondition.extend('branch_name','string')
        qCondition['branch_name'].setData(name)
        qResult=coral.AttributeList()
        qResult.extend('nchildren','unsigned int')
        qHandle.defineOutput(qResult)
        conditionStr='BRANCH_NAME=:branch_name'
        qHandle.setCondition(conditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            if cursor.currentRow()['nchildren'].data()>0:
                result='branch'                
        del qHandle
        return result
    except :
        raise 
#def revisionsInBranch(schema,branchid):
#    '''
#    returns all revision values in a branch/tag
#    result=[revision_id]
#    select r.revision_id from revisions r where r.branch_id=:branchid and r.revision_id not in (select distinct a.branch_id from revisions a where a.branch_id>:branchid)
#    '''
#    result=[]
#    try:
#        qHandle=schema.newQuery()
#        subquery=qHandle.defineSubQuery('B')
#        subquery.addToTableList( nameDealer.revisionTableName(),'a' )
#        subquery.addToOutputList('distinct a.BRANCH_ID')
#        subqueryCondition=coral.AttributeList()
#        subqueryCondition.extend('branchid','unsigned long long')
#        subqueryCondition['branchid'].setData(branchid)
#        subquery.setCondition('a.BRANCH_ID>:branchid',subqueryCondition)
#        
#        qHandle.addToTableList( nameDealer.revisionTableName(),'r' )
#        qHandle.addToTableList( 'B')
#        qHandle.addToOutputList('r.REVISION_ID','revision_id')
#        qCondition=coral.AttributeList()
#        qCondition.extend('branchid','unsigned long long')
#        qCondition['branchid'].setData(branchid)
#        qResult=coral.AttributeList()
#        qResult.extend('revision_id','unsigned long long')
#        qHandle.defineOutput(qResult)
#        conditionStr='r.BRANCH_ID=:branchid AND r.REVISION_ID NOT IN B'
#        qHandle.setCondition(conditionStr,qCondition)
#        cursor=qHandle.execute()
#        while cursor.next():
#            result.append(cursor.currentRow()['revision_id'].data())
#        del qHandle
#        return result
#    except :
#        raise 
    
def revisionsInBranchName(schema,branchname):
    '''
    returns all revisions in a branch/tag by name
    '''
    result=[]
    try:
        (revision_id,branch_id)=branchInfoByName(schema,branchname)
        result=revisionsInBranch(schema,revision_id)
        return result
    except :
        raise 
def entryInBranch(schema,datatableName,entryname,branch):
    '''
    whether an entry(by name) already exists in the given branch
    select e.entry_id from entrytable e,revisiontable r where r.revision_id=e.revision_id and e.name=:entryname and r.branch_name=branchname/branch_id
    input:
        if isinstance(branch,str):byname
        else: byid
    output:entry_id/None
    '''
    try:
        result=None
        byname=False
        if isinstance(branch,str):
            byname=True
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
            result=entry_id
        del qHandle
        return result
    except :
        raise 

def dataRevisionsOfEntry(schema,datatableName,entry,revrange):
    '''
    all data version of the given entry whose revision falls in branch revision range
    select d.data_id,r.revision_id from datatable d, datarevmaptable r where d.entry_id(or name )=:entry and d.data_id=r.data_id
    input: if isinstance(entry,str): d.entry_name=:entry ; else d.entry_id=:entry
    output: [data_id]
    '''
    qHandle=schema.newQuery()
    try:
        result=[]
        byname=False
        if isinstance(entry,str):
            byname=True
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
            if revision_id in revrange:
                result.append(data_id)
        return result
    except :
        del qHandle
        raise

def latestDataRevisionOfEntry(schema,datatableName,entry,revrange):
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
         revision_id=None
         branch_id=None
         while cursor.next():
             revision_id=cursor.currentRow()['revision_id'].data()
             branch_id=cursor.currentRow()['branch_id'].data()
         del qHandle
         return (revision_id,branch_id)
    except Exception,e :
        raise RuntimeError(' revisionDML.branchInfoByName: '+str(e))
    

#=======================================================
#
#   INSERT requires in update transaction
#
#=======================================================
def bookNewEntry(schema,datatableName):
    '''
    allocate new revision_id,entry_id,data_id
    '''
    try:
        entrytableName=nameDealer.entryTableName(datatableName)
        iddealer=idDealer.idDealer(schema)
        revision_id=iddealer.generateNextIDForTable( nameDealer.revisionTableName() )
        data_id=iddealer.generateNextIDForTable( datatableName)
        entry_id=iddealer.generateNextIDForTable( nameDealer.entryTableName(datatableName) )
        return (revision_id,entry_id,data_id)
    except:
        raise
    
def bookNewRevision(schema,datatableName):
    '''
    allocate new revision_id,data_id
    '''
    try:
        iddealer=idDealer.idDealer(schema)
        revision_id=iddealer.generateNextIDForTable( nameDealer.revisionTableName() )
        data_id=iddealer.generateNextIDForTable(datatableName)
        return (revision_id,data_id)
    except:
        raise
     
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
    try:
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
    except:
        raise
    
def addRevision(schema,datatableName,revisioninfo,branchinfo):
    '''
    1.insert a new revision into the revisions table
    2.insert into data_id, revision_id pair to  datatable_revmap
    insert into revisions(revision_id,branch_id,branch_name,ctime) values()
    insert into datatable_rev(data_id,revision_id) values())
    input:
         revisioninfo (revision_id(0),data_id(1))
         branchinfo  (branch_id(0),branch_name(1))
    '''
    try:
        revisiontableName=nameDealer.revisionTableName()
        revtableName=nameDealer.revmapTableName(datatableName)
        
        db=dbUtil.dbUtil(schema)
        tabrowDefDict={}
        tabrowDefDict['REVISION_ID']='unsigned long long'
        tabrowDefDict['BRANCH_ID']='unsigned long long'
        tabrowDefDict['BRANCH_NAME']='string'
        tabrowDefDict['CTIME']='time stamp'

        tabrowValueDict={}
        tabrowValueDict['REVISION_ID']=revisioninfo[0]
        tabrowValueDict['BRANCH_ID']=branchinfo[0]
        tabrowValueDict['BRANCH_NAME']=branchinfo[1]
        tabrowValueDict['CTIME']=coral.TimeStamp()
        
        db.insertOneRow(revisiontableName,tabrowDefDict,tabrowValueDict)
        
        tabrowDefDict={}
        tabrowDefDict['REVISION_ID']='unsigned long long'
        tabrowDefDict['DATA_ID']='unsigned long long'
        tabrowValueDict={}
        tabrowValueDict['REVISION_ID']=revisioninfo[0]
        tabrowValueDict['DATA_ID']=revisioninfo[1]
        db.insertOneRow(revtableName,tabrowDefDict,tabrowValueDict)
    except:
        raise    
def createBranch(schema,name,parentname,comment=''):
    '''
    create a new branch/tag under given parentnode
    insert into revisions(revision_id,branch_id,branch_name,name,comment,ctime) values()
    return (revisionid,parentid,parentname)
    '''
    try:
        parentid=None
        revisionid=0       
        if not parentname is None:
            qHandle=schema.newQuery()
            qHandle.addToTableList( nameDealer.revisionTableName() )
            qHandle.addToOutputList( 'REVISION_ID','revision_id' )
            qCondition=coral.AttributeList()
            qCondition.extend('parentname','string')
            qCondition['parentname'].setData(parentname)
            qResult=coral.AttributeList()
            qResult.extend('revision_id','unsigned long long')
            qHandle.defineOutput(qResult)
            qHandle.setCondition('NAME=:parentname',qCondition)
            cursor=qHandle.execute()
            while cursor.next():
                parentid=cursor.currentRow()['revision_id'].data()
            del qHandle
        else:
            parentname='ROOT'
        iddealer=idDealer.idDealer(schema)
        revisionid=iddealer.generateNextIDForTable( nameDealer.revisionTableName() )
        db=dbUtil.dbUtil(schema)
        tabrowDefDict={}
        tabrowDefDict['REVISION_ID']='unsigned long long'
        tabrowDefDict['BRANCH_ID']='unsigned long long'
        tabrowDefDict['BRANCH_NAME']='string'
        tabrowDefDict['NAME']='string'
        tabrowDefDict['COMMENT']='string'
        tabrowDefDict['CTIME']='time stamp'
        tabrowValueDict={}
        tabrowValueDict['REVISION_ID']=revisionid
        tabrowValueDict['BRANCH_ID']=parentid
        tabrowValueDict['BRANCH_NAME']=parentname
        tabrowValueDict['NAME']=name
        tabrowValueDict['COMMENT']=comment
        tabrowValueDict['CTIME']=coral.TimeStamp()
        db.insertOneRow(nameDealer.revisionTableName(),tabrowDefDict, tabrowValueDict )
        return (revisionid,parentid,parentname)
    except:
        raise
    
################################################################
# Data Tagging  API
################################################################
def createDataTag(schema,tagname,lumitype='HF'):
    '''
    insert into tags(tagname,tagid,creationtime) values()
    output:
        tagname,tagid,creationtime
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    if lumitype=='HF':
        tagstablename=nameDealer.tagsTableName()
    else:
        tagstablename=nameDealer.pixeltagsTableName()
    try:
        iddealer=idDealer.idDealer(schema)
        tagid=iddealer.generateNextIDForTable( tagstablename )
        db=dbUtil.dbUtil(schema)
        tabrowDefDict={}
        tabrowDefDict['TAGNAME']='string'
        tabrowDefDict['TAGID']='unsigned long long'
        tabrowDefDict['CREATIONTIME']='time stamp'
        tabrowValueDict={}
        tabrowValueDict['TAGNAME']=tagname
        tabrowValueDict['TAGID']=tagid
        creationtime=coral.TimeStamp()
        tabrowValueDict['CREATIONTIME']=creationtime
        db.insertOneRow(tagstablename,tabrowDefDict, tabrowValueDict )
        return (tagname,tagid,creationtime)
    except:
        raise

def currentDataTag(schema,lumitype='HF'):
    '''
    select tagid,tagname from tags
    output:(tagid,tagname)
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    if lumitype=='HF':
        tagstablename=nameDealer.tagsTableName()
    else:
        tagstablename=nameDealer.pixeltagsTableName()
    tagmap={}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList( tagstablename )
        qHandle.addToOutputList('TAGID')
        qHandle.addToOutputList('TAGNAME')
        qResult=coral.AttributeList()
        qResult.extend('TAGID','unsigned long long')
        qResult.extend('TAGNAME','string')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        currenttagid=0
        while cursor.next():
            tagid=cursor.currentRow()['TAGID'].data()
            tagname=cursor.currentRow()['TAGNAME'].data()
            tagmap[tagid]=tagname
        del qHandle
        if len(tagmap)!=0:
            currenttagid=max(tagmap.keys())
        if currenttagid==0:
            raise 'currentDataTag: no tag available'
        return (currenttagid,tagmap[currenttagid])
    except:
        raise
        
def addRunToCurrentDataTag(schema,runnum,lumiid,trgid,hltid,lumitype='HF',comment=''):
    '''
    select tagid from tags
    insert into tagruns(tagid,runnum,lumidataid,trgdataid,hltdataid,creationtime,comment) values(tagid,runnum,lumiid,trgid,hltid,creationtime,comment)
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    if lumitype=='HF':
        tagrunstablename=nameDealer.tagrunsTableName()
    else:
        tagrunstablename=nameDealer.pixeltagRunsTableName()
    currenttagid=currentDataTag(schema,lumitype=lumitype)[0]
    try:
        db=dbUtil.dbUtil(schema)
        tabrowDefDict={}
        tabrowDefDict['TAGID']='unsigned long long'
        tabrowDefDict['RUNNUM']='unsigned int'
        tabrowDefDict['LUMIDATAID']='unsigned long long'
        tabrowDefDict['TRGDATAID']='unsigned long long'
        tabrowDefDict['HLTDATAID']='unsigned long long'
        tabrowDefDict['CREATIONTIME']='time stamp'
        tabrowDefDict['COMMENT']='string'
        tabrowValueDict={}
        tabrowValueDict['TAGID']=currenttagid
        tabrowValueDict['RUNNUM']=runnum
        tabrowValueDict['LUMIDATAID']=lumiid
        tabrowValueDict['TRGDATAID']=trgid
        tabrowValueDict['HLTDATAID']=hltid
        tabrowValueDict['CREATIONTIME']=coral.TimeStamp()
        tabrowValueDict['COMMENT']=comment
        db.insertOneRow( tagrunstablename,tabrowDefDict, tabrowValueDict )
    except:
        raise

def alldataTags(schema,lumitype='HF'):
    '''
    select tagname,tagid from tags,tagruns  
        if number of tags==1->open tag
        if tagid is max ->open tag
        for closed tag:
           max run=max(runnum) where tagid=:tagid
           min run
              select min(runnum) from tagruns where tagid<=:tagid
        for open tag:
           max run=None
           min run
              select min(runnum) from tagruns where tagid<=:tagid
    output:
       {tagid:(name,minrun,maxrun,creationtime)}
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    if lumitype=='HF':
        tagstablename=nameDealer.tagsTableName()
        tagrunstablename=nameDealer.tagRunsTableName()
    else:
        tagstablename=nameDealer.pixeltagsTableName()
        tagrunstablename=nameDealer.pixeltagRunsTableName()
    tagmap={}#{tagid:[tagname,minrun,maxrun,creationtime]}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList( tagstablename )
        qCondition=coral.AttributeList()
        qHandle.addToOutputList('TAGNAME')
        qHandle.addToOutputList('TAGID')
        qHandle.addToOutputList("TO_CHAR(CREATIONTIME,\'MM/DD/YY HH24:MI:SS\')",'creationtime')
        qResult=coral.AttributeList()        
        qResult.extend('TAGNAME','string')
        qResult.extend('TAGID','unsigned long long')
        qResult.extend('creationtime','string')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        while cursor.next():
            tagname=cursor.currentRow()['TAGNAME'].data()
            tagid=cursor.currentRow()['TAGID'].data()
            creationtime=cursor.currentRow()['creationtime'].data()
            tagmap[tagid]=[tagname,0,0,creationtime]
        del qHandle
        
        tagids=tagmap.keys()
        allruns=set()
        for tagid in tagids:
            qConditionStr='TAGID<=:tagid'
            qCondition=coral.AttributeList()
            qCondition.extend('tagid','unsigned long long')
            qCondition['tagid'].setData(tagid)
            qHandle=schema.newQuery()
            qHandle.addToTableList( tagrunstablename )
            qResult=coral.AttributeList()
            qResult.extend('RUNNUM','unsigned int')
            qHandle.defineOutput(qResult)
            qHandle.setCondition(qConditionStr,qCondition)
            qHandle.addToOutputList('RUNNUM')
            cursor=qHandle.execute()
            while cursor.next():
                rnum=cursor.currentRow()['RUNNUM'].data()
                allruns.add(rnum)
            minrun=0
            maxrun=0
            if len(allruns)!=0:
                minrun=min(allruns)
                maxrun=max(allruns)
            tagmap[tagid][1]=minrun
            if len(tagmap)>1 and tagid!=max(tagids):
                tagmap[tagid][2]=maxrun   
    except:
        raise
    return tagmap

def dataIdsByTagName(schema,tagname,runlist=None,withcomment=False,lumitype='HF'):
    '''
    select tagid from tags where tagname=:tagname
    input:
        runlist: select run list, if None, all
    output:
        {run:(lumidataid,trgdataid,hltdataid,(creationtime,comment)}
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    if lumitype=='HF':
        tagstablename=nameDealer.tagsTableName()
    else:
        tagstablename=nameDealer.pixeltagsTableName()        
    tagid=None
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList( tagstablename )
        qConditionStr='TAGNAME=:tagname'
        qCondition=coral.AttributeList()
        qCondition.extend('tagname','string')
        qCondition['tagname'].setData(tagname)
        qHandle.addToOutputList('TAGID')
        qResult=coral.AttributeList()        
        qResult.extend('TAGID','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            if not cursor.currentRow()['TAGID'].isNull():
                tagid=cursor.currentRow()['TAGID'].data()
        del qHandle
    except:
        raise
    if tagid is None:
        return {}
    return dataIdsByTagId(schema,tagid,runlist=runlist,withcomment=withcomment,lumitype=lumitype)


def dataTagInfo(schema,tagname,runlist=None,lumitype='HF'):
    '''
    select tagid from tags where tagname=:tagname
    select runnum,comment from tagruns where tagid<=:tagid
    input:
        runlist: select run list, if None, all
    output:
       {tagid:(name,minrun,maxrun,creationtime)}
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    if lumitype=='HF':
        tagstablename=nameDealer.tagsTableName()
        tagrunstablename=nameDealer.tagRunsTableName()
    else:
        tagstablename=nameDealer.pixeltagsTableName()
        tagrunstablename=nameDealer.pixeltagRunsTableName()
    tagmap={}#{tagid:[tagname,minrun,maxrun,creationtime]}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList( tagstablename )
        qCondition=coral.AttributeList()
        qHandle.addToOutputList('TAGNAME')
        qHandle.addToOutputList('TAGID')
        qHandle.addToOutputList("TO_CHAR(CREATIONTIME,\'MM/DD/YY HH24:MI:SS\')",'creationtime')
        qResult=coral.AttributeList()        
        qResult.extend('TAGNAME','string')
        qResult.extend('TAGID','unsigned long long')
        qResult.extend('creationtime','string')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        while cursor.next():
            tagname=cursor.currentRow()['TAGNAME'].data()
            tagid=cursor.currentRow()['TAGID'].data()
            creationtime=cursor.currentRow()['creationtime'].data()
            tagmap[tagid]=[tagname,0,0,creationtime]
        del qHandle
        
        tagids=tagmap.keys()
        allruns=set()
        for tagid in tagids:
            qConditionStr='TAGID<=:tagid'
            qCondition=coral.AttributeList()
            qCondition.extend('tagid','unsigned long long')
            qCondition['tagid'].setData(tagid)
            qHandle=schema.newQuery()
            qHandle.addToTableList(tagrunstablename)
            qResult=coral.AttributeList()
            qResult.extend('RUNNUM','unsigned int')
            qHandle.defineOutput(qResult)
            qHandle.setCondition(qConditionStr,qCondition)
            qHandle.addToOutputList('RUNNUM')
            cursor=qHandle.execute()
            while cursor.next():
                rnum=cursor.currentRow()['RUNNUM'].data()
                if runlist is not None and rnum not in runlist:
                    continue
                allruns.add(rnum)
            minrun=0
            maxrun=0
            if len(allruns)!=0:
                minrun=min(allruns)
                maxrun=max(allruns)
            tagmap[tagid][1]=minrun
            if len(tagmap)>1 and tagid!=max(tagids):
                tagmap[tagid][2]=maxrun   
    except:
        raise
    return tagmap

def dataIdsByTagId(schema,tagid,runlist=None,withcomment=False,lumitype='HF'):
    '''
    select runnum,lumidataid,trgdataid,hltdataid,comment from tagruns where TAGID<=:tagid;
    input:
        runlist: select run list, if None, all
    output:
        {run:(lumidataid,trgdataid,hltdataid,(creationtime,comment))}
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    if lumitype=='HF':
        tagrunstablename=nameDealer.tagRunsTableName()
    else:
        tagrunstablename=nameDealer.pixeltagRunsTableName()
    result={}#{run:[lumiid,trgid,hltid,comment(optional)]} 
    commentdict={}#{(lumiid,trgid,hltid):[ctimestr,comment]}
    try:
        qHandle=schema.newQuery()
        qHandle.addToTableList(tagrunstablename)
        qConditionStr='TAGID<=:tagid'
        qCondition=coral.AttributeList()
        qCondition.extend('tagid','unsigned long long')
        qCondition['tagid'].setData(tagid)
        qResult=coral.AttributeList()        
        qResult.extend('RUNNUM','unsigned int')
        qResult.extend('LUMIDATAID','unsigned long long')
        qResult.extend('TRGDATAID','unsigned long long')
        qResult.extend('HLTDATAID','unsigned long long')
        if withcomment:
            qResult.extend('COMMENT','string')
            qResult.extend('creationtime','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        qHandle.addToOutputList('RUNNUM')
        qHandle.addToOutputList('LUMIDATAID')
        qHandle.addToOutputList('TRGDATAID')
        qHandle.addToOutputList('HLTDATAID')
        if withcomment:
            qHandle.addToOutputList('COMMENT')
            qHandle.addToOutputList("TO_CHAR(CREATIONTIME,\'MM/DD/YY HH24:MI:SS\')",'creationtime')
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['RUNNUM'].data()
            if runlist is not None and runnum not in runlist:
                continue
            lumidataid=0
            if not cursor.currentRow()['LUMIDATAID'].isNull():
                lumidataid=cursor.currentRow()['LUMIDATAID'].data()
            trgdataid=0
            if not cursor.currentRow()['TRGDATAID'].isNull():
                trgdataid=cursor.currentRow()['TRGDATAID'].data()
            hltdataid=0
            if not cursor.currentRow()['HLTDATAID'].isNull():
                hltdataid=cursor.currentRow()['HLTDATAID'].data()
            if not result.has_key(runnum):
                result[runnum]=[0,0,0]
            if lumidataid>result[runnum][0]:
                result[runnum][0]=lumidataid
            if trgdataid>result[runnum][1]:
                result[runnum][1]=trgdataid
            if hltdataid>result[runnum][2]:
                result[runnum][2]=hltdataid    
            if withcomment:
                comment=''
                creationtime=''
                if not cursor.currentRow()['creationtime'].isNull():
                    creationtime=cursor.currentRow()['creationtime'].data()
                if not cursor.currentRow()['COMMENT'].isNull():
                    comment=cursor.currentRow()['COMMENT'].data()
                commentdict[(lumidataid,trgdataid,hltdataid)]=(creationtime,comment)
        del qHandle
        if withcomment:
            for run,resultentry in result.items():
                lumiid=resultentry[0]
                trgid=resultentry[1]
                hltid=resultentry[2]
                if commentdict.has_key((lumiid,trgid,hltid)):
                    resultentry.append(commentdict[(lumiid,trgid,hltid)])
                elif commentdict.has_key((lumiid,0,0)):
                    resultentry.append(commentdict[(lumiid,0,0)])
                elif commentdict.has_ley((0,trgid,0)):
                    resultentry.append(commentdict[(0,trgid,0)])
                elif commentdict.has_ley((0,0,hltid)):
                    resultentry.append(commentdict[(0,0,hltid)])
                else:
                    resultentry.append(())
                    
    except:
        raise
    return result
    
def dataIdsByCurrentTag(schema,runlist=None,lumitype='HF'):
    '''
    dataIdsByTagId(schema,currenttagid,runlist)
    output:
       (currenttagname,{run:(lumidataid,trgdataid,hltdataid)})
    '''
    (currenttagid,currenttagname)=currentDataTag(schema)
    result=dataIdsByTagId(schema,currenttagid,runlist=runlist,withcomment=False,lumitype=lumitype)
    return (currenttagname,result)

if __name__ == "__main__":
    import sessionManager
    import lumidbDDL
    #myconstr='oracle://cms_orcoff_prep/cms_lumi_dev_offline'
    #authpath='/afs/cern.ch/user/x/xiezhen'
    myconstr='sqlite_file:test.db'
    svc=sessionManager.sessionManager(myconstr,debugON=False)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    schema=session.nominalSchema()
    session.transaction().start(False)
    tables=lumidbDDL.createTables(schema)
    trunkinfo=createBranch(schema,'TRUNK',None,comment='main')
    #print trunkinfo
    datainfo=createBranch(schema,'DATA','TRUNK',comment='hold data')
    #print datainfo
    norminfo=createBranch(schema,'NORM','TRUNK',comment='hold normalization factor')
    #print norminfo
    (branchid,branchparent)=branchInfoByName(schema,'DATA')
    databranchinfo=(branchid,'DATA')
    print databranchinfo
    for runnum in [1200,1211,1222,1233,1345,1222,1200]:
        lumientryid=entryInBranch(schema,nameDealer.lumidataTableName(),str(runnum),'DATA')
        trgentryid=entryInBranch(schema,nameDealer.trgdataTableName(),str(runnum),'DATA')
        hltentryid=entryInBranch(schema,nameDealer.hltdataTableName(),str(runnum),'DATA')
        if lumientryid is None:
            (revision_id,entry_id,data_id)=bookNewEntry( schema,nameDealer.lumidataTableName() )
            entryinfo=(revision_id,entry_id,str(runnum),data_id)
            addEntry(schema,nameDealer.lumidataTableName(),entryinfo,databranchinfo)
            #add data here
        else:
            revisioninfo=bookNewRevision( schema,nameDealer.lumidataTableName() )
            addRevision(schema,nameDealer.lumidataTableName(),revisioninfo,databranchinfo)
            #add data here
        if trgentryid is None:
            (revision_id,entry_id,data_id)=bookNewEntry( schema,nameDealer.trgdataTableName() )
            entryinfo=(revision_id,entry_id,str(runnum),data_id)
            addEntry(schema,nameDealer.trgdataTableName(),entryinfo,databranchinfo)
            #add data here
        else:
            revisioninfo=bookNewRevision( schema,nameDealer.trgdataTableName() )
            addRevision(schema,nameDealer.trgdataTableName(),revisioninfo,databranchinfo)      
             #add data here
        if hltentryid is None:
            (revision_id,entry_id,data_id)=bookNewEntry( schema,nameDealer.hltdataTableName() )
            entryinfo=(revision_id,entry_id,str(runnum),data_id)
            addEntry(schema,nameDealer.hltdataTableName(),entryinfo,databranchinfo)
            #add data here
        else:
            revisioninfo=bookNewRevision( schema,nameDealer.hltdataTableName() )
            addRevision(schema,nameDealer.hltdataTableName(),revisioninfo,databranchinfo)
            #add data here
        
    session.transaction().commit()
    print 'test reading'
    session.transaction().start(True)
    print branchType(schema,'DATA')
    revlist=revisionsInBranchName(schema,'DATA')
    print 'DATA revlist ',revlist
    lumientry_id=entryInBranch(schema,nameDealer.lumidataTableName(),'1211','DATA')
    print lumientry_id
    latestrevision=latestDataRevisionOfEntry(schema,nameDealer.lumidataTableName(),lumientry_id,revlist)
    print 'latest data_id for run 1211 ',latestrevision
    session.transaction().commit()
    del session
