'''
compare lumibyls result file of a run with corresponding result in HFLUMIRESULT table , return runnumber if considered different, criteria
1)different number of lumils or cmsls
2)difference in beamstatus
3)difference in beamenergy
4)difference in delivered
5)difference in recorded
'''
import csv,coral
from RecoLuminosity.LumiDB import sessionManager
def parselumibylsfile(runnum,lumifilename):
    '''
    parse lumibyls file
          skip firstline,skip lines start with #
    output:
    [lumils,cmsls,beamstatus,beamenergy,delivered,recorded]
    '''
    result=[]
    f=None
    try:
        f=open(lumifilename,'rb')
    except IOError:
        print 'failed to open file ',lumifilename
        return result
    freader=csv.reader(f,delimiter=',')
    idx=0
    for row in freader:
        if idx==0:
            idx=1
            continue
        if row[0].find('#')==1:
            continue
        [lumils,cmsls]=map(lambda i:int(i),row[1].split(':'))
        beamstatus=row[3]
        beamenergy=float(row[4])
        delivered=float(row[5])
        recorded=float(row[6])
        result.append([lumils,cmsls,beamstatus,beamenergy,delivered,recorded])
    return result
def getresultfromdb(schema,runnum):
    '''
    select LS,CMSLS,BEAM_STATUS,ENERGY,DELIVERED,RECORDED from HFLUMIRESULT where RUNNUM=:runnum
    output:{lumils:[cmsls,beamstatus,beamenergy,delivered,recorded]}    
    '''
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList( 'HFLUMIRESULT' )
        qHandle.addToOutputList('LS')
        qHandle.addToOutputList('CMSLS')
        qHandle.addToOutputList('BEAM_STATUS')
        qHandle.addToOutputList('ENERGY')
        qHandle.addToOutputList('DELIVERED')
        qHandle.addToOutputList('RECORDED')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qResult=coral.AttributeList()
        qResult.extend('LS','unsigned int')
        qResult.extend('CMSLS','unsigned int')
        qResult.extend('BEAM_STATUS','string')
        qResult.extend('ENERGY','unsigned int')
        qResult.extend('DELIVERED','float')
        qResult.extend('RECORDED','float')
        qHandle.defineOutput(qResult)
        qCondition['runnum'].setData(runnum)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            lumils=cursor.currentRow()['LS'].data()
            cmsls=cursor.currentRow()['CMSLS'].data()
            beamstatus=cursor.currentRow()['BEAM_STATUS'].data()
            beamenergy=cursor.currentRow()['ENERGY'].data()
            beamenergy=int(round(beamenergy))
            delivered=cursor.currentRow()['DELIVERED'].data()
            recorded=cursor.currentRow()['RECORDED'].data()
            result[lumils]=[cmsls,beamstatus,beamenergy,delivered,recorded]
        del qHandle
    except:
        if qHandle:del qHandle
        raise
    return result

def isdifferent(resultlist,resultmap):
    '''
    input:
    resultlist [lumils,cmsls,beamstatus,beamenergy,delivered,recorded]
    resultmap {lumils:[cmsls,beamstatus,beamenergy,delivered,recorded]}  
    '''
    result=False
    if len(resultlist)!=len(resultmap): return False
    for perlumidata in resultlist:
        [lumilsa,cmslsa,beamstatusa,beamenergya,delivereda,recordeda]=perlumidata
        if not resultmap.has_key(lumilsa): return False
        [cmslsb,beamstatusb,beamenergyb,deliveredb,recordedb]=resultmap[lumilsa]
        if cmslsa!=cmslsb: return False
        if beamstatusa!=beamstatusb: return False
        if beamenergya!=beamenergyb: return False
        if delivereda!=deliveredb:
            print delivereda,deliveredb
            return False
        if recordeda!=recordedb:
            print recordeda,recordedb
            return False
        return True
if __name__ == "__main__" :
    sourcestr='oracle://cms_orcon_prod/cms_lumi_prod'
    pth='/nfshome0/xiezhen'
    sourcesvc=sessionManager.sessionManager(sourcestr,authpath=pth,debugON=False)
    sourcesession=sourcesvc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    sourcesession.transaction().start(True)
    runnum=209151
    dbresult=getresultfromdb(sourcesession.nominalSchema(),runnum)
    sourcesession.transaction().commit()
    #print dbresult
    lumifilename=str(runnum)+'.csv'
    fileresult=parselumibylsfile(runnum,lumifilename)
    #print fileresult
    isdifferent=isdifferent(fileresult,dbresult)
    print isdifferent
