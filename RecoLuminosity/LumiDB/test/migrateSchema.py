#!/usr/bin/env python
VERSION='1.02'
import os,sys,array
import coral
from RecoLuminosity.LumiDB import argparse,idDealer,nameDealer,CommonUtil,lumidbDDL,dbUtil
#
# data transfer section
#
def getOldTrgData(schema,runnum):
    '''
    generate new data_id for trgdata
    select cmslsnum,deadtime,bitname,trgcount,prescale from trg where runnum=:runnum and bitnum=0 order by cmslsnum;
    select cmslsnum,bitnum,trgcount,deadtime,prescale from trg where runnum=:runnum order by cmslsnum
    output [bitnames,databuffer]
    '''
    bitnames=''
    databuffer={} #{cmslsnum:[deadtime,bitzeroname,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.trgTableName())
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('DEADTIME','deadtime')
        qHandle.addToOutputList('BITNAME','bitname')
        qHandle.addToOutputList('TRGCOUNT','trgcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition.extend('bitnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qCondition['bitnum'].setData(int(0))
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('deadtime','unsigned long long')
        qResult.extend('bitname','string')
        qResult.extend('trgcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum AND BITNUM=:bitnum',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            deadtime=cursor.currentRow()['deadtime'].data()
            bitname=cursor.currentRow()['bitname'].data()
            bitcount=cursor.currentRow()['trgcount'].data()
            prescale=cursor.currentRow()['prescale'].data()
            if not databuffer.has_key(cmslsnum):
                databuffer[cmslsnum]=[]
            databuffer[cmslsnum].append(deadtime)
            databuffer[cmslsnum].append(bitname)
            databuffer[cmslsnum].append(bitcount)
            databuffer[cmslsnum].append(prescale)
        del qHandle
        qHandle=dbsession.nominalSchema().newQuery()
        qHandle.addToTableList(n.trgtable)
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('BITNUM','bitnum')
        qHandle.addToOutputList('BITNAME','bitname')
        qHandle.addToOutputList('TRGCOUNT','trgcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qHandle.addToOrderList('cmslsnum')
        qHandle.addToOrderList('bitnum')
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('bitnum','unsigned int')
        qResult.extend('bitname','string')
        qResult.extend('trgcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        bitnameList=[]
        trgcountArray=array.array('l')
        prescaleArray=array.array('l')
        counter=0
        previouscmslsnum=0
        cmslsnum=-1
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            bitnum=cursor.currentRow()['bitnum'].data()
            bitname=cursor.currentRow()['bitname'].data()
            trgcount=cursor.currentRow()['trgcount'].data()        
            prescale=cursor.currentRow()['prescale'].data()
            
            if bitnum==0 and counter!=0:
                trgcountBlob=CommonUtil.packArraytoBlob(trgcountArray)
                prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)
                databuffer[previouslsnum].append(trgcountBlob)
                databuffer[previouslsnum].append(prescaleBlob)
                bitnameList=[]
                trgcountArray=array.array('l')
                prescaleArray=array.array('l')
            else:
                previouslsnum=cmslsnum
            bitnameList.append(bitname)
            trgcountArray.append(trgcount)
            prescaleArray.append(prescale)
            counter+=1
        if cmslsnum>0:
            bitnames=','.join(bitnameList)
            trgcountBlob=CommonUtil.packArraytoBlob(trgcountArray)
            prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)

            databuffer[cmslsnum].append(trgcountBlob)
            databuffer[cmslsnum].append(prescaleBlob)
        del qHandle
        return [bitnames,databuffer]
    except:
        del qHandle
        raise 

def getOldHLTData(schema,runnum):
    '''
    select count(distinct pathname) from hlt where runnum=:runnum
    select cmslsnum,pathname,inputcount,acceptcount,prescale from hlt where runnum=:runnum order by cmslsnum,pathname
    [pathnames,databuffer]
    '''
    
    databuffer={} #{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}
    pathnames=''
    try:
        npath=0
        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.hltTableName() )
        qHandle.addToOutputList('COUNT(DISTINCT PATHNAME)','npath')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qResult=coral.AttributeList()
        qResult.extend('npath','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            npath=cursor.currentRow()['npath'].data()
        del qHandle
        #print 'npath ',npath

        qHandle=schema.newQuery()
        qHandle.addToTableList( nameDealer.hltTableName() )
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('PATHNAME','pathname')
        qHandle.addToOutputList('INPUTCOUNT','inputcount')
        qHandle.addToOutputList('ACCEPTCOUNT','acceptcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('pathname','string')
        qResult.extend('inputcount','unsigned int')
        qResult.extend('acceptcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qHandle.addToOrderList('cmslsnum')
        qHandle.addToOrderList('pathname')
        cursor=qHandle.execute()
        pathnameList=[]
        inputcountArray=array.array('l')
        acceptcountArray=array.array('l')
        prescaleArray=array.array('l')
        ipath=0
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            pathname=cursor.currentRow()['pathname'].data()
            ipath+=1
            inputcount=cursor.currentRow()['inputcount'].data()
            acceptcount=cursor.currentRow()['acceptcount'].data()
            prescale=cursor.currentRow()['prescale'].data()
            pathnameList.append(pathname)
            inputcountArray.append(inputcount)
            acceptcountArray.append(acceptcount)
            prescaleArray.append(prescale)
            if ipath==npath:
                pathnames=','.join(pathnameList)
                inputcountBlob=CommonUtil.packArraytoBlob(inputcountArray)
                acceptcountBlob=CommonUtil.packArraytoBlob(acceptcountArray)
                prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)
                databuffer[cmslsnum]=[inputcountBlob,acceptcountBlob,prescaleBlob]
                pathnameList=[]
                inputcountArray=array.array('l')
                acceptcountArray=array.array('l')
                prescaleArray=array.array('l')
                ipath=0
        del qHandle
        return [pathnames,databuffer]
    except :
        del qHandle        
        raise 

def main():
    from RecoLuminosity.LumiDB import sessionManager
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="migrate lumidb schema",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',action='store',required=False,default='oracle://cms_orcoff_prep/CMS_LUMI_DEV_OFFLINE',help='connect string to trigger DB(required)')
    parser.add_argument('-P',dest='authpath',action='store',required=False,default='/afs/cern.ch/user/x/xiezhen',help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',required=True,help='run number')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=int(args.runnumber)
    #print 'processing run ',runnumber
    svc=sessionManager.sessionManager(args.connect,authpath=args.authpath,debugON=args.debug)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    session.transaction().start(False)
    schema=session.nominalSchema()
    #lumidbDDL.newToOld(schema)
    lumidbDDL.oldToNew(schema)
    session.transaction().commit()
    del session
    del svc
if __name__=='__main__':
    main()
    
