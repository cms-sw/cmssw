import csv,os,sys,coral,array
from RecoLuminosity.LumiDB import argparse,sessionManager,CommonUtil,idDealer,dbUtil,dataDML,revisionDML
beamenergy=4.0e03
beamstatus='STABLE BEAMS'
numorbit=262144
startorbit=0
constfactor=23.31*1000.
beam1intensity=9124580336.0
beam2intensity=8932813306.0

def convertlist(l):
    '''yield successive pairs for l
    '''
    for i in xrange(0,len(l),2):
        idx=int(l[i])
        val=float(l[i+1])
        yield (idx,val)
        
def fetchOldData(schema,oldlumidataid):
    '''
    fetch old perbunch data if the run exists
    select CMSBXINDEXBLOB,BEAMINTENSITYBLOB_1,BEAMINTENSITYBLOB_2,BXLUMIVALUE_OCC1,BXLUMIVALUE_OCC2,BXLUMIVALUE_ET,BXLUMIERROR_OCC1,BXLUMIERROR_OCC2,BXLUMIERROR_ET,BXLUMIQUALITY_OCC1,BXLUMIQUALITY_OCC2,BXLUMIQUALITY_ET from lumisummaryv2 where data_id=:oldlumidataid;
    output:
        {lumilsnum:[bxvalueblob_occ1(0),bxerrblob_occ1(1),bxqualityblob_occ1(2),bxvalueblob_occ2(3),bxerrblob_occ2(4),bxqualityblob_occ2(5),bxvalueblob_et(6),bxerrblob_et(7),bxqualityblob_et(8),bxindexblob(9),beam1intensity(10),beam2intensity(11),beamstatus(12),beamenergy(13),instlumierror(14),instlumiquality(15)],...}
    '''
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList('LUMISUMMARYV2')
        qHandle.addToOutputList('LUMILSNUM','lumilsnum')
        qHandle.addToOutputList('BEAMSTATUS','beamstatus')
        qHandle.addToOutputList('BEAMENERGY','beamenergy')
        qHandle.addToOutputList('INSTLUMIERROR','instlumierror')
        qHandle.addToOutputList('INSTLUMIQUALITY','instlumiquality')
        qHandle.addToOutputList('BXLUMIVALUE_OCC1','bxvalue_occ1')
        qHandle.addToOutputList('BXLUMIERROR_OCC1','bxerror_occ1')
        qHandle.addToOutputList('BXLUMIQUALITY_OCC1','bxquality_occ1')
        qHandle.addToOutputList('BXLUMIVALUE_OCC2','bxvalue_occ2')
        qHandle.addToOutputList('BXLUMIERROR_OCC2','bxerror_occ2')
        qHandle.addToOutputList('BXLUMIQUALITY_OCC2','bxquality_occ2')
        qHandle.addToOutputList('BXLUMIVALUE_ET','bxvalue_et')
        qHandle.addToOutputList('BXLUMIERROR_ET','bxerror_et')
        qHandle.addToOutputList('BXLUMIQUALITY_ET','bxquality_et')
        qHandle.addToOutputList('CMSBXINDEXBLOB','bxindexblob')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_1','beam1intensity')
        qHandle.addToOutputList('BEAMINTENSITYBLOB_2','beam2intensity')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        print 'oldlumidataid ',oldlumidataid
        qCondition['dataid'].setData(int(oldlumidataid))
        qResult=coral.AttributeList()
        qResult.extend('lumilsnum','unsigned int')
        qResult.extend('beamstatus','string')
        qResult.extend('beamenergy','float')
        qResult.extend('instlumierror','float')
        qResult.extend('instlumiquality','short')
        qResult.extend('bxvalue_occ1','blob')
        qResult.extend('bxerror_occ1','blob')
        qResult.extend('bxquality_occ1','blob')
        qResult.extend('bxvalue_occ2','blob')
        qResult.extend('bxerror_occ2','blob')
        qResult.extend('bxquality_occ2','blob')
        qResult.extend('bxvalue_et','blob')
        qResult.extend('bxerror_et','blob')
        qResult.extend('bxquality_et','blob')
        qResult.extend('bxindexblob','blob')
        qResult.extend('beam1intensity','blob')
        qResult.extend('beam2intensity','blob')        
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            lumilsnum=cursor.currentRow()['lumilsnum'].data()
            beamstatus=cursor.currentRow()['beamstatus'].data()
            beamenergy=cursor.currentRow()['beamenergy'].data()
            instlumierror=cursor.currentRow()['instlumierror'].data()
            instlumiquality=cursor.currentRow()['instlumiquality'].data()
            bxvalueblob_occ1=cursor.currentRow()['bxvalue_occ1'].data()
            bxvalueblob_occ1Array=CommonUtil.unpackBlobtoArray(bxvalueblob_occ1,'f')
            bxerrblob_occ1=cursor.currentRow()['bxerror_occ1'].data()
            bxqualityblob_occ1=cursor.currentRow()['bxquality_occ1'].data()
            bxvalueblob_occ2=cursor.currentRow()['bxvalue_occ2'].data()
            bxerrblob_occ2=cursor.currentRow()['bxerror_occ2'].data()
            bxqualityblob_occ2=cursor.currentRow()['bxquality_occ2'].data()
            bxvalueblob_et=cursor.currentRow()['bxvalue_et'].data()
            bxerrblob_et=cursor.currentRow()['bxerror_et'].data()
            bxqualityblob_et=cursor.currentRow()['bxquality_et'].data()
            bxindexblob=cursor.currentRow()['bxindexblob'].data()
            beam1intensity=cursor.currentRow()['beam1intensity'].data()
            beam2intensity=cursor.currentRow()['beam2intensity'].data()
            result[lumilsnum]=[bxvalueblob_occ1,bxerrblob_occ1,bxqualityblob_occ1,bxvalueblob_occ2,bxerrblob_occ2,bxqualityblob_occ2,bxvalueblob_et,bxerrblob_et,bxqualityblob_et,bxindexblob,beam1intensity,beam2intensity,beamstatus,beamenergy,instlumierror,instlumiquality]
            
    except :
        del qHandle
        raise 
    del qHandle
    return result
            
def insertLumischemaV2(dbsession,runnum,datasource,perlsrawdata,perbunchrawdata,bxdistribution,withDetails=False,deliveredonly=False):
    '''
    input:
    lumirundata[datasource]
    perlsrawdata: {cmslsnum:instlumi}
    perbunchrawdata: {bxidx:lumifraction}

    lumilsdata {lumilsnum:[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beam1intensityblob,beam2intensityblob,bxlumivalue_occ1,bxlumierror_occ1,bxlumiquality_occ1,bxlumivalue_occ2,bxlumierror_occ2,bxlumiquality_occ2,bxlumivalue_et,bxlumierror_et,bxlumiquality_et]}
    '''
    branchrevision_id=3#databranch_id
    branchinfo=(branchrevision_id,'DATA')
    lumirundata=[datasource]
    lumilsdata={}
    for cmslsnum,instlumi in perlsrawdata.items():
        mystartorbit=startorbit+numorbit*(cmslsnum-1)
        bxdataocc1blob=None
        bxdataocc2blob=None
        bxdataetblob=None
        bxerrorocc1blob=None
        bxerrorocc2blob=None
        bxerroretblob=None
        bxqualityocc1blob=None
        bxqualityocc2blob=None
        bxqualityetblob=None
        cmsbxindexblob=None
        beam1intensityblob=None
        beam2intensityblob=None
        beamstatus='STABLE BEAMS'
        beamenergy=4000.
        instlumierror=0.
        instlumiquality=1
        if perbunchrawdata:
            bxdataocc1blob=perbunchrawdata[cmslsnum][0]
            bxerrorocc1blob=perbunchrawdata[cmslsnum][1]
            bxqualityocc1blob=perbunchrawdata[cmslsnum][2]
            bxdataocc2blob=perbunchrawdata[cmslsnum][3]
            bxerrorocc2blob=perbunchrawdata[cmslsnum][4]
            bxqualityocc2blob=perbunchrawdata[cmslsnum][5]
            bxdataetblob=perbunchrawdata[cmslsnum][6]
            bxerroretblob=perbunchrawdata[cmslsnum][7]
            bxqualityetblob=perbunchrawdata[cmslsnum][8]
            bxindexblob=perbunchrawdata[cmslsnum][9]
            beam1intensityblob=perbunchrawdata[cmslsnum][10]
            beam2intensityblob=perbunchrawdata[cmslsnum][11]
            beamstatus=perbunchrawdata[cmslsnum][12]
            beamenergy=perbunchrawdata[cmslsnum][13]
            instlumierror=perbunchrawdata[cmslsnum][14]
            instlumiquality=perbunchrawdata[cmslsnum][15]
        elif bxdistribution:
            bxdataArray=array.array('f')
            bxerrorArray=array.array('f')
            bxqualityArray=array.array('h')
            cmsbxindexArray=array.array('h')
            beam1intensityArray=array.array('f')
            beam2intensityArray=array.array('f')
            for bxidx in range(1,3565):
                lumifraction=0.0
                if perbunchrawdata.has_key(bxidx):
                    lumifraction=perbunchrawdata[bxidx]
                bxlumivalue=float(instlumi*lumifraction)
                bxdataArray.append(bxlumivalue)
                beam1intensityArray.append(9124580336.0)
                beam1intensityArray.append(8932813306.0)
                cmsbxindexArray.append(bxidx)
                bxqualityArray.append(1)
                bxerrorArray.append(0.0)           
            bxdataocc1blob=CommonUtil.packArraytoBlob(bxdataArray)
            bxdataocc2blob=CommonUtil.packArraytoBlob(bxdataArray)
            bxdataetblob=CommonUtil.packArraytoBlob(bxdataArray)
            bxerrorocc1blob=CommonUtil.packArraytoBlob(bxerrorArray)
            bxerrorocc2blob=CommonUtil.packArraytoBlob(bxerrorArray)
            bxerroretblob=CommonUtil.packArraytoBlob(bxerrorArray)
            bxqualityocc1blob=CommonUtil.packArraytoBlob(bxqualityArray)
            bxqualityocc2blob=CommonUtil.packArraytoBlob(bxqualityArray)
            bxqualityetblob=CommonUtil.packArraytoBlob(bxqualityArray)         
            cmsbxindexblob=CommonUtil.packArraytoBlob(cmsbxindexArray)
            beam1intensityblob=CommonUtil.packArraytoBlob(beam1intensityArray)
            beam2intensityblob=CommonUtil.packArraytoBlob(beam2intensityArray)
        if deliveredonly:
            perlsdata=[0,float(instlumi),instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,mystartorbit,cmsbxindexblob,beam1intensityblob,beam2intensityblob,bxdataocc1blob,bxerrorocc1blob,bxqualityocc1blob,bxdataocc2blob,bxerrorocc2blob,bxqualityocc2blob,bxdataetblob,bxerroretblob,bxqualityetblob]
        else:
            perlsdata=[cmslsnum,float(instlumi),instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,mystartorbit,cmsbxindexblob,beam1intensityblob,beam2intensityblob,bxdataocc1blob,bxerrorocc1blob,bxqualityocc1blob,bxdataocc2blob,bxerrorocc2blob,bxqualityocc2blob,bxdataetblob,bxerroretblob,bxqualityetblob]
        lumilsdata[cmslsnum]=perlsdata
    #print 'toinsert from scratch',lumilsdata
    print lumilsdata
    dbsession.transaction().start(False)
    (revision_id,entry_id,data_id)=dataDML.addLumiRunDataToBranch(dbsession.nominalSchema(),runnum,lumirundata,branchinfo,'LUMIDATA')
    dataDML.bulkInsertLumiLSSummary(dbsession,runnum,data_id,lumilsdata,'LUMISUMMARYV2',withDetails=withDetails)
    dbsession.transaction().commit()
    
def parsebunchFile(ifilename):
    '''
    perbunchrawdata :{bunchidx:lumifraction}
    '''
    perbunchdata={}
    result=[]
    try:
        csvfile=open(ifilename,'rb')
        reader=csv.reader(csvfile,delimiter=' ',skipinitialspace=True)
        for row in reader:
            row=[x for x in row if len(x)>0]
            result+=row
        for i in convertlist(result):
            perbunchdata[i[0]]=i[1]
        return perbunchdata
    except Exception,e:
        raise RuntimeError(str(e))
    
def parseLSFile(ifilename):
    perlsdata={}
    result=[]
    try:
        csvfile=open(ifilename,'rb')
        reader=csv.reader(csvfile,delimiter=' ',skipinitialspace=True)
        for row in reader:
            row=[x for x in row if len(x)>0]
            result+=row
        for i in convertlist(result):
            perlsdata[i[0]]=i[1]/constfactor
        return perlsdata
    except Exception,e:
        raise RuntimeError(str(e))

def main(*args):
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi fake",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',action='store',
                        required=True,
                        help='connect string to lumiDB,optional',
                        default=None)
    parser.add_argument('-P',dest='authpath',action='store',
                        required=True,
                        help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',
                        type=int,
                        required=True,
                        help='run number')
    parser.add_argument('-isummary',dest='summaryfile',action='store',
                        required=True,
                        help='lumi summary file ')
    parser.add_argument('-idetail',dest='detailfile',action='store',
                        required=False,
                        help='lumi detail file ')
    parser.add_argument('--delivered-only',dest='deliveredonly',action='store_true',
                        help='without trigger' )
    #
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug')
    options=parser.parse_args()
    os.environ['CORAL_AUTH_PATH'] = options.authpath
        
    perlsrawdata=parseLSFile(options.summaryfile)
    
    #print perlsrawdata
    perbunchrawdata={}
    bxdistribution={}
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    dbsession=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    dbsession.transaction().start(True)
    oldlumidataid=dataDML.guessLumiDataIdByRunInBranch(dbsession.nominalSchema(),options.runnumber,'LUMIDATA','DATA')
    if oldlumidataid:
        perbunchrawdata=fetchOldData(dbsession.nominalSchema(),oldlumidataid)
    elif options.detailfile:
        bxdistribution=parsebunchFile(options.detailfile)
    dbsession.transaction().commit()
    #print perlsrawdata
    #print perbunchrawdata
    insertLumischemaV2(dbsession,options.runnumber,options.summaryfile,perlsrawdata,perbunchrawdata,bxdistribution,deliveredonly=options.deliveredonly)
    del dbsession
    del svc

if __name__=='__main__':
    sys.exit(main(*sys.argv))
