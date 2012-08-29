import csv,os,sys,coral,array
from RecoLuminosity.LumiDB import argparse,sessionManager,CommonUtil,idDealer,dbUtil,dataDML,revisionDML
beamenergy=4.0e03
beamstatus='STABLE BEAMS'
numorbit=262144
startorbit=0
#lslength=23.31
beam1intensity=9124580336.0
beam2intensity=8932813306.0

def convertlist(l):
    '''yield successive pairs for l
    '''
    for i in xrange(0,len(l),2):
        idx=int(l[i])
        val=float(l[i+1])
        yield (idx,val)

def insertLumischemaV2(dbsession,runnum,datasource,perlsrawdata,perbunchrawdata,deliveredonly=False):
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
    dbsession.transaction().start(True)
    oldlumidataid=dataDML.guessLumiDataIdByRunInBranch(dbsession.nominalSchema(),runnum,'LUMIDATA','DATA')
    dbsession.transaction().commit()

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
        if perbunchrawdata:
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
            perlsdata=[0,float(instlumi),0.0,1,'STABLE BEAMS',beamenergy,numorbit,mystartorbit,cmsbxindexblob,beam1intensityblob,beam2intensityblob,bxdataocc1blob,bxerrorocc1blob,bxqualityocc1blob,bxdataocc2blob,bxerrorocc2blob,bxqualityocc2blob,bxdataetblob,bxerroretblob,bxqualityetblob]
        else:
            perlsdata=[cmslsnum,float(instlumi),0.0,1,'STABLE BEAMS',beamenergy,numorbit,mystartorbit,cmsbxindexblob,beam1intensityblob,beam2intensityblob,bxdataocc1blob,bxerrorocc1blob,bxqualityocc1blob,bxdataocc2blob,bxerrorocc2blob,bxqualityocc2blob,bxdataetblob,bxerroretblob,bxqualityetblob]
        lumilsdata[cmslsnum]=perlsdata
    #print 'toinsert from scratch',lumilsdata
    
    dbsession.transaction().start(False)
    (revision_id,entry_id,data_id)=dataDML.addLumiRunDataToBranch(dbsession.nominalSchema(),runnum,lumirundata,branchinfo,'LUMIDATA')
    newlumilsnumMin=min(lumilsdata.keys())
    newlumilsnumMax=max(lumilsdata.keys())
    if oldlumidataid:
        #update id of existing to new
        #update lumisummaryv2 set DATA_ID=:data_id where DATA_ID=:oldid and lumilsnum<newlumilsnumMin or lumilsnum>newlumilsnumMax
        #lumidataid is not None  update lumilsdata set lumidataid=:newlumidataid where runnum=:run a
        inputData=coral.AttributeList()
        inputData.extend('dataid','unsigned long long')
        inputData.extend('oldid','unsigned long long')
        inputData.extend('lumilsnumMin','unsigned int')
        inputData.extend('lumilsnumMax','unsigned int')
        inputData['dataid'].setData( data_id )
        inputData['oldid'].setData( oldlumidataid )
        inputData['lumilsnumMin'].setData( newlumilsnumMin )
        inputData['lumilsnumMax'].setData( newlumilsnumMax )
        db=dbUtil.dbUtil(dbsession.nominalSchema())
        db.singleUpdate('LUMISUMMARYV2','DATA_ID=:dataid','DATA_ID=:oldid AND (LUMILSNUM<:lumilsnumMin OR lumilsnum>:lumilsnumMax)',inputData)
        print 'to update existing id ',oldlumidataid,' outside region ',newlumilsnumMin,' , ',newlumilsnumMax
    dataDML.bulkInsertLumiLSSummary(dbsession,runnum,data_id,lumilsdata,'LUMISUMMARYV2',withDetails=False)
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
            perlsdata[i[0]]=i[1]
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
    print perlsrawdata
    perbunchrawdata={}
    if options.detailfile:
        perbunchrawdata=parsebunchFile(options.detailfile)
        print sum(perbunchrawdata.values())
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    dbsession=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    insertLumischemaV2(dbsession,options.runnumber,options.summaryfile,perlsrawdata,perbunchrawdata,deliveredonly=options.deliveredonly)
    del dbsession
    del svc

if __name__=='__main__':
    sys.exit(main(*sys.argv))
