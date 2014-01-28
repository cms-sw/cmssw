import csv,os,sys,coral,array
from RecoLuminosity.LumiDB import CommonUtil,idDealer,dbUtil
ilsfilename='Run152474-ls.txt'
ibunchfilename='bunchdistribution.txt'
conn='oracle://cms_orcoff_prep/cms_lumi_prod'
beamenergy=3.5e03
beamstatus='STABLE BEAMS'
beamintensity=0
runnum=152474
lumiversion='0001'
dtnorm=1.0
lhcnorm=1.0
cmsalive=1
numorbit=262144
startorbit=0
#lslength=23.357
beam1intensity=9124580336.0
beam2intensity=8932813306.0

def convertlist(l):
    '''yield successive pairs for l
    '''
    for i in xrange(0,len(l),2):
        idx=int(l[i])
        val=float(l[i+1])
        yield (idx,val)
def insertLumiSummarydata(dbsession,perlsrawdata):
    '''
    input: perlsrawdata {cmslsnum:instlumi}
    insert into lumisummary(lumisummary_id,runnum,cmslsnum,lumilsnum,lumiversion,dtnorm,lhcnorm,instlumi,instlumierror,instlumiquality,cmsalive,startorbit,numorbit,lumisectionquality,beamenergy,beamstatus) values()
    '''
    summaryidlsmap={}
    dataDef=[]
    dataDef.append(('LUMISUMMARY_ID','unsigned long long'))
    dataDef.append(('RUNNUM','unsigned int'))
    dataDef.append(('CMSLSNUM','unsigned int'))
    dataDef.append(('LUMILSNUM','unsigned int'))
    dataDef.append(('LUMIVERSION','string'))
    dataDef.append(('DTNORM','float'))
    dataDef.append(('LHCNORM','float'))
    dataDef.append(('INSTLUMI','float'))
    dataDef.append(('INSTLUMIERROR','float'))
    dataDef.append(('INSTLUMIQUALITY','short'))
    dataDef.append(('CMSALIVE','short'))
    dataDef.append(('STARTORBIT','unsigned int'))
    dataDef.append(('NUMORBIT','unsigned int'))
    dataDef.append(('LUMISECTIONQUALITY','short'))
    dataDef.append(('BEAMENERGY','float'))
    dataDef.append(('BEAMSTATUS','string'))
    
    perlsiData=[]
    dbsession.transaction().start(False)
    iddealer=idDealer.idDealer(dbsession.nominalSchema())
    db=dbUtil.dbUtil(dbsession.nominalSchema())
    lumisummary_id=0
    for cmslsnum,instlumi in perlsrawdata.items():
        mystartorbit=startorbit+numorbit*(cmslsnum-1)
        lumisummary_id=iddealer.generateNextIDForTable('LUMISUMMARY')
        summaryidlsmap[cmslsnum]=lumisummary_id
        perlsiData.append([('LUMISUMMARY_ID',lumisummary_id),('RUNNUM',runnum),('CMSLSNUM',cmslsnum),('LUMILSNUM',cmslsnum),('LUMIVERSION',lumiversion),('DTNORM',dtnorm),('LHCNORM',lhcnorm),('INSTLUMI',instlumi),('INSTLUMIERROR',0.0),('CMSALIVE',cmsalive),('STARTORBIT',mystartorbit),('NUMORBIT',numorbit),('LUMISECTIONQUALITY',1),('BEAMENERGY',beamenergy),('BEAMSTATUS',beamstatus)])
    db.bulkInsert('LUMISUMMARY',dataDef,perlsiData)
    dbsession.transaction().commit()
    print 'lumisummary to insert : ',perlsiData
    print 'summaryidlsmap ',summaryidlsmap
    return summaryidlsmap
    
def insertLumiDetaildata(dbsession,perlsrawdata,perbunchrawdata,summaryidlsmap):
    dataDef=[]
    dataDef.append(('LUMISUMMARY_ID','unsigned long long'))
    dataDef.append(('LUMIDETAIL_ID','unsigned long long'))
    dataDef.append(('BXLUMIVALUE','blob'))
    dataDef.append(('BXLUMIERROR','blob'))
    dataDef.append(('BXLUMIQUALITY','blob'))
    dataDef.append(('ALGONAME','string'))
    perbunchiData=[]
    dbsession.transaction().start(False)
    iddealer=idDealer.idDealer(dbsession.nominalSchema())
    db=dbUtil.dbUtil(dbsession.nominalSchema())
    print 'to insert lumidetail '
    for algoname in ['OCC1','OCC2','ET']:
        for cmslsnum,instlumi in perlsrawdata.items():
            lumisummary_id=summaryidlsmap[cmslsnum]
            lumidetail_id=iddealer.generateNextIDForTable('LUMIDETAIL')
            print 'cmslsnum ',lumidetail_id,lumisummary_id
            bxdataocc1=array.array('f')
            bxdataocc2=array.array('f')
            bxdataet=array.array('f')
            bxerror=array.array('f')
            bxquality=array.array('h')
            for bxidx in range(1,3565):
                lumifraction=0.0
                if perbunchrawdata.has_key(bxidx):
                    lumifraction=perbunchrawdata[bxidx]
                bxlumivalue=float(instlumi*lumifraction)
                bxdataocc1.append(bxlumivalue)
                bxdataocc2.append(bxlumivalue)
                bxdataet.append(bxlumivalue)
            bxdataocc1Blob=CommonUtil.packArraytoBlob(bxdataocc1)
            bxdataocc2Blob=CommonUtil.packArraytoBlob(bxdataocc2)
            bxdataetBlob=CommonUtil.packArraytoBlob(bxdataet)
            bxqualityBlob=CommonUtil.packArraytoBlob(bxquality)
            perbunchiData.append([('LUMISUMMARY_ID',lumisummary_id),('LUMIDETAIL_ID',lumidetail_id),('BXLUMIVALUE',bxdataocc1Blob),('BXLUMIERROR',bxdataocc2Blob),('BXLUMIQUALITY',bxqualityBlob),('ALGONAME',algoname)])
    db.bulkInsert('LUMIDETAIL',dataDef,perbunchiData)
    print perbunchiData
    dbsession.transaction().commit()
    return 

def parsebunchFile(ifilename):
    print 0
    perbunchdata={}
    result=[]
    try:
        csvfile=open(ifilename,'rb')
        reader=csv.reader(csvfile,delimiter=' ',skipinitialspace=True)
        for row in reader:
            r=[elem for elem in row if elem not in ['',' ','\n','\t']]
            result+=r
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
            r=[elem for elem in row if elem not in ['',' ','\n','\t']]
            result+=r
        for i in convertlist(result):
            perlsdata[i[0]]=i[1]
        return perlsdata
    except Exception,e:
        raise RuntimeError(str(e))
    
def main(*args):
    perlsrawdata=parseLSFile(ilsfilename)
    #print perlsrawdata
    perbunchrawdata=parsebunchFile(ibunchfilename)
    #print perbunchrawdata
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/user/x/xiezhen'
    svc = coral.ConnectionService()
    dbsession=svc.connect(conn,accessMode=coral.access_Update)
    summaryidlsmap=insertLumiSummarydata(dbsession,perlsrawdata)
    insertLumiDetaildata(dbsession,perlsrawdata,perbunchrawdata,summaryidlsmap)
    del dbsession
    del svc
if __name__=='__main__':
    sys.exit(main(*sys.argv))
