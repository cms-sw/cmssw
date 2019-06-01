from __future__ import print_function
from builtins import range
import csv,os,sys,coral,array
from RecoLuminosity.LumiDB import CommonUtil,idDealer,dbUtil,dataDML,revisionDML
ilsfilename='/build/zx/patch/Run170899-ls.txt'
ibunchfilename='/build/zx/patch/Run170899-bunch.txt'
runnum=170899
conn='oracle://cms_orcoff_prep/cms_lumi_dev_offline'
beamenergy=3.5e03
beamstatus='STABLE BEAMS'
lumiversion='0001'
dtnorm=1.0
lhcnorm=1.0
cmsalive=1
numorbit=262144
startorbit=0
lslength=23.357
bunchnorm=6.37
beam1intensity=9124580336.0
beam2intensity=8932813306.0

def convertlist(l):
    '''yield successive pairs for l
    '''
    for i in range(0,len(l),2):
        idx=int(l[i])
        val=float(l[i+1])
        yield (idx,val)

def insertLumischemaV2(dbsession,runnum,datasource,perlsrawdata,perbunchrawdata):
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
        bxdataArray=array.array('f')
        bxerrorArray=array.array('f')
        bxqualityArray=array.array('h')
        cmsbxindexArray=array.array('h')
        beam1intensityArray=array.array('f')
        beam2intensityArray=array.array('f')
        for bxidx in range(1,3565):
            lumifraction=0.0
            if bxidx in perbunchrawdata:
                lumifraction=perbunchrawdata[bxidx]
            bxlumivalue=float(instlumi*lumifraction)/float(bunchnorm)
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
        perlsdata=[cmslsnum,float(instlumi)/float(6370),0.0,1,'STABLE BEAMS',beamenergy,numorbit,mystartorbit,cmsbxindexblob,beam1intensityblob,beam2intensityblob,bxdataocc1blob,bxerrorocc1blob,bxqualityocc1blob,bxdataocc2blob,bxerrorocc2blob,bxqualityocc2blob,bxdataetblob,bxerroretblob,bxqualityetblob]
        lumilsdata[cmslsnum]=perlsdata
    print(lumilsdata)
    dbsession.transaction().start(False)
    (revision_id,entry_id,data_id)=dataDML.addLumiRunDataToBranch(dbsession.nominalSchema(),runnum,lumirundata,branchinfo)
    dataDML.bulkInsertLumiLSSummary(dbsession,runnum,data_id,lumilsdata)
    dbsession.transaction().commit()
    
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
    print('lumisummary to insert : ',perlsiData)
    print('summaryidlsmap ',summaryidlsmap)
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
    print('to insert lumidetail ')
    for algoname in ['OCC1','OCC2','ET']:
        for cmslsnum,instlumi in perlsrawdata.items():
            lumisummary_id=summaryidlsmap[cmslsnum]
            lumidetail_id=iddealer.generateNextIDForTable('LUMIDETAIL')
            bxdata=array.array('f')
            bxerror=array.array('f')
            bxquality=array.array('h')
            for bxidx in range(1,3565):
                lumifraction=0.0
                if bxidx in perbunchrawdata:
                    lumifraction=perbunchrawdata[bxidx]
                bxlumivalue=float(instlumi*lumifraction)/float(bunchnorm)
                bxdata.append(bxlumivalue)
                bxerror.append(0.0)
                bxquality.append(1)
            bxdataBlob=CommonUtil.packArraytoBlob(bxdata)
            bxerrorBlob=CommonUtil.packArraytoBlob(bxerror)
            bxqualityBlob=CommonUtil.packArraytoBlob(bxquality)
            perbunchiData.append([('LUMISUMMARY_ID',lumisummary_id),('LUMIDETAIL_ID',lumidetail_id),('BXLUMIVALUE',bxdataBlob),('BXLUMIERROR',bxerrorBlob),('BXLUMIQUALITY',bxqualityBlob),('ALGONAME',algoname)])
    db.bulkInsert('LUMIDETAIL',dataDef,perbunchiData)
    dbsession.transaction().commit()
    return 

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
            result+=row
        for i in convertlist(result):
            perbunchdata[i[0]]=i[1]
        return perbunchdata
    except Exception as e:
        raise RuntimeError(str(e))
    
def parseLSFile(ifilename):
    perlsdata={}
    result=[]
    try:
        csvfile=open(ifilename,'rb')
        reader=csv.reader(csvfile,delimiter=' ',skipinitialspace=True)
        for row in reader:
            result+=row
        for i in convertlist(result):
            perlsdata[i[0]]=i[1]/float(lslength)
        return perlsdata
    except Exception as e:
        raise RuntimeError(str(e))
    
def main(*args):
    perlsrawdata=parseLSFile(ilsfilename)
    print(perlsrawdata)
    perbunchrawdata=parsebunchFile(ibunchfilename)
    print(sum(perbunchrawdata.values()))
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/user/x/xiezhen'
    svc = coral.ConnectionService()
    dbsession=svc.connect(conn,accessMode=coral.access_Update)
    print(len(args))
    if len(args)>1 and args[1]=='--v2':
        insertLumischemaV2(dbsession,runnum,ilsfilename,perlsrawdata,perbunchrawdata)
    else:
        summaryidlsmap=insertLumiSummarydata(dbsession,perlsrawdata)
        insertLumiDetaildata(dbsession,perlsrawdata,perbunchrawdata,summaryidlsmap)
    del dbsession
    del svc
#
#change valriables ilsfilename,ibunchfilename,runnum
#
if __name__=='__main__':
    sys.exit(main(*sys.argv))
