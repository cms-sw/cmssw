import array,coral
from RecoLuminosity.LumiDB import CommonUtil
def lumiSummary(schema,nlumils):
    '''
    input:
    output: [datasource,{lumilsnum:[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beamintensityblob_1,beamintensitublob_2,bxlumivalue_occ1,bxlumierror_occ1,bxlumiquality_occ1,bxlumivalue_occ2,bxlumierror_occ2,bxlumiquality_occ2,bxlumivalue_et,bxlumierror_et,bxlumiquality_et]}]
    '''
    o=['file:fake.root']
    perlsdata={}
    for lumilsnum in range(1,nlumils+1):
        cmslsnum=0
        if lumilsnum<100:
            cmslsnum=lumilsnum
        instlumi=2.37
        instlumierror=0.56
        instlumiquality=2
        beamstatus='STABLE BEAMS'
        beamenergy=3.5e03
        numorbit=12345
        startorbit=numorbit*lumilsnum
        if cmslsnum==0:
            cmsbxindex=None
            beam1intensity=None
            beam2intensity=None
        else:
            cmsbxindex=array.array('I')
            beam1intensity=array.array('f')
            beam2intensity=array.array('f')
            for idx in range(1,3565):
                cmsbxindex.append(idx)
                beam1intensity.append(1.5e09)
                beam2intensity.append(5.5e09)
                cmsbxindexBlob=CommonUtil.packArraytoBlob(cmsbxindex)
                beam1intensityBlob=CommonUtil.packArraytoBlob(beam1intensity)
                beam2intensityBlob=CommonUtil.packArraytoBlob(beam2intensity)
        bxlumivalue=array.array('f')
        bxlumierror=array.array('f')
        bxlumiquality=array.array('I')
        for idx in range(1,3565):
            bxlumivalue.append(2.3)
            bxlumierror.append(0.4)
            bxlumiquality.append(2)
        bxlumivalueBlob=CommonUtil.packArraytoBlob(bxlumivalue)
        bxlumierrorBlob=CommonUtil.packArraytoBlob(bxlumierror)
        bxlumiqualityBlob=CommonUtil.packArraytoBlob(bxlumiquality)
        if not perlsdata.has_key(lumilsnum):
            perlsdata[lumilsnum]=[]
        perlsdata[lumilsnum].extend([cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexBlob,beam1intensityBlob,beam2intensityBlob,bxlumivalueBlob,bxlumierrorBlob,bxlumiqualityBlob,bxlumivalueBlob,bxlumierrorBlob,bxlumiqualityBlob,bxlumivalueBlob,bxlumierrorBlob,bxlumiqualityBlob])
    o.append(perlsdata)
    return o
#def lumiDetail(schema,nlumils):
#    '''
#    input:
#    output:[(algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]})]
#    '''
#    o=[]
#    algos=['OCC1','OCC2','ET']
#    for algoname in algos:
#        perlsdata={}
#        for lumilsnum in range(1,nlumils+1):
#            cmslsnum=0
#            cmsalive=0
#            if lumilsnum<100:
#                cmslsnum=lumilsnum
#            bxlumivalue=array.array('f')
#            bxlumierror=array.array('f')
#            bxlumiquality=array.array('I')
#            for idx in range(1,3565):
#                bxlumivalue.append(2.3)
#                bxlumierror.append(0.4)
#                bxlumiquality.append(2)
#            bxlumivalueBlob=CommonUtil.packArraytoBlob(bxlumivalue)
#            bxlumierrorBlob=CommonUtil.packArraytoBlob(bxlumierror)
#            bxlumiqualityBlob=CommonUtil.packArraytoBlob(bxlumiquality)
#            if not perlsdata.has_key(lumilsnum):
#                perlsdata[lumilsnum]=[]
#            perlsdata[lumilsnum].extend([cmslsnum,bxlumivalueBlob,bxlumierrorBlob,bxlumiqualityBlob])           
#        o.append((algoname,perlsdata))
#    return o
def trg(schema,nls):
    '''
    input:
    output: [datasource,bitzeroname,bitnameclob,{cmslsnum:[deadtime,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}]
    '''
    o=['oracle://cms_orcon_prod/cms_gtmon','L1_ZeroBias']
    bitnameclob='L1_ZeroBias,False,L1_SingleHfBitCountsRing1_1,L1_SingleHfBitCountsRing2_1,L1_SingleMu15,L1SingleJet,Jura'
    o.append(bitnameclob)
    perlsdata={}
    for cmslsnum in range(1,nls+1):
        deadtime=99+cmslsnum
        bitzerocount=897865
        bitzeroprescale=17
        trgcounts=array.array('I')
        prescalecounts=array.array('I')
        for i in range(1,192):
            trgcounts.append(778899+i)
            prescalecounts.append(17)
        trgcountsBlob=CommonUtil.packArraytoBlob(trgcounts)
        prescalecountsBlob=CommonUtil.packArraytoBlob(prescalecounts)
        if not perlsdata.has_key(cmslsnum):
            perlsdata[cmslsnum]=[]
        perlsdata[cmslsnum].extend([deadtime,bitzerocount,bitzeroprescale,trgcountsBlob,prescalecountsBlob])
    o.append(perlsdata)
    return o
def hlt(schema,nls):
    '''
    input:
    output: [datasource,pathnameclob,{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}]
    '''
    o=['oracle://cms_orcon_prod/cms_runinfo']
    pathnameclob='HLT_PixelTracks_Multiplicity70,HLT_PixelTracks_Multiplicity85,HLT_PixelTracks_Multiplicity100,HLT_GlobalRunHPDNoise,HLT_TechTrigHCALNoise'
    o.append(pathnameclob)
    perlsdata={}
    for cmslsnum in range(1,nls+1):
        inputcounts=array.array('I')
        acceptcounts=array.array('I')
        prescalecounts=array.array('I')
        for i in range(1,201):
            inputcounts.append(6677889 )
            acceptcounts.append(3344565)
            prescalecounts.append(17)
        inputcountsBlob=CommonUtil.packArraytoBlob(inputcounts)
        acceptcountsBlob=CommonUtil.packArraytoBlob(acceptcounts)
        prescalecountsBlob=CommonUtil.packArraytoBlob(prescalecounts)
        if not perlsdata.has_key(cmslsnum):
            perlsdata[cmslsnum]=[]
        perlsdata[cmslsnum].extend([inputcountsBlob,acceptcountsBlob,prescalecountsBlob])
    o.append(perlsdata)
    return o
def hlttrgmap(schema):
    '''
    input:
    output:[hltkey,{hltpahtname:l1seed}]
    '''
    o=['/cdaq/physics/firstCollisions10/v2.0/HLT_7TeV/V5']
    hlttrgmap={}
    hlttrgmap['HLT_L1Tech_BSC_halo']='4'
    hlttrgmap['HLT_PixelTracks_Multiplicity70']='L1_ETT60'
    hlttrgmap['HLT_PixelTracks_Multiplicity85']='L1_ETT60'
    hlttrgmap['HLT_PixelTracks_Multiplicity100']='L1_ETT100'
    hlttrgmap['HLT_GlobalRunHPDNoise']="L1_SingleJet10U_NotBptxOR"
    hlttrgmap['HLT_TechTrigHCALNoise']="11 OR 12"
    o.append(hlttrgmap)
    return o
def runsummary(schema,amodetag,egev):
    '''
    input:
    output:[l1key,amodetag,egev,hltkey,fillnum,sequence,starttime,stoptime]
    '''
    o=['collisioncollision','PROTPHYS',3500,'/cdaq/physics/firstCollisions10/v2.0/HLT_7TeV/V5',1005,'GLOBAL-RUN']
    starttime=coral.TimeStamp(2010,11,1,0,0,0,0)
    stoptime=coral.TimeStamp(2010,11,1,11,0,0,0)
    o.append(starttime)
    o.append(stoptime)
    return o
    
if __name__ == "__main__":
    pass
