import coral
'''
This module defines lowlevel SQL query API for lumiDB 
We do not like range queries so far because of performance of range scan. 
The principle is to query by runnumber and per each coral queryhandle
Try reuse db session/transaction and just renew query handle each time to reduce metadata queries.
Avoid explicit order by, should be solved by asc index in the schema.
'''
def runsummaryByrun(queryHandle,runnum):
    '''
    select fillnum,sequence,hltkey,starttime,stoptime from cmsrunsummary where runnum=:runnum
    output: [fillnum,sequence,hltkey,starttime,stoptime]
    '''
    pass

def lumisummaryByrun(queryHandle,runnum,lumiversion):
    '''
    select cmslsnum,instlumi,numorbit,startorbit,beamstatus,beamenery from lumisummary where runnum=:runnum and lumiversion=:lumiversion
    output: {cmslsnum:[instlumi,numorbit,startorbit,beamstatus,beamenergy]}
    '''
    pass

def lumisumByrun(queryHandle,runnum,lumiversion):
    '''
    select sum(instlumi) from lumisummary where runnum=:runnum and lumiversion=:lumiversion
    output: float totallumi
    Note: the output is the raw result, need to apply LS length in time(sec)
    '''
    pass

def trgbitzeroByrun(queryHandle,runnum):
    '''
    select cmslsnum,trgcount,deadtime,bitname,prescale from trg where runnum=:runnum and bitnum=0;
    output: {cmslsnum:[trgcount,deadtime,bitname,prescale]}
    '''
    pass

def trgBybitnameByrun(queryHandle,runnum,bitname):
    '''
    select cmslsnum,trgcount,deadtime,bitnum,prescale from trg where runnum=:runnum and bitname=:bitname;
    output: {cmslsnum:[trgcount,deadtime,bitnum,prescale]}
    '''
    pass

def trgAllbitsByrun(queryHandle,runnum):
    '''
    select cmslsnum,trgcount,deadtime,bitnum,bitname,prescale from trg where runnum=:runnum
    this can be changed to blob query later
    output: {cmslsnum:{bitname:[bitnum,trgcount,deadtime,prescale]}}
    '''
    pass

def hltBypathByrun(queryHandle,runnum.hltpath):
    '''
    select cmslsnum,inputcount,acceptcount,prescale from hlt where runnum=:runnum and pathname=:hltpath
    output: {cmslsnum:[inputcount,acceptcount,prescale]}
    '''
    pass

def hltAllpathByrun(queryHandle,runum):
    '''
    select cmslsnum,inputcount,acceptcount,prescale,hltpath from hlt where runnum=:runnum
    this can be changed to blob query later
    output: {cmslsnum:{hltpath:[inputcount,acceptcount,prescale]}}
    '''
    pass

def lumidetailByrunByAlgo(queryHandle,runum,algoname='OCC1'):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnumber and d.algoname=:algoname and s.lumisummary_id=d.lumisummary_id order by s.startorbit
    output: [cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit]
    since the output is ordered by time, it has to be in seq list format
    '''
    pass

def lumidetailAllalgosByrun(queryHandle,runum):
    '''
    select s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname from LUMIDETAIL d,LUMISUMMARY s where s.runnum=:runnumber and s.lumisummary_id=d.lumisummary_id order by s.startorbit
    output: {algoname:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality,startorbit]}
    '''
    pass

def hlttrgMappingByrun(queryHandle,runnum):
    '''
    select trghltmap.hltpathname,trghltmap.l1seed from cmsrunsummary,trghltmap where cmsrunsummary.runnum=:runnum and trghltmap.hltkey=cmsrunsummary.hltkey
    output: {hltpath:l1seed}
    '''
    pass
