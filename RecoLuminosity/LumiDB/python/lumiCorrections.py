import os,coral,re
from RecoLuminosity.LumiDB import nameDealer

def constCorrectionByEnergy(schema,nominalenergy):
    '''
    1.075
    '''
    if nominalenergy*1.2>3500 and nominalenergy*0.8<3500:
        return 1.075
    else: 
        return 1.0
    
def fillCorrectionsMap(schema):
    '''
    select fillschemepattern,correctionfactor from fillscheme; 
       [(fillschemepattern,afterglow),...]
    select distinct fillnum,fillscheme,ncollidingbunches from cmsrunsummary;
       [(fillnum,fillscheme,ncollidingbunches),...]
       
    output: {fillnum:(afterglowfactor,nonlinearfactor)}
    afterglowfactor= (default 1.0)
    nonlinearfactor=0.076/nbx (default 0.0)
    '''
    result={}
    afterglows=[]
    
    qHandle=schema.newQuery()
    r=nameDealer.cmsrunsummaryTableName()
    s=nameDealer.fillschemeTableName()
    try:
        qHandle.addToTableList(s)
        qResult=coral.AttributeList()
        qResult.extend('FILLSCHEMEPATTERN','string')
        qResult.extend('CORRECTIONFACTOR','float')
        qHandle.defineOutput(qResult)
        qHandle.addToOutputList('FILLSCHEMEPATTERN')
        qHandle.addToOutputList('CORRECTIONFACTOR')
        cursor=qHandle.execute()
        while cursor.next():
            fillschemePattern=cursor.currentRow()['FILLSCHEMEPATTERN'].data()
            afterglowfac=cursor.currentRow()['CORRECTIONFACTOR'].data()
            afterglows.append((fillschemePattern,afterglowfac))
    except :
        del qHandle
        raise
    del qHandle
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(r)
        qHandle.addToOutputList('distinct FILLNUM', 'fillnum')
        qHandle.addToOutputList('FILLSCHEME','fillscheme')
        qHandle.addToOutputList('NCOLLIDINGBUNCHES','ncollidingbunches')
        qResult=coral.AttributeList()
        qResult.extend('fillnum','unsigned int')
        qResult.extend('fillscheme','string')
        qResult.extend('ncollidingbunches','unsigned int')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        while cursor.next():
            afterglow=1.0
            nonlinear=0.076
            nonlinearPerBX=0.0
            fillnum=cursor.currentRow()['fillnum'].data()
            ncollidingbunches=0
            if cursor.currentRow()['ncollidingbunches']:
                ncollidingbunches=cursor.currentRow()['ncollidingbunches'].data()
            fillscheme=''
            if cursor.currentRow()['fillscheme']:
                fillscheme=cursor.currentRow()['fillscheme'].data()
            if fillscheme and len(fillscheme)!=0:
                afterglow=afterglowByFillscheme(fillscheme,afterglows)
            if ncollidingbunches and ncollidingbunches!=0:
                nonlinearPerBX=float(1)/float(ncollidingbunches)
            nonlinear=nonlinearPerBX*nonlinear
            if fillnum and fillnum!=0:
                result[fillnum]=(afterglow,nonlinear)
    except :
        del qHandle
        raise
    del qHandle
    return result

def afterglowByFillscheme(fillscheme,afterglowPatterns):
    for (apattern,cfactor) in afterglowPatterns:
        if re.match(apattern,fillscheme):
            return cfactor
    return 1.0

if __name__ == "__main__":
    import sessionManager
    myconstr='oracle://cms_orcoff_prep/cms_lumi_dev_offline'
    svc=sessionManager.sessionManager(myconstr,authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    schema=session.nominalSchema()
    session.transaction().start(True)
    fillcmap=fillCorrectionsMap(schema)
    session.transaction().commit()
    del session
    print len(fillcmap)
