import os,coral,re
from RecoLuminosity.LumiDB import nameDealer

class correctionTerm(object):
    constfactor=1.141 # const upshift , same for everyone     

class nonlinearSingle(correctionTerm):
    t1=0.076  #slop
    
class nonlinearV2(correctionTerm):
    drift=0.01258 # drift 
    t1=0.063  # slop1
    t2=-0.0037# slop2
        
class nonlinearV3(correctionTerm):
    drift=0.00813# drift 
    t1=0.073    # slop1
    t2=-0.0037 # slop2
        
def afterglowByFillscheme(fillscheme,afterglowPatterns):
    '''
    search in the list of (pattern,afterglowfactor) for a match in regex
    '''
    for (apattern,cfactor) in afterglowPatterns:
        if re.match(apattern,fillscheme):
            print apattern,cfactor
            return cfactor
    return 1.0

#=======================================================================================================
#below : correction formula version_2
#======================================================================================================
def driftcorrectionsForRange(schema,inputRange,correctionTerm,startrun=160403):
    '''
    select intglumi from intglumi where runnum=:runnum and startrun=:startrun
    input : inputRange. str if a single run, [runs] if a list of runs
    output: {run:driftcorrection}
    '''
    result={}
    runs=[]
    if isinstance(inputRange,str):
        runs.append(int(inputRange))
    else:
        runs=inputRange
    for r in runs:
        defaultresult=1.0
        intglumi=0.0
        lint=0.0 
        if r<150008 :# no drift corrections for 2010 data
            result[r]=defaultresult
            continue
        if r>189738: # no drift correction for 2012 data
            result[r]=defaultresult
            continue
        qHandle=schema.newQuery()
        try:
            qHandle.addToTableList(nameDealer.intglumiTableName())
            qResult=coral.AttributeList()
            qResult.extend('INTGLUMI','float')
            qHandle.addToOutputList('INTGLUMI')
            qConditionStr='RUNNUM=:runnum AND STARTRUN=:startrun'
            qCondition=coral.AttributeList()
            qCondition.extend('runnum','unsigned int')
            qCondition.extend('startrun','unsigned int')
            qCondition['runnum'].setData(int(r))
            qCondition['startrun'].setData(int(startrun))
            qHandle.setCondition(qConditionStr,qCondition)
            qHandle.defineOutput(qResult)
            cursor=qHandle.execute()
            while cursor.next():
                intglumi=cursor.currentRow()['INTGLUMI'].data()
            lint=intglumi*6.37*1.0e-9 #(convert to /fb)
            #print lint
        except :
            del qHandle
            raise
        del qHandle
        if not lint:
            print '[WARNING] null intglumi for run ',r,' '
        result[r]=defaultresult+correctionTerm.drift*lint
    #print 'lint ',lint,' result ',result
    return result

def applyfinecorrectionBXV2(bxlumi,avglumi,norm,constfactor,afterglowfactor,ncollidingbx,nonlinear_1,nonlinear_2,driftfactor):
    if bxlumi<=0:#do nothing about the negative bx lumi
        return bxlumi
    correctbxlumi=bxlumi*norm*constfactor*afterglowfactor*driftfactor
    if ncollidingbx and ncollidingbx!=0:
        dldt=avglumi/float(ncollidingbx)
        nonlinearTerm=1.0+nonlinear_1*dldt+nonlinear_2*dldt*dldt
        correctbxlumi=correctbxlumi/nonlinearTerm
        #print 'avglumi,nonlinearfactor,nonlinearTerm ',avglumi,nonlinearfactor,nonlinearTerm
    #print 'bxlumi,avglumi,norm,const,after',bxlumi,avglumi,norm,constfactor,afterglowfactor,correctbxlumi
    return correctbxlumi

def applyfinecorrectionV2(avglumi,constfactor,afterglowfactor,ncollidingbx,nonlinear_1,nonlinear_2,driftfactor):
    '''
    input :
         avglumi : normalized lumi with 6370
         constfactor,afterglowfactor,ncollidingbx,nonlinear_1,nonlinear_2
         driftfactor: default
    '''
    #print avglumi,constfactor,afterglowfactor,ncollidingbx,nonlinear_1,nonlinear_2,driftfactor
    instlumi=avglumi*afterglowfactor*constfactor*driftfactor
    if ncollidingbx and ncollidingbx!=0:
        dldt=avglumi/float(ncollidingbx)
        nonlinearTerm=1.0+nonlinear_1*dldt+nonlinear_2*dldt*dldt
        instlumi=instlumi/nonlinearTerm
    #print 'avglumi,const,after,nonlinear,instlumi ',avglumi,constfactor,afterglowfactor,nonlinearfactor,instlumi
    return instlumi

def correctionsForRangeV2(schema,inputRange,correctionTerm):
    '''
    decide on the corrections to apply in the input range depending on amodetag,egev and runrange
    select fillschemepattern,correctionfactor from fillscheme; 
       [(fillschemepattern,afterglow),...]
    select fillnum,runnum,fillscheme,ncollidingbunches,egev from cmsrunsummary where amodetag='PROTPYHS' and egev>3000
        {runnum: (fillnum,fillscheme,ncollidingbunches),...}
    input: correctionTerm correction terms used in the formula
    output:
        {runnum:(constantfactor,afterglowfactor,ncollidingbx,nonlinearfactor1,nonlinearfactor2)}
    '''
    runs=[]
    result={}
    constfactor=1.0 #default
    afterglow=1.0 #default
    ncollidingbunches=0 #default
    nonlinear_1=1.0 #default
    nonlinear_2=1.0 #default
    if isinstance(inputRange,str):
        runs.append(int(inputRange))
    else:
        runs=inputRange
    for r in runs:
        if r<150008 :
            result[r]=(constfactor,afterglow,ncollidingbunches,nonlinear_1, nonlinear_2)
    afterglows=[]
    s=nameDealer.fillschemeTableName()
    r=nameDealer.cmsrunsummaryTableName()
    qHandle=schema.newQuery()
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
        qHandle.addToOutputList('FILLNUM', 'fillnum')
        qHandle.addToOutputList('RUNNUM', 'runnum')
        qHandle.addToOutputList('FILLSCHEME','fillscheme')
        qHandle.addToOutputList('NCOLLIDINGBUNCHES','ncollidingbunches')
        qResult=coral.AttributeList()
        qResult.extend('fillnum','unsigned int')
        qResult.extend('runnum','unsigned int')
        qResult.extend('fillscheme','string')
        qResult.extend('ncollidingbunches','unsigned int')
        qConditionStr='AMODETAG=:amodetag AND EGEV>=:egev'#filter out lowenergy and non-proton runs
        qCondition=coral.AttributeList()
        qCondition.extend('amodetag','string')
        qCondition.extend('egev','unsigned int')
        qCondition['amodetag'].setData('PROTPHYS')
        qCondition['egev'].setData(3000)
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            #print 'runnum ',runnum 
            if runnum not in runs or result.has_key(runnum):
                continue
            fillnum=cursor.currentRow()['fillnum'].data()
            afterglow=1.0
            constfactor=correctionTerm.constfactor
            nonlinear_1=correctionTerm.t1
            nonlinear_2=correctionTerm.t2
            ncollidingbunches=0
            if cursor.currentRow()['ncollidingbunches']:
                ncollidingbunches=cursor.currentRow()['ncollidingbunches'].data()
            fillscheme=''
            if cursor.currentRow()['fillscheme']:
                fillscheme=cursor.currentRow()['fillscheme'].data()
            if fillscheme and len(fillscheme)!=0:
                if fillnum>=2124: #afterglow'salready applied by lumidaq in hf root for fill<2124                 
                    afterglow=afterglowByFillscheme(fillscheme,afterglows)           
            result[runnum]=(constfactor,afterglow,ncollidingbunches,nonlinear_1,nonlinear_2)
    except :
        del qHandle
        raise
    del qHandle
    for run in runs:
        if run not in result.keys():
            result[run]=(constfactor,afterglow,ncollidingbunches,nonlinear_1,nonlinear_2) 
    return result
#=======================================================================================================
#below : below correction formula version_1,  default untill April 2012, no more used.
#======================================================================================================
#def applyfinecorrectionBX(bxlumi,avglumi,norm,constfactor,afterglowfactor,nonlinearfactor):
#    if bxlumi<=0:
#        return bxlumi
#    correctbxlumi=bxlumi*norm*constfactor*afterglowfactor
#    if constfactor!=1.0 and nonlinearfactor!=0:
#        if avglumi<0:
#            avglumi=0.0
#        nonlinearTerm=1.0+avglumi*nonlinearfactor#0.076/ncollidinbunches
#        correctbxlumi=correctbxlumi/nonlinearTerm
#        #print 'avglumi,nonlinearfactor,nonlinearTerm ',avglumi,nonlinearfactor,nonlinearTerm
#    #print 'bxlumi,avglumi,norm,const,after',bxlumi,avglumi,norm,constfactor,afterglowfactor,correctbxlumi
#    return correctbxlumi

#def applyfinecorrection(avglumi,constfactor,afterglowfactor,nonlinearfactor):
#    instlumi=avglumi*afterglowfactor*constfactor
#    if nonlinearfactor!=0 and constfactor!=1.0:
#        nonlinearTerm=1.0+avglumi*nonlinearfactor#0.076/ncollidinbunches
#        instlumi=instlumi/nonlinearTerm
#    #print 'avglumi,const,after,nonlinear,instlumi ',avglumi,constfactor,afterglowfactor,nonlinearfactor,instlumi
#    return instlumi

#def correctionsForRange(schema,inputRange,correctionTerm):
#    '''
#    select fillschemepattern,correctionfactor from fillscheme; 
#       [(fillschemepattern,afterglow),...]
#    select fillnum,runnum,fillscheme,ncollidingbunches,egev from cmsrunsummary where amodetag='PROTPYHS' and egev>3000
#        {runnum: (fillnum,fillscheme,ncollidingbunches),...}
#    output:
#        {runnum:(constantfactor,afterglowfactor,nonlinearfactor)}
#    '''
#    runs=[]
#    result={}
#    if isinstance(inputRange,str):
#        runs.append(int(inputRange))
#    else:
#        runs=inputRange
#    for r in runs:
#        if r<150008 :
#            result[r]=(1.0,1.0,0.0)
#    afterglows=[]
#    constfactor=correctionTerm.constfactor
#    s=nameDealer.fillschemeTableName()
#    r=nameDealer.cmsrunsummaryTableName()
#    qHandle=schema.newQuery()
#    try:
#        qHandle.addToTableList(s)
#        qResult=coral.AttributeList()
#        qResult.extend('FILLSCHEMEPATTERN','string')
#        qResult.extend('CORRECTIONFACTOR','float')
#        qHandle.defineOutput(qResult)
#        qHandle.addToOutputList('FILLSCHEMEPATTERN')
#        qHandle.addToOutputList('CORRECTIONFACTOR')
#        cursor=qHandle.execute()
#        while cursor.next():
#            fillschemePattern=cursor.currentRow()['FILLSCHEMEPATTERN'].data()
#            afterglowfac=cursor.currentRow()['CORRECTIONFACTOR'].data()
#            afterglows.append((fillschemePattern,afterglowfac))
#    except :
#        del qHandle
#        raise
#    del qHandle
#    qHandle=schema.newQuery()
#    try:
#        qHandle.addToTableList(r)
#        qHandle.addToOutputList('FILLNUM', 'fillnum')
#        qHandle.addToOutputList('RUNNUM', 'runnum')
#        qHandle.addToOutputList('FILLSCHEME','fillscheme')
#        qHandle.addToOutputList('NCOLLIDINGBUNCHES','ncollidingbunches')
#        qResult=coral.AttributeList()
#        qResult.extend('fillnum','unsigned int')
#        qResult.extend('runnum','unsigned int')
#        qResult.extend('fillscheme','string')
#        qResult.extend('ncollidingbunches','unsigned int')
#        qConditionStr='AMODETAG=:amodetag AND EGEV>=:egev'
#        qCondition=coral.AttributeList()
#        qCondition.extend('amodetag','string')
#        qCondition.extend('egev','unsigned int')
#        qCondition['amodetag'].setData('PROTPHYS')
#        qCondition['egev'].setData(3000)
#        qHandle.defineOutput(qResult)
#        qHandle.setCondition(qConditionStr,qCondition)
#        cursor=qHandle.execute()
#        while cursor.next():
#            runnum=cursor.currentRow()['runnum'].data()
#            #print 'runnum ',runnum 
#            if runnum not in runs or result.has_key(runnum):
#                continue
#            fillnum=cursor.currentRow()['fillnum'].data()
#            constfactor=correctionTerm.constfactor
#            afterglow=1.0
#            nonlinear=correctionTerm.t1
#            nonlinearPerBX=0.0
#            ncollidingbunches=0
#            if cursor.currentRow()['ncollidingbunches']:
#                ncollidingbunches=cursor.currentRow()['ncollidingbunches'].data()
#            fillscheme=''
#            if cursor.currentRow()['fillscheme']:
#                fillscheme=cursor.currentRow()['fillscheme'].data()
#            if fillscheme and len(fillscheme)!=0:
#                afterglow=afterglowByFillscheme(fillscheme,afterglows)
#            if ncollidingbunches and ncollidingbunches!=0:
#                nonlinearPerBX=float(1)/float(ncollidingbunches)
#            nonlinear=nonlinearPerBX*nonlinear
#            result[runnum]=(constfactor,afterglow,nonlinear)
#    except :
#        del qHandle
#        raise
#    del qHandle
#    for run in runs:
#        if run not in result.keys():
#            result[run]=(1.0,1.0,0.0) #those have no fillscheme 2011 runs
#    return result
#=======================================================================================================
#below : correction on pixellumi, afterglow only
#======================================================================================================
def pixelcorrectionsForRange(schema,inputRange):
    '''
    select fillschemepattern,correctionfactor from fillscheme; 
       [(fillschemepattern,afterglow),...]
    select fillnum,runnum,fillscheme from cmsrunsummary where amodetag='PROTPHYS' 
        {runnum: (fillnum,fillscheme),...}
    output:
        {runnum:(afterglowfactor)}
    '''
    runs=[]
    result={}
    if isinstance(inputRange,str):
        runs.append(int(inputRange))
    else:
        runs=inputRange
    if not runs:
        return {}
    minrun=min(runs)
    afterglows=[]
    if minrun>190380:
        s=nameDealer.pixelfillschemeTableName(2012)
    else:
        s=nameDealer.fillschemeTableName()
    r=nameDealer.cmsrunsummaryTableName()
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(s)
        qResult=coral.AttributeList()
        qResult.extend('FILLSCHEMEPATTERN','string')
        qResult.extend('PIXELCORRECTIONFACTOR','float')
        qHandle.defineOutput(qResult)
        qHandle.addToOutputList('FILLSCHEMEPATTERN')
        qHandle.addToOutputList('PIXELCORRECTIONFACTOR')
        cursor=qHandle.execute()
        while cursor.next():
            fillschemePattern=cursor.currentRow()['FILLSCHEMEPATTERN'].data()
            afterglowfac=cursor.currentRow()['PIXELCORRECTIONFACTOR'].data()
            afterglows.append((fillschemePattern,afterglowfac))
    except :
        del qHandle
        raise
    del qHandle
    qHandle=schema.newQuery()
    try:
        qConditionStr='FILLNUM>:minfillnum'
        qCondition=coral.AttributeList()
        qCondition.extend('minfillnum','unsigned int')
        qCondition['minfillnum'].setData(1600)
        qHandle.addToTableList(r)
        qHandle.addToOutputList('FILLNUM', 'fillnum')
        qHandle.addToOutputList('RUNNUM', 'runnum')
        qHandle.addToOutputList('FILLSCHEME','fillscheme')
        qResult=coral.AttributeList()
        qResult.extend('fillnum','unsigned int')
        qResult.extend('runnum','unsigned int')
        qResult.extend('fillscheme','string')
        qHandle.setCondition(qConditionStr,qCondition)
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            if runnum not in runs or result.has_key(runnum):
                continue
            fillnum=cursor.currentRow()['fillnum'].data()
            afterglow=1.0
            fillscheme=''
            if cursor.currentRow()['fillscheme']:
                fillscheme=cursor.currentRow()['fillscheme'].data()
            if fillscheme and len(fillscheme)!=0:
                afterglow=afterglowByFillscheme(fillscheme,afterglows)
            result[runnum]=afterglow
    except :
        del qHandle
        raise
    del qHandle
    for run in runs:
        if run not in result.keys():
            result[run]=1.0 #those have no fillscheme 
    return result

if __name__ == "__main__":
    import sessionManager
    #myconstr='oracle://cms_orcoff_prod/cms_lumi_prod'
    myconstr='oracle://cms_orcoff_prep/cms_lumi_dev_offline'
    svc=sessionManager.sessionManager(myconstr,authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    #runrange=[163337,163387,163385,163664,163757,163269,1234,152611]
    schema=session.nominalSchema()
    session.transaction().start(True)
    driftresult=driftcorrectionsForRange(schema,[160467,152611])
    print driftresult
    #result=correctionsForRange(schema,runrange)
    session.transaction().commit()
    del session

    
