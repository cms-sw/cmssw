import re,ast
class normFunctionFactory(object):
    '''
    luminorm and correction functions.
    The result of the functions are correction factors, not final luminosity
    all functions take 5 run time parameters, and arbituary named params
    '''

    def fPoly(self,luminonorm,intglumi,nBXs,whatev,whatav,a0=1.0,a1=0.0,a2=0.0,drift=0.0,c1=0.0,afterglow=''):
        '''
        input: luminonorm unit Hz/ub
        output: correction factor to be applied on lumi in Hz/ub
        '''
        avglumi=0.
        if c1 and nBXs>0:
            avglumi=c1*luminonorm/nBXs
        Afterglow=1.0
        if len(afterglow)!=0:
            afterglowmap=ast.literal_eval(afterglow)
            for (bxthreshold,correction) in afterglowmap:
                if nBXs >= bxthreshold :
                    Afterglow = correction
        driftterm=1.0
        if drift and intglumi:
            driftterm=1.0+drift*intglumi
        result=a0*Afterglow/(1+a1*avglumi+a2*avglumi*avglumi)*driftterm
        return result

    def fPolyScheme(self,luminonorm,intglumi,nBXs,fillschemeStr,fillschemePatterns,a0=1.0,a1=0.0,a2=0.0,drift=0.0,c1=0.0):
        '''
        input: luminonorm unit Hz/ub
        input: fillschemePatterns [(patternStr,afterglow])
        output: correction factor to be applied on lumi in Hz/ub
        '''
        avglumi=0.
        if c1 and nBXs>0:
            avglumi=c1*luminonorm/nBXs
        Afterglow=1.0
        if fillschemeStr and fillschemePatterns:
            for apattern,cfactor in fillschemePatterns.items():
                if re.match(apattern,fillschemeStr):
                    Afterglow=cfactor
        driftterm=1.0
        if drift and intglumi:
            driftterm=1.0+drift*intglumi
        result=a0*Afterglow/(1+a1*avglumi+a2*avglumi*avglumi)*driftterm
        return result
    
def normFunctionCaller(funcName,*args,**kwds):
    fac=normFunctionFactory()
    try:
        myfunc=getattr(fac,funcName,None)
    except AttributeError:
        print '[ERROR] unknown correction function '+funcName
        raise
    if callable(myfunc):
        return myfunc(*args,**kwds)
    else:
        raise ValueError('uncallable function '+funcName)
if __name__ == '__main__':
    #sim run 176796,ls=6
    luminonorm=0.5061*1.0e3
    intglumi=3.309 #/fb
    nBXs=1331
    constParams={'a0':1.0}    
    argvals=[luminonorm,intglumi,nBXs,0.0,0.0]
    print 'no correction lumi in Hz/ub ',luminonorm*normFunctionCaller('fPoly',*argvals,**constParams)
    polyParams={'a0':7.268,'a1':0.063,'a2':-0.0037,'drift':0.01258,'c1':6.37,'afterglow':'[(213,0.992), (321,0.99), (423,0.988), (597,0.985), (700,0.984), (873,0.981), (1041,0.979), (1179,0.977),(1317,0.975)]'}
    print 'poly corrected lumi in Hz/ub',luminonorm*normFunctionCaller('fPoly',*argvals,**polyParams)
    polyParams={'a0':7.268,'a1':0.063,'a2':-0.0037,'drift':0.0,'c1':6.37,'afterglow':'[(213,0.992), (321,0.99), (423,0.988), (597,0.985), (700,0.984), (873,0.981), (1041,0.979), (1179,0.977),(1317,0.975)]'}
    print 'poly corrected without drift in Hz/ub ',luminonorm*normFunctionCaller('fPoly',*argvals,**polyParams)
    constParams={'a0':7.268}
    print 'const corrected lumi in Hz/ub',luminonorm*normFunctionCaller('fPoly',*argvals,**constParams)
