import re
def findUniqueSeed(hltPath,ExprStr):
    '''
    given a hltpath and its L1SeedExpression, find the L1 bit name
    can return None
    
    if hltPath contains the following, skip do not parse seed.
    
    FakeHLTPATH*, HLT_Physics*, HLT_*Calibration*, HLT_HFThreashold,
    HLT_MiniBias*,HLT_Random*,HLTTriggerFinalPath,HLT_PixelFED*

    parse hltpath contains at most 2 logics x OR y, x AND y, and return left val
    do not consider path containing operator NOT

    '''
    if re.match('FakeHLTPATH',hltPath)!=None :
        return None
    if re.match('HLT_Physics',hltPath)!=None :
        return None
    if re.match('HLT_[aA-zZ]*Calibration',hltPath)!=None :
        return None
    if re.match('HLT_[aA-zZ]*Threshold',hltPath)!=None :
        return None
    if re.match('HLT_MiniBias',hltPath)!=None :
        return None
    if re.match('HLT_Random',hltPath)!=None :
        return None
    if re.match('HLTriggerFinalPath',hltPath)!=None :
        return None
    if re.match('HLT_[aA-zZ]*FEDSize',hltPath)!=None :
        return None
    if ExprStr.find('(')!=-1 : #we don't parse expression with ()
        return None
    sep=re.compile('\sAND\s|\sOR\s')
    result=re.split(sep,ExprStr)
    if len(result)>2:
        return None
    for r in result:
        if r.find('NOT ')!=-1 : #we don't know what to do with NOT
            return None
    #the result we return the first one
    return result[0]

if __name__=='__main__':
    print findUniqueSeed("HLT_ForwardBSC","36 OR 37")
    print findUniqueSeed("HLT_HcalNZS_8E29","L1_SingleEG1 OR L1_SingleEG2 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10")
    print findUniqueSeed("HLT_ZeroBiasPixel_SingleTrack","L1_ZeroBias")
    
