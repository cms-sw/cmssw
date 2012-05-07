#!/usr/bin/env python

"""
_recoTLR_

Util to retrieve the appropriate set of reco top-level patches
from the Configuration/GlobalRuns package based on the release name

"""


# two helper functions
def getRelease():
    import os

    try:
        release=os.environ['CMSSW_VERSION']
    except:
        msg='No CMSSW_VERSION envvar defined'
        raise RuntimeError,msg

    sp=release.split('_')
    if len(sp)<2:
        raise RuntimeError,'Unexpected release name'+release
    
    return ''.join(sp[1:3])+'X'

    #__import__(moduleName,globals(),locals(),[scenarioName])
    #except ImportError, ex:


def getCustomProcess(process,myname):
    rel=getRelease()
    try:
        _temp=__import__('Configuration.GlobalRuns.reco_TLR_'+rel,globals(),locals(),[myname])
    except ImportError,ex:
        msg= 'Unable to import reco TLR configuration ' + str(ex)
        raise RuntimeError,msg
    
    return getattr(_temp,myname)(process)
    

##############################################################################
import sys
def customisePPData(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)

def customiseVALSKIM(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)
        
##############################################################################
def customisePPMC(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)

##############################################################################
def customiseCosmicData(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)

##############################################################################
def customiseCosmicMC(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)

##############################################################################
def customiseExpress(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)

##############################################################################
def customisePrompt(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)

##############################################################################
def customiseExpressHI(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)

##############################################################################
def customisePromptHI(process):
    myname=sys._getframe().f_code.co_name
    return getCustomProcess(process,myname)
