#
# Define Ecal convenience functions
# author Stefano Argiro
# version $Id: EcalPyUtils.py,v 1.1 2009/07/09 22:56:14 argiro Exp $
#

from pluginEcalPyUtils import *

def unhashEBIndex(idx) :
    
    tmp= hashedIndexToEtaPhi(idx)
    return tmp[0],tmp[1]

def unhashEEIndex(idx) :
    
    tmp=hashedIndexToXY(idx)
    return tmp[0],tmp[1],tmp[2]

def fromXML(filename):
    barrel=barrelfromXML(filename)
    endcap=endcapfromXML(filename)
    return barrel,endcap
