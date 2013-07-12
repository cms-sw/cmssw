#
# Define Ecal convenience functions
# author Stefano Argiro
# version $Id$
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
