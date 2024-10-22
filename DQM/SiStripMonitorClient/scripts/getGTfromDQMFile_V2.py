#!/usr/bin/env python3
from math import *
from ROOT import TFile, TObject, TTree
from array import array
from ROOT import gDirectory
import sys
import os
from __future__ import print_function

def getGTfromDQMFile(DQMfile, RunNumber, globalTagVar):
    if not os.path.isfile(DQMfile):#    print "Error: file", DQMfile, "not found, exit" 
        sys.exit(0)
    thefile = TFile( DQMfile )
    globalTagDir = 'DQMData/Run ' + RunNumber + '/Info/Run summary/CMSSWInfo'
    if not gDirectory.GetDirectory( globalTagDir ):
        #    print "Warning: globalTag not found in DQM file"
        sys.exit(0)    
    keys = gDirectory.GetDirectory( globalTagDir ).GetListOfKeys()
    key = keys[0]
    globalTag = ''
    while key:
        obj = key.ReadObj()
        if globalTagVar in obj.GetName():
            globalTag = obj.GetName()[len("<"+globalTagVar+">s="):-len("</"+globalTagVar+">")]
            break
        key = keys.After(key)
    if len(globalTag) > 1:
        if globalTag.find('::') >= 0: 
            print(globalTag[0:globalTag.find('::')])
        else:
            print(globalTag)
    return globalTag

