#!/usr/bin/env python

import FWCore.ParameterSet.Config as cms


def lumiList( json ):
    import FWCore.PythonUtilities.LumiList as LumiList
    myLumis = LumiList.LumiList(filename = json ).getCMSSWString().split(',')
    return myLumis

def applyJSON( process, json ):  

    # import PhysicsTools.PythonAnalysis.LumiList as LumiList
    # import FWCore.ParameterSet.Types as CfgTypes
    # myLumis = LumiList.LumiList(filename = json ).getCMSSWString().split(',')

    myLumis = lumiList( json )
    
    import FWCore.ParameterSet.Types as CfgTypes
    process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
    process.source.lumisToProcess.extend(myLumis)

    # print process.source.lumisToProcess

if __name__ == '__main__':
    
    import sys
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.usage = "%prog <json>\nPrints the selected lumi sections in a JSON file in EDM format."
    
    (options,args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        sys.exit(1)

    json = args[0]

    import pprint
    lumis = lumiList(json)
    pprint.pprint( lumis )
    
