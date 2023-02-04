import FWCore.ParameterSet.Config as cms

# Full configuration for Tracker Geometry Comparison Tool
from Alignment.OfflineValidation.trackerGeometryCompare_cfi import trackerGeometryCompare as _trackerGeometryCompare
TrackerGeometryCompare = _trackerGeometryCompare.clone(
    fromDD4hep     = False,
    writeToDB      = False,
    outputFile     = 'output.root',
    setCommonTrackerSystem = 'NONE', ##must be "NONE" if you don't want to use this option
    detIdFlag      = False,
    detIdFlagFile  = 'blah.txt',
    weightById     = False,
    #	untracked vstring levels = {"PixelEndcap","PixelHalfBarrel","TID","HalfBarrel","Endcap","DetUnit"}
    levels         = ['Det'],
    weightBy       = 'DetUnit',
    weightByIdFile = 'blah2.txt',
    treeNameAlign  = 'alignTree',
    treeNameDeform = 'alignTreeDeformations',
    inputROOTFile1 = 'IDEAL',
    inputROOTFile2 = 'idealtracker2.root',
    moduleList     = 'moduleList.txt',
    surfDir        = '.'
)



