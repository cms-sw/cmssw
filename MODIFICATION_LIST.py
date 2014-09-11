#!/usr/bin/env python

from subprocess import Popen
import os

tagList = [
#'Alignment/CommonAlignment',
'Alignment/ReferenceTrajectories',
'Alignment/TrackerAlignment',
'CalibTracker/SiStripESProducers',
'CalibTracker/SiStripLorentzAngle',
'DataFormats/TrackerRecHit2D',
'DataFormats/TrackingRecHit',
'DataFormats/GeometryCommonDetAlgo',
'DataFormats/GeometrySurface',
'Geometry/TrackingGeometryAligner',
'FastSimulation/TrackingRecHitProducer',
'FastSimulation/TrajectoryManager',
'Fireworks/Geometry',
'Geometry/CSCGeometry',
'Geometry/CommonDetUnit',
'Geometry/DTGeometry',
'Geometry/GEMGeometry',
'Geometry/RPCGeometry',
'Geometry/TrackerGeometryBuilder',
'RecoLocalTracker/SiStripRecHitConverter',
'RecoMuon/TransientTrackingRecHit',
'RecoTracker/DebugTools',
'RecoTracker/MeasurementDet',
'RecoTracker/TransientTrackingRecHit',
'TrackingTools/TransientTrackingRecHit']

#for subsystem in tagList:
#    Popen('git cms-addpkg '+subsystem,shell=True).wait()
#    #Popen('cp -r /afs/cern.ch/cms/sw/ReleaseCandidates/vol0/slc6_amd64_gcc481/cms/cmssw-patch/CMSSW_7_2_X_2014-08-23-0200/src/'+subsystem+' .',shell=True).wait()
#    print 'git cms-addpkg '+subsystem

fileModList = [
'DataFormats/GeometrySurface/interface/LocalErrorExtended.h',
'DataFormats/TrackingRecHit/src/AlignmentPositionError.cc',
'Alignment/CommonAlignment/src/AlignableDetUnit.cc',
'Alignment/CommonAlignment/src/AlignableDet.cc',
'Alignment/CommonAlignment/src/AlignableModifier.cc',
'Alignment/CommonAlignment/src/AlignableBeamSpot.cc',
'Alignment/CommonAlignment/src/AlignableComposite.cc',
'Alignment/TrackerAlignment/test/ApeAdder.cpp'
#'DataFormats/GeometryCommonDetAlgo/interface/GlobalErrorBaseExtended.h',
#'DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h',
#'DataFormats/GeometrySurface/interface/LocalErrorBaseExtended.h',
#'DataFormats/GeometryCommonDetAlgo/interface/LocalError.h',
#'DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h', #to be changed
#'DataFormats/TrackingRecHit/interface/AlignmentPositionError.h',       #to be changed
#'Geometry/CommonDetUnit/src/TrackerGeomDet.cc',
#'RecoMuon/TransientTrackingRecHit/src/MuonTransientTrackingRecHit.cc', #to be changed
#'Geometry/TrackingGeometryAligner/interface/GeometryAligner.h']        #to be changed
#in addition modify the GeometryBuilder. but not permanently 
]

INIT_PATH = '/afs/cern.ch/user/a/asvyatko/APEStudyDev_Repo/CMSSW_7_2_X_2014-08-23-0200/src/'

for file in fileModList:
    #Popen('cp '+INIT_PATH+file+' '+file,'shell=True').wait()
    print 'cp '+INIT_PATH+file+' '+file

#RecoMuon/TransientTrackingRecHit/src/MuonTransientTrackingRecHit.cc
#Geometry/CommonDetUnit/src/GeomDet.cc
