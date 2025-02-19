import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the CSC geometry model
# with corrections for real MTCC geometry Oct-2006
#
from Geometry.CSCGeometry.cscGeometry_cfi import *
CSCGeometryESModule.useCentreTIOffsets = True

