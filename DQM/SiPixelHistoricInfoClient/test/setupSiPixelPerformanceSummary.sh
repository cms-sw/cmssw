#!/bin/bash

eval `scramv1 ru -sh`

cd ${CMSSW_BASE}/src

cvs co -r ${CMSSW_VERSION} CondFormats/SiPixelObjects

cvs co -r V00-13-13 CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h
cvs co -r V00-13-13 CondFormats/SiPixelObjects/src/SiPixelPerformanceSummary.cc
cvs co -r V00-13-13 CondFormats/SiPixelObjects/xml/SiPixelPerformanceSummary_basic_0.xml
 
scramv1 build

cd - 
