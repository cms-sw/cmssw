#!/bin/sh

eval `scramv1 runtime -sh`
cmsRun SiPixelInclusiveSmearedConditionsReader.cfg
cmsRun SiPixelInclusiveReader.cfg
root check_SiPixelCondObjects.C
