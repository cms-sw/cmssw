#!/bin/sh

eval `scramv1 runtime -sh`
cmsRun SiPixelInclusiveSmearedConditionsReader.cfg
cmsRun SiPixelInclusiveReader.cfg
rm -rf tempin
echo ".q" > tempin
root check_SiPixelCondObjects.C < tempin
rm -rf tempin
