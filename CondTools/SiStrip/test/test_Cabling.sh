#!/bin/bash

# set env
eval `scramv1 runtime -sh`
kinit

# initial checkout
cd $CMSSW_BASE/src/
addpkg CalibFormats/SiStripObjects
addpkg CalibTracker/SiStripESProducers
addpkg CondFormats/SiStripObjects
addpkg CondTools/SiStrip

# temp patches
cvs update -A CalibFormats/SiStripObjects/interface/SiStripFecCabling.h
cvs update -A CalibFormats/SiStripObjects/interface/SiStripDetCabling.h
cvs update -A CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h
cvs update -A CalibFormats/SiStripObjects/src/SiStripFecCabling.cc
cvs update -A CalibFormats/SiStripObjects/src/SiStripDetCabling.cc
cvs update -A CalibFormats/SiStripObjects/src/SiStripRegionCabling.cc
cvs update -A CondTools/SiStrip/plugins/SiStripFedCablingBuilder.h
cvs update -A CondTools/SiStrip/plugins/SiStripFedCablingReader.h
cvs update -A CondTools/SiStrip/plugins/SiStripFedCablingBuilder.cc
cvs update -A CondTools/SiStrip/plugins/SiStripFedCablingReader.cc
cvs update -A CondTools/SiStrip/test/SiStripFedCablingBuilder_cfg.py
cvs update -A CondTools/SiStrip/test/SiStripFedCablingReader_cfg.py
cvs update -A CondTools/SiStrip/test/test_Cabling.sh

# old structure, old interface
cd $CMSSW_BASE/src/
NAME="SiStripFedCabling.h."`date +%s`
mv CondFormats/SiStripObjects/interface/SiStripFedCabling.h CondFormats/SiStripObjects/interface/$NAME
cvs co -r V13-04-06 CondFormats/SiStripObjects
showtags

# build
scramv1 b distclean
eval `scramv1 runtime -sh`
scramv1 b 

# run
cd $CMSSW_BASE/src/CondTools/SiStrip/test/
rm *.db
cmsRun SiStripFedCablingBuilder_cfg.py 
cmsRun SiStripFedCablingReader_cfg.py 
mv cablingBuilder.log cablingBuilder1.log
mv cablingReader.log cablingReader1.log

diff cablingBuilder1.log cablingReader1.log >& diffBuilder1Reader1.log

# new structure, old interface
cd $CMSSW_BASE/src/
cvs co -r V14-00-03 CondFormats/SiStripObjects
showtags

# build
scramv1 b distclean
eval `scramv1 runtime -sh`
scramv1 b 

# run
cd $CMSSW_BASE/src/CondTools/SiStrip/test/
rm *.db
cmsRun SiStripFedCablingBuilder_cfg.py 
cmsRun SiStripFedCablingReader_cfg.py 
mv cablingBuilder.log cablingBuilder2.log
mv cablingReader.log cablingReader2.log

diff cablingBuilder2.log cablingReader2.log >& diffBuilder2Reader2.log
diff cablingBuilder1.log cablingBuilder2.log >& diffBuilder1Builder2.log
diff cablingReader1.log cablingReader2.log >& diffReader1Reader2.log

# new structure, new interface
cd $CMSSW_BASE/src/
cvs co -r V14-01-00 CondFormats/SiStripObjects
showtags

# build
scramv1 b distclean
eval `scramv1 runtime -sh`
scramv1 b 

# run
cd $CMSSW_BASE/src/CondTools/SiStrip/test/
rm *.db
cmsRun SiStripFedCablingBuilder_cfg.py 
cmsRun SiStripFedCablingReader_cfg.py 
mv cablingBuilder.log cablingBuilder3.log
mv cablingReader.log cablingReader3.log

diff cablingBuilder3.log cablingReader3.log >& diffBuilder3Reader3.log
diff cablingBuilder2.log cablingBuilder3.log >& diffBuilder2Builder3.log
diff cablingReader2.log cablingReader3.log >& diffReader2Reader3.log

# end
echo
echo
echo "==========>>>>> Done! <<<<<=========="

