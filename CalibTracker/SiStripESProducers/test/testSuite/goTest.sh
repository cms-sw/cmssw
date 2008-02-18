#!/bin/sh

eval `scramv1 runtime -sh`

mkdir TkMap

rm -f dbfile.db

echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Create DB \n&&&&&&&&&&&&&&&&&&&&&&&\n"
cmscond_bootstrap_detector.pl --offline_connect sqlite_file:dbfile.db --auth /afs/cern.ch/cms/DB/conddb/authentication.xml STRIP

echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Fill DB \n&&&&&&&&&&&&&&&&&&&&&&&\n"
rm -f *out
echo "cmsRun SiStripBadChannelBuilder_1.cfg      > bc_1.out"
cmsRun SiStripBadChannelBuilder_1.cfg      | grep -v '%MSG' > bc_1.out 
echo "cmsRun SiStripBadChannelBuilder_2.cfg      > bc_2.out"
cmsRun SiStripBadChannelBuilder_2.cfg      | grep -v '%MSG' > bc_2.out 
echo "cmsRun SiStripBadFiberBuilder_1.cfg        > bf_1.out" 
cmsRun SiStripBadFiberBuilder_1.cfg        | grep -v '%MSG' > bf_1.out 
echo "cmsRun SiStripBadModuleByHandBuilder_1.cfg > bm_1.out"
cmsRun SiStripBadModuleByHandBuilder_1.cfg | grep -v '%MSG' > bm_1.out
echo "cmsRun SiStripBadModuleByHandBuilder_2.cfg > bm_2.out"
cmsRun SiStripBadModuleByHandBuilder_2.cfg | grep -v '%MSG' > bm_2.out
echo "cmsRun SiStripBadModuleByHandBuilder_3.cfg > bm_3.out"
cmsRun SiStripBadModuleByHandBuilder_3.cfg | grep -v '%MSG' > bm_3.out


echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Read DB \n&&&&&&&&&&&&&&&&&&&&&&&\n"
echo "cmsRun testSiStripQualityESProducer.cfg > es.out"
cmsRun testSiStripQualityESProducer.cfg | grep -v '%MSG' > es.out

echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n IOV sequence \n&&&&&&&&&&&&&&&&&&&&&&&\n"
grep -B5 "produce called" es.out | grep Begin
