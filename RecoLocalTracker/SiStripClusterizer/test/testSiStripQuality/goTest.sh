#!/bin/sh


eval `scramv1 runtime -sh`

echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Run with Fake SiStripQuality \n&&&&&&&&&&&&&&&&&&&&&&&\n"
cmsRun testClusterizer_FakeQuality.cfg > Fake.out
cat Fake.out | awk '$0~/Seed/{aprint=1} {if(aprint==1) print $0} $0~/Cluster accepted/{aprint=0} $0~/Cluster rejected/{aprint=0}' > Fake.ClusterList.out


echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Extracting bad components \n&&&&&&&&&&&&&&&&&&&&&&&\n"

Number_BadModules=1
Number_BadApv=4
Number_BadChannels=10


cat Fake.ClusterList.out | grep -A1 "Strips on the" | grep detID  | awk 'BEGIN{counter=0;apv=-1;detid=-1}{if(counter<c){counter++;print $2" "$4;detid=$2;apv=int($4/128)}else if(counter<b+c){if(detid!=$2 && apv!=int($4/128)){counter++;apv=int($4/128);print $2" "$4;detid=$2}}else if(counter<a+b+c){if(detid!=$2){detid=$2;counter++;print $2" "$4}}}' a=$Number_BadModules b=$Number_BadApv c=$Number_BadChannels > Fake.DetId_Strip_List.out


C=`cat Fake.DetId_Strip_List.out | head -$Number_BadChannels | awk ' BEGIN{detid=-1;a="";flag=0} func printFunc(val){if(flag){print ","}; print "{ uint32 BadModule = "detid" vuint32 BadChannelList = {"val"}}" ; flag=1}{if(detid==-1){detid=$1;a=$2;} if(detid!=$1){printFunc(a); detid=$1;  a=$2}else{a=sprintf("%s , %d",a,$2);} } END{printFunc(a)}'`

echo
echo $C

let tot=$Number_BadChannels+$Number_BadApv

B=`cat Fake.DetId_Strip_List.out | head -$tot | tail -$Number_BadApv | awk '{counter++;if(counter!=1){print ","};print "{ uint32 BadModule = "$1 " vuint32 BadApvList = {"int($2/128)"} }" }'`

echo
echo $B

let tot=$Number_BadChannels+$Number_BadApv+$Number_BadModules

A=`cat Fake.DetId_Strip_List.out | head -$tot | tail -$Number_BadModules | awk '{counter++;if(counter!=1){print ","};print $1 }'` 

echo
echo $A 


echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Create DB \n&&&&&&&&&&&&&&&&&&&&&&&\n"
rm -f dbfile.db
cmscond_bootstrap_detector.pl --offline_connect sqlite_file:dbfile.db --auth /afs/cern.ch/cms/DB/conddb/authentication.xml STRIP


echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Fill DB \n&&&&&&&&&&&&&&&&&&&&&&&\n"

echo -e "create cfg from template"
cat SiStripBadChannel.tpl | sed -e "s@insert_BadModuleList@`echo $A`@" -e "s@insert_BadApvList@`echo $B`@" -e "s@insert_BadChannelList@`echo $C`@" > SiStripBadChannel.cfg
cat SiStripBadApv.tpl | sed -e "s@insert_BadModuleList@`echo $A`@" -e "s@insert_BadApvList@`echo $B`@" -e "s@insert_BadChannelList@`echo $C`@" > SiStripBadApv.cfg
cat SiStripBadModule.tpl | sed -e "s@insert_BadModuleList@`echo $A`@" -e "s@insert_BadApvList@`echo $B`@" -e "s@insert_BadChannelList@`echo $C`@" > SiStripBadModule.cfg

echo "cmsRun SiStripBadModule.cfg > fillDb_module.out"
cmsRun SiStripBadModule.cfg > fillDb_module.out

echo "cmsRun SiStripBadApv.cfg > fillDb_apv.out"
cmsRun SiStripBadApv.cfg > fillDb_apv.out

echo "cmsRun SiStripBadChannel.cfg > fillDb_channel.out"
cmsRun SiStripBadChannel.cfg > fillDb_channel.out


echo "cmsRun testSiStripQualityESProducer.cfg"
cmsRun testSiStripQualityESProducer.cfg

echo -e "\n&&&&&&&&&&&&&&&&&&&&&&&\n Run with Real SiStripQuality \n&&&&&&&&&&&&&&&&&&&&&&&\n"
cmsRun testClusterizer_RealQuality.cfg > Real.out
cat Real.out | awk '$0~/Seed/{aprint=1} {if(aprint==1) print $0} $0~/Cluster accepted/{aprint=0} $0~/Cluster rejected/{aprint=0}' > Real.ClusterList.out

sdiff Real.ClusterList.out Fake.ClusterList.out > thediff.out

echo "look at the thediff.out file, or with emacs"

echo "emacs Real.ClusterList.out Fake.ClusterList.out"
