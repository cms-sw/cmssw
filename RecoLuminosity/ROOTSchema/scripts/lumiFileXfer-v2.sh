#! /bin/sh
#
# Author Y. Guo
# Date June11 08
# 
# This script is used to call  ~tier0/scripts/injectFileIntoTransferSystem.pl script automatically.
#
# Example to run Storage Manage script to notify T0 transfer system 
# ~tier0/scripts/injectFileIntoTransferSystem.pl --filename=CMS_LUMI_RAW_20080508_000043541_0.root 
# --path=/cms/mon/data/dqm/lumi/root/store/lumi/200805 --filesize=71443592 --runnumber=43541 --lumisection=0 
# --type=lumi --hostname=srv-c2c06-02.cms --appversion=CVS_HEAD --appname=LumiROOTFileWriter 
#
usage="Usage: $0 -d dirname [-f \"filename1 filename2 ...\"]"
while getopts ":d:f:" opt;do
    case $opt in
      d  ) dirname=$OPTARG;;
           #echo $OPTIND;
           #echo $dirname;;
      f  ) files=$OPTARG;;
           #echo $OPTIND;
           #echo $files;; 
      \? ) echo $usage
           exit 1;; 
     esac
done  

#the max command line input is 4
if [ -n "$5" ]; then
    echo $usage
    exit 1;
fi

run=""
ls=""
size=""
type="lumi"
#type="lumi-vdm" #temporarily changed for VdM scans

#this function takes a file name as input
function getData()
{   
    thisFile=$1 
    size=`stat -c %s $thisFile`
    runLS=${thisFile/CMS\_LUMI\_RAW\_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]\_/""}
    rL=${runLS/\.root/""}
    run=${rL/\_*/""}
    ls=${rL/*\_/""}
}
#
if [ $OPTIND = 3 ]; then
    #echo $dirname
    cd $dirname
    #ls -l $dirname
    xferFiles=`ls *.root`
    #echo $xferFiles
    for theFile in $xferFiles; do
      getData $theFile
      #echo $theFile $dirname $size $run $ls $type
      echo $theFile  $size 
      ~tier0/scripts/injectFileIntoTransferSystem.pl --filename=$theFile \
           --path=$dirname --filesize=$size --runnumber=$run --lumisection=$ls \
           --type=$type --hostname=srv-c2c06-02.cms --appversion=CVS_HEAD --appname=LumiROOTFileWriter 
    done
elif [ $OPTIND = 5 ]; then
    #echo $5;
    #echo $dirname $files;
    cd $dirname
    for theFile in $files;do
        getData $theFile
        #echo $theFile $dirname $size $run $ls $type
        echo $theFile $size 
        ~tier0/scripts/injectFileIntoTransferSystem.pl --filename=$theFile \
           --path=$dirname --filesize=$size --runnumber=$run --lumisection=$ls \
           --type=$type --hostname=srv-c2c06-02.cms --appversion=CVS_HEAD --appname=LumiROOTFileWriter
    done
else
    echo $usage
    exit 1
fi
