#! python3

from __future__ import print_function

### this script takes a file with a list of IOVs and subdir name where separated xmls files are located
### the xml files have the iov start runno in the name and in the body (iov node of the xml) which should match
### for large xml with multiple IOVs we preferred to break into single-iov xmls to check for identical consecutive payloads
### for each new IOV start cmsRun write... is called to update the sqlite file
### the resulting sqlite file has a table of several IOV start and payloads

import subprocess
import sys

iovs_file = open(sys.argv[1])
path="."
if len(sys.argv)>2:
    path=sys.argv[2]
iovs_list = iovs_file.readlines()
iovs_list.sort()
iovs_file.close()

for i in range(0,len(iovs_list)):
    runno = iovs_list[i].strip()
    if i >0 and runno != "286693" and runno != "309055": ## runs where rpix change
        previov = iovs_list[i-1].strip()
        a= subprocess.check_output(
            "diff "+path+"/real_alignment_iov"+runno+".xml "+path+"/real_alignment_iov"+previov+".xml  | wc"
            ,shell=True)
        if int(a.split()[0] ) ==4 :
            #print("skipping file of IOV "+runno)
            continue
    print("Processing payload for IOV start "+runno)
    print (subprocess.check_output(
        "cmsRun write-ctpps-rprealalignment_table_cfg.py "+runno+"  "+path
        ,shell=True )
        )

print("finished")
