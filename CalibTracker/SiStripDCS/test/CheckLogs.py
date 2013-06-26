#! /usr/bin/env python
#Quick and dirty script to "analyze" the log of the CheckAllIOVs.py to validate
#plotting macro ExtractTrends.C
import sys
#print "Opening logfile %s"%sys.argv[1]
log=open(sys.argv[1])
#log=open("CheckAllIOVs_11_12_122009_2033_800.log","r")
HVOff=0
LVOff=0
for line in log:
    if "start" in line:
        print "%s modules with HV off"%HVOff
        print "%s modules with LV off\n"%LVOff 
        print line
        HVOff=0
        LVOff=0
    else:
        try:
            (HVStatus,LVStatus)=line.rstrip().split()[1:]
            if HVStatus=="OFF":
                HVOff=HVOff+1
            if LVStatus=="OFF":
                LVOff=LVOff+1
        except:
            print "Line read did not contain the IOV or the Voltage status:\n%s"%line
if HVOff or LVOff: #Catch the case of the last IOV when there are no exception lines ;)
    print "%s modules with HV off"%HVOff
    print "%s modules with LV off"%LVOff
