#!/usr/bin/env python

import sys
import os
import string

def computeAvg(mylist):
    avg=0
    if (len(mylist)==0):
        return avg
    for i in mylist:
        avg=avg+i
    avg=avg/len(mylist)
    return avg


myfile=sys.argv[1:]
print "\n\n"

for F in myfile:
    print "Results for file %s:"%F
    print "------- --- ---- --------"

    mylines=open(F,'r').readlines()

    myTimers={}

    for i in mylines:
        if string.find(i,"TIMER::")>-1:
            temp=string.strip(i)
            #print temp
            temp=string.split(temp,"->")
            #print "..%s.."%temp[1]
            val=string.atof(temp[1])
            key=string.split(temp[0],"::")[1]
            key=string.strip(key)
            if key not in myTimers.keys():
                myTimers[key]=[val]
            else:
                myTimers[key].append(val)

    mykeys=myTimers.keys()
    mykeys.sort()
    myTime={}
    mySubTime={}
    sum=0
    sumsub=0
    for i in mykeys:
        temp=computeAvg(myTimers[i])
        if i[0:4]=="Hcal":
            mySubTime[temp]=i
            sumsub=sumsub+temp
        else:
            myTime[temp]=i
            sum=sum+temp

    mykeys=myTime.keys()
    mykeys.sort()
    mykeys.reverse()

    for i in mykeys:

        print "%.5f :"%i,
        print "  %s"%(myTime[i])

    print
    print "%.5f :  TOTAL TIME"%sum
    print "\n"
    mysubkeys=mySubTime.keys()
    mysubkeys.sort()
    mysubkeys.reverse()
    for i in mysubkeys:
        print "%.5f : \t%s"%(i,mySubTime[i])
    print "%.5f : TOTAL SUBPROCESS TIME"%sum
    print "\n\n"

