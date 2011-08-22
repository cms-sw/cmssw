#!/usr/bin/python

import time
import os

def runTheAlarm(file, type):

    currentTime = time.time()
    minModulesOn = 200
    maxModulesOn = 15148 - 200

    timeDiff = 0
    firstIOV = True

    previousIOV = 0
    for item in file.read().split("]"):
        # print item
        if( len(item.split("[")) > 1 ):
            element = item.split("[")[1]
            if element.find(", ") != -1:
                IOVtime = int(element.split(", ")[0])/1000
                if firstIOV == False:
                    firstIOV == True
                    previousIOVtime = IOVtime
                else:
                    # print currentTime
                    # 7200 are two hours in seconds = 2*60*60
                    # if (currentTime - IOVtime) > 0:
                    if (currentTime - IOVtime) < 7200:
                        modulesOn = int(element.split(", ")[1])
                        if modulesOn > minModulesOn and modulesOn < maxModulesOn:
                            # print "warning, modules on are", modulesOn, "at", time.localtime(IOVtime)
                            timeDiff += IOVtime - previousIOV
                            previousIOV = IOVtime
                        else:
                            # print "modulesOn =", modulesOn
                            # 10 minutes = 600 seconds
                            if timeDiff > 600:
                                print "WARNING: Bad IOV for", type, "lasting for =", timeDiff, "seconds and ending at", time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(IOVtime))
                                os.system("echo \"WARNING: Bad IOV for "+type+" lasting for = "+str(timeDiff)+" seconds and ending at "+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(IOVtime))+"\" > mail.txt")
                                os.system("mail -s \"DCS WARNING\" \"marco.de.mattia@cern.ch\" < \"mail.txt\"")
                                os.system("mail -s \"DCS WARNING\" \"gabriele.benelli@cern.ch\" < \"mail.txt\"")
                                timeDiff = 0
                                previousIOV = IOVtime
    if timeDiff > 300:
        print "WARNING: timeDiff =", timeDiff
        os.system("echo \"WARNING: Bad IOV for "+type+" lasting for = "+str(timeDiff)+" seconds and ending at "+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(IOVtime))+"\" > mail.txt")
        os.system("mail -s \"DCS WARNING\" \"marco.de.mattia@cern.ch\" < \"mail.txt\"")
        os.system("mail -s \"DCS WARNING\" \"gabriele.benelli@cern.ch\" < \"mail.txt\"")


fileHV = open("/afs/cern.ch/cms/tracker/sistrcalib/DCSTrend/oneMonth_hv.js")
fileLV = open("/afs/cern.ch/cms/tracker/sistrcalib/DCSTrend/oneMonth_hv.js")

runTheAlarm(fileHV, "HV")
runTheAlarm(fileLV, "LV")

