from IMProv.IMProvQuery import IMProvQuery
from IMProv.IMProvLoader import loadIMProvFile

# script to parse Crab XML report obtained with crab -status
#
# needs to be run in crab_..../share directory after
# crab -status -USER.xml_report=RReport.xml
#
# status codes are explained in
# CRAB_2_2_X/ProdCommon/ProdCommon/BossLite/Scheduler/GLiteLBQuery.py
# i.e.
#
#    statusMap = {
#        'Undefined':'UN',
#        'Submitted':'SU',
#        'Waiting':'SW',
#        'Ready':'SR',
#        'Scheduled':'SS',
#        'Running':'R',
#        'Done':'SD',
#        'Cleared':'E',
#        'Aborted':'A',
#        'Cancelled':'K',
#        'Unknown':'UN',
#        'Done(failed)':'DA'
#                }
# more status code come from CrabServer and are here
# https://twiki.cern.ch/twiki/bin/view/CMS/TaskTracking#Notes
#
    
# map from new (left) to old (right) code
statusMap = {
    'Undefined':'U',
    'Created':'C',
    'Submitting':'B',
    'Submitted':'B',
    'Waiting':'S',
    'Ready':'S',
    'Scheduled':'S',
    'Running':'R',
    'Done':'D',
    'Done (Failed)':'D',
    'Cleared':'Y',
    'Retrieved':'Y',
    'Killing':'K',
    'Killed':'K',
    'CannotSubmit':'A',
    'Aborted':'A',
    'NotSubmitted':'A',
    'Cancelled':'K',
    'Cancelled by user':'K',
    'Unknown':'U',
    'Done(failed)':'D'   
    }

def queryStatusXML(filename):

    try:
        report = loadIMProvFile(filename)
    except StandardError, ex:
        msg = "Error reading CRAB Status Report: %s\n" % filename
        msg += str(ex)
        raise RuntimeError, msg

    query = IMProvQuery("Task/TaskJobs/Job/RunningJob")
    Jobs = query(report)

    return Jobs
  
def printCrabStatusFromReport(filename):
    Jobs = queryStatusXML(filename)
    print "Crab Id: StatusScheduler | Status | ProcessStatus | State | GridId |"
    for j in Jobs:
        crabId = int(j.attrs.get("jobId",None))
        statusScheduler = str(j.attrs.get("statusScheduler",None))
        status = str(j.attrs.get("status",None))
        processStatus = str(j.attrs.get("processStatus",None))
        state  = str(j.attrs.get("state",None))
        gridId = str(j.attrs.get("schedulerId",None))

        # print crabId, processStatus, statusScheduler, status, state, gridId
        print "%d : %s | %s | %s | %s | %s " % (crabId,statusScheduler,status,processStatus,state,gridId)

        """
        # remap into old status codes from BOSS for use in JobRobot
        if state == 'SubRequested' : status = 'Submitting'
        if state == 'Terminated' : status = 'Done'
        ost = statusMap[statusScheduler]
        # old bossId starts from 0
        bossId = crabId-1
        print ("%d|%1s||%s|%d|") % (bossId, ost, gridId, crabId)
        """

def crabStatusFromReport(filename):
    Jobs = queryStatusXML(filename)
    #statusField = "statusScheduler"
    #statusField = "state"
    statusField = "status"
    summary = {}
    nJobs = 0
    for j in Jobs:
        jobStatus = str(j.attrs.get(statusField,None))
        if summary.has_key(jobStatus): summary[jobStatus] += 1
        else: summary[jobStatus] = 1
        nJobs += 1

    for item in summary: summary[item] = 100.*summary[item]/nJobs

    return summary

