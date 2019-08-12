from __future__ import print_function
import re
import os
import errno
shortcuts = {}

# regex matching on key, replacement of groups on value
# implement any other shortcuts that you want to use
#sources
shortcuts["mp([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp{0}/jobData/jobm/alignments_MP.db"
shortcuts["mp([0-9]*)_jobm([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/mp{0}/jobData/jobm{1}/alignments_MP.db"
shortcuts["sm([0-9]*)_iter([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/sm{0}/alignments_iter{1}.db"
shortcuts["um([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/um{0}/jobData/jobm/um{0}.db"
shortcuts["um([0-9]*)_jobm([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/MP/MPproduction/um{0}/jobData/jobm{1}/um{0}.db"
shortcuts["hp([0-9]*)_iter([0-9]*)"] = "sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/HipPy/alignments/hp{0}/alignments_iter{1}.db"
shortcuts["prod"] = "frontier://FrontierProd/CMS_CONDITIONS"

# Exact numbers don't really matter, but it is important that each one has a unique
# number, so that states are distinguishable
STATE_NONE = -1
STATE_ITERATION_START=0
STATE_BJOBS_WAITING=1
STATE_BJOBS_DONE=2
STATE_BJOBS_FAILED=12
STATE_MERGE_WAITING=3
STATE_MERGE_DONE=4
STATE_MERGE_FAILED=14
STATE_SUMMARY_WAITING=5
STATE_SUMMARY_DONE=6
STATE_SUMMARY_FAILED=16
STATE_LOCAL_WAITING=7
STATE_LOCAL_DONE=8
STATE_LOCAL_FAILED=18
STATE_FINISHED=9

status_map = {}
status_map[STATE_NONE] = "none"
status_map[STATE_ITERATION_START] = "starting iteration"
status_map[STATE_BJOBS_WAITING] = "waiting for jobs"
status_map[STATE_BJOBS_DONE] = "jobs finished"
status_map[STATE_BJOBS_FAILED] = "jobs failed"
status_map[STATE_MERGE_WAITING] = "waiting for merging"
status_map[STATE_MERGE_DONE] = "merging done"
status_map[STATE_MERGE_FAILED] = "merging failed"
status_map[STATE_SUMMARY_WAITING] = "waiting for APE determination"
status_map[STATE_SUMMARY_DONE] = "APE determination done"
status_map[STATE_SUMMARY_FAILED] = "APE determination failed"
status_map[STATE_LOCAL_WAITING] = "waiting for APE saving"
status_map[STATE_LOCAL_DONE] = "APE saving done"
status_map[STATE_LOCAL_FAILED] = "APE saving failed"
status_map[STATE_FINISHED] = "finished"

records = {}
records["Alignments"] = "TrackerAlignmentRcd"
records["TrackerAlignment"] = "TrackerAlignmentRcd"
records["Deformations"] = "TrackerSurfaceDeformationRcd"
records["TrackerSurfaceDeformations"] = "TrackerSurfaceDeformationRcd"
records["SiPixelTemplateDBObject"] = "SiPixelTemplateDBObjectRcd"
records["BeamSpotObjects"] = "BeamSpotObjectsRcd"


def ensurePathExists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def replaceAllRanges(string):
    if "[" in string and "]" in string:
        strings = []
        posS = string.find("[")
        posE = string.find("]")
        nums = string[posS+1:posE].split(",")
        expression = string[posS:posE+1]

        nums = string[string.find("[")+1:string.find("]")]
        for interval in nums.split(","):
            interval = interval.strip()
            if "-" in interval:
                lowNum = int(interval.split("-")[0])
                upNum = int(interval.split("-")[1])
                for i in range(lowNum, upNum+1):
                    newstring = string[0:posS]+str(i)+string[posE+1:]
                    newstring = replaceAllRanges(newstring)
                    strings += newstring
            else:
                newstring = string[0:posS]+interval+string[posE+1:]
                newstring = replaceAllRanges(newstring)
                strings += newstring
        return strings
    else:
        return [string,]


def replaceShortcuts(toScan):
    global shortcuts
    for key, value in shortcuts.items():
        match = re.search(key, toScan)
        if match and match.group(0) == toScan:
            return value.format(*match.groups())
    # no match
    return toScan

def loadConditions(dictionary):
    hasAlignmentCondition = False
    conditions = []
    for key, value in dictionary.items():
        if key.startswith("condition"):
            if len(value.split(" ")) > 1: # structure is "condition rcd:source tag"
                record = key.split(" ")[1]
                connect, tag = value.split(" ")
                if record == "TrackerAlignmentRcd":
                    hasAlignmentCondition = True
                conditions.append({"record":record, "connect":replaceShortcuts(connect), "tag":tag})
            else:
                # structure is "condition tag:source", so we have to guess rcd from the tag. might also be "condition tag1+tag2+...+tagN:source"
                global records
                connect = value.strip()
                tags = key.split(" ")[1]
                for tag in tags.split("+"):
                    foundTag = False
                    for possibleTag, possibleRcd in records.items():
                        if tag.startswith(possibleTag):
                            conditions.append({"record":possibleRcd, "connect":replaceShortcuts(connect), "tag":tag})
                            if possibleRcd == "TrackerAlignmentRcd":
                                hasAlignmentCondition = True
                            foundTag = True
                            break
                    if not foundTag:
                        print("Did not find a record corresponding to {} tag".format(tag))
                        exit()

    return conditions, hasAlignmentCondition 
