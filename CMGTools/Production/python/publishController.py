#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## September 27th 2011

from CMGTools.Production.cmgdbToolsApi import CmgdbToolsApi
from datetime import *
import sys, re


class PublishController(object):
    """This class controls the interactions between a user and the CMGDB
    publishing platform"""
    def __init__(self, username, development=False):
        """Initialise CMGDB and set username of who is publishing

        'force' takes a boolean value which determines whether
        a lack of log file can be ignored
        """
        self.development = development
        self._username = username
        self._cmgdbAPI=CmgdbToolsApi(self.development)
        self._cmgdbAPI.connect()

    def cmgdbOnline(self):
        """Returns True if CMGDB is online and working"""
        if self._cmgdbAPI is not None: return True
        else: return False

    def cmgdbPublish(self, datasetDetails):
        """Publish dataset information to CMGDB, and return unique CMGDB dataset ID

        'datasetDetails' takes a dict object which contains all of the datasets details. This is a strictly defined stucture.
        """
        if self._cmgdbAPI is None:
            return None


        # See if cmgdb already has record of ds with sav
        datasetDetails['CMGDBID'] = self._cmgdbAPI.getDatasetIDWithName(datasetDetails['CMGDBName'])

        # If not add dataset
        if datasetDetails['CMGDBID'] is None:
            datasetDetails['CMGDBID'] = self._cmgdbAPI.addDataset(datasetDetails['CMGDBName'],
                                                                  datasetDetails['SampleName'],
                                                                  datasetDetails["LFN"],
                                                                  datasetDetails['FileOwner'],
                                                                  datasetDetails['ParentCMGDBID'],
                                                                  self._username)
        else:
            if datasetDetails['ParentCMGDBID'] is not None: 
                self._cmgdbAPI.setParentID(datasetDetails['CMGDBID'], 
                                           datasetDetails['ParentCMGDBID'])
        # Clear 4 tables relating to bad files & jobs, and missing & duplicate files
        self._cmgdbAPI.clearDatasetBadFiles(datasetDetails['CMGDBName'],
                                            datasetDetails['CMGDBID'])
        self._cmgdbAPI.clearDatasetMissingFiles(datasetDetails['CMGDBName'],
                                                datasetDetails['CMGDBID'])
        self._cmgdbAPI.clearDatasetBadJobs(datasetDetails['CMGDBName'],
                                           datasetDetails['CMGDBID'])


        if datasetDetails["TotalJobs"] is not None:
            self._cmgdbAPI.addTotalJobs(datasetDetails['CMGDBID'], 
                                        datasetDetails["TotalJobs"])
        if datasetDetails["TotalFilesMissing"] is not None:
            self._cmgdbAPI.addMissingFileNum(datasetDetails['CMGDBID'],
                                             datasetDetails["TotalFilesMissing"])
        if datasetDetails["TotalFilesGood"] is not None:
            self._cmgdbAPI.addGoodFileNum(datasetDetails['CMGDBID'],
                                          datasetDetails["TotalFilesGood"])
        if datasetDetails["TotalFilesBad"] is not None:
            self._cmgdbAPI.addBadFileNum(datasetDetails['CMGDBID'], 
                                         datasetDetails["TotalFilesBad"])
        for badJob in datasetDetails["BadJobs"]:
            self._cmgdbAPI.addBadJob(datasetDetails['CMGDBID'],
                                     badJob)
        for group_name in datasetDetails['FileGroups']:
            group_id = self._cmgdbAPI.addFileGroup(group_name,
                                                   datasetDetails['CMGDBID'])
            print "Group: %s added with ID: %d" % (group_name, 
                                                   group_id)
            if datasetDetails['FileGroups'][group_name]["BadFiles"] is not None:
                for badFile in datasetDetails['FileGroups'][group_name]["BadFiles"]:
                    self._cmgdbAPI.addBadFile(datasetDetails['CMGDBName'],
                                              datasetDetails['CMGDBID'],
                                              badFile.split('/')[-1],
                                              group_id)

            if datasetDetails['FileGroups'][group_name]["MissingFiles"] is not None:
                for missingFile in datasetDetails['FileGroups'][group_name]["MissingFiles"]:
                    self._cmgdbAPI.addMissingFile(datasetDetails['CMGDBName'],
                                                  datasetDetails['CMGDBID'],
                                                  missingFile.split('/')[-1],
                                                  group_id)

            if datasetDetails['FileGroups'][group_name]["NumberMissingFiles"] is not None:
                self._cmgdbAPI.addGroupMissingFileNum(group_id,
                                                      datasetDetails['FileGroups'][group_name]["NumberMissingFiles"])
            if datasetDetails['FileGroups'][group_name]["NumberBadFiles"] is not None:
                self._cmgdbAPI.addGroupBadFileNum(group_id,
                                                  datasetDetails['FileGroups'][group_name]["NumberBadFiles"])
            if datasetDetails['FileGroups'][group_name]["NumberGoodFiles"] is not None:
                self._cmgdbAPI.addGroupGoodFileNum(group_id,
                                                   datasetDetails['FileGroups'][group_name]["NumberGoodFiles"])
            if datasetDetails['FileGroups'][group_name]["PrimaryDatasetFraction"] is not None:
                self._cmgdbAPI.addGroupPrimaryDatasetFraction(group_id,
                                                              datasetDetails['FileGroups'][group_name]["PrimaryDatasetFraction"])
            if datasetDetails['FileGroups'][group_name]["SizeInTB"] is not None:
                self._cmgdbAPI.addDatasetSize(group_id,
                                              datasetDetails['FileGroups'][group_name]["SizeInTB"])
            if datasetDetails['FileGroups'][group_name]["FileEntries"] is not None:
                self._cmgdbAPI.addGroupFileEntries(group_id,
                                                   datasetDetails['FileGroups'][group_name]["FileEntries"])

        if datasetDetails["PrimaryDatasetFraction"] is not None:
            self._cmgdbAPI.addPrimaryDatasetFraction(datasetDetails['CMGDBID'],
                                                     datasetDetails["PrimaryDatasetFraction"])
        if datasetDetails["PrimaryDatasetEntries"] is not None:
            self._cmgdbAPI.addPrimaryDatasetEntries(datasetDetails['CMGDBID'],
                                                    datasetDetails["PrimaryDatasetEntries"])
        if datasetDetails["FileEntries"] is not None:
            self._cmgdbAPI.addFileEntries(datasetDetails['CMGDBID'],
                                          datasetDetails["FileEntries"])
        if datasetDetails["DirectorySizeInTB"] is not None:
            self._cmgdbAPI.addDirectorySize(datasetDetails['CMGDBID'],
                                            datasetDetails["DirectorySizeInTB"])

        # Add tags to CMGDB
        if datasetDetails['Tags'] is None or len(datasetDetails['Tags']) is 0:
            print "No tags could be added to CMGDB as none were found"
            return datasetDetails['CMGDBID']
        tagIDs = []

        # Create hash code for the tag set
        taghash = []
        for i in datasetDetails['Tags']:
            a=hash((i['package'],i['tag']))
            taghash.append(a)
        taghash.sort()
        endhash = hash(tuple(taghash))

        # check if tag set is already on CMGDB
        tagSetID = self._cmgdbAPI.getTagSetID(endhash)

        # If it isn't found, add the tags, and the tag set
        if tagSetID is None:
            if datasetDetails['Tags']:
                tagIDs
                for row in datasetDetails['Tags']:
                    tagID = self._cmgdbAPI.addTag(row["package"],
                                                  row["tag"])
                    if tagID is not None:
                        tagIDs.append(tagID)

                tagSetID = self._cmgdbAPI.addTagSet(datasetDetails['Release'],
                                                    endhash)
                for tagID in tagIDs:
                    self._cmgdbAPI.addTagToSet(tagID,
                                               tagSetID)

        if tagSetID is not None:
            self._cmgdbAPI.addTagSetID(tagSetID, datasetDetails['CMGDBID'])
        return datasetDetails['CMGDBID']
