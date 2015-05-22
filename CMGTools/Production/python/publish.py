#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## September 27th 2011

import os, sys, re
from CMGTools.Production.publishController import PublishController
from CMGTools.Production.nameOps import *
from CMGTools.Production.castorBaseDir import getUserAndArea
from CMGTools.Production.datasetInformation import DatasetInformation


def publish(sampleName,fileown,comment,test,username,force,
            primary, run_range = None, development = False ):
    """Publish the given dataset to CMGDB and Savannah

    'sampleName' takes the name of the dataset, in either format
    'fileown' takes the NICE username of the space on EOS in
    which the dataset resides
    'comment' takes a users comment for publishing to Savannah or None
    'username' takes the name of the user publishing the dataset
    'test' takes True/False on whether the posting is a test or not
    'development' takes True/False depending on whether
    wants to publish on the official or the devdb11 database
    """

    def checkName(sampleName, fileown):
        # Validate name, and escape if name is invalidate
        # Convert name to EOS format (castor)
        if re.search("---",sampleName):
            fileown = getFileOwner(sampleName)
            sampleName = getSampleName(sampleName)
            if sampleName is None:
                print "\nError, dataset name is not valid, please use valid name\n"
                return None

        # Check the length of the dataset name
        if len(sampleName.lstrip(os.sep).rstrip(os.sep).split(os.sep)) < 3:
            print "Error, " + sampleName + " is not valid, please use valid name."
            return None
        elif len(sampleName.lstrip(os.sep).rstrip(os.sep).split(os.sep)) < 4:
            print "Dataset "+sampleName+"is a CMS base dataset and cannot be published, please use DAS."
            return None
        return sampleName, fileown

    datasetDetails = None
    try:
        if not primary:
            sampleName, fileown = checkName(sampleName, fileown)
        if sampleName is None: return None
        print "\n\t-------Publishing New Dataset-------"
        print sampleName+"\n"

        # Initialise PublishController
        publishController = PublishController(username,development)

        # Get DS Information
        datasetDetails = DatasetInformation(sampleName, 
                                            fileown, 
                                            comment, 
                                            force,
                                            test,
                                            primary, 
                                            development)

        # Build all reports on the dataset
        if datasetDetails is None:
            return None
        datasetDetails.buildAllReports()
        if datasetDetails.dataset_details is None:
            return None
        # Print dataset names
        print "\n------DataSet Information------"
        print datasetDetails.createDirectoryDetailString()
        for group_name in datasetDetails.dataset_details['FileGroups']:
            print datasetDetails.createFileGroupDetailString(group_name)

        if datasetDetails.dataset_details['TaskID'] is not None:
            status = 'Success'

        # Sent data (with updated task ID) to CMGDB
        if publishController.cmgdbOnline():
            print "\n-------CMGDB-------\n"
            cmgdbid = publishController.cmgdbPublish(datasetDetails.dataset_details)

        return datasetDetails.dataset_details
    except KeyboardInterrupt:
        raise
    except ValueError as err:
        print err.args, '.\nDataset not published'
        return None
    except NameError as err:
        print err.args[0]
        return datasetDetails
