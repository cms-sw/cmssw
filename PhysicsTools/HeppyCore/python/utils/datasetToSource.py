import os
from dataset import Dataset, CMSDataset, LocalDataset, createDataset, PrivateDataset, createMyDataset

import FWCore.ParameterSet.Config as cms

def datasetToSource( user, dataset, pattern='.*root', readCache=False):

    # print user, dataset, pattern
    data = createDataset(user, dataset, pattern, readCache)
    
    source = cms.Source(
	"PoolSource",
	noEventSort = cms.untracked.bool(True),
	duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
	fileNames = cms.untracked.vstring()
        )
    
    source.fileNames.extend( data.listOfGoodFiles() )

    return source

### MM
def myDatasetToSource( user, dataset, pattern='.*root', dbsInstance=None, readCache=False):

    #print user, dataset, pattern, dbsInstance
    data = createMyDataset(user, dataset, pattern, dbsInstance, readCache)

    source = cms.Source(
        "PoolSource",
        noEventSort = cms.untracked.bool(True),
        duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
        fileNames = cms.untracked.vstring()
        )

    #print data.listOfGoodFiles()
    source.fileNames.extend( data.listOfGoodFiles() )

    return source
### MM
