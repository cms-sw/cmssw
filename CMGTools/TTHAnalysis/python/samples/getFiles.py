def getFiles(dataset, user, pattern):
    from CMGTools.Production.datasetToSource import datasetToSource
    # print 'getting files for', dataset,user,pattern
    ds = datasetToSource( user, dataset, pattern, True )
    files = ds.fileNames
    #return ['root://eoscms//eos/cms%s' % f for f in files]
    return ['root://eoscms.cern.ch//eos/cms%s' % f for f in files]

