def getMyFiles(dataset, user, pattern, dbsInstance):
    from CMGTools.Production.datasetToSource import MyDatasetToSource
    # print 'getting files for', dataset,user,pattern
    ds = MyDatasetToSource( user, dataset, pattern, dbsInstance, True )
    files = ds.fileNames
    return ['root://eoscms//eos/cms%s' % f for f in files]

