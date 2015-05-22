from CMGTools.Production.datasetToSource import datasetToSource

def getFiles(dataset, user, regex, useCache=True):
    '''Returns the list of files corresponding to a given dataset
    and matching a regex pattern.

    Each logical file name in the list is appended a prefix
    root://eoscms//eos/cms/<LFN>
    where <LFN> starts with /store/...
    so that the files can be directly opened in root or FWLite,
    that is also in the python analysis framework.
    '''
    from CMGTools.Production.datasetToSource import datasetToSource
    # print 'getting files for', dataset,user,pattern
    ds = datasetToSource( user, dataset, regex, useCache )
    files = ds.fileNames
    return ['root://eoscms//eos/cms%s' % f for f in files]


if __name__ == '__main__':

    import pprint 
    pprint.pprint( getFiles('/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/Summer12_DR53X-PU_S10_START53_V7A-v1/AODSIM/V5_B/PAT_CMG_V5_16_0', 'cmgtools', '.*root') )
