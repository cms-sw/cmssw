def splitFactor(comp, nEventsPerJob=2e4):
    '''
    For component comp, returns the split factor needed to split the component in chunks
    containing roughly the same number of events, nEventsPerJob.

    comp is assumed to have the following attributes:
    - dataset_entries: number of events in the corresponding dataset. This attribute is set automatically
    by the connect function (see connect.py in this directory)
    - files: list of files in the dataset, also set automatically by connect.
    '''
    split = int(comp.dataset_entries / nEventsPerJob)
    if split>len(comp.files): split = len(comp.files)
    if split==0: split = 1
    return split
    
