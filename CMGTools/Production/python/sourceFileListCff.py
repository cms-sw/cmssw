def sourceFileListCff( files, bad_files = {}):

    str = '''
import FWCore.ParameterSet.Config as cms

source = cms.Source(
\t"PoolSource",
\tnoEventSort = cms.untracked.bool(True),
\tduplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
\tfileNames = cms.untracked.vstring()
)
source.fileNames.extend([
'''
    for file in files:
        file = file.replace('//','/')
        #     file = file.replace( protocol+'/castor/cern.ch/cms/store', '/store')  
        if not bad_files.has_key(file):
            fileLine = "\t\t'%s'," % file
        else:
            reason = bad_files[file]
            fileLine = "###%s\t'%s'," % (reason,file)
        fileLine += '\n'
        str += fileLine 
    str += "])"

    return str
