import os
import re 

# from CMGTools.H2TauTau.officialJSONS import jsonMap
# from CMGTools.H2TauTau.dcsJSONS import dcsJsonMap

def lfnToDataset( lfn ):
    '''If lfn contains A/CMG/B, returns /B. Otherwise, returns lfn.'''
    # The CMG that we capture in the regexp below corresponds to the
    # CMG directory that every user should have. The group space is supported
    pattern = re.compile( '.*/cmst3/+[^/]+/+[^/]+/+CMG(\S+)' )
    match = pattern.match( lfn )
    if match is not None:
        dataset = match.group(1)
        # print dataset
        return dataset
    return lfn


def jsonPick( dataset, jsonMap):
    """
    Returns the path to the json file corresponding to a given dataset. 
    dataset is a path to a dataset in the CMG system
    (eos or LFN path, path to a directory or a root file... should work!)

    jsonMap is a dictionary mapping a dataset regexp pattern to
    the path of a json file,
    
    e.g.:

    dcs2011 = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/DCSOnly/json_DCSONLY.txt'
    dcs2012 = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/DCSOnly/json_DCSONLY.txt'
    
    dcsJsonMap = {
    '.*Run2011.*':dcs2011,
    '.*Run2012.*':dcs2012,
    '.*_2011.*':dcs2011,
    '.*_2012.*':dcs2012, 
    }

    This map will e.g. give the dcs2011 json for all datasets containing Run2011
    in their name. 
    """
    
    dataset = lfnToDataset(dataset)

    # stripping out the last part of the dataset name
    # to keep only the base official dataset name
    dsfields = dataset.lstrip('/').split('/')[0:3]
    # print dsfields
    baseDataSet = '/'+'/'.join( dsfields )
        
    # jsonFile = jmap[ baseDataSet ]
    jsonFiles = []
    jsonFile = None
    for key, value in  jsonMap.iteritems():
        pat = re.compile(key)
        if pat.match(baseDataSet):
            jsonFiles.append(value)
    if len(jsonFiles)==0:
        raise ValueError('No json file found for ' + dataset)
    elif len(jsonFiles)>1:
        raise ValueError('Too many json files found')
    else:
        jsonFile = jsonFiles[0]
            
    jsonAbsPath = jsonFile
    if not os.path.isfile(jsonAbsPath):
        raise ValueError( ' '.join([jsonAbsPath,
                                    'does not exist.']) )
    return jsonAbsPath

if __name__ == '__main__':

    samples = [
               '/TauPlusX/Run2011A-May10ReReco-v1/AOD/foo/bar',
               '/TauPlusX/Run2011B-PromptReco-v1/AOD/blah',
               '/TauPlusX/Run2012B-PromptReco-v1/AOD/blah',
               '/TauPlusX/Run2012C-24Aug2012-v1/AOD/blah',

               '/Tau/Run2012A-22Jan2013-v1/AOD'      ,
               '/TauParked/Run2012B-22Jan2013-v1/AOD',
               '/TauParked/Run2012C-22Jan2013-v1/AOD',
               '/TauParked/Run2012D-22Jan2013-v1/AOD',
               '/DoubleMu/StoreResults-Run2012A_22Jan2013_v1_RHembedded_trans1_tau132_pthad1_30had2_30_v1-f456bdbb960236e5c696adfe9b04eaae/USER',
               '/DoubleMuParked/StoreResults-Run2012B_22Jan2013_v1_RHembedded_trans1_tau132_pthad1_30had2_30_v1-f456bdbb960236e5c696adfe9b04eaae/USER',
               '/DoubleMuParked/StoreResults-Run2012C_22Jan2013_v1_RHembedded_trans1_tau132_pthad1_30had2_30_v1-f456bdbb960236e5c696adfe9b04eaae/USER',
               '/DoubleMuParked/StoreResults-Run2012D_22Jan2013_v1_RHembedded_trans1_tau132_pthad1_30had2_30_v1-f456bdbb960236e5c696adfe9b04eaae/USER',

               '/DoubleMu/StoreResults-DoubleMu_Run2012B_13Jul2012_v4_embedded_trans1_tau115_ptelec1_17had1_17_v1-f456bdbb960236e5c696adfe9b04eaae/USER/blah',
               '/store/cmst3/user/cmgtools/CMG/TauPlusX/Run2011A-PromptReco-v4/AOD/V2/PAT_CMG_V2_4_0/tree_CMG_648.root',
               'root://eoscms//eos/cms/store/cmst3/user/cmgtools/CMG/DoubleMu/Run2012A-22Jan2013-v1/AOD/CMGPF_V5_16_0/cmgTuple_1.root',
               'should_fail_for_this_sample_name'
               ]

    from CMGTools.RootTools.json.dcsJSONS import dcsJsonMap

    for sample in samples:
        print 'Sample', sample
        print '\tJSON (DCS)     =', jsonPick( sample, dcsJsonMap )


    
