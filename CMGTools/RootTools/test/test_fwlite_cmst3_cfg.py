import copy
import CMGTools.RootTools.fwlite.Config as cfg

treeAna = cfg.Analyzer(
    'TestTreeAnalyzer'
    )

def getFiles(datasets, user, pattern):
    from CMGTools.Production.datasetToSource import datasetToSource
    files = []
    for d in datasets:
        ds = datasetToSource( user,
                              d,
                              pattern )
        files.extend(ds.fileNames)
    return ['root://eoscms//eos/cms%s' % f for f in files]

# getting the first 10 files 
files = getFiles(['/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola/Fall11-PU_Chamonix12_START44_V10-v2/AODSIM/PAT_CMG_V3_0_0'], 'cmgtools','tree.*root')[:10]

print files

DYJets = cfg.MCComponent(
    name = 'DYJets',
    files = files,
    xSection = 3048.,
    nGenEvents = 34915945,
    triggers = [],
    effCorrFactor = 1 )

selectedComponents =  [DYJets]

sequence = cfg.Sequence( [
    treeAna
    ] )

config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

DYJets.splitFactor = 1
