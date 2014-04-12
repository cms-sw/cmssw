import FWCore.ParameterSet.Config as cms

def documentSkims():
    import Configuration.StandardSequences.Skims_cff as Skims

    listOfOptions=[]
    for skim in Skims.__dict__:
        skimstream = getattr(Skims,skim)
        if (not isinstance(skimstream,cms.FilteredStream)):
            continue
        
        shortname = skim.replace('SKIMStream','')
        print shortname
        if shortname!=skimstream['name']:
            print '#### ERROR ####'
            print 'skim name and stream name should be the same for consistency',shortname,'!=',skimstream['name']
            
        for token in ['name','responsible','dataTier']:
            print token,":",skimstream[token]
            
        listOfOptions.append(skimstream['name'])

    print 'possible cmsDriver options for skimming:'
    print 'SKIM:'+'+'.join(listOfOptions)


### DPG skims ###
from DPGAnalysis.Skims.Skims_DPG_cff import *


### Central Skims ###
from Configuration.Skimming.Skims_PDWG_cff import *
