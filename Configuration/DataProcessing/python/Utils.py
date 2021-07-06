#!/usr/bin/env python3
"""
_Utils_

Module containing some utility tools

"""

def stepALCAPRODUCER(skims):
    """
    _stepALCAPRODUCER_

    Creates and returns the configuration string for the ALCAPRODUCER step
    starting from the list of AlcaReco path to be run.

    """

    step = ''
    if len(skims) >0:
        step = ',ALCAPRODUCER:'+('+'.join(skims))
    return step


def stepSKIMPRODUCER(PhysicsSkims):
    """
    _stepSKIMPRODUCER_

    Creates and returns the configuration string for the SKIM step
    starting from the list of skims to be run.

    """

    step = ''
    if len(PhysicsSkims) >0 :
        step = ',SKIM:'+('+'.join(PhysicsSkims))
    return step

def addMonitoring(process):
    """
    _addMonitoring_
    
    Add the monitoring services to the process provided
    in order to write out performance summaries to the framework job report
    """
    import FWCore.ParameterSet.Config as cms
    
    process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                            jobReportOutputOnly = cms.untracked.bool(True)
                                            )
    process.Timing = cms.Service("Timing",
                                 summaryOnly = cms.untracked.bool(True)
                                 )
    
    return process


def validateProcess(process):
    """
    _validateProcess_
    
    Check attributes of process are appropriate for production
    This method returns nothing but will throw a RuntimeError for any issues it finds
    likely to cause problems in the production system
    
    """
    
    schedule=process.schedule_()
    paths=process.paths_()
    endpaths=process.endpaths_()
    
    # check output mods are in paths and have appropriate settings
    for outputModName in process.outputModules_().keys():
        outputMod = getattr(process, outputModName)
        if not hasattr(outputMod, 'dataset'):
            msg = "Process contains output module without dataset PSET: %s \n" % outputModName
            msg += " You need to add this PSET to this module to set dataTier and filterName\n"
            raise RuntimeError(msg)
        ds=getattr(outputMod,'dataset')
        if not hasattr(ds, "dataTier"):
            msg = "Process contains output module without dataTier parameter: %s \n" % outputModName
            msg += " You need to add an untracked parameter to the dataset PSET of this module to set dataTier\n"
            raise RuntimeError(msg)

        # check module in path or whatever (not sure of exact syntax for endpath)
        omRun=False

        if schedule==None:
            for path in paths:
                if outputModName in getattr(process,path).moduleNames():
                    omRun=True
            for path in endpaths:
                if outputModName in getattr(process,path).moduleNames():
                    omRun=True
        else:
            for path in schedule:
                if outputModName in path.moduleNames():
                    omRun=True
        if omRun==False:
            msg = "Output Module %s not in endPath" % outputModName
            raise RuntimeError(msg)

        
def dqmIOSource(args):
    import FWCore.ParameterSet.Config as cms
    if args.get('newDQMIO', False):
        return cms.Source("DQMRootSource",
                          fileNames = cms.untracked(cms.vstring())
                          )
    else:
        return cms.Source("PoolSource",
                          fileNames = cms.untracked(cms.vstring())
                          )

def harvestingMode(process, datasetName, args,rANDl=True):
    import FWCore.ParameterSet.Config as cms
    if rANDl and (not args.get('newDQMIO', False)):
        process.source.processingMode = cms.untracked.string('RunsAndLumis')
    process.dqmSaver.workflow = datasetName
    process.dqmSaver.saveByLumiSection = 1

def dictIO(options,args):
    if 'outputs' in args:
        options.outputDefinition = args['outputs'].__str__()
    else:
        writeTiers = args.get('writeTiers', [])
        options.eventcontent = ','.join(writeTiers)
        options.datatier = ','.join(writeTiers)

def dqmSeq(args,default):
    if 'dqmSeq' in args and len(args['dqmSeq'])!=0:
        return ':'+('+'.join(args['dqmSeq']))
    else:
        return default

def gtNameAndConnect(globalTag, args):
    if 'globalTagConnect' in args and args['globalTagConnect'] != '':
        return globalTag + ','+args['globalTagConnect']        
    # we override here the default in the release which uses the FrontierProd servlet not suited for Tier0 activity
    return globalTag +',frontier://PromptProd/CMS_CONDITIONS'
