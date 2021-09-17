import FWCore.ParameterSet.Config as cms

def customizeHLTforNewDatasetDefinition(process):
    # Loop over streams
    for stream in process.streams.parameterNames_():
        streamPaths = cms.vstring()
        # Loop over datasets
        for dataset in getattr( process.streams, stream ):
            # Define dataset prescaler
            setattr( process, 'hltPreDataset'+dataset, cms.EDFilter( "HLTPrescaler", L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ), offset = cms.uint32( 0 ) ) )
            # Define dataset selection
            paths = getattr( process.datasets, dataset )
            setattr( process, 'hltDataset'+dataset, cms.EDFilter( "PathStatusFilter" , verbose = cms.untracked.bool( False ), logicalExpression = cms.string( ' or '.join(paths) ) ) )
            # Create dataset path
            datasetPath = 'Dataset_'+dataset+'_v1'
            setattr( process, datasetPath, cms.Path( process.hltGtStage2Digis + getattr( process , 'hltPreDataset'+dataset ) +  getattr( process, 'hltDataset'+dataset ) + process.HLTEndSequence ) )
            # Append dataset path
            process.HLTSchedule.insert( process.HLTSchedule.index( process.HLTriggerFinalPath ), getattr( process, datasetPath ) )
            setattr( process.datasets, dataset, cms.vstring( datasetPath ) )
            streamPaths.append( datasetPath )
        # Set stream paths
        getattr( process, 'hltOutput'+stream ).SelectEvents.SelectEvents = streamPaths

    return process
