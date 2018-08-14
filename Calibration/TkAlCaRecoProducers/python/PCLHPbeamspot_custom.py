"""
_Utils_

Tools to customise the PCL workflow which computes beamspot from a dedicated express-like stream

"""

def customise_HPbeamspot(process):

    # write to sqlite the HP tag and use the HP medatata for uploading it  to the dropbox
    # ByLumi
    if ( hasattr(process,'PoolDBOutputService')   and
         hasattr(process,'pclMetadataWriter')     and
         hasattr(process,'ALCAHARVESTBeamSpotByLumi')  ):
        for onePset in process.PoolDBOutputService.toPut:
            if onePset.record == 'BeamSpotObjectsRcdByLumi':
                onePset.record = 'BeamSpotObjectsRcdHPByLumi'
                onePset.tag    = 'BeamSpotObjectHP_ByLumi'
        for onePset in process.pclMetadataWriter.recordsToMap:
            if onePset.record == 'BeamSpotObjectsRcdByLumi':
                onePset.record = 'BeamSpotObjectsRcdHPByLumi'
        if process.ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName == 'BeamSpotObjectsRcdByLumi':
            process.ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.outputRecordName = 'BeamSpotObjectsRcdHPByLumi'
            process.ALCAHARVESTBeamSpotByLumi.AlcaBeamSpotHarvesterParameters.DumpTxt = True
    # ByRun
    if ( hasattr(process,'PoolDBOutputService')   and
         hasattr(process,'pclMetadataWriter')     and
         hasattr(process,'ALCAHARVESTBeamSpotByRun')  ):
        for onePset in process.PoolDBOutputService.toPut:
            if onePset.record == 'BeamSpotObjectsRcdByRun':
                onePset.record = 'BeamSpotObjectsRcdHPByRun'
                onePset.tag    = 'BeamSpotObjectHP_ByRun'
        for onePset in process.pclMetadataWriter.recordsToMap:
            if onePset.record == 'BeamSpotObjectsRcdByRun':
                onePset.record = 'BeamSpotObjectsRcdHPByRun'
        if process.ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.outputRecordName == 'BeamSpotObjectsRcdByRun':
            process.ALCAHARVESTBeamSpotByRun.AlcaBeamSpotHarvesterParameters.outputRecordName = 'BeamSpotObjectsRcdHPByRun'

    # ALCARECOTkAlMinBiasTkAlDQM is part of the ALCARECO sequence we want and needs caloJets
    # which are not available when running tracking only reco => remove it from the sequence
    if hasattr(process,'ALCARECOTkAlMinBiasDQM') and 'ALCARECOTkAlMinBiasTkAlDQM' in process.ALCARECOTkAlMinBiasDQM.moduleNames() :
        process.ALCARECOTkAlMinBiasDQM.remove(process.ALCARECOTkAlMinBiasTkAlDQM)

    return process
