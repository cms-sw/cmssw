def recover(process):
    if hasattr(process,'siStripDigis'):
        process.siStripDigis.UnpackBadChannels = cms.bool(True)
    return (process)

