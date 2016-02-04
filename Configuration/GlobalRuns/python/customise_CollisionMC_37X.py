
def customise(process):
    from Configuration.GlobalRuns.reco_TLR_37X import customisePPMC
    process=customisePPMC(process)
    return process
