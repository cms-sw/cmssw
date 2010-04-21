
def customise(process):
    from Configuration.GlobalRuns.reco_TLR_36X import customisePPMC
    process=customisePPMC(process)
    return process
