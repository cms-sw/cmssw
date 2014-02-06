
def customiseMergedTruth(process) :
    process.mergedtruth.HepMCDataLabels = ['hiSignal']
    process.mergedtruth.useMultipleHepMCLabels = False
    return process
