def removeL1FastJetJECs(process):
    for label in process.producerNames().split():
        module = getattr(process, label)
        if module.type_() == "PATPFJetMETcorrInputProducer":
            module.offsetCorrLabel = ''

def removeJECsForMC(process):
    for label in process.producerNames().split():
        module = getattr(process, label)
        if module.type_() == "PATPFJetMETcorrInputProducer":
            module.jetCorrLabel = 'Uncorrected'

    process.basicJetsForMet.jetCorrLabel = 'Uncorrected'
