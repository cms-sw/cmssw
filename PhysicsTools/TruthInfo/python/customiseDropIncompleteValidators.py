"""Harness-only customise for producing HGCAL validation plots from standalone
RECO-from-step2 jobs. It removes validators that need EventSetup records or
collections a standalone offline job does not provide (HCAL trigger-primitive
record, HLT-only reco), keeping the offline HGCal validator and its associators
intact. Not for PR configs: the CI workflows run the full chain and provide
these inputs."""


def customise(process):
    drop = [
        "AllHcalDigisValidation",   # needs CaloTPGRecord
        "hcalDigisValidationSequence",
        "hltHgcalValidator",        # HLT TICL reco not run in an offline RECO job
        "tpHltGsfTrackAssociation",
        "hltGsfTrackValidator",
        "hltTrackValidator",
        "hltMultiTrackValidator",
    ]
    for label in drop:
        if hasattr(process, label):
            process.__delattr__(label)
    # Also blank any path/sequence referencing the dropped labels leniently by
    # rebuilding the schedule without empty leftovers is unnecessary: EDM prunes
    # unscheduled modules, and removing the module makes the sequence skip it.
    return process
