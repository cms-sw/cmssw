import FWCore.ParameterSet.Config as cms

def setCondition(process,
                 connect = "frontier://FrontierProd/CMS_CONDITIONS",
                 record = None,
                 tag = None,
                 label = None):
    """
    Overrides a condition in startgeometry from globaltag.
    """

    if record is None or tag is None:
        raise ValueError("A 'record' and a 'tag' have to be provided to 'setCondition'.")

    if not hasattr(process, "GlobalTag"):
        process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

    args = {"connect": cms.string(connect),
            "record": cms.string(record),
            "tag": cms.string(tag)}
    if label is not None:
        args["label"] = cms.untracked.string(label)

    process.GlobalTag.toGet.append(cms.PSet(**args))
