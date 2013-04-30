import FWCore.ParameterSet.Config as cms

def AutoCondGlobalTag(GlobalTag,GTKey):
    """Modify the GlobalTag module configuration, allowing to use the extended
    Configuration/AlCa/python/autoCond.py functionality of specifying symbolic
    globaltags (i.e., actual global tags, plus additional payloads to get from
    the DB as customisations).
    The actual code is taken from cmsDriver's ConfigBuilder.py, adapted here to
    apply the changes directly to the process object and its module GlobalTag."""

    if GlobalTag==None:
        return GlobalTag

    # rely on the same functionality implemented in Configuration.AlCa.GlobalTag
    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
    GlobalTag = customiseGlobalTag(GlobalTag, GTKey)

    return GlobalTag
