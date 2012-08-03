import FWCore.ParameterSet.Config as cms

def GlobalTag(essource = None, globaltag = None, conditions = None):
    """Modify the GlobalTag module configuration, allowing to use the extended
    Configuration/AlCa/python/autoCond.py functionality of specifying symbolic
    globaltags (i.e., actual global tags, plus additional payloads to get from
    the DB as customisations).
    The actual code is taken from cmsDriver's ConfigBuilder.py, adapted here to
    apply the changes directly to the process object and its module GlobalTag."""

    # custom conditions: 
    #  - start from an empty map
    #  - add any conditions coming from autoCond.py
    #  - then add and override them with any conditions explicitly requested
    custom_conditions = {}

    # if no GlobalTag ESSource module is given, load a "default" configuration
    if essource is None:
        from Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi import GlobalTag as essource

    # if a Global Tag is given, check for an "auto" tag, and look it up in autoCond.py
    # if a match is found, load the actual Global Tag and optional conditions
    if globaltag is not None:
        if globaltag.startswith('auto:'):
            from Configuration.AlCa.autoCond import autoCond
            globaltag = globaltag[5:]
            if globaltag not in autoCond:
                raise Exception('no correspondence for '+globaltag+'\navailable keys are\n'+','.join(autoCond.keys()))

            autoKey = autoCond[globaltag]
            if isinstance(autoKey, tuple) or isinstance(autoKey, list):
                globaltag = autoKey[0]
                # TODO backward compatible code: to be removed after migrating autoCond.py to use a map for custom conditions
                map = {}
                for entry in autoKey[1:]:
                  entry = entry.split(',')
                  record     = entry[1]
                  label      = len(entry) > 3 and entry[3] or None
                  tag        = entry[0]
                  connection = len(entry) > 2 and entry[2] or None
                  map[ (record, label) ] = (tag, connection)
                custom_conditions.update( map )
            else:
                globaltag = autoKey

        # if a GlobalTag globaltag is given or loaded from autoCond.py, check for optional connection string and pfn prefix
        globaltag = globaltag.split(',')
        if len(globaltag) > 0 :
            essource.globaltag = cms.string( str(globaltag[0]) )
        if len(globaltag) > 1:
            essource.connect   = cms.string( str(globaltag[1]) )
        if len(globaltag) > 2:
            essource.pfnPrefix = cms.untracked.string( str(globaltag[2]) )

    # add any explicitly requested conditions, possibly overriding those from autoCond.py
    if conditions is not None:
        # TODO backward compatible code: to be removed after migrating ConfigBuilder.py and confdb.py to use a map for custom conditions
        if isinstance(conditions, basestring): 
          if conditions:
            map = {}
            for entry in conditions.split('+'):
                entry = entry.split(',')
                record     = entry[1]
                label      = len(entry) > 3 and entry[3] or None
                tag        = entry[0]
                connection = len(entry) > 2 and entry[2] or None
                map[ (record, label) ] = (tag, connection)
            custom_conditions.update( map )
        elif isinstance(conditions, dict):
          custom_conditions.update( conditions )
        else:
          raise TypeError, "the 'conditions' argument should be either a string or a dictionary"

    # explicit payloads toGet from DB
    if custom_conditions:
        for ( (record, label), (tag, connection) ) in custom_conditions.iteritems():
            payload = cms.PSet()
            payload.record = cms.string( record )
            if label:
                payload.label = cms.untracked.string( label )
            payload.tag = cms.string( tag )
            if connection:
                payload.connection = cms.untracked.string( connection )
            essource.toGet.append( payload )

    return essource
