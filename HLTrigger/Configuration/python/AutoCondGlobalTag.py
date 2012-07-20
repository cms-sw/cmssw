import FWCore.ParameterSet.Config as cms

def AutoCondGlobalTag(process,gtkey):
    """Modify the process.GlobalTag module configuration, allowing to use the
    extended Configuration/AlCa/python/autoCond.py customisation functionality of
    specifying symbolic globaltags (actual global tags plus additional payloads
    to get from the DB as customisations).
    The actual code is taken from cmsDriver's ConfigBuilder.py, adapted here to
    apply the changes directly to the process object and its module GlobalTag."""

    if not 'GlobalTag' in process.__dict__:
        return process
  
    # offline use
    process.GlobalTag.connect   = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG')
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

    # starting point for conditions to process
    conditions = gtkey
    
    # check for auto globaltag
    if 'auto:' in conditions:
        from Configuration.AlCa.autoCond import autoCond
        key=conditions.split(':')[-1]
        if key not in autoCond:
            raise Exception('no correspondance for '+key+'\n available keys are '+','.join(autoCond.keys()))
        else:
            conditions = autoCond[key]

    # check format/type of customisation
    if isinstance(conditions,tuple):
        condition = conditions[0]                    # actual globaltag
        custom_conditions = '+'.join(conditions[1:]) # payloads
    else:
        condition = conditions                       # actual globaltag
        custom_conditions = ""                       # payloads
    
    # actual globaltag (required) plus connect/pfnPrefix strings (optional)
    cond = condition.split(',')
    process.GlobalTag.globaltag = cms.string(str(cond[0]))
    if len(cond) > 1:
        process.GlobalTag.connect = cms.string(str(cond[1]))
        if len(cond) > 2:
            process.GlobalTag.pfnPrefix = cms.untracked.string(str(cond[2]))

    # explicit payloads toGet from DB
    if custom_conditions!="":    
        specs = custom_conditions.split('+')
        for spec in specs:
            items = spec.split(',')
            payloadSpec={}
            allowedFields=['tag','record','connect','label']
            for i,item in enumerate(items):
                payloadSpec[allowedFields[i]]=item

            if (not 'record' in payloadSpec) or (not 'tag' in payloadSpec):
                raise Exception('conditions cannot be customised with: '+repr(payloadSpec)+' no record or tag field available')
            payloadSpecToAppend=cms.PSet()
            for i,item in enumerate(allowedFields):
                if not item in payloadSpec: continue
                if not payloadSpec[item]: continue
                if i==0:
                    payloadSpecToAppend.tag=cms.string(str(payloadSpec[item]))
                if i==1:    
                    payloadSpecToAppend.record=cms.string(str(payloadSpec[item]))
                if i==2:
                    payloadSpecToAppend.connect=cms.untracked.string(str(payloadSpec[item]))
                if i==3:
                    payloadSpecToAppend.label=cms.untracked.string(str(payloadSpec[item]))
#           print 'customising the GlobalTag with:',payloadSpecToAppend
            process.GlobalTag.toGet.append(payloadSpecToAppend)

    #
    return process
