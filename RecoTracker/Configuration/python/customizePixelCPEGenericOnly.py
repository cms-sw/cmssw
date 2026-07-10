import FWCore.ParameterSet.Config as cms

def replacePixelCPETemplateWithGeneric(process):
    # In all PSets and EDProducers, replace 'WithAngleAndTemplate' with 'WithTrackAngle'
    def replace_in_pset(pset):
        # Iterate over the parameters in the PSet, recursively replacing any string that contains 'WithAngleAndTemplate' with 'WithTrackAngle'
        for name in pset.parameters_():
            value = getattr(pset, name)
            if isinstance(value, cms.PSet):
                replace_in_pset(value)
            elif isinstance(value, cms.string) and 'WithAngleAndTemplate' in value.value(): # WithAngleAndTemplate, WithAngleAndTemplateWithoutProbQ 
                setattr(pset, name, cms.string('WithTrackAngle'))

    for name in dir(process):
        obj = getattr(process, name)
        if isinstance(obj, cms.EDProducer):
            for attr_name in dir(obj):
                attr_value = getattr(obj, attr_name)
                if isinstance(attr_value, cms.PSet):
                    replace_in_pset(attr_value)
                elif isinstance(attr_value, cms.string) and 'WithAngleAndTemplate' in attr_value.value():
                    setattr(obj, attr_name, cms.string('WithTrackAngle'))
                # lowPtGsfElectronSeeds have TTRHBuilder = cms.ESInputTag("","WithAngleAndTemplate") instead of cms.string()
                elif isinstance(attr_value, cms.ESInputTag) and 'WithAngleAndTemplate' in attr_value.value():
                    setattr(obj, attr_name, cms.ESInputTag("", 'WithTrackAngle'))
        
        if isinstance(obj, cms.PSet):
            replace_in_pset(obj)

    return process