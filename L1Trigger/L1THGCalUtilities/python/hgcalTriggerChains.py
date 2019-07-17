import FWCore.ParameterSet.Config as cms

class HGCalTriggerChains:
    def __init__(self):
        self.vfe = {}
        self.concentrator = {}
        self.backend1 = {}
        self.backend2 = {}
        self.selector = {}
        self.ntuple = {}
        self.chain = []

    def register_vfe(self, name, generator):
        self.vfe[name] = generator

    def register_concentrator(self, name, generator):
        self.concentrator[name] = generator

    def register_backend1(self, name, generator):
        self.backend1[name] = generator

    def register_backend2(self, name, generator):
        self.backend2[name] = generator

    def register_selector(self, name, generator):
        self.selector[name] = generator

    def register_ntuple(self, name, generator):
        self.ntuple[name] = generator

    def register_chain(self, vfe, concentrator, backend1, backend2, selector='', ntuple=''):
        if not vfe in self.vfe: 
            raise KeyError('{} not registered as VFE producer'.format(vfe))
        if not concentrator in self.concentrator: 
            raise KeyError('{} not registered as concentrator producer'.format(concentrator))
        if not backend1 in self.backend1: 
            raise KeyError('{} not registered as backend1 producer'.format(backend1))
        if not backend2 in self.backend2: 
            raise KeyError('{} not registered as backend2 producer'.format(backend2))
        if selector!='' and not selector in self.selector:
            raise KeyError('{} not registered as selector'.format(selector))
        if ntuple!='' and not ntuple in self.ntuple: 
            raise KeyError('{} not registered as ntuplizer'.format(ntuple))
        self.chain.append( (vfe, concentrator, backend1, backend2, selector, ntuple) )
    

    def create_sequences(self, process):
        if not hasattr(process, 'hgcalTriggerSelector'):
            process.load('L1Trigger.L1THGCalUtilities.HGC3DClusterGenMatchSelector_cff')
        tmp = cms.SequencePlaceholder("tmp")
        vfe_sequence = cms.Sequence(tmp)
        concentrator_sequence = cms.Sequence(tmp)
        backend1_sequence = cms.Sequence(tmp)
        backend2_sequence = cms.Sequence(tmp)
        selector_sequence = cms.Sequence(tmp)
        ntuple_sequence = cms.Sequence(tmp)
        for vfe,concentrator,backend1,backend2,selector,ntuple in self.chain:
            concentrator_name = '{0}{1}'.format(vfe, concentrator)
            backend1_name = '{0}{1}{2}'.format(vfe, concentrator, backend1)
            backend2_name = '{0}{1}{2}{3}'.format(vfe, concentrator, backend1, backend2)
            selector_name = '{0}{1}{2}{3}{4}'.format(vfe, concentrator, backend1, backend2, selector)
            ntuple_name = '{0}{1}{2}{3}{4}{5}'.format(vfe, concentrator, backend1, backend2, selector, ntuple)
            if selector=='':
                ntuple_inputs = [
                        concentrator_name+':HGCalConcentratorProcessorSelection',
                        backend1_name+':HGCalBackendLayer1Processor2DClustering',
                        backend2_name+':HGCalBackendLayer2Processor3DClustering'
                        ]
            else:
                ntuple_inputs = [
                        concentrator_name+':HGCalConcentratorProcessorSelection',
                        backend1_name+':HGCalBackendLayer1Processor2DClustering',
                        selector_name]
            if not hasattr(process, vfe):
                setattr(process, vfe, self.vfe[vfe](process))
                vfe_task.add(getattr(process, vfe))
            if not hasattr(process, concentrator_name):
                setattr(process, concentrator_name, self.concentrator[concentrator](process, vfe))
                concentrator_task.add(getattr(process, concentrator_name))
            if not hasattr(process, backend1_name):
                setattr(process, backend1_name, self.backend1[backend1](process, concentrator_name))
                backend1_task.add(getattr(process, backend1_name))
            if not hasattr(process, backend2_name):
                setattr(process, backend2_name, self.backend2[backend2](process, backend1_name))
                backend2_sequence *= getattr(process, backend2_name)
            if selector!='' and not hasattr(process, selector_name):
                setattr(process, selector_name, self.selector[selector](process, backend2_name))
                selector_sequence *= getattr(process, selector_name)
            if ntuple!='' and not hasattr(process, ntuple_name):
                setattr(process, ntuple_name, self.ntuple[ntuple](process, ntuple_inputs))
                ntuple_sequence *= getattr(process, ntuple_name)
        vfe_sequence.remove(tmp)
        concentrator_sequence.remove(tmp)
        backend1_sequence.remove(tmp)
        backend2_sequence.remove(tmp)
        selector_sequence.remove(tmp)
        ntuple_sequence.remove(tmp)
        process.globalReplace('hgcalVFE', vfe_sequence)
        process.globalReplace('hgcalConcentrator', concentrator_sequence)
        process.globalReplace('hgcalBackEndLayer1', backend1_sequence)
        process.globalReplace('hgcalBackEndLayer2', backend2_sequence)
        process.globalReplace('hgcalTriggerSelector', selector_sequence)
        process.globalReplace('hgcalTriggerNtuples', ntuple_sequence)
        return process
