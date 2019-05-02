import FWCore.ParameterSet.Config as cms

class HGCalTriggerChains:
    def __init__(self):
        self.vfe = {}
        self.concentrator = {}
        self.backend1 = {}
        self.backend2 = {}
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

    def register_ntuple(self, name, generator):
        self.ntuple[name] = generator

    def register_chain(self, vfe, concentrator, backend1, backend2, ntuple=''):
        if not vfe in self.vfe: 
            raise KeyError('{} not registered as VFE producer'.format(vfe))
        if not concentrator in self.concentrator: 
            raise KeyError('{} not registered as concentrator producer'.format(concentrator))
        if not backend1 in self.backend1: 
            raise KeyError('{} not registered as backend1 producer'.format(backend1))
        if not backend2 in self.backend2: 
            raise KeyError('{} not registered as backend2 producer'.format(backend2))
        if ntuple!='' and not ntuple in self.ntuple: 
            raise KeyError('{} not registered as ntuplizer'.format(ntuple))
        self.chain.append( (vfe, concentrator, backend1, backend2, ntuple) )
    

    def create_sequences(self, process):
        tmp = cms.TaskPlaceholder("tmp")
        tmpseq = cms.SequencePlaceholder("tmp")
        vfe_task = cms.Task(tmp)
        concentrator_task = cms.Task(tmp)
        backend1_task = cms.Task(tmp)
        backend2_task = cms.Task(tmp)
        ntuple_sequence = cms.Sequence(tmpseq)
        for vfe,concentrator,backend1,backend2,ntuple in self.chain:
            concentrator_name = '{0}{1}'.format(vfe, concentrator)
            backend1_name = '{0}{1}{2}'.format(vfe, concentrator, backend1)
            backend2_name = '{0}{1}{2}{3}'.format(vfe, concentrator, backend1, backend2)
            ntuple_name = '{0}{1}{2}{3}{4}'.format(vfe, concentrator, backend1, backend2, ntuple)
            ntuple_inputs = [concentrator_name, backend1_name, backend2_name]
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
                backend2_task.add(getattr(process, backend2_name))
            if ntuple!='' and not hasattr(process, ntuple_name):
                setattr(process, ntuple_name, self.ntuple[ntuple](process, ntuple_inputs))
                ntuple_sequence *= getattr(process, ntuple_name)
        vfe_task.remove(tmp)
        concentrator_task.remove(tmp)
        backend1_task.remove(tmp)
        backend2_task.remove(tmp)
        ntuple_task.remove(tmpseq)
        process.globalReplace('hgcalVFE', vfe_task)
        process.globalReplace('hgcalConcentrator', concentrator_task)
        process.globalReplace('hgcalBackEndLayer1', backend1_task)
        process.globalReplace('hgcalBackEndLayer2', backend2_task)
        process.globalReplace('hgcalTriggerNtuples', ntuple_sequence)
        return process
