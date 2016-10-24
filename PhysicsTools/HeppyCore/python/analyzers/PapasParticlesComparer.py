from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.papas.data.comparer import ParticlesComparer
from PhysicsTools.HeppyCore.papas.data.history import History
from PhysicsTools.HeppyCore.papas.data.pfevent import  PFEvent

class PapasParticlesComparer(Analyzer):
    ''' Unsophisticated testing Module that checks that two lists of sorted particles match
       
        Usage:
            from PhysicsTools.HeppyCore.analyzers.PapasParticlesComparer import PapasParticlesComparer 
            particlescomparer = cfg.Analyzer(
                 PapasParticlesComparer ,
                 particlesA = 'papas_PFreconstruction_particles_list',
                 particlesB = 'papas_rec_particles_no_leptons'
             )

    '''
    def __init__(self, *args, **kwargs):
        super(PapasParticlesComparer, self).__init__(*args, **kwargs)
        self.particlesA_name = self.cfg_ana.particlesA
        self.particlesB_name = self.cfg_ana.particlesB
                
    def process(self, event): #think about if argument is correct
        ''' calls a particle comparer to compare two lists of pre-sorted particles
        arguments
            event: must contain baseline_particles (the original reconstruction from simulation)
                   and reconstructed_particles made from the new BlockBuilder approach
        '''
        ParticlesComparer(getattr(event, self.particlesA_name), getattr(event, self.particlesB_name))

        