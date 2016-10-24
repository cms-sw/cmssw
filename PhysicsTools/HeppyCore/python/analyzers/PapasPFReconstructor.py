from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.papas.pfalgo.pfreconstructor import PFReconstructor as PFReconstructor
from PhysicsTools.HeppyCore.papas.data.pfevent import PFEvent
from PhysicsTools.HeppyCore.papas.pfalgo.distance  import Distance
from PhysicsTools.HeppyCore.papas.data.history import History

class PapasPFReconstructor(Analyzer):
    ''' Module to reconstruct particles from blocks of events
         
        Usage:
        pfreconstruct = cfg.Analyzer(
            PapasPFReconstructor,
            instance_label = 'papas_PFreconstruction', 
            detector = CMS(),
            input_blocks = 'reconstruction_blocks',
            input_history = 'history_nodes', 
            output_history = 'history_nodes',     
            output_particles_dict = 'particles_dict', 
            output_particles_list = 'particles_list'
        )
        
        input_blocks: Name of the the blocks dict in the event
        history: Name of history_nodes
        output_particles_dict = Name for recosntructed particles (as dict), 
        output_particles_list =  Name for recosntructed particles (as list)
    '''
    
    def __init__(self, *args, **kwargs):
        super(PapasPFReconstructor, self).__init__(*args, **kwargs)  
        self.detector = self.cfg_ana.detector
        self.reconstructed = PFReconstructor(self.detector, self.logger)
        self.blocksname =  self.cfg_ana.input_blocks
        self.historyname = self.cfg_ana.history   
        self.output_particlesdictname = '_'.join([self.instance_label,
                                                  self.cfg_ana.output_particles_dict])
        self.output_particleslistname = '_'.join([self.instance_label,
                                                  self.cfg_ana.output_particles_list])
                
    def process(self, event):
        ''' Calls the particle reconstruction algorithm and returns the 
           reconstructed paricles and updated history_nodes to the event object
           arguments:
                    event must contain blocks made using BlockBuilder'''
        
        self.reconstructed.reconstruct(event,  self.blocksname, self.historyname)
        
        #setattr(event, self.historyname, self.reconstructed.history_nodes)
        setattr(event, self.output_particlesdictname, self.reconstructed.particles)
        
        #hist = History(event.history_nodes,PFEvent(event))
        #for block in event.blocks:
        #    hist.summary_of_links(block)
        
        #for particle comparison we want a list of particles (instead of a dict) so that we can sort and compare
        reconstructed_particle_list = sorted( self.reconstructed.particles.values(),
                                                   key = lambda ptc: ptc.e(),
                                                   reverse=True)
        
        setattr(event, self.output_particleslistname, reconstructed_particle_list)
