from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.fcc.particle import Particle
from PhysicsTools.HeppyCore.particles.fcc.jet import Jet
from PhysicsTools.HeppyCore.particles.fcc.vertex import Vertex 
from PhysicsTools.HeppyCore.particles.fcc.met import Met
import PhysicsTools.HeppyCore.configuration

import math

class MissingCollection(Exception):
    pass

class Reader(Analyzer):
    '''Reads events in FCC EDM format, and creates lists of objects adapted to an
    analysis in python.

    Configuration: 
    ----------------------
    
    Example: 
    
    from PhysicsTools.HeppyCore.analyzers.fcc.Reader import Reader
    source = cfg.Analyzer(
      Reader,
      # all the parameters below are optional: 
      gen_particles = 'GenParticle',
      # gen_vertices = '<gen_vertices_name>', 
      # gen_jets = '<gen_jets_name>',
      # jets = '<jets_name>',
    )

    * gen_particles: name of the collection of gen particles
    in the input FCC-EDM file
    * gen_vertices: name of the collection of gen vertices
    * gen_jets: name of the collection of gen jets.
    * jets: name of the collection of reconstructed jets
    
    
    You can find out about the names of the collections by opening
    the root file with root, and by printing the events TTree.

    Creates: 
    --------    

    if self.cfg_ana.gen_particles is set: 
    - event.gen_particles: gen particles
    - event.gen_particles_stable: stable gen_particles except neutrinos

    if the respective parameter is set (see above): 
    - event.gen_vertices: gen vertices (needed for gen particle history)
    - event.gen_jets: gen jets
    - event.jets: reconstructed jets  
    '''
    
    def process(self, event):
        store = event.input

        def get_collection(class_object, coll_label, sort=True):
            pycoll = None
            if hasattr(self.cfg_ana, coll_label):
                coll_name = getattr( self.cfg_ana, coll_label)
                coll = store.get( coll_name )
                if coll == None:
                    raise MissingCollection(
                        'collection {} is missing'.format(coll_name)
                        )
                pycoll = map(class_object, coll)
                if sort:
                    #    pycoll.sort(key = self.sort_key, reverse=True)
                    pycoll.sort(reverse=True)
                setattr(event, coll_label, pycoll )
            return pycoll

        get_collection(Particle, 'gen_particles')
        get_collection(Vertex, 'gen_vertices', False)
        get_collection(Jet, 'gen_jets')
        jetcoll = get_collection(Jet, 'jets')
        if jetcoll:
            jets = dict()
            for jet in jetcoll:
                jets[jet] = jet
            if hasattr(self.cfg_ana, 'bTags') and \
               hasattr(self.cfg_ana, 'jetsToBTags'):
                for tt in store.get(self.cfg_ana.jetsToBTags):
                    jets[Jet(tt.Jet())].tags['bf'] = tt.Tag().Value()

        class Iso(object):
            def __init__(self):
                self.sumpt=-9999
                self.sume=-9999
                self.num=-9999

        electrons = dict()
        if hasattr(self.cfg_ana, 'electrons'):
            event.electrons = map(Particle, store.get(self.cfg_ana.electrons))
            event.electrons.sort(reverse=True)
            for ele in event.electrons:
                ele.iso = Iso()
                electrons[ele]=ele
        if  hasattr(self.cfg_ana, 'electronsToITags') and hasattr(self.cfg_ana, 'electronITags'):
            for ele in store.get(self.cfg_ana.electronsToITags):
                electrons[Particle(ele.Particle())].iso = Iso()
                electrons[Particle(ele.Particle())].iso.sumpt = electrons[Particle(ele.Particle())].pt()*ele.Tag().Value()

        muons = dict()
        if hasattr(self.cfg_ana, 'muons'):
            event.muons = map(Particle, store.get(self.cfg_ana.muons))
            event.muons.sort(reverse=True)   
            for mu in event.muons:
                mu.iso = Iso()
                muons[mu]=mu
        if  hasattr(self.cfg_ana, 'muonsToITags') and hasattr(self.cfg_ana, 'muonITags'):
            for mu in store.get(self.cfg_ana.muonsToITags):
                #import pdb; pdb.set_trace()
                muons[Particle(mu.Particle())].iso = Iso()
                muons[Particle(mu.Particle())].iso.sumpt = muons[Particle(mu.Particle())].pt()*mu.Tag().Value()
                
        
        get_collection(Particle, 'photons')
        met = get_collection(Met, 'met', False)
        if met:
            event.met = event.met[0]
