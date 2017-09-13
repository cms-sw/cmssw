from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.particles.tlv.jet import Jet
from PhysicsTools.HeppyCore.particles.jet import JetConstituents

import os 

from ROOT import gSystem
CCJetClusterizer = None
if os.environ.get('FCCPHYSICS'):
    gSystem.Load("libfccphysics-tools")
    from ROOT import JetClusterizer as CCJetClusterizer
elif os.environ.get('CMSSW_BASE'):
    gSystem.Load("libColinPFSim")
    from ROOT import heppy
    CCJetClusterizer = PhysicsTools.HeppyCore.JetClusterizer

import math
    
class JetClusterizer(Analyzer):
    '''Jet clusterizer. 
    
    Makes use of the JetClusterizer class compiled in the analysis-cpp package
    (this external package is the only dependence to the FCC software).

    Example configuration: 

    from PhysicsTools.HeppyCore.analyzers.fcc.JetClusterizer import JetClusterizer
    jets = cfg.Analyzer(
      JetClusterizer,
      output = 'jets',
      particles = 'particles_not_zed',
      fastjet_args = dict( njets = 2)  
    )
    
    * output: name of the output collection of Jets. 
    Each jet is attached a JetConstituents object as jet.constituents.
    See the Jet and JetConstituents classes. 

    * particles: name of the input collection of particle-like objects. 
    These objects should have a p4(). 
    
    you should provide either one or the other of the following arguments:
    - ptmin : pt threshold for exclusive jet reconstruction 
    - njets : number of jets for inclusive jet reconstruction 

    A more flexible interface can easily be provided if needed, 
    contact Colin.
    '''

    def __init__(self, *args, **kwargs):
        super(JetClusterizer, self).__init__(*args, **kwargs)
        args = self.cfg_ana.fastjet_args
        self.clusterize = None
        if 'ptmin' in args and 'njets' in args:
            raise ValueError('cannot specify both ptmin and njets arguments')
        if 'ptmin' in args:
            self.clusterizer = CCJetClusterizer(0)
            def clusterize():
                return self.clusterizer.make_inclusive_jets(args['ptmin']) 
            self.clusterize = clusterize
        elif 'njets' in args:
            self.clusterizer = CCJetClusterizer(1)
            def clusterize():
                return self.clusterizer.make_exclusive_jets(args['njets']) 
            self.clusterize = clusterize
        else:
            raise ValueError('specify either ptmin or njets') 
        
    def validate(self, jet):
        constits = jet.constituents
        keys = set(jet.constituents.keys())
        all_possible = set([211, 22, 130, 11, 13, 1, 2])
        if not keys.issubset(all_possible):
            print constits
            assert(False)
        sume = 0. 
        for component in jet.constituents.values():
            if component.e() - jet.e() > 1e-5:
                import pdb; pdb.set_trace()
            sume += component.e()
        if jet.e() - sume > 1e-5:
            import pdb; pdb.set_trace()
                
                
    def process(self, event):
        particles = getattr(event, self.cfg_ana.particles)
        # removing neutrinos
        particles = [ptc for ptc in particles if abs(ptc.pdgid()) not in [12,14,16]]
        self.clusterizer.clear();
        for ptc in particles:
            self.clusterizer.add_p4( ptc.p4() )
        self.clusterize()
        jets = []
        for jeti in range(self.clusterizer.n_jets()):
            jet = Jet( self.clusterizer.jet(jeti) )
            jet.constituents = JetConstituents()
            jets.append( jet )
            for consti in range(self.clusterizer.n_constituents(jeti)):
                constituent_index = self.clusterizer.constituent_index(jeti, consti)
                constituent = particles[constituent_index]
                jet.constituents.append(constituent)
            jet.constituents.sort()
            self.validate(jet)
        setattr(event, self.cfg_ana.output, jets)
