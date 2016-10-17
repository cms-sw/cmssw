from heppy.framework.analyzer import Analyzer
from heppy.particles.isolation import IsolationComputer, IsolationInfo


pdgids = [211, 22, 130]

class LeptonAnalyzer(Analyzer):
    '''Compute lepton isolation.  

    Example:

    from heppy.analyzers.LeptonAnalyzer import LeptonAnalyzer
    from heppy.particles.isolation import EtaPhiCircle
    iso_leptons = cfg.Analyzer(
    LeptonAnalyzer,
      leptons = 'leptons',
      particles = 'particles',
      iso_area = EtaPhiCircle(0.4)
    )
    
    * leptons : collection of leptons for which the isolation should be computed

    * particles : collection of particles w/r to which the leptons should be isolated. 

    The particles are assumed to have a pdgid equal to +- 211 (all charged hadrons), 
    22 (photons), or 130 (neutral hadrons). 

    For each pdgid, the isolation result is attached to the lepton.
    For example, to keep track of isolation w/r to charged hadrons, an 
    attribute lepton.iso_211 is attached to each lepton. It contains:
    - lepton.iso_211.sumpt: sum pT of all charged hadrons in a cone around the lepton
    - lepton.iso_211.sume: sum E for these charged hadrons
    - lepton.iso_211.num: number of such charged hadrons 

    Additionally, the attribute lepton.iso is attached to the lepton. it contains 
    sumpt, sume, and num for charged hadrons, photons, and neutral hadrons together. 
    
    See IsolationComputer and IsolationInfo for more information.
    '''
    
    def beginLoop(self, setup):
        super(LeptonAnalyzer, self).beginLoop(setup)
        # now using same isolation definition for all pdgids
        self.iso_computers = dict()
        for pdgid in pdgids:
            self.iso_computers[pdgid] = IsolationComputer(
                [self.cfg_ana.iso_area],
                label='iso{pdgid}'.format(pdgid=str(pdgid))
            )
            
    def process(self, event):
        particles = getattr(event, self.cfg_ana.particles)
        leptons = getattr(event, self.cfg_ana.leptons)
        for lepton in leptons:
            isosum = IsolationInfo('all', lepton)
            for pdgid in pdgids:
                sel_ptcs = [ptc for ptc in particles if ptc.pdgid()==pdgid]
                iso = self.iso_computers[pdgid].compute(lepton, sel_ptcs)
                isosum += iso 
                setattr(lepton, 'iso_{pdgid}'.format(pdgid=pdgid), iso)
            lepton.iso = isosum
        
  
