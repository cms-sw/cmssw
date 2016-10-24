from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.isolation import IsolationComputer, IsolationInfo

import pprint 

pdgids = [211, 22, 130, 11, 13]

class IsolationAnalyzer(Analyzer):
    '''Compute lepton isolation.  

    Example:

    from PhysicsTools.HeppyCore.analyzers.IsolationAnalyzer import IsolationAnalyzer
    from PhysicsTools.HeppyCore.particles.isolation import EtaPhiCircle
    iso_leptons = cfg.Analyzer(
    IsolationAnalyzer,
      leptons = 'leptons',
      particles = 'particles',
      iso_area = EtaPhiCircle(0.4)
    )
    
    * leptons : collection of leptons for which the isolation should be computed

    * particles : collection of particles w/r to which the leptons should be isolated. 

    The particles are assumed to have a pdgid equal to 
    +- 11 (electrons)
    +- 13 (muons)
    +- 211 (all other charged particles) 
    22 (photons)
    130 (all other neutral particles) 

    If one of the particles considered in the isolation calculation is 
    the lepton (is the same python object), it is discarded. 

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
        super(IsolationAnalyzer, self).beginLoop(setup)
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
            self.logger.info(str(lepton))
            for pdgid in pdgids:
                sel_ptcs = [ptc for ptc in particles if abs(self.pdgid(ptc))==pdgid]
                iso = self.iso_computers[pdgid].compute(lepton, sel_ptcs)
                isosum += iso 
                setattr(lepton, 'iso_{pdgid}'.format(pdgid=pdgid), iso)
                self.logger.info(str(iso))
                if iso.num:
                    self.logger.info(pprint.pformat(iso.particles))             
            lepton.iso = isosum
            self.logger.info(str(isosum))
        
    def pdgid(self, ptc): 
        '''returns summary pdg id.
        - e or mu -> +- 11 or +- 13
        - charged -> +- 211
        - photon -> 22
        - other -> neutral hadron
        '''
        if ptc.pdgid() in [11,13]:
            return ptc.pdgid() 
        elif ptc.q(): 
            return ptc.q() * 211
        elif ptc.pdgid() == 22: 
            return 22
        else: 
            return 130
