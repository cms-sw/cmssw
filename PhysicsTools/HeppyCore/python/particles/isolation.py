from PhysicsTools.HeppyCore.utils.deltar import deltaR2

class Area(object):
    '''Base Area interface.'''
    def is_inside(self, *args):
        '''returns True if *args describes a particle inside the EtaPhiCircle. 

        *args may be the particle itself, assuming it has eta() and phi() methods, 
        or eta, phi. 
        '''
        pass

class EtaPhiCircle(Area):
    '''Circle in (eta, phi) space.
    When running on a lepton collider, eta is replaced by theta. 
    '''
    def  __init__(self, R):
        '''Create a circle of radius R'''
        self.R = R
        self._R2 = R**2

    def is_inside(self, *args):
        dR2 = deltaR2(*args)
        return dR2 < self._R2


class IsolationInfo(object):
    '''Holds the results of an isolation calculation.'''
    def __init__(self, label, lepton):
        '''Create an IsolationInfo.

        Attributes:
         lepton    = the lepton 
         particles = list of particles around the lepton used in the calculation.
                     the following quantities are computed for these particles
         sumpt = total pT for the particles
         sume  = total energy for the particles 
         num   = total number of particles 
        '''
        self.particles = []
        self.label = label
        self.lepton = lepton
        self.sumpt = 0
        self.sume = 0
        self.num = 0

    def add_particle(self, ptc):
        '''Add a new particle and update counters.'''
        self.particles.append(ptc)
        self.sumpt += ptc.pt()
        self.sume += ptc.e()
        self.num += 1 

    def __iadd__(self, other):
        self.particles.extend(other.particles)
        self.sumpt += other.sumpt
        self.sume += other.sume
        self.num += other.num
        return self
        
    def __str__(self):
        return 'iso {label:>3}: sumpt = {sumpt:5.2f}, sume = {sume:5.2f}, num = {num}'.format(
            label = self.label,
            sumpt = self.sumpt,
            sume = self.sume,
            num = self.num
        )
                
        
    
class IsolationComputer(object):
    '''Computes isolation for a given lepton.'''

    def __init__(self, on_areas, off_areas=None,
                 pt_thresh=0, e_thresh=0, label=''):
        '''Creates the isolation computer. 

        Particles around the lepton are considered in the isolation if:
        - they pass both thresholds:
          pt_thresh : pt threshold
          e_thresh  : energy threshold

        - they are in an active area around the lepton
        areas should 

        on_areas and off_areas are lists of areas in which particles 
        around the should be considered 
        or ignored, respectively.
        for a given particle 

        ''' 

        self.on_areas = on_areas
        if off_areas is None:
            off_areas = []
        self.off_areas = off_areas
        self.pt_thresh = pt_thresh
        self.e_thresh = e_thresh
        self.label = label

        
    def compute(self, lepton, particles):
        '''Compute the isolation for lepton, using particles.
        returns an IsolationInfo.
        '''
        isolation = IsolationInfo(self.label, lepton)
        for ptc in particles:
            if ptc is lepton:
                continue
            if ptc.e()<self.e_thresh or \
               ptc.pt()<self.pt_thresh:
                continue
            is_on = False
            for area in self.on_areas:
                if area.is_inside(lepton.eta(), lepton.phi(),
                                  ptc.eta(), ptc.phi() ):
                    is_on = True
                    break
            if not is_on:
                continue        
            for area in self.off_areas:
                if area.is_inside(lepton.eta(), lepton.phi(),
                                  ptc.eta(), ptc.phi() ):
                    is_on = False
                    break
            if is_on:
                isolation.add_particle(ptc)        
        return isolation
    
