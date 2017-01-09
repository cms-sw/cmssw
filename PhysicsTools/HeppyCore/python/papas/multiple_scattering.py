import math
from scipy import constants
from numpy import sign
from ROOT import TLorentzVector, TVector3
import PhysicsTools.HeppyCore.statistics.rrandom as random

from PhysicsTools.HeppyCore.papas.path import Helix
from PhysicsTools.HeppyCore.papas.pfobjects import Particle

# propagate untill surface
#_______________________________________________________________________________
# find t_scat, time when scattering :

def multiple_scattering( particle, detector_element, field ):
    '''This function computes the scattering of a particle while propagating through the detector.
    
    As described in the pdg booklet, Passage of particles through matter, multiple scattering through small angles.
    the direction of a charged particle is modified.
    
    This function takes a particle (that has been propagated until the detector element
    where it will be scattered) and the detector element responsible for the scattering.
    The magnetic field has to be specified in order to create the new trajectory.
    
    Then this function computes the new direction, randomly choosen according to
    Moliere's theory of multiple scattering (see pdg booklet) and replaces the
    initial path of the particle by this new scattered path.
    
    The particle can now be propagated in the next part of the detector.
    '''

    if not particle.q():
        return
    # reject particles that could not be extrapolated to detector element
    # (particle created too late, out of the detector element)
    surface_in = '{}_in'.format(detector_element.name)
    surface_out = '{}_out'.format(detector_element.name)
    if not surface_in in particle.path.points or \
        not surface_out in particle.path.points:
        return
    
    #TODOCOLIN : check usage of private attributes
    in_point = particle.path.points[surface_in]
    out_point = particle.path.points[surface_out]
    phi_in = particle.path.phi( in_point.X(), in_point.Y())
    phi_out = particle.path.phi( out_point.X(), out_point.Y())
    t_scat = particle.path.time_at_phi((phi_in+phi_out)*0.5)
    # compute p4_t = p4 at t_scat :
    p4_0 = particle.path.p4.Clone()
    p4tx = p4_0.X()*math.cos(particle.path.omega*t_scat)\
           + p4_0.Y()*math.sin(particle.path.omega*t_scat)
    p4ty =-p4_0.X()*math.sin(particle.path.omega*t_scat)\
           + p4_0.Y()*math.cos(particle.path.omega*t_scat)
    p4tz = p4_0.Z()
    p4tt = p4_0.T()
    p4_t = TLorentzVector(p4tx, p4ty, p4tz, p4tt)

    # now, p4t will be modified with respect to the multiple scattering
    # first one has to determine theta_0 the width of the gaussian :
    P = p4_t.Vect().Dot(p4_t.Vect().Unit())
    deltat = particle.path.time_at_phi(phi_out)-particle.path.time_at_phi(phi_in)
    x = abs(particle.path.path_length(deltat))
    X_0 = detector_element.material.x0

    theta_0 = 1.0*13.6e-3/(1.0*particle.path.speed/constants.c*P)*abs(particle.path.charge)
    theta_0 *= (1.0*x/X_0)**(1.0/2)*(1+0.038*math.log(1.0*x/X_0))

    # now, make p4_t change due to scattering :
    theta_space = random.gauss(0, theta_0*2.0**(1.0/2))
    psi = constants.pi*random.uniform(0,1) #double checked
    p3i = p4_t.Vect().Clone()
    e_z = TVector3(0,0,1)
    #first rotation : theta, in the xy plane
    a = p3i.Cross(e_z)
    #this may change the sign, but randomly, as the sign of theta already is
    p4_t.Rotate(theta_space,a)
    #second rotation : psi (isotropic around initial direction)
    p4_t.Rotate(psi,p3i.Unit())

    # creating new helix, ref at scattering point :
    helix_new_t = Helix(field, particle.path.charge, p4_t,
                        particle.path.point_at_time(t_scat))

    # now, back to t=0
    p4sx = p4_t.X()*math.cos(-particle.path.omega*t_scat)\
           + p4_t.Y()*math.sin(-particle.path.omega*t_scat)
    p4sy =-p4_t.X()*math.sin(-particle.path.omega*t_scat)\
           + p4_t.Y()*math.cos(-particle.path.omega*t_scat)
    p4sz = p4_t.Z()
    p4st = p4_t.T()
    p4_scat = TLorentzVector(p4sx, p4sy, p4sz, p4st)

    # creating new helix, ref at new t0 point :
    helix_new_0 = Helix(field, particle.path.charge, p4_scat,
                        helix_new_t.point_at_time(-t_scat))

    # replacing the particle's path with the scatterd one :
    particle.set_path(helix_new_0, option = 'w')
        
        
