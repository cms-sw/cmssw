import math
from scipy import constants
from ROOT import TVector3, TLorentzVector
from PhysicsTools.HeppyCore.utils.deltar import deltaPhi
from collections import OrderedDict
import scipy.optimize as opti # need to compute impact parameters
from numpy import sign
import PhysicsTools.HeppyCore.statistics.rrandom as random

class Path(object):
    '''Path followed by a particle in 3D space. 
    Assumes constant speed magnitude both along the z axis and in the transverse plane.
    '''
        
    def __init__(self, p4, origin):
        self.p4 = p4
        self.udir = p4.Vect().Unit()
        self.origin = origin
        self.speed = self.p4.Beta() * constants.c
        self.points = OrderedDict()
        self.points['vertex'] = origin

    def time_at_z(self, z):
        dest_time = (z - self.origin.Z())/self.vz()
        return dest_time

    def deltat(self, path_length):
        '''Time needed to follow a given path length'''
        return path_length / self.speed

    def point_at_time(self, time):
        '''Returns the 3D point on the path at a given time'''
        return self.origin + self.udir * self.speed * time
        
    def vz(self):
        '''Speed magnitude along z axis'''
        return self.p4.Beta() * constants.c * self.udir.Z()

    def vperp(self):
        '''Speed magnitude in the transverse plane'''
        return self.p4.Beta() * constants.c * self.udir.Perp()

    
class StraightLine(Path):
    pass
    
    
class Helix(Path):
    def __init__(self, field, charge, p4, origin):
        super(Helix, self).__init__(p4, origin)
        self.charge = charge
        self.rho = p4.Perp() / (abs(charge)*field) * 1e9/constants.c
        self.v_over_omega = p4.Vect()
        self.v_over_omega *= 1./(charge*field)*1e9/constants.c
        self.omega = charge*field*constants.c**2 / (p4.M()*p4.Gamma()*1e9)
        momperp_xy = TVector3(-p4.Y(), p4.X(), 0.).Unit()
        origin_xy = TVector3(origin.X(), origin.Y(), 0.)
        self.center_xy = origin_xy - charge * momperp_xy * self.rho
        self.extreme_point_xy = TVector3(self.rho, 0, 0) 
        if self.center_xy.X()!=0 or self.center_xy.Y()!=0:
            self.extreme_point_xy = self.center_xy + self.center_xy.Unit() * self.rho
        # calculate phi range with the origin at the center,
        # for display purposes
        center_to_origin = origin_xy - self.center_xy
        self.phi0 = center_to_origin.Phi()
        self.phi_min = self.phi0 * 180 / math.pi
        self.phi_max = self.phi_min + 360.

    def polar_at_time(self, time):
        z = self.vz() * time + self.origin.Z()
        rho = self.rho
        phi = - self.omega * time + self.phi0
        return rho, z, phi

    def time_at_phi(self, phi):
        time = deltaPhi(self.phi0, phi) / self.omega
        return time

    def phi(self, x, y):
        xy = TVector3(x,y,0)
        xy -= self.center_xy
        return xy.Phi()
        
    def point_from_polar(self, polar):
        rho,z,phi = polar
        xy = self.center_xy + self.rho * TVector3(math.cos(phi), math.sin(phi), 0)
        return TVector3(xy.X(), xy.Y(), z)
        
    def point_at_time(self, time):
        '''return a TVector3 with cartesian coordinates at time t'''
        x,y,z = self.coord_at_time(time)
        return TVector3(x, y, z)
    
    def path_length(self, deltat):
        '''ds2 = dx2+dy2+dz2 = [w2rho2 + vz2] dt2'''
        return math.sqrt(self.omega**2 * self.rho**2 + self.vz()**2)*deltat
 
 #______________________________________________________________________________   
    def coord_at_time(self, time):
        '''returns x,y,z at time t'''
        x = self.origin.X() + \
            self.v_over_omega.Y() * (1-math.cos(self.omega*time)) \
            + self.v_over_omega.X() * math.sin(self.omega*time)
        y = self.origin.Y() - \
            self.v_over_omega.X() * (1-math.cos(self.omega*time)) \
            + self.v_over_omega.Y() * math.sin(self.omega*time)
        z = self.vz() * time + self.origin.Z()
        return x,y,z
        
    def compute_IP(self, vertex,jet):
        '''find the impact parameter of the trajectory with respect to a given
        point (vertex). The impact parameter has the same sign as the scalar product of
        the vector pointing from the given vertex to  the point of closest
        approach with the given jet direction.
        
        new attributes :
        *   closest_t = time of closest approach to the primary vertex.
        *   IP = signed impact parameter
        *   IPcoord = TVector3 of the point of closest approach to the
            primary vertex
        '''
        self.vertex_IP = vertex
        def distquad (time):
            x,y,z = self.coord_at_time(time)
            dist2 = (x-vertex.x())**2 + (y-vertex.y())**2\
            + (z-vertex.z())**2 
            return dist2
        minim_answer = opti.bracket(distquad, xa = -0.5e-14, xb = 0.5e-14)
        self.closest_t = minim_answer[1]
        vector_IP = self.point_at_time(minim_answer[1]) - vertex
        Pj = jet.p4().Vect().Unit()
        signIP  = vector_IP.Dot(Pj)
        self.IP = minim_answer[4]**(1.0/2)*sign(signIP)
        
        x,y,z = self.coord_at_time(minim_answer[1])
        self.IPcoord = TVector3(x, y, z)
       
    def compute_theta_0(self, x, X_0):
        '''Computes the square root of the variance, sigma, of the multiple
        scattering angle due to matter interactions, using the formula in PDG
        booklet, Passage of particles through matter, multiple scattering through small angles
        equation 10.'''
        P = self.p4.Vect().Dot(self.udir)
        self.theta_0 = 1.0*13.6e-3/(1.0*self.speed/constants.c*P)
        self.theta_0 *= abs(self.charge)*(1.0*abs(x/X_0))**(1.0/2)*(1+0.038*math.log(1.0*abs(x/X_0)))
        self.xX_0 = 1.0*x/X_0
    
    def compute_IP_signif(self, IP, theta_0, scat_point):
        # ! are we sure sigma_IP_due_IP_algo_precise isnt overestimated ?
        # it is an approximation : we stay here in a plan containing the primary
        # vertex, the IP_point and the deviated one. But geometrically the new
        # IP_point isnt in that plan (cos(theta) factor ~ 1)
        delta_t = 1e-15
        delta_s = delta_t * self.speed *1.0
        sigma_s = delta_s
        sigma_IP_due_IP_algo_precise = IP*1.0/(math.cos(math.atan(sigma_s/IP)))-IP
        sigma_IP_due_other = 1e-5
        
        if theta_0 == None or scat_point == None:
            self.IP_signif = IP*1.0/(sigma_IP_due_IP_algo_precise**2+sigma_IP_due_other**2)**0.5
        else :        
            phi_t_scat = self.phi( scat_point.X(), scat_point.Y())
            t_scat = self.time_at_phi(phi_t_scat)
            fly_distance = self.speed * 1.0 * t_scat
            # for the IP significance : estimation 
            sigma_IP_due_scattering = fly_distance*math.tan((2)**0.5*theta_0)
            sigma_IP_tot = ( sigma_IP_due_IP_algo_precise**2 + sigma_IP_due_scattering**2 + sigma_IP_due_other**2 )**0.5
            self.IP_signif = IP*1.0/sigma_IP_tot
            self.IP_sigma = sigma_IP_tot
        
            
 #______________________________________________________________________________    

    # def deltat(self, path_length):
    #     #TODO: shouldn't this just use beta????
    #     d1 = path_length / (self.p4.Beta()*constants.c)
    #     # d2 = path_length / math.sqrt(self.omega**2 * self.rho**2 + self.vz()**2)
    #     return d1
        
    
if __name__ == '__main__':

    from ROOT import TLorentzVector, TVector3
    p4 = TLorentzVector()
    p4.SetPtEtaPhiM(1, 0, 0, 5.11e-4)
    helix = Helix(3.8, 1, p4, TVector3(0,0,0))
    length = helix.path_length(1e-9)
    helix.deltat(length)
