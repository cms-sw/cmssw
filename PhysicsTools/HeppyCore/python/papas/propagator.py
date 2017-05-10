from vectors import Point
import math
import copy
from ROOT import TVector3
from geotools import circle_intersection
from papas_exceptions import PropagationError
from path import Helix, StraightLine

class Info(object):
    pass

class Propagator(object):

    def propagate(self, particles, cylinders, *args, **kwargs):
        for ptc in particles:
            for cyl in cylinders:
                self.propagate_one(ptc, cyl, *args, **kwargs)
                
                
class StraightLinePropagator(Propagator):        

    def propagate_one(self, particle, cylinder, dummy=None):
        line = StraightLine(particle.p4(), particle.vertex) 
        particle.set_path( line ) # TODO 
        theta = line.udir.Theta()
        if abs(line.origin.Z()) > cylinder.z or \
           line.origin.Perp() > cylinder.rad:
            return # particle created outside the cylinder
        if line.udir.Z(): 
            destz = cylinder.z if line.udir.Z() > 0. else -cylinder.z
            length = (destz - line.origin.Z())/math.cos(theta)
            if length < 0:
                print 'HERE!!'
                import pdb; pdb.set_trace()
                raise PropagationError(particle)
            destination = line.origin + line.udir * length
            rdest = destination.Perp()
            if rdest > cylinder.rad:
                udirxy = TVector3(line.udir.X(), line.udir.Y(), 0.)
                originxy = TVector3(line.origin.X(), line.origin.Y(), 0.)
                # solve 2nd degree equation for intersection
                # between the straight line and the cylinder
                # in the xy plane to get k,
                # the propagation length
                a = udirxy.Mag2()
                b= 2*udirxy.Dot(originxy)
                c= originxy.Mag2()-cylinder.rad**2
                delta = b**2 - 4*a*c
                if delta<0:
                    return 
                    # raise PropagationError(particle)
                km = (-b - math.sqrt(delta))/(2*a)
                # positive propagation -> correct solution.
                kp = (-b + math.sqrt(delta))/(2*a)
                # print delta, km, kp
                destination = line.origin + line.udir * kp  
        #TODO deal with Z == 0 
        #TODO deal with overlapping cylinders
        particle.points[cylinder.name] = destination

        
class HelixPropagator(Propagator):
    
    def propagate_one(self, particle, cylinder, field, debug_info=None):
        helix = Helix(field, particle.q(), particle.p4(),
                      particle.vertex)
        particle.set_path(helix)
        is_looper = helix.extreme_point_xy.Mag() < cylinder.rad
        is_positive = particle.p4().Z() > 0.
        if not is_looper:
            try: 
                xm, ym, xp, yp = circle_intersection(helix.center_xy.X(),
                                                     helix.center_xy.Y(),
                                                     helix.rho,
                                                     cylinder.rad )
            except ValueError:
                return
                # raise PropagationError(particle)
            # particle.points[cylinder.name+'_m'] = Point(xm,ym,0)
            # particle.points[cylinder.name+'_p'] = Point(xp,yp,0)
            phi_m = helix.phi(xm, ym)
            phi_p = helix.phi(xp, yp)
            dest_time = helix.time_at_phi(phi_p)
            destination = helix.point_at_time(dest_time)
            if destination.Z()*helix.udir.Z()<0.:
                dest_time = helix.time_at_phi(phi_m)
                destination = helix.point_at_time(dest_time)
            if abs(destination.Z())<cylinder.z:
                particle.points[cylinder.name] = destination
            else:
                is_looper = True
        if is_looper:
            # extrapolating to endcap
            destz = cylinder.z if helix.udir.Z() > 0. else -cylinder.z
            dest_time = helix.time_at_z(destz)
            destination = helix.point_at_time(dest_time)
            # destz = cylinder.z if positive else -cylinder.z
            particle.points[cylinder.name] = destination

            
        info = Info()
        info.is_positive = is_positive
        info.is_looper = is_looper
        return info
        
straight_line = StraightLinePropagator()

helix = HelixPropagator() 
