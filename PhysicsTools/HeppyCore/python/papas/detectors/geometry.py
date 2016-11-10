
class SurfaceCylinder(object):

    def __init__(self, name, rad, z):
        self.name = name
        self.rad = rad
        self.z = z

    def __str__(self):
        return '{} : {}, R={:5.2f}, z={:5.2f}'.format(
            self.__class__.__name__,
            self.name,
            self.rad,
            self.z
            )

            
class VolumeCylinder(object):
    '''Implement sub even for pipes, and consistency test: all space must be filled.'''
    
    def __init__(self, name, orad, oz, irad=None, iz=None):
        if not isinstance(name, basestring):
            raise ValueError('first parameter must be a string')
        self.name = name
        self.outer = SurfaceCylinder('_'.join([self.name, 'out']), orad, oz)
        self.inner = None
        if irad and iz: 
            if irad > orad:
                raise ValueError('outer radius of subtracted cylinder must be smaller')
            if iz > oz :
                raise ValueError('outer z of subtracted cylinder must be smaller')
            if irad is None or iz is None:
                raise ValueError('must specify both irad and iz.')    
            self.inner = SurfaceCylinder('_'.join([self.name, 'in']), irad, iz)

    def contains(self, point):
        perp = point.Perp()
        if abs(point.Z())<self.inner.z:
            return perp >= self.inner.rad and perp < self.outer.rad
        elif abs(point.Z())<self.outer.z:
            return perp < self.outer.rad
        else:
            return False
