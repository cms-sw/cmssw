class ParticleDataAccessor(object):
    """ This class provides access to the underlying data model.
    """
    
    LINE_STYLE_SOLID = 0
    LINE_STYLE_DASH = 1
    LINE_STYLE_WAVE = 2
    LINE_STYLE_SPIRAL = 3
    LINE_VERTEX = 4
    
    def id(self, object):
        """ Returns an id to identify given object. 
        
        Usually it is sufficient to identify python objects directly with themselves.
        Overwrite this function if this is not true for your objects. 
        """
        return id(object)

    def particleId(self, object):
        raise NotImplementedError
        
    def isQuark(self, object):
        raise NotImplementedError
        
    def isLepton(self, object):
        raise NotImplementedError
        
    def isGluon(self, object):
        raise NotImplementedError
        
    def isBoson(self, object):
        raise NotImplementedError
    
    def color(self, object):
        raise NotImplementedError
    
    def lineStyle(self, object):
        raise NotImplementedError

    def createParticle(self):
        raise NotImplementedError

    def charge(self, object):
        raise NotImplementedError
    
    def linkMother(self, object, mother):
        raise NotImplementedError
        
    def linkDaughter(self, object, daughter):
        raise NotImplementedError
        