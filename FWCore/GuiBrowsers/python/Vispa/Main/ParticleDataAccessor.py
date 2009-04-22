class ParticleDataAccessor(object):
    """ This class provides access to the underlying data model.
    """
    
    def id(self, object):
        """ Return the particle id.
        """
        raise NotImplementedError

    def isQuark(self, object):
        raise NotImplementedError
        
    def isLepton(self, object):
        raise NotImplementedError
        
    def isGluon(self, object):
        raise NotImplementedError
        
    def isBoson(self, object):
        raise NotImplementedError
        
