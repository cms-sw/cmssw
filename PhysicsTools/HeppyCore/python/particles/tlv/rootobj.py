from itertools import count
import copy

class RootObj(object):
    '''Base class for all objects based on ROOT,
    typically created on the fly in analysis code instead 
    of being read from an EDM file.'''

    _ids = count(0)
    
    def __init__(self, *args, **kwargs):
        super(RootObj, self).__init__(*args, **kwargs)
        self._objid = self._ids.next()

    def __eq__(self, other):
        '''compares two objects for equality. 
        True if object id is the same. 
        So if an object is copied, the two copies are equal.
        '''
        return self._objid == other._objid
    
    def __hash__(self):
        '''returns a hash built on the object id. 
        '''
        return hash( self._objid )

                
