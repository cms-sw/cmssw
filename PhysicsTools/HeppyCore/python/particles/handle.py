import copy

class Handle(object):
    '''Extends the Handle functionalities.

    This class wraps a C++ Handle.
    The user can call all functions of the C++ Handle,
    and can also attach new attributes to objects from this class.
    '''

    def __init__(self, handle):
        self.handle = handle
        super(Handle, self).__init__()

    def __copy__(self):
        '''Very dirty trick, the handle is deepcopied...'''
        handle = copy.deepcopy( self.handle )
        newone = type(self)(handle)
        newone.__dict__.update(self.__dict__)
        newone.handle = handle
        return newone        
        
    def __getattr__(self,name):
        '''all accessors  from cmg::DiTau are transferred to this class.'''
        return getattr(self.handle, name)

    def __eq__(self,other):
        if( hasattr(other, 'handle') ):
            # the two python Handles have the same C++ Handle
            return self.handle == other.handle
        else:
            # can compare a python Handle with a cpp Handle directly
            return self.handle == other 

