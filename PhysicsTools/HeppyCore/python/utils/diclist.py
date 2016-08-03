# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

class diclist( list ):
    '''list with an internal dictionary for indexing, 
    allowing to keep dictionary elements ordered. 
    keys can be everything except an integer.
    '''

    def __init__(self):
        super( diclist, self).__init__()
        # internal dictionary, will contain key -> index in list
        self.dico = {}

    def add( self, key, value ):
        if isinstance(key, (int, long)):
            raise ValueError("key cannot be an integer")
        if key in self.dico:
            raise ValueError("key '{key}' already exists".format(key=key) )
        self.dico[key] = len(self)
        self.append(value)

    def __getitem__(self, index):
        '''index can be a dictionary key, or an integer specifying 
        the rank of the value to be accessed
        '''
        try:
            # if index is an integer (the rank), use the list. 
            return super(diclist, self).__getitem__(index)
        except TypeError, ValueError:
            # else it's the dictionary key.
            # use the internal dictionary to get the index, 
            # and return the corresponding value from the list
            return super(diclist, self).__getitem__( self.dico[index] )
            
    def __setitem__(self, index, value):
        '''These functions are quite risky...'''
        try:
            return super(diclist, self).__setitem__(index, value)
        except TypeError as ValueError:
            return super(diclist, self).__setitem__( self.dico[index], value )
            

    
