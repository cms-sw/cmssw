import collections

class Event(object):
    '''Event class.

    The Looper passes the Event object to each of its Analyzers, which in turn can:
    - read some information
    - add more information
    - modify existing information.

    The attributes of the Event object are dynamically modified as allowed by python.
    The new attributes can be of any type.'''
    
    def __init__(self, iEv):
        self.iEv = iEv
        #WARNING do I really want to define the weight here?
        self.eventWeight = 1
        
    def __str__(self):
        '''A clever printout :-).'''
        header = '{type}: {iEv}'.format( type=self.__class__.__name__,
                                         iEv = self.iEv)
        varlines = []
        for var,value in sorted(vars(self).iteritems()):
            # if hasattr(value, '__dict__'):
            #    value = str( vars(value) )
##             tmp = None
##             try:
##                 if str(iter( value )).startswith('<ROOT.reco::candidate'):
##                     # a single reco::Candidate appears to be iterable...
##                     # here, I want to consider it as an object, not a sequence.
##                     raise TypeError('is not a vector')
##                 tmp = map(str, value)
##             except TypeError:
##                 tmp = value
            tmp = value
            if isinstance( value, collections.Iterable ) and \
                   not isinstance( value, (str,unicode)) and \
                   not str(iter( value )).startswith('<ROOT.reco::candidate'):
                tmp = map(str, value)
            
            varlines.append( '\t{var:<15}:   {value}'.format(var=var, value=tmp) )
        all = [ header ]
        all.extend(varlines)
        return '\n'.join( all )

