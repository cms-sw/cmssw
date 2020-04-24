import collections
from ROOT import TChain

class Event(object):
    '''Event class.

    The Looper passes the Event object to each of its Analyzers,
    which in turn can:
    - read some information
    - add more information
    - modify existing information.

    Attributes:
      iEv = event processing index, starting at 0
      eventWeight = a weight, set to 1 at the beginning of the processing
      input = input, as determined by the looper
    #TODO: provide a clear interface for access control (put, get, del products) - we should keep track of the name and id of the analyzer.
    '''

    def __init__(self, iEv, input_data=None, setup=None, eventWeight=1 ):
        self.iEv = iEv
        self.input = input_data
        self.setup = setup
        self.eventWeight = eventWeight

    def __str__(self):
        header = '{type}: {iEv}'.format( type=self.__class__.__name__,
                                         iEv = self.iEv)
        varlines = []
        for var,value in sorted(vars(self).iteritems()):
            tmp = value
            # check for recursivity
            recursive = False
            if hasattr(value, '__getitem__') and \
               not isinstance(value, collections.Mapping) and \
               (len(value)>0 and value[0].__class__ == value.__class__):
                    recursive = True
            if hasattr(value, '__contains__') and \
                   not isinstance(value, (str,unicode)) and \
                   not isinstance(value, TChain) and \
                   not recursive :
                tmp = map(str, value)

            varlines.append( '\t{var:<15}:   {value}'.format(var=var, value=tmp) )
        all = [ header ]
        all.extend(varlines)
        return '\n'.join( all )
