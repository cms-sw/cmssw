import ROOT as rt

class RootIterator(object):
    """A wrapper around the ROOT iterator so that it can be used in python"""
    def __init__(self, o):
        if hasattr(o,'Class') and o.Class().InheritsFrom('TIterator'):
            self.iter = o
        elif hasattr(o,'createIterator'):
            self.iter = o.createIterator()
        elif hasattr(o,'MakeIterator'):
            self.iter = o.MakeIterator()
        elif hasattr(o,'componentIterator'):
            self.iter = o.componentIterator()
        else:
            self.iter = None
    def __iter__(self):
        return self
    def next(self):
        n = self.iter.Next()
        if n is None or not n:
            raise StopIteration()
        return n