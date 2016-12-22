from pyLCIO import IOIMPL

class Events(object):
    
    def __init__(self, filename, dummy):
        self.reader = IOIMPL.LCFactory.getInstance().createLCReader() 
        self.reader.open(filename)
        
    def __len__(self):
        return self.reader.getNumberOfEvents()

    def __getattr__(self, key):
        return getattr(self.events, key)

    def __iter__(self):
        return iter(self.reader)
