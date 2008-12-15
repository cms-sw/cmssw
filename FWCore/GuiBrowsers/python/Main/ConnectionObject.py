class Plug(object):
    def __init__(self,label="",connection=None):
        """ constructor """
        self.label=label
        self.connection=connection

class Connection(object):
    def __init__(self):
        """ constructor """
        self.sink=None
        self.source=None
        
    def setSink(self,sink,name):
        self.sink=sink
        sink.sinks+=[Plug(name,self)]

    def setSource(self,source,name):
        self.source=source
        source.sources+=[Plug(name,self)]
