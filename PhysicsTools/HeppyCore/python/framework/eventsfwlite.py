from DataFormats.FWLite import Events as FWLiteEvents

class Events(object):
    def __init__(self, files, tree_name):
        self.events = FWLiteEvents(files)

    def __len__(self):
        return self.events.size()

    def __getattr__(self, key):
        return getattr(self.events, key)

    def __getitem__(self, iEv):
        self.events.to(iEv)
        return self
