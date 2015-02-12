from DataFormats.FWLite import Events as FWLiteEvents

class Events(object):
    def __init__(self, files, tree_name,  **kwargs):
	if "options" in kwargs :
		options=kwargs["options"]
		if not hasattr(options,"inputFiles"):
		 	options.inputFiles=files
		if not hasattr(options,"maxEvents"):
			options.maxEvents = 0	
		if not hasattr(options,"secondaryInputFiles"):
			options.secondaryInputFiles = []
	        self.events = FWLiteEvents(options=options)
	else :
	        self.events = FWLiteEvents(files)

    def __len__(self):
        return self.events.size()

    def __getattr__(self, key):
        return getattr(self.events, key)

    def __getitem__(self, iEv):
        self.events.to(iEv)
        return self
