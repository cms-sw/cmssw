#ROOTTOOLS
from DataFormats.FWLite import Events, Handle
        

class AutoHandle( Handle, object ):
    '''Handle + label.'''

    handles = {}
    
    def __init__(self, label, type, mayFail=False, fallbackLabel=None):
        '''Note: label can be a tuple : (module_label, collection_label, process)'''
        self.label = label
        self.fallbackLabel = fallbackLabel
        self.type = type
        self.mayFail = mayFail
        Handle.__init__(self, self.type)
    def product(self):
	if not self.isLoaded :
		self.ReallyLoad(self.event)
		self.isLoaded=True
	return super(AutoHandle,self).product()

    def Load(self, event):  #is actually a reset state
	self.event=event
 	self.isLoaded=False

    def ReallyLoad(self, event):
        '''Load self from a given event.

        Call this function, and then just call self.product() to get the collection'''
        try:
            event.getByLabel( self.label, self)
            if not self.isValid(): raise RuntimeError    
        except RuntimeError:
            Handle.__init__(self, self.type) # must re-init, since otherwise after a failure it becomes unusable
            errstr = '''
            Cannot find collection with:
            type = {type}
            label = {label}
            '''.format(type = self.type, label = self.label)
            if not self.mayFail and self.fallbackLabel == None:
                raise Exception(errstr)
            if self.fallbackLabel != None:
                try:
                    event.getByLabel( self.fallbackLabel, self)
                    if not self.isValid(): raise RuntimeError
                    ## if I succeeded, swap default and fallback assuming that the next event will be like this one
                    self.fallbackLabel, self.label = self.label, self.fallbackLabel
                except RuntimeError:
                    Handle.__init__(self, self.type) # must re-init, since otherwise after a failure it becomes unusable
                    errstr = '''
                    Cannot find collection with:
                    type = {type}
                    label = {label} or {lab2}
                    '''.format(type = self.type, label = self.label, lab2 = self.fallbackLabel)
                    if not self.mayFail:
                        raise Exception(errstr)


