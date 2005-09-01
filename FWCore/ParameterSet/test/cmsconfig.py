#------------------------------------------------------------
#
# cmsconfig: a class to provide convenient access to the Python form
# of a parsed CMS configuration file.
#
# Note: we have not worried about security. Be careful about strings
# you put into this; we use a naked 'eval'!
#
#------------------------------------------------------------

class cmsconfig:
    """A class to provide convenient access to the contents of a
    parsed CMS configuration file."""
    
    def __init__(self, stringrep):
        """Create a cmsconfig object from the contents of the (Python)
        exchange format for configuration files."""
        self.psdata = eval(stringrep)

    def numberOfModules(self):
        return len(self.psdata['modules'])

    def numberOfOutputModules(self):
        return len(self.getOutputModuleNames())

    def moduleNames(self):
        return self.psdata['modules'].keys()

    def module(self, name):
        """Get the module with this name. Exception raised if name is
        not known. Returns a dictionary."""
        return self.psdata['modules'][name]

    def outputModuleNames(self):
        return self.psdata['output_modules']

    def pathNames(self):
        return self.psdata['paths'].keys()

    def path(self, name):
        """Get the path description for the path of the given
        name. Exception raised if name is not known. Returns a
        string."""
        return self.psdata['paths'][name]

    def sequenceNames(self):
        return self.psdata['sequences'].keys()

    def sequence(self, name):
        """Get the sequence description for the sequence of the given
        name. Exception raised if name is not known. Returns a
        string."""
        return self.psdata['sequences'][name]

    def endpath(self):
        """Return the endpath description, as a string."""
        return self.psdata['endpath']

    def mainInputSource(self):
        """Return the description of the main input source, as a
        dictionary."""
        return self.psdata['main_input']

    def procName(self):
        """Return the process name, a string"""
        return self.psdata['procname']

        
        
