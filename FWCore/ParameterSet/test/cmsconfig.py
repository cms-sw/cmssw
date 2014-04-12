#------------------------------------------------------------
#
#
# cmsconfig: a class to provide convenient access to the Python form
# of a parsed CMS configuration file.
#
# Note: we have not worried about security. Be careful about strings
# you put into this; we use a naked 'eval'!
#
#------------------------------------------------------------
#
# Tests for this class need to be run 'by hand', until we figure out
# how to make use of the SCRAMV1 tools for running tests on Python
# code.
#
#------------------------------------------------------------
# TODO: We need some refactoring to handle the writing of module-like
# objects more gracefully. Right now, we have to pull the classname
# out of the dictionary in more than one place. Try making a class
# which represents the module, which contains the dictionary now used,
# and knows about special features: the class name, now to print the
# guts without repeating the classname, etc.
#------------------------------------------------------------

import cStringIO
import types

# TODO: Refactor pset_dict_to_string and class printable_parameter to
# have a consistent view of the problem. Perhaps have a class
# representing the configuration data for a PSet object, rather than
# just using a dictionary instance. See also __write_module_guts,
# which should be refactored at the same time.

def pset_dict_to_string(psetDict):
    """Convert dictionary representing a PSet to a string consistent
    with the configuration grammar."""
    stream = cStringIO.StringIO()
    stream.write('\n{\n')

    for name, value in psetDict.iteritems():
        stream.write('%s' % printable_parameter(name, value))
        stream.write('\n')        
    
    stream.write('}\n')
    return stream.getvalue()


def secsource_dict_to_string(secSourceDict):
    """Make a string representing the secsource"""
    stream = cStringIO.StringIO()
    stream.write("%s\n{\n" %  secSourceDict["@classname"][2])
    for name, value in secSourceDict.iteritems():
        if name[0] != '@':
            stream.write('%s' % printable_parameter(name, value))
            stream.write('\n')

    stream.write('}\n')
    return stream.getvalue()


class printable_parameter:
    """A class to provide automatic unpacking of the tuple (triplet)
    representation of a single parameter, suitable for printing.

    Note that 'value' may in fact be a list."""
    
    def __init__(self, aName, aValueTuple):
        self.name = aName
        self.type, self.trackedCode, self.value = aValueTuple
        # Because the configuration grammar treats tracked status as
        # the default, we only have to write 'untracked' as the
        # tracking code if the parameter is untracked.        
        if self.trackedCode == "tracked":
            self.trackedCode = ""
        else:
            self.trackedCode = "untracked " # trailing space is needed

        # We need special handling of some of the parameter types.
        if self.type in ["vbool", "vint32", "vuint32", "vdouble", "vstring", "VInputTag", "VESInputTag"]:   
            # TODO: Consider using cStringIO, if this is observed
            # to be a bottleneck. This may happen if many large
            # vectors are used in parameter sets.
            temp = '{'
            # Write out values as a comma-separated list
            temp += ", ".join(self.value)
            temp += '}'
            self.value = temp

        if self.type == "PSet":
            self.value = pset_dict_to_string(self.value)
        if self.type == "secsource":
            self.value = secsource_dict_to_string(self.value)
        if self.type == "VPSet":
            temp = '{'
            tup = [ pset_dict_to_string(x) for x in self.value ]
            temp += ", ".join( tup )
            temp += '}'
            self.value = temp

    def __str__(self):
        """Print this parameter in the right format for a
        configuration file."""
        s = "%(trackedCode)s%(type)s %(name)s = %(value)s" % self.__dict__
        return s    

# I'm not using new-style classes, because I'm not sure that we can
# rely on a new enough version of Python to support their use.

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
        return len(self.outputModuleNames())

    def moduleNames(self):
        """Return the names of modules. Returns a list."""
        return self.psdata['modules'].keys()

    def module(self, name):
        """Get the module with this name. Exception raised if name is
        not known. Returns a dictionary."""
        return self.psdata['modules'][name]

    def psetNames(self):
        """Return the names of psets. Returns a list."""
        return self.psdata['psets'].keys()

    def pset(self, name):
        """Get the pset with this name. Exception raised if name is
        not known. Returns a dictionary."""
        return self.psdata['psets'][name]

    def outputModuleNames(self):
        return self.psdata['output_modules']

    def moduleNamesWithSecSources(self):
        return self.psdata['modules_with_secsources']

    def esSourceNames(self):
        """Return the names of all ESSources. Names are of the form '<C++ type>@<label>' where
        label can be empty. Returns a list."""
        return self.psdata['es_sources'].keys()

    def esSource(self, name):
        """Get the ESSource with this name. Exception raised if name is
        not known. Returns a dictionary."""
        return self.psdata['es_sources'][name]

    def esModuleNames(self):
        """Return the names of all ESModules. Names are of the form '<C++ type>@<label>' where
        label can be empty. Returns a list."""
        return self.psdata['es_modules'].keys()

    def esModule(self, name):
        """Get the ESModule with this name. Exception raised if name is
        not known. Returns a dictionary."""
        return self.psdata['es_modules'][name]

    def esPreferNames(self):
        """Return the names of all es_prefer statements. Names are of the form 'esprefer_<C++ type>@<label>' where
        label can be empty. Returns a list."""
        return self.psdata['es_prefers'].keys()

    def esPrefer(self, name):
        """Get the es_prefer statement with this name. Exception raised if name is
        not known. Returns a dictionary."""
        return self.psdata['es_prefers'][name]

    def serviceNames(self):
        """Return the names of all Services. Names are actually the C++ class names
        Returns a list."""
        return self.psdata['services'].keys()

    def service(self, name):
        """Get the Service with this name. Exception raised if name is
        not known. Returns a dictionary."""
        return self.psdata['services'][name]

    def pathNames(self):
        return self.psdata['paths'].keys()

    def path(self, name):
        """Get the path description for the path of the given
        name. Exception raised if name is not known. Returns a
        string."""
        return self.psdata['paths'][name]

    def schedule(self):
        return self.psdata['schedule']

    def sequenceNames(self):
        return self.psdata['sequences'].keys()

    def sequence(self, name):
        """Get the sequence description for the sequence of the given
        name. Exception raised if name is not known. Returns a
        string."""
        return self.psdata['sequences'][name]

    def endpathNames(self):
        return self.psdata['endpaths'].keys()

    def endpath(self, name):
        """Return the endpath description, as a string."""
        return self.psdata['endpaths'][name]

    def mainInputSource(self):
        """Return the description of the main input source, as a
        dictionary."""
        return self.psdata['main_input']

    def looper(self):
        """Return the description of the looper, as a
        dictionary."""
        return self.psdata['looper']

    def procName(self):
        """Return the process name, a string"""
        return self.psdata['procname']

    def asConfigurationString(self):
        """Return a string conforming to the configuration file
        grammar, encoding this configuration."""

        # Let's try to make sure we lose no resources if something
        # fails in formatting...
        result = ""
        
        try:
            stream = cStringIO.StringIO()
            self.__write_self_to_stream(stream)
            result = stream.getvalue()

        finally:
            stream.close()
        
        return result

    def asPythonString(self):
       """Return a string containing the python psdata source of
       this object to facilitate saving and loading of python format"""
       result = "#!/usr/bin/env python\n"
       result += str(self.psdata)
       return result 

    def __write_self_to_stream(self, fileobj):
        """Private method.
        Return None.
        Write the contents of self to the file-like object fileobj."""

        # Write out the process block
        fileobj.write('process %s = \n{\n' % self.procName())
        self.__write_process_block_guts(fileobj)
        fileobj.write('}\n')

    def __write_process_block_guts(self, fileobj):
        """Private method.
        Return None.
        Write the guts of the process block to the file-like object
        fileobj."""

        # TODO: introduce, and deal with, top-level PSet objects and
        # top-level block objects.        
        self.__write_main_source(fileobj)
        self.__write_looper(fileobj)
        self.__write_psets(fileobj)
        self.__write_es_sources(fileobj)        
        self.__write_es_modules(fileobj)
        self.__write_es_prefers(fileobj)
        self.__write_modules(fileobj)
        self.__write_services(fileobj)
        self.__write_sequences(fileobj)
        self.__write_paths(fileobj)
        self.__write_endpaths(fileobj)
        self.__write_schedule(fileobj)

    def __write_psets(self, fileobj):
        """Private method.
        Return None
        Write all the psets to the file-like object fileobj."""
        for name in self.psetNames():
            psettuple = self.pset(name)
            # 8/2006: Wasn't writing trackedness!  Just re-use code
            # for embedded PSets
            fileobj.write('%s' % printable_parameter(name, psettuple))
            #fileobj.write("PSet %s = \n{\n" % (name) )
            #psetdict = psettuple[2]
            #self.__write_module_guts(psetdict, fileobj)
            #fileobj.write('}\n')

    def __write_modules(self, fileobj):
        """Private method.
        Return None
        Write all the modules to the file-like object fileobj."""
        for name in self.moduleNames():
            moddict = self.module(name)
            fileobj.write("module %s = %s\n{\n" % (name, moddict['@classname'][2]))
            self.__write_module_guts(moddict, fileobj)
            fileobj.write('}\n')

    def __write_es_sources(self, fileobj):
        """Private method.
        Return None
        Write all ESSources to the file-like object
        fileobj."""
        for name in self.esSourceNames():
            es_source_dict = self.esSource(name)
            fileobj.write("es_source %s = %s\n{\n" % (es_source_dict['@label'][2], es_source_dict['@classname'][2]))
            self.__write_module_guts(es_source_dict, fileobj)
            fileobj.write('}\n')

    def __write_es_modules(self, fileobj):
        """Private method.
        Return None
        Write all ESModules to the file-like object
        fileobj."""
        for name in self.esModuleNames():
            es_mod_dict = self.esModule(name)
            fileobj.write("es_module %s = %s\n{\n" % (es_mod_dict['@label'][2], es_mod_dict['@classname'][2]))
            self.__write_module_guts(es_mod_dict, fileobj)
            fileobj.write('}\n')

    def __write_es_prefers(self, fileobj):
        """Private method.
        Return None
        Write all es_prefer statements to the file-like object
        fileobj."""
        for name in self.esPreferNames():
            es_mod_dict = self.esPrefer(name)
            fileobj.write("es_prefer %s = %s\n{\n" % (es_mod_dict['@label'][2], es_mod_dict['@classname'][2]))
            self.__write_module_guts(es_mod_dict, fileobj)
            fileobj.write('}\n')

    def __write_services(self, fileobj):
        """Private method.
        Return None
        Write all Services to the file-like object
        fileobj."""
        for name in self.serviceNames():
            es_mod_dict = self.service(name)
            fileobj.write("service = %s\n{\n" % (es_mod_dict['@classname'][2]))
            self.__write_module_guts(es_mod_dict, fileobj)
            fileobj.write('}\n')

    def __write_sequences(self, fileobj):
        """Private method.
        Return None
        Write all the sequences to the file-like object fileobj."""
        for name in self.sequenceNames():
            fileobj.write("sequence %s = {%s}\n"  % (name, self.sequence(name)))


    def __write_paths(self, fileobj):
        """Private method.
        Return None
        Write all the paths to the file-like object fileobj."""
        for name in self.pathNames():
            fileobj.write("path %s = {%s}\n" % (name, self.path(name)))

        
    def __write_endpaths(self, fileobj):
        """Private method.
        Return None
        Write all the endpaths to the file-like object
        fileobj."""
        for name in self.endpathNames():
            fileobj.write("endpath %s = {%s}\n" % (name, self.endpath(name)))

    def __write_schedule(self, fileobj):
        fileobj.write("schedule = {%s}\n" % self.schedule())

    def __write_main_source(self, fileobj):
        """Private method.
        Return None
        Write the (main) source block to the file-like object
        fileobj."""
        mis = self.mainInputSource()  # this is a dictionary
        if mis:
        	fileobj.write('source = %s\n{\n' % mis['@classname'][2])
        	self.__write_module_guts(mis, fileobj)
        	fileobj.write('}\n')

    def __write_looper(self, fileobj):
        """Private method.
        Return None
        Write the looper block to the file-like object
        fileobj."""
        mis = self.looper()  # this is a dictionary
        if mis:
        	fileobj.write('looper = %s\n{\n' % mis['@classname'][2])
        	self.__write_module_guts(mis, fileobj)
        	fileobj.write('}\n')

    
    def __write_module_guts(self, moddict, fileobj):
        """Private method.
        Return None
        Print the body of the block for this 'module'. This includes
        all the dictionary contents except for the classname (because
        the classname does not appear within the block).

        NOTE: This should probably be a static method, because it doesn't
        use any member data of the object, but I'm not sure we can
        rely on a new-enough version of Python to make use of static
        methods."""
        for name, value in moddict.iteritems():
            if name[0] != '@':
                fileobj.write('%s' % printable_parameter(name, value))
                fileobj.write('\n')

            
        
if __name__ == "__main__":
    from sys import argv
    filename = "complete.pycfg"
    if len(argv) > 1:
	filename = argv[1]

    txt = file(filename).read()
    cfg = cmsconfig(txt)
    print cfg.asConfigurationString()
    

