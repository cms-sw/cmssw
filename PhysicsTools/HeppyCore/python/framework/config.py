# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

from weight import Weight
import glob
import analyzer
import copy

# Forbidding PyROOT to hijack help system,
# in case the configuration module is used as a script.
import ROOT 
ROOT.PyConfig.IgnoreCommandLineOptions = True

def printComps(comps, details=False):
    '''
    Summary printout for  a list of components comps.
    The components are assumed to have a name, and a list of files,
    like the ones from this module.
    '''
    nJobs = 0
    nCompsWithFiles = 0
    for c in comps:
        if not hasattr(c, 'splitFactor'):
            c.splitFactor = 1
        print c.name, c.splitFactor, len(c.files)
        if len(c.files)==0:
            continue
        else:
            if details:
                print c.files[0]
            nJobs += c.splitFactor
            nCompsWithFiles += 1

    print '-'*70
    print '# components with files = ', nCompsWithFiles
    print '# jobs                  = ', nJobs

def split(comps):
    '''takes a list of components, split the ones that need to be splitted, 
    and return a new (bigger) list'''

    def chunks(l, n):
        '''split list l in n chunks. The last one can be smaller.'''
        return [l[i:i+n] for i in range(0, len(l), n)]

    splitComps = []
    for comp in comps:
        if hasattr( comp, 'fineSplitFactor') and comp.fineSplitFactor>1:
            subchunks = range(comp.fineSplitFactor)
            for ichunk, chunk in enumerate([(f,i) for f in comp.files for i in subchunks]):
                newComp = copy.deepcopy(comp)
                newComp.files = [chunk[0]]
                newComp.fineSplit = ( chunk[1], comp.fineSplitFactor )
                newComp.name = '{name}_Chunk{index}'.format(name=newComp.name,
                                                       index=ichunk)
                splitComps.append( newComp )
        elif hasattr( comp, 'splitFactor') and comp.splitFactor>1:
            chunkSize = len(comp.files) / comp.splitFactor
            if len(comp.files) % comp.splitFactor:
                chunkSize += 1
            # print 'chunk size',chunkSize, len(comp.files), comp.splitFactor
            for ichunk, chunk in enumerate(chunks(comp.files, chunkSize)):
                newComp = copy.deepcopy(comp)
                newComp.files = chunk
                newComp.name = '{name}_Chunk{index}'.format(name=newComp.name,
                                                            index=ichunk)
                splitComps.append( newComp )
        else:
            splitComps.append( comp )
    return splitComps


class CFG(object):
    '''Base configuration class. The attributes are used to store parameters of any type'''
    def __init__(self, **kwargs):
        '''All keyword arguments are added as attributes.'''
        self.__dict__.update( **kwargs )
        self.name = None

    def __str__(self):
        '''A useful printout'''
        header = '{type}: {name}'.format( type=self.__class__.__name__,
                                          name=self.name)
        varlines = ['\t{var:<15}:   {value}'.format(var=var, value=value) \
                    for var,value in sorted(vars(self).iteritems()) \
                    if var is not 'name']
        all = [ header ]
        all.extend(varlines)
        return '\n'.join( all )

    def clone(self, **kwargs):
        '''Make a copy of this object, redefining (or adding) some parameters, just
           like in the CMSSW python configuration files. 

           For example, you can do
              module1 = cfg.Analyzer(SomeClass, 
                          param1 = value1, 
                          param2 = value2, 
                          param3 = value3, 
                          ...)
              module2 = module1.clone(
                         param2 = othervalue,
                         newparam = newvalue)
           and module2 will inherit the configuration of module2 except for
           the value of param2, and for having an extra newparam of value newvalue
           (the latter may be useful if e.g. newparam were optional, and needed
           only when param2 == othervalue)

           Note that, just like in CMSSW, this is a shallow copy and not a deep copy,
           i.e. if in the example above value1 were to be an object, them module1 and
           module2 will share the same instance of value1, and not have two copies.
        '''
        other = copy.copy(self)
        for k,v in kwargs.iteritems():
            setattr(other, k, v)
        return other
    
class Analyzer( CFG ):
    '''Base analyzer configuration, see constructor'''
    names = set()
    
    def __init__(self, class_object, instance_label='1', 
                 verbose=False, **kwargs):
        '''
        One could for example define the analyzer configuration for a
        di-muon framework.Analyzer.Analyzer in the following way:

        ZMuMuAna = cfg.Analyzer(
          ZMuMuAnalyzer,
          'zmumu', # optional!
          pt1 = 20,
          pt2 = 20,
          iso1 = 0.1,
          iso2 = 0.1,
          eta1 = 2,
          eta2 = 2,
          m_min = 0,
          m_max = 200
        )


        The first argument is your analyzer class. 
        It should inherit from heppy.framework.analyzer.Analyser (standalone)
        or from PhysicsTools.HeppyCore.framework.analyzer (in CMS)

        The second argument is optional.
        If you have several analyzers of the same class, 
        e.g. ZEleEleAna and ZMuMuAna, 
        you may choose to provide it to keep track of the output 
        of these analyzers. 
        If you don't so so, the instance labels of the analyzers will
        automatically be set to 1, 2, etc.

        Finally, any kinds of keyword arguments can be added.
        
        This analyzer configuration object will become available 
        as self.cfg_ana in your ZMuMuAnalyzer.
        '''
        super(Analyzer, self).__init__(**kwargs)
        errmsg = None
        if type(class_object) is not type: 
            errmsg = 'The first argument should be a class'
        elif not analyzer.Analyzer in class_object.__mro__:
            try:
                #TODO: we also should be able to use analyzers
                #TODO: in PhysicsTools.HeppyCore...
                #TODO: a bit of a hack anyway, can we do something cleaner?
                from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer as CMSBaseAnalyzer
                if CMSBaseAnalyzer in class_object.__mro__:
                    errmsg = None
            except: 
                errmsg = 'The first argument should be a class inheriting from {anaclass}'.format(anaclass=analyzer.Analyzer)
        if errmsg: 
            msg = 'Error creating {selfclass} object. {errmsg}. Instead, you gave {classobjectclass}'.format( 
                selfclass=self.__class__,
                errmsg=errmsg, 
                classobjectclass=class_object )
            raise ValueError(msg)
        self.class_object = class_object
        self.instance_label = instance_label # calls _build_name
        self.verbose = verbose

    def __setattr__(self, name, value):
        '''You may decide to copy an existing analyzer and change
        its instance_label. In that case, one must stay consistent.'''
        self.__dict__[name] = value
        if name == 'instance_label':
            self.name = self._build_name()   

    def _build_name(self):
        class_name = '.'.join([self.class_object.__module__, 
                               self.class_object.__name__])
        while 1:
            # if class_name == 'heppy.analyzers.ResonanceBuilder.ResonanceBuilder':
            #    import pdb; pdb.set_trace()
            name = '_'.join([class_name, self.instance_label])
            if name not in self.__class__.names:
                self.__class__.names.add(name)
                break
            else:
                # cannot set attr directly or infinite recursion,
                # see setattr
                iinst = None
                try:
                    iinst = int(self.instance_label)
                    self.__dict__['instance_label'] = str(iinst+1)
                except ValueError:
                    # here, reloading module in ipython
                    self.__class__.names = set()
                    self.__dict__['instance_label'] = self.instance_label
        return name 

    def clone(self, **kwargs):
        other = super(Analyzer, self).clone(**kwargs)
        if 'class_object' in kwargs and 'name' not in kwargs:
            other.name = other._build_name()
        return other

    def __repr__(self):
        baserepr = super(Analyzer, self).__repr__()
        return ':'.join([baserepr, self.name])

    
class Service( CFG ):
    
    def __init__(self, class_object, instance_label='1', 
                 verbose=False, **kwargs):
        super(Service, self).__init__(**kwargs)
        self.class_object = class_object
        self.instance_label = instance_label
        self.name = self._build_name()
        self.verbose = verbose

    def _build_name(self):
        class_name = '.'.join([self.class_object.__module__, 
                               self.class_object.__name__])
        name = '_'.join([class_name, self.instance_label])
        return name 

    def __setattr__(self, name, value):
        '''You may decide to copy an existing analyzer and change
        its instance_label. In that case, one must stay consistent.'''
        self.__dict__[name] = value
        if name == 'instance_label':
            self.name = self._build_name()   

    def clone(self, **kwargs):
        other = super(Service, self).clone(**kwargs)
        if 'class_object' in kwargs and 'name' not in kwargs:
            other.name = other._build_name()
        return other


class Sequence( list ):
    '''A list with print functionalities.

    Used to define a sequence of analyzers.'''
    def __init__(self, *args):
        for arg in args:
            if isinstance(arg, list):
                self.extend(arg)
            elif not hasattr(arg, '__iter__'):
                self.append(arg)
            else:
                raise ValueError(
'''
Sequence only accepts lists or non iterable objects.
You provided an object of type {}
'''.format(arg.__class__)
                )
        
    def __str__(self):
        tmp = []
        for index, ana in enumerate( self ):
            tmp.append( '{index} :'.format(index=index) )
            tmp.append( '{ana} :'.format(ana=ana) )
        return '\n'.join(tmp)

#TODO review inheritance, and in particular constructor args - this is a mess.

class Component( CFG ):
    '''Base component class.

    See the child classes:
    DataComponent, MCComponent, EmbedComponent
    for more information.'''
    def __init__(self, name, files, tree_name=None, triggers=None, **kwargs):
        if isinstance(triggers, basestring):
            triggers = [triggers]
        if type(files) == str:
            files = sorted(glob.glob(files))
        super( Component, self).__init__( name = name,
                                          files = files,
                                          tree_name = tree_name,
                                          triggers = triggers, **kwargs)
        self.name = name 
        self.dataset_entries = 0
        self.isData = False
        self.isMC = False
        self.isEmbed = False
        


class DataComponent( Component ):

    def __init__(self, name, files, intLumi=None, triggers=[], json=None):
        super(DataComponent, self).__init__(name, files, triggers=triggers)
        self.isData = True
        self.intLumi = intLumi
        self.json = json

    def getWeight( self, intLumi = None):
        return Weight( genNEvents = -1,
                       xSection = None,
                       genEff = -1,
                       intLumi = self.intLumi,
                       addWeight = 1. )



class MCComponent( Component ):
    def __init__(self, name, files, triggers=[], xSection=1,
                 nGenEvents=None,
                 effCorrFactor=None, **kwargs ):
        super( MCComponent, self).__init__( name = name,
                                            files = files,
                                            triggers = triggers, **kwargs )
        self.xSection = xSection
        self.nGenEvents = nGenEvents
        self.effCorrFactor = effCorrFactor
        self.isMC = True
        self.intLumi = 1.
        self.addWeight = 1.

    def getWeight( self, intLumi = None):
        # if intLumi is None:
        #    intLumi = Weight.FBINV
        #COLIN THIS WEIGHT STUFF IS REALLY BAD!!
        # use the existing Weight class or not? guess so...
        return Weight( genNEvents = self.nGenEvents,
                       xSection = self.xSection,
                       intLumi = self.intLumi,
                       genEff = 1/self.effCorrFactor,
                       addWeight = self.addWeight )

class Config( object ):
    '''Main configuration object, holds a sequence of analyzers, and
    a list of components.'''
    def __init__(self, components, sequence, services, events_class,preprocessor=None):
        self.preprocessor = preprocessor
        self.components = components
        self.sequence = sequence
        self.services = services
        self.events_class = events_class

    def __str__(self):
        comp = '\n'.join(map(str, self.components))
        sequence = str(self.sequence)
        services = '\n'.join( map(str, self.services))
        return '\n'.join([comp, sequence, services])


