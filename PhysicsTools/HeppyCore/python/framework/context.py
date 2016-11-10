import os 

def get_name(env=None):
    '''Returns the name of the context is which heppy is used.
    
    If several contexts are defined, throws a ValueError.

    @return: 'cms', 'fcc' or None if no context is defined.
    '''
    if env is None: 
        env = os.environ
    contexts = dict( (key, False) for key in ['cms','fcc'])
    reldir = env.get('CMSSW_BASE', None)
    if reldir and os.path.isdir(reldir):
        contexts['cms'] = True
    fcc_envs = set(['PODIO', 'FCCPHYSICS', 'FCCEDM', 'HEPPY'])
    if fcc_envs.issubset( env ): 
        contexts['fcc'] = True
    defined = [key for key,defined in contexts.iteritems() 
               if defined is True]
    if len(defined)>1: 
        raise ValueError('several contexts defined: ' + str(defined) )
    elif len(defined)==0:
        return None
    else: 
        return defined.pop()

def heppy_path(): 
    context = get_name()
    if context == 'cms':
        return '/'.join([os.environ['CMSSW_BASE'], 
                         'src/PhysicsTools/HeppyCore/python']) 
    elif context == 'fcc': 
        return os.environ['HEPPY']
    


name = get_name()

heppy_path = heppy_path()
