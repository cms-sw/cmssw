import inspect


class ConfigError(Exception):
    """the most basic Error for CMS config"""
    pass


class ModuleCloneError(ConfigError):
    pass


def format_outerframe(number):
    """formats the outer frame 'number' to output like:
       In file foo.py, line 8:
          process.aPath = cms.Path(module1*module2)

       'number' is the number of frames to go back relative to caller.  
    """
    frame = inspect.stack()[number+1] #+1 because this routine adds another call
    return "In file %s, line %s:\n    %s" %(frame[1], frame[2], frame[4][0])

    
def format_typename(object):
    """format the typename and return only the last part""" 
    return str(type(object)).split("'")[1].split(".")[-1]
