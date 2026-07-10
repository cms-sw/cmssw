import os
import sys
import types
from importlib.machinery import SourceFileLoader

def processFromFile(filename:str, args=None):
    # make processFromFile behave like cmsRun
    sys.path.insert(0, os.path.dirname(os.path.abspath(filename)))

    # pass any optional arguments to the file being loaded
    old_sys_argv = None
    if args is not None:
        old_sys_argv = sys.argv[:]
        sys.argv = [filename]+args

    loader = SourceFileLoader("pycfg", filename)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    process = mod.process

    # restore the original arguments
    if old_sys_argv is not None:
        sys.argv = old_sys_argv

    # restore the original environment
    sys.path.pop(0)

    return process
