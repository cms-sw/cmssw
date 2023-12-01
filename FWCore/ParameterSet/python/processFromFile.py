import sys
import types
from importlib.machinery import SourceFileLoader

def processFromFile(filename, args=None):
    old_sys_argv = None
    if args is not None:
        old_sys_argv = sys.argv[:]
        sys.argv = [filename]+args

    loader = SourceFileLoader("pycfg", filename)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    process = mod.process

    if old_sys_argv is not None:
        sys.argv = old_sys_argv

    return process
