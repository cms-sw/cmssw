import os
from os.path import sep, join
import importlib

class _ModuleProxy (object):
    def __init__(self, package, name):
        self._package = package
        self._name = name
        self._caller = None
    def __call__(self,*arg, **kwargs):
        if not self._caller:
            self._caller = getattr(importlib.import_module(self._package+'.'+self._name),self._name)
        return self._caller(*arg, **kwargs)


def _setupProxies(fullName:str):
    _cmssw_package_name='.'.join(fullName.split(sep)[-3:-1])
    basename = fullName.split(sep)[-1]
    pathname = fullName[:-1*len(basename)]
    proxies = dict()
    for filename in ( x for x in os.listdir(pathname) if (len(x) > 3 and x[-3:] == '.py' and x != basename and ((len(x) < 6) or (x[-6:] != 'cfi.py')))):
        name = filename[:-3]
        proxies[name] = _ModuleProxy(_cmssw_package_name, name)
    return proxies
