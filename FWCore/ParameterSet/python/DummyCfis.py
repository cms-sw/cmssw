import importlib
from FWCore.ParameterSet.ModulesProxy import _ModuleProxy
from FWCore.ParameterSet.Types import _ProxyParameter, _RequiredParameter, _OptionalParameter
import FWCore.ParameterSet.Config as cms

#setup defaults for each type
cms.int32.dummyDefault = 999
cms.uint32.dummyDefault = 999
cms.int64.dummyDefault = 999
cms.uint64.dummyDefault = 999
cms.string.dummyDefault="__default__"
cms.double.dummyDefault = 9999
cms.vdouble.dummyDefault = []
cms.float.dummyDefault = 9999
cms.vfloat.dummyDefault = []
cms.vint32.dummyDefault = []
cms.vuint32.dummyDefault = []
cms.vint64.dummyDefault = []
cms.vuint64.dummyDefault=[]
cms.vstring.dummyDefault=[]
cms.bool.dummyDefault = False
cms.PSet.dummyDefault = cms.PSet()
cms.VPSet.dummyDefault = cms.VPSet()
cms.InputTag.dummyDefault = "__dummy__"
cms.VInputTag.dummyDefault = []
cms.ESInputTag.dummyDefault=":__dummy__"
cms.VESInputTag.dummyValue = []
cms.EventID.dummyDefault="0:0:0"
cms.VEventID.dummyDefault =[]
cms.LuminosityBlockID.dummyDefault = "0:0"
cms.VLuminosityBlockID.dummyDefault=[]
cms.EventRange.dummyDefault="0:0"
cms.VEventRange.dummyDefault=[]
cms.LuminosityBlockRange.dummyDefault="0:0"
cms.VLuminosityBlockID.dummyDefault=[]
cms.FileInPath.dummyDefault="__dummy__"



def create_cfis(modName: str, writeRequired, writeOptional):
    modules = importlib.import_module(modName+".modules")
    for (n,m) in (x for x in modules.__dict__.items() if isinstance(x[1], _ModuleProxy)):
        print(modName +'.'+n)
        write_cfi(modName+'.'+n, writeRequired, writeOptional)

def write_cfi(pythonModuleName, writeRequired, writeOptional):
    parts = pythonModuleName.split('.')
    filename = parts[-1][0].lower()+parts[-1][1:]
    f = open(filename+"_cfi.py",'x')
    f.writelines(["import FWCore.ParameterSet.DummyCfis as dc\n",
                  "import sys\n",
                  "dc.create_module('{}', sys.modules[__name__], {}, {})\n".format(pythonModuleName, writeRequired, writeOptional)])
    f.close()
    
def setDefaultInPSet(pset: cms.PSet, writeRequired, writeOptional):
    for n in pset.parameterNames_():
        setADefault(pset, n, writeRequired, writeOptional)

def setADefault(obj, paramName, writeRequired, writeOptional):
    p = getattr(obj, paramName)
    #print(p)
    if (isinstance(p, _RequiredParameter) and writeRequired) or (isinstance(p, _OptionalParameter) and writeOptional):
        p.setValue(p._ProxyParameter__type.dummyDefault)
    if isinstance(p, cms.PSet):
        setDefaultInPSet(p, writeRequired, writeOptional)
    if isinstance(p, cms.VPSet):
        for pset in p:
            setDefaultInPSet(pset, writeRequired, writeOptional)

def setDefaultsInModule(mod, writeRequired, writeOptional):
    for n in mod.parameterNames_():
        setADefault(mod, n, writeRequired, writeOptional)
    return mod
    
def create_module(pythonModuleName: str, localPythonModule, writeRequired, writeOptional ):
    parts = pythonModuleName.split('.')
    pmod = importlib.import_module(pythonModuleName)
    setattr(localPythonModule, parts[-1][0].lower()+parts[-1][1:], setDefaultsInModule(getattr(pmod, parts[-1])(), writeRequired, writeOptional ) )
    
    
#create_cfis("FWCore.Integration")

if __name__ == '__main__':
    import FWCore.ParameterSet.Config as cms
    test = cms.EDAnalyzer("Foo",
                          a = cms.optional.int32,
                          b = cms.optional.string,
                          c = cms.optional.PSet,
                          d = cms.untracked.PSet(a=cms.int32(1), b= cms.optional.untracked.PSet),
                          e = cms.required.EventID,
                          f = cms.optional.LuminosityBlockID,
                          g = cms.optional.EventRange,
                          h = cms.optional.LuminosityBlockRange,
                          j = cms.optional.InputTag,
                          k = cms.optional.ESInputTag,
                          l = cms.optional.FileInPath
    )
    print(test.dumpPython())
    setDefaultsInModule(test, True, False)
    print(test.dumpPython())
    
