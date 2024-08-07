import sys
from FWCore.ParameterSet.VarParsing import VarParsing

def parse(argv):
  sys.argv = ['test.py','maxEvents=100']+argv # emulate user arguments
  opts = VarParsing('standard')
  def add(n, d, m, t):
    opts.register(n, d, m, t)
  add('myInts0',   '',      VarParsing.multiplicity.list, VarParsing.varType.int)
  add('myInts1',   0,       VarParsing.multiplicity.list, VarParsing.varType.int)
  add('myInts2',   [0],     VarParsing.multiplicity.list, VarParsing.varType.int)
  add('myFloats0', '',      VarParsing.multiplicity.list, VarParsing.varType.float)
  add('myFloats1', 0,       VarParsing.multiplicity.list, VarParsing.varType.float)
  add('myFloats2', [0],     VarParsing.multiplicity.list, VarParsing.varType.float)
  add('myBools0',  '',      VarParsing.multiplicity.list, VarParsing.varType.bool)
  add('myBools1',  True,    VarParsing.multiplicity.list, VarParsing.varType.bool)
  add('myBools2',  [True],  VarParsing.multiplicity.list, VarParsing.varType.bool)
  add('myStrs0',   '',      VarParsing.multiplicity.list, VarParsing.varType.string)
  add('myStrs1',   'foo',   VarParsing.multiplicity.list, VarParsing.varType.string)
  add('myStrs2',   ['foo'], VarParsing.multiplicity.list, VarParsing.varType.string)

# parse without user arguments
parse([ ])

# parse with user arguments
parse(['myInts1=0,1','myBools1=True,False','myStrs1=foo,bar','myStrs2=foo,bar'])