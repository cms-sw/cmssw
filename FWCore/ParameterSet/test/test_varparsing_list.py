import sys
from FWCore.ParameterSet.VarParsing import VarParsing

def parse(argv):
  sys.argv = ['test_varparsing_list.py','maxEvents=100']+argv # emulate user arguments
  opts = VarParsing('standard')
  opts.register('myInts0',   '',      VarParsing.multiplicity.list, VarParsing.varType.int)
  opts.register('myInts1',   [],      VarParsing.multiplicity.list, VarParsing.varType.int)
  opts.register('myInts2',   0,       VarParsing.multiplicity.list, VarParsing.varType.int)
  opts.register('myInts3',   [0],     VarParsing.multiplicity.list, VarParsing.varType.int)
  opts.register('myFloats0', '',      VarParsing.multiplicity.list, VarParsing.varType.float)
  opts.register('myFloats1', [],      VarParsing.multiplicity.list, VarParsing.varType.float)
  opts.register('myFloats2', 0,       VarParsing.multiplicity.list, VarParsing.varType.float)
  opts.register('myFloats3', [0],     VarParsing.multiplicity.list, VarParsing.varType.float)
  opts.register('myBools0',  '',      VarParsing.multiplicity.list, VarParsing.varType.bool)
  opts.register('myBools1',  [],      VarParsing.multiplicity.list, VarParsing.varType.bool)
  opts.register('myBools2',  True,    VarParsing.multiplicity.list, VarParsing.varType.bool)
  opts.register('myBools3',  [True],  VarParsing.multiplicity.list, VarParsing.varType.bool)
  opts.register('myStrs0',   '',      VarParsing.multiplicity.list, VarParsing.varType.string)
  opts.register('myStrs1',   [],      VarParsing.multiplicity.list, VarParsing.varType.string)
  opts.register('myStrs2',   'foo',   VarParsing.multiplicity.list, VarParsing.varType.string)
  opts.register('myStrs3',   ['foo'], VarParsing.multiplicity.list, VarParsing.varType.string)
  opts.parseArguments()
  #print(f">>> Parsed: {sys.argv} -> lists={opts._lists}")

# parse without user arguments
parse([ ])

# parse with user arguments
parse(['myInts1=0,1,-1','myFloats1=3.14,0,0.0,-1.0','myBools1=True,False','myStrs1=foo,bar'])