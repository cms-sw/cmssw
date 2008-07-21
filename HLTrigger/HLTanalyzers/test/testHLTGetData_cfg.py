Traceback (most recent call last):
  File "cfg2py.py", line 8, in ?
    print cmsParse.dumpCfg(fileInPath)
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parseConfig.py", line 1614, in dumpCfg
    return cfgDumper.parseFile(_fileFactory(fileName))[0]
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 990, in parseFile
    return self.parseString(file_contents)
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 770, in parseString
    loc, tokens = self._parse( instring.expandtabs(), 0 )
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 663, in _parseNoCache
    loc,tokens = self.parseImpl( instring, preloc, doActions )
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 1810, in parseImpl
    loc, resultlist = self.exprs[0]._parse( instring, loc, doActions )
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 689, in _parseNoCache
    tokens = fn( instring, tokensStart, retTokens )
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parseConfig.py", line 1388, in _dumpCfg
    values = _getCompressedNodes(s, loc, list(iter(toks[0][1])) )
  File "/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/python/FWCore/ParameterSet/parseConfig.py", line 1380, in _getCompressedNodes
    raise pp.ParseFatalException(s,loc,"the process contains the error \n"+str(e))
FWCore.ParameterSet.parsecf.pyparsing.ParseFatalException: the process contains the error 
Unable to find file 'FWCore/MessageLogger/data/MessageLogger.cfi' using the search path ${'CMSSW_SEARCH_PATH'} 
/afs/cern.ch/user/j/jjhollar/scratch0/Testrels/CMSSW_2_1_0_pre9/src:/afs/cern.ch/user/j/jjhollar/scratch0/Testrels/CMSSW_2_1_0_pre9/share:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/src:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_2_1_0_pre9/share:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-CondCore-SQLiteData/24:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-FastSimulation-MaterialEffects/20:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-FastSimulation-PileUpProducer/22:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Geometry-CaloTopology/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-MagneticField-Interpolation/22:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoMuon-MuonIdentification/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoParticleFlow-PFBlockProducer/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoParticleFlow-PFTracking/22-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoTracker-RingESSource/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoTracker-RoadMapESSource/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-SimG4CMS-Calo/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-EcalDigis/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-EcalHits/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-EcalRecHits/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-Geometry/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-HcalHits/19-cms (at char 0), (line:1, col:1)
