# The following comments couldn't be translated into the new config version:

#replace FEVTEventContent.outputCommands += TrackingToolsFEVT.outputCommands

#replace FEVTEventContent.outputCommands += RecoBTauFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoBTagFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoTauTagFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoVertexFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoEgammaFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoPixelVertexingFEVT.outputCommands

#replace RECOEventContent.outputCommands += TrackingToolsRECO.outputCommands

#replace RECOEventContent.outputCommands += RecoBTauRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoBTagRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoTauTagRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoVertexRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoEgammaRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoPixelVertexingRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoParticleFlowRECO.outputCommands

#replace AODEventContent.outputCommands += TrackingToolsAOD.outputCommands

#replace AODEventContent.outputCommands += RecoBTauAOD.outputCommands
#replace AODEventContent.outputCommands += RecoBTagAOD.outputCommands
#replace AODEventContent.outputCommands += RecoTauTagAOD.outputCommands
#replace AODEventContent.outputCommands += RecoVertexAOD.outputCommands
#replace AODEventContent.outputCommands += RecoEgammaAOD.outputCommands
#replace AODEventContent.outputCommands += RecoParticleFlowAOD.outputCommands

Traceback (most recent call last):
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/wed/2.1-wed-02/CMSSW_2_1_X_2008-06-11-0200/src/FWCore/ParameterSet/python/cfg2py.py", line 10, in ?
    print cmsParse.dumpCff(fileInPath)
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/wed/2.1-wed-02/CMSSW_2_1_X_2008-06-11-0200/src/FWCore/ParameterSet/python/parseConfig.py", line 1621, in dumpCff
    compressedValues = _getCompressedNodes(fileName, 0, values)
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/wed/2.1-wed-02/CMSSW_2_1_X_2008-06-11-0200/src/FWCore/ParameterSet/python/parseConfig.py", line 1380, in _getCompressedNodes
    raise pp.ParseFatalException(s,loc,"the process contains the error \n"+str(e))
FWCore.ParameterSet.parsecf.pyparsing.ParseFatalException: the process contains the error 
Unable to find file 'RecoLocalCalo/Configuration/data/RecoLocalCalo_EventContentCosmics.cff' using the search path ${'CMSSW_SEARCH_PATH'} 
/build/filippo/CMSSW_2_1_X_2008-06-11-0200/src:/build/filippo/CMSSW_2_1_X_2008-06-11-0200/share:/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/wed/2.1-wed-02/CMSSW_2_1_X_2008-06-11-0200/src:/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/wed/2.1-wed-02/CMSSW_2_1_X_2008-06-11-0200/share:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-CondCore-SQLiteData/24:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-FastSimulation-MaterialEffects/20:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-FastSimulation-PileUpProducer/21:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Geometry-CaloTopology/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-MagneticField-Interpolation/22:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoMuon-MuonIdentification/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoParticleFlow-PFBlockProducer/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoParticleFlow-PFTracking/22:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoTracker-RingESSource/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-RecoTracker-RoadMapESSource/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-SimG4CMS-Calo/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-EcalDigis/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-EcalHits/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-EcalRecHits/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-Geometry/19-cms:/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/data-Validation-HcalHits/19-cms (at char 0), (line:1, col:1)
