import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

pfClient = DQMEDHarvester("PFClient",
                          FolderNames = cms.vstring("PFJet/CompWithGenJet","PFJet/CompWithCaloJet"),
                          HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_"),
                          CreateEfficiencyPlots = cms.bool(False),
                          HistogramNamesForEfficiencyPlots = cms.vstring( " " ), 
                          HistogramNamesForProjectionPlots = cms.vstring( " " ),
                          CreateProfilePlots = cms.bool(False),
                          HistogramNamesForProfilePlots = cms.vstring( " " ),
                          )

# need a different Client to store the slices  
pfClientJetRes = DQMEDHarvester("PFClient_JetRes",
                                  FolderNames = cms.vstring("PFJet/CompWithGenJet","PFJet/CompWithCaloJet"),
                                  HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_"),
                                  CreateEfficiencyPlots = cms.bool(False),
                                  HistogramNamesForEfficiencyPlots = cms.vstring( " " ),
                                  CreateProfilePlots = cms.bool(False),
                                  HistogramNamesForProfilePlots = cms.vstring( " " ),
                                  #VariablePtBins  = cms.vint32(0,1,2,5,10,20,50,100,200,400,1000)
                                  VariablePtBins  = cms.vint32(20,40,60,80,100,150,200,250,300,400,500,750)
                                 )
