import FWCore.ParameterSet.Config as cms

ecalpedestalPCL =cms.EDAnalyzer('ECALpedestalPCLworker',
                                BarrelDigis=cms.InputTag('ecalDigis','ebDigis'),
                                EndcapDigis=cms.InputTag('ecalDigis','eeDigis'))
