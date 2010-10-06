import FWCore.ParameterSet.Config as cms

## Electron ID Based on Simple Cuts: Spring10 MC tuned selections
#
#  Instructions on how to use this file
#  ====================================
#
#  The selections that are implemented in this python cfg are
#  explained in this twiki page:
#  https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
#  In summary, they come in 6 different tightness levels. For
#  each tightness, the user can select whether they want
#  combined isolation or relative isolations.
#
#  In order to use this cfg file you have to include it from the
#  python directory that you have placed it, clone some selection
#  of your preference and run it in your sequence
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  from ElectroWeakAnalysis.WENu.simpleCutBasedElectronID_cfi import *  
#
#  simpleEleId_95relIso = simpleCutBasedElectronID.clone()
#  simpleEleId_95relIso.electronQuality = '_95relIso_'
#  mySequence = cms.Sequence(...+...+..+simpleEleId95relIso+...)
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Warning: make sure that you use the correct tags for the
#  RecoEgamma/ElectronIdentification package
#  consult this twiki to obtain the latest information:
#
#  https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
#
#  this version of the file needs 
#  V00-03-07-03   RecoEgamma/ElectronIdentification

simpleCutBasedElectronID = cms.EDProducer("EleIdCutBasedExtProducer",

#   import here your collections
    src = cms.InputTag("gsfElectrons"),
    #reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    #reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    # Spring10 uses these names:
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    # if you want the vertices or the offline beam spot
    verticesCollection = cms.InputTag("offlineBeamSpot"),
    dataMagneticFieldSetUp = cms.bool(False),
    dcsTag = cms.InputTag("scalersRawToDigi"),                                          
    algorithm = cms.string('eIDCB'),

    #electronIDType: robust  for the simple Cut-Based
    #electronQuality: see later
    #electronVersion: use V03 with the offline beam spot
    electronIDType  = cms.string('robust'),
    electronQuality = cms.string('test'),
    electronVersion = cms.string('V04'),

####
#### Selections with Relative Isolation                                          
    robust95relIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(1.5e-01, 1.0e-02, 8.0e-01, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 1.5e-01, 
                                 2.0e+00, 1.2e-01, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.0, 0.0, ),
#           endcap =  cms.vdouble(7.0e-02, 3.0e-02, 7.0e-01, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 8.0e-02, 
#                                 6.0e-02, 5.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.0, 0.0, ),
           endcap =  cms.vdouble(7.0e-02, 3.0e-02, 7.0e-01, 1.0e-02, -1, -1, 9999., 9999., 9999., 9999., 9999., 8.0e-02, 
                                 6.0e-02, 5.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.0, 0.0, ),
    ),
    robust90relIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(1.2e-01, 1.0e-02, 8.0e-01, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 1.2e-01, 
                                 9.0e-02, 1.0e-01, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(5.0e-02, 3.0e-02, 7.0e-01, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 5.0e-02, 
#                                 6.0e-02, 3.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),      
           endcap =  cms.vdouble(5.0e-02, 3.0e-02, 7.0e-01, 9.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 5.0e-02, 
                                 6.0e-02, 3.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
    ),
    robust85relIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(4.0e-02, 1.0e-02, 6.0e-02, 6.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9.0e-02, 
                                 8.0e-02, 1.0e-01, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 4.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 5.0e-02, 
#                                 5.0e-02, 2.5e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 4.0e-02, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 5.0e-02, 
                                 5.0e-02, 2.5e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
    ),
    robust80relIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(4.0e-02, 1.0e-02, 6.0e-02, 4.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9.0e-02, 
                                 7.0e-02, 1.0e-01, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 3.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 4.0e-02, 
#                                 5.0e-02, 2.5e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),      
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 3.0e-02, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 4.0e-02, 
                                 5.0e-02, 2.5e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
    ),
    # 70% point modified with restricting cuts to physical values                                                                   
    robust70relIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(2.5e-02, 1.0e-02, 3.0e-02, 4.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 5.0e-02, 
                                 6.0e-02, 3.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 2.5e-02, 
#                                 2.5e-02, 2.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 5.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 2.5e-02, 
                                 2.5e-02, 2.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
    ),
    # 60% point modified with restricting cuts to physical values                      
    robust60relIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(2.5e-02, 1.0e-02, 2.5e-02, 4.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 4.0e-02, 
                                 4.0e-02, 3.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 2.5e-02, 
#                                 2.0e-02, 2.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),      
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 5.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 2.5e-02, 
                                 2.0e-02, 2.0e-02, 9999., 9999., 9999., 9999., 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
    ),

####
#### Selections with Combined Isolation

    robust95cIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(1.5e-01, 1.0e-02, 8.0e-01, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 1.5e-01, 0.0, -9999., 9999., 9999., 1, -1, 0.0, 0.0, ),
#           endcap =  cms.vdouble(7.0e-02, 3.0e-02, 7.0e-01, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
#                                 9999., 9999., 9999., 9999., 9999., 1.0e-01, 0.0, -9999., 9999., 9999., 1, -1, 0.0, 0.0, ),           
           endcap =  cms.vdouble(7.0e-02, 3.0e-02, 7.0e-01, 1.0e-02, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 1.0e-01, 0.0, -9999., 9999., 9999., 1, -1, 0.0, 0.0, ),
    ),
    robust90cIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(1.2e-01, 1.0e-02, 8.0e-01, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 1.0e-01, 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(5.0e-02, 3.0e-02, 7.0e-01, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
#                                 9999., 9999., 9999., 9999., 9999., 7.0e-02, 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),        
           endcap =  cms.vdouble(5.0e-02, 3.0e-02, 7.0e-01, 9.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 7.0e-02, 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
    ),
    robust85cIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(4.0e-02, 1.0e-02, 6.0e-02, 6.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 9.0e-02, 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 4.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
#                                 9999., 9999., 9999., 9999., 9999., 6.0e-02, 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 4.0e-02, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 6.0e-02, 0.0, -9999., 9999., 9999., 1, -1, 0.02, 0.02, ),
    ),
    robust80cIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(4.0e-02, 1.0e-02, 6.0e-02, 4.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 7.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 3.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
#                                 9999., 9999., 9999., 9999., 9999., 6.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),        
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 3.0e-02, 7.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 6.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
    ),
    # 70% point modified with restricting cuts to physical values                                          
    robust70cIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(2.5e-02, 1.0e-02, 3.0e-02, 4.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 4.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
#                                 9999., 9999., 9999., 9999., 9999., 3.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 5.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 3.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
    ),
    # 60% point modified with restricting cuts to physical values
    robust60cIsoEleIDCutsV04 = cms.PSet(
           barrel =  cms.vdouble(2.5e-02, 1.0e-02, 2.5e-02, 4.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 3.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
#           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 9999., -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
#                                 9999., 9999., 9999., 9999., 9999., 2.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
           endcap =  cms.vdouble(2.5e-02, 3.0e-02, 2.0e-02, 5.0e-03, -1, -1, 9999., 9999., 9999., 9999., 9999., 9999., 
                                 9999., 9999., 9999., 9999., 9999., 2.0e-02, 0.0, -9999., 9999., 9999., 0, -1, 0.02, 0.02, ),
    ),

)

