
import FWCore.ParameterSet.Config as cms


def customizePFECALClustering(process,scenario):

    if scenario ==1:
        print '-------------ECAL CLUSTERING-------------'
        print 'SCENARIO 1: Default ECAL clustering & ECAL time reconstruction'
        print 'Timing cuts are applied to the PF cluster level'
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')
        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)

    elif scenario ==2:
        print '-------------ECAL CLUSTERING-------------'
        print 'SCENARIO 2: ECAL clustering with time & default ECAL time reconstruction'
        print 'Timing cuts are applied to the PF cluster level'

        from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionECALBarrel,_timeResolutionECALEndcap

        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterECALTimeSelected_cfi')

        process.particleFlowRecHitECAL.navigator.name = "PFRecHitECALNavigatorWithTime"
        process.particleFlowRecHitECAL.navigator.barrel = cms.PSet(
             sigmaCut = cms.double(5.0),
             timeResolutionCalc = _timeResolutionECALBarrel 
        )
        process.particleFlowRecHitECAL.navigator.endcap = cms.PSet(
            sigmaCut = cms.double(5.0),
            timeResolutionCalc = _timeResolutionECALEndcap
        )
        
        for p in process.particleFlowRecHitECAL.producers:
            for t in p.qualityTests:
                if t.name == 'PFRecHitQTestECAL':
                    t.timingCleaning = cms.bool(False) 

        process.particleFlowClusterECALUncorrected.recHitsSource = cms.InputTag("particleFlowRecHitECAL")
        process.particleFlowClusterECALTimeSelected.src = cms.InputTag('particleFlowClusterECALUncorrected')
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterECALTimeSelected')

        i = process.pfClusteringECAL.index(process.particleFlowClusterECALUncorrected)
        process.pfClusteringECAL.insert(i+1,process.particleFlowClusterECALTimeSelected)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.algoName = cms.string("PFlow2DClusterizerWithTime")

        process.particleFlowClusterECALUncorrected.pfClusterBuilder.showerSigma = cms.double(1.5)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeSigmaEB = cms.double(10.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeSigmaEE = cms.double(10.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.maxNSigmaTime = cms.double(10.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.minChi2Prob = cms.double(0.)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.clusterTimeResFromSeed = cms.bool(False)
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeResolutionCalcBarrel = _timeResolutionECALBarrel
        process.particleFlowClusterECALUncorrected.pfClusterBuilder.timeResolutionCalcEndcap = _timeResolutionECALEndcap

        
        

def customizePFHCALClustering(process,scenario,data = False):

    if scenario ==0:
        print '-------------HCAL CLUSTERING-------------'
        print 'SCENARIO 0: HCAL clustering from hits+old HCAL time reconstruction'
        print 'Timing cuts are N OTapplied'
        if not data:
            print 'You are trying to run on MC, if not set data to True'
        else:   
            print 'You are trying to run on data, if not set data to False'
 
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHETimeSelected_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi')

        process.pfClusteringHBHEHF.remove(process.towerMakerPF)
        process.pfClusteringHBHEHF.remove(process.particleFlowRecHitHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFHAD)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFEM)

        process.pfClusteringHBHEHF+=process.particleFlowRecHitHBHE
        process.pfClusteringHBHEHF+=process.particleFlowRecHitHF
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHE
        process.pfClusteringHBHEHF+=process.particleFlowClusterHF


        process.particleFlowClusterHCAL = cms.EDProducer('PFMultiDepthClusterProducer',
               clustersSource = cms.InputTag("particleFlowClusterHBHE"),
               pfClusterBuilder =cms.PSet(
                      algoName = cms.string("PFMultiDepthClusterizer"),
                      nSigmaEta = cms.double(2.),
                      nSigmaPhi = cms.double(2.),
                      #pf clustering parameters
                      minFractionToKeep = cms.double(1e-7),
                      allCellsPositionCalc = cms.PSet(
                         algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                         minFractionInCalc = cms.double(1e-9),    
                         posCalcNCrystals = cms.int32(-1),
                         logWeightDenominator = cms.double(0.8),#same as gathering threshold
                         minAllowedNormalization = cms.double(1e-9)
                      )
               ),
               positionReCalc = cms.PSet(
               ),
               energyCorrector = cms.PSet()
         )

        process.pfClusteringHBHEHF+=process.particleFlowClusterHCAL

        process.particleFlowTmp.cleanedHF = cms.VInputTag(cms.InputTag("particleFlowRecHitHF","Cleaned"), cms.InputTag("particleFlowClusterHF","Cleaned"))
        
        ##customize block since we have only one HF cluster collection now
        importers = process.particleFlowBlock.elementImporters
        newImporters  = cms.VPSet()

        for pset in importers:
            if pset.importerName =="GenericClusterImporter":

                if pset.source.moduleLabel == 'particleFlowClusterHFEM':
                    pset.source = cms.InputTag('particleFlowClusterHF')
                    newImporters.append(pset)
                elif  pset.source.moduleLabel != 'particleFlowClusterHFHAD':
                    newImporters.append(pset)
            else:
                newImporters.append(pset)
        
                
        process.particleFlowBlock.elementImporters = newImporters
        

        for norm in process.particleFlowClusterHBHE.seedFinder.thresholdsByDetector:
            if norm.detector.value() == 'HCAL_BARREL1':
                norm.seedingThreshold = 0.80
            if norm.detector.value() == 'HCAL_ENDCAP':
                norm.seedingThreshold = 0.80



        # Enable OOT pileup corrections for HBHE in MC processing
        process.hbheprereco.mcOOTCorrectionName = cms.string("HBHE")

        # Uncomment next line to enable OOT pileup corrections in data processing
        if data:
            process.hbheprereco.dataOOTCorrectionName = cms.string("HBHE")

        # Configure database access for the OOT pileup corrections
        import os
        process.load("CondCore.CondDB.CondDBboost_cfi")
        process.CondDBboost.connect = "sqlite_file:%s/src/CondTools/Hcal/data/testOOTPileupCorrection.db" % os.environ["CMSSW_RELEASE_BASE"]

        process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                              process.CondDBboost,
                                              toGet = cms.VPSet(cms.PSet(
                    record = cms.string("HcalOOTPileupCorrectionRcd"),
                    tag = cms.string("test")
                    )
            )
        )



    if scenario ==1:
        print '-------------HCAL CLUSTERING-------------'
        print 'SCENARIO 1: HCAL clustering from hits'
        print 'Timing cuts are applied'
        if not data:
            print 'You are trying to run on MC, if not set data to True'
        else:   
            print 'You are trying to run on data, if not set data to False'
 
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHETimeSelected_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi')

        process.pfClusteringHBHEHF.remove(process.towerMakerPF)
        process.pfClusteringHBHEHF.remove(process.particleFlowRecHitHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFHAD)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFEM)

        process.pfClusteringHBHEHF+=process.particleFlowRecHitHBHE
        process.pfClusteringHBHEHF+=process.particleFlowRecHitHF
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHE
#        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHETimeSelected
        process.pfClusteringHBHEHF+=process.particleFlowClusterHF


        process.particleFlowClusterHCAL = cms.EDProducer('PFMultiDepthClusterProducer',
               clustersSource = cms.InputTag("particleFlowClusterHBHE"),
               pfClusterBuilder =cms.PSet(
                      algoName = cms.string("PFMultiDepthClusterizer"),
                      nSigmaEta = cms.double(2.),
                      nSigmaPhi = cms.double(2.),
                      #pf clustering parameters
                      minFractionToKeep = cms.double(1e-7),
                      allCellsPositionCalc = cms.PSet(
                         algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                         minFractionInCalc = cms.double(1e-9),    
                         posCalcNCrystals = cms.int32(-1),
                         logWeightDenominator = cms.double(0.8),#same as gathering threshold
                         minAllowedNormalization = cms.double(1e-9)
                      )
               ),
               positionReCalc = cms.PSet(
               ),
               energyCorrector = cms.PSet()
         )

        process.pfClusteringHBHEHF+=process.particleFlowClusterHCAL

        process.particleFlowTmp.cleanedHF = cms.VInputTag(cms.InputTag("particleFlowRecHitHF","Cleaned"), cms.InputTag("particleFlowClusterHF","Cleaned"))
        
        ##customize block since we have only one HF cluster collection now
        importers = process.particleFlowBlock.elementImporters
        newImporters  = cms.VPSet()

        for pset in importers:
            if pset.importerName =="GenericClusterImporter":

                if pset.source.moduleLabel == 'particleFlowClusterHFEM':
                    pset.source = cms.InputTag('particleFlowClusterHF')
                    newImporters.append(pset)
                elif  pset.source.moduleLabel != 'particleFlowClusterHFHAD':
                    newImporters.append(pset)
            else:
                newImporters.append(pset)
        
                
        process.particleFlowBlock.elementImporters = newImporters
        

        for norm in process.particleFlowClusterHBHE.seedFinder.thresholdsByDetector:
            if norm.detector.value() == 'HCAL_BARREL1':
                norm.seedingThreshold = 0.80
            if norm.detector.value() == 'HCAL_ENDCAP':
                norm.seedingThreshold = 0.80



        # Enable OOT pileup corrections for HBHE in MC processing
        process.hbheprereco.mcOOTCorrectionName = cms.string("HBHE")

        # Uncomment next line to enable OOT pileup corrections in data processing
        if data:
            process.hbheprereco.dataOOTCorrectionName = cms.string("HBHE")

        # Configure database access for the OOT pileup corrections
        import os
        process.load("CondCore.CondDB.CondDBboost_cfi")
        process.CondDBboost.connect = "sqlite_file:%s/src/CondTools/Hcal/data/testOOTPileupCorrection.db" % os.environ["CMSSW_RELEASE_BASE"]

        process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                              process.CondDBboost,
                                              toGet = cms.VPSet(cms.PSet(
                    record = cms.string("HcalOOTPileupCorrectionRcd"),
                    tag = cms.string("test")
                    )
            )
        )


    if scenario ==2:
        print '-------------HCAL CLUSTERING-------------'
        print 'SCENARIO 2: HCAL clustering from hits +3D clustering with time '
        print 'Timing cuts are applied'
        if not data:
            print 'You are trying to run on MC, if not set data to True'
        else:   
            print 'You are trying to run on data, if not set data to False'
 
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHETimeSelected_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi')

        from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCAL
        

        process.pfClusteringHBHEHF.remove(process.towerMakerPF)
        process.pfClusteringHBHEHF.remove(process.particleFlowRecHitHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFHAD)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFEM)

        process.pfClusteringHBHEHF+=process.particleFlowRecHitHBHE
        process.pfClusteringHBHEHF+=process.particleFlowRecHitHF
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHE
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHETimeSelected
        process.pfClusteringHBHEHF+=process.particleFlowClusterHF


        process.particleFlowClusterHCAL = cms.EDProducer('PFMultiDepthClusterProducer',
               clustersSource = cms.InputTag("particleFlowClusterHBHETimeSelected"),
               pfClusterBuilder =cms.PSet(
                      algoName = cms.string("PFMultiDepthClusterizer"),
                      nSigmaEta = cms.double(2.),
                      nSigmaPhi = cms.double(2.),
                      #pf clustering parameters
                      minFractionToKeep = cms.double(1e-7),
                      allCellsPositionCalc = cms.PSet(
                         algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                         minFractionInCalc = cms.double(1e-9),    
                         posCalcNCrystals = cms.int32(-1),
                         logWeightDenominator = cms.double(0.8),#same as gathering threshold
                         minAllowedNormalization = cms.double(1e-9)
                      )
               ),
               positionReCalc = cms.PSet(
               ),
               energyCorrector = cms.PSet()
         )

        process.pfClusteringHBHEHF+=process.particleFlowClusterHCAL
        
        ##customize block since we have only one HF cluster collection now
        importers = process.particleFlowBlock.elementImporters
        newImporters  = cms.VPSet()

        for pset in importers:
            if pset.importerName =="GenericClusterImporter":

                if pset.source.moduleLabel == 'particleFlowClusterHFEM':
                    pset.source = cms.InputTag('particleFlowClusterHF')
                    newImporters.append(pset)
                elif  pset.source.moduleLabel != 'particleFlowClusterHFHAD':
                    newImporters.append(pset)
            else:
                newImporters.append(pset)
        
                
        process.particleFlowBlock.elementImporters = newImporters
        



        for norm in process.particleFlowClusterHBHE.seedFinder.thresholdsByDetector:
            if norm.detector.value() == 'HCAL_BARREL1':
                norm.seedingThreshold = 0.80
            if norm.detector.value() == 'HCAL_ENDCAP':
                norm.seedingThreshold = 0.80



        # Enable OOT pileup corrections for HBHE in MC processing
        process.hbheprereco.mcOOTCorrectionName = cms.string("HBHE")

        # Uncomment next line to enable OOT pileup corrections in data processing
        if data:
            process.hbheprereco.dataOOTCorrectionName = cms.string("HBHE")

        # Configure database access for the OOT pileup corrections
        import os
        process.load("CondCore.CondDB.CondDBboost_cfi")
        process.CondDBboost.connect = "sqlite_file:%s/src/CondTools/Hcal/data/testOOTPileupCorrection.db" % os.environ["CMSSW_RELEASE_BASE"]

        process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                              process.CondDBboost,
                                              toGet = cms.VPSet(cms.PSet(
                    record = cms.string("HcalOOTPileupCorrectionRcd"),
                    tag = cms.string("test")
                    )
            )
        )
    
        process.particleFlowRecHitHBHE.navigator = cms.PSet(
            name = cms.string("PFRecHitHCALNavigatorWithTime"),
            sigmaCut = cms.double(5.0),
            timeResolutionCalc = _timeResolutionHCAL
            )
        process.particleFlowClusterHBHE.pfClusterBuilder.timeSigmaEB = cms.double(10.)
        process.particleFlowClusterHBHE.pfClusterBuilder.timeSigmaEE = cms.double(10.)
        process.particleFlowClusterHBHE.pfClusterBuilder.maxNSigmaTime = cms.double(10.)
        process.particleFlowClusterHBHE.pfClusterBuilder.minChi2Prob = cms.double(0.)
        process.particleFlowClusterHBHE.pfClusterBuilder.clusterTimeResFromSeed = cms.bool(False)
        process.particleFlowClusterHBHE.pfClusterBuilder.timeResolutionCalcBarrel = _timeResolutionHCAL
        process.particleFlowClusterHBHE.pfClusterBuilder.timeResolutionCalcEndcap = _timeResolutionHCAL
        process.particleFlowClusterHBHE.pfClusterBuilder.algoName = cms.string("PFlow2DClusterizerWithTime")
        process.particleFlowTmp.cleanedHF = cms.VInputTag(cms.InputTag("particleFlowRecHitHF","Cleaned"), cms.InputTag("particleFlowClusterHF","Cleaned"))



    if scenario ==3:
        print '-------------HCAL CLUSTERING-------------'
        print 'SCENARIO 3: Max time sample + HCAL clustering from hits +3D clustering with time '
        print 'Timing cuts are applied'
        if not data:
            print 'You are trying to run on MC, if not set data to True'
        else:   
            print 'You are trying to run on data, if not set data to False'
 
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHEMaxSampleTimeSelected_cfi')
        process.load('RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi')

        from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample
        

        process.pfClusteringHBHEHF.remove(process.towerMakerPF)
        process.pfClusteringHBHEHF.remove(process.particleFlowRecHitHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHCAL)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFHAD)
        process.pfClusteringHBHEHF.remove(process.particleFlowClusterHFEM)

        process.pfClusteringHBHEHF+=process.particleFlowRecHitHBHE
        process.pfClusteringHBHEHF+=process.particleFlowRecHitHF
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHE
        process.pfClusteringHBHEHF+=process.particleFlowClusterHBHETimeSelected
        process.pfClusteringHBHEHF+=process.particleFlowClusterHF



        for p in process.particleFlowRecHitHBHE.producers:
            p.name = cms.string(p.name.value()+'MaxSample')



        process.particleFlowClusterHCAL = cms.EDProducer('PFMultiDepthClusterProducer',
#               clustersSource = cms.InputTag("particleFlowClusterHBHE"),

               clustersSource = cms.InputTag("particleFlowClusterHBHETimeSelected"),
               pfClusterBuilder =cms.PSet(
                      algoName = cms.string("PFMultiDepthClusterizer"),
                      nSigmaEta = cms.double(2.),
                      nSigmaPhi = cms.double(2.),
                      #pf clustering parameters
                      minFractionToKeep = cms.double(1e-7),
                      allCellsPositionCalc = cms.PSet(
                         algoName = cms.string("Basic2DGenericPFlowPositionCalc"),
                         minFractionInCalc = cms.double(1e-9),    
                         posCalcNCrystals = cms.int32(-1),
                         logWeightDenominator = cms.double(0.8),#same as gathering threshold
                         minAllowedNormalization = cms.double(1e-9)
                      )
               ),
               positionReCalc = cms.PSet(
               ),
               energyCorrector = cms.PSet()
         )

        process.pfClusteringHBHEHF+=process.particleFlowClusterHCAL
        
        ##customize block since we have only one HF cluster collection now
        importers = process.particleFlowBlock.elementImporters
        newImporters  = cms.VPSet()

        for pset in importers:
            if pset.importerName =="GenericClusterImporter":

                if pset.source.moduleLabel == 'particleFlowClusterHFEM':
                    pset.source = cms.InputTag('particleFlowClusterHF')
                    newImporters.append(pset)
                elif  pset.source.moduleLabel != 'particleFlowClusterHFHAD':
                    newImporters.append(pset)
            else:
                newImporters.append(pset)
        
                
        process.particleFlowBlock.elementImporters = newImporters
        


        process.particleFlowRecHitHBHE.navigator = cms.PSet(
            name = cms.string("PFRecHitHCALNavigatorWithTime"),
            sigmaCut = cms.double(5.0),
            timeResolutionCalc = _timeResolutionHCALMaxSample
        )


        


        process.particleFlowClusterHBHE.pfClusterBuilder.timeSigmaEB = cms.double(10.)
        process.particleFlowClusterHBHE.pfClusterBuilder.timeSigmaEE = cms.double(10.)
        process.particleFlowClusterHBHE.pfClusterBuilder.maxNSigmaTime = cms.double(10.)
        process.particleFlowClusterHBHE.pfClusterBuilder.minChi2Prob = cms.double(0.)
        process.particleFlowClusterHBHE.pfClusterBuilder.clusterTimeResFromSeed = cms.bool(False)
        process.particleFlowClusterHBHE.pfClusterBuilder.timeResolutionCalcBarrel = _timeResolutionHCALMaxSample
        process.particleFlowClusterHBHE.pfClusterBuilder.timeResolutionCalcEndcap = _timeResolutionHCALMaxSample
        process.particleFlowClusterHBHE.pfClusterBuilder.algoName = cms.string("PFlow2DClusterizerWithTime")



        for norm in process.particleFlowClusterHBHE.seedFinder.thresholdsByDetector:
            if norm.detector.value() == 'HCAL_BARREL1':
                norm.seedingThreshold = 0.80
            if norm.detector.value() == 'HCAL_ENDCAP':
                norm.seedingThreshold = 0.80
                
        
        
        process.particleFlowTmp.cleanedHF = cms.VInputTag(cms.InputTag("particleFlowRecHitHF","Cleaned"), cms.InputTag("particleFlowClusterHF","Cleaned"))
