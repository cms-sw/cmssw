import FWCore.ParameterSet.Config as cms

# replace the standard SiStripClusterizer with the switch producer
# meant primarily for testing
def customizeStripClustersFromRaw(process):
    if hasattr(process, 'striptrackerlocalrecoTask'):
        process.striptrackerlocalrecoTask.remove(process.siStripClusters)
        process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizerOnDemand_cfi")
        # CPU should emulate the full detector clusterizer
        process.siStripClusters.cpu.onDemand = cms.bool(False)
        process.striptrackerlocalrecoTask.add(process.siStripClustersTask)

    return process

def customizeHLTStripClustersFromRaw(process):
    if hasattr(process, 'hltSiStripRawToClustersFacility'):
        import RecoLocalTracker.SiStripClusterizer.SiStripClusterizerOnDemand_cfi as SiStripClusterizerOnDemand_cfi

        process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizerOnDemand_cfi")
        process.hltSiStripRawToClustersFacility = SiStripClusterizerOnDemand_cfi.siStripClusters.clone()
        process.HLTDoLocalStripSequence.replace(process.hltSiStripRawToClustersFacility,
                                   cms.Sequence(process.hltSiStripRawToClustersFacility, process.siStripClustersTaskCUDA))

    return process


def customizeHLTStripClustersFromRaw_alpaka(process: cms.Process, maxClSz:int = 16, sequences: cms.Sequence = None):
    if hasattr(process, 'hltSiStripRawToClustersFacility'):
        from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import DefaultAlgorithms
        from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import DefaultClusterizer
        # This is a super messy modifier. Someone with experience in these stuffs should review this
        obj = process.hltSiStripRawToClustersFacility.clone()
        defaultProducerPars = {
            "ProductLabel" : cms.InputTag("rawDataCollector"),
            "ConditionsLabel" : cms.string(""),
            "CablingConditionsLabel" : cms.string(""),
            #
            "onDemand" : cms.bool(False),
            "DoAPVEmulatorCheck" : cms.bool(False),
            "LegacyUnpacker": cms.bool(False),
            "HybridZeroSuppressed": cms.bool(False),
            #
            "Clusterizer": DefaultClusterizer,
            "Algorithms": DefaultAlgorithms
            }
        # Override the parameters in the originalProducerPars with those user-configured so far in the process.hltSiStripRawToClustersFacility 
        parameters = {}
        for par in defaultProducerPars:
            try:
                if (obj.hasParameter(par)):
                    defaultProducerPars[par] = obj.getParameter(par)
            except AttributeError:
                pass
            
        
        # Create the alpaka-producer and replace the one in process        
        hltSiStripRawToClustersFacilityAlpaka = cms.EDProducer("SiStripRawToCluster@alpaka")
        rmArgs = ['Algorithms', 'DoAPVEmulatorCheck', 'HybridZeroSuppressed', 'LegacyUnpacker', 'onDemand']        
        for par in defaultProducerPars:
            if par in rmArgs: continue
            setattr(hltSiStripRawToClustersFacilityAlpaka, par, defaultProducerPars[par])
        # Make sure there is maxClusterSize (if not present in the original producer)
        if not hasattr(hltSiStripRawToClustersFacilityAlpaka.Clusterizer, 'MaxClusterSize'):
            hltSiStripRawToClustersFacilityAlpaka.Clusterizer.MaxClusterSize = cms.uint32(maxClSz)
            print(f"MaxClusterSize not in process.hltSiStripRawToClustersFacility.Clusterizer PSet. Setting MaxClusterSize to {maxClSz}")
        
        # Create the legacy objects
        hltSiStripClustersToLegacy = cms.EDProducer("SiStripClustersToLegacy@alpaka",
            source = cms.InputTag("hltSiStripRawToClustersFacilityAlpaka")
        )
        
        ####### Produce ES for the alpaka clusterizer #######
        # Produce the SiStripClusterizerConditionsHost object
        hltSiStripClusterizerConditionsESProducerAlpaka = cms.ESProducer("SiStripClusterizerConditionsESProducerAlpaka@alpaka",
            QualityLabel = cms.string(""),
            Label = cms.string(""),
        )
        
        process.hltSiStripClusterizerConditionsESProducerAlpaka = hltSiStripClusterizerConditionsESProducerAlpaka
        process.hltSiStripRawToClustersFacilityAlpaka = hltSiStripRawToClustersFacilityAlpaka
        process.hltSiStripRawToClustersFacility = hltSiStripClustersToLegacy
        
        sequencesToFix = ['HLTDoLocalStripSequence']
        for seq in sequencesToFix:
            if hasattr(process, seq):
                sequence = getattr(process, seq)
                print("Sequence before:")
                print(sequence)
                sequence.replace(process.hltSiStripRawToClustersFacility, process.hltSiStripRawToClustersFacilityAlpaka + process.hltSiStripRawToClustersFacility)
                print("Sequence after:")
                print(sequence)
                
    return process