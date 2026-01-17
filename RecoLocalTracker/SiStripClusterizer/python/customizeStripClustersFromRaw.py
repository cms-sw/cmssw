import FWCore.ParameterSet.Config as cms

# replace the standard SiStripClusterizer with the switch producer
# meant primarily for testing
def customizeStripClustersFromRaw(process):
    if hasattr(process, 'striptrackerlocalrecoTask'):
        process.striptrackerlocalrecoTask.remove(process.siStripClusters)
        process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizerOnDemand_cfi")
        # CPU should emulate the full detector clusterizer
        process.siStripClusters.onDemand = cms.bool(False)
        process.striptrackerlocalrecoTask.add(process.siStripClustersTask)

    return process

def customizeHLTStripClustersFromRaw_alpaka(process: cms.Process, MaxClusterSize:int = 768, doNotReplaceInPath = []):
    if hasattr(process, 'hltSiStripRawToClustersFacility'):
        from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import DefaultAlgorithms
        from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import DefaultClusterizer
        
        # Take the parameters from the original producer
        # (parameters_() makes a copy, this could cause issues if the parameters are changed after this customizer is applied)
        initialPars = process.hltSiStripRawToClustersFacility.parameters_()
        
        # Make sure the module configuration is compatible with the heterogeneous producer
        if 'LegacyUnpacker' in initialPars and initialPars['LegacyUnpacker'] == True:
            raise RuntimeError("The LegacyUnpacker is not supported by the alpaka producer. Please set LegacyUnpacker to False.")
        if 'HybridZeroSuppressed' in initialPars and initialPars['HybridZeroSuppressed'] == True:
            # The alpaka version supports READOUT_MODE_ZERO_SUPPRESSED* modes
            raise RuntimeError("The HybridZeroSuppressed is not supported by the alpaka producer. Please set HybridZeroSuppressed to False.")
        
        # Create the alpaka-producer and set its parameters from the original one
        hltSiStripRawToClustersFacilityAlpaka = cms.EDProducer("sistrip::SiStripRawToCluster@alpaka", **initialPars)
        
        # Add the extra pars (if not present)
        if not hasattr(hltSiStripRawToClustersFacilityAlpaka, "CablingConditionsLabel"): hltSiStripRawToClustersFacilityAlpaka.CablingConditionsLabel = cms.string("")        
        ## Make sure the Clusterizer PSet has the MaxClusterSize argument
        if not hasattr(hltSiStripRawToClustersFacilityAlpaka.Clusterizer, "MaxClusterSize"):
            # print(f"[hltSiStripRawToClustersFacility] No Clusterizer.MaxClusterSize defined. Defaulting to {MaxClusterSize}")
            hltSiStripRawToClustersFacilityAlpaka.Clusterizer.MaxClusterSize = cms.uint32(MaxClusterSize)
        ## Add the MaxSeedStrips to the clusterizer PSet if not present
        if not hasattr(hltSiStripRawToClustersFacilityAlpaka.Clusterizer, "MaxSeedStrips"):
            hltSiStripRawToClustersFacilityAlpaka.Clusterizer.MaxSeedStrips = cms.uint32(200000)
                
        # Remove illegal parameters from the configuration from comparison of 
        # the legacy `edmPluginHelp -b -p SiStripClusterizerFromRaw` and
        # the heterogeneous `edmPluginHelp -b -p sistrip::SiStripRawToCluster@alpaka` parameters.
        for par in ['onDemand', 'LegacyUnpacker', 'HybridZeroSuppressed', 'Algorithms']:
            if hasattr(hltSiStripRawToClustersFacilityAlpaka, par): delattr(hltSiStripRawToClustersFacilityAlpaka, par)
        
        # Create the converter bringing the alpaka-made cluster into legacy objects
        hltSiStripClustersToLegacy = cms.EDProducer("sistrip::SiStripClustersToLegacy",
            source = cms.InputTag("hltSiStripRawToClustersFacilityAlpaka")
        )
        
        ####### Produce ES for the alpaka clusterizer #######
        # Produce the SiStripClusterizerConditionsHost object
        hltSiStripClusterizerConditionsESProducerAlpaka = cms.ESProducer("sistrip::SiStripClusterizerConditionsESProducerAlpaka@alpaka",
            QualityLabel = cms.ESInputTag(""),
            Label = cms.ESInputTag(""),
        )
        
        # Add to process, if not already present
        if not hasattr(process, "hltSiStripClusterizerConditionsESProducerAlpaka"):
            process.hltSiStripClusterizerConditionsESProducerAlpaka = hltSiStripClusterizerConditionsESProducerAlpaka
        process.hltSiStripRawToClustersFacilityAlpaka = hltSiStripRawToClustersFacilityAlpaka
        process.hltSiStripRawToClustersFacility = hltSiStripClustersToLegacy
        
        sequencesToFix = []
        for path_name, path in process.paths_().items():
          if path.contains(process.hltSiStripRawToClustersFacility):
            # If Path is in doNotReplaceInPath, then don't replace there
            if path_name in doNotReplaceInPath: continue
            
            sequencesToFix.append(path_name)
            pathToFix_ptr = getattr(process, path_name)
            pathToFix_ptr.replace(process.hltSiStripRawToClustersFacility, process.hltSiStripRawToClustersFacilityAlpaka + process.hltSiStripRawToClustersFacility)
                
    return process
