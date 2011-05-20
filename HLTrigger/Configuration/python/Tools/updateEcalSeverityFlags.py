"""
Customize the HLT menu dump to run with CMSSW_4_2_0_pre5 plus
    V01-04-05       Calibration/IsolatedParticles
    V00-18-32       DQM/EcalBarrelMonitorTasks
    V00-18-32       DQM/EcalEndcapMonitorTasks
    V02-02-06       DataFormats/EcalRecHit
    V01-23-11       FastSimulation/Configuration
    V04-09-04       FastSimulation/HighLevelTrigger
    V00-09-09       RecoEcal/EgammaClusterProducers
    V05-07-02       RecoEcal/EgammaCoreTools
    V01-25-01       RecoEgamma/EgammaElectronProducers
    V00-05-07       RecoEgamma/EgammaHLTProducers
    V00-04-02       RecoEgamma/EgammaIsolationAlgos
    V01-01-03       RecoEgamma/PhotonIdentification
    V00-14-04       RecoLocalCalo/CaloTowersCreator
    V00-13-11       RecoLocalCalo/EcalRecAlgos
    V01-00-06       RecoLocalCalo/HcalRecAlgos
"""


def update(process):

    import FWCore.ParameterSet.Config as cms
    translate = {
        0: 'kGood',
        1: 'kProblematic',
        2: 'kRecovered',
        3: 'kTime',
        4: 'kWeird',
        5: 'kBad'
    }
    
    def fixEcalSeverityFlagsForEcalClusters(module, parameter):
        # "translate" the severity flags
        oldValue = getattr(module, parameter)
        newValue = cms.vstring( translate[flag] for flag in oldValue )
        setattr(module, parameter, newValue)


    def fixEcalSeverityFlagsForCaloTowers(module):
        # list all severity flags >= the old vlue
        oldValue = module.EcalAcceptSeverityLevel.value()
        module.EcalRecHitSeveritiesToBeExcluded = cms.vstring( translate[flag] for flag in range(oldValue, len(translate)) )
        # remove the old parameter
        delattr(module, 'EcalAcceptSeverityLevel')
        # add a new (empty) parameter
        module.EcalSeveritiesToBeUsedInBadTowers = cms.vstring()



    # run over all EDProducers
    import helper
    for name, module in helper.findEDProducers(process).iteritems():
        if module.type_() in ('HybridClusterProducer', 'EgammaHLTHybridClusterProducer', ):
            print 'fixing', name
            fixEcalSeverityFlagsForEcalClusters( module, 'RecHitFlagToBeExcluded' )
            fixEcalSeverityFlagsForEcalClusters( module, 'RecHitSeverityToBeExcluded' )

        if module.type_() in ('Multi5x5ClusterProducer', 'EgammaHLTMulti5x5ClusterProducer', ):
            print 'fixing', name
            fixEcalSeverityFlagsForEcalClusters( module, 'RecHitFlagToBeExcluded' )

        if module.type_() in ('CaloTowersCreator', ):
            print 'fixing', name
            fixEcalSeverityFlagsForCaloTowers( module )
