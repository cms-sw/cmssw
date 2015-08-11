import FWCore.ParameterSet.Config as cms

reducedEgamma = cms.EDProducer("ReducedEGProducer",
  keepPhotons = cms.string("pt > 14 && hadTowOverEm()<0.15"), #keep in output
  slimRelinkPhotons = cms.string("pt > 14 && hadTowOverEm()<0.15"), #keep only slimmed SuperCluster plus seed cluster
  relinkPhotons = cms.string("(r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"), #keep all associated clusters/rechits/conversions
  keepGsfElectrons = cms.string(""), #keep in output
  slimRelinkGsfElectrons = cms.string(""), #keep only slimmed SuperCluster plus seed cluster
  relinkGsfElectrons = cms.string("pt>5"), #keep all associated clusters/rechits/conversions
  photons = cms.InputTag("gedPhotons"),
  gsfElectrons = cms.InputTag("gedGsfElectrons"),
  conversions = cms.InputTag("allConversions"),
  singleConversions = cms.InputTag("particleFlowEGamma"),
  barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB"),
  endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE"),
  preshowerEcalHits = cms.InputTag("reducedEcalRecHitsES"),
  photonsPFValMap = cms.InputTag("particleBasedIsolation","gedPhotons"),
  gsfElectronsPFValMap = cms.InputTag("particleBasedIsolation","gedGsfElectrons"),
  photonIDSources = cms.VInputTag(
    cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDLoose"),
    cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDLooseEM"),    
    cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDTight")
  ),
  photonIDOutput = cms.vstring(
    "PhotonCutBasedIDLoose",
    "PhotonCutBasedIDLooseEM",
    "PhotonCutBasedIDTight",
  ),
  gsfElectronIDSources = cms.VInputTag(
    cms.InputTag("eidLoose"),
    cms.InputTag("eidRobustHighEnergy"),
    cms.InputTag("eidRobustLoose"),
    cms.InputTag("eidRobustTight"),
    cms.InputTag("eidTight"),
  ),
  gsfElectronIDOutput = cms.vstring(
    "eidLoose",
    "eidRobustHighEnergy",
    "eidRobustLoose",
    "eidRobustTight",
    "eidTight",
    ),
  photonPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("photonEcalPFClusterIsolationProducer"),
        cms.InputTag("photonHcalPFClusterIsolationProducer"),
  ),
  photonPFClusterIsoOutput = cms.vstring(
        "phoEcalPFClusIso",
        "phoHcalPFClusIso",
  ),
  gsfElectronPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("electronEcalPFClusterIsolationProducer"),
        cms.InputTag("electronHcalPFClusterIsolationProducer"),
  ),
  gsfElectronPFClusterIsoOutput = cms.vstring(
        "eleEcalPFClusIso",
        "eleHcalPFClusIso",
  ),
)

from CondCore.DBCommon.CondDBCommon_cfi import CondDBCommon
gedelectronGBRESSource = cms.ESSource("PoolDBESSource",
                                      CondDBCommon,
                                      DumpStat=cms.untracked.bool(False),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('GBRWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_p4combination_25ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_p4combination_25ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EBCorrection_25ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EBCorrection_25ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EBUncertainty_25ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EBUncertainty_25ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EECorrection_25ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EECorrection_25ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EEUncertainty_25ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EEUncertainty_25ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_p4combination_50ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_p4combination_50ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EBCorrection_50ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EBCorrection_50ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EBUncertainty_50ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EBUncertainty_50ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EECorrection_50ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EECorrection_50ns')
                                                                 ),
                                                        cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                                 tag = cms.string('GBRForestD_gedelectron_EEUncertainty_50ns_v0'),
                                                                 label = cms.untracked.string('gedelectron_EEUncertainty_50ns')
                                                                 ),
                                                        )
                                      )
gedelectronGBRESSource.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
gedelectronPrefer = cms.ESPrefer('PoolDBESSource',
                                 'gedelectronGBRESSource',
                                 GBRWrapperRcd  = cms.vstring('GBRForest/gedelectron_p4combination_25ns',
                                                              'GBRForest/gedelectron_p4combination_50ns'),
                                 GBRDWrapperRcd = cms.vstring('GBRForestD/gedelectron_EBCorrection_25ns',
                                                              'GBRForestD/gedelectron_EBUncertainty_25ns',
                                                              'GBRForestD/gedelectron_EECorrection_25ns',
                                                              'GBRForestD/gedelectron_EEUncertainty_25ns',
                                                              'GBRForestD/gedelectron_EBCorrection_50ns',
                                                              'GBRForestD/gedelectron_EBUncertainty_50ns',
                                                              'GBRForestD/gedelectron_EECorrection_50ns',
                                                              'GBRForestD/gedelectron_EEUncertainty_50ns',
                                                              )
                                 )

gedphotonGBRESSource = cms.ESSource("PoolDBESSource",
                                    CondDBCommon,
                                    DumpStat=cms.untracked.bool(False),
                                    toGet = cms.VPSet(cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EBCorrection_25ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EBCorrection_25ns')
                                                               ),
                                                      cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EBUncertainty_25ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EBUncertainty_25ns')
                                                               ),
                                                      cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EECorrection_25ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EECorrection_25ns')
                                                               ),
                                                      cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EEUncertainty_25ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EEUncertainty_25ns')
                                                               ),
                                                      cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EBCorrection_50ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EBCorrection_50ns')
                                                               ),
                                                      cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EBUncertainty_50ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EBUncertainty_50ns')
                                                               ),
                                                      cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EECorrection_50ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EECorrection_50ns')
                                                               ),
                                                      cms.PSet(record = cms.string('GBRDWrapperRcd'),
                                                               tag = cms.string('GBRForestD_gedphoton_EEUncertainty_50ns_v0'),
                                                               label = cms.untracked.string('gedphoton_EEUncertainty_50ns')
                                                               ),
                                                      )
                                    )
gedphotonGBRESSource.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
gedphotonPrefer = cms.ESPrefer('PoolDBESSource',
                               'gedphotonGBRESSource',
                               GBRDWrapperRcd = cms.vstring('GBRForestD/gedphoton_EBCorrection_25ns',
                                                            'GBRForestD/gedphoton_EBUncertainty_25ns',
                                                            'GBRForestD/gedphoton_EECorrection_25ns',
                                                            'GBRForestD/gedphoton_EEUncertainty_25ns',
                                                            'GBRForestD/gedphoton_EBCorrection_50ns',
                                                            'GBRForestD/gedphoton_EBUncertainty_50ns',
                                                            'GBRForestD/gedphoton_EECorrection_50ns',
                                                            'GBRForestD/gedphoton_EEUncertainty_50ns',
                                                            )
                               )



