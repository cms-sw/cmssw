from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import ecalDrivenGsfElectrons

gedGsfElectronsTmp = ecalDrivenGsfElectrons.clone(

    # input collections
    gsfElectronCoresTag = "gedGsfElectronCores",

    # steering
    resetMvaValuesUsingPFCandidates = True,
    applyPreselection = True,
    ecalDrivenEcalEnergyFromClassBasedParameterization = False,
    ecalDrivenEcalErrorFromClassBasedParameterization = False,
    useEcalRegression = True,
    useCombinationRegression = True,

    # regression. The labels are needed in all cases.
    ecalRefinedRegressionWeightLabels = ["gedelectron_EBCorrection_offline_v1",
                                         "gedelectron_EECorrection_offline_v1",
                                         "gedelectron_EBUncertainty_offline_v1",
                                         "gedelectron_EEUncertainty_offline_v1"],
    combinationRegressionWeightLabels = ["gedelectron_p4combination_offline"],
)


from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(gedGsfElectronsTmp.preselection, minSCEtBarrel = 15.0)
pp_on_AA.toModify(gedGsfElectronsTmp.preselection, minSCEtEndcaps = 15.0)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(gedGsfElectronsTmp.preselection,
                                minSCEtBarrel = 1.0, 
                                minSCEtEndcaps = 1.0)
egamma_lowPt_exclusive.toModify(gedGsfElectronsTmp, applyPreselection = False) 
