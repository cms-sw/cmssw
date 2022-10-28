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

    #Egamma PFID DNN model configuration
    EleDNNPFid= dict(
        outputTensorName = "sequential_1/FinalLayer/Softmax",
        modelsFiles = [
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/lowpT/lowpT_modelDNN.pb",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EB_highpT/barrel_highpT_modelDNN.pb",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EE_highpT/endcap_highpT_modelDNN.pb",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Winter22_122X/exteta1/modelDNN.pb",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Winter22_122X/exteta2/modelDNN.pb"
        ],
        scalersFiles = [
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/lowpT/lowpT_scaler.txt",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EB_highpT/barrel_highpT_scaler.txt",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Summer21_120X/EE_highpT/endcap_highpT_scaler.txt",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Winter22_122X/exteta1/scaler.txt",
            "RecoEgamma/ElectronIdentification/data/Ele_PFID_dnn/Run3Winter22_122X/exteta2/scaler.txt"
        ],
        outputDim = [5,5,5,5,3]
    )    
)



from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(gedGsfElectronsTmp.preselection, minSCEtBarrel = 15.0)
pp_on_AA.toModify(gedGsfElectronsTmp.preselection, minSCEtEndcaps = 15.0)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(gedGsfElectronsTmp.preselection,
                                minSCEtBarrel = 1.0, 
                                minSCEtEndcaps = 1.0)
egamma_lowPt_exclusive.toModify(gedGsfElectronsTmp, applyPreselection = False) 


# Activate the Egamma PFID dnn only for Run3
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(gedGsfElectronsTmp.EleDNNPFid,
    enabled = True
)
