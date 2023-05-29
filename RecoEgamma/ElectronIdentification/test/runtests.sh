function die { echo $1: status $2 ;  exit $2; }

ids_to_test=(
    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV60_cff'
    'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff'
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_50ns_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_50ns_V2_cff'
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff'
    'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff'
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff'
    'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff'
                    
		    
)

for id_set in "${ids_to_test[@]}"; do
    echo Checking: $id_set
    cmsRun ${SCRAM_TEST_PATH}/runElectron_VID.py 1 $id_set || die "Failure using runElectron_VID.py on AOD $id_set" $?
    cmsRun ${SCRAM_TEST_PATH}/runElectron_VID.py 0 $id_set || die "Failure using runElectron_VID.py on MiniAOD $id_set" $?
done
