function die { echo $1: status $2 ;  exit $2; }

ids_to_test=(
"RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_PHYS14_PU20bx25_V2_cff"
"RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff"
"RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_50ns_V1_cff"
"RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV60_cff"
"RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff"
)

for id_set in "${ids_to_test[@]}"; do
    echo Checking: $id_set
    cmsRun ${LOCAL_TEST_DIR}/runElectron_VID.py 1 $id_set || die "Failure using runElectron_VID.py on AOD $id_set" $?
    cmsRun ${LOCAL_TEST_DIR}/runElectron_VID.py 0 $id_set || die "Failure using runElectron_VID.py on MiniAOD $id_set" $?
done