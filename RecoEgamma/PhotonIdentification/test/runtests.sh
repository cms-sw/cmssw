function die { echo $1: status $2 ;  exit $2; }

ids_to_test=(
    'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_25ns_V1_cff'
    'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_50ns_V1_cff'
    'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring16_V2p2_cff'
    'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V1_TrueVtx_cff'
    'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff'
    'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1_cff'
    'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff'
    'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff'
)

for id_set in "${ids_to_test[@]}"; do
    echo Checking: $id_set
    cmsRun ${LOCAL_TEST_DIR}/runPhoton_VID.py 1 $id_set || die "Failure using runPhoton_VID.py on AOD $id_set" $?
    cmsRun ${LOCAL_TEST_DIR}/runPhoton_VID.py 0 $id_set || die "Failure using runPhoton_VID.py on MiniAOD $id_set" $?
done
