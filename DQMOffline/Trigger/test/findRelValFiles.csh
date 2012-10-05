


# set myTarget = "/RelValTTbar/CMSSW_5_2_0_pre1-START50_V9-v1/GEN-SIM-RECO"
# dbs search  --query="find file.name where dataset = '$myTarget'"

# echo "=========== Getting list of sites =============="
# dbs search  --query="find site.name where dataset = '$myTarget'"


echo "=========== Getting list of datasets =============="
#dbs search  --query="find dataset.name,site.name,file.name where dataset = '*RelValTTbar*CMSSW_6_1_0_pre*GEN-SIM-RECO' and site.name like '*fnal.gov'"
dbs search  --query="find dataset.name,site.name,file.name where dataset = '*RelValTTbar*CMSSW_6_1_0_pre*GEN-SIM-RECO' "

