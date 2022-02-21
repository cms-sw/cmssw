#! /bin/bash

# command line input
suite=${1:-"forPR"} # which set of benchmarks to run: full, forPR, forConf
useARCH=${2:-0} # which computer cluster to run on. 0=phi3, 1=lnx, 2= phi3+lnx, 3=phi123, 4=phi123+lnx
lnxuser=${3:-${USER}} #username for lnx computers

# samples
export sample=${sample:-"CMSSW_TTbar_PU50"}

# Validation architecture
export val_arch=SKL-SP

# vars for KNL
export KNL_HOST=${USER}@phi2.t2.ucsd.edu
export KNL_WORKDIR=/data1/work/${USER}
export KNL_TEMPDIR=tmp

# vars for SNB
export SNB_HOST=${USER}@phi1.t2.ucsd.edu
export SNB_WORKDIR=/data2/nfsmic/${USER}
export SNB_TEMPDIR=tmp

# vars for LNX7188
export LNXG_HOST=${lnxuser}@lnx7188.classe.cornell.edu
export LNXG_WORKDIR=/home/${lnxuser}
export LNXG_TEMPDIR=/tmp/tmp7188

# vars for LNX4108
export LNXS_HOST=${lnxuser}@lnx4108.classe.cornell.edu
export LNXS_WORKDIR=/home/${lnxuser}
export LNXS_TEMPDIR=/tmp/tmp4108

# SSH options
function SSHO()
{
    ssh -o StrictHostKeyChecking=no < /dev/null "$@"
}
export -f SSHO

#################
## Build Types ##
#################

export BH="BH bh"
export STD="STD std"
export CE="CE ce"

# which set of builds to use based on input from command line
if [[ "${suite}" == "full" ]]
then
    declare -a ben_builds=(BH STD CE)
    declare -a val_builds=(BH STD CE)
elif [[ "${suite}" == "forPR" ]]
then
    declare -a ben_builds=(BH CE)
    declare -a val_builds=(STD CE)
elif [[ "${suite}" == "forConf" ]]
then
    declare -a ben_builds=(CE)
    declare -a val_builds=(CE)
elif [[ "${suite}" == "val" || "${suite}" == "valMT1" ]]
then
    declare -a ben_builds=()
    declare -a val_builds=(STD CE)
else
    echo ${suite} "is not a valid benchmarking suite option! Exiting..."
    exit
fi

# set dependent arrays
th_builds=() ## for parallelization tests
vu_builds=() ## for vectorization tests
meif_builds=() ## for multiple-events-in-flight tests
text_builds=() ## for text dump comparison tests

# loop over ben_builds and set dependent arrays, export when done
for ben_build in "${ben_builds[@]}"
do
    # set th builds : all benchmarks!
    th_builds+=("${ben_build}")
    vu_builds+=("${ben_build}")
    
    # set meif builds : only do CE
    if [[ "${ben_build}" == "CE" ]]
    then
	meif_builds+=("${ben_build}")
    fi
done
export ben_builds val_builds th_builds vu_builds meif_builds

# th checking
function CheckIfTH ()
{
    local build=${1}
    local result="false"

    for th_build in "${th_builds[@]}"
    do 
	if [[ "${th_build}" == "${build}" ]]
	then
	    result="true"
	    break
	fi
    done
    
    echo "${result}"
}
export -f CheckIfTH

# vu checking
function CheckIfVU ()
{
    local build=${1}
    local result="false"

    for vu_build in "${vu_builds[@]}"
    do 
	if [[ "${vu_build}" == "${build}" ]]
	then
	    result="true"
	    break
	fi
    done
    
    echo "${result}"
}
export -f CheckIfVU

# meif checking
function CheckIfMEIF ()
{
    local build=${1}
    local result="false"

    for meif_build in "${meif_builds[@]}"
    do 
	if [[ "${meif_build}" == "${build}" ]]
	then
	    result="true"
	    break
	fi
    done
    
    echo "${result}"
}
export -f CheckIfMEIF

# set text dump builds: need builds matched in both TH and VU tests
for ben_build in "${ben_builds[@]}"
do 
    check_th=$( CheckIfTH ${ben_build} )
    check_vu=$( CheckIfVU ${ben_build} )

    if [[ "${check_th}" == "true" ]] && [[ "${check_vu}" == "true" ]]
    then
	text_builds+=("${ben_build}")
    fi
done

export text_builds

# text checking
function CheckIfText ()
{
    local build=${1}
    local result="false"

    for text_build in "${text_builds[@]}"
    do 
	if [[ "${text_build}" == "${build}" ]]
	then
	    result="true"
	    break
	fi
    done

    echo "${result}"
}
export -f CheckIfText

Base_Test="NVU1_NTH1"
if [[ ${useARCH} -eq 0 ]]
then
    arch_array=(SKL-SP)
    arch_array_textdump=("SKL-SP ${Base_Test}" "SKL-SP NVU16int_NTH64")
    arch_array_benchmark=("SKL-SP skl-sp")
elif [[ ${useARCH} -eq 1 ]]
then
    arch_array=(LNX-G LNX-S)
    arch_array_textdump=("LNX-G ${Base_Test}" "LNX-G NVU16int_NTH64" "LNX-S ${Base_Test}" "LNX-S NVU16int_NTH64")
    arch_array_benchmark=("LNX-G lnx-g" "LNX-S lnx-s")
elif [[ ${useARCH} -eq 2 ]]
then
    arch_array=(SKL-SP LNX-G LNX-S)
    arch_array_textdump=("SKL-SP ${Base_Test}" "SKL-SP NVU16int_NTH64" "LNX-G ${Base_Test}" "LNX-G NVU16int_NTH64" "LNX-S ${Base_Test}" "LNX-S NVU16int_NTH64")
    arch_array_benchmark=("SKL-SP skl-sp" "LNX-G lnx-g" "LNX-S lnx-s")
elif [[ ${useARCH} -eq 3 ]]
then
    arch_array=(SNB KNL SKL-SP)
    arch_array_textdump=("SNB ${Base_Test}" "SNB NVU8int_NTH24" "KNL ${Base_Test}" "KNL NVU16int_NTH256" "SKL-SP ${Base_Test}" "SKL-SP NVU16int_NTH64")
    arch_array_benchmark=("SNB snb" "KNL knl" "SKL-SP skl-sp")
elif [[ ${useARCH} -eq 4 ]]
then
    arch_array=(SNB KNL SKL-SP LNX-G LNX-S)
    arch_array_textdump=("SNB ${Base_Test}" "SNB NVU8int_NTH24" "KNL ${Base_Test}" "KNL NVU16int_NTH256" "SKL-SP ${Base_Test}" "SKL-SP NVU16int_NTH64" "LNX-G ${Base_Test}" "LNX-G NVU16int_NTH64" "LNX-S ${Base_Test}" "LNX-S NVU16int_NTH64")
    arch_array_benchmark=("SNB snb" "KNL knl" "SKL-SP skl-sp" "LNX-G lnx-g" "LNX-S lnx-s")
else
    echo "${useARCH} is not a valid useARCH option! Exiting..."
    exit
fi
export arch_array arch_array_textdump arch_array_benchmark
