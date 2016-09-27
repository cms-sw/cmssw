#!/bin/bash

dataset="/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISummer15GS-MCRUN2_71_V1-v1/GEN-SIM"
query="file dataset=${dataset} site=T2_CH_CERN"
queue="cmscaf1nh"

################################################################################
if [ x${CMSSW_BASE} = 'x' ]
then
    echo "Please source a CMSSW environment."
    exit 1
fi

################################################################################
eos=/afs/cern.ch/project/eos/installation/cms/bin/eos.select

out_name=$(echo ${dataset} | sed 's|^/||;s|/GEN-SIM||;s|/|_|g;')

submit_dir="submit_${out_name}"
current_dir=$(pwd -P)
eos_dir=/eos/cms/store/caf/user/${USER}/ZMuMuMassConstraintParameterFinder/${out_name}/

rm -rf ${submit_dir}
mkdir ${submit_dir}
${eos} mkdir -p ${eos_dir}

input_files=$(das_client --limit=0 --query="${query}")


# create skim jobs for each input file
cd ${submit_dir}
count=1
for input in ${input_files}
do
    formatted_count=$(printf %04d ${count})
    output=dimuon_mass_${formatted_count}.root
    script_name=dimuon_mass_${formatted_count}.sh
    cat > ${script_name} <<EOF
#!/bin/bash
CWD=\$(pwd -P)
cd ${CMSSW_BASE}/src
eval \`scramv1 ru -sh\`
cd \${CWD}
echo \${CWD}
cp $(readlink -e ../zmumudistribution_cfg.py) .
cmsRun zmumudistribution_cfg.py inputFiles=${input} outputFile=${output}
${eos} cp ${output} ${eos_dir}
EOF
    chmod +x ${script_name}
    bsub_output=$(bsub -q ${queue} ${script_name})
    echo ${bsub_output}
    job_id=$(echo ${bsub_output} | sed 's|^Job <\([0-9]\{9\}\)>.*|\1|')
    if [[ ${count} -eq 1 ]]
    then
        conditions="ended(${job_id})"
    else
        conditions="${conditions} && ended(${job_id})"
    fi
    count=$((count + 1))
done


# merge the output of the skim jobs
output=dimuon_mass.root
script_name=dimuon_mass_merge.sh
cat > ${script_name} <<EOF
#!/bin/bash
CWD=\$(pwd -P)
cd ${CMSSW_BASE}/src
eval \`scramv1 ru -sh\`
cd \${CWD}
echo \${CWD}
${eos} cp -r ${eos_dir} ROOT_FILES/
cd ROOT_FILES
rm ${output}
hadd ${output} *.root
${eos} cp ${output} ${eos_dir}
root -l -b -q "${current_dir}/printParameters.C(\"${output}\")" > ${current_dir}/${submit_dir}/zMuMuMassConstraintParameters.txt
EOF
chmod +x ${script_name}
bsub -q ${queue} -w "${conditions}" ${script_name}
cd ${current_dir}
