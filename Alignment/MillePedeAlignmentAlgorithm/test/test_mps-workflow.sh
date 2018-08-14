#!/bin/bash

MPS_TEST_DIR=${LOCALTOP}/tmp/$(date '+%G-%m-%d_%H.%M.%S.%N')_${RANDOM}
MPprod_dir=${MPS_TEST_DIR}/MPproduction

check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}

check_for_failure() {
    "${@}" && exit 1 || echo -e "\n ---> Passed test of '${@}'\n\n"
}

clean_up() {
    cd
    rm -rf ${MPS_TEST_DIR}
}
trap clean_up EXIT


echo "========================================"
echo "Testing MPS workflow in '${MPS_TEST_DIR}'."
echo "----------------------------------------"
echo


# create dummy MPproduction area
rm -rf ${MPprod_dir}
mkdir -p ${MPprod_dir}
cd ${MPprod_dir}


# create dummy input file list
input_file_list=${MPprod_dir}/dummy_input_file_list.txt
rm -f ${input_file_list}
for i in $(seq 5)
do
    echo "/this/is/file/number/${i}.root" >> ${input_file_list}
done


# create dummy (previous) campaign
campaign_id=${RANDOM}
mkdir mp$(printf %04d ${campaign_id})


# setup of MP campaigns
check_for_failure mps_setup_new_align.py
check_for_failure mps_setup_new_align.py -t mc -d 'dummy campaign'
check_for_failure mps_setup_new_align.py -t dat -d 'dummy campaign'
check_for_failure mps_setup_new_align.py -t MC
check_for_failure mps_setup_new_align.py -d 'dummy campaign'
check_for_success mps_setup_new_align.py -t data -d 'dummy data campaign'
campaign_id=$((${campaign_id}+1))
check_for_success mps_setup_new_align.py -t MC -d 'dummy MC campaign'
campaign_id=$((${campaign_id}+1))
check_for_success mps_setup_new_align.py -t MC -d 'dummy MC campaign' -c mp$(printf %04d ${campaign_id})
campaign_id=$((${campaign_id}+1))


# proceed with last created campaign
cd mp$(printf %04d ${campaign_id})


# create input db file
input_db_file=test_input.db
check_for_success mps_prepare_input_db.py -g auto:run2_mc -r ${RANDOM} -o ${input_db_file}
surface_tag=$(conddb --db ${input_db_file} listTags | awk '$3 ~ /AlignmentSurfaceDeformations/ {print $1}')


# modify the templates
sed -i "s|\(inputFileList\s*=\).*$|\1 ${input_file_list}|" alignment_config.ini
cat <<EOF >> alignment_config.ini
[dataset:Cosmics0T]
collection     = ALCARECOTkAlCosmicsCTF0T
inputFileList  = ${input_file_list}
cosmicsDecoMode  = true
cosmicsZeroTesla = true
njobs            = 3

[dataset:Cosmics3.8T_PEAK]
collection     = ALCARECOTkAlCosmicsCTF0T
inputFileList  = ${input_file_list}
cosmicsDecoMode  = false
cosmicsZeroTesla = false
njobs            = 10

[dataset:Cosmics0T_PEAK]
collection     = ALCARECOTkAlCosmicsCTF0T
inputFileList  = ${input_file_list}
cosmicsDecoMode  = false
cosmicsZeroTesla = true
njobs            = 1
EOF
cat <<EOF >> universalConfigTemplate.py
tagwriter.setCondition(process,
       connect = "frontier://FrontierProd/CMS_CONDITIONS",
       record = "TrackerAlignmentErrorExtendedRcd",
       tag = "TrackerIdealGeometryErrorsExtended210_mc")
tagwriter.setCondition(process,
       connect = "sqlite_file:$(pwd)/${input_db_file}",
       record = "TrackerSurfaceDeformationRcd",
       tag = "${surface_tag}")
EOF


# checking the setup of the job folders of a campaign
check_for_failure mps_alisetup.py
sed -i "s|\(\[general\]\)\s*$|\1\ntestMode = true|" alignment_config.ini
check_for_failure mps_alisetup.py alignment_config.ini
sed -i "s|\(FirstRunForStartGeometry\s*=\).*$|\1 1\nmassStorageDir = /store/nothing/MSS|" alignment_config.ini
check_for_success mps_alisetup.py alignment_config.ini


# checking the weight assignment
cat <<EOF >> alignment_config.ini
weight = peak_cosmics

[weights]
peak_cosmics = 3.0
EOF
check_for_failure mps_alisetup.py -w
check_for_success mps_alisetup.py -w alignment_config.ini


# clean up
clean_up
