#!/bin/bash
#
# Run script template for Pede job, copying binary files from mass storage to local disk.
#
# Adjustments might be needed for CMSSW environment.

# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo -e "Running at $(date) \n        on ${HOSTNAME} \n        in directory ${BATCH_DIR}."

# set up the CMS environment
cd CMSSW_RELEASE_AREA
eval `scram runtime -sh`
hash -r

cd ${BATCH_DIR}
echo Running directory changed to $(pwd).

# these defaults will be overwritten by MPS
RUNDIR=${HOME}/scratch0/some/path
MSSDIR=/castor/cern.ch/user/u/username/another/path
MSSDIRPOOL=
CONFIG_FILE=

export X509_USER_PROXY=${RUNDIR}/.user_proxy

#get list of treefiles
TREEFILELIST=
if [ "${MSSDIRPOOL}" != "cmscafuser" ]
then
    :                           # do nothing
else
    TREEFILELIST=$(ls -l ${MSSDIR}/tree_files)
fi
if [[ -z "${TREEFILELIST}" ]]
then
    echo -e "\nThe list of treefiles seems to be empty.\n"
fi

clean_up () {
#try to recover log files and root files
    echo try to recover log files and root files ...
    cp -p pede.dump* ${RUNDIR}
    cp -p *.txt.* ${RUNDIR}
    cp -p *.log ${RUNDIR}
    cp -p *.log.gz ${RUNDIR}
    cp -p millePedeMonitor*root ${RUNDIR}
    cp -p millepede.res* ${RUNDIR}
    cp -p millepede.end ${RUNDIR}
    cp -p millepede.his* ${RUNDIR}
    cp -p *.db ${RUNDIR}
    exit
}
#LSF signals according to http://batch.web.cern.ch/batch/lsf-return-codes.html
trap clean_up HUP INT TERM SEGV USR2 XCPU XFSZ IO


# a helper function to repeatedly try failing copy commands
untilSuccess () {
# trying "${1} ${2} ${3} > /dev/null" until success, if ${4} is a
# positive number run {1} with -f flag,
# break after ${5} tries (with four arguments do up to 5 tries).
    if  [[ ${#} -lt 4 || ${#} -gt 5 ]]
    then
        echo ${0} needs 4 or 5 arguments
        return 1
    fi

    TRIES=0
    MAX_TRIES=5
    if [[ ${#} -eq 5 ]]
    then
        MAX_TRIES=${5}
    fi


    if [[ ${4} -gt 0 ]]
    then 
        ${1} -f ${2} ${3} > /dev/null
    else 
        ${1} ${2} ${3} > /dev/null
    fi
    while [[ ${?} -ne 0 ]]
    do # if not successfull, retry...
        if [[ ${TRIES} -ge ${MAX_TRIES} ]]
        then # ... but not until infinity!
            if [[ ${4} -gt 0 ]]
            then
                echo ${0}: Give up doing \"${1} -f ${2} ${3} \> /dev/null\".
                return 1
            else
                echo ${0}: Give up doing \"${1} ${2} ${3} \> /dev/null\".
                return 1
            fi
        fi
        TRIES=$((${TRIES}+1))
        if [[ ${4} -gt 0 ]]
        then
            echo ${0}: WARNING, problems with \"${1} -f ${2} ${3} \> /dev/null\", try again.
            sleep $((${TRIES}*5)) # for before each wait a litte longer...
            ${1} -f ${2} ${3} > /dev/null
        else
            echo ${0}: WARNING, problems with \"${1} ${2} ${3} \> /dev/null\", try again.
            sleep $((${TRIES}*5)) # for before each wait a litte longer...
            ${1} ${2} ${3} > /dev/null
        fi
    done

    if [[ ${4} -gt 0 ]]
    then
        echo successfully executed \"${1} -f ${2} ${3} \> /dev/null\"
    else
        echo successfully executed \"${1} ${2} ${3} \> /dev/null\"
    fi
    return 0
}

copytreefile () {
    CHECKFILE=`echo ${TREEFILELIST} | grep -i ${2}`
    if [[ -z "${TREEFILELIST}" ]]
    then
        untilSuccess ${1} ${2} ${3} ${4}
    else
        if [[ -n "${CHECKFILE}" ]]
        then
            untilSuccess ${1} ${2} ${3} ${4}
        fi
    fi
}

# stage and copy the binary file(s), first set castor pool for binary files in ${MSSDIR} area
export -f untilSuccess
export -f copytreefile
rm -rf stager_get-commands.txt; touch stager_get-commands.txt
rm -rf parallel-copy-commands.txt; touch parallel-copy-commands.txt
if [ "${MSSDIRPOOL}" != "cmscafuser" ]; then
# Not using cmscafuser pool => rfcp command must be used
  export STAGE_SVCCLASS=${MSSDIRPOOL}
  export STAGER_TRACE=
  echo stager_get -M ${MSSDIR}/milleBinaryISN.dat.gz >> stager_get-commands.txt
  echo untilSuccess rfcp ${MSSDIR}/milleBinaryISN.dat.gz ${BATCH_DIR} 0 >> parallel-copy-commands.txt
  echo stager_get -M ${MSSDIR}/treeFileISN.root >> stager_get-commands.txt
  echo copytreefile rfcp ${MSSDIR}/treeFileISN.root ${BATCH_DIR} 0 >> parallel-copy-commands.txt
else
  MSSCAFDIR=`echo ${MSSDIR} | perl -pe 's/\/castor\/cern.ch\/cms//gi'`
  echo untilSuccess xrdcp ${MSSCAFDIR}/binaries/milleBinaryISN.dat.gz milleBinaryISN.dat.gz 1 >> parallel-copy-commands.txt
  echo copytreefile xrdcp ${MSSCAFDIR}/tree_files/treeFileISN.root treeFileISN.root 1 >> parallel-copy-commands.txt
fi
xargs -a stager_get-commands.txt -n 1 -P 10 -I {} bash -c '$@' _ {}
xargs -a parallel-copy-commands.txt -n 1 -P 10 -I {} bash -c '$@' _ {}
rm stager_get-commands.txt
rm parallel-copy-commands.txt


# We have gzipped binaries, but the python config looks for .dat
# (could also try to substitute in config ".dat" with ".dat.gz"
#  ONLY for lines which contain "milleBinary" using "sed '/milleBinary/s/.dat/.dat.gz/g'"):
ln -s milleBinaryISN.dat.gz milleBinaryISN.dat

cd ${BATCH_DIR}
echo Running directory changed to $(pwd).

echo -e "\nDirectory content before running cmsRun:"
ls -lh
# Execute. The cfg file name will be overwritten by MPS
time cmsRun ${CONFIG_FILE}

# clean up what has been staged in (to avoid copy mistakes...)
rm treeFileISN.root
rm milleBinaryISN.dat.gz milleBinaryISN.dat

# Gzip one by one in case one argument cannot be expanded:
gzip -f *.log
gzip -f *.txt
gzip -f *.dump

#Try to merge millepede monitor files. This only works successfully if names were assigned to jobs.
mps_merge_millepedemonitor.pl ${RUNDIR}/../../mps.db ${RUNDIR}/../../

# Merge possible alignment monitor and millepede monitor hists...
# ...and remove individual histogram files after merging to save space (if success):
# NOTE: the names "histograms.root" and "millePedeMonitor.root" must match what is in
#      your  alignment_cfg.py!
#hadd histograms_merge.root ${RUNDIR}/../job???/histograms.root
#if [ $? -eq 0 ]; then
#    rm ${RUNDIR}/../job???/histograms.root
#fi
hadd millePedeMonitor_merge.root ${RUNDIR}/../job???/millePedeMonitor*.root
if [[ ${?} -eq 0 ]]
then
    rm ${RUNDIR}/../job???/millePedeMonitor*.root
else
    rm millePedeMonitor_merge.root
fi

# Macro creating chi2ndfperbinary.pdf with pede chi2/ndf information hists:
if [[ -e ${CMSSW_BASE}/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C ]]
then
    # Checked out version if existing:
    cp ${CMSSW_BASE}/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C .
else
    # If nothing checked out, take from release:
    cp ${CMSSW_RELEASE_BASE}/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C .
fi
mps_parse_pedechi2hist.py -d ${RUNDIR}/../../mps.db --his millepede.his -c ${CONFIG_FILE}
if [[ -f chi2pedehis.txt ]]
then
    root -l -x -b -q 'createChi2ndfplot.C+("chi2pedehis.txt")'
fi

# Macro creating millepede.his.pdf with pede information hists:
if [[ -e ${CMSSW_BASE}/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C ]]
then
    # Checked out version if existing:
    cp ${CMSSW_BASE}/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C .
else
    # If nothing checked out, take from release:
    cp ${CMSSW_RELEASE_BASE}/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C .
fi
root -b -q "readPedeHists.C+(\"print nodraw\")"

# zip plot files:
gzip -f *.pdf
# now zip .his and .res:
gzip -f millepede.*s
# in case of diagonalisation zip this:
gzip -f millepede.eve
# zip monitoring file:
gzip -f millepede.mon

#list IOVs
for tag in $(sqlite3 alignments_MP.db  "SELECT NAME FROM TAG;")
do
    conddb --db alignments_MP.db list ${tag}
done

#split the IOVs
aligncond_split_iov.sh alignments_MP.db alignments_split_MP.db

echo -e "\nDirectory content after running cmsRun, zipping log file and merging histogram files:"
ls -lh
# Copy everything you need to MPS directory of your job
# (separate cp's for each item, otherwise you loose all if one file is missing):
cp -p *.root ${RUNDIR}
cp -p *.gz ${RUNDIR}
cp -p *.db ${RUNDIR}
cp -p *.end ${RUNDIR}

# create symlinks of the monitoring files in other (possibly non-existing) jobm folders
nTry=0
while true
do
    if [[ ${nTry} -ge 10 ]]     # wait up to 10 times for monitoring files
    then
        break
    fi

    monFiles=$(ls --color=never ${RUNDIR}/../jobm*/*.root | egrep -i 'millepedemonitor_.+\.root$')
    if [[ ${?} -eq 0 ]]
    then
        monFiles=$(echo ${monFiles} | xargs -n 1 readlink -e)
        break
    else
        sleep 60                # wait a minute and retry
        nTry=$((${nTry} + 1))
    fi
done
jobmFolders=$(ls --color=never -d ${RUNDIR}/../jobm* | xargs -n 1 readlink -e)
for folder in ${jobmFolders}
do
    for mon in ${monFiles}
    do
        ln -s ${mon} ${folder}/ > /dev/null 2>&1
        ln -s ${mon} > /dev/null 2>&1 # make them also available here
    done
done

# copy aligment_merge.py for mps_validate.py
ln -s ${CONFIG_FILE} alignment_merge.py
ln -s ${RUNDIR}/.TrackerTree.root
ln -s ${RUNDIR}/.weights.pkl
# run mps_validate.py
campaign=`basename ${MSSDIR}`
mps_validate.py -m ${campaign} -p ./

cp -pr validation_output ${RUNDIR}
