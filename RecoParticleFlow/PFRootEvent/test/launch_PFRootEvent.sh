#!/bin/sh

castorDir=/castor/cern.ch/user/p/pjanot/CMSSW331/
castorShDir=\/castor\/cern.ch\/user\/p\/pjanot\/cmst3\/CMSSW331\/
cmsswDir=$CMSSW_BASE/src
rootDir=$CMSSW_BASE/src/RecoParticleFlow/PFRootEvent/test/

for ((file=1;file<=3;file++));
    do
    echo "file = "$file
    firstEvent=0
    totalEvent=9000
    nJobs=10
    for ((job=0;job<nJobs;job++));
	do
	nEvent=$(( ($totalEvent)/($nJobs) ))
	lastEvent=$(( ($firstEvent)+($nEvent) ))
	input="display_QCDForPF_Full_00"${file}".root"
	name="PFRoot_"${file}"_"${job}
	log="Events_"${file}"_"${job}".txt"
	jetout="pfjetBenchmark_Full_"${file}"_"${job}".root"
	metout="pfmetBenchmark_Full_"${file}"_"${job}".root"
	echo $name
        sed -e "s/==FIRST==/${firstEvent}/" -e "s/==LAST==/${lastEvent}/" Macros/particleFlowProcess.C > tmp.C
        sed -e "s/==INPUT==/\/castor\/cern.ch\/user\/p\/pjanot\/cmst3\/CMSSW331\/${input}/" -e "s/==OUTJET==/${jetout}/" -e "s/==OUTMET==/${metout}/" particleFlow.opt > tmp.opt
	firstEvent=$(( lastEvent ))
#Start to write the script
cat > job_${name}.sh << EOF
#!/bin/sh
cd $cmsswDir
eval \`scramv1 runtime -sh\`
export STAGE_HOST=castorcms
export STAGE_SVCCLASS=cmst3
cd -
#commande pour decoder le .cfg
cat > particleFlow.opt << "EOF"
EOF


# add the opt file to the scripr
cat  tmp.opt >> job_${name}.sh

# On poursuit le script
echo "EOF" >> job_${name}.sh
cat >> job_${name}.sh << EOF

cat > particleFlowProcess.C << "EOF"
EOF

# add the .C file to the scripr
cat  tmp.C >> job_${name}.sh

# On poursuit le script
echo "EOF" >> job_${name}.sh
cat >> job_${name}.sh << EOF

root -b particleFlowProcess.C > ${log}

cp $jetout $rootDir.
cp $metout $rootDir.
cp $log $rootDir.

EOF
chmod 755 job_${name}.sh
bsub -q cms8nht3 -J $name -R "mem>2000" $PWD/job_${name}.sh

done
done

