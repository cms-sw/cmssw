#!/bin/tcsh -f

#Script for submiting job using condor
# parameters
#
# $1 - bin number 
# $2 - etaInner
# $3 - etaOuter
# $4 - run number 

echo " submitting: "  ${1}_${2}_${3}_${4} "into the condor queue executing:" 
/bin/rm -f  condor_${1}_${2}_${3}_${4}

cat > condor_${1}_${2}_${3}_${4} << EOF

universe = vanilla
Executable = /uscms/home/${user}/CMSSW_1_2_0/src/RecoJets/JetAnalyzers/test/DijetRatio/submitDiJetAnalysis.csh
Requirements   = (Memory >= 499 && OpSys == "LINUX" && (Arch == "INTEL" || Arch =="x86_64") && (Disk >= DiskUsage) && (TARGET.FileSystemDomain == MY.FileSystemDomain))
Should_Transfer_Files = NO
Output  = /uscms/home/${user}/CMSSW_1_2_0/src/RecoJets/JetAnalyzers/test/DijetRatio/condor_${1}_${2}_${3}_${4}\$(Cluster)_\$(Process).stdout
Error = /uscms/home/${user}/CMSSW_1_2_0/src/RecoJets/JetAnalyzers/test/DijetRatio/condor_${1}_${2}_${3}_${4}\$(Cluster)_\$(Process).stderr
Log = /uscms/home/${user}/CMSSW_1_2_0/src/RecoJets/JetAnalyzers/test/DijetRatio/condor_${1}_${2}_${3}_${4}\$(Cluster)_\$(Process).log

notify_user = ${user}@fnal.gov
#notify_user = hepmkj@gmail.com

Arguments = ${1} ${2} ${3} ${4}
Queue 1

EOF

/opt/condor/bin/condor_submit condor_${1}_${2}_${3}_${4}

