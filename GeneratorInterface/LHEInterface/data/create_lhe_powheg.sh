#!/bin/bash

#set -o verbose
EXPECTED_ARGS=6

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` repository name process card Nevents RandomSeed "
  echo "process names are: Dijet Zj WW hvq WZ  W_ew-BW Wbb Wj VBF_Hgg_H W Z  Wp_Wp_J_J VBF_Wp_Wp ZZ"  
  echo "Example: ./create_lhe_powheg.sh slc5_ia32_gcc434/powheg/V1.0/src powhegboxv1.0_Jan2012 Z slc5_ia32_gcc434/powheg/V1.0/8TeV_Summer12/DYToEE_M-20_8TeV-powheg/v1/DYToEE_M-20_8TeV-powheg.input 1000 1212" 
  exit 1
fi

echo "   ______________________________________     "
echo "         Running Powheg                       "
echo "   ______________________________________     "

repo=${1}
echo "%MSG-POWHEG repository = $repo"

name=${2} 
echo "%%MSG-POWHEG name = $name"

process=${3}
echo "%MSG-POWHEG process = $process"

cardinput=${4}
echo "%MSG-POWHEG location of the card = $cardinput"

nevt=${5}
echo "%MSG-POWHEG number of events requested = $nevt"

rnum=${6}
echo "%MSG-POWHEG random seed used for the run = $rnum"


seed=$rnum
file="events"
# Release to be used to define the environment and the compiler needed
export PRODHOME=`pwd`
export SCRAM_ARCH=slc5_amd64_gcc462
export RELEASE=${CMSSW_VERSION}
export WORKDIR=`pwd`

# Get the input card
wget --no-check-certificate http://cms-project-generators.web.cern.ch/cms-project-generators/${cardinput} -O powheg.input
card="$WORKDIR/powheg.input"

# initialize the CMS environment 
scram project -n ${name} CMSSW ${RELEASE} ; cd ${name} ; mkdir -p work ; cd work
eval `scram runtime -sh`

# force the f77 compiler to be the CMS defined one
ln -s `which gfortran` f77
ln -s `which gfortran` g77
export PATH=`pwd`:${PATH}

# FastJet and LHAPDF
#fastjet-config comes with the paths used at build time.
#we need this to replace with the correct paths obtained from scram tool info fastjet

newinstallationdir=`scram tool info fastjet | grep FASTJET_BASE |cut -d "=" -f2`
cp ${newinstallationdir}/bin/fastjet-config ./fastjet-config.orig

oldinstallationdir=`cat fastjet-config.orig | grep installationdir | head -n 1 | cut -d"=" -f2`
sed -e "s#${oldinstallationdir}#${newinstallationdir}#g" fastjet-config.orig > fastjet-config 
chmod +x fastjet-config

#same for lhapdf
newinstallationdirlha=`scram tool info lhapdf | grep LHAPDF_BASE |cut -d "=" -f2`
cp ${newinstallationdirlha}/bin/lhapdf-config ./lhapdf-config.orig
oldinstallationdirlha=`cat lhapdf-config.orig | grep prefix | head -n 1 | cut -d"=" -f2`
sed -e "s#prefix=${oldinstallationdirlha}#prefix=${newinstallationdirlha}#g" lhapdf-config.orig > lhapdf-config
chmod +x lhapdf-config

#svn checkout --username anonymous --password anonymous svn://powhegbox.mib.infn.it/trunk/POWHEG-BOX
# # retrieve the wanted POWHEG-BOX from the official repository 

wget --no-check-certificate http://cms-project-generators.web.cern.ch/cms-project-generators/${repo}/${name}.tar.gz 
tar xzf ${name}.tar.gz

cd POWHEG-BOX/${process}
mv Makefile Makefile.orig
cat Makefile.orig | sed -e "s#STATIC[ \t]*=[ \t]*-static#STATIC=-dynamic#g" | sed -e "s#PDF[ \t]*=[ \t]*native#PDF=lhapdf#g"> Makefile
make pwhg_main
mkdir workdir
cd workdir
cat ${card} | sed -e "s#SEED#${seed}#g" | sed -e "s#NEVENTS#${nevt}#g" > powheg.input
cat powheg.input
../pwhg_main &> log_${process}_${seed}.txt
mv pwgevents.lhe ${file}_final.lhe
cp ${file}_final.lhe $WORKDIR/.

echo "Output ready with log_${process}_${seed}.txt and ${file}_final.lhe at `pwd`"
echo "End of job on " `date`
exit 0;
