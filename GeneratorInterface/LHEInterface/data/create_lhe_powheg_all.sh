#!/bin/bash

fail_exit() { echo "$@" 1>&2; exit 1; }

#set -o verbose
EXPECTED_ARGS=10

if [ $# -ne $EXPECTED_ARGS ]
then
    echo "Usage: `basename $0` repository name process card preCompiled createTarball tarballRepository tarballName Nevents RandomSeed "
    echo "process names are: Dijet Zj WW hvq WZ  W_ew-BW Wbb Wj VBF_Hgg_H W Z  Wp_Wp_J_J VBF_Wp_Wp ZZ"  
    echo "Example: ./create_lhe_powheg_all.sh slc5_ia32_gcc434/powheg/V1.0/src powhegboxv1.0_Jan2012 Z slc5_ia32_gcc434/powheg/V1.0/8TeV_Summer12/DYToEE_M-20_8TeV-powheg/v1/DYToEE_M-20_8TeV-powheg.input false true slc5_amd64_gcc462/8TeV/powheg Z 1000 1212" 
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

precompile=${5}
echo "%MSG-POWHEG Precompiled or not   = $precompile"

createTarball=${6}
echo "%MSG-POWHEG create tarball or not   = $createTarball"

tarballRepo=${7}
echo "%MSG-POWHEG tarball repository = $tarballRepo"

tarball=${8}
echo "%MSG-POWHEG tar ball file name = $tarball"

nevt=${9}
echo "%MSG-POWHEG number of events requested = $nevt"

rnum=${10}
echo "%MSG-POWHEG random seed used for the run = $rnum"


seed=$rnum
file="events"
# Release to be used to define the environment and the compiler needed
export PRODHOME=`pwd`
export RELEASE=${CMSSW_VERSION}
export WORKDIR=`pwd`

# Get the input card
wget --no-check-certificate http://cms-project-generators.web.cern.ch/cms-project-generators/${cardinput} -O powheg.input  || fail_exit "Failed to obtain input card" ${cardinput}
card="$WORKDIR/powheg.input"

# initialize the CMS environment 
if [[ -e ${name} ]]; then
  mv ${name} old_${name}
  mv output.lhe old_output.lhe
fi

scram project -n ${name} CMSSW ${RELEASE}; cd ${name} ; mkdir -p work ; cd work  
eval `scram runtime -sh`

# force the f77 compiler to be the CMS defined one
#ln -s `which gfortran` f77
#ln -s `which gfortran` g77
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

if [ "$precompile" == "false" ];
then 
    echo "Compile during the run"

    wget --no-check-certificate http://cms-project-generators.web.cern.ch/cms-project-generators/${repo}/${name}.tar.gz  -O ${name}.tar.gz || fail_exit "Failed to get powheg tar ball " ${name}
    tar xzf ${name}.tar.gz

#remove from Powheg the LENOCC function which is already defined in LHAPDF library
    patch POWHEG-BOX/cernroutines.f <<EOF
*** POWHEG-BOX/cernroutines_orig.f	Wed Mar 14 11:48:14 2012
--- POWHEG-BOX/cernroutines.f	Wed Mar 14 11:48:29 2012
***************
*** 790,815 ****
  
  
  
! c# 10 "lenocc.F" 2
!       FUNCTION LENOCC (CHV)
! C
! C CERN PROGLIB# M507    LENOCC          .VERSION KERNFOR  4.21  890323
! C ORIG. March 85, A.Petrilli, re-write 21/02/89, JZ
! C
! C-    Find last non-blank character in CHV
! 
!       CHARACTER    CHV*(*)
! 
!       N = LEN(CHV)
! 
!       DO 17  JJ= N,1,-1
!       IF (CHV(JJ:JJ).NE.' ') GO TO 99
!    17 CONTINUE
!       JJ = 0
! 
!    99 LENOCC = JJ
!       RETURN
!       END
  c# 1 "mtlset.F"
  c# 1 "<built-in>"
  c# 1 "<command line>"
--- 790,815 ----
  
  
  
! ccccccc# 10 "lenocc.F" 2
! cccccc      FUNCTION LENOCC (CHV)
! ccccccC
! ccccccC CERN PROGLIB# M507    LENOCC          .VERSION KERNFOR  4.21  890323
! ccccccC ORIG. March 85, A.Petrilli, re-write 21/02/89, JZ
! ccccccC
! ccccccC-    Find last non-blank character in CHV
! cccccc
! cccccc      CHARACTER    CHV*(*)
! cccccc
! cccccc      N = LEN(CHV)
! cccccc
! cccccc      DO 17  JJ= N,1,-1
! cccccc      IF (CHV(JJ:JJ).NE.' ') GO TO 99
! cccccc   17 CONTINUE
! cccccc      JJ = 0
! cccccc
! cccccc   99 LENOCC = JJ
! cccccc      RETURN
! cccccc      END
  c# 1 "mtlset.F"
  c# 1 "<built-in>"
  c# 1 "<command line>"

EOF

    cd POWHEG-BOX/${process}

    mv Makefile Makefile.orig
    cat Makefile.orig | sed -e "s#STATIC[ \t]*=[ \t]*-static#STATIC=-dynamic#g" | sed -e "s#PDF[ \t]*=[ \t]*native#PDF=lhapdf#g"> Makefile
    echo "LIBS+=-lz -lstdc++" >> Makefile


    LHA_BASE="`readlink -f "$LHAPATH/../../../"`"

#slc5_amd64_gcc462/external/lhapdf/5.8.5 has a bug. if this version is used, replace it by 5.8.5-cms:
    if [ `basename $LHA_BASE` == "5.8.5" ]
    then  
	LHA_BASE="`echo "$LHA_BASE" | sed 's@slc5_amd64_gcc462/external/lhapdf/5.8.5@slc5_amd64_gcc462/external/lhapdf/5.8.5-cms@'`"
    fi

    LHA_BASE_OLD="`$LHA_BASE/bin/lhapdf-config --prefix`"
    cat > lhapdf-config-wrap <<EOF
#!/bin/bash
"$LHA_BASE/bin/lhapdf-config" "\$@" | sed "s|$LHA_BASE_OLD|$LHA_BASE|g"
EOF
    chmod a+x lhapdf-config-wrap

    make LHAPDF_CONFIG="`pwd`/lhapdf-config-wrap" pwhg_main || fail_exit "Failed to compile pwhg_main"

    if [ "$createTarball" == "true" ]
    then
	rm -rf testrun*
	rm -rf Docs
        rm -rf .svn 
        rm -rf obj
        rm -rf *.f
        cd ..
        tar chvzf ${tarball}.tar.gz ${process}
	cp -p ${tarball}.tar.gz ${WORKDIR}/.
        cd ${process}
    fi

else if [ "$precompile" == "true" ];
then
    echo "Using a precompiled tar ball $tarball.tar.gz"
#    wget --no-check-certificate http://cms-project-generators.web.cern.ch/cms-project-generators/${tarballRepo}/${tarball}.tar.gz
    fn-fileget -c `cmsGetFnConnect frontier://smallfiles` ${tarballRepo}/${tarball}.tar.gz || true

    if [[ -e ./${tarball}.tar.gz ]]; then
	tar xvzf ${tarball}.tar.gz
	cd ${process}
    else
	echo "Error! The tar ball $tarball.tar.gz does not exist!"
	exit 1
    fi
fi
fi


mkdir workdir
cd workdir
cat ${card} | sed -e "s#SEED#${seed}#g" | sed -e "s#NEVENTS#${nevt}#g" > powheg.input
cat powheg.input
../pwhg_main &> log_${process}_${seed}.txt
#remove the spurious random seed output that is non LHE standard 
cat pwgevents.lhe | grep -v "Random number generator exit values" > ${file}_final.lhe
ls -l ${file}_final.lhe
pwd
cp ${file}_final.lhe ${WORKDIR}/.
#cp ${file}_final.lhe ${WORKDIR}/${file}_final.lhe
#cp ${file}_final.lhe ${WORKDIR}/output.lhe

echo "Output ready with log_${process}_${seed}.txt and ${file}_final.lhe at `pwd` and $WORKDIR"
echo "End of job on " `date`
exit 0;
