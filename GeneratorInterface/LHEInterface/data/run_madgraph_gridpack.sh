#!/bin/bash

#set -o verbose
set -e

echo "   ______________________________________     "
echo "         Running Madgraph5....                "
echo "   ______________________________________     "

repo=${1}
echo "%MSG-MG5 repository = $repo"

name=${2} 
echo "%MSG-MG5 gridpack = $name"

decay=${3}
echo "%MSG-MG5 run decay = $decay"

replace=${4}
echo "%MSG-MG5 replace = $replace"

process=${5}
echo "%MSG-MG5 process = $process"

maxjetflavor=${6}
echo "%MSG-MG5 maxjetflavor = $maxjetflavor"

qcut=${7}
echo "%MSG-MG5 qcut = $qcut"

minmax_jet=${8}
echo "%MSG-MG5 minmax_jet = $minmax_jet"

min_jets=${9}
max_jets=${10}
echo "%MSG-MG5 min/max jet multiplicity = $min_jets / $max_jets"

nevt=${11}
echo "%MSG-MG5 number of events requested = $nevt"

rnum=${12}
echo "%MSG-MG5 random seed used for the run = $rnum"

# retrieve the wanted gridpack from the official repository
## || true is a temporary workaround to work with option -e and overcome the fact that the exist code of fn-fileget is 1 even if it succeeds
fn-fileget -c `cmsGetFnConnect frontier://smallfiles` ${repo}/${name}_gridpack.tar.gz || true

# force the f77 compiler to be the CMS defined one
ln -sf `which gfortran` f77
ln -sf `which gfortran` g77
PATH=`pwd`:${PATH}

tar xzf ${name}_gridpack.tar.gz ; rm -f ${name}_gridpack.tar.gz ; cd madevent
## compile according to MG version 1.3.30 or 1.4.3 

########### BEGIN - REPLACE process ################
# REPLACE script is runned automatically by run.sh if REPLACE dir is found ###
# REPLACE will replace el with el/mu/taus by default, if you need something else you need to edit the replace_card1.dat
if [ ${replace} == true ] ; then
  echo "%MSG-MG5: REPLACE is not working on SLC6 nodes and had been disabled." 
  echo "%MSG-MG5: Please produce your gridpack with explicit lepton flavours in the proc_card.dat"
#########THE USE OF REPLACE HAS BEEN DISABLED!
#########this is due to a clash between zlib in cmssw and Compress::Zlib in SLC6 nodes 
#    cd ..
#    mkdir REPLACE
#    cat > replace_card1.dat <<EOF
## Enter here any particles you want replaced in the event file after ME run
## In the syntax PID : PID1 PID2 PID3 ...
## End with "done" or <newline>
#11:11 13 15
#-12: -12 -14 -16
#-11:-11 -13 -15
#12: 12 14 16
#done
#EOF
#    cp  ./madevent/bin/replace.pl  ./replace_card1.dat  ./REPLACE/
#    chmod a+x ./REPLACE/replace.pl
#    cd -
  exit 1
fi	
########## END - REPLACE #########################


###### BEGIN - DECAY process #################
# Decay is runned automatically by run.sh if DECAY dir is found
# To avoid compilation problem DECAY dir has been created and compiled when gridpack is created
# if not needed DECAY dir is deleted

if [ "${decay}" != true ] ; then
    rm -rf DECAY
else
  cd ..
  cp  -r ./madevent/bin/internal/DECAY .
  cd -
fi

######END - DECAY #####

# run the production stage
cd ..
./run.sh ${nevt} ${rnum}

ls -al

file="events"

if [ ! -f ${file}.lhe.gz ]; then
        echo "%MSG-MG5 events.lhe.gz file is not in the same folder with run.sh script, abort  !!! "
        exit 1

fi

cp ${file}.lhe.gz ${file}_orig.lhe.gz
gzip -d ${file}.lhe.gz

#_______________________________________________________________________________________
# check the seed number in LHE file.

echo "   ______________________________________     "
echo "         post processing started              "
echo "   ______________________________________     "

echo 
if [ -f ${file}.lhe ] ; then
        seed=`awk 'BEGIN{FS=" = gseed  "}/gseed/{print $1}' ${file}.lhe`
        number_event=`grep -c "</event>" ${file}.lhe`
fi

if [ $seed -eq $rnum ] ;then
                echo "GSEED  :$seed"
                if [ $number_event -eq $nevt ] ;then
                        echo "NEVENT :  $nevt "
                else
                        echo "%MSG-MG5 Error: The are less events ( $number_event ) Post Production is cancelled."
                        # TO-DO You might want to save the events in case of inspection the events.
                        exit 1
                fi
else
        echo "%MSG-MG5 Error: Seed numbers doesnt match ( $seed )"
        exit 1
fi

#_______________________________________________________________________________________
# post-process the LHE file.


#__________________________________________
# wjets/zjets

if [[ ${process} == wjets || ${process} == zjets ]] ; then
	echo "%MSG-MG5 process V+jets"
	python madevent/bin/mgPostProcv2.py -o ${file}_qcut${qcut}_mgPostv2.lhe -m -w -j ${maxjetflavor} -q ${qcut} -e 5 -s ${file}.lhe
fi

# qcd 
if [ ${process} == qcd ] ; then
	echo "%MSG-MG5 process QCD"
	python madevent/bin/mgPostProcv2.py -o ${file}_qcut${qcut}_mgPostv2.lhe -q ${qcut} -j ${maxjetflavor} -e 5 -s ${file}.lhe
fi

# ttbar
if [ ${process} == ttbar ] ; then
	echo "%MSG-MG5 process ttbar"
	python madevent/bin/mgPostProcv2.py -o ${file}_qcut${qcut}_mgPostv2.lhe  -m -w -t -j ${maxjetflavor} -q ${qcut} -e 5 -s ${file}.lhe
	sed -i -e '/Rnd seed/d'  -e '/MC partial width/d' -e '/Number of Events/d' -e '/Max wgt/d' -e '/Average wgt/d'   -e '/Integrated weight/d' ${file}_qcut${qcut}_mgPostv2.lhe
fi

#__________________________________________
# If you have HT binned samples min/max jets might be different from file to file. 
# So you can override the min/max jets decision and put by hand these from the command line 

if [ $minmax_jet == true ] ;then

	sed -i "s/ [0-9]* = minjets    ! Smallest number of additional light flavour jets/ $min_jets = minjets    ! Smallest number of additional light flavour jets/g" \
	${file}_qcut${qcut}_mgPostv2.lhe 
	sed -i "s/ [0-9]* = maxjets    ! Largest number (inclusive ktMLM matching multipl.)/ $max_jets = maxjets    ! Largest number  (inclusive ktMLM matching multipl.)/g" \
	${file}_qcut${qcut}_mgPostv2.lhe 
fi

mv ${file}_qcut${qcut}_mgPostv2.lhe ${name}_final.lhe 

ls -l
echo

exit 0
