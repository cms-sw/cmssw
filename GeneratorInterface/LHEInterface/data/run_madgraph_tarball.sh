#!/bin/bash

#set -o verbose

echo "   ______________________________________     "
echo "         Running Madgraph5                    "
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


if [[ -d madevent ]]
    then
    echo 'madevent directory found'
    echo 'Setting up the environment'
    rm -rf madevent
fi
mkdir madevent; cd madevent

# retrieve the wanted gridpack from the official repository 
fn-fileget -c `cmsGetFnConnect frontier://smallfiles` ${repo}/${name}_tarball.tar.gz 


#check the structure of the tarball
tar xzf ${name}_tarball.tar.gz ; rm -f ${name}_tarball.tar.gz ;

# force the f77 compiler to be the CMS defined one
ln -sf `which gfortran` f77
ln -sf `which gfortran` g77
PATH=`pwd`:${PATH}

#replace the seed in the run card with ${rnum}
run_card_seed=`awk 'BEGIN{FS=" = iseed"}/iseed/{print $1}' Cards/run_card.dat`
echo "run_card_seed = ${run_card_seed}"
sed -i -e "s#${run_card_seed}.*.= iseed#${rnum}  = iseed#g" Cards/run_card.dat
new_run_card_seed=`awk 'BEGIN{FS=" = iseed"}/iseed/{print $1}' Cards/run_card.dat`
echo "new_run_card_seed = ${new_run_card_seed}"

#replace the number of events in the run_card
run_card_nevents=`awk 'BEGIN{FS=" = nevents"}/nevents/{print $1}' Cards/run_card.dat`
echo "run_card_nevents = ${run_card_nevents}"
sed -i -e "s#${run_card_nevents}.*.= nevents#${nevt}  = nevents#g" Cards/run_card.dat
new_run_card_nevents=`awk 'BEGIN{FS=" = nevents"}/nevents/{print $1}' Cards/run_card.dat`
echo "new_run_card_nevents = ${new_run_card_nevents}"


version=`cat MGMEVersion.txt | grep -c "1.4"`

#generate events
bin/generate_events 0 ${name}
cd ../

file="${name}_unweighted_events"
if [ "$version" -eq "0" ] ; then
mv madevent/Events/${file}.lhe.gz ./
fi

if [ "$version" -eq "1" ] ; then
mv madevent/Events/${name}/unweighted_events.lhe.gz ${file}.lhe.gz
fi

if [ ! -f ${file}.lhe.gz ]; then
        echo "%MSG-MG5 events.lhe.gz file is not in the same folder with run.sh script, abort  !!! "
        exit
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
        seed=`awk 'BEGIN{FS=" = iseed  "}/iseed/{print $1}' ${file}.lhe`
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
# DECAY process
if [ "${decay}" == true ] ; then

    echo "%MSG-MG5 Running DECAY..."
    bm=`grep -c "# MB" ${file}.lhe`
    zero=0;
    if [ $bm -eq $zero ] ;then
      sed 's/  5 0.000000 # b : 0.0/  5  4.800000 # b/' ${file}.lhe > ${file}_in.lhe ; rm -f ${file}.lhe
    fi

    if [ $bm -gt $zero ] ;then
      sed  's/5 0.000000e+00 # MB/5 4.800000e+00 # MB/g' ${file}.lhe > ${file}_in.lhe ; rm -f ${file}.lhe
    fi
    # if you want to do not-inclusive top-decays you have to modify the switch in the decay_1.in and decay_2.in
    for (( i = 1; i <=2; i++)) ; do
        if [ -f ${file}.lhe ] ; then
           mv ${file}.lhe ${file}_in.lhe
        fi
        madevent/bin/decay < madevent/bin/decay_$i\.in
     done
fi
#__________________________________________
# REPLACE process
# REPLACE will replace el with el/mu/taus by default, if you need something else you need to edit the replace_card1.dat

cat > replace_card1.dat <<EOF
# Enter here any particles you want replaced in the event file after ME run
# In the syntax PID : PID1 PID2 PID3 ...
# End with "done" or <newline>
11:11 13 15
-12: -12 -14 -16
-11:-11 -13 -15
12: 12 14 16
done
EOF

if [ ${replace} == true ] ; then
    echo "%MSG-MG5 Runnig REPLACE..."
	if [ -f ${file}.lhe ] ; then
		mv ${file}.lhe ${file}_in.lhe
	fi
	perl madevent/bin/replace.pl ${file}_in.lhe ${file}.lhe < replace_card1.dat
fi	

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

mv ${file}_qcut${qcut}_mgPostv2.lhe ${file}_final.lhe 


ls -l
echo

exit 0
