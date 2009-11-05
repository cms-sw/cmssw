#! /bin/bash

echo "#############################################################################"
echo "#####                                                                   #####"
echo "#####      Starting the TK Alignment Skim&Prescale workflow!!!          #####"
echo "#####                                                                   #####"
echo "#####                                                                   #####"
echo "#####                                                                   #####"
echo "#############################################################################"
echo
echo
echo "Program launched from $(hostname) at $(date) "
echo
echo

CASTOR_OUT="/castor/cern.ch/cms/store/user/bonato/CRAFTReproSkims/Summer09/MinBias/"
MYCMSSW_RELEASE="$CMSSW_BASE"

#prepare the scripts
replace "<CASTOROUT>" $CASTOR_OUT < SkimLooper.tpl > SkimLooper.sh
replace "<CASTOROUT>" $CASTOR_OUT  "<MYCMSSW>" $MYCMSSW_RELEASE < PrescaleLooper.tpl > PrescaleLooper.sh
replace "<MYCMSSW>" $MYCMSSW_RELEASE < skim_exec.tpl > skim_exec.sh
replace "<MYCMSSW>" $MYCMSSW_RELEASE < presc_exec.tpl > presc_exec.sh



#prepare list of files counting the eevents from DBS
for ALCATAG in  $( cat "taglist.txt" )
do
echo ""
echo "Counting events for ${ALCATAG} ; Log in ./log_nevents_${ALCATAG}.out"
time ./cntevts_in_file.sh  "./data/${ALCATAG}.dat" $ALCATAG &> log_nevents_${ALCATAG}.out
done


#first loop: SKIMMING
echo ""
echo "~~~ Starting SkimLooper at $(date)"
echo
./SkimLooper.sh "taglist.txt" 3

sleep 1200 #wait for half an hour

#check that all the files from SkimLooper are done
SKIM_DONE=1
echo ""
echo "Checking the status of SkimLooper: "
while [ $SKIM_DONE == 1 ]
do

node=$(basename $(hostname) .cern.ch)
bjobs -u $USER | grep $USER | grep $node > tmpskimjobs.tmp
if [ $(wc -l tmpskimjobs.tmp | awk '{print $1}') == 0 ]
then
SKIM_DONE=0
else
sleep 600 #wait for ten minutes before checking again
fi

rm -f  tmpskimjobs
echo "Status is ${SKIM_DONE}"
done

#SkimLooper has finished, go to the prescaling phase
#second loop: PRESCALING
echo ""
echo "~~~ Starting PrescaleLooper at $(date)"
echo
./PrescaleLooper.sh "taglist.txt" 3

echo
echo
echo "Finished operations at $(date)"
