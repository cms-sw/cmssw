#! /bin/bash
source /afs/cern.ch/cms/caf/setup.sh

echo ""
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

##########CASTOR_OUT="/castor/cern.ch/cms/store/caf/user/bonato/Collisions2010/Oct2010/"
CASTOR_OUT="/store/caf/user/bonato/Collisions2010/Run2010B/Sept2010/"
###CASTOR_OUT="/store/caf/user/bonato/Collisions2010/Run2010A-v2/"
MYCMSSW_RELEASE="$CMSSW_BASE"

###prepare the scripts
sed -e "s|<CASTOROUT>|${CASTOR_OUT}|g" < SkimLooper.tpl > SkimLooper.sh
sed -e "s|<CASTOROUT>|${CASTOR_OUT}|g" -e "s|<MYCMSSW>|$MYCMSSW_RELEASE|g"  < PrescaleLooper.tpl > PrescaleLooper.sh
sed -e "s|<MYCMSSW>|$MYCMSSW_RELEASE|g"  < skim_exec.tpl > skim_exec.sh
sed -e  "s|<MYCMSSW>|$MYCMSSW_RELEASE|g" < presc_exec.tpl > presc_exec.sh


#prepare list of files counting the eevents from DBS
for ALCATAG in  $( cat "taglist.txt" )
do
echo ""
echo "Counting events for ${ALCATAG} ; Log in ./log_nevents_${ALCATAG}.out"
#time ./cntevts_in_file.sh  "../data/${ALCATAG}.dat" $ALCATAG &> log_nevents_${ALCATAG}.out
done
########exit 0

#first loop: SKIMMING
echo ""
echo "~~~ Starting SkimLooper at $(date)"
echo
chmod 711 SkimLooper.sh
chmod 711 skim_exec.sh
./SkimLooper.sh "taglist.txt" 3
if [ $? != 0 ]
    then
    exit 2
fi

sleep 1800 #wait for half an hour

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
chmod 711 PrescaleLooper.sh
chmod 711 presc_exec.sh
./PrescaleLooper.sh "taglist.txt" 3
if [ $? != 0 ]
    then
    exit 3
fi
echo
echo
echo "Finished operations at $(date)"
