#!/bin/bash

rm index_selection.html

echo "11111111111111"
runList=`cat ${1}`
#echo "${runList}"

echo "22222222222"

# Check the runList and correct the correctables
# Replace ',' and ';' by empty spaces
runList=`echo "${runList}" | sed 'sk,k\ kg' | sed 'sk;k\ kg'`
ok=1
for r in ${runList} ; do
    if [ ! ${#r} -eq 6 ] ; then
	echo "run numbers are expected to be of length 6. Check <$r>"
	ok=0
    fi
    debug_loc=0
    if [ "$r" -eq "$r" ] 2>/dev/null ; then
	if [ ${debug_loc} -eq 1 ] ; then echo "run variable <$r> is a number (ok)"; fi
    else
	echo "error: run variable <$r> is not an integer number"
	ok=0
    fi
done
echo "333"

#echo "Tested `wc -w <<< "${runList}"` runs from file ${fileName}"
if [ ${ok} -eq 0 ] ; then
    echo "errors in the file ${fileName} with run numbers"
    exit 3
else
    if [ ${#fileName} -gt 0 ] ; then
	echo "run numbers in ${fileName} verified ok"
    fi
fi
echo "444"
echo "555"

#echo 'Numbers of NEW runs for processing'
#echo "${runList}"
#echo -e "runList complete\n"

#processing skipped
echo "6"


# #  #  # # # # # # # # # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # #####

#echo -e '\n\nRun numbers:'
runListEOS=`echo $runList | tee _runlist_`

echo "7"


#echo "${runListEOS}"
#echo -e "Full runList for EOS complete\n"



echo "8"




#########################################################################################################
for i in ${runListEOS} ; do
 
runnumber=${i}
#if [[ "$runnumber" > 243400 ]] ; then
#dasgoclient --query="file dataset=/HcalNZS/Run2018A-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
#dasgoclient --query="file dataset=/HcalNZS/Run2018B-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
#dasgoclient --query="file dataset=/HcalNZS/Run2018C-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
#
#dasgoclient --query="file dataset=/HcalNZS/Run2018D-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
#
echo "runnumber:"
dasgoclient --query="file dataset=/HcalNZS/Commissioning2021-HcalCalMinBias-PromptReco-v1/ALCARECO  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp

echo "${runnumber}"


#
#dasgoclient --query="file dataset=/HcalNZS/Run2018E-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
#dasgoclient --query="run dataset=/HIHcalNZS/HIRun2018A-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
timetmp=`cat tmp | awk '{print $3}'`
############################################################################################################
type='NZS'
timetmp2=`date -d @${timetmp} +%Y-%m-%d:%H-%M-%S`
sizetmp=`cat tmp | awk '{print $1}'`
neventstmp=`cat tmp | awk '{print $2}'`
commentariy='Com21-ALCARECO'
##cat runs_info
#echo 'RUN Type = '$type
#echo ${sizetmp} ${neventstmp} ${timetmp2}
#echo 'RUN Comment = '$commentariy
#
#
#
#adding entry to list of file index_selection.html
let "raw = (k % 2) + 2"
echo '<tr>'>> index_selection.html
echo '<td class="s1" align="center">'$k'</td>'>> index_selection.html
echo '<td class="s'$raw'" align="center">'$runnumber'</td>'>> index_selection.html
echo '<td class="s'$raw'" align="center">'$type'</td>'>> index_selection.html
echo '<td class="s'$raw'" align="center">'$timetmp2'</td>'>> index_selection.html
echo '<td class="s'$raw'" align="center">'$sizetmp'</td>'>> index_selection.html
echo '<td class="s'$raw'" align="center">'$neventstmp'</td>'>> index_selection.html
echo '<td class="s'$raw'" align="center">'$commentariy'</td>'>> index_selection.html

done

echo "9"

mv index_selection.html index_selectionNZS_ALCARECO.html
#mv index_selection.html index_selectionA.html
#mv index_selection.html index_selectionB.html
#mv index_selection.html index_selectionC.html
#mv index_selection.html index_selectionD.html
#mv index_selection.html index_selectionE.html
#mv index_selection.html index_selectionHI.html
############################################################################################################
rm tmp
rm \_runlist\_


echo " done"
