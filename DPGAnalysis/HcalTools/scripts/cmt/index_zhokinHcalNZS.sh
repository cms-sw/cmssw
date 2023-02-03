#!/bin/bash

WebDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring'
WebSite='https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring'
HistoDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos'


cmsenv 2>/dev/null
if [ $? == 0 ] ; then
    eval `scramv1 runtime -sh`
fi

#echo "0"

# Process arguments and set the flags
fileName=$1

# Obtain the runList from a file, if needed
runList=""
if [ ${#fileName} -gt 0 ] ; then
  if [ -s ${fileName} ] ; then
      runList=`cat ${fileName}`
  else
      echo "<${fileName}> does not seem to be a valid file"
      exit 2
  fi
else
    if [ ${ignoreFile} -eq 0 ] ; then
	echo " ! no file provided"
    fi
    echo " ! will produce only the global html page"
fi


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

echo "Tested `wc -w <<< "${runList}"` runs from file ${fileName}"
if [ ${ok} -eq 0 ] ; then
    echo "errors in the file ${fileName} with run numbers"
    exit 3
else
    if [ ${#fileName} -gt 0 ] ; then
	echo "run numbers in ${fileName} verified ok"
    fi
fi

echo 
echo 
echo 'Numbers of NEW runs for processing'
echo "${runList}"
echo -e "runList complete\n"

echo -e '\n\nRun numbers:'
runListEOS=`echo $runList | tee _runlist_`
echo "${runListEOS}"
echo -e "Full runList for EOS complete\n"


# copy index.html from EOS:
echo 'next message is Fine: '
rm index.html
eoscp $WebDir/CMT/index.html index.html
cp index.html OLDindex.html

# delete last line of copied index_draft.html which close the table
cat index.html | head -n -1 > index_draft.html

#extract run numbers for correct continuation
#k=0
#for i in ${runListEOSall} ; do
#let "k = k + 1"
#done

k=643


########################################## type by hands number of new runs k=k-number:
#let "k = k - 1"
echo ' ================>>>    k in old list = '$k

for i in ${runListEOS} ; do
 
#runnumber=$(echo $i | sed -e 's/[^0-9]*//g')
#runnumber=$(echo $i | awk -F 'run' '{print $2}'| awk -F '.' '{print $1}')
runnumber=${i}
#if [[ "$runnumber" > 243400 ]] ; then
let "k = k + 1"
echo
echo ' ================>>> new k in loop = '$k
echo
echo
echo 'RUN number = '$runnumber

# extract the date of file
dasInfo=${DAS_DIR}/das_${runnumber}.txt
got=0
#echo "1"
if [[ ${dasCache} == "1" ]] ; then
    rm -f tmp
    if [ -s ${dasInfo} ] ; then
	cp ${dasInfo} tmp
	got=1
    else
	echo "no ${dasInfo} found. Will use dasgoclient"
    fi
fi
#echo "2"

if [ ${got} -eq 0 ] ; then
#echo "3"
#echo "runnumber:"
#dasgoclient  --query="file dataset=/HcalNZS/Run2018D-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
#dasgoclient  --query="file dataset=/Cosmics/Commissioning2021-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp
#dasgoclient  --query="file dataset=/HcalNZS/Commissioning2021-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp

dasgoclient  --query="file dataset=/HcalNZS/Run2022G-v1/RAW  run=${i} | grep file.size, file.nevents, file.modification_time "  > tmp


#echo "${runnumber}"
fi

timetmp=`cat tmp | head -n 1  | awk '{print $3}'`
############################################################################################################ printout:
#type='Cosmics'
type='HcalNZS'
#type='ZeroBias0'
timetmp2=`date -d @${timetmp} +%Y-%m-%d:%H-%M-%S`
sizetmp=`cat tmp | head -n 1  | awk '{print $1}'`
neventstmp=`cat tmp | head -n 1  | awk '{print $2}'`
#commentariy='CRUZET2021'
#commentariy='CRAFT2021'
#commentariy='Commissioning2021'
commentariy='Run3 2022G-v1'
#cat runs_info
echo 'RUN Type = '$type
echo ${sizetmp} ${neventstmp} ${timetmp2}
echo 'RUN Comment = '$commentariy

#echo "4"


#adding entry to list of file index_draft.html
let "raw = (k % 2) + 2"
echo '<tr>'>> index_draft.html
echo '<td class="s1" align="center">'$k'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runnumber'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$type'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$timetmp2'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$sizetmp'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$neventstmp'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center"><a href="'$WebSite'/CMT/GLOBAL_'$runnumber'/LumiList.html">CMT_'$runnumber'</a></td>'>> index_draft.html
echo '<td class="s'$raw'" align="center"><a href="'$WebSite'/GlobalRMT/GLOBAL_'$runnumber'/MAP.html">RMT_'$runnumber'</a></td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$commentariy'</td>'>> index_draft.html


rm tmp
#echo "5"

############################################################################################################   record index_draft.html

if [ ${#comment} -gt 0 ] ; then
    #echo "runList=${runList}, check ${runnumber}"
    temp_var=${runList/${runnumber}/}
    if [ ${#temp_var} -lt ${#runList} ] ; then
	echo "adding a commentary for this run"
	echo "<td class=\"s${raw}\" align=\"center\">${comment}</td>" >> index_draft.html
    fi
fi
echo '</tr>'>> index_draft.html
prev=$i

#fi
done
#echo "6"


# print footer to index.html 
echo `cat footer.txt`>> index_draft.html


status=0
if [[ ${debug} == "1" ]] ; then
    echo "debug=${debug}. No upload to eos"
    status=-1
else
###    echo "Commented by me:  eoscp index_draft.html $WebDir/CMT/index.html No upload to eos"
#  eoscp OLDindex.html $WebDir/CMT/OLDindex.html
#  eoscp index_draft.html $WebDir/CMT/index.html
    status="$?"
# rm index_draft.html
fi
#echo "7"

# delete temp files

if [[ ${debug} == "0" ]] ; then
#    rm -f *.root
    rm -f _runlist_
    rm -f _runlistEOSall_
fi
#echo "8"

# check eos-upload exit code
if [[ "${status}" == "0" ]]; then
  echo "Successfully uploaded!"
else
  echo "ERROR: Auto-uploading failed: do it by hands !!!"
  exit 1
fi

echo "index script done"
