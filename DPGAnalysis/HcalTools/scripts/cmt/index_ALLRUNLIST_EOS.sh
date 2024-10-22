#!/bin/bash

WebDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring'
WebSite='https://cms-conddb-dev.cern.ch/eosweb/hcal/HcalRemoteMonitoring'
HistoDir='/eos/cms/store/group/dpg_hcal/comm_hcal/www/HcalRemoteMonitoring/CMT/histos'
eos='/afs/cern.ch/project/eos/installation/0.3.15/bin/eos.select'

# print usage info
if [[ "$1" == "" ]]; then
  echo "Usage:"
  echo "  $0 file [comment] [-ignore-file] [-das-cache]"
  echo "    file  -- a file with run numbers"
  echo "    comment  -- add a comment line (instead of spaces use '_')"
  echo "    -ignore-file   -- skips production of run html pages. Produces"
  echo "                      only the global page. File name is not needed."
  echo "    -das-cache   -- whether to save DAS information locally for a reuse"
  echo
  echo "example: ./GLOBAL.sh Run_List.txt"
  exit 1
fi

cmsenv 2>/dev/null
if [ $? == 0 ] ; then
    eval `scramv1 runtime -sh`
fi
temp_var=`ls ${eos}`
status="$?"
echo "using eos command <${temp_var}>"
if [ ! ${status} -eq 0 ] ; then
    echo "failed to find eos command"
    # exit 1
fi


# Process arguments and set the flags
fileName=$1
comment=$2
if [ ${#comment} -gt 0 ] && [ "${comment:0:1}" == "-" ] ; then comment=""; fi
ignoreFile=0
debug=0
dasCache=0
DAS_DIR="d-DAS-info"

for a in $@ ; do
    if [ "$a" == "-ignore-file" ] ; then
	echo " ** file will be ignored"
	fileName=""
	ignoreFile=1
    elif [ "$a" == "-das-cache" ] ; then
	echo " ** DAS cache ${DAS_DIR} enabled"
	dasCache=1
	if [ ! -d ${DAS_DIR} ] ; then mkdir ${DAS_DIR}; fi
    else
	temp_var=${a/-debug/}
	if [ ${#a} -gt ${#temp_var} ] ; then
	    debug=${temp_var}
	    echo " ** debug detected (debug=${debug})"
	fi
    fi
done

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

comment=`echo ${comment} | sed sk\_k\ kg`
if [ ${#comment} -gt 0 ] ; then
    echo "comment \"${comment}\" will be added to the pages"
fi

if [ ${debug} -eq 3 ] ; then exit; fi


echo 
echo 
echo 
echo 'Numbers of NEW runs for processing'
echo "${runList}"
echo -e "runList complete\n"

#processing skipped


# #  #  # # # # # # # # # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # ### # # #####
# Create global web page
#

echo "Get list of runss in ${HistoDir}"
#eos ls $HistoDir | grep root | awk  '{print $5}' | awk -F / '{print $10}' > rtmp
#cat rtmp | awk -F _ '{print $2}' | awk -F . '{print $1}' > _runlist_

histoFiles=`${eos} ls $HistoDir | grep root | awk -F '_' '{print $2}' | awk -F '.' '{print $1}'`
echo -e '\n\nRun numbers on EOS:'
runListEOSall=`echo $histoFiles | tee _runlistEOSall_`
echo "Got `wc -w <<< "${runListEOSall}"` runs from HistDir ${HistoDir}"

echo -e '\n\nRun numbers:'
runListEOS=`echo $runList | tee _runlist_`
echo "${runListEOS}"
echo -e "Full runList for EOS complete\n"





### use to clarify full list of runs:
echo "Get OLD list of runs in ${HistoDir}"
echo -e '\n\nRun numbers to be on EOS:'
runListEOS=`echo $histoFiles `
echo "${runListEOS}"
echo -e "OLD list complete\n"


#############                 making table

# skip/comment header:
## print header to index.html 
#if [ ${#comment} -eq 0 ] ; then
#    echo `cat header_GLOBAL_EOS.txt` > index_draft.html
#else
#    echo `head -n -1 header_GLOBAL_EOS.txt` > index_draft.html
#    echo -e "<td class=\"s1\" align=\"center\">Comment</td>\n</tr>\n" \
#	>> index_draft.html
#fi

# copy index.html from EOS:
echo 'next message is Fine: '
rm index.html
#cmsStage $WebDir/CMT/index.html index.html
eoscp $WebDir/CMT/index.html index.html
cp index.html OLDindex.html

# delete last line of copied index_draft.html which close the table
cat index.html | head -n -1 > index_draft.html

#extract run numbers for correct continuation
#k=0
#for i in ${runListEOSall} ; do
#let "k = k + 1"
#done
k=1429
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
if [ ${dasCache} -eq 1 ] ; then
    rm -f tmp
    if [ -s ${dasInfo} ] ; then
	cp ${dasInfo} tmp
	got=1
    else
	echo "no ${dasInfo} found. Will use dasgoclient"
    fi
fi

if [ ${got} -eq 0 ] ; then
#####  old:                                       1            2          3           4            5               6                 7 8            
##   ./das_client.py --query="run=${i} | grep run.beam_e,run.bfield,run.nlumis,run.lhcFill,run.delivered_lumi,run.duration,run.start_time,run.end_time" --limit=0 > tmp
###### AZ  used now:                                   1            2          3           4            5               6                 7 8 9      10 11 12  
dasgoclient  --query="run=${i} | grep run.lhcEnergy ,run.bfield ,run.lsCount ,run.lhcFill ,  run.lsRanges ,    run.runClassName ,run.startTime ,run.stopTime "  > tmp
# for test:
#dasgoclient  --query="run=${i} | grep run.aaa ,run.lhcEnergy ,run.bfield ,run.lsCount,run.lhcFill ,  run.lsRanges ,    run.runClassName ,run.startTime ,run.stopTime , run.runCreated ,run.modified " > tmp

#    if [ ${dasCache} -eq 1 ] ; then cp tmp ${dasInfo}; fi
fi


##### AZ used now:                          
# delete last lines of copied tmp file
cat tmp | head -n -1 > ztmp
rm tmp
#cat ztmp 

###### AZ  used now:                          
date=`cat ztmp | awk '{print $7" "$8" "$9}'`
date_end=`cat ztmp | awk '{print $10" "$11" "$12}'`
#old:
#date=`cat ztmp | awk '{print $7" "$8}'`
#date_end=`cat ztmp | awk '{print $9" "$10}'`
E=`cat ztmp | awk '{print $1}'`
B=`cat ztmp | awk '{print $2}'`
nL=`cat ztmp | awk '{print $3}'`
Fill=`cat ztmp | awk '{print $4}'`
dLumi=`cat ztmp | awk '{print $5}'`
D=`cat ztmp | awk '{print $6}'`

###### AZ  used now:                          
rm ztmp

#echo 'ver 1'
#${eos} ls $HistoDir/Global_$i.root
#echo 'ver 2'
#${eos} ls -l $HistoDir/Global_$i.root
#old Date_obr=`${eos} ls -l $HistoDir/Global_$i.root | awk '{print $3" "$4}'`

fileinfo=`${eos} ls -l $HistoDir/Global_$i.root`
Date_obr=`echo ${fileinfo} | awk '{print $6" "$7" "$8}'`
#echo "Date_obr=$Date_obr"

# extract run type, data, time and number of events
type='Cosmic'
commentariy=''
#cat runs_info

#  for j in $(cat runs_info); do
#    echo $j
#    k= `echo $j | awk  '{print $1}'`
#    if [[ "$runnumber" == "$k" ]] ; then
#      type= `echo $i | awk  '{print $2}'`
#      commentariy=`echo $i | awk  '{print $3}'`
#    fi
#  done

#echo 'RUN Type = '$type
echo 'RUN Start Date = '$date
echo 'RUN Duration = '$D
echo 'RUN End Date = '$date_end
echo 'RUN Energy = '$E
echo 'RUN Magnet field = '$B
echo 'RUN LS number = '$nL
echo 'RUN LHC Fill = '$Fill
echo 'RUN Delivered Luminosity = '$dLumi
echo 'RUN Date processing = '$Date_obr
#echo 'RUN Comment = '$commentariy

#adding entry to list of file index_draft.html
let "raw = (k % 2) + 2"
echo '<tr>'>> index_draft.html
echo '<td class="s1" align="center">'$k'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$runnumber'</td>'>> index_draft.html
#echo '<td class="s'$raw'" align="center">'$type'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$nL'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$Fill'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$date'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$D'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$date_end'</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center"><a href="'$WebSite'/CMT/GLOBAL_'$runnumber'/LumiList.html">CMT_'$runnumber'</a></td>'>> index_draft.html
echo '<td class="s'$raw'" align="center"><a href="'$WebSite'/GlobalRMT/GLOBAL_'$runnumber'/MAP.html">RMT_'$runnumber'</a></td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$B' T</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$E' GeV</td>'>> index_draft.html
#echo '<td class="s'$raw'" align="center">'$dLumi' /nb</td>'>> index_draft.html
echo '<td class="s'$raw'" align="center">'$Date_obr'</td>'>> index_draft.html
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


# print footer to index.html 
echo `cat footer.txt`>> index_draft.html


status=0
if [ ${debug} -gt 0 ] ; then
    echo "debug=${debug}. No upload to eos"
    status=-1
else
###    echo "Commented by me:  eoscp index_draft.html $WebDir/CMT/index.html No upload to eos"
#  eoscp OLDindex.html $WebDir/CMT/OLDindex.html
#  eoscp index_draft.html $WebDir/CMT/index.html


    status="$?"
# rm index_draft.html
fi

# delete temp files

if [ ${debug} -eq 0 ] ; then
#    rm -f *.root
    rm -f _runlist_
    rm -f _runlistEOSall_
fi

# check eos-upload exit code
if [[ "${status}" == "0" ]]; then
  echo "Successfully uploaded!"
else
  echo "ERROR: Uploading failed"
  exit 1
fi

echo "index script done"
