#!/bin/bash

workdir=`dirname $0`
cd $workdir

subdet=strip
[ "c$1" != "c" ] && subdet=$1

if [ "c${CMSSW_RELEASE_BASE}" == "c" ]; then
    echo -e "\nSetting scramv1 runtime"
    cd /afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_1_6_0
    eval `scramv1 runtime -sh`
    cd -
fi
export TNS_ADMIN=/afs/cern.ch/project/oracle/admin

[ ! -e log ] && mkdir log

dblist=(`echo "select distinct db from DBtags ;" | sqlite3 dbfile.db`)
accountlist=(`echo "select distinct account from DBtags where account like \"%$subdet%\" order by account;" | sqlite3 dbfile.db` )


nc=${#dblist[@]}
nr=${#accountlist[@]}

#echo $nc $nr

ir=0
for account in ${accountlist[@]}
  do
  ic=0
  for db in ${dblist[@]}
    do
    let comp=$ir*$nc+$ic
    value[$comp]=`echo "select count(tag) from DBtags where (db='$db' and account='$account') ;" | sqlite3 dbfile.db`
    
    file=log/TagList_for_${db}_${account}.txt
    echo -e "DB:\t\t$db\nAccount:\t$account\n\ntags\n-----------------------" > $file
    echo "select tag from DBtags where (db='$db' and account='$account') ;" | sqlite3 dbfile.db>> $file
    #echo
    let ic=$ic+1
    done
  let ir=$ir+1
done

#echo ${value[@]}
echo -e "\t\t\t ${dblist[@]}"

export value

ir=0
while [ $ir -lt $nr ];
  do
  let start=$ir*$nc
  let stop=$start+$nc
  echo -e "${accountlist[$ir]} \t\t ${value[@]:$start:$nc}"
  let ir=$ir+1
done


export webadd="http://test"
export htmlpath=`echo $webpath | sed -e "s@/data1@$webadd@"`

webdir=/afs/cern.ch/user/g/giordano/WWW/MonitorCondDBSiStripAccount
webfile=$webdir/table_$subdet.html
webfiletmp=tmptest.html

rm -f ${webfiletmp}*
    

    #Header
echo "<html><head><title>Summary Page $vTag</title></head>" > ${webfiletmp}

#echo "<h2>Summary Page for tag $vTag</h2>&nbsp;&nbsp;&nbsp; list of uploaded IOVs: <a href=$htmlpath/O2ORuns.txt>here </a><br><br>" #>> ${webfiletmp}
#echo "<HR>" #>>  $webfiletmp

echo "<TABLE  BORDER=1 ALIGN=CENTER> " > ${webfiletmp}

echo ${dblist[@]} | awk 'BEGIN{stringa=" <TD align=center>"} function addColumn(value){stringa=sprintf("%s <TD align=center> %s",stringa,value) } {
for(i=1;i<=NF;++i) addColumn($i);} END{print stringa "<TR>"}'  >> ${webfiletmp}


ir=0
while [ $ir -lt $nr ];
  do
  let start=$ir*$nc
  let stop=$start+$nc
  echo -e "${accountlist[$ir]} ${value[@]:$start:$nc} ${dblist[@]}" | awk 'function addColumn(value,filename,account){stringa=sprintf("%s <TD align=center> <a href=log/TagList_for_%s_%s.txt>%s</a>",stringa,filename,account,value) } {stringa=sprintf("<TD align=center>%s",$1); for(i=2;i<=(NF-1)/2+1;++i){j=i+NF/2-.5; addColumn($i,$j,$1);}} END{print stringa "<TR>"}' >> ${webfiletmp}

  let ir=$ir+1
done



#    echo -e "$tableSeparator  $tableSeparator Nsubmitted $tableSeparator Ncleared $tableSeparator EXIT CODE 0 $tableSeparator <a href=$sortedwebadd> Nevents </a><TR> " 

 #   echo -e "<TD><a href=$webpath/$dir> $dir </a>  $Separator  $fontColor $Ncreated </font>  $Separator $fontColor $Nsubmitted </font>  $Separator $fontColor  $Ncleared </font> $Separator $fontColor $Ndone </font> $tableSeparator $fontColor $Nevents </font> <TR> " | sed -e "s@hex@$color@g">> ${webfiletmp}_jobtable 
      
    echo "</TABLE> " >> ${webfiletmp}

cp ${webfiletmp} ${webfile}
cp -r log $webdir/.

