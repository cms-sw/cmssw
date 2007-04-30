#!/bin/sh


function orderTableXRun(){

    echo "<TABLE  BORDER=1 ALIGN=CENTER> " > $sortedfile
    echo -e "<TD><a href=$htmlwebadd>Job</a> $tableSeparator Ncreated $tableSeparator Nsubmitted $tableSeparator Ncleared $tableSeparator EXIT CODE 0 $tableSeparator <a href=$sortedwebadd> Nevents </a><TR> " >> $sortedfile
#    echo -e "<TD>Job $tableSeparator Ncreated $tableSeparator Nsubmitted $tableSeparator Ncleared $tableSeparator EXIT CODE 0 $tableSeparator Nevents<TR> " >> $sortedfile
    cat $webfile | grep font | sort -n -r -t ">" -k 18 >> $sortedfile 
    echo "</TABLE> " >> $sortedfile
}

function CreateHtml(){
    path=$1
    webadd="http://cmstac11.cern.ch:8080/analysis/"

    mkdir -p /data1/CrabAnalysis/Summaries/$3

    export webfile=/data1/CrabAnalysis/Summaries/$3/$2

    export htmlwebadd=`echo $webfile | sed -e "s@/data1/@$webadd@"`

    export sortedfile=`echo $webfile | sed -e "s#.html#_sorted.html#"`
    export sortedwebadd=`echo $sortedfile | sed -e "s@/data1/@$webadd@"`

    webpath=`echo $1 | sed -e "s@/data1/@$webadd@"`

    tableSeparator="<TD align=center>"
    tableHSeparator="<TH align=center>"
    fontColor="<font color=\"hex\">"
    greenColor="#00FF00"
    redColor="#FF0000"
    blackColor="#000000"

    rm -f $webfile
    rm -f $sortedfile
    echo "<TABLE  BORDER=1 ALIGN=CENTER> " > $webfile
    echo -e "<TD><a href=$htmlwebadd>Job</a> $tableSeparator Ncreated $tableSeparator Nsubmitted $tableSeparator Ncleared $tableSeparator EXIT CODE 0 $tableSeparator <a href=$sortedwebadd> Nevents </a><TR> " >> $webfile 

    for dir in `ls $path`;
      do

      Nevents=0
      [ -e $path/$dir/logs/create_log.txt ] &&
      Nevents=`grep "number of available events" $path/$dir/logs/create_log.txt | awk '{print $8}'`
      [ -e $path/$dir/create_log.txt ] &&
      Nevents=`grep "number of available events" $path/$dir/create_log.txt | awk '{print $8}'`
      [ "$Nevents" == "" ] && Nevents=0
      
      [ -e $path/$dir/logs/create_log.txt ] &&
      Ncreated=`grep created $path/$dir/logs/create_log.txt | awk '{if(NR==1) print $4}'`
      [ -e $path/$dir/create_log.txt ] &&
      Ncreated=`grep created $path/$dir/create_log.txt | awk '{if(NR==1) print $4}'`
      [ "$Ncreated" == "" ] && Ncreated=0
      
      [ -e $path/$dir/logs/submission_log.txt ] &&
      Nsubmitted=`grep submitted $path/$dir/logs/submission_log.txt | awk '{if (NR==1) print $4}'`
      [ -e $path/$dir/submission_log.txt ] &&
      Nsubmitted=`grep submitted $path/$dir/submission_log.txt | awk '{if (NR==1) print $4}'`
      [ "$Nsubmitted" == "" ] && Nsubmitted=0
      
      Ndone=0
      [ -e $path/$dir/logs/status_$dir.txt ] &&
      Ndone=`grep "EXE_EXIT_CODE: 0" $path/$dir/logs/status_$dir.txt | awk '{if (NR==1) print $1}'`
      [ -e $path/$dir/status_$dir.txt ] &&
      Ndone=`grep "EXE_EXIT_CODE: 0" $path/$dir/status_$dir.txt | awk '{if (NR==1) print $1}'`
      [ "$Ndone" == "" ] && Ndone=0

      Ncleared=0
      [ -e $path/$dir/logs/status_$dir.txt ] &&
      Ncleared=`grep "Jobs cleared" $path/$dir/logs/status_$dir.txt | awk '{if (NR==1) print $2}'`
      [ -e $path/$dir/status_$dir.txt ] &&
      Ncleared=`grep "Jobs cleared" $path/$dir/status_$dir.txt | awk '{if (NR==1) print $2}'`
      [ "$Ncleared" == "" ] && Ncleared=0

      color=$blackColor
      
      [ $Ndone -eq $Nsubmitted ] && color=$greenColor
      [ $Ncleared -ne 0 ] && [ $Ndone -ne $Ncleared ] && color=$redColor #&& echo -e "Job $dir\n\tNcleared=$Ncleared\t Ndone=$Ndone\n$webpath/$dir" | mail -s "CRAB Job Problem" domenico.giordano@cern.ch

      [ $Ncreated -eq 0 ] && color=$redColor

      Separator=$tableSeparator
      [ $Ncreated -ne $Nsubmitted ] && Separator=$tableHSeparator && color=$redColor

      [ ! -d $path/$dir ] && Ncleared=0 && Ncreated=0 && Nsubmitted=0 && Ndone=0 && color=$blackColor

      echo -e "<TD><a href=$webpath/$dir> $dir </a>  $Separator  $fontColor $Ncreated </font>  $Separator $fontColor $Nsubmitted </font>  $Separator $fontColor  $Ncleared </font> $Separator $fontColor $Ndone </font> $tableSeparator $fontColor $Nevents </font> <TR> " | sed -e "s@hex@$color@g">> $webfile 
      
    done
    
    echo "</TABLE> " >> $webfile

    orderTableXRun 
}


############
##  MAIN  ##
############

Tpath=/data1/CrabAnalysis/ClusterAnalysis
for path in `ls $Tpath`
  do
  CreateHtml $Tpath/$path ClusterAnalysis_$path.html $path
done

Tpath=/data1/CrabAnalysis/OLDClusterAnalysis
for path in `ls $Tpath`
  do
  CreateHtml $Tpath/$path ClusterAnalysis_$path.html $path
done

Tpath=/data1/CrabAnalysis/TIFNtupleMaker
for path in `ls $Tpath`
  do
  CreateHtml $Tpath/$path TIFNtupleMaker_$path.html $path
done

Tpath=/data1/CrabAnalysis/TIFNtupleMakerZS
for path in `ls $Tpath`
  do
  CreateHtml $Tpath/$path TIFNtupleMakerZS_$path.html $path
done


