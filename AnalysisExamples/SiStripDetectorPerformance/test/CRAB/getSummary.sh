#!/bin/sh


function orderTableXRun(){

    echo "<TABLE  BORDER=1 ALIGN=CENTER> " > ${sortedfiletmp}_jobtable
    echo -e "<TD><a href=$htmlwebadd>Job</a> $tableSeparator Ncreated $tableSeparator Nsubmitted $tableSeparator Ncleared $tableSeparator EXIT CODE 0 $tableSeparator <a href=$sortedwebadd> Nevents </a><TR> " >> ${sortedfiletmp}_jobtable
    cat ${webfiletmp}_jobtable | grep font | sort -n -r -t ">" -k 18 >> ${sortedfiletmp}_jobtable 
    echo "</TABLE> " >> ${sortedfiletmp}_jobtable
}

function CreateJobTable(){
    echo "<TABLE  BORDER=1 ALIGN=CENTER> " > ${webfiletmp}_jobtable
    echo -e "<TD><a href=$htmlwebadd>Job</a> $tableSeparator Ncreated $tableSeparator Nsubmitted $tableSeparator Ncleared $tableSeparator EXIT CODE 0 $tableSeparator <a href=$sortedwebadd> Nevents </a><TR> " >> ${webfiletmp}_jobtable 

    for dir in `ls $path`;
      do

      [ ! -d $path/$dir ] || [ "$dir" == "AllSummaries" ] && continue 

      let NJobs++

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

      [ $Ncreated -gt 0 ] && let NJobs_cre++

      
      [ -e $path/$dir/logs/submission_log.txt ] &&
      Nsubmitted=`grep submitted $path/$dir/logs/submission_log.txt | awk '{if (NR==1) print $4}'`
      [ -e $path/$dir/submission_log.txt ] &&
      Nsubmitted=`grep submitted $path/$dir/submission_log.txt | awk '{if (NR==1) print $4}'`
      [ "$Nsubmitted" == "" ] && Nsubmitted=0

      [ $Nsubmitted -gt 0 ] && let NJobs_sub++      

      
      Ndone=0
      [ -e $path/$dir/logs/status_$dir.txt ] &&
      Ndone=`grep "EXE_EXIT_CODE: 0" $path/$dir/logs/status_$dir.txt | awk '{if (NR==1) print $1}'`
      [ -e $path/$dir/status_$dir.txt ] &&
      Ndone=`grep "EXE_EXIT_CODE: 0" $path/$dir/status_$dir.txt | awk '{if (NR==1) print $1}'`
      [ "$Ndone" == "" ] && Ndone=0

      [ $Ndone -gt 0 ] && let NJobs_good++


      Ncleared=0
      [ -e $path/$dir/logs/status_$dir.txt ] &&
      Ncleared=`grep "Jobs cleared" $path/$dir/logs/status_$dir.txt | awk '{if (NR==1) print $2}'`
      [ -e $path/$dir/status_$dir.txt ] &&
      Ncleared=`grep "Jobs cleared" $path/$dir/status_$dir.txt | awk '{if (NR==1) print $2}'`
      [ "$Ncleared" == "" ] && Ncleared=0

      [ $Ncleared -gt 0 ] && let NJobs_cle++


      color=$blackColor
      
      [ $Ndone -eq $Nsubmitted ] && color=$greenColor
      [ $Ncleared -ne 0 ] && [ $Ndone -ne $Ncleared ] && color=$redColor #&& echo -e "Job $dir\n\tNcleared=$Ncleared\t Ndone=$Ndone\n$webpath/$dir" | mail -s "CRAB Job Problem" domenico.giordano@cern.ch

      [ $Ncreated -eq 0 ] && color=$redColor

      Separator=$tableSeparator
      [ $Ncreated -ne $Nsubmitted ] && Separator=$tableHSeparator && color=$redColor


      echo -e "<TD><a href=$webpath/$dir> $dir </a>  $Separator  $fontColor $Ncreated </font>  $Separator $fontColor $Nsubmitted </font>  $Separator $fontColor  $Ncleared </font> $Separator $fontColor $Ndone </font> $tableSeparator $fontColor $Nevents </font> <TR> " | sed -e "s@hex@$color@g">> ${webfiletmp}_jobtable 
      
    done
    
    echo "</TABLE> " >> ${webfiletmp}_jobtable

    orderTableXRun 
}

function CreateSummaries(){


   echo "<TABLE  BORDER=1 ALIGN=CENTER> " > ${webfiletmp}_jobtable
    echo -e "<TD><a href=$htmlwebadd>Job</a> $tableSeparator Ncreated $tableSeparator Nsubmitted $tableSeparator Ncleared $tableSeparator EXIT CODE 0 $tableSeparator <a href=$sortedwebadd> Nevents </a><TR> " >> ${webfiletmp}_jobtable 

    for file in `ls $path/AllSummaries 2>/dev/null`;
      do
      [ -d $path/AllSummaries/$file ] && continue 

      if [ `echo $file | grep -c .txt` != 0 ]; then
	  echo "<hr><br><br>""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href=$webpath/AllSummaries/$file>$file</a>&nbsp;&nbsp;<a href="`echo $webpath/AllSummaries/$file | sed -e 's@.txt@.html@'`">html</a>""<br><br>" >> ${webfiletmp}_smry

	  name=`echo $file | sed -e "s@.txt@*@"`
	  for giffile in `ls $path/AllSummaries/$name 2>/dev/null | awk -F"/" '{print $NF}'`;
	    do 
	    [ `echo $giffile | grep -c .gif` != 0 ] && echo "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href=$webpath/AllSummaries/$giffile>$giffile</a>&nbsp;&nbsp;" >> ${webfiletmp}_smry #&& echo "<br>" >> ${webfiletmp}_smry 
	    done
      fi

    done
}

function CreateHtml(){
    path=$1
    webadd="http://cmstac11.cern.ch:8080/analysis/"

    mkdir -p /data1/CrabAnalysis/Summaries/$3

    webfile=/data1/CrabAnalysis/Summaries/$3/$2
    htmlwebadd=`echo $webfile | sed -e "s@/data1/@$webadd@"`

    sortedfile=`echo $webfile | sed -e "s#.html#_sorted.html#"`
    sortedwebadd=`echo $sortedfile | sed -e "s@/data1/@$webadd@"`

    webpath=`echo $1 | sed -e "s@/data1/@$webadd@"`

    webfiletmp=/tmp/cmstac/$2
    sortedfiletmp=`echo $webfiletmp | sed -e "s#.html#_sorted.html#"`


    tableSeparator="<TD align=center>"
    tableHSeparator="<TH align=center>"
    fontColor="<font color=\"hex\">"
    greenColor="#00FF00"
    redColor="#FF0000"
    blackColor="#000000"

    rm -f $webfile
    rm -f $sortedfile
    rm -f ${webfiletmp}_*
    rm -f ${sortedfiletmp}_*
 
 
    #Header
    echo "<html><head><title>Summary Page</title></head>" | tee $sortedfile > $webfile

    #CRAB
    echo "<h3>CRAB Job List</h3>&nbsp;&nbsp;&nbsp; Report of monitored quantities related to the CRAB jobs, created and submitted for each scheduled run<br><br>&nbsp;&nbsp;&nbsp; The Job Name provides a link to the web page where logs and results for the given job are collected<br>" | tee -a $sortedfile >> $webfile

    echo "<h4>&nbsp;&nbsp;&nbsp;Statistics</h4>" | tee -a $sortedfile >> $webfile

    #CRAB JOB TABLE
 
    ###Summary report
    export NJobs=0
    export NJobs_cre=0
    export NJobs_sub=0
    export NJobs_cle=0
    export NJobs_good=0

    CreateJobTable

    echo "<br><TABLE  BORDER=1 ALIGN=CENTER> " | tee -a $sortedfile >> $webfile
    echo -e "$tableSeparator Total Num of Identified Runs $tableSeparator  Runs with created jobs $tableSeparator  Runs with submitted jobs $tableSeparator  Runs with cleared jobs $tableSeparator  Runs with exit status 0 jobs <TR> " | tee -a $sortedfile >> $webfile
    echo -e "$tableSeparator $NJobs $tableSeparator $NJobs_cre $tableSeparator $NJobs_sub $tableSeparator $NJobs_cle $tableSeparator $NJobs_good <TR> "        | tee -a $sortedfile >> $webfile
    echo "</TABLE><br><br><br> " | tee -a $sortedfile >> $webfile

    cat ${webfiletmp}_jobtable >> $webfile
    cat ${sortedfiletmp}_jobtable >>$sortedfile


    #Summaries
    echo "<body><h3>Summary Results List</h3><br>Links to summary pages, related to specific analyses (Quality Tests) performed on the CRAB job's outputs (root histograms)<br><br>" | tee -a $sortedfile >> $webfile
    
    CreateSummaries
    [ -e ${webfiletmp}_smry ] && cat ${webfiletmp}_smry | tee -a $sortedfile >> $webfile


    #END
    echo "<br><br>Webmaster: domenico.giordano@cern.ch"| tee -a $sortedfile >> $webfile
    echo "</body></html>" | tee -a $sortedfile >> $webfile


    rm -f ${webfiletmp}_*
    rm -f ${sortedfiletmp}_*
}


############
##  MAIN  ##
############

Version=""
[ "$1" != "" ] && Version=$1
 
Tpath=/data1/CrabAnalysis/ClusterAnalysis
for path in `ls $Tpath`
  do
  [ "$Version" != "" ] && [ "$Version" != "$path" ] && continue
  echo CreateHtml $Tpath/$path ClusterAnalysis_$path.html $path  
  CreateHtml $Tpath/$path ClusterAnalysis_$path.html $path
done

Tpath=/data1/CrabAnalysis/OLDClusterAnalysis
for path in `ls $Tpath`
  do
  [ "$Version" != "" ] && [ "$Version" != "$path" ] && continue
  echo CreateHtml $Tpath/$path ClusterAnalysis_$path.html $path
  CreateHtml $Tpath/$path ClusterAnalysis_$path.html $path
done

Tpath=/data1/CrabAnalysis/TIFNtupleMaker
for path in `ls $Tpath`
  do
  [ "$Version" != "" ] && [ "$Version" != "$path" ] && continue
  echo CreateHtml $Tpath/$path TIFNtupleMaker_$path.html $path
  CreateHtml $Tpath/$path TIFNtupleMaker_$path.html $path
done

Tpath=/data1/CrabAnalysis/TIFNtupleMakerZS
for path in `ls $Tpath`
  do
  [ "$Version" != "" ] && [ "$Version" != "$path" ] && continue
  echo CreateHtml $Tpath/$path TIFNtupleMakerZS_$path.html $path
  CreateHtml $Tpath/$path TIFNtupleMakerZS_$path.html $path
done


