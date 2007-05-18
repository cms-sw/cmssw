#!/bin/sh

function CreateHtml(){

    path=$1
    webadd="http://cmstac11.cern.ch:8080/analysis/"

    export webfile=$path/`echo $2 | sed -e "s@.txt@.html@g"`
    export htmlwebadd=`echo $webfile | sed -e "s@/data1/@$webadd@"`

    tableSeparator="<TD align=center>"

    rm -f $webfile
    echo "<TABLE  BORDER=1 ALIGN=CENTER> " > $webfile  

    cat $path/$2 | grep "|" | sed -e "s@[ \t]*@@g" | awk -F"|" '
BEGIN{
 Separator[0]="<TD align=center>"; 
 Separator[1]="</font><TD align=center>"; 
 fontColor[0]="<font color=\"#FF0000\">";
 fontColor[1]="<font color=\"#00000\">";
 ci=0;
 N=0;
}
{
  if (NR==1){
    label="";
    for (i=1;i<=NF;i++){
      label=sprintf("%s %s %s",label,Separator[0],$i);
    }      
    print label " <TR> ";
  }
  if (NR!=1){
       if (int($1)!=N) {N=int($1); if (ci==0){ci=1}else{ci=0} }
       pswebadd=""; label="<TD>"; labelA="??? INSERT INTO Summary values (";
       for (i=1;i<=NF;i++){
         val=$i;
         if (i>3){
           if (length($i)==0){val="-";}
           else{ val="<a href=pswebadd> X </a>"; 
             pswebadd=sprintf("%s | %s/../ClusterAnalysis_*%d/res/%s/*%s*.gif",pswebadd,path,$2,$i,$3);
           }
         }
         label=sprintf("%s %s %s %s",label,fontColor[ci],val,Separator[1]);
         if (i>1)
         labelA=sprintf("%s ,\"%s\"",labelA,val);
        } 
       print label " <TR> " pswebadd;
       print labelA "); " pswebadd;
  }
}' path=$path | while read line; do
	aline=`echo "$line" | grep -v "???"`
	[ "$aline" != "" ] && echo  `echo $aline | awk -F"|" '{i=2;while(match($1,"pswebadd")){sub("pswebadd",$i,$1);i++};gsub("/data1/",webadd,$1);print $1}' webadd="$webadd"` >> $webfile

	aline=`echo "$line" | grep "???"`
	[ "$aline" != "" ] && echo  `echo $aline | awk -F"|" '{i=2;while(match($1,"pswebadd")){sub("<a href=pswebadd> X </a>",$i,$1);i++};gsub("/data1/",webadd,$1);print $1}' webadd="$webadd"` | sed -e "s@???@@g" -e "s@,@@" | sqlite3 /tmp/cmstac/prova_${Version}.db
    done
    echo "</TABLE> " >> $webfile
}

############
## MAIN  ###
############

export Version=""
[ "$1" != "" ] && Version=$1

export outFile
basePath=/analysis/sw/CRAB
Tpath=/data1/CrabAnalysis/ClusterAnalysis

macroPath=${basePath}/macros

cd ${basePath}/CMSSW/CMSSW_1_3_0/src
eval `scramv1 runtime -sh`
cd -

for path in `ls $Tpath`
  do
  [ "$Version" != "" ] && [ "$Version" != "$path" ] && continue
  #[ "$path" != "FNAL_pre6_v17" ] && [ "$path" != "FNAL_pre6_v17" ]&& continue  
  echo "...Running on $Tpath/$path"
  workdir=$Tpath/$path/AllSummaries

  [ ! -e $workdir ] && continue 
  cd $workdir

  rm tmp
  rm -vf SummaryTable.*
  count=0
  TotFlag="ID \t|\t Run \t|\t Module "
  for file in `ls Asummary_*.txt | grep -v "DBBadStrips"`
#  for file in `ls Asummary_*.txt `
    do
    let count++
    flag=`echo $file | sed -e "s@Asummary_@@g" -e "s@.txt@@g"`
    echo "-------------- " $flag
    grep -v "\-\-\-\-" $file | grep -v "Run " | awk -F "|" '{print $3" "$2"\t"acount" "aflag}' aflag=$flag acount=$count | sort  >> SummaryTable.tmp
    TotFlag=`echo "$TotFlag \t|\t $flag"`
  done

  echo -e "$TotFlag" > SummaryTable.txt

  cat SummaryTable.tmp | sort | awk '
function report(){
while(lastEntry!=MaxEntry){labels=sprintf("%s\t| \t ",labels);lastEntry++;}
print MajCount"."MinCount" \t|\t "Run" \t|\t "DetID" "labels;labels="";lastEntry=0;
}
BEGIN{Run=-1;DetID=0;labels="";lastEntry=0;MajCount=0;MinCount=1;}
{
if($1!=Run){
if(Run!=-1)report(); 
print "-------------------------------------------------------------------------------------------------------";
Run=$1;
DetID=$2;
MajCount++; 
MinCount=1;}else{
            if($2!=DetID){report();DetID=$2;MinCount++;}
}
while(lastEntry+1!=$3){labels=sprintf("%s\t| \t ",labels);lastEntry++;}
lastEntry++;
labels=sprintf("%s\t| %s ",labels,$4);
}
' MaxEntry=$count >> SummaryTable.txt

  rm -vf  SummaryTable.tmp

  sqlite3 /tmp/cmstac/prova_$Version.db < /analysis/sw/CRAB/macros/SummaryTable.sql

  echo "...   CreateHtml `pwd` SummaryTable.txt"
  CreateHtml `pwd` SummaryTable.txt
  cd -
done  

