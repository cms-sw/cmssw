#!/bin/sh

function atLeast(){
    grep "&&&&" $outFile | grep $1 | sort -n -k 4 | awk '{if ($4>=a) print $0}' a=$2 | wc -l | tr -d [:blank:] 
}

function CreateHtml(){

    path=$1
    webadd="http://cmstac11.cern.ch:8080/analysis/"

    export webfile=$path/`echo $2 | sed -e "s@.txt@.html@g"`
    export htmlwebadd=`echo $webfile | sed -e "s@/data1/@$webadd@"`

    tableSeparator="<TD align=center>"

    rm -f $webfile
    echo "<TABLE  BORDER=1 ALIGN=CENTER> " > $webfile  
#Commented table with % columns
#    echo -e "<TD>N $tableSeparator Module $tableSeparator Run Number  $tableSeparator %   All  $tableSeparator N All $tableSeparator % inside $tableSeparator N inside $tableSeparator % leftEdge $tableSeparator N leftEdge $tableSeparator % rightEdge $tableSeparator N rightEdge  <TR> " >> $webfile
    echo -e "<TD>N $tableSeparator Module $tableSeparator Run Number  $tableSeparator HistoEntries $tableSeparator%   All  $tableSeparator N All  $tableSeparator N inside  $tableSeparator N leftEdge $tableSeparator N rightEdge  <TR> " >> $webfile

    cat $path/$2 | grep "|" | grep -v "Run" | sed -e "s@(@|@g" -e "s@)@@g" -e "s@[ \t]*@@g" | awk -F"|" '
BEGIN{
Separator[0]="<TD align=center>"; 
Separator[1]="</font><TD align=center>"; 
fontColor[0]="<font color=\"#FF0000\">";
fontColor[1]="<font color=\"#00000\">";
N=0;
}
{
if (int($1)!=N) {N=int($1); if (i==0){i=1}else{i=0} }
pswebadd=sprintf("%s/../ClusterAnalysis_*%d/res/HotStrips/cPos_SingleDet_%s*.gif",path,$3,$2);
smrywebadd=sprintf("%s/../ClusterAnalysis_*%d/res/ClusterAnalysis_*%d*HotStrips.smry",path,$3,$3);
#Commented table with % columns
#print "<TD> " fontColor[i] "<a href=pswebadd>" $1 "</a>" Separator[0] fontColor[i] $2 Separator[1] fontColor[i] "<a href=smrywebadd>" $3 "</a>" Separator[1] fontColor[i] $4 Separator[1] fontColor[i] $5 Separator[1] fontColor[i] $6 Separator[1] fontColor[i] $7 Separator[1] fontColor[i] $8 Separator[1] fontColor[i] $9 Separator[1] fontColor[i] $10 Separator[1] fontColor[i] $11"</font> <TR> | " pswebadd " | " smrywebadd  
print "<TD> " fontColor[i] "<a href=pswebadd>" $1 "</a>" Separator[0] fontColor[i] $2 Separator[1] fontColor[i] "<a href=smrywebadd>" $3 "</a>" Separator[1] fontColor[i] $12 Separator[1] fontColor[i] $4 Separator[1] fontColor[i] $5 Separator[1] fontColor[i] $7 Separator[1] fontColor[i] $9  Separator[1] fontColor[i] $11"</font> <TR> | " pswebadd " | " smrywebadd  
}' path=$path | while read line; do
	echo `echo $line | awk -F"|" '{sub("pswebadd",$2,$1);sub("smrywebadd",$3,$1);gsub("/data1/",webadd,$1); print $1}' webadd="$webadd"` >> $webfile
    done
    echo "</TABLE> " >> $webfile
}



function extractInfo(){

    #Run=`echo $1 |  sed -e "s@[a-zA-Z]*.txt@@g" -e "s@_[0-9]*@@g" -e "s@[[:alpha:]]@@g" | awk '{print int($0)}'`
    Run=`echo $1 | awk -F"_" '{print int($3)}'`

############
# summary
############

    echo -e "*Run $Run"
    echo -e "\n------------------------------------------------------------------"
    echo TotalNumber of Modules with bad strips: `atLeast T 1`
    echo -e "------------------------------------------------------------------"
    echo -e "SubDet \t | \t >0  \t >4 \t >9 \t >24 \t >49 \t >99"
    echo -e "------------------------------------------------------------------"
    for SubDet in TIB TID TOB TEC
      do
      echo -e "$SubDet \t | \t `atLeast $SubDet 1` \t `atLeast $SubDet 5` \t `atLeast $SubDet 10` \t `atLeast $SubDet 25` \t `atLeast $SubDet 50` \t `atLeast $SubDet 100`"
    done 
    echo -e "------------------------------------------------------------------\n"

############
# list
############

    echo -e "#BadStrips \t\t\t\t|\t Detector\t\t|\t\tBadStrips List" 
    echo -e "#(all,inside,lEdge,rEdge)\t\t|\t\t\t\t|"
    echo -e "# out of Nstrips - HistoEntries\t|\t\t\t\t|" 

    echo -e "------------------------------------------------------------------------------------------------------------------------------------\n"

    #grep "&&&&" $outFile | sort -n -k 4  | awk '{print $4 "\t<-->\t" $2}' | sed -e "s@Det_@@" -e "s@_[0-9]*@ @5" -e "s@_@ @g"  

    for det in `grep "&&&&" $outFile | sort -n -k 2  | awk '{print $2}'` 
      do
      listBad=`grep $det $outFile | grep -v "ooooo" | awk '{if ($1 != "&&&&&&") print $3}'`
      grep $det $outFile | awk '{if ($1 == "&&&&&&") print $4"\t"$18"\t"$10"\t"$15"\t"$22"\t"$24"\t|\t" $2} ' | sed -e "s@Det_@@" -e "s@_[0-9]*@ @5" -e "s@_@ @g" | cut -d " " -f "-6" | awk '{print "* "$0 "\t|\t" a}' a="`echo $listBad`"
    done 

    echo -e "*------------------------------------------------------------------------------------------------------------------------------------\n"
}

function SummaryInfo(){
    cd $Tpath/$path/
    rm -rf ASummaries
    rm -f Asummary*
    mkdir -p $Tpath/$path/AllSummaries
    
    cd $Tpath/$path/AllSummaries

    runSmryFile=`ls ../*/res/*_HotStrips.smry 2>/dev/null`
    [ "`echo $runSmryFile`" == "" ] && return

    #echo $runSmryFile


      #Summaryes x each SubDet to extract trends

    echo "...creating .gif files with root"
    rm -f Asummary_HotStrips_T*
    echo "Run >=0 >=5 >=10 >=25 >=50 >=100" > Asummary_HotStrips_TIB.dat
    echo "Run >=0 >=5 >=10 >=25 >=50 >=100" > Asummary_HotStrips_TID.dat
    echo "Run >=0 >=5 >=10 >=25 >=50 >=100" > Asummary_HotStrips_TOB.dat
    echo "Run >=0 >=5 >=10 >=25 >=50 >=100" > Asummary_HotStrips_TEC.dat
    for file in `echo $runSmryFile`
      do
      Run=`cat $file | head -1 | awk '{print $2}'`
      EntriesTIB=`cat $file | head -11 | grep TIB | awk -F "|" '{print $2}'`
      EntriesTID=`cat $file | head -11 | grep TID | awk -F "|" '{print $2}'`
      EntriesTOB=`cat $file | head -11 | grep TOB | awk -F "|" '{print $2}'`
      EntriesTEC=`cat $file | head -11 | grep TEC | awk -F "|" '{print $2}'`
      echo $Run $EntriesTIB >> Asummary_HotStrips_TIB.dat
      echo $Run $EntriesTID >> Asummary_HotStrips_TID.dat
      echo $Run $EntriesTOB >> Asummary_HotStrips_TOB.dat
      echo $Run $EntriesTEC >> Asummary_HotStrips_TEC.dat

    done

    root.exe -b -q -l "$macroPath/CreateSinglePlotFromTable.C(\"$Tpath/$path/AllSummaries/Asummary_HotStrips_TIB.dat\",\"Run\",\"N modules\")"
    root.exe -b -q -l "$macroPath/CreateSinglePlotFromTable.C(\"$Tpath/$path/AllSummaries/Asummary_HotStrips_TID.dat\",\"Run\",\"N modules\")"
    root.exe -b -q -l "$macroPath/CreateSinglePlotFromTable.C(\"$Tpath/$path/AllSummaries/Asummary_HotStrips_TOB.dat\",\"Run\",\"N modules\")"
    root.exe -b -q -l "$macroPath/CreateSinglePlotFromTable.C(\"$Tpath/$path/AllSummaries/Asummary_HotStrips_TEC.dat\",\"Run\",\"N modules\")"

    echo "...creating Asummary_HotStrips.txt"
    
    echo -e "N |\t\t Module |\t Run |\t %(Num Bads) all |\t inside |\t leftEdge |\t rigthEdge |\t HistoEntries" > Asummary_HotStrips.txt

    cat $runSmryFile | grep "^*" | awk -F"|" '{print $2" "$1}' | sed -e "s@*@@" | awk 'function perc(n,d){p=-1;if(d>0){p=int(n/d*10000)/100;}; return sprintf("%3.2f",p);}{if(index($1,"--")){print $1}else if(index($1,"Run")){print $1" "$2}else{print $1"_"$2"_"$3"_"$4"_"$5" "perc($6,$10)"("$6")?"perc($7,$10)"("$7")?"perc($8,$10)"("$8")?"perc($9,$10)"("$9")?"$11}}' | awk '
BEGIN{Run=1} 
{ 
if( index($1,"--") == 0){
if (index($1,"Run")){ Run=$2}
else{ print $1"\t|"Run"\t|"$2} 
}
}
' | sort  | sed -e "s@(@ (@g" -e "s@?@|\t@g" | awk 'BEGIN{det=0;count=0;count2=0}{if ($1!=det){count++;count2=0;det=$1;print"----------------------------------------------------------------------------------------------------------------"};count2++;print count"."count2"|\t"$0}' >> Asummary_HotStrips.txt

    echo "...creating Asummary_HotStrips.html"
    rm -f  Asummary_HotStrips.html
    CreateHtml `pwd` Asummary_HotStrips.txt
    
    cd -
}

############
## MAIN  ###
############

Version=""
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
  for dir in `ls $Tpath/$path`
    do
    #workdir=$Tpath/$path/$dir
    #[ -e $workdir/res ] && workdir=$workdir/res  #to take into account the recent movement in res dir of results
    workdir=$Tpath/$path/$dir/res

    [ ! -e $workdir ] && continue 
    cd $workdir
 
    rootFile=$dir.root
    if [ ! -e $rootFile ]; then
	rootFile=`ls -1 | grep ".root" | head -1`
    fi
    [ "$rootFile" == "" ] && continue
    pwd
    outFile=`echo $rootFile | sed -e "s@.root@_HotStrips.txt@"`  
    binFile=`echo $rootFile | sed -e "s@.root@_HotStrips.bin@"`  
    smryFile=`echo $rootFile | sed -e "s@.root@_HotStrips.smry@"`  

    if [ ! -e $outFile ]; then

	rm -fv *_HotStrips*

        ############
        # root macro
        ############
	
	echo "...Running root"
	echo "root.exe -q -b -l \"$macroPath/RunBadStripsFromPosition.C(\"$rootFile\",\"$binFile\")\" > $outFile"
	root.exe -q -b -l "$macroPath/RunBadStripsFromPosition.C(\"$rootFile\",\"$binFile\")" > $outFile
	exitStatus=$?
	if [ `ls cPos*.gif 2>/dev/null | wc -l` -ne 0 ]; then
	    mkdir -p HotStrips
	    mv -f cPos*.gif HotStrips 2>/dev/null
	fi
	[ "$exitStatus" != "0" ] && continue
        echo "...Running extractInfo"
	extractInfo $outFile > $smryFile
    fi
    cd -
  done

  echo "...Running SummaryInfo"
  cd $Tpath/$path/
  SummaryInfo 
done



