#!/bin/sh

function getVal(){
    grep "&&&&" $outFile | grep $1 | grep "Layer_$2" | awk '{print $8}'  
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
    echo -e "<TD>N $tableSeparator Module $tableSeparator Run Number  $tableSeparator %  Bad Apvs  $tableSeparator N Bad Apvs  <TR> " >> $webfile

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
pswebadd=sprintf("%s/../ClusterAnalysis_*%d/res/TracksQT/DBPedestals_SingleDet_%s*.gif",path,$3,$2);
smrywebadd=sprintf("%s/../ClusterAnalysis_*%d/res/ClusterAnalysis_*%d*TracksQT.smry",path,$3,$3);
#Commented table with % columns
#print "<TD> " fontColor[i] "<a href=pswebadd>" $1 "</a>" Separator[0] fontColor[i] $2 Separator[1] fontColor[i] "<a href=smrywebadd>" $3 "</a>" Separator[1] fontColor[i] $4 Separator[1] fontColor[i] $5 Separator[1] fontColor[i] $6 Separator[1] fontColor[i] $7 Separator[1] fontColor[i] $8 Separator[1] fontColor[i] $9 Separator[1] fontColor[i] $10 Separator[1] fontColor[i] $11"</font> <TR> | " pswebadd " | " smrywebadd  
print "<TD> " fontColor[i] "<a href=pswebadd>" $1 "</a>" Separator[0] fontColor[i] $2 Separator[1] fontColor[i] "<a href=smrywebadd>" $3 "</a>" Separator[1] fontColor[i] $4 Separator[1] fontColor[i] $5 "</font> <TR> | " pswebadd " | " smrywebadd  
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
    echo "AngleVsPhi (zero val) for each Layer"
    echo -e "------------------------------------------------------------------"
    echo -e "SubDet \t | \t L1 \t L2 \t L3 \t L4 \t L5 \t L6 "
    echo -e "------------------------------------------------------------------"
    for SubDet in TIB TOB 
      do
      echo -e "$SubDet \t | \t `getVal $SubDet 1` \t `getVal $SubDet 2` \t `getVal $SubDet 3` \t `getVal $SubDet 4` \t `getVal $SubDet 5` \t `getVal $SubDet 6`"
    done 
    echo -e "------------------------------------------------------------------\n"

############
# list
############

    echo -e "#Phi@zero\t ErrPhi \t FitPar \t errFitPar \t|\t SubDet_Layer" 
    echo -e "------------------------------------------------------------------------------------------------------------------------------------\n"

    grep "&&&&&" $outFile | awk '{print "* "$8" \t"$10" \t"$4" \t"$6"\t\t|\t" $2} ' | sed -e "s@AngleVsPhi_@@" -e "s@_onTrack@@"
    
    echo -e "*------------------------------------------------------------------------------------------------------------------------------------\n"
}

function SummaryInfo(){
    cd $Tpath/$path/
    rm -rf ASummaries
    rm -f Asummary*
    mkdir -p $Tpath/$path/AllSummaries
    
    cd $Tpath/$path/AllSummaries

    runSmryFile=`ls ../*/res/*_TracksQT.smry 2>/dev/null`
    [ "`echo $runSmryFile`" == "" ] && return

    #echo $runSmryFile


      #Summaryes x each SubDet to extract trends

    echo "...creating .gif files with root"
    rm -f Asummary_TracksQT_T*
    echo "Run L1 L2 L3 L4" > Asummary_TracksQT_TIB.dat
    echo "Run L1 L2 L3 L4 L5 L6" > Asummary_TracksQT_TOB.dat
    echo "Run _L1 L2 L3 L4 L5 L6 L7 L8 L9 L10" > Asummary_TracksQT_TIBTOB.dat
    for file in `echo $runSmryFile`
      do
      Run=`cat $file | head -1 | awk '{print $2}'`
      EntriesTIB=`cat $file | head -11 | grep TIB | awk -F "|" '{print $2}'`
      EntriesTOB=`cat $file | head -11 | grep TOB | awk -F "|" '{print $2}'`
      echo $Run $EntriesTIB >> Asummary_TracksQT_TIB.dat
      echo $Run $EntriesTOB >> Asummary_TracksQT_TOB.dat
      echo $Run $EntriesTIB $EntriesTOB >> Asummary_TracksQT_TIBTOB.dat
    done

    root.exe -b -q -l "$macroPath/CreateSinglePlotFromTable.C(\"$Tpath/$path/AllSummaries/Asummary_TracksQT_TIB.dat\",\"Run\",\"Phi @ normal incidence (deg)\",false)"
    root.exe -b -q -l "$macroPath/CreateSinglePlotFromTable.C(\"$Tpath/$path/AllSummaries/Asummary_TracksQT_TOB.dat\",\"Run\",\"Phi @ normal incidence (deg)\",false)"
    root.exe -b -q -l "$macroPath/CreateSinglePlotFromTable.C(\"$Tpath/$path/AllSummaries/Asummary_TracksQT_TIBTOB.dat\",\"Run\",\"Phi @ normal incidence (deg)\",false)"

    echo "...creating Asummary_TracksQT.txt"
    
    echo -e "N |\t\t SubDet_Layer |\t Run |\t Phi(at Zero) |\t PhiErr |\t FitPar |\t ErrFitPar " > Asummary_TracksQT.txt

    cat $runSmryFile | grep "^*" | awk -F"|" '{print $2" "$1}' | sed -e "s@*@@" | awk '{if(index($1,"--")){print $1}else if(index($1,"Run")){print $1" "$2}else{print $1"  "$2"?"$4"?"$3"?"$5}}' | awk '
BEGIN{Run=1} 
{ 
if( index($1,"--") == 0){
if (index($1,"Run")){ Run=$2}
else{ print $1"\t|"Run"\t|"$2} 
}
}
' | sort  | sed -e "s@?@|\t@g" | awk 'BEGIN{det=0;count=0;count2=0}{if ($1!=det){count++;count2=0;det=$1;print"----------------------------------------------------------------------------------------------------------------"};count2++;print count"."count2"|\t"$0}' >> Asummary_TracksQT.txt
    
    echo "...creating Asummary_TracksQT.html"
    rm -f  Asummary_TracksQT.html
#    CreateHtml `pwd` Asummary_TracksQT.txt
    
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

cd ${basePath}/CMSSW/CMSSW_1_3_0_V01-01-10/src
#eval `scramv1 runtime -sh`
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
    outFile=`echo $rootFile | sed -e "s@.root@_TracksQT.txt@"`  
    smryFile=`echo $rootFile | sed -e "s@.root@_TracksQT.smry@"`  

    if [ ! -e $outFile ]; then

	rm -fv *_TracksQT.*

        ############
        # root macro
        ############
	
	echo "...Running root"
	echo "root.exe -q -b -l \"$macroPath/RunTracksQT.C(\"$rootFile\",\"prova\")\" > $outFile"
	root.exe -q -b -l "$macroPath/RunTracksQT.C(\"$rootFile\",\"prova\")" > $outFile
	exitStatus=$?
	if [ `ls *TracksQT*.gif 2>/dev/null | wc -l` -ne 0 ]; then
	    mkdir -p TracksQT
	    mv -f *TracksQT*.gif TracksQT 2>/dev/null
	    mv -f *TracksQT*.eps TracksQT 2>/dev/null
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



