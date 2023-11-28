#!/bin/bash -f

#CreateIndex ()
#{
#    COUNTER=0
#    LASTUPDATE=`date`
#
#    for Plot in `ls *.png`; do
#        if [[ $COUNTER%2 -eq 0 ]]; then
#            cat >> index_new.html  << EOF
#<TR> <TD align=center> <a href="$Plot"><img src="$Plot"hspace=5 vspace=5 border=0 style="width: 90%" ALT="$Plot"></a> 
#  <br> $Plot </TD>
#EOF
#        else
#            cat >> index_new.html  << EOF
#  <TD align=center> <a href="$Plot"><img src="$Plot"hspace=5 vspace=5 border=0 style="width: 90%" ALT="$Plot"></a> 
#  <br> $Plot </TD> </TR> 
#EOF
#        fi
#
#        let COUNTER++
#    done
#
#    cat /afs/cern.ch/cms/tracker/sistrvalidation/WWW/template_index_foot.html | sed -e "s@insertDate@$LASTUPDATE@g" >> index_new.html
#
#    mv -f index_new.html index.html
#}

CreateIndex ()
{  
    LASTUPDATE=`date`
    
            cat >> index_new.html  << EOF
<TR> <TD align=center> <a href="SiStripHitEffTKMapBad.png"><img src="SiStripHitEffTKMapBad.png"hspace=5 vspace=5 border=0 style="width: 90%" ALT="SiStripHitEffTKMapBad.png"></a> 
  <br> SiStripHitEffTKMapBad.png </TD>
EOF

            cat >> index_new.html  << EOF
  <TD align=center> <a href="SiStripHitEffTKMap.png"><img src="SiStripHitEffTKMap.png"hspace=5 vspace=5 border=0 style="width: 90%" ALT="SiStripHitEffTKMap.png"></a> 
  <br> SiStripHitEffTKMap.png </TD> </TR> 
EOF

            cat >> index_new.html  << EOF
  <TR><TD align=center> <a href="Summary.png"><img src="Summary.png"hspace=5 vspace=5 border=0 style="width: 90%" ALT="Summary.png"></a> 
  <br> Summary.png </TD> </TR> 
EOF

    cat /afs/cern.ch/cms/tracker/sistrvalidation/WWW/template_index_foot.html | sed -e "s@insertDate@$LASTUPDATE@g" >> index_new.html

    mv -f index_new.html index.html
}

if [ $# != 1 ]; then
  echo "Usage: $0 CalibTreeFullPath"
  echo "Runs the Hit Efficiency Study"
  exit 0;
fi

fullpath=$1
echo "The full file path is $fullpath"

#Create a macro file to execute to extract the run number
rm -f run_extract.C
echo '{' >> run_extract.C
echo 'TFile* f = TFile::Open("'$1'");' >> run_extract.C
echo 'f->cd("anEff");' >> run_extract.C
echo 'CalibTree = (TTree*)(gDirectory->Get("traj"));' >> run_extract.C
echo 'TLeaf* runLf = CalibTree->GetLeaf("run");' >> run_extract.C
echo 'CalibTree->GetEvent(1);' >> run_extract.C
echo 'double runno = (double)runLf->GetValue();' >> run_extract.C
echo 'cout << runno << endl;' >> run_extract.C
echo '}'>> run_extract.C

root -b -l << EOF >& runnumber1.txt
.x run_extract.C
.q
EOF

runnumber=`cat runnumber1.txt`
echo "The run number is $runnumber"

rm -f runnumber1.txt

cp dbfile_31X_IdealConditions.db dbfile.db
#cp dbfile_31X_IdealConditions.db fakedb.db

cp SiStripHitEff_template.py "SiStripHitEff_run$runnumber.py"
sed -i "s/newrun/$runnumber/g" "SiStripHitEff_run$runnumber.py"
sed -i "s|newfilelocation|$fullpath|g" "SiStripHitEff_run$runnumber.py"

#cp Summary_template.py "Summary_run$runnumber.py"
#sed -i "s/newrun/$runnumber/g" "Summary_run$runnumber.py"
#sed -i "s|newfilelocation|$fullpath|g" "Summary_run$runnumber.py"

cmsRun "SiStripHitEff_run$runnumber.py" >& "run_$runnumber.log" 
#cmsRun "Summary_run$runnumber.py" >& "runSummary_$runnumber.log"

cat run_$runnumber.log | awk 'BEGIN{doprint=0}{if(match($0,"New IOV")!=0) doprint=1;if(match($0,"%MSG")!=0) {doprint=0;} if(match($0,"Message")!=0) {doprint=0;} if(doprint==1) print $0}' > InefficientModules_$runnumber.txt
cat run_$runnumber.log | awk 'BEGIN{doprint=0}{if(match($0,"occupancy")!=0) doprint=1;if(match($0,"efficiency")!=0) doprint=1; if(match($0,"%MSG")!=0) {doprint=0;} if(match($0,"tempfilename")!=0) {doprint=0;} if(match($0,"New IOV")!=0) {doprint=0;} if(match($0,"generation")!=0) {doprint=0;} if(doprint==1) print $0}' > EfficiencyResults_$runnumber.txt

#rm run_$runnumber.log

rm -f run_extract.C

#Now publish all of the relevant files
#Create the relevant directories on a per run basis
#mkdir "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CalibrationValidation/HitEfficiency/run_$runnumber"
#mkdir "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber"
#mkdir "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/cfg"
#mkdir "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/Plots"
#mkdir "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/sqlite"
#mkdir "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/QualityLog"

#Move the config file
#mv "SiStripHitEff_run$runnumber.py" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/cfg"
#mv "plotHitEffSummary_run$runnumber.C" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/cfg"

#Move the log files
#mv "InefficientModules_$runnumber.txt" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/QualityLog"
#mv "EfficiencyResults_$runnumber.txt" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/QualityLog"

#Move the root file containing hot cold maps
#mv "SiStripHitEffHistos_run$runnumber.root" "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERCALIB/SiStrip/CalibrationValidation/HitEfficiency/run_$runnumber"

#Generate an index.html file to hold the TKMaps
#cat /afs/cern.ch/cms/tracker/sistrvalidation/WWW/template_index_header.html | sed -e "s@insertPageName@Validation Plots --- Hit Efficiency Study --- Tracker Maps@g" > index_new.html
#CreateIndex

#mv index.html "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/Plots"
#mv "SiStripHitEffTKMapBad.png" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/Plots"
#mv "SiStripHitEffTKMap.png" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/Plots"
#mv "Summary.png" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/Plots/Summary.png"

# Create the sqlite- and metadata-files
#echo "Preparing the sqlite and metadata files"

#ID1=`uuidgen -t`
#cp dbfile.db SiStripHitEffBadModules@${ID1}.db
#cat template_SiStripHitEffBadModules.txt | sed -e "s@insertFirstRun@$runnumber@g" -e "s@insertIOV@$runnumber@" > SiStripHitEffBadModules@${ID1}.txt

#mv "SiStripHitEffBadModules@${ID1}.db" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/sqlite"
#mv "SiStripHitEffBadModules@${ID1}.txt" "/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/HitEfficiency/run_$runnumber/sqlite"
