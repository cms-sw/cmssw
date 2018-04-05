# Methode to publish png pictures on the web. Will be used at the end of the script:
CreateIndex ()
{
    COUNTER=0
    LASTUPDATE=`date`

    for Plot in `ls *.png`; do
	if [[ $COUNTER%3 -eq 0 ]]; then
	    cat >> index_gain.html  << EOF
<TR> <TD align=center> <a href="$Plot"><img src="$Plot"hspace=5 vspace=5 border=0 style="width: 80%" ALT="$Plot"></a> 
  <br> $Plot </TD>
EOF
	else if [[ $COUNTER%3 -eq 1 ]]; then
	    cat >> index_gain.html  << EOF
  <TD align=center> <a href="$Plot"><img src="$Plot"hspace=5 vspace=5 border=0 style="width: 80%" ALT="$Plot"></a> 
  <br> $Plot </TD>
EOF
	else
	    cat >> index_gain.html  << EOF
  <TD align=center> <a href="$Plot"><img src="$Plot"hspace=5 vspace=5 border=0 style="width: 80%" ALT="$Plot"></a> 
  <br> $Plot </TD> </TR> 
EOF
	fi
	fi

	let COUNTER++
    done

    cat /afs/cern.ch/cms/tracker/sistrvalidation/WWW/template_index_foot.html | sed -e "s@insertDate@$LASTUPDATE@g" >> index_gain.html

    mv -f index_gain.html index.html
}
# end of publication methode

if [ "$#" != '1' ]; then
#   cp Empty_Sqlite.db Gains_Sqlite.db
   echo "Running: cmsRun Gains_Compute_cfg.py"
   cmsRun Gains_Compute_cfg.py
   root -l -b -q KeepOnlyGain.C+

   #can not run validation from PCL inputs
   if [[ $1 != *"PCL"* ]]; then
      echo "Running: cmsRun Validation_Compute_cfg.py"
      cmsRun Validation_Compute_cfg.py
   fi

   if [ "$#" == '0' ]; then
      sh PlotMacro.sh
   fi

   if [ "$#" == '3' ]; then
      sh PlotMacro.sh "\"$2\"" "\"$3\""
   fi
else
   WORKDIR=$PWD
   DIRPATH=/afs/cern.ch/cms/tracker/sistrvalidation/WWW/CalibrationValidation/ParticleGain/$1
   echo "Creating directories"
   mkdir $DIRPATH
   mkdir $DIRPATH/cfg
   mkdir $DIRPATH/log
   mkdir $DIRPATH/sqlite
   mkdir $DIRPATH/plots_gain
   mkdir $DIRPATH/plots_validation
   echo "Move results to the respective directories"
   cp Gains_Compute_cfg.py $DIRPATH/cfg/.
   cp Validation_Compute_cfg.py $DIRPATH/cfg/.
   cp FileList_cfg.py $DIRPATH/cfg/.
   cp Gains_Sqlite.db $DIRPATH/sqlite/.
   cp Gains.root $DIRPATH/sqlite/.
   cp Gains_ASCII.txt $DIRPATH/log/.
   cp Validation_ASCII.txt $DIRPATH/log/.
   cp Pictures/*.txt $DIRPATH/log/.
   cp Pictures/Gains_*.png $DIRPATH/plots_gain/.
   rm $DIRPATH/plots_gain/Gains_Charge.png
   rm $DIRPATH/plots_gain/Gains_MPVs.png
   rm $DIRPATH/plots_gain/Gains_MPVsSubDet.png
   rm $DIRPATH/plots_gain/Gains_MPVsSubDetAndStat.png
   cp Pictures/Validation_*.png $DIRPATH/plots_validation/.
   rm $DIRPATH/plots_validation/Validation_Charge.png
   rm $DIRPATH/plots_validation/Validation_MPVs.png
   rm $DIRPATH/plots_validation/Validation_MPVsSubDet.png
   rm $DIRPATH/plots_validation/Validation_MPVsSubDetAndStat.png
   CASTORPATH=/castor/cern.ch/cms/store/group/tracker/strip/calibration/validation/ParticleGain/$1
   rfmkdir $CASTORPATH
   rfcp Gains_Tree.root $CASTORPATH/.
   rfcp Validation_Tree.root $CASTORPATH/.
#   rm Gains_Tree.root
#   rm Validation_Tree.root

   echo "Publish the validation plots on the web"
   INDEXPATH=/afs/cern.ch/cms/tracker/sistrvalidation/WWW
   cd $DIRPATH/plots_gain
   cat $INDEXPATH/template_index_header.html | sed -e "s@insertPageName@Validation Plots --- Particle Gain Calibration --- $1@g" > index_gain.html
   CreateIndex
   cd $DIRPATH/plots_validation
   cat $INDEXPATH/template_index_header.html | sed -e "s@insertPageName@Validation Plots --- Particle Gain Validation --- $1@g" > index_gain.html
   CreateIndex
   cd $WORKDIR
fi
