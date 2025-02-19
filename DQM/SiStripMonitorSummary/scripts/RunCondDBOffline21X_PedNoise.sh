    #!/bin/sh
    export PATH=$PATH:/afs/cern.ch/cms/sw/common/
    export FRONTIER_FORCERELOAD=long
    #===============Setting parameters=====================
    Tag=GR_21X_v2_hlt 
    FedCablingTag=SiStripFedCabling_$Tag
    NoiseTag=SiStripNoise_$Tag
    PedestalTag=SiStripPedestals_$Tag
    ThresholdTag=SiStripThreshold_$Tag
    CMSCondAccount=CMS_COND_21X_STRIP
    QtestsFileName=CondDBQtests.xml
    search_IoV=$PedestalTag
    BaseDir=/afs/cern.ch/user/h/hashemim/scratch0/     
    logDir=log
    outDir=/tmp/hashemim/
    CMSSWVersion=CMSSW_2_2_5
    #======================================================
    cd `dirname $0`
    WorkDir=`pwd`
    cd $BaseDir/$CMSSWVersion/src/
    eval `scramv1 runtime -sh`
    cd $BaseDir/$CMSSWVersion/src/DQM/SiStripMonitorSummary/python/
    scramv1 b
    cd $WorkDir


    [ ! -e $logDir ] && mkdir $logDir
    [ ! -e $outDir ] && mkdir $outDir

    cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_21X_STRIP -t  $search_IoV | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > $logDir/list_Iov.txt
    


    touch $logDir/WhiteList_${Tag}.txt

    grep Run_In $logDir/list_Iov.txt | awk '{print $2}'

    
    


    for Run_In_number in `grep Run_In $logDir/list_Iov.txt | awk '{print $2}'`; 
      do
      [ $Run_In_number == "Total" ] && continue 
      
      RunNb=$Run_In_number
      RootFile_name="CondDB_"$Tag"_"$RunNb


      [ "`grep -c "$RunNb RUN_TAG-OK" $logDir/WhiteList_${Tag}.txt`" != "0" ] && echo "run done already, skipping!" &&	continue  

    # Build the cfg files:
      cat $BaseDir/$CMSSWVersion/src/DQM/SiStripMonitorSummary/scripts/TemplateCfg21X_PedNoise_cfg.py | sed -e "s@insert_FedCablingTag@$FedCablingTag@g" \
	  -e "s@insert_ThresholdTag@$ThresholdTag@g" \
	  -e "s@insert_NoiseTag@$NoiseTag@g" \
	  -e "s@insert_PedestalTag@$PedestalTag@g" \
	  -e "s@insertAccount@$CMSCondAccount@g" \
	  -e "s@insert_runnumber@$RunNb@g" \
	  -e "s@insert_QtestsFileName@$QtestsFileName@g" > $logDir/MainCfg_${RunNb}_cfg.py  

#      cd $logDir      
#      scramv1 b
#      cd $WorkDir

      echo @@@ Running on run number $RunNb
      cmsRun $logDir/MainCfg_${RunNb}_cfg.py >  $logDir/output_${RunNb}.log
      exitStatus=$?

      if [ "$exitStatus" == "0" ]; then 
# 	if `mv SiStrip*.root $WorkDir/$outDir/${RootFile_name}.root` ; then
	  if `mv DQM*.root $outDir/${RootFile_name}.root` ; then
	      echo $RunNb" RUN_TAG-OK" >> $logDir/WhiteList_${Tag}.txt
	  fi
      else
	  echo $RunNb" RUN_TAG-BAD" >> $logDir/WhiteList_${Tag}.txt
      fi
      
    done
    








    
