    #!/bin/sh
    export PATH=$PATH:/afs/cern.ch/cms/sw/common/
    export FRONTIER_FORCERELOAD=long
    #===============Setting parameters=====================
    Tag=GR_21X_v2_hlt
    QualityTag=HotStrip_CRAFT_v3_offline
    FedCablingTag=SiStripFedCabling_$Tag
    NoiseTag=SiStripNoise_$Tag
    PedestalTag=SiStripPedestals_$Tag
    ThresholdTag=SiStripThreshold_$Tag
    CMSCondAccount=CMS_COND_21X_STRIP
    QtestsFileName=CondDBQtests.xml
    search_IoV=SiStripBadChannel_HotStrip_CRAFT_v3_offline
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

    cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_21X_STRIP -t $search_IoV | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > $logDir/list_Iov_Quality_${QualityTag}.txt


    touch $logDir/WhiteList_${QualityTag}_SiStripQuality.txt

    grep Run_In $logDir/list_Iov_Quality_${QualityTag}.txt | awk '{print $2}'




    
    for Run_In_number in `grep Run_In $logDir/list_Iov_Quality_${QualityTag}.txt | awk '{print $2}'`;
      do
      
      [ $Run_In_number == "Total" ] && continue 
      
      RunNb=$Run_In_number
      RootFile_name="Quality_"$QualityTag"_"$RunNb


      [ "`grep -c "$RunNb RUN_TAG-OK" $logDir/WhiteList_${QualityTag}_SiStripQuality.txt`" != "0" ] && echo "run done already, skipping!" &&	continue  
   # Build the cfg files:
      cat $BaseDir/$CMSSWVersion/src/DQM/SiStripMonitorSummary/scripts/TemplateCfg21X_Quality_cfg.py | sed -e "s@insert_FedCablingTag@$FedCablingTag@g" \
          -e "s@insert_ThresholdTag@$ThresholdTag@g" \
	  -e "s@insert_NoiseTag@$NoiseTag@g" \
	  -e "s@insert_PedestalTag@$PedestalTag@g" \
	  -e "s@insertAccount@$CMSCondAccount@g" \
          -e "s@insert_DB_Tag@$search_IoV@g" \
          -e "s@insert_runnumber@$RunNb@g" \
	  -e "s@insert_QtestsFileName@$QtestsFileName@g" > $logDir/MainCfg_${RunNb}_QualityOnly_cfg.py

      echo @@@ Running on run number $RunNb
      cmsRun $logDir/MainCfg_${RunNb}_QualityOnly_cfg.py >  $logDir/output_${RunNb}_QualityOnly.log
      exitStatus=$?

      if [ "$exitStatus" == "0" ]; then 
	  if `mv SiStrip*.root ${outDir}/${RootFile_name}.root` ; then
	      echo $RunNb" RUN_TAG-OK" >> $logDir/WhiteList_${QualityTag}_SiStripQuality.txt
	  fi
      else
	  echo $RunNb" RUN_TAG-BAD" >> $logDir/WhiteList_${QualityTag}_SiStripQuality.txt
      fi

    done









    
