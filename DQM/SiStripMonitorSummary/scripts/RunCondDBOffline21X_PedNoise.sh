    #!/bin/sh
    export PATH=$PATH:/afs/cern.ch/cms/sw/common/
    #===============Setting parameters=====================
    Tag=GR_21X_v2_hlt ## Global Tag here
    FedCablingTag=SiStripFedCabling_$Tag
    NoiseTag=SiStripNoise_$Tag
    PedestalTag=SiStripPedestals_$Tag
    ThresholdTag=SiStripThreshold_$Tag
    CMSCondAccount=CMS_COND_21X_STRIP
    DB_Tag=temp_CAF ## Global Tag here
    QtestsFileName=CondDBQtests.xml
    search_IoV=$PedestalTag
    BaseDir=/home/cmstacuser/CMSSWReleasesForCondDB
    logDir=log21X
    outdir=/storage/data1/SiStrip/SiStripDQM/output/conddb
    #======================================================

    cd `dirname $0`
    WorkDir=`pwd`
    cd $BaseDir/CMSSW_2_1_4/src/
    eval `scramv1 runtime -sh`
    cd $WorkDir

   # get the list of IoV
#   cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_21X_STRIP -t  $search_IoV 

#oracle://cms_orcoff_prod/$CMSCondAccount -P /afs/cern.ch/cms/DB/conddb -t $search_IoV 

    cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_21X_STRIP -t  $search_IoV | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > list_Iov.txt
 
    [ ! -e $logDir ] && mkdir $logDir


    touch $logDir/WhiteList_${Tag}.txt

    grep Run_In list_Iov.txt | awk '{print $2}'
    
    for Run_In_number in `grep Run_In list_Iov.txt | awk '{print $2}'`; 
      do

      [ $Run_In_number == "Total" ] && continue 

    RunNb=$Run_In_number
    RootFile_name="CondDB_"$Tag"_"$RunNb


    [ "`grep -c "$RunNb RUN_TAG-OK" $logDir/WhiteList_${Tag}.txt`" != "0" ] && echo "run done already, skipping!" &&	continue  

    ## Build the cff and cfg files:
    cat TemplateCfg21X_PedNoise.cfg | sed -e "s@insert_runnumber@$RunNb@g" \
    -e "s@insert_DB_Tag@$DB_Tag@g" \
    -e "s@insert_FedCablingTag@$FedCablingTag@g" \
    -e "s@insert_ThresholdTag@$ThresholdTag@g" \
    -e "s@insert_QtestsFileName@$QtestsFileName@g" \
    -e "s@insert_NoiseTag@$NoiseTag@g" \
    -e "s@insert_PedestalTag@$PedestalTag@g" \
    -e "s@insertAccount@$CMSCondAccount@g" > $logDir/MainCfg_${RunNb}.cfg  

    echo @@@ Running on run number $RunNb
    cmsRun $logDir/MainCfg_${RunNb}.cfg >  $logDir/output_${RunNb}.log
    exitStatus=$?

    if [ "$exitStatus" == "0" ]; then 
	if `mv DQM_SiStrip_R*$RunNb.root ${outdir}/${RootFile_name}.root` ; then
	     echo $RunNb" RUN_TAG-OK" >> $logDir/WhiteList_${Tag}.txt
	fi
    else
	    echo $RunNb" RUN_TAG-BAD" >> $logDir/WhiteList_${Tag}.txt
    fi

    done









  
