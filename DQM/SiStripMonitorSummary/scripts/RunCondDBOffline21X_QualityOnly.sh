    #!/bin/sh
    export PATH=$PATH:/afs/cern.ch/cms/sw/common/
    #===============Setting parameters=====================
    Tag=GR_21X_v2_hlt
    FedCablingTag=SiStripFedCabling_$Tag
    NoiseTag=SiStripNoise_$Tag
    PedestalTag=SiStripPedestals_$Tag
    ThresholdTag=SiStripThreshold_$Tag
    CMSCondAccount=CMS_COND_21X_STRIP
    DB_Tag=SiStripBadChannel_v1 ## temporary tag for test
    ####DB_Tag=temp_CAF ## Global Tag here
    QtestsFileName=CondDBQtests.xml
    search_IoV=SiStripBadChannel_TKCC_21X_v2_offline
    
    BaseDir=/home/cmstacuser/CMSSWReleasesForCondDB     

    outdir=/storage/data1/SiStrip/SiStripDQM/output/conddb

    #======================================================

    cd `dirname $0`
    WorkDir=$BaseDir/CMSSW_2_1_12/src/DQM/SiStripMonitorSummary/ ## to be changed in vers on TAC
   cd $WorkDir
   eval `scramv1 runtime -sh`


   # get the list of IoV
#    cmscond_list_iov -c oracle://cms_orcoff_prod/$CMSCondAccount -P /afs/cern.ch/cms/DB/conddb -t $search_IoV 
#    cmscond_list_iov -c sqlite_file:dbfile.db -t $DB_Tag
    cmscond_list_iov -c frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_21X_STRIP -t $search_IoV | awk '{if(NR>4) print "Run_In "$1 " Run_End " $2}' > $WorkDir/$logDir/list_Iov_Quality.txt


    [ ! -e $logDir ] && mkdir $logDir


    touch $WorkDir/$logDir/WhiteList_${Tag}_SiStripQuality.txt

    grep Run_In list_Iov_Quality.txt | awk '{print $2}'



    ## Build the cff files:
    cd $WorkDir/scripts
    cat Template_Tags21X_Quality_cff.py | sed -e "s@insert_FedCablingTag@$FedCablingTag@g" \
    -e "s@insert_ThresholdTag@$ThresholdTag@g" \
      -e "s@insert_NoiseTag@$NoiseTag@g" \
    -e "s@insert_PedestalTag@$PedestalTag@g" \
    -e "s@insertAccount@$CMSCondAccount@g" > $WorkDir/python/Tags21X_Quality_cff.py 
    cat Template_tagsQuality_cfi.py | sed -e "s@insert_DB_Tag@$DB_Tag@g" > $WorkDir/python/tagsQuality_cfi.py 

# compile cff in the first occurence:
cd $WorkDir/python 
scramv1 b


    
     for Run_In_number in `grep Run_In $WorkDir/$logDir/list_Iov_Quality.txt | awk '{print $2}'`;
      do

       [ $Run_In_number == "Total" ] && continue 

    RunNb=$Run_In_number
    RootFile_name="Quality_"$Tag"_"$RunNb


     [ "`grep -c "$RunNb RUN_TAG-OK" $WorkDir/$logDir/WhiteList_${Tag}_SiStripQuality.txt`" != "0" ] && echo "run done already, skipping!" &&	continue  

   # Build the cfg files:
    cd $WorkDir/scripts
    cat TemplateCfg21X_Quality_cfg.py | sed -e "s@insert_runnumber@$RunNb@g" \
    -e "s@insert_DB_Tag@$search_IoV@g" \
    -e "s@insert_QtestsFileName@$QtestsFileName@g" > $WorkDir/test/MainCfg_${RunNb}_QualityOnly_cfg.py

    cd $WorkDir/test
    echo @@@ Running on run number $RunNb
    cmsRun MainCfg_${RunNb}_QualityOnly_cfg.py >  $WorkDir/$logDir/output_${RunNb}_QualityOnly.log
    exitStatus=$?

    if [ "$exitStatus" == "0" ]; then 
	if `mv SiStrip*.root ${outdir}/${RootFile_name}.root` ; then
	     echo $RunNb" RUN_TAG-OK" >> $WorkDir/$logDir/WhiteList_${Tag}_SiStripQuality.txt
	fi
    else
	    echo $RunNb" RUN_TAG-BAD" >> $WorkDir/$logDir/WhiteList_${Tag}_SiStripQuality.txt
    fi

    done









  
