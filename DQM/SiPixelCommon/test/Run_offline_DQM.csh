#!/bin/csh
if(-e ../../SiPixelMonitorClient/test/sipixel_monitorelement_backup.xml) then
    cp ../../SiPixelMonitorClient/test/sipixel_monitorelement_backup.xml ../../SiPixelMonitorClient/test/sipixel_monitorelement_config.xml
endif

if(-e ../../SiPixelMonitorClient/test/sipixel_monitorelement_skeleton_backup.xml) then
    cp ../../SiPixelMonitorClient/test/sipixel_monitorelement_skeleton_backup.xml ../../SiPixelMonitorClient/test/sipixel_monitorelement_skeleton.xml
    endif

   set all_flag = "false"
    set default_flag = "false"
    set physics_flag = "false"
    set calib_flag = "false"
    set opt_flag = "true"
    set var_flag = ( 0 0 0 0 0 0 0 0 0 0 )
if( !(-d ../../../DQM/SiPixelMonitorClient/test)) then
    echo "Please check out the DQM/SiPixelMonitorClient package"
endif

if($#argv > 1) then
    cd ../../../DQM/SiPixelMonitorClient/test
    cp sipixel_monitorelement_config.xml sipixel_monitorelement_backup.xml
    cp sipixel_monitorelement_skeleton.xml sipixel_monitorelement_skeleton_backup.xml
    set num_args = $#argv
    
    set i = 2
 
    
    while ($i <= $num_args)
	set monlist = $argv[$i]
	    switch($monlist)
		case All:
		    set all_flag = "true"
		    cp sipixel_monitorelement_config_all.xml sipixel_monitorelement_config.xml
		    breaksw
		case RawData:
		    set reglist = "me_parts/rawdataMEs.txt"
		    set grandlist = "me_parts/rawdataMEs.txt"
		    set regspot = "RAWDATA"
		    set grandspot = "RAWDATA"
		    set var_flag[1] = 1
		    
		breaksw
		case Digi:
		    set reglist = "me_parts/digiMEs.txt"
		    set grandlist = "me_parts/digiMEs.txt"
		    set regspot = "DIGIS"
		    set grandspot = "DIGIS"
		    set var_flag[2] = 1
		    
		breaksw
		case Cluster:
		    set reglist = "me_parts/clusterMEs.txt"
		    set grandlist = "me_parts/clusterMEs.txt"
		    set regspot = "CLUSTERS"
		    set grandspot = "CLUSTERS"
		    set var_flag[3] = 1
		    breaksw
		case RecHit:
		    set reglist = "me_parts/rechitMEs.txt"
		    set grandlist = "me_parts/rechitMEs.txt"
		    set regspot = "RECHITS"
		    set grandspot = "RECHITS"
		    set var_flag[4] = 1
		    breaksw
		case Track:
		    set reglist = "me_parts/trackMEs.txt"
		    set grandlist = "me_parts/grandtrackMEs.txt"
		    set regspot = "REGTRACKS"
		    set grandspot = "GRANDTRACKS"
		    set var_flag[5] = 1
		    breaksw
		case Gain:
		    set reglist = "me_parts/gainMEs.txt"
		    set grandlist = "me_parts/grandgainMEs.txt"
		    set regspot = "REGGAIN"
		    set grandspot = "GRANDGAIN"
		    set var_flag[6] = 1
		    set var_flag[7] = 1
		    breaksw
		case SCurve:
		    set reglist = "me_parts/scurveMEs.txt"
		    set grandlist = "me_parts/grandscurveMEs.txt"
		    set regspot = "REGSCURVE"
		    set grandspot = "GRANDSCURVE"
		    set var_flag[6] = 1
		    set var_flag[8] = 1
		    breaksw
		case PixelAlive:
		    set reglist = "me_parts/pixelMEs.txt"
		    set grandlist = "me_parts/grandpixelMEs.txt"
		    set regspot = "REGPIXEL"
		    set grandspot = "GRANDPIXEL"
		     set var_flag[6] = 1
		    set var_flag[9] = 1
		    breaksw
		case Physics:
		    set physics_flag = "true"
		    cp sipixel_monitorelement_config_physicsdata.xml sipixel_monitorelement_config.xml    
		    breaksw
		case Calibration:
		    set calib_flag = "true"
		    cp sipixel_monitorelement_config_calibrations.xml sipixel_monitorelement_config.xml
		    breaksw
		default:
		    echo "${monlist} is not a valid monitor element choice.  Valid options are Physics or Calibration."
		    breaksw
		endsw
	      if( $physics_flag != "true" && $calib_flag != "true" ) then
		
		sed "/$regspot/ r $reglist" < sipixel_monitorelement_skeleton.xml > temp.xml
		cp temp.xml sipixel_monitorelement_skeleton.xml
		rm temp.xml
		if($grandspot != $regspot) then
		    sed "/$grandspot/ r $grandlist" < sipixel_monitorelement_skeleton.xml > temp.xml
		    cp temp.xml sipixel_monitorelement_skeleton.xml
		    rm temp.xml
		endif
	    endif
		@ i = $i + 1
    end

    if ($all_flag == "false" && $calib_flag == "false" && $physics_flag == "false" ) then
       cp sipixel_monitorelement_skeleton.xml sipixel_monitorelement_config.xml
       foreach me_name (RAWDATA DIGIS CLUSTERS GRANDTRACKS RECHITS GRANDGAIN GRANDSCURVE GRANDPIXEL REGTRACKS REGGAIN REGSCURVE REGPIXEL)
	    sed "/$me_name/d" < sipixel_monitorelement_config.xml > temp.xml
	    cp temp.xml sipixel_monitorelement_config.xml
	    rm temp.xml

       end

    endif

    cd ../../SiPixelCommon/test

else
    set opt_flag = "false"
    echo "No option specified!  Please choose Calibration or Physics"
endif

if ( $opt_flag == "true" ) then
    set xx = 2
    set depth = 0
    while ( $xx < 6 )
	if ( $var_flag[$xx] == 1 ) then
	    set depth = $xx
	endif
	@ xx = $xx + 1
    end  

    set runscript = "runme.csh"
    set filelist = $argv[1]
    set file_counter = 1
    foreach filename ( `more $filelist` )

    	if(-e Run_offline_DQM_${file_counter}_cfg.py) then
		    rm Run_offline_DQM_${file_counter}_cfg.py
    	endif

	set rsys = "rfio:"
	set osys = "file:"
	set iscastor = `echo $filename | grep -o /castor/cern.ch`
	set isdata = `echo $filename | grep -o /store/data`
	set calibtype = `echo $filename | grep -o -e "PixelAlive" -e "SCurve" -e "GainCalibration"`
	set file_extension = `echo $filename | grep -o -e ".dmp" -e ".root" -e ".dat"`
	set endrun = `echo $filename | grep -o -e "_[0-9]\{2,\}\."`
	set runnumber = `echo $endrun | grep -o -e ".*[^\.]"`
        set rundefault = "_default"
	

	
	if($calibtype == "PixelAlive" || $calibtype == "SCurve" || $calibtype == "GainCalibration") then
	    set tagnumber = $calibtype$runnumber	    
	else if ($calib_flag == "true") then
	    set tagnumber = "PixelAlive_default"
	else
	    set tagnumber = ""
	endif


	if(  $iscastor == "/castor/cern.ch" ) then
	    set filetorun = $rsys$filename
	else if ( $isdata == "/store/data" ) then
	    set filetorun = $filename
	else
	    set filetorun = $osys$filename
	endif

	if( $physics_flag == "true" ) then
	sed "s#FILENAME#$filetorun#" < client_template_physics_cfg.py > Run_offline_DQM_${file_counter}_cfg.py
	else if ($calib_flag == "true" ) then
	sed "s#FILENAME#$filetorun#" < client_template_calib_cfg.py > Run_offline_DQM_${file_counter}_cfg.py
	else
	sed "s#FILENAME#$filetorun#" < client_template_cfg.py > Run_offline_DQM_${file_counter}_cfg.py
	endif
	
	if( $all_flag == "true" ) then
	    rm Run_offline_DQM_${file_counter}_cfg.py
	sed "s#FILENAME#$filetorun#" < client_template_all_cfg.py > Run_offline_DQM_${file_counter}_cfg.py
	sed "s#CALIBRATIONTAG#$tagnumber#" < Run_offline_DQM_${file_counter}_cfg.py > temp_cfg.py
	cp temp_cfg.py Run_offline_DQM_${file_counter}_cfg.py
	rm temp_cfg.py
	endif
       
        sed "s#CALIBRATIONTAG#$tagnumber#" < Run_offline_DQM_${file_counter}_cfg.py > temp_cfg.py
	cp temp_cfg.py Run_offline_DQM_${file_counter}_cfg.py
	rm temp_cfg.py

	if( $file_extension == ".dat" ) then
	    echo ".dat files are not supported actively by this script, but a config file will be generated anyway"
	    sed 's/DAT//' < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	    rm temp.xml
	    set source_type = "NewEventStreamFileReader"
	    set first_param = " max_event_size = cms.int32(7000000),"
	    set second_param = " max_queue_depth = cms.int32(5),"
	    set converter = "datconverter,"
	    sed "/siPixelDigis/ i\ $converter" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	    
	else
	    sed '/^DAT/d' < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml

	endif

	if( $file_extension == ".dmp" ) then
	    set source_type = PixelSLinkDataInputSource
	    set first_param = " fedid = cms.untracked.int32(-1)"
	    set second_param = "runNumber = cms.untracked.int32(-1)"
	endif
	set has_calibdigis = "false"
	set has_digis = "false"
	set has_clusters = "false"
	set has_rechits = "false"
	if( $file_extension == ".root" ) then
	    set source_type = PoolSource
	    set first_param = ""
	    set second_param = ""
	    edmEventSize -v $filename >& es.log 
	    set has_calibdigis = `grep -m 1 -o -e "siPixelCalibDigis" es.log`
	    if( $has_calibdigis == "siPixelCalibDigis" ) then
		set has_calibdigis = "true"
	    else
		set has_calibdigis = "false"
	    endif


	    set has_digis = `grep -m 1 -o -e "siPixelDigis" es.log`
	    if( $has_digis == "siPixelDigis" ) then
		set has_digis = "true"
	    else
		set has_digis = "false"
	    endif


	    set has_clusters = ` grep -m 1 -o -e "siPixelClusters" es.log`
	    if( $has_clusters == "siPixelClusters" ) then
		set has_clusters = "true"
	    else
		set has_clusters = "false"
	    endif


	    set has_rechits = `grep -m 1 -o -e "siPixelRecHits" es.log`
	    if( $has_rechits == "siPixelRecHits" ) then
		set has_rechits = "true"
	    else
		set has_rechits = "false"
	    endif
	    rm es.log
	endif

    if( $all_flag == "true" || $default_flag == "true" ) then
	if( $has_digis == "true" ) then
	    sed "/siPixelDigis,/d" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	    rm temp.xml
	endif
	if( $has_clusters == "true" ) then
	    sed "/siPixelClusters,/d" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	    rm temp.xml
	endif
	if( $has_rechits == "true" ) then
	    sed "/siPixelRecHits,/d" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	    rm temp.xml
	endif
	if( $has_calibdigis == "true" ) then
	    sed "/siPixelCalibDigis,/d" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	    rm temp.xml
	endif
    endif	
	if( $depth > 1 || $var_flag[6] == 1 ) then
	    if( $depth > 2 ) then
		if( $depth > 3) then
		    if( $has_rechits != "true" ) then
			sed "s/RECSPOT/process.siPixelRecHits*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
			cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
			rm temp.xml
		    endif
		endif
		if( $has_clusters != "true" ) then
			sed "s/CLUSPOT/process.siPixelClusters*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
			cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
			rm temp.xml
		endif
	    endif
	    if( $has_digis != "true" ) then
			sed "s/DIGISPOT/process.siPixelDigis*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
			cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
			rm temp.xml
	    endif
	endif


	if( $calib_flag == "true") then
		if( $has_calibdigis != "true" ) then
	    		sed "s/CDSPOT/process.siPixelCalibDigis*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    		cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    		rm temp.xml
			
		endif
		set calibration_tag = "CRZT210_V1P::All"
			sed "s/GLOBALCALIB/$calibtype/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
		cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
		rm temp.xml
		
	else if( $var_flag[6] != 1 ) then
	  set calibration_tag = "CRUZET4_V5P::All"
	   set connect_string = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
	   sed '/^CALIB/d' < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
		cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
		rm temp.xml
	    sed 's/PHYS//' < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	     cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	     rm temp.xml

	endif
	 
	sed "s/GTAG/$calibration_tag/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	rm temp.xml


	if( $var_flag[1] == 1 ) then
	    sed "s/RAWMONSPOT/process.RAWmonitor*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	endif
	if( $var_flag[2] == 1 ) then
	    sed "s/DIGMONSPOT/process.DIGImonitor*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	endif
	if( $var_flag[3] == 1 ) then
	    sed "s/CLUMONSPOT/process.CLUmonitor*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	endif
	if( $var_flag[4] == 1 ) then
	    sed "s/RECMONSPOT/process.RECmonitor*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	endif
	if( $var_flag[7] == 1 ) then
	    sed "s/GAINSPOT/process.siPixelGainCalibrationAnalysis*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	endif
	if( $var_flag[8] == 1 ) then
	    sed "s/SCURVESPOT/process.siPixelSCurveAnalysis*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	endif
	if( $var_flag[9] == 1 ) then
	    sed "s/PIXELSPOT/process.siPixelIsAliveCalibration*/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	    rm temp.xml
	endif

	foreach mon_name (DIGISPOT CLUSPOT RECSPOT CDSPOT SCURVESPOT GAINSPOT PIXELSPOT RAWMONSPOT DIGMONSPOT CLUMONSPOT RECMONSPOT)
	    sed "s/$mon_name//" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	    rm temp.xml

       end



	

	sed "s/SOURCETYPE/$source_type/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	rm temp.xml
	sed "s/ONEPARAM/$first_param/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	rm temp.xml
	sed "s/TWOPARAM/$second_param/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}_cfg.py 
	rm temp.xml
	set logname = "DQM_text_output_"
	set logfile = $logname$file_counter
	sed "s/TEXTFILE/$logfile/" < Run_offline_DQM_${file_counter}_cfg.py > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}_cfg.py
	rm temp.xml       
	echo "created file Run_offline_DQM_${file_counter}_cfg.py to run on file ${filename}"
	@ file_counter = $file_counter + 1
	
    end

endif
