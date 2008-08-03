#!/bin/csh
if(-e ../../SiPixelMonitorClient/test/sipixel_monitorelement_backup.xml) then
    cp ../../SiPixelMonitorClient/test/sipixel_monitorelement_backup.xml ../../SiPixelMonitorClient/test/sipixel_monitorelement_config.xml
endif

if(-e ../../SiPixelMonitorClient/test/sipixel_monitorelement_skeleton_backup.xml) then
    cp ../../SiPixelMonitorClient/test/sipixel_monitorelement_skeleton_backup.xml ../../SiPixelMonitorClient/test/sipixel_monitorelement_skeleton.xml
    endif

   set all_flag = "false"
    set default_flag = "false"
    set var_flag = ( 0 0 0 0 0 0 0 0 0 )
if( !(-d ../../../DQM/SiPixelMonitorClient/test)) then
    echo "Please check out the DQM/SiPixelMonitorClient package"
else if($#argv > 1) then
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
		default:
		    echo "${monlist} is not a valid monitor element choice.  Valid options are: RawData, Digi, Cluster, RecHit, Track, Gain, SCurve, PixelAlive, All.  Choosing All overrides all other options."
		    breaksw
		endsw
	      
	    sed "/$regspot/ r $reglist" < sipixel_monitorelement_skeleton.xml > temp.xml
		cp temp.xml sipixel_monitorelement_skeleton.xml
		rm temp.xml
	    if($grandspot != $regspot) then
	    sed "/$grandspot/ r $grandlist" < sipixel_monitorelement_skeleton.xml > temp.xml
		cp temp.xml sipixel_monitorelement_skeleton.xml
		rm temp.xml
	    endif
		@ i = $i + 1
    end

    if( $var_flag[6] == 1 ) then
	set calib_use = 1
    else
	set calib_use = 0
    endif
    sed "s/CALIBUSE/$calib_use/" < sipixel_monitorelement_skeleton.xml > temp.xml
    cp temp.xml sipixel_monitorelement_skeleton.xml
    rm temp.xml



    if ($all_flag == "false") then
       cp sipixel_monitorelement_skeleton.xml sipixel_monitorelement_config.xml
       foreach me_name (RAWDATA DIGIS CLUSTERS GRANDTRACKS RECHITS GRANDGAIN GRANDSCURVE GRANDPIXEL REGTRACKS REGGAIN REGSCURVE REGPIXEL)
	    sed "/$me_name/d" < sipixel_monitorelement_config.xml > temp.xml
	    cp temp.xml sipixel_monitorelement_config.xml
	    rm temp.xml

       end

    endif

    cd ../../SiPixelCommon/test

else

    echo "No monitor elements specified.  Using default configuration (RawData/Digis/Calibrations)"
    set default_flag = "true"
endif


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

    	if(-e Run_offline_DQM_${file_counter}.cfg) then
		    rm Run_offline_DQM_${file_counter}.cfg
    	endif

	set rsys = "rfio:"
	set osys = "file:"
	set iscastor = `echo $filename | grep -o castor`
	set calibtype = `echo $filename | grep -o -e "PixelAlive" -e "SCurve" -e "GainCalibration"`
	set file_extension = `echo $filename | grep -o -e ".dmp" -e ".root" -e ".dat"`
	set endrun = `echo $filename | grep -o -e "_[0-9]\{2,\}\."`
	set runnumber = `echo $endrun | grep -o -e ".*[^\.]"`
        set rundefault = "_default"
	

	
	if($calibtype == "PixelAlive" || $calibtype == "SCurve" || $calibtype == "GainCalibration") then
	    set tagnumber = $calibtype$rundefault	    
	else
	    set tagnumber = "PixelAlive_default"
	    echo "No calibrations detected for file ${filename}, using default tag"
	endif


	if(  $iscastor == "castor" ) then
	    set filetorun = $rsys$filename
	else
	    set filetorun = $osys$filename
	endif

	if( $default_flag == "false" ) then
	sed "s#FILENAME#$filetorun#" < client_template.cfg > Run_offline_DQM_${file_counter}.cfg
	else
	sed "s#FILENAME#$filetorun#" < client_template_default.cfg > Run_offline_DQM_${file_counter}.cfg
	endif
	
	if( $all_flag == "true" ) then
	    rm Run_offline_DQM_${file_counter}.cfg
	sed "s#FILENAME#$filetorun#" < client_template_all.cfg > Run_offline_DQM_${file_counter}.cfg
	sed "s#TAG#$tagnumber#" < Run_offline_DQM_${file_counter}.cfg > temp.cfg
	cp temp.cfg Run_offline_DQM_${file_counter}.cfg
	rm temp.cfg
	endif

	

       
        sed "s#TAG#$tagnumber#" < Run_offline_DQM_${file_counter}.cfg > temp.cfg
	cp temp.cfg Run_offline_DQM_${file_counter}.cfg
	rm temp.cfg

	if( $file_extension == ".dat" ) then
	    sed 's/DAT//' < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg
	    rm temp.xml
	    set source_type = "NewEventStreamFileReader"
	    set first_param = "int32 max_event_size = 7000000"
	    set second_param = "int32 max_queue_depth = 5"
	    set converter = "datconverter,"
	    sed "/siPixelDigis/ i\ $converter" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	    
	else
	    sed '/^DAT/d' < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml

	endif

	if( $file_extension == ".dmp" ) then
	    set source_type = PixelSLinkDataInputSource
	    set first_param = "untracked int32 fedid = -1"
	    set second_param = "untracked int32 runNumber = -1"
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
	    sed "/siPixelDigis,/d" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg
	    rm temp.xml
	endif
	if( $has_clusters == "true" ) then
	    sed "/siPixelClusters,/d" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg
	    rm temp.xml
	endif
	if( $has_rechits == "true" ) then
	    sed "/siPixelRecHits,/d" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg
	    rm temp.xml
	endif
	if( $has_calibdigis == "true" ) then
	    sed "/siPixelCalibDigis,/d" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg
	    rm temp.xml
	endif
    endif	
	if( $depth > 1 || $var_flag[6] == 1 ) then
	    if( $depth > 2 ) then
		if( $depth > 3) then
		    if( $has_rechits != "true" ) then
			sed "s/RECSPOT,/siPixelRecHits,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
			cp temp.xml Run_offline_DQM_${file_counter}.cfg 
			rm temp.xml
		    endif
		endif
		if( $has_clusters != "true" ) then
			sed "s/CLUSPOT,/siPixelClusters,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
			cp temp.xml Run_offline_DQM_${file_counter}.cfg 
			rm temp.xml
		endif
	    endif
	    if( $has_digis != "true" ) then
			sed "s/DIGISPOT,/siPixelDigis,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
			cp temp.xml Run_offline_DQM_${file_counter}.cfg 
			rm temp.xml
	    endif
	endif


	if( $var_flag[6] == 1) then
		if( $has_calibdigis != "true" ) then
	    		sed "s/CDSPOT,/siPixelCalibDigis,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    		cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    		rm temp.xml
		endif
	endif

	if( $var_flag[1] == 1 ) then
	    sed "s/RAWMONSPOT,/RAWmonitor,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	endif
	if( $var_flag[2] == 1 ) then
	    sed "s/DIGMONSPOT,/DIGImonitor,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	endif
	if( $var_flag[3] == 1 ) then
	    sed "s/CLUMONSPOT,/CLUmonitor,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	endif
	if( $var_flag[4] == 1 ) then
	    sed "s/RECMONSPOT,/RECmonitor,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	endif
	if( $var_flag[7] == 1 ) then
	    sed "s/GAINSPOT,/siPixelGainCalibrationAnalysis,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	endif
	if( $var_flag[8] == 1 ) then
	    sed "s/SCURVESPOT,/siPixelSCurveAnalysis,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	endif
	if( $var_flag[9] == 1 ) then
	    sed "s/PIXELSPOT,/siPixelIsAliveCalibration,/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	    rm temp.xml
	endif

	foreach mon_name (DIGISPOT CLUSPOT RECSPOT CDSPOT SCURVESPOT GAINSPOT PIXELSPOT RAWMONSPOT DIGMONSPOT CLUMONSPOT RECMONSPOT)
	    sed "/$mon_name/d" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	    cp temp.xml Run_offline_DQM_${file_counter}.cfg
	    rm temp.xml

       end



	

	sed "s/SOURCETYPE/$source_type/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	rm temp.xml
	sed "s/ONEPARAM/$first_param/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	rm temp.xml
	sed "s/TWOPARAM/$second_param/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}.cfg 
	rm temp.xml
	set logname = "DQM_text_output_"
	set logfile = $logname$file_counter
	sed "s/TEXTFILE/$logfile/" < Run_offline_DQM_${file_counter}.cfg > temp.xml
	cp temp.xml Run_offline_DQM_${file_counter}.cfg
	rm temp.xml       
	echo "created file Run_offline_DQM_${file_counter}.cfg to run on file ${filename}"
	@ file_counter = $file_counter + 1
	
    end


