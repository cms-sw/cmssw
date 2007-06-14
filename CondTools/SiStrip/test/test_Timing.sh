#!/bin/sh

function test_db() {

    [ $# = 0 ] && echo -e "usage: test_db -mode<write/read> -what<gain,ped,noise,pednoise,modulehv,badstrip> -stream<blob,noblob> -geom_mtcc<-geom_ideal> <-debug>\n"

    mode=""
    [ `echo $@ | grep -c "\-write[ ]*"` = 1 ] && mode=write 
    [ `echo $@ | grep -c "\-read[ ]*"`  = 1 ] && mode=read 

    [ "$mode" != "write" ] && [ "$mode" != "read" ] && return
    
    blobflag=noblob
    [ `echo $@ | grep -c "\-blob[ ]*"` = 1 ] && blobflag=blob 
    [ `echo $@ | grep -c "\-badstrip[ ]*"`      = 1 ] && module=badstrip     
    [ `echo $@ | grep -c "\-modulehv[ ]*"`      = 1 ] && module=modulehv     
    [ `echo $@ | grep -c "\-gain[ ]*"`      = 1 ] && module=gain     
    [ `echo $@ | grep -c "\-ped[ ]*"`      = 1 ] && module=ped     
    [ `echo $@ | grep -c "\-noise[ ]*"`    = 1 ] && module=noise   
    [ `echo $@ | grep -c "\-pednoise[ ]*"` = 1 ] && module=pednoise

    [ `echo $@ | grep -c "\-geom_mtcc[ ]*"`  = 1 ] && geom=geom_mtcc
    [ `echo $@ | grep -c "\-geom_ideal[ ]*"` = 1 ] && geom=geom_ideal

    debugflag="false"
    [ `echo $@ | grep -c "\-debug[ ]*"` = 1 ] && debugflag=debug 

    logfile=${mode}_${module}_${geom}_${blobflag}.log
    dbfile=${workdir}/${module}_${geom}_${blobflag}.db
    dbcatalog=${workdir}/${module}_${geom}_${blobflag}.xml

    cfgfile=${workdir}/${mode}_${module}_${geom}_${blobflag}.cfg


    eval `scramv1 runtime -sh`
    SealPluginRefresh

    if [ ${blobflag} == "blob" ] && [ "${mode}" == "write" ]; then

        [ -e ${dbfile} ] && rm -f ${dbfile}

	echo "cmscond_bootstrap_detector.pl --offline_connect sqlite_file:$dbfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP "
	cmscond_bootstrap_detector.pl --offline_connect sqlite_file:$dbfile --auth ${CORAL_AUTH_PATH}/authentication.xml STRIP 
	pool_insertFileToCatalog -u file:${dbcatalog} -t POOL_RDBMS sqlite_file:${dbfile}
	echo " " 

    fi

    cat template_Timing_${mode}.cfg | sed \
        -e "s@insert_workdir@${workdir}@"  \
        -e "s@insert_logfile@${logfile}@"  \
        -e "s@insert_dbfile@${dbfile}@"  \
        -e "s@insert_dbcatalog@${dbcatalog}@"  \
        -e "s@\#${geom}@@"  \
        -e "s@\#${blobflag}@@"  \
        -e "s@\#${debugflag}[ ]* @@g" \
        -e "s@\#${module}[ ]* @@g" \
        > ${cfgfile}

    echo -e "\ncmsRun ${cfgfile} | tee ${workdir}/out \n"
    cmsRun ${cfgfile} | tee ${workdir}/out

    export timereport=`grep "Time report complete in" ${workdir}/out | awk '{print $6}'`
}

function setEnvironment(){
    [ -n "$setEnvironment" ] && return
    setEnvironment=1
    
    echo -e "Creating sqlite db schema from custom mapping"
    
    export workdir=/tmp/$USER/$$
    export CORAL_AUTH_USER="me"
    export CORAL_AUTH_PASSWORD="me"
    
    if [ ! -n "$CORAL_AUTH_PATH" ];
	then
	export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
	echo -e "\nWARNING: CORAL_AUTH_PATH environment variable is not defined in your shell\n default value will be used CORAL_AUTH_PATH=$CORAL_AUTH_PATH"
    fi
    

    echo -e "workdir $workdir\n"

    [ -e ${workdir} ] && rm -rf ${workdir} 
    mkdir ${workdir}
}
#%%%%%%%%%%%%%%%%%%%%
#   Main
#%%%%%%%%%%%%%%%%%%%%

setEnvironment

if [ "$1" == "doLoop" ];
    then
    i=0
    for stream in blob noblob; 
      do
      for mode in write read;
	do
	for what in badstrip gain ped noise pednoise;
	  do
	      #echo -e "\n\n$mode $what with $stream on geometry mtcc\n\n"      
	      #test_db -$mode -$what -$stream -geom_mtcc -debug
	      #timeis[$i]=$timereport
	      #let i++
	  echo -e "\n\n$mode $what with $stream on geometry ideal\n\n"      
	  test_db -$mode -$what -$stream -geom_ideal
	  timeis[$i]=$timereport
	  let i++
	done
      done
    done

    echo -e "\n\nTime Report\n\n"


    i=0
    for stream in blob noblob; 
      do
      for mode in write read;
	do
	for what in modulehv badstrip gain ped noise pednoise;
	  do
	     # echo -e "$mode \t$what \twith $stream on geometry mtcc debug \t\t" ${timeis[$i]}     
	     # let i++
	  echo -e "$mode \t$what \twith $stream on geometry ideal      \t\t" ${timeis[$i]}     
	  let i++
	done
      done
    done
else
    echo -e "\n[usage]:  "
    echo -e "\n\ttest_Timing.sh doLoop"
    echo -e "OR\n\ttest_db -mode<write/read> -what<modulehv, badstrip,gain,ped,noise,pednoise> -stream<blob,noblob> -geom_mtcc<-geom_ideal> <-debug>\n"
fi
