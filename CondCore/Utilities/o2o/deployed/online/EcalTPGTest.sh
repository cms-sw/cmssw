#!/bin/bash
#setting up environment variables
export HOME=/nfshome0/popconpro
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/nfshome0/popconpro/bin

#JCPORT=9999

#while getopts ":t:r:p:k" options; do
#    case $options in
#        t ) TPG_KEY=$OPTARG;;
#        r ) RUN_NUMBER=$OPTARG;;
#        p ) JCPORT=$OPTARG;;
#        k ) KILLSWITCH=1;;
#    esac
#done

source /data/O2O/scripts/setupO2O.sh -s Ecal -j TPGTest

#log "-----------------------------------------------------------------------"
#log "EcalTPG.sh"
#log "PID $$"
#log "HOSTNAME $HOSTNAME"
#log "JCPORT $JCPORT"
#log "TPG_KEY $TPG_KEY"
#log "RUN_NUMBER $RUN_NUMBER"
#log "date `date`"
#log "-----------------------------------------------------------------------"

SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python

# run the O2Os...
submit_test_cmsRun copyBadTT $SRCDIR/copyBadTT_cfg.py
submit_test_cmsRun copyLin copyLin_cfg.py
submit_test_cmsRun copyPed copyPed_cfg.py
submit_test_cmsRun copyPhysConst copyPhysConst_cfg.py
submit_test_cmsRun copySli copySli_cfg.py
submit_test_cmsRun updateADCToGeV_express updateADCToGeV_express.py

# END OF CHANGES
#log "-----------------------------------------------------------------------"
#if [ -n "$KILLSWITCH" ]; then
#    log "Killswitch activated"
#ADDR="http://$HOSTNAME:$JCPORT/urn:xdaq-application:service=jobcontrol/ProcKill?kill=$$"

#KILLCMD="curl $ADDR"

#log $KILLCMD
#$KILLCMD > /dev/null

#fi

#log DONE


exit 0 
