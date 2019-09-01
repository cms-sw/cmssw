#!/bin/bash
#setting up environment variables
export HOME=@home
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:@home/bin

JCPORT=9999

while getopts ":t:r:p:k" options; do
    case $options in
        t ) TPG_KEY=$OPTARG;;
        r ) RUN_NUMBER=$OPTARG;;
        p ) JCPORT=$OPTARG;;
        k ) KILLSWITCH=1;;
    esac
done

source @root/scripts/setup.sh -j EcalTPG

log "-----------------------------------------------------------------------"
log "EcalTPG.sh"
log "PID $$"
log "HOSTNAME $HOSTNAME"
log "JCPORT $JCPORT"
log "TPG_KEY $TPG_KEY"
log "RUN_NUMBER $RUN_NUMBER"
log "date `date`"
log "-----------------------------------------------------------------------"

SRCDIR=$RELEASEDIR/src/CondTools/Ecal/python

# run the O2Os...
submit_cmsRun EcalTPGTowerStatus $SRCDIR/copyBadTT_cfg.py
submit_cmsRun EcalTPGCrystalStatus $SRCDIR/copyBadXT_cfg.py
submit_cmsRun EcalTPGFineGrainEBGroup $SRCDIR/copyFgrGroup_cfg.py
submit_cmsRun EcalTPGFineGrainEBIdMap $SRCDIR/copyFgrIdMap_cfg.py
submit_cmsRun EcalTPGFineGrainStripEE $SRCDIR/copyFgrStripEE_cfg.py
submit_cmsRun EcalTPGFineGrainTowerEE $SRCDIR/copyFgrTowerEE_cfg.py
submit_cmsRun EcalTPGLinearizationConst $SRCDIR/copyLin_cfg.py
submit_cmsRun EcalTPGLutGroup $SRCDIR/copyLutGroup_cfg.py
submit_cmsRun EcalTPGLutIdMap $SRCDIR/copyLutIdMap_cfg.py
submit_cmsRun EcalTPGPedestals $SRCDIR/copyPed_cfg.py
submit_cmsRun EcalTPGPhysicsConst $SRCDIR/copyPhysConst_cfg.py
submit_cmsRun EcalTPGSlidingWindow $SRCDIR/copySli_cfg.py
submit_cmsRun EcalTPGWeightGroup $SRCDIR/copyWGroup_cfg.py
submit_cmsRun EcalTPGWeightIdMap $SRCDIR/copyWIdMap_cfg.py
submit_cmsRun EcalTPGSpike $SRCDIR/copySpikeTh_cfg.py
submit_cmsRun EcalTPGStripStatus $SRCDIR/copyBadStrip_cfg.py
submit_command EcalADCToGeV_express "cmsRun $SRCDIR/EcalADCToGeVConstantPopConBTransitionAnalyzer_cfg.py runNumber=$RUN_NUMBER destinationDatabase={db} destinationTag={tag} tagForRunInfo={runInfoTag} tagForBOff={boffTag} tagForBOn={bonTag}"
submit_command EcalADCToGeV_hlt "cmsRun $SRCDIR/EcalADCToGeVConstantPopConBTransitionAnalyzer_cfg.py runNumber=$RUN_NUMBER destinationDatabase={db} destinationTag={tag} tagForRunInfo={runInfoTag} tagForBOff={boffTag} tagForBOn={bonTag}"
submit_command EcalIntercalibConstants_express "cmsRun $SRCDIR/EcalIntercalibConstantsPopConBTransitionAnalyzer_cfg.py runNumber=$RUN_NUMBER destinationDatabase={db} destinationTag={tag} tagForRunInfo={runInfoTag} tagForBOff={boffTag} tagForBOn={bonTag}"
submit_command EcalIntercalibConstants_hlt "cmsRun $SRCDIR/EcalIntercalibConstantsPopConBTransitionAnalyzer_cfg.py runNumber=$RUN_NUMBER destinationDatabase={db} destinationTag={tag} tagForRunInfo={runInfoTag} tagForBOff={boffTag} tagForBOn={bonTag}"

log "-----------------------------------------------------------------------"
if [ -n "$KILLSWITCH" ]; then
    log "Killswitch activated"
ADDR="http://$HOSTNAME:$JCPORT/urn:xdaq-application:service=jobcontrol/ProcKill?kill=$$"

KILLCMD="curl $ADDR"

log $KILLCMD
$KILLCMD > /dev/null

fi

log DONE


exit 0 
