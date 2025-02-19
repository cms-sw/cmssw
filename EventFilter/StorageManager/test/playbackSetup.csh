#
# 30-Nov-2009, KAB - This script sets up a csh environment for a
# storage manager playback system.
# It needs to be run from the test directory of the SM development area.
#

# simply re-use the testSetup.csh script and over-ride the necessary
# configuration parameters
source ./testSetup.csh
alias startEverything "cd $STMGR_DIR/bin; ./startEverything.sh playback"
setenv SMDEV_FU_PROCESS_COUNT 1
setenv STMGR_CONFIG $STMGR_DIR/cfg/sm_playback.xml
