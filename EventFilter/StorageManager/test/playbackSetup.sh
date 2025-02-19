#
# 30-Nov-2009, KAB - This script sets up a bash environment for a
# storage manager playback system.
# It needs to be run from the test directory of the SM development area.
#

# simply re-use the testSetup.sh script and over-ride the necessary
# configuration parameters
source ./testSetup.sh
alias startEverything="cd $STMGR_DIR/bin; source ./startEverything.sh playback"
export SMDEV_FU_PROCESS_COUNT=1
export STMGR_CONFIG=$STMGR_DIR/cfg/sm_playback.xml
