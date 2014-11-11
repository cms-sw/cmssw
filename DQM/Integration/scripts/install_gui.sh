
source ~cmssw/cmsset_default.sh
CMSSW_VERSION=CMSSW_1_8_0
DQM_BASE=/home/dqm                  # Choose a directory of your liking
GUICONF=$DQM_BASE/config/server-conf-online.py

ARCH=slc4_ia32_gcc345               # Architecture, only SL4 supported now
DISTAREA=~cmssw/$ARCH               # Where CMS software is installed
mkdir -p $DQM_BASE/rpms
mkdir -p $DQM_BASE/gui

cd $DQM_BASE/rpms
wget -O bootstrap-$ARCH.sh http://cmsrep.cern.ch/cmssw/bootstrap-$ARCH.sh
sh -x ./bootstrap-$ARCH.sh setup -path $PWD >& $PWD/bootstrap.log </dev/null
source ./slc4_ia32_gcc345/external/apt/0.5.15lorg3.2-CMS3/etc/profile.d/init.sh
apt-get update
apt-get install cms+webtools+1.0.0

cd $DQM_BASE
scramv1 p CMSSW $CMSSW_VERSION
cd $CMSSW_VERSION

scramv1 b ExternalLinks
ln -s $DISTAREA/external/libpng/1.2.10-CMS3/lib/lib*.so* lib/$ARCH
ln -s $DISTAREA/external/libtiff/3.8.2-CMS3/lib/lib*.so* lib/$ARCH

cd src
export CVS_RSH=ssh CVSROOT=:pserver:anonymous@cmscvs.cern.ch:/cvs_server/repositories/CMSSW
cvs co VisMonitoring/DQMServer

cvs co DQM/Integration
cvs co DQM/RenderPlugins
scramv1 b

ln -s $CMSSW_VERSION/src/DQM/Integration/config $DQM_BASE/config

eval `cd $DQM_BASE/$CMSSW_VERSION && scramv1 run -sh`

visDQMControl start all from $GUICONF

tail -f $DQM_BASE/gui/*/log


