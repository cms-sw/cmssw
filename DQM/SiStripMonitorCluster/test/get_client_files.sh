#!/bin/sh

setenv WHICH_RELEASE CMSSW_0_8_0_pre3

cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/.SiStripClient.xml > .SiStripClient.xml
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/.WebTest.xml > .WebTest.xml
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/.profile.xml > .profile.xml
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/.startMonitorClient > .startMonitorClient
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/WebLib.js > WebLib.js
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/embedded_svg.html > embedded_svg.html
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/sendCmdToApp.pl > sendCmdToApp.pl
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/setup.sh > setup.sh
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/sistrip_monitorelement_config.xml > sistrip_monitorelement_config.xml
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/sistrip_qualitytest_config.xml > sistrip_qualitytest_config.xml
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/style.css > style.css
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/tracker.dat > tracker.dat
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/tracker_mtcc.dat > tracker_mtcc.dat
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/trackermap.txt > trackermap.txt
cvs -q co -r $WHICH_RELEASE -p  DQM/SiStripMonitorClient/test/webPingXDAQ.pl > webPingXDAQ.pl

chmod u+x ./setup.sh
