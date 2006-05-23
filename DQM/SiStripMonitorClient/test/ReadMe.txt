1. Description of Classes :
=============================

    SiStripWebClient (web client for the SiStrip, inherited from DQMWbClient)
    SiStripQualityTester (defines and attaches QualityTests to the MEs. the
                         test can be defined in test.txt file [for the moment])


    TrackerMap & TmModule (creates the TrackerMap SVG file)



2. Auxiliary files in test directory
=====================================

  o test.txt   : Quality test of ME is defined here  which has the following 
                 format
                 QTest  Title of QTest  Error Pro. Warning Prob  Parameters)
  
  o tracker.dat            : needed to create TrackerMap
    trackermap.txt         : header of the SVG file to be created

  o sendCmdToApp.pl        : scripts needed to start the xdaq.exe
    webPingXDAQ.pl       
    WebLib.js               
   
  o setup.sh               : creates necessary xml and other files needed
                             for a given environment

  o .WebTest.xml           : used by setup.sh to create real ones
    .profile.xml            WebTest.xml profile.xml 
  
  o  .startMonitorClient   : used by  setup.sh to create start script 
                             startMonitorClient

3. Running
=================

  - start the collector    (execute DQLCollector)
  - start the source       (execute cmsRun -p DQM_digicluster.cfg in 
                           DQM/SiStripMonitorCluster/test directory)
  - start xdaq executable (execute startMonitorClient in 
                           DQM/SiStripMonitorClient/test directory)
  - in a web browser open the link
      http://MACHINE_NAME:1972  (e.g http://lxplus020.cern.ch:1972)
      