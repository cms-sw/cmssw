1. Description of Classes :
=============================

    SiStripWebInterface (web interface  for the SiStrip, inherited from WebInterface)
    SiStripQualityTester (defines and attaches QualityTests to the MEs. the
                         tests are defined in test/sistrip_qualitytest_config.xml)
    SiStripActionExecutor (performs various actions as requested by WebInterface)
   
    SiStripWebClient [OBSOLETE] (web client for the SiStrip, inherited from DQMWbClient)


    TrackerMap & TmModule (creates the TrackerMap SVG file)



2. Auxiliary files in test directory
=====================================

  o sistrip_qualitytest_config.xml   : Quality tests and the association of tests with ME 
                                        is defined here
  
  o tracker.dat                      : needed to create TrackerMap
    trackermap.txt                   : header of the SVG file to be created

  o sendCmdToApp.pl                  : scripts needed to start the xdaq.exe
    webPingXDAQ.pl       
    WebLib.js               
   
  o setup.sh                         : creates necessary xml and other files needed
                                       for a given environment

  o .SiStripClient.xml                : used by setup.sh to create real ones
    .profile.xml                      SiStripClient.xml profile.xml 
  
  o  .startMonitorClient             : used by  setup.sh to create start script 
                                       startMonitorClient

  o  style.css                       : color, border... etc

3. Running
=================

  - do not forget to execute
     eval `scramv1 runtime -csh`

  - collector
        execute DQLCollector from any directory 

  - source
        execute cmsRun -p DQM_digicluster.cfg in DQM/StripMonitorCluster/test directory


  - client (start xdaq executable)
       in DQM/SiStripMonitorClient/test directory

        o execute setup.sh script to setup the machine name
        o execute startMonitorClient script to start xdaq executable

        o in a web browser open the link
           http://MACHINE_NAME:1972  (e.g http://lxplus020.cern.ch:1972)

