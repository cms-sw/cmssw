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
  
  o tracker.dat                      : needed to create TrackerMap
    trackermap.txt                   : header of the SVG file to be created

  o sendCmdToApp.pl                  : scripts needed to start the xdaq.exe
    webPingXDAQ.pl 
      
  o ConfigBox.js, GifDisplay.js,     : Java scripts for the Web Widjets
    Navigator.js,  WebLib.js, 
    Select.js, ContentViewer.js  
    Messages.js

  o setup.sh                         : creates necessary xml and scripts needed
                                       for a given environment

  o .SiStripClient.xml                : used by setup.sh to create real ones
    .profile.xml                      SiStripClient.xml profile.xml 
  
  o  .startMonitorClient             : used by  setup.sh to create start script 
                                       startMonitorClient

  o  style.css                       : color, border... etc

  o  sistrip_qualitytest_config.xml  : quality Test Configuration file 
                                       where the tests and the attachments 
                                       of tests with ME are defined
  o  sistrip_monitorelement_config.xml : the MEs to be used in TrackerMap 
                                         and the summary plot are defined
                                          here


3. Running
=================

   The Client needs have Source and the Collector to run. The Collector and the Client 
    must not run in the same machine. It is probably better to start Collector, Client 
    and Source on three different machines.

  - Collector
       o login to a (lxplus) machine
       o go to the CMSSW working directory
       o do a eval `scramv1 runtime -csh`
       o execute "DQMCollector"

   - Source
       o login to a (lxplus) machine (possibly different than the Collector one)
       o go to DQM/StripMonitorCluster/test area of CMSSW working directory
       o do a eval `scramv1 runtime -csh`
       o put the Collector machine name in DQM/StripMonitorCluster/data/MonitorDaemon.cfi
         config file (in the field DestinationAddress)
       o execute "cmsRun  OnlyDQM.cfg"


   - Client (start xdaq executable)
       in DQM/SiStripMonitorClient/test directory

       o login to a machine which must be different wrt the Collector one
       o go to DQM/SiStripMonitorClient/test of the CMSSW working directory, 
       o do a eval  `scramv1 runtime -csh`
       o execute setup.sh script to setup the machine names for Collector and Client
         (the Client machine is taken automatically, instead the Collector machine
          name should be put in the argument)

          setup.sh COLLECTOR_MACHINE_NAME

       o execute startMonitorClient script to start xdaq executable

       o in a web browser open the link
           http://CLIENT_MCHINE_NAME:1972  (e.g http://lxplus020.cern.ch:1972)
       o select SiStripClient application from XDAQ window
       o click "Configure" and "Enable" buttons .... then the Client is ready to run!

   - IGUANA-CMS GUI

      one can start the IGUANA-CMS GUI using the Client as the server. To do that
      one has to use "actAsServer" as true in .SiStripClient.xml before executing
      setup.sh before starting the Client

      o go to any directory inside CMSSW working area
      o do a eval  `scramv1 runtime -csh`     
      o execute "iguana"
      o select "Vis Example--NTuple browser"
      o specify Client machine name (or the Collector) and the port (9090)
      
