SUMMARY
    Configuration file runs next steps on Simulated Data:
      - Digitization
      - Clusterization
      - Tracks finders
      - ClusterInfo
      - TrackInfo
  Produced collections of objects are saved in output ROOT tuple file.

  [Note: At the moment only Cosmic Track Finder is used]

CONTACT INFO
  Samvel Khalatian ( samvel at fnal dot gov)

INSTRUCTIONS

  Simple Run
  ----------
    1. Copy config file with custom name say MyConfig.cfg
    2. Edit MyConfig.cfg
         - Set input file by replacing {INPUT_FILE}.

         - Set maximum number of events to be processed by replacing 
           {EVENTS_NUM} with integer number

           [Hint: -1 stands for all event processing]

    3. cmsRun MyConfig.cfg 

  Advanced Run
  ------------

TROUBLESHOOTING
