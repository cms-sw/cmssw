SUMMARY
    Config starts reconstruction chain that is standart to MTCC (except for Digitization step) involving:
      - Clusterization
      - Tracks finders
      - TrackInfo production 
  Produced collections of objects are saved in output ROOT tuple file.

  [Note: At the moment only Cosmic Track Finder is used]

  [Note: most objects labels are replaced with different values]

  [Note: The purpose of given configuration file is to perform reconstruction
         chain after Digis were amplified by MTCCAmplifyDigis_default.cfg]

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
