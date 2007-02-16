SUMMARY
    Config starts modules that produce Infos for basic objects such as
  SiStripClusters, etc. Such Info objects are useful in analysis
  and simplifies access to many properties.

  [Note: at the moment only SiStripClusterInfo's produced]

CONTACT INFO
  Samvel Khalatian (samvel at fnal dot gov)

INSTRUCTION
  
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
