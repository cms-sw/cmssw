SUMMARY
    Modules does not perform any analysis. It's purpose is to be a plug for 
  configuration. Such chain can be used with any ROOT tuple to extract some
  number of events that is less than what is actually saved in input root or
  to reduce number of objects that are saved in input file.

CONTACT INFO
  Samvel Khalatian (samvel at fnal dot gov) 

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
