SUMMARY
    Config starts reconstruction chain that is standart to MTCC involving:
      - Digitization
      - Clusterization
      - Tracks finders
      - TrackInfo production 
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

  1. In case NoiseService does not work follow next steps:
       [1] Copy DB setup files
           cp /afs/cern.ch/cms/DB/conddb <UserDir>
       [2] Edit authentication.xml in <UserDir> and replace in each ENTITY 
           path to files with <UserDir>
       [3] Open MyConfig.cfg from "Simple Run" instructions and replace 
           authenticationPath value with <UserDir> in PoolDBESSource
