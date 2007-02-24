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

  1. Noise service default configuration should be working in CMSSW releases
     >= 1_3_0_pre2. Recommended are any >= 1_3_0_pre4. For anything  
     < 1_3_0_pre2 use next configuration of DataBase [simply replace Offline 
     DB part]:

      es_source = PoolDBESSource { 
        VPSet toGet = { 
          { string record = "SiStripPedestalsRcd"  
            string tag    = "SiStripPedNoise_MTCC_v1_p"},

          { string record = "SiStripNoisesRcd"     
            string tag    = "SiStripPedNoise_MTCC_v1_n"},

          { string record = "SiStripFedCablingRcd" 
            string tag    = "SiStripCabling_MTCC_v1"}
        }
      
        untracked bool siteLocalConfig = true
        string connect  = "frontier://cms_conditions_data/CMS_COND_STRIP"
        string timetype = "runnumber" 
        untracked uint32 messagelevel   = 0
        untracked bool loadBlobStreamer = true
        untracked uint32 authenticationMethod = 0
      }

  2. In case error access to DB try to setup environment variable CORAL_AUTH_PATH:

       Bash:  export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
       Tcsh:  setenv CORAL_AUTH_PATH /afs/cern.ch/cms/DB/conddb

  3. In case of authentication.xml file access error:

       [1] Copy DB setup files into some local folder
           cp /afs/cern.ch/cms/DB/conddb <UserDir>

       [2] Edit authentication.xml in <UserDir> and replace in each ENTITY 
           path to files with <UserDir>

       [3] Open MyConfig.cfg from "Simple Run" instructions and replace 
           authenticationPath value with <UserDir> in PoolDBESSource
