Collection of scripts for the automatic population of the database tags and production of the trend plots.

The WatchDog.sh script is called by the acrontab to run the python scritps.

The HDQMDatabaseProducer.py script uses the class defined in HDQMDatabaseProducerConfiguration.py to read
the parameters from a configuration file (e.g. HDQMDatabaseProducerConfiguration_StreamExpress.cfg). It can
discover all the type or reco files present and produce a different tag for each type, or use the types specified
in the cfg.
This script:
- loops on all the types
- uses the class in DiscoverDQMFiles.py to discovers the DQM root files in the source directory specified in the cfg
- uses the class in DiscoverProcessedRuns.py to build the list of runs that are already in the database
- compares the two lists to find the DQM root files it needs to process
- uses the class in PopulateDB.py to put the information in the database from each DQM root file
- if some new run was added it uses the class in ProducePlots.py to make the plots

The ProducePlots class uses the class in SelectRuns.py to select the good runs using the runreg.py script with the
condition specified in the cfg. It then produces plots for all runs, last 40 runs both without any selection (all
the available runs) and with the good runs selection. It copies the results in the web area specified in the cfg.