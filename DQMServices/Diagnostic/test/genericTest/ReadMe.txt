Instruction about how to install and run a simple HDQM test


#**********
#to install
#**********

#*** Version currently used: CMSSW_3_1_0

#*** needed tags


#****************
#populate DB
#*****************

cmsenv
rm -f *db

#**** Create and fill the dbfile.db (and log.db) 

#*** This will open the IOV sequence creating an IOV at 1
cmsRun test_SiStripHistoryDQMService_cfg_69912.py

 
#****************
#Create Trends
#***************

root test_Inspector.cc
