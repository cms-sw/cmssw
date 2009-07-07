Instruction about how to install and run a simple HDQM test


#**********
#to install
#**********

#*** Version currently used: CMSSW_3_1_0

#*** needed tags
cvs co -r V03-00-00      DQM/SiPixelHistoricInfoClient                    


#****************
#populate DB
#*****************

cmsenv
rm -f *db

#**** Create and fill the dbfile.db (and log.db) 

#*** This will open the IOV sequence creating an IOV at 1
cmsRun test_SiStripHistoryDQMService_cfg_69912.py

#*** These will open the IOVs at the specified run number (note that the test uploads runs in an unordered list)
cmsRun test_SiStripHistoryDQMService_cfg_69912.py
cmsRun test_SiStripHistoryDQMService_cfg_69572.py
cmsRun test_SiStripHistoryDQMService_cfg_70416.py

#*** This is to check that an already uploaded run will be not uploaded anymore
cmsRun test_SiStripHistoryDQMService_cfg_69572.py

#*** Upload summaries for pixels
cmsRun test_SiPixelpHistoryDQMService_cfg_69912.py
 
#****************
#Create Trends
#***************

root test_Inspector.cc
