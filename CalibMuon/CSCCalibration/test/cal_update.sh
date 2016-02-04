#
#
#   Calibration Update Script
#
# Authors: Tom Nummy     Northeastern University
#          Oana Boeriu   Northeastern University 
#
# Description: This script's purpose is to work towards automation of the
#              generation, validation, and updating of calbration constants
#              for CSC gains, pedestals, crosstalk, and the noise matrix
#
#
#============================================================================

#### MAKE SURE YOU HAVE THE MOST RECENT DB VALUES FIRST!!! ####

#### Run readDBGains_cfg.py, readDBPedestals_cfg.py, readDBNoiseMatrix_cfg.py,
#### and readDBCrosstalk_cfg.py to be sure.

echo ==================================================================
echo "Beginning new Calibration Update"
echo ==================================================================
echo
sleep 3
echo "Checking the CFEB_04 run log"
echo 
sleep 2

#--- Update the directory list ----
ls -l -t -d /data/dqm/calib/Test_CFEB04/run*.plots | awk '{print $9}' > /nfshome0/nummy/Logs/CFEB_04/Test_CFEB04_dirlist.dat

#--- Compile and run the Log checker to find if there are any unprocessed runs in the directory ---
g++ /nfshome0/nummy/Logs/CFEB_04/LogChecker_CFEB04.cpp -o /nfshome0/nummy/Logs/CFEB_04/LC04
/nfshome0/nummy/Logs/CFEB_04/LC04

#--- Compile the C++ validator for later (outside loop b/c we only need to compile once) ---
g++ readGains.cpp -o readGains
chmod +x readGains

#echo
#echo "here's the job list:"
#cat /nfshome0/nummy/Logs/CFEB_04/tmpruns_04.dat

#--- This is the list of folders in the directory that haven't been processed yet (the job list) ---
#runqueue="/nfshome0/nummy/Logs/CFEB_04/Test_CFEB04_dirlist.dat"
runqueue="/nfshome0/nummy/Logs/CFEB_04/tmpruns_04.dat"

exec <$runqueue
i=0
#--- Looping over the job list --- 
while read CFEB4RUNDIR
  do
  echo
  echo ----------------------------------------------------------
  echo
  echo "Processing " $CFEB4RUNDIR
  i=`expr $i + 1`
  echo
  #--- get all the ascii files from all the chambers and put into a list for the upcoming loop ---
  CFEB4ASCIIFILES=`ls -l $CFEB4RUNDIR/*/*_DB.dat | awk '{print $9}'`
  echo
  
  if [ "$CFEB4ASCIIFILES" != "" ]
      then
      #--- merging the ascii files into one summary file ---
      for asciifile in $CFEB4ASCIIFILES
	do
	cat $asciifile >> /nfshome0/nummy/merged_data/gainSummary.dat
      done
      echo "merged ASCII files"
      echo
  fi
  
  #--- Run the C++ validator ---
  sleep 2
  echo "Running gains validator, Please wait..."
  ./readGains
  sleep 2
  echo "...complete"
  echo

  #--- Until CMSSW is installed on the CSC-DQM machine we have to run all CMSSW programs on cmsusr0 ---  #ssh cmsusr0 ./getGainsDB.sh
  #mv dbgains.dat DBValues/
  
  #--- Run the CMSSW comparison analyzer ---
  cd /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test
  source /nfshome0/cmssw2/scripts/setup.sh
  eval `scramv1 runt -sh`

  echo "beginning CMSSW analyser, this may take a few moments"
  cmsRun gains_compare_cfg.py
  echo
  echo "Comparison complete!"
  echo
  # files need to be moved to correct path
  cp /nfshome0/nummy/Good_data/goodGains.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/gains.dat
  cp /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/DBValues/dbgains.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/old_dbgains.dat

  #--- Generate SQlite Files ---
  #cmsRun CSCDBGainsPopCon_cfg.py
  cd ~
  
  
  #---Some housekeeping, making folders by run number and moving summary and C++ ouput data there ---
  newfoldername=`echo $CFEB4RUNDIR | awk '{ n=split($1,path,"/"); print path[n] }'`
  if [ -d /nfshome0/nummy/Good_data/$newfoldername ]
      then
      echo "The directory for the processed data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/Good_data/$newfoldername
  fi 
  
  if [ -d /nfshome0/nummy/merged_data/$newfoldername ]
      then
      echo "The directory for the merged data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/merged_data/$newfoldername
  fi 
  
  if [ -d /nfshome0/nummy/Diff_Output/$newfoldername ]
      then
      echo "The directory for the diff output already exists, overwriting previous entries"
      echo
  else
      mkdir /nfshome0/nummy/Diff_Output/$newfoldername
  fi
  
  #--- write buffers to the corresponding run folders
  cp /nfshome0/nummy/merged_data/gainSummary.dat /nfshome0/nummy/merged_data/$newfoldername/
  cp /nfshome0/nummy/Good_data/goodGains.dat /nfshome0/nummy/Good_data/$newfoldername/
  cp /nfshome0/nummy/Diff_Output/diffGains.dat /nfshome0/nummy/Diff_Output/$newfoldername/

  #--- clear the buffer for the next run ---
  echo "" > /nfshome0/nummy/merged_data/gainSummary.dat
  
done

echo "Processed " $i " gain calibration runs"
sleep 3

echo
echo ==================================================
echo "Getting ready to process Crosstalk runs"
echo ==================================================
echo
sleep 2
echo "Checking the CFEB_03 run log"
echo
sleep 2

#--- Update the directory list ----
ls -l -t -d /data/dqm/calib/Test_CFEB03/run*.plots | awk '{print $9}' > /nfshome0/nummy/Logs/CFEB_03/Test_CFEB03_dirlist.dat

#--- Compile and run the Log checker to find if there are any unprocessed runs in the directory ---
g++ /nfshome0/nummy/Logs/CFEB_03/LogChecker_CFEB03.cpp -o /nfshome0/nummy/Logs/CFEB_03/LC03
/nfshome0/nummy/Logs/CFEB_03/LC03

#--- Compile the C++ validator for later (outside loop b/c we only need to compile once) ---
g++ readXtalk.cpp -o readXtalk
chmod +x readXtalk

#echo
#echo "here's the job list:"
#cat /nfshome0/nummy/Logs/CFEB_03/tmpruns_03.dat

#--- This is the list of folders in the directory that haven't been processed yet (the job list) ---
#runqueue="/nfshome0/nummy/Logs/CFEB_03/Test_CFEB03_dirlist.dat"
runqueue="/nfshome0/nummy/Logs/CFEB_03/tmpruns_03.dat"

exec <$runqueue
i=0
#--- Looping over the job list --- 
while read CFEB3RUNDIR
  do
  echo
  echo ----------------------------------------------------------
  echo
  echo "Processing " $CFEB3RUNDIR
  i=`expr $i + 1`
  echo
  #--- get all the ascii files from all the chambers and put into a list for the upcoming loop ---
  CFEB3ASCIIFILES=`ls -l $CFEB3RUNDIR/*/*Xtalk.dat | awk '{print $9}'`
  echo
  
  if [ "$CFEB3ASCIIFILES" != "" ]
      then
      echo "Found the crosstalk ascii files"
      #--- merging the ascii files into one summary file ---
      for asciifile in $CFEB3ASCIIFILES
	do
	cat $asciifile >> /nfshome0/nummy/merged_data/xtalkSummary.dat
      done
  fi
  echo "merged ASCII files"
  echo
  
  #--- Run the C++ validator ---
  sleep 2
  echo "Running XTalk validator, Please wait..."
  ./readXtalk
  sleep 2
  echo "...complete"
  echo

  #--- Run the CMSSW comparison analyzer ---
  cd /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test
  source /nfshome0/cmssw2/scripts/setup.sh
  eval `scramv1 runt -sh`

  echo "beginning CMSSW analyser, this may take a few moments"
  cmsRun xtalk_compare_cfg.py
  echo
  echo "Comparison complete!"
  echo
  # move files to correct paths
  cp /nfshome0/nummy/Good_data/goodXtalk.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/xtalk.dat
  cp /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/DBValues/dbxtalk.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/old_dbxtalk.dat

  #--- Generate SQlite Files ---
  #cmsRun CSCDBCrosstalkPopCon_cfg.py
  cd ~
  
  #---Some housekeeping, making folders by run number and moving summary and C++ ouput data there ---
  newfoldername=`echo $CFEB3RUNDIR | awk '{ n=split($1,path,"/"); print path[n] }'`
  if [ -d /nfshome0/nummy/Good_data/$newfoldername ]
      then
      echo "The directory for the processed data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/Good_data/$newfoldername
  fi 
  
  if [ -d /nfshome0/nummy/merged_data/$newfoldername ]
      then
      echo "The directory for the merged data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/merged_data/$newfoldername
  fi 
  
  if [ -d /nfshome0/nummy/Diff_Output/$newfoldername ]
      then
      echo "The directory for the diff output data already exists, overwriting previous entries"
      echo
  else
      mkdir /nfshome0/nummy/Diff_Output/$newfoldername
  fi
  
  cp /nfshome0/nummy/merged_data/xtalkSummary.dat /nfshome0/nummy/merged_data/$newfoldername/
  cp /nfshome0/nummy/Good_data/goodXtalk.dat /nfshome0/nummy/Good_data/$newfoldername/
  cp /nfshome0/nummy/Diff_Output/diffXtalk.dat /nfshome0/nummy/Diff_Output/$newfoldername/
  
  echo "" > /nfshome0/nummy/merged_data/xtalkSummary.dat
  
done

echo "Processed " $i " crosstalk calibration runs"
sleep 2

echo
echo ==================================================
echo "Getting ready to process Pedestal/Noise Matrix runs"
echo ==================================================
echo
sleep 3
echo "Checking the CFEB_02 log"
echo
sleep 2

#--- Update the directory list ----
ls -l -t -d /data/dqm/calib/Test_CFEB02/run*.plots | awk '{print $9}' > /nfshome0/nummy/Logs/CFEB_02/Test_CFEB02_dirlist.dat

#--- Compile and run the Log checker to find if there are any unprocessed runs in the directory ---
g++ /nfshome0/nummy/Logs/CFEB_02/LogChecker_CFEB02.cpp -o /nfshome0/nummy/Logs/CFEB_02/LC02
/nfshome0/nummy/Logs/CFEB_02/LC02

#--- Compile the C++ validator for later (outside loop b/c we only need to compile once) ---
g++ readPeds.cpp -o readPeds
chmod +x readPeds

g++ readMatrix.cpp -o readMatrix
chmod +x readMatrix

#echo
#echo "here's the job list:"
#cat /nfshome0/nummy/Logs/CFEB_02/tmpruns_02.dat

#--- This is the list of folders in the directory that haven't been processed yet (the job list) ---
#runqueue="/nfshome0/nummy/Logs/CFEB_02/Test_CFEB02_dirlist.dat"
runqueue="/nfshome0/nummy/Logs/CFEB_02/tmpruns_02.dat"

exec <$runqueue
i=0
#--- Looping over the job list --- 
while read CFEB2RUNDIR
  do
  echo
  echo ----------------------------------------------------------
  echo
  echo "Processing " $CFEB2RUNDIR
  i=`expr $i + 1`
  echo
  #--- get all the ascii files from all the chambers and put into a list for the upcoming loop ---
  CFEB2ASCIIFILES=`ls -l $CFEB2RUNDIR/*/*_DB.dat | awk '{print $9}'`
  echo
  
  if [ "$CFEB2ASCIIFILES" != "" ]
      then
      #--- merging the ascii files into one summary file ---
      for asciifile in $CFEB2ASCIIFILES
	do
	cat $asciifile >> /nfshome0/nummy/merged_data/pedSummary.dat
      done
  fi
  echo "merged pedestal ASCII files"
  echo
  
  #--- Run the C++ validator ---
  sleep 2
  echo "Running Pedestal validator, Please wait..."
  ./readPeds
  sleep 2
  echo "...complete"
  echo
  
  echo "goodPeds.dat:"
  head /nfshome0/nummy/Good_data/goodPeds.dat
  echo
  echo "dbpeds.dat:"
  head /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/DBValues/dbpeds.dat
  echo

   #--- Run the CMSSW comparison analyzer ---
  cd /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test
  source /nfshome0/cmssw2/scripts/setup.sh
  eval `scramv1 runt -sh`

  echo "beginning CMSSW analyser, this may take a few moments"
  cmsRun peds_compare_cfg.py
  echo
  echo "Comparison complete!"
  echo
   # files need to be moved to correct path
  cp /nfshome0/nummy/Good_data/goodPeds.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/peds.dat
  cp /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/DBValues/dbpeds.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/old_dbpeds.dat

  #--- Generate SQlite Files ---
  #cmsRun CSCDBPedestalsPopCon_cfg.py
  cd ~
  
  #---Some housekeeping, making folders by run number and moving summary and C++ ouput data there ---
  newfoldername=`echo $CFEB2RUNDIR | awk '{ n=split($1,path,"/"); print path[n] }'`
  if [ -d /nfshome0/nummy/Good_data/$newfoldername ]
      then
      echo "The directory for the processed data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/Good_data/$newfoldername
  fi 
  
  if [ -d /nfshome0/nummy/merged_data/$newfoldername ]
      then
      echo "The directory for the merged data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/merged_data/$newfoldername
  fi 
  
  if [ -d /nfshome0/nummy/Diff_Output/$newfoldername ]
      then
      echo "The directory for the diff output  already exists, overwriting previous entries"
      echo
  else
      mkdir /nfshome0/nummy/Diff_Output/$newfoldername
  fi

  cp /nfshome0/nummy/merged_data/pedSummary.dat /nfshome0/nummy/merged_data/$newfoldername/
  cp /nfshome0/nummy/Good_data/goodPeds.dat /nfshome0/nummy/Good_data/$newfoldername/
  cp /nfshome0/nummy/Diff_Output/diffPeds.dat /nfshome0/nummy/Diff_Output/$newfoldername/

  echo "" > /nfshome0/nummy/merged_data/pedSummary.dat
  
done

echo "Processed " $i " Pedestal calibration runs"
sleep 2

#----------------------------------------------------------------

exec <$runqueue
i=0
#--- Looping over the job list --- 
while read CFEB2RUNDIR
  do
  echo
  echo ----------------------------------------------------------
  echo
  echo "Processing " $CFEB2RUNDIR
  i=`expr $i + 1`
  echo
  #--- get all the ascii files from all the chambers and put into a list for the upcoming loop ---
  CFEB2ASCIIFILES=`ls -l $CFEB2RUNDIR/*/*_DB_NoiseMatrix.dat | awk '{print $9}'`
  echo
  
  if [ "$CFEB2ASCIIFILES" != "" ]
      then
      #--- merging the ascii files into one summary file ---
      for asciifile in $CFEB2ASCIIFILES
	do
	cat $asciifile >> /nfshome0/nummy/merged_data/matrixSummary.dat
      done
  fi
  echo "merged noise matrix ASCII files"
  echo
  
  #--- Run the C++ validator ---
  sleep 2
  echo "Running Noise Matrix validator, Please wait..."
  ./readMatrix
  sleep 2
  echo "...complete"
  echo
  
  echo "goodMatrix.dat:"
  head /nfshome0/nummy/Good_data/goodMatrix.dat
  echo
  echo "dbmatrix.dat:"
  head /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/DBValues/dbmatrix.dat
  echo
  
  #--- Run the CMSSW comparison analyzer ---
  cd /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test
  source /nfshome0/cmssw2/scripts/setup.sh
  eval `scramv1 runt -sh`

  echo "beginning CMSSW analyser, this may take a few moments"
  cmsRun noisematrix_compare_cfg.py
  echo
  echo "Comparison complete!"
  echo
   # files need to be moved to correct path
  cp /nfshome0/nummy/Good_data/goodMatrix.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/matrix.dat
  cp /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/DBValues/dbmatrix.dat /nfshome0/nummy/CMSSW_3_2_5/src/CalibMuon/CSCCalibration/test/old_dbmatrix.dat

  #--- Generate SQlite Files ---
  #cmsRun CSCDBNoiseMatrixPopCon_cfg.py
  cd ~


  #---Some housekeeping, making folders by run number and moving summary, C++, and Diff ouput data there ---
  newfoldername=`echo $CFEB2RUNDIR | awk '{ n=split($1,path,"/"); print path[n] }'`
  if [ -d /nfshome0/nummy/Good_data/$newfoldername ]
      then
      echo "The directory for the processed data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/Good_data/$newfoldername
  fi 
  
  if [ -d /nfshome0/nummy/merged_data/$newfoldername ]
      then
      echo "The directory for the merged data already exists, overwriting previous entries"
  else
      mkdir /nfshome0/nummy/merged_data/$newfoldername
  fi 

  if [ -d /nfshome0/nummy/Diff_Output/$newfoldername ]
      then
      echo "The directory for the diff output  already exists, overwriting previous entries"
      echo
  else
      mkdir /nfshome0/nummy/Diff_Output/$newfoldername
  fi
  
  cp /nfshome0/nummy/merged_data/matrixSummary.dat /nfshome0/nummy/merged_data/$newfoldername/
  cp /nfshome0/nummy/Good_data/goodMatrix.dat /nfshome0/nummy/Good_data/$newfoldername/
  cp /nfshome0/nummy/Diff_Output/diffMatrix.dat /nfshome0/nummy/Diff_Output/$newfoldername/

  echo "" > /nfshome0/nummy/merged_data/matrixSummary.dat
  
done

echo "Processed " $i " noise matrix calibration runs"
sleep 2

echo
echo "Finished Calibration update.."
echo
sleep 2
#cat einstein.dat
