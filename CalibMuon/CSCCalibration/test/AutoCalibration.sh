#!/bin/sh

echo Analyzing for run: $1

       ##########################################################################
       ######      CSC Automated Calibration  -   Revision 11-24-2010      ######
       ######      Darin Baumgartel, Northeastern University               ######
       ##########################################################################

echo " "
echo "                  ***************************************************"
echo "                  ****     Beginning new Calibration Update      ****"
echo "                  ***************************************************"; echo " "

# Date label
NOW=$(date +"%b_%d_%Y__%R_%S")
echo $NOW
######################################################################################
###        MERGE THE GAINS, PEDS, XTALK, MATRIX VALUES INTO SUMMARY FILES           ### 
#######################################################################################

# Replace the run numbes with your desired run numbers (things like "run_######_Calib_CFEB_[----].plots" directories)

echo " ";echo "                  ****     Creating Summary Files:";echo " "
# CFEB02 - SCAPed is used for both noise-matrix (*_DB_NoiseMatrix.dat files) and pedestals (*_DB.dat files)
matrixname="matrixSummary_$NOW.dat"
for i in /nfshome0/cscdqm/results/calib/Test_CFEB02/run_$1_Calib_CFEB_SCAPed.plots/*/*_DB_NoiseMatrix.dat
do
cat $i >> $matrixname
done
echo "CFEB02: NoiseMatrix Summary File:      $matrixname    has been produced."; echo " "

pedsname="pedSummary_$NOW.dat"
for i in /nfshome0/cscdqm/results/calib/Test_CFEB02/run_$1_Calib_CFEB_SCAPed.plots/*/*_DB.dat
do
cat $i >> $pedsname
done
echo "CFEB02: Pedestals Summary File:        $pedsname       has been produced."; echo " "

# CFEB03 - Crosstalk is used for Crosstalk (*_DB_XTalk.dat files) 
xtalkname="xtalkSummary_$NOW.dat"
for i in /nfshome0/cscdqm/results/calib/Test_CFEB03/run_$1_Calib_CFEB_CrossTalk.plots/*/*_DB_Xtalk.dat
do
cat $i >> $xtalkname
done
echo "CFEB03: Crosstalk Summary File:        $xtalkname     has been produced."; echo " "

# CFEB04 - Gains is used for Gains (*_DB.dat files)
gainsname="gainSummary_$NOW.dat"
for i in /nfshome0/cscdqm/results/calib/Test_CFEB04/run_$1_Calib_CFEB_Gains.plots/*/*.dat
do
cat $i >> $gainsname
done
echo "CFEB04: Gains Summary File:            $gainsname      has been produced."; echo " "

#######################################################################################
###        Copy the read[----].cpp template files to date-labeled files and           ### 
###        filenames (replacing the string "FileName" with the names above)          ### 
#######################################################################################

echo " ";echo "                  ****     The following \"read\" files have been produced from templates:";echo " "
cat readMatrix.cpp | sed -e 's/FileName/'$matrixname'/' > readMatrix_$NOW.cpp; echo "readMatrix_$NOW.cpp"
cat readPeds.cpp | sed -e 's/FileName/'$pedsname'/' > readPeds_$NOW.cpp; echo "readPeds_$NOW.cpp"
cat readXtalk.cpp | sed -e 's/FileName/'$xtalkname'/' > readXtalk_$NOW.cpp; echo "readXtalk_$NOW.cpp"
cat readGains.cpp | sed -e 's/FileName/'$gainsname'/' > readGains_$NOW.cpp; echo "readGains_$NOW.cpp"; echo " "


#######################################################################################
###        Run the read[----].cpp files                                              ###
#######################################################################################

echo " ";echo "                  ****     Running read____.cpp files. Succesful completetion:";echo " "
echo "             READ___.CPP FILE:                           INPUT_FILE                    -->                       OUTPUT_FILE";echo " "

g++ readMatrix_$NOW.cpp -o readMatrix_$NOW  
./readMatrix_$NOW
echo "readMatrix_$NOW.cpp:      $matrixname     -->     GoodVals_$matrixname";

g++ readPeds_$NOW.cpp -o readPeds_$NOW  
./readPeds_$NOW
echo "readPeds_$NOW.cpp:        $pedsname        -->     GoodVals_$pedsname";

g++ readXtalk_$NOW.cpp -o readXtalk_$NOW  
./readXtalk_$NOW
echo "readXtalk_$NOW.cpp:       $xtalkname      -->     GoodVals_$xtalkname";

g++ readGains_$NOW.cpp -o readGains_$NOW  
./readGains_$NOW
echo "readGains_$NOW.cpp:       $gainsname       -->     GoodVals_$gainsname";

echo " "

#######################################################################################
###    Read DataBase Values, designate them as "old_db[---].dat" for comparison     ###
#######################################################################################

echo " ";echo " ";
echo "          ---------------------------------------------------------"
echo "          -----    RETRIEVING DATABASE VALUES AND STORING     -----"
echo "          ---------------------------------------------------------"
echo " ";echo "         >>>>>>>>>>>>>> Retrieve Crosstalk from DataBase <<<<<<<<<<<<<<";echo " ";cmsRun readDBCrosstalk_cfg.py 
echo " ";echo "         >>>>>>>>>>>>>> Retrieve Matrix from DataBase <<<<<<<<<<<<<<";echo " ";cmsRun readDBNoiseMatrix_cfg.py
echo " ";echo "         >>>>>>>>>>>>>> Retrieve Gains from DataBase <<<<<<<<<<<<<<";echo " ";cmsRun readDBGains_cfg.py 
echo " ";echo "         >>>>>>>>>>>>>> Retrieve Pedestals from DataBase <<<<<<<<<<<<<<";echo " ";cmsRun readDBPedestals_cfg.py; echo " "
echo " ";echo " ";
echo "          ---------------------------------------------------------"
echo "          -----          DATABASE VALUES RETRIEVED            -----"
echo "          ---------------------------------------------------------"

echo " ";echo " ";echo "                  ****     Reading Database Values: ";echo " "
rm old_dbxtalk.dat;rm old_dbmatrix.dat;rm old_dbgains.dat;rm old_dbpeds.dat
mv dbxtalk.dat old_dbxtalk.dat;                      echo "Crosstalk     -- DataBase file:   old_dbxtalk.dat    has been produced.";
mv dbmatrix.dat old_dbmatrix.dat;                    echo "Noise Matrix  -- DataBase file:   old_dbmatrix.dat   has been produced.";
mv dbgains.dat old_dbgains.dat;                      echo "Gains         -- DataBase file:   old_dbgains.dat    has been produced.";
mv dbpeds.dat old_dbpeds.dat;                        echo "Pedestals     -- DataBase file:   old_dbpeds.dat     has been produced."; echo " "

# Here I remove unnecessary characters in the matrix file not suitable for comparison with the merged file...
mv old_dbmatrix.dat old_dbmatrix_orig.dat
cat old_dbmatrix_orig.dat | sed -e 's/E:[0-9][0-9]//;s/E:[0-9]//;s/S:[0-9][0-9]//;s/S:[0-9]//;s/R:[0-9][0-9]//;s/R:[0-9]//;s/C:[0-9][0-9]//;s/C:[0-9]//;s/L:[0-9][0-9]//;s/L:[0-9]//;s/chan [0-9][0-9]//;s/chan [0-9]//;s/          /  /' > old_dbmatrix.dat
rm old_dbmatrix_orig.dat

#######################################################################################
###      Save values into SQLite File                                               ###
#######################################################################################

echo " ";echo " ";
echo "          ---------------------------------------------------------"
echo "          -----             Creating SQLite Files             -----"
echo "          ---------------------------------------------------------"

mv GoodVals_$gainsname gains.dat; mv GoodVals_$pedsname peds.dat; mv GoodVals_$matrixname matrix.dat; mv GoodVals_$xtalkname xtalk.dat;
cmsRun CSCDBCrosstalkPopCon_cfg.py; cmsRun CSCDBGainsPopCon_cfg.py; cmsRun CSCDBNoiseMatrixPopCon_cfg.py; cmsRun CSCDBPedestalsPopCon_cfg.py;
mv gains.dat GoodVals_$gainsname; mv peds.dat GoodVals_$pedsname; mv matrix.dat GoodVals_$matrixname; mv xtalk.dat GoodVals_$xtalkname;
mv DBCrossTalk.db DBCrossTalk_$NOW.db; mv DBGains.db DBGains_$NOW.db; mv DBNoiseMatrix.db DBNoiseMatrix_$NOW.db; mv DBPedestals.db DBPedestals_$NOW.db;
echo " "; echo " "; echo " SQLite File Creation Complete"

#######################################################################################
###       Create and Run Python Comparison Module                                   ###
#######################################################################################
echo " ";echo " ";
echo "          ---------------------------------------------------------"
echo "          -----    Running Comparison With Python Modules     -----"
echo "          ---------------------------------------------------------"

rm stubs/Compare.cc;
cat Compare_template.txt | sed -e 's/Matrix_FileName/'GoodVals_$matrixname'/;s/Peds_FileName/'GoodVals_$pedsname'/;s/Gains_FileName/'GoodVals_$gainsname'/;s/Xtalk_FileName/'GoodVals_$xtalkname'/' > stubs/Compare.cc;
cd ../../../
scramv1 b -j 4
cd -
echo "              "
echo "                    stubs/Compare.C File Created"; echo " "
 echo        "THIS MAY TAKE SEVERAL MINUTES, GO HAVE A CUP OF COFFEE"
echo " "; echo "                    Running Comparisons... ";

cmsRun compare_cfg.py

#######################################################################################
###      Load Results into Root NTuple                                              ###
#######################################################################################

echo " ";echo " ";
echo "          ---------------------------------------------------------"
echo "          -----      Creating/Running Root-NTuple Scripts     -----"
echo "          ---------------------------------------------------------"


cat diffGainsNtuple.C | sed -e 's/Gains_FileName/'GoodVals_$gainsname'/;s/RootFile/'diffGains_$NOW.root'/' >  "diffGainsNtuple_$NOW.C"
cat diffPedsNtuple.C | sed -e 's/Peds_FileName/'GoodVals_$pedsname'/;s/RootFile/'diffPeds_$NOW.root'/' >  "diffPedsNtuple_$NOW.C"
cat diffMatrixNtuple.C | sed -e 's/Matrix_FileName/'GoodVals_$matrixname'/;s/RootFile/'diffMatrix_$NOW.root'/' >  "diffMatrixNtuple_$NOW.C"
cat diffXtalkNtuple.C | sed -e 's/Xtalk_FileName/'GoodVals_$xtalkname'/;s/RootFile/'diffXtalk_$NOW.root'/' >  "diffXtalkNtuple_$NOW.C"

echo "               Scripts Produced:"; echo " "
echo "diffGainsNtuple_$NOW.C"; echo "diffPedsNtuple_$NOW.C";echo "diffMatrixNtuple_$NOW.C";echo "diffXtalkNtuple_$NOW.C"; echo " ";echo " "
echo "               Running Scripts:"; echo " "
 
rm RootProcesses
echo "{gROOT->ProcessLine(\"gROOT->Reset()\"); gROOT->ProcessLine(\".x diffGainsNtuple_$NOW.C\"); gROOT->ProcessLine(\"gROOT->Reset()\"); gROOT->ProcessLine(\".x diffPedsNtuple_$NOW.C\"); gROOT->ProcessLine(\"gROOT->Reset()\"); gROOT->ProcessLine(\".x diffMatrixNtuple_$NOW.C\"); gROOT->ProcessLine(\"gROOT->Reset()\"); gROOT->ProcessLine(\".x diffXtalkNtuple_$NOW.C\"); gROOT->ProcessLine(\".q\");}" >RootProcesses
root -l RootProcesses

echo " "; echo " ";echo "Root N-tuple production complete. "


#######################################################################################
###      Store Results                                                              ###
#######################################################################################

echo " ";echo " ";
echo "          ---------------------------------------------------------"
echo "          -----      Transferring Results to Dated Folder     -----"
echo "          ---------------------------------------------------------"

mkdir "Calibration_$NOW"
mv *$NOW* Calibration_$NOW
mv old_db*dat Calibration_$NOW
echo "          Results/Files stored in folder Calibration_$NOW"

echo " "
echo "                  ***************************************************"
echo "                  ****        Calibration Run is Complete        ****"
echo "                  ***************************************************"; echo " "






