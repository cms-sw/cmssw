for i in /data/dqm/calib/Test_CFEB02/run_00000001_Calib_CFEB_SCAPed_090730_154028.plots/*/*_DB.dat
do
cat $i >> pedSummary.dat
done
for i in /data/dqm/calib/Test_CFEB04/run_00000001_Calib_CFEB_Gains_090730_152454.plots/*/*_DB.dat
do
cat $i >> gainSummary.dat
done
for i in /data/dqm/calib/Test_CFEB02/run_00000001_Calib_CFEB_SCAPed_090730_154028.plots/*/*_DB_NoiseMatrix.dat
do
cat $i >> matrixSummary.dat
done
for i in /data/dqm/calib/Test_CFEB03/run_00000001_Calib_CFEB_CrossTalk_090730_153558.plots/*/*_DB_Xtalk.dat
do
cat $i >> xtalkSummary.dat
done
mv pedSummary.dat merged_data/pedSummary2009_07_30_run00000001.dat
mv gainSummary.dat merged_data/gainSummary2009_07_30_run00000001.dat
mv matrixSummary.dat merged_data/matrixSummary2009_07_30_run00000001.dat
mv xtalkSummary.dat merged_data/xtalkSummary2009_07_30_run00000001.dat
cd CMSSW_2_1_0_pre3/src/
g++ readGains.cpp -o readGains
g++ readMatrix.cpp -o readMatrix
g++ readPeds.cpp -o readPeds
g++ readXtalk.cpp -o readXtalk
g++ compareGains.cpp -o compareGains
g++ comparePeds.cpp -o comparePeds
g++ compareXtalk.cpp -o compareXtalk
g++ compareMatrix.cpp -o compareMatrix
