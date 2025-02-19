#merge all summary tables of each test in a single ASCII file
./merge_peds.sh
./merge_gains.sh
./merge_matrix.sh
./merge_xtalk.sh

#compile the validation files and execute + copy to new names.dat
g++ readGains.cpp -o readGains
./readGains
cp goodGains.dat gains.dat
g++ readMatrix.cpp -o readMatrix
./readMatrix
cp goodMatrix.dat matrix.dat
g++ readPeds.cpp -o readPeds 
./readPeds
cp goodPeds.dat peds.dat
g++ readXtalk.cpp -o readXtalk
./readXtalk
cp goodXtalk.dat xtalk.dat

#read what is in DB already
cmsRun readDBCrosstalk_cfg.py
cmsRun readDBGains_cfg.py
cmsRun readDBNoiseMatrix_cfg.py
cmsRun readDBPedestals_cfg.py

#replace with new data and transfer to DB
cmsRun CSCDBGainsPopCon_cfg.py
cmsRun CSCDBNoiseMatrixPopCon_cfg.py
cmsRun CSCDBPedestalsPopCon_cfg.py
cmsRun CSCDBCrosstalkPopCon_cfg.py
