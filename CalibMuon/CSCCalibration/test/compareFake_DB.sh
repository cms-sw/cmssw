#!/bin/tcsh

#if no output after running this shell script -> everything is OK!
#ignore croostalk errors at the moment
eval `scramv1 runt -csh`
#cmsRun CSCValidationFakes.cfg
cmsRun CSCDBCrosstalkPopCon.cfg
cmsRun CSCDBGainsPopCon.cfg
cmsRun CSCDBNoiseMatrixPopCon.cfg
cmsRun CSCDBPedestalsPopCon.cfg
cmsRun readDBCrosstalk.cfg
cmsRun readDBGains.cfg
cmsRun readDBNoiseMatrix.cfg
cmsRun readDBPedestals.cfg

mv dbgains.dat fakegains.dat
mv dbmatrix.dat fakematrix.dat
mv dbpeds.dat fakepeds.dat
mv dbxtalk.dat fakextalk.dat

cmsRun readDBCrosstalk.cfg
cmsRun readDBGains.cfg
cmsRun readDBNoiseMatrix.cfg
cmsRun readDBPedestals.cfg


#cmsRun readNewVectorConstants.cfg
g++ CSCValidation.cpp -o CSCValidation
./CSCValidation >! test
rm Fake*.db 
 
