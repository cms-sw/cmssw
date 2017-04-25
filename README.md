## Build

Initialise environment for `CMSSW_9_1_0_pre2`, go to the base directory of this project and do:
```
make clean
rootcint -f ./stub/FitCint.cc -c -I`root-config --incdir` -Iinterface -I. TMultiDimFet.h LHCOpticsApproximator.h TNtupleDcorr.h ./stub/FitCintLinkDef.h
make -j8
make exe
```
