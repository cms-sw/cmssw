# optics_generator

## Running

Login to lxplus (CC7!) as totemprd.

```ssh totemprd@lxplus7.cern.ch```

Do not activate any CMSSW enviroment, but simply add paths to the shared libraries needed by `FindApproximation` exec:

```
export PATH=$PATH:/afs/cern.ch/user/m/mad/bin
. /afs/cern.ch/sw/lcg/app/releases/ROOT/6.06.06/x86_64-slc6-gcc48-opt/root/bin/thisroot.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cern.ch/sw/lcg/external/XercesC/3.1.1/x86_64-slc6-gcc47-opt/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cern.ch/sw/lcg/external/clhep/2.2.0.4/x86_64-slc6-gcc48-opt/lib
```

Go to the directory with your input files:
```
cd /afs/cern.ch/work/t/totemprd/totem/ctpps_optics_2016
```

`FindApproximation` exec  wasn't properly linked and looks for some libraries in directories relative to its
own location, so we temporarily copy them here
```
mkdir lib
cp /afs/cern.ch/work/t/totemprd/totem/ctpps_optics_2016/hubert/leszek/github/optics_generator-master/lib/libFit.so lib/
cp /afs/cern.ch/work/t/totemprd/totem/ctpps_optics_2016/hubert/leszek/github/optics_generator-master/stub/FitCint_rdict.pcm .
```

Finally we run it:
```
/afs/cern.ch/work/t/totemprd/totem/ctpps_optics_2016/hubert/leszek/github/optics_generator-master/bin/FindApproximation configuration_beam_1_ip_150.xml
```


## Compilation

TODO

```
cd /afs/cern.ch/work/t/totemprd/totem/ctpps_optics_2016/hubert/leszek/github/optics_generator-master
make clean
rootcint -f ./stub/FitCint.cc -c -I`root-config --incdir` -Iinterface -I. TMultiDimFet.h LHCOpticsApproximator.h TNtupleDcorr.h ./stub/FitCintLinkDef.h
make -j8
make exe
```