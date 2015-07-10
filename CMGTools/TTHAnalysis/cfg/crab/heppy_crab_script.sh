# extract exported necessary stuff
tar xzf cmgdataset.tar.gz --directory $HOME
tar xzf python.tar.gz --directory $CMSSW_BASE
tar xzf cafpython.tar.gz --directory $CMSSW_BASE

# uncomment for debuging purposes
#ls -lR .
#echo "ARGS:"
#echo $@
#echo "ENV..................................."
#env
#echo "VOMS"
#voms-proxy-info -all
#echo "CMSSW BASE, python path, pwd, home"
#echo $CMSSW_BASE
#echo $PYTHONPATH
#echo $PWD
#echo $HOME
#echo $ROOTSYS

# copy auxiliarity data to the right place (json, pu, lep eff, jet corr, ...)
cp -r lib/* $CMSSW_BASE/lib/
mkdir $CMSSW_BASE/src/CMGTools/
cp -r src/CMGTools/* $CMSSW_BASE/src/CMGTools/
for i in `find src/ -name data -type d`
do
    echo $i
    mkdir -p  $CMSSW_BASE/$i
    cp -r $i/* $CMSSW_BASE/$i
done

#ls -lR 

python heppy_crab_script.py $@
