echo "================== START OF HEPPY CRAB SCRIPT========================="
echo "Unpacking libs"
rm -rf $CMSSW_BASE/lib/
rm -rf $CMSSW_BASE/src/
rm -rf $CMSSW_BASE/module/
rm -rf $CMSSW_BASE/python/
mv lib $CMSSW_BASE/lib
mv src $CMSSW_BASE/src
mv module $CMSSW_BASE/module
mv python $CMSSW_BASE/python
echo "Running of Heppy"
python heppy_crab_script.py $1
echo "============= END OF HEPPY CRAB SCRIPT ========================="
