# Procedure to clone just enough of CMSSW to build mkFit for standalone runs.
# In "git clone" below, substitute the CMSSW branch you want to start from.

git clone --branch master --single-branch --no-checkout --reference /cvmfs/cms-ib.cern.ch/git/cms-sw/cmssw.git git@github.com:cms-sw/cmssw.git mkFitFromCMSSW
cd mkFitFromCMSSW
git config core.sparsecheckout true
echo -e "/.gitignore\n/.clang-tidy\n/.clang-format" > .git/info/sparse-checkout
echo -e "/RecoTracker/MkFit/\n/RecoTracker/MkFitCMS/\n/RecoTracker/MkFitCore/" >> .git/info/sparse-checkout
echo -e "/FWCore/Utilities/interface/" >> .git/info/sparse-checkout
git checkout  # enter detached-head state
./RecoTracker/MkFitCore/standalone/configure $PWD
source ./RecoTracker/MkFitCore/standalone/xeon_scripts/init-env.sh
unset INTEL_LICENSE_FILE
make -j 16 AVX_512:=1 WITH_ROOT=1

# To build with icc, do the above except for make, then:
# if [ -z ${INTEL_LICENSE_FILE+x} ]; then export INTEL_LICENSE_FILE=1; fi
# source /opt/intel/oneapi/compiler/latest/env/vars.sh
# source /opt/intel/oneapi/tbb/latest/env/vars.sh
# make -j 16 AVX_512:=1 WITH_ROOT=1
