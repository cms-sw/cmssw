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
unset INTEL_LICENSE_FILE

# To build with gcc:
source ./RecoTracker/MkFitCore/standalone/xeon_scripts/init-env.sh
make -j 16 AVX_512:=1 WITH_ROOT=1

# To build with icpx, do this instead (note, WITH_ROOT doesn't work yet):
# source /opt/intel/oneapi/compiler/latest/env/vars.sh
# source /opt/intel/oneapi/tbb/latest/env/vars.sh
# make -j 16 AVX_512:=1 CXX=icpx

# To build with icc (obsolete), source the gcc AND icpx scripts above, then:
# if [ -z ${INTEL_LICENSE_FILE+x} ]; then export INTEL_LICENSE_FILE=1; fi
# make -j 16 AVX_512:=1 WITH_ROOT=1
