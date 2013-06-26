#!/bin/tcsh -f
#
# Script for running dijet ratio. results stored in dCache.
# Used by Manoj Jha.
#
# parameters
#
# $1 - bin number 
# $2 - etaInner
# $3 - etaOuter
# $4 - run number 

setenv SCRAM_ARCH slc3_ia32_gcc323
source /uscms/home/manoj/data/cshrc uaf

cd /uscms_data/d1/manoj/after-phd/diJet/cmssw/CMSSW_1_2_0/src
eval `scramv1 runtime -csh`

cd ${_CONDOR_SCRATCH_DIR}

set ranseed = ${1}${2}
echo "random seed: " $ranseed

cat > reco.cfg <<EOF
process myprocess =  {
#keep the logging output to a nice level

include "FWCore/MessageLogger/data/MessageLogger.cfi"

source = PoolSource {
untracked vstring fileNames = { 
"dcap://cmsgridftp.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/2/manoj/work/after-phd/QcdDijet/MCData/QcdBackgrd/mc_jetcor_120_qcd_pt_${1}.root"
}
untracked int32 maxEvents = -1
}
service = Tracer { untracked string indention = "$$"}

module myanalysis = MassAnalyzer {

# names of modules, producing object collections
string Mid5GenJets = "midPointCone5GenJets"
string Mid5CaloJets = "midPointCone5CaloJets"
string Mid5CorRecJets = "corJetMcone5"
string HepMcSrc  =  "VtxSmeared"
double v_etaInner = $2
double v_etaOuter = $3

# name of output root file with histograms
untracked string HistOutFile = "DiJetAnalysis.root"	

}

# module dump = EventContentAnalyzer {}

# path p = {dump}
path p = {myanalysis}

}
EOF

cmsRun  reco.cfg

ls -ltr | tail -4

set storage_dir = /pnfs/cms/WAX/2/${user}/work/after-phd/QcdDijet/Analysis/QcdBackgrd/etaOptimization

if (! -d ${storage_dir}/${2}_${3}) then
	mkdir ${storage_dir}/${2}_${3}
else
	echo "Directory ${storage_dir}/${2}_${3} exists"
endif

dccp DiJetAnalysis.root ${storage_dir}/${2}_${3}/QcdBackgrd_${1}_${4}.root
