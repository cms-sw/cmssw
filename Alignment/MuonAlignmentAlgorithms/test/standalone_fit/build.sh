cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFrphiFitter.h $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/src/MuonResiduals6DOFrphiFitter.cc $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFFitter.h $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/src/MuonResiduals6DOFFitter.cc $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals5DOFFitter.h $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/src/MuonResiduals5DOFFitter.cc $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/src/MuonResidualsFitter.cc $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit/

cd $CMSSW_BASE/src/Alignment/MuonAlignmentAlgorithms/test/standalone_fit
make -j4
cd -
