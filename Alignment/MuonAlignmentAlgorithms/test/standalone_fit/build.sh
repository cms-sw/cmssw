cp Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFrphiFitter.h Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp Alignment/MuonAlignmentAlgorithms/src/MuonResiduals6DOFrphiFitter.cc Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals6DOFFitter.h  Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp Alignment/MuonAlignmentAlgorithms/src/MuonResiduals6DOFFitter.cc  Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp Alignment/MuonAlignmentAlgorithms/interface/MuonResiduals5DOFFitter.h  Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp Alignment/MuonAlignmentAlgorithms/src/MuonResiduals5DOFFitter.cc  Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp Alignment/MuonAlignmentAlgorithms/src/MuonResidualsFitter.cc  Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cp Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h  Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
#cp $ROOTSYS/etc/Makefile.arch Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
cd Alignment/MuonAlignmentAlgorithms/test/standalone_fit/
make -j4
cd -

