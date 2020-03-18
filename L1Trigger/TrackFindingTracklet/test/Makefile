include Makefile.inc

default: fpga

fpga: fpga.o imath.a
	$(LD) -o $@ $^ $(LDFLAGS) $(LIBS)

fpga.o: ../interface/slhcevent.h ../interface/L1TStub.h ../interface/Constants.h ../interface/Sector.h  ../interface/Stub.h ../interface/FPGAWord.h ../interface/MemoryBase.h ../interface/ProcessBase.h ../interface/VMStubsMEMemory.h ../interface/VMStubsTEMemory.h ../interface/VMStubTE.h ../interface/TrackletProjectionsMemory.h ../interface/InverseTable.h ../interface/TrackletCalculator.h ../interface/TrackletProcessor.h ../interface/TrackletCalculatorDisplaced.h ../interface/MatchCalculator.h ../interface/MatchProcessor.h ../interface/TrackletEngine.h ../interface/TrackletEngineDisplaced.h ../interface/TripletEngine.h ../interface/MatchEngine.h ../interface/FitTrack.h ../interface/ProjectionRouter.h ../interface/InputLinkMemory.h ../plugins/FPGA.icc ../interface/TrackFitMemory.h ../interface/Tracklet.h ../interface/TrackletParametersMemory.h ../interface/Track.h ../interface/AllStubsMemory.h ../interface/StubPairsMemory.h ../interface/StubTripletsMemory.h ../interface/TrackDerTable.h ../interface/FullMatchMemory.h ../interface/TrackDer.h ../interface/VMProjectionsMemory.h ../interface/LayerProjection.h ../interface/LayerResidual.h ../interface/DTC.h ../interface/DTCLink.h ../interface/Util.h ../interface/PurgeDuplicate.h ../interface/CleanTrackMemory.h FPGATree.icc ../interface/TETableOuter.h ../interface/TETableInner.h ../interface/TETableOuterDisk.h ../interface/TETableInnerDisk.h ../interface/TETableInnerOverlap.h ../interface/TETableBase.h ../interface/DiskProjection.h ../interface/DiskResidual.h ../interface/VMRouterPhiCorrTable.h ../interface/CandidateMatchMemory.h ../interface/IMATH_TrackletCalculator.h ../interface/IMATH_TrackletCalculatorDisk.h ../interface/IMATH_TrackletCalculatorOverlap.h ../interface/AllProjectionsMemory.h ../interface/Cabling.h ../interface/VMRouter.h ../interface/StubVariance.h ../interface/imath.h ../plugins/WriteDesign.icc ../plugins/WriteInvTables.icc ../interface/GlobalHistTruth.h ../interface/HistBase.h ../interface/HistImp.h ../interface/TrackPars.h

imath.a: imath.o imath_calculate.o imath_HLS.o imath_Verilog.o
	$(AR) $(ARFLAGS) $@ $?

imath.o: ../src/imath.cc ../interface/imath.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

imath_calculate.o: ../src/imath_calculate.cc ../interface/imath.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

imath_HLS.o: ../src/imath_HLS.cc ../interface/imath.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

imath_Verilog.o: ../src/imath_Verilog.cc ../interface/imath.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: fpga

clean:
	@rm -f *.o
	@rm -f matching
	@rm -f main
	@rm -f fpga
	@rm -f *~
	@rm -f ../data/MemPrints/InputStubs/InputStubs*.dat
	@rm -f ../data/MemPrints/InputStubs/Link_*.dat
	@rm -f ../data/MemPrints/Stubs/AllStubs_AS_*.dat
	@rm -f ../data/MemPrints/Stubs/LayerStubs_SL*.dat
	@rm -f ../data/MemPrints/VMStubsTE/VMStubs_VMS*.dat
	@rm -f ../data/MemPrints/VMStubsME/VMStubs_VMS*.dat
	@rm -f ../data/MemPrints/FitTrack/TrackFit_TF_*.dat
	@rm -f ../data/MemPrints/CleanTrack/CleanTrack_CT_*.dat
	@rm -f ../data/MemPrints/Stubs/DiskStubs_SD*.dat
	@rm -f ../data/MemPrints/StubPairs/StubPairs_SP_*.dat
	@rm -f InvTable_TC_*.dat
	@rm -f FitDerTable_*.txt
	@rm -f TEBinTable*.txt
	@rm -f ../data/MemPrints/TrackletProjections/TrackletProjections_TPROJ_*.dat
	@rm -f ../data/MemPrints/TrackletProjections/AllProj_AP_*.dat
	@rm -f ../data/MemPrints/VMProjections/VMProjections_VMPROJ_*.dat
	@rm -f ../data/MemPrints/Matches/FullMatches_FM_*.dat
	@rm -f ../data/MemPrints/Matches/CandidateMatches_CM_*.dat
	@rm -f ../data/MemPrints/TrackletParameters/TrackletParameters_TPAR_*.dat
	@rm -f layeroccupancy.txt
	@rm -f VMTable*.txt
	@rm -f TE_*_*_*.txt
	@rm -f moduleoccupancy.txt
	@rm -f trackletcombinations.txt
	@rm -f trackletlayercombinations.txt
	@rm -f trackletlayers.txt
	@rm -f trackletparameters.txt
	@rm -f tracklets.txt
	@rm -f trackletpars.txt
	@rm -f trackletparsdisk.txt
	@rm -f trackproj.txt
	@rm -f diskmatch.txt
	@rm -f neighborproj.txt
	@rm -f vmfullmatch.txt
	@rm -f vmmatches.txt
	@rm -f vmstubs.txt
	@rm -f allprojections.txt
	@rm -f stubs.txt
	@rm -f stubsdisk.txt
	@rm -f stubslayer.txt
	@rm -f stubpairs.txt
	@rm -f stubpairsdisk.txt
	@rm -f trackletcands.txt
	@rm -f trackletslayer.txt
	@rm -f trackres.txt
	@rm -f vmprojections.txt
	@rm -f trackeff.txt
	@rm -f allstubs.txt
	@rm -f diskrouter.txt
	@rm -f fittrack.txt
	@rm -f fittrack1.txt
	@rm -f layerrouter.txt
	@rm -f matchcalculator.txt
	@rm -f projectioncounts.txt
	@rm -f alphadep.txt
	@rm -f diskmatch*.txt
	@rm -f dtclinks.txt
	@rm -f vmoccupancy.txt
	@rm -f vmoccupancyme.txt
	@rm -f vmoccupancyte.txt
	@rm -f layerresiduals.txt
	@rm -f diskresiduals.txt
	@rm -f chisq.txt
	@rm -f VMPhiCorrL*.txt
	@rm -f newvmoccupancy.txt
	@rm -f nmatches*.txt
	@rm -f hitpattern.txt
	@rm -f matchengine.txt
	@rm -f stubslayerpersector.txt
	@rm -f stubsdiskpersector.txt
	@rm -f z0_and_rinv.txt
	@rm -f trackletengine.txt
	@rm -f trackletcalculator.txt
	@rm -f matchtransceiver.txt
	@rm -f matchtransceiver_cc*
	@rm -f projectiontransceiver.txt
	@rm -f projectiontransceiver_cc*
	@rm -f trackletcalculator.txt
	@rm -f allstubste.txt
	@rm -f allstubsme.txt
	@rm -f variance.txt
	@rm -f hiteff.txt
	@rm -f dtcphirange.txt
	@rm -f inputlink.txt
	@rm -f unusedproj.txt
	@rm -f trackprojocc.txt
	@rm -f trackprojocc.dat
	@rm -f trackpars.txt
	@rm -f AutoDict_*.cxx
	@rm -f AutoDict_*cxx.d
	@rm -f AutoDict_*cxx.so
	@rm -f *_drinv.dat
	@rm -f *_tinv.dat
	@rm -f *_drinv.h
	@rm -f *_tinv.h
	@rm -f matcheff.txt
	@rm -f TC_*.v
	@rm -f TC_*.cpp
	@rm -f imath.root
	@rm -f *_cc.d
	@rm -f *_cc.so
	@rm -f *_dict_rdict.pcm
	@rm -f imath.a





