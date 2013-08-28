all: Toy JPsiMuMu GEN

# test generating a JPsiMuMu fastsim sample, producing the fit tree and fitting it
JPsiMuMu: testTagProbeFitTreeAnalyzer_JPsiMuMu.root

JpsiMM_7TeV_cfi_py_GEN_FASTSIM.py:
	cmsDriver.py  JpsiMM_7TeV_cfi.py -s GEN,FASTSIM -n 1000 --pileup=NoPileUp --geometry DB --conditions auto:startup --datatier 'GEN-SIM-RECO' --eventcontent RECOSIM --no_exec

JpsiMM_7TeV_cfi_py_GEN_FASTSIM.root: JpsiMM_7TeV_cfi_py_GEN_FASTSIM.py
	cmsRun JpsiMM_7TeV_cfi_py_GEN_FASTSIM.py

testTagProbeFitTreeProducer_JPsiMuMu.root: JpsiMM_7TeV_cfi_py_GEN_FASTSIM.root testTagProbeFitTreeProducer_JPsiMuMu.py
	cmsRun testTagProbeFitTreeProducer_JPsiMuMu.py

testTagProbeFitTreeAnalyzer_JPsiMuMu.root: testTagProbeFitTreeProducer_JPsiMuMu.root testTagProbeFitTreeAnalyzer_JPsiMuMu.py
	cmsRun testTagProbeFitTreeAnalyzer_JPsiMuMu.py

# test generating a JPsiMuMu fastsim sample, producing the fit tree and fitting it
GEN: testTagProbeFitTreeAnalyzer_GEN.root

ZMM_7TeV_cfi_py_GEN.py:
	cmsDriver.py  ZMM_7TeV_cfi.py -s GEN -n 1000 --pileup=NoPileUp --geometry DB --conditions auto:startup --datatier 'GEN' --eventcontent RECOSIM --no_exec

ZMM_7TeV_cfi_py_GEN.root: ZMM_7TeV_cfi_py_GEN.py
	cmsRun ZMM_7TeV_cfi_py_GEN.py

testTagProbeFitTreeProducer_GEN.root: ZMM_7TeV_cfi_py_GEN.root testTagProbeFitTreeProducer_GEN.py
	cmsRun testTagProbeFitTreeProducer_GEN.py

testTagProbeFitTreeAnalyzer_GEN.root: testTagProbeFitTreeProducer_GEN.root testTagProbeFitTreeAnalyzer_GEN.py
	cmsRun testTagProbeFitTreeAnalyzer_GEN.py


# test generating a Toy experiment with known efficiency and fitting it
Toy: testTagProbeFitTreeAnalyzer_Toy.root

testTagProbeFitTreeProducer_Toy.root: testTagProbeFitTreeProducer_Toy.C
	root -q -l testTagProbeFitTreeProducer_Toy.C

testTagProbeFitTreeAnalyzer_Toy.root: testTagProbeFitTreeProducer_Toy.root testTagProbeFitTreeAnalyzer_Toy.py
	cmsRun testTagProbeFitTreeAnalyzer_Toy.py

