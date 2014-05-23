analyze: tester.ps
	#gv tester.ps &

tester.ps: mc-results.tex booklet.tex tester.tex
	latex tester.tex
	latex tester.tex
	latex tester.tex
	latex tester.tex
	dvipdf tester.dvi tester.pdf
	rm -rf tester.dvi tester.aux tester.log texput.log tester.toc
	rm -rf mc-results.aux booklet.aux


mc-results.tex mc-results.root: prod1/mc-tester.root prod2/mc-tester.root
	root -b -q ANALYZE.C

booklet.tex: mc-results.root
	root -b -q BOOKLET.C


display:
	root DISPLAY.C

clean:
	rm -rf booklet
	rm -rf mc-results.root mc-results.tex
	rm -rf booklet.tex
	rm -rf tester.pdf tester.log tester.aux tester.dvi texput.log tester.toc mc-results.aux booklet.aux
	rm -rf MC-TESTER.DAT
Clean: clean
	rm -rf prod1/mc-tester.root
	rm -rf prod2/mc-tester.root
