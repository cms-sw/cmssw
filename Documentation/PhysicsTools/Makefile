all: document.pdf

document.dvi: document.tex Makefile memarticle.cls
	latex  document.tex
	latex  document.tex
	latex  document.tex

document.ps: document.dvi
	dvips  -o document.ps document.dvi

document.pdf: document.ps
	ps2pdf document.ps

clean:
	rm -f document.out document.toc document.aux document.log document.dvi document.ps document.tex.bak document.tex~ Makefile~

clobber: clean
	rm -f document.pdf


