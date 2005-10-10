############################################################
#
# GNUmakefile for generation of PDF document from LaTeX source.
#
# This is not a very clever makefile; if you have an installation of
# MiKTeX, use 'makefile' instead
#
# $Id: miktex.mk,v 1.2 2005/05/09 13:25:20 paterno Exp $
#
#
############################################################

all: view

view: RoadMap.pdf
	cygstart $<

RoadMap.dvi: RoadMap.tex GNUmakefile memarticle.cls
	texify --silent RoadMap.tex

#RoadMap.ps: RoadMap.dvi
#	dvips  -o RoadMap.ps -Ppdf RoadMap.dvi

#RoadMap.pdf: RoadMap.ps
#	ps2pdf RoadMap.ps

RoadMap.pdf: RoadMap.dvi
	dvipdfm RoadMap.dvi

clean:
	rm -f RoadMap.out RoadMap.toc RoadMap.qst RoadMap.aux RoadMap.log RoadMap.dvi RoadMap.ps RoadMap.tex.bak RoadMap.tex~ GNUmakefile~

clobber: clean
	rm -f RoadMap.pdf


