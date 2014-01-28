# $Id: Makefile,v 1.1 2008/05/26 08:31:19 loizides Exp $
#############################


TEXSDIR   = ./
PICSDIR   = ./ ./Software ./Hardware
FILENAME  = smreviewdoc2

###

TEXSRCS   = $(shell find $(TEXSDIR) -maxdepth 2 -iname "*.tex")
PICS      = $(shell find $(PICSDIR) -maxdepth 1 -iname "*.eps")

ADDFILES  = Makefile 

BACKFILES = $(PICS) $(TEXSRCS) cmspaper.cls

ARCHFILES = $(TEXSRCS) $(FILENAME).bbl hypernat.sty ${PICS}

PDFPICS   = $(patsubst %.eps,%.pdf,$(PICS))

LATEX	  = latex
PDFTEX    = pdflatex
BIBTEX	  = bibtex
DVIPS	  = dvips
PS2PDF    = ps2pdf
EPS2EPS   = eps2eps
EPSTOPDF  = epstopdf
THUMBPDF  = thumbpdf
#EPSOPTION = --hires 

default:	$(FILENAME).pdf

$(FILENAME):		$(FILENAME).dvi

$(FILENAME).pdf:	$(FILENAME).tex $(TEXSRCS) $(ADDFILES) $(PDFPICS) 
	$(PDFTEX) $(FILENAME)
	@if test -n "`grep \"undefined citations\" ${FILENAME}.log`"; then \
         $(BIBTEX) $(FILENAME); \
         $(PDFTEX) $(FILENAME); \
	fi
	@if test -n "`grep \"undefined\" ${FILENAME}.log`"; then \
         $(PDFTEX) $(FILENAME); \
        fi

$(FILENAME).dvi:	$(FILENAME).tex $(TEXSRCS) $(ADDFILES)
	$(LATEX) 	$(FILENAME)
	@if test -n "`grep "undefined citations" ${FILENAME}.log`"; then \
         ${BIBTEX} $(FILENAME); \
         $(LATEX) $(FILENAME); \
	fi
	@if test -n "`grep "undefined" ${FILENAME}.log`"; then \
         $(LATEX) $(FILENAME); \
        fi
	$(LATEX) 	$(FILENAME)
	$(LATEX) 	$(FILENAME)

$(FILENAME).ps:		$(FILENAME).dvi $(PICS)
	$(DVIPS) $(FILENAME).dvi -o $(FILENAME).ps


$(FILENAME).tar.gz:	backup

dvi:	$(FILENAME).dvi

ps:	$(FILENAME).ps

gzip:	ps
	gzip $(FILENAME).ps

pdf:	$(FILENAME).pdf

ps2pdf:	$(FILENAME).dvi
	$(DVIPS) -Ppdf $(FILENAME).dvi -o $(FILENAME).ps
	$(PS2PDF) $(FILENAME).ps

print:  $(FILENAME).ps
	lpr $(FILENAME).ps

vdvi:	$(FILENAME).dvi
	xdvi $(FILENAME).dvi &

vps:	$(FILENAME).ps
	gv $(FILENAME).ps &

vpdf:	$(FILENAME).pdf
	xpdf $(FILENAME).pdf &

edit:
	emacs $(FILENAME).tex

clean:
	rm -f $(FILENAME).dvi $(FILENAME).toc .tmp.eps	\
	$(FILENAME).aux $(FILENAME).out $(FILENAME).tpt \
	$(FILENAME).bbl $(FILENAME).blg $(FILENAME).log \
	$(FILENAME).lof $(FILENAME).lot $(FILENAME).idx \
	$(FILENAME).ind $(FILENAME).ilg $(FILENAME).glo \
	$(FILENAME).blg $(FILENAME).brf missfont.log \
        $(FILENAME).end texput.log core *.aux *.tmp
	find $(TEXSDIR) $(PICSDIR) -name "*~" -type f -exec rm -f "{}" \;
	find $(TEXSDIR) $(PICSDIR) -type d -exec chmod 755 "{}" \;
	find $(TEXSDIR) $(PICSDIR) -type f -exec chmod 644 "{}" \;
	find $(TEXSDIR) -name "*.sh" -type f -exec chmod 755 "{}" \;

realclean: clean
	rm -f $(FILENAME).ps.gz $(FILENAME).pdf $(PDFPICS) $(FILENAME).ps $(FILENAME).tar.gz archiv.tar.gz

archiv:
	tar -cvzf archiv.tar.gz $(ARCHFILES)

backup:
	tar -cvzf $(FILENAME).tar.gz $(BACKFILES) $(ADDFILES)


.phony: print clean backup

$(PDFPICS): %.pdf: %.eps
	$(EPSTOPDF) $(EPSOPTION) $< --outfile=$@
#	$(EPS2EPS) $< .tmp.eps && $(EPSTOPDF) $(EPSOPTION) .tmp.eps --outfile=$@ && rm -f .tmp.eps
