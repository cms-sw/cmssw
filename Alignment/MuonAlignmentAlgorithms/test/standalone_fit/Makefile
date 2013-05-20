include $(ROOTSYS)/etc/Makefile.arch
#------------------------------------------------------------------------------

FITO        = \
	MuonResidualsFitter.$(ObjSuf) \
	MuonResiduals6DOFFitter.$(ObjSuf) \
	MuonResiduals5DOFFitter.$(ObjSuf) \
	MuonResiduals6DOFrphiFitter.$(ObjSuf) \
	MuonResidualsFitterDict.$(ObjSuf)

FITS        = \
	MuonResidualsFitter.$(SrcSuf) \
	MuonResiduals6DOFFitter.$(SrcSuf) \
	MuonResiduals5DOFFitter.$(SrcSuf) \
	MuonResiduals6DOFrphiFitter.$(SrcSuf) \
	MuonResidualsFitterDict.$(SrcSuf)

FITSO       = libFit.$(DllSuf)
FITX        = Fit$(ExeSuf)
FITLIB      = $(shell pwd)/$(FITSO)

OBJS          = $(FITO)

PROGRAMS      = $(FITX)

CXXFLAGS     += -D STANDALONE_FITTER

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)
#.PHONY:    Aclock Hello Tetris

all:            $(FITSO)

$(FITSO):     $(FITO)
		$(LD) $(SOFLAGS) $(LDFLAGS) $(ROOTLIBS) -lMinuit $^ $(OutPutOpt) $@ $(EXPLLINKLIBS)
		@echo "$@ done"


$(FITX):       $(FITSO)
		$(LD) $(LDFLAGS) $(FITO) $(LIBS) $(OutPutOpt)$@
		$(MT_EXE)
		@echo "$@ done"

clean:
		@rm -f $(OBJS) core.* *Dict.*

distclean:      clean
		@rm -f $(PROGRAMS) $(EVENTSO) $(EVENTLIB) *Dict.* *.def *.exp \
		   *.root *.ps *.so *.lib *.dll *.d *.log .def so_locations \
		   files/*
		@rm -rf cxx_repository
#		-@cd RootShower && $(MAKE) distclean

.SUFFIXES: .$(SrcSuf)

###

MuonResidualsFitter.$(ObjSuf): MuonResidualsFitter.h MuonResiduals6DOFFitter.h MuonResiduals5DOFFitter.h MuonResiduals6DOFrphiFitter.h
MuonResidualsFitterDict.$(SrcSuf): MuonResidualsFitter.h MuonResiduals6DOFFitter.h MuonResiduals5DOFFitter.h MuonResiduals6DOFrphiFitter.h LinkDef.h
	@echo "Generating dictionary $@..."
	$(ROOTCINT) -f $@ -c -DSTANDALONE_FITTER $^

.$(SrcSuf).$(ObjSuf):
	$(CXX)  $(CXXFLAGS) -c $<
