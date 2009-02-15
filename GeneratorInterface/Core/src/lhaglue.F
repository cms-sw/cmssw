C*********************************************************************

C...  LHAGLUE Interface to LHAPDF library of modern parton
C...   density functions (PDF) with uncertainties
C...
C...Authors for v4: Dimitri Bourilkov, Craig Group, Mike Whalley
C...
C...Authors for v3: Dimitri Bourilkov, Craig Group, Mike Whalley
C...
C...Author for v1 and v2: Dimitri Bourilkov  bourilkov@mailaps.org
C...                      University of Florida
C...
C...HERWIG interface by Dimitri Bourilkov and Craig Group
C...
C...New numbering scheme and upgrade for LHAPDF v2.1
C...by Dimitri Bourilkov and Mike Whalley
C...
C...For more information, or when you cite this interface, currently
C...the official reference is:
C...D.Bourilkov, "Study of Parton Density Function Uncertainties with
C...LHAPDF and PYTHIA at LHC", hep-ph/0305126.
C...
C...The official LHAPDF page is:
C...
C...   http://durpdg.dur.ac.uk/lhapdf/index.html 
C...
C...The interface contains four subroutines (similar to PDFLIB).
C...It can be used seamlessly by Monte Carlo generators 
C...interfaced to PDFLIB or in stand-alone mode.
C...
C...    For initialization (called once)
C...
C...    PDFSET(PARM,VALUE)
C...
C...    For the proton/pion structure functions
C...
C...    STRUCTM(X,Q,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU)
C...
C...    For the photon structure functions
C...
C...    STRUCTP(X,Q2,P2,IP2,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU)
C...
C...    For statistics ON structure functions (under/over-flows)
C...
C...    PDFSTA
C...
C...This interface can be invoked in 3 ways depending
C...on the value of PARM(1) provided by the user when
C...calling PDFSET(PARM,VALUE):
C...
C...    For PYTHIA:         PARM(1).EQ.'NPTYPE'
C...      (this is set automatically by PYTHIA)
C...
C...    For HERWIG:         PARM(1).EQ.'HWLHAPDF'
C...      (set by the USER e.g. in the main program like this:
C...          AUTPDF(1) = 'HWLHAPDF'
C...          AUTPDF(2) = 'HWLHAPDF'                         )
C...
C...    For Stand-alone:    PARM(1).EQ.'DEFAULT'
C...      (can be used for PDF studies or when interfacing
C...       new generators)
C...
C...The LHAPDF set/member is selected depending on the value of:
C...
C...        PYTHIA:   ABS(MSTP(51)) - proton
C...                  ABS(MSTP(53)) - pion
C...                  ABS(MSTP(55)) - photon
C...
C...        HERWIG:   ABS(INT(VALUE(1)))
C...
C...   STAND-ALONE:   ABS(INT(VALUE(1)))
C...
C...
C...        CONTROL switches:
C...       ==================
C...
C...     THE LOCATION OF THE LHAPDF LIBRARY HAS TO BE SPECIFIED
C...     AS DESCRIBED BELOW (the rest is optional)
C...
C...     if the user does nothing, sensible defaults
C...     are active; to change the behaviour, the corresponding
C...     values of LHAPARM() should be set to the values given below
C...
C...  Location of the LHAPDF library of PDFs (pathname):
C...     uses common block /LHAPDFC/
C...
C...   If the user does nothing => default = subdir PDFsets of the 
C...                              current directory (can be real subdir
C...                              OR a soft link to the real location)
C...   If the user sets LHAPATH => supplied by the USER who defines the
C...                        path in common block COMMON/LHAPDFC/LHAPATH
C...                        BEFORE calling PDFSET
C...
C...   Other controls:
C...   ===============
C...     use common block /LHACONTROL/
C...
C...  Collect statistics on under/over-flow requests for PDFs
C...  outside their validity ranges in X and Q**2
C...  (call PDFSTA at end of run to print it out)
C...
C...      LHAPARM(16).EQ.'NOSTAT' => No statistics (faster)
C...      LHAPARM(16).NE.'NOSTAT' => Default: collect statistics
C...
C...  Option to use the values for the strong coupling alpha_s
C...  as computed in LHAPDF in the MC generator
C...  (to ensure uniformity between the MC generator and the PDF set)
C...  WARNING: implemented ONLY for PYTHIA in LHAPDFv4
C...
C...      LHAPARM(17).EQ.'LHAPDF' => Use alpha_s from LHAPDF
C...      LHAPARM(17).NE.'LHAPDF' => Default (same as LHAPDF v1/v3)
C...
C...  Extrapolation of PDFs outside LHAPDF validity range given by
C...  [Xmin,Xmax] and [Q2min,Q2max]; DEFAULT => PDFs "frozen" at the
C...  boundaries
C...
C...      LHAPARM(18).EQ.'EXTRAPOLATE' => Extrapolate PDFs on OWN RISK
C...                          WARNING: Crazy values can be returned
C...
C...  Printout of initialization information in PDFSET (by default)
C...
C...      LHAPARM(19).EQ.'SILENT' => No printout (silent mode)
C...      LHAPARM(19).EQ.'LOWKEY' => Print 5 times (almost silent mode)
C...
C...
C...v5.0  06-Oct-2005  Major change to allow multiset-initializations 
C...v4.0  28-Apr-2005  PDFSTA routine; option to use Alfa_s from LHAPDF
C...v4.0  21-Mar-2005  Photon/pion/new p PDFs, updated for LHAPDF v4
C...v3.1  26-Apr-2004  New numbering scheme, updated for LHAPDF v2/v3
C...v3.0  23-Jan-2004  HERWIG interface added
C...v2.0  20-Sep-2003  PDFLIB style adopted
C...v1.0  05-Mar-2003  First working version from PYTHIA to LHAPDF v1
C...
C...interface to LHAPDF library

C*********************************************************************

      BLOCK DATA LHAPDFSET
c... additions for multiset use
#include "parmsetup.inc"
#include "pathsetup.inc"
c      character*172 LHANAMES(nmxset)
      integer LHASET, LHAMEMB
      common/LHAPDF/LHANAME, LHASET, LHAMEMB
      integer LHAMEMBERS(nmxset),LHANUMBERS(nmxset)
      common/LHASETS/LHANAMES,LHANUMBERS,LHAMEMBERS,nsets
      data nsets/0/
c...
c      CHARACTER*132 LHAPATH
      COMMON/LHAPDFC/LHAPATH
      SAVE /LHAPDFC/
      DATA LHAPATH/'PDFsets'/ ! Default = PDFsets (below current dir)
      CHARACTER*20 LHAPARM(20)
      DOUBLE PRECISION LHAVALUE(20)
      COMMON/LHACONTROL/LHAPARM,LHAVALUE
      SAVE/LHACONTROL/
      DATA LHAPARM /20*' '/
      DATA LHAVALUE /20*0.0D0/
      DOUBLE PRECISION XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      COMMON/LHAGLSTA/ XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      SAVE/LHAGLSTA/
      DATA XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM/5*0.D0/
      DATA XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP/5*0.D0/
      END

C...PDFSET
C...Initialization for use of parton distributions
C... according to the LHAPDF interface.
C...
C...v4.0  28-Apr-2005  Option to use Alfa_s from LHAPDF
C...v4.0  21-Mar-2005  Photon/pion/new p PDFs, updated for LHAPDF v4
C...v3.1  26-Apr-2004  New numbering scheme
C...v3.0  23-Jan-2004  HERWIG interface added
C...
C...interface to LHAPDF library

      SUBROUTINE PDFSET(PARM,VALUE)
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c...additions for multiset use
#include "parmsetup.inc"
#include "pathsetup.inc"
c      character*172 LHANAMES(nmxset)
      integer LHAMEMBERS(nmxset),LHANUMBERS(nmxset)
      common/LHASETS/LHANAMES,LHANUMBERS,LHAMEMBERS,nsets
      real*8 xxmin(nmxset),xxmax(nmxset),qq2min(nmxset),qq2max(nmxset)
      save xxmin,xxmax,qq2min,qq2max
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      SAVE /PYPARS/
c... following 2 for earlier Pythia versions
      COMMON/LUDAT1/MSTU5(200),PARU5(200),MSTJ5(200),PARJ5(200)
      SAVE /LUDAT1/
C...Interface to LHAPDFLIB.
c      CHARACTER*172 LHANAME
      INTEGER LHASET, LHAMEMB
      COMMON/LHAPDF/LHANAME, LHASET, LHAMEMB
      SAVE /LHAPDF/
      DOUBLE PRECISION QCDLHA4, QCDLHA5
      INTEGER NFLLHA
      COMMON/LHAPDFR/QCDLHA4, QCDLHA5, NFLLHA
      SAVE /LHAPDFR/
c      CHARACTER*132 LHAPATH
      COMMON/LHAPDFC/LHAPATH
      SAVE /LHAPDFC/
      CHARACTER*20 LHAPARM(20)
      DOUBLE PRECISION LHAVALUE(20)
      COMMON/LHACONTROL/LHAPARM,LHAVALUE
      SAVE/LHACONTROL/
      INTEGER LHAEXTRP
      COMMON/LHAPDFE/LHAEXTRP
      SAVE /LHAPDFE/
      INTEGER LHASILENT
      COMMON/LHASILENT/LHASILENT
      SAVE /LHASILENT/
      DOUBLE PRECISION XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      COMMON/LHAGLSTA/ XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      SAVE/LHAGLSTA/
C...Interface to PDFLIB.
      COMMON/W50511/ NPTYPEPDFL,NGROUPPDFL,NSETPDFL,MODEPDFL,
     >               NFLPDFL,LOPDFL,TMASPDFL
      SAVE /W50511/
      DOUBLE PRECISION TMASPDFL
C...Interface to PDFLIB.
      COMMON/W50512/QCDL4,QCDL5
      SAVE /W50512/
      DOUBLE PRECISION QCDL4,QCDL5
C...Interface to PDFLIB.
      COMMON/W50513/XMIN,XMAX,Q2MIN,Q2MAX
      SAVE /W50513/
      DOUBLE PRECISION XMIN,XMAX,Q2MIN,Q2MAX
C...Local arrays and character variables (NOT USED here DB)
      CHARACTER*20 PARM(20)
      DOUBLE PRECISION VALUE(20)
      INTEGER LHAPATHLEN
      INTEGER LHAINPUT
      INTEGER LHASELECT
      INTEGER LHAPRINT
      INTEGER LHAONCE
      INTEGER LHAFIVE
      SAVE LHAONCE
      SAVE LHAFIVE
      DATA LHAONCE/0/
      DATA LHAFIVE/0/
      logical first
      data first/.TRUE./
      save first
 
      if(first .AND. (LHAPARM(20).NE.'LHAPATH')) then
c...overide the default PDFsets path
c ... check first if the environmental variable LHAPATH is set ... of not the
*     Use the lhapdf-config script to get the path to the PDF sets
      call getenv('LHAPATH',lhapath)
      if(lhapath.eq.'') then
      call system("lhapdf-config --pdfsets-path > /tmp/lhapdf-pdfsets-pa
     $th")
      open(unit=8, file="/tmp/lhapdf-pdfsets-path", status="old", iostat
     $=ierror)
      read (8,'(A)') LHAPATH
      close(8)
      endif
      first=.FALSE.
      endif
c
*
C...Init
      LHAEXTRP = 0
      IF(LHAPARM(18).EQ.'EXTRAPOLATE')
     >   THEN  ! Extrapolate PDFs on own risk
         LHAEXTRP = 1
      ENDIF
      LHASILENT = 0
      IF(LHAPARM(19).EQ.'SILENT') THEN    !  No printout (silent MODE)
         LHASILENT = 1
      ELSEIF(LHAPARM(19).EQ.'LOWKEY') THEN ! Print 5 times (lowkey MODE)
         IF(LHAFIVE .LT. 6) THEN
            LHAFIVE = LHAFIVE + 1
         ELSE
            LHASILENT = 1
         ENDIF
      ENDIF
      IF(PARM(1).EQ.'NPTYPE') THEN        !  PYTHIA
         if(MSTP(181).ge.6) then
           LHAPRINT = MSTU(11)
         else
           LHAPRINT = MSTU5(11)
         endif
         IF(VALUE(1) .EQ. 1) THEN         !   nucleon
           LHAINPUT = ABS(MSTP(51))
         ELSEIF(VALUE(1) .EQ. 2) THEN     !   pion
           LHAINPUT = ABS(MSTP(53))
         ELSEIF(VALUE(1) .EQ. 3) THEN     !   photon
           LHAINPUT = ABS(MSTP(55))
         ENDIF
         IF(LHASILENT .NE. 1) 
     >    PRINT *,'==== PYTHIA WILL USE LHAPDF ===='
      ELSEIF(PARM(1).EQ.'HWLHAPDF') THEN  !  HERWIG
         LHAINPUT = ABS(INT(VALUE(1)))
         IF(LHAONCE.EQ.LHAINPUT) RETURN
         IF(LHASILENT .NE. 1) 
     >    PRINT *,'==== HERWIG WILL USE LHAPDF ===='
         LHAPRINT = 6
         LHAONCE = LHAINPUT
      ELSEIF(PARM(1).EQ.'DEFAULT') THEN  !  Stand-alone
         LHAINPUT = ABS(INT(VALUE(1)))
         IF(LHAONCE.EQ.LHAINPUT) RETURN
         IF(LHASILENT .NE. 1) 
     >    PRINT *,'==== STAND-ALONE LHAGLUE MODE TO USE LHAPDF ===='
         LHAPRINT = 6
         LHAONCE = LHAINPUT
      ELSE
         PRINT *,'== UNKNOWN LHAPDF INTERFACE CALL! STOP EXECUTION! =='
         STOP
      ENDIF
C...Initialize parton distributions: LHAPDFLIB.
        LHAPATHLEN=INDEX(LHAPATH,' ')-1
        LHASET = LHAINPUT
        XMIN = 1.0D-6      ! X_min for current PDF set
        XMAX = 1.0D0       ! X_max for current PDF set
        Q2MIN = 1.0D0**2   ! Q**2_min scale for current PDF set [GeV]
        Q2MAX = 1.0D5**2   ! Q**2_max scale for current PDF set [GeV]
C...
C...Protons
C...
C...CTEQ Family
        IF((LHAINPUT .GE. 10000) .AND. (LHAINPUT .LE. 19999)) THEN
	  Q2MAX = 1.0D08
          IF((LHAINPUT .GE. 10000) .AND. (LHAINPUT .LE. 10040)) THEN
           LHASET = 10000
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq6.LHpdf'
	   Q2MIN = 1.69D0
          ELSEIF((LHAINPUT .GE. 10041) .AND. (LHAINPUT .LE. 10041)) THEN
           LHASET = 10041
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq6l.LHpdf'
	   Q2MIN = 1.69D0
          ELSEIF((LHAINPUT .GE. 10042) .AND. (LHAINPUT .LE. 10042)) THEN
           LHASET = 10042
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq6ll.LHpdf'
	   Q2MIN = 1.69D0
          ELSEIF((LHAINPUT .GE. 10050) .AND. (LHAINPUT .LE. 10090)) THEN
           LHASET = 10050
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq6mE.LHgrid'
	   Q2MIN = 1.69D0
          ELSEIF((LHAINPUT .GE. 10100) .AND. (LHAINPUT .LE. 10140)) THEN
           LHASET = 10100
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq61.LHpdf'
	   Q2MIN = 1.69D0
          ELSEIF((LHAINPUT .GE. 10150) .AND. (LHAINPUT .LE. 10190)) THEN
           LHASET = 10150
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq61.LHgrid'
	   Q2MIN = 1.69D0
          ELSEIF((LHAINPUT .GE. 10250) .AND. (LHAINPUT .LE. 10269)) THEN
           LHASET = 10250
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq6AB.LHgrid'
	   Q2MIN = 1.69D0
          ELSEIF((LHAINPUT .GE. 19050) .AND. (LHAINPUT .LE. 19050)) THEN
           LHASET = 19050
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq5m.LHgrid'
           XMIN=1.0D-5
          ELSEIF((LHAINPUT .GE. 19051) .AND. (LHAINPUT .LE. 19051)) THEN
           LHASET = 19051
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq5m1.LHgrid'
           XMIN=1.0D-5
          ELSEIF((LHAINPUT .GE. 19060) .AND. (LHAINPUT .LE. 19060)) THEN
           LHASET = 19060
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq5d.LHgrid'
           XMIN=1.0D-5
          ELSEIF((LHAINPUT .GE. 19070) .AND. (LHAINPUT .LE. 19070)) THEN
           LHASET = 19070
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq5l.LHgrid'
           XMIN=1.0D-5
          ELSEIF((LHAINPUT .GE. 19150) .AND. (LHAINPUT .LE. 19150)) THEN
           LHASET = 19150
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq4m.LHgrid'
	   Q2MIN = 2.56D0
           XMIN=1.0D-5
          ELSEIF((LHAINPUT .GE. 19160) .AND. (LHAINPUT .LE. 19160)) THEN
           LHASET = 19160
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq4d.LHgrid'
	   Q2MIN = 2.56D0
           XMIN=1.0D-5
          ELSEIF((LHAINPUT .GE. 19170) .AND. (LHAINPUT .LE. 19170)) THEN
           LHASET = 19170
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/cteq4l.LHgrid'
	   Q2MIN = 2.56D0
           XMIN=1.0D-5
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...MRST Family
        ELSEIF((LHAINPUT .GE. 20000) .AND. (LHAINPUT .LE. 29999)) THEN
          Q2MIN = 1.25D0
	  Q2MAX = 1.0D07
	  XMIN = 1.0D-5
          IF((LHAINPUT .GE. 20000) .AND. (LHAINPUT .LE. 20004)) THEN
           LHASET = 20000
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2001nlo.LHpdf'
          ELSEIF((LHAINPUT .GE. 20050) .AND. (LHAINPUT .LE. 20054)) THEN
           LHASET = 20050
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2001nlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20060) .AND. (LHAINPUT .LE. 20061)) THEN
           LHASET = 20060
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2001lo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20070) .AND. (LHAINPUT .LE. 20074)) THEN
           LHASET = 20070
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2001nnlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20100) .AND. (LHAINPUT .LE. 20130)) THEN
           LHASET = 20100
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2001E.LHpdf'
          ELSEIF((LHAINPUT .GE. 20150) .AND. (LHAINPUT .LE. 20180)) THEN
           LHASET = 20150
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2001E.LHgrid'
          ELSEIF((LHAINPUT .GE. 20200) .AND. (LHAINPUT .LE. 20201)) THEN
           LHASET = 20200
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2002nlo.LHpdf'
          ELSEIF((LHAINPUT .GE. 20250) .AND. (LHAINPUT .LE. 20251)) THEN
           LHASET = 20250
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2002nlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20270) .AND. (LHAINPUT .LE. 20271)) THEN
           LHASET = 20270
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2002nnlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20300) .AND. (LHAINPUT .LE. 20301)) THEN
           LHASET = 20300
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2003cnlo.LHpdf'
           Q2MIN = 10.0D0
 	   XMIN = 1.0D-3
          ELSEIF((LHAINPUT .GE. 20350) .AND. (LHAINPUT .LE. 20351)) THEN
           LHASET = 20350
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2003cnlo.LHgrid'
           Q2MIN = 10.0D0
 	   XMIN = 1.0D-3
          ELSEIF((LHAINPUT .GE. 20370) .AND. (LHAINPUT .LE. 20371)) THEN
           LHASET = 20370
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2003cnnlo.LHgrid'
           Q2MIN = 7.0D0
 	   XMIN = 1.0D-3
          ELSEIF((LHAINPUT .GE. 20400) .AND. (LHAINPUT .LE. 20401)) THEN
           LHASET = 20400
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004nlo.LHpdf'
          ELSEIF((LHAINPUT .GE. 20406) .AND. (LHAINPUT .LE. 20407)) THEN
           LHASET = 20406
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004FF3nlo.LHpdf'
          ELSEIF((LHAINPUT .GE. 20408) .AND. (LHAINPUT .LE. 20409)) THEN
           LHASET = 20408
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004FF4nlo.LHpdf'
          ELSEIF((LHAINPUT .GE. 20450) .AND. (LHAINPUT .LE. 20451)) THEN
           LHASET = 20450
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004nlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20452) .AND. (LHAINPUT .LE. 20453)) THEN
           LHASET = 20452
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004FF3lo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20454) .AND. (LHAINPUT .LE. 20455)) THEN
           LHASET = 20454
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004FF4lo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20456) .AND. (LHAINPUT .LE. 20457)) THEN
           LHASET = 20456
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004FF3nlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20458) .AND. (LHAINPUT .LE. 20459)) THEN
           LHASET = 20458
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004FF4nlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 20470) .AND. (LHAINPUT .LE. 20471)) THEN
           LHASET = 20470
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST2004nnlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 29000) .AND. (LHAINPUT .LE. 29003)) THEN
           LHASET = 29000
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST98.LHpdf'
          ELSEIF((LHAINPUT .GE. 29040) .AND. (LHAINPUT .LE. 29045)) THEN
           LHASET = 29040
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST98lo.LHgrid'
          ELSEIF((LHAINPUT .GE. 29050) .AND. (LHAINPUT .LE. 29055)) THEN
           LHASET = 29050
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST98nlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 29060) .AND. (LHAINPUT .LE. 29065)) THEN
           LHASET = 29060
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST98dis.LHgrid'
          ELSEIF((LHAINPUT .GE. 29070) .AND. (LHAINPUT .LE. 29071)) THEN
           LHASET = 29070
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/MRST98ht.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...Fermi Family
        ELSEIF((LHAINPUT .GE. 30000) .AND. (LHAINPUT .LE. 39999)) THEN
          IF((LHAINPUT .GE. 30100) .AND. (LHAINPUT .LE. 30200)) THEN
           LHASET = 30100
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/Fermi2002_100.LHpdf'
          ELSEIF((LHAINPUT .GE. 31000) .AND. (LHAINPUT .LE. 32000)) THEN
           LHASET = 31000
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/Fermi2002_1000.LHpdf'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...Alekhin Family
        ELSEIF((LHAINPUT .GE. 40000) .AND. (LHAINPUT .LE. 49999)) THEN
          IF((LHAINPUT .GE. 40100) .AND. (LHAINPUT .LE. 40200)) THEN
           LHASET = 40100
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/Alekhin_100.LHpdf'
          ELSEIF((LHAINPUT .GE. 41000) .AND. (LHAINPUT .LE. 41999)) THEN
           LHASET = 41000
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/Alekhin_1000.LHpdf'
          ELSEIF((LHAINPUT .GE. 40350) .AND. (LHAINPUT .LE. 40367)) THEN
           LHASET = 40350
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/a02m_lo.LHgrid'
           XMIN = 1.0D-7
	   Q2MIN = 0.8D0
	   Q2MAX = 2.0D08
          ELSEIF((LHAINPUT .GE. 40450) .AND. (LHAINPUT .LE. 40467)) THEN
           LHASET = 40450
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/a02m_nlo.LHgrid'
           XMIN = 1.0D-7
	   Q2MIN = 0.8D0
	   Q2MAX = 2.0D08
          ELSEIF((LHAINPUT .GE. 40550) .AND. (LHAINPUT .LE. 40567)) THEN
           LHASET = 40550 
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/a02m_nnlo.LHgrid'
           XMIN = 1.0D-7
	   Q2MIN = 0.8D0
	   Q2MAX = 2.0D08
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...Botje Family
        ELSEIF((LHAINPUT .GE. 50000) .AND. (LHAINPUT .LE. 59999)) THEN
          IF((LHAINPUT .GE. 50100) .AND. (LHAINPUT .LE. 50200)) THEN
           LHASET = 50100
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/Botje_100.LHpdf'
          ELSEIF((LHAINPUT .GE. 51000) .AND. (LHAINPUT .LE. 51999)) THEN
           LHASET = 51000
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/Botje_1000.LHpdf'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...ZEUS Family
        ELSEIF((LHAINPUT .GE. 60000) .AND. (LHAINPUT .LE. 69999)) THEN
	  Q2MIN = 0.3D0
	  Q2MAX = 2.0D05
          IF((LHAINPUT .GE. 60000) .AND. (LHAINPUT .LE. 60022)) THEN
           LHASET = 60000
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/ZEUS2002_TR.LHpdf'
          ELSEIF((LHAINPUT .GE. 60100) .AND. (LHAINPUT .LE. 60122)) THEN
           LHASET = 60100
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/ZEUS2002_ZM.LHpdf'
          ELSEIF((LHAINPUT .GE. 60200) .AND. (LHAINPUT .LE. 60222)) THEN
           LHASET = 60200
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/ZEUS2002_FF.LHpdf'
          ELSEIF((LHAINPUT .GE. 60300) .AND. (LHAINPUT .LE. 60322)) THEN
           LHASET = 60300
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/ZEUS2005_ZJ.LHpdf'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...H1 Family
        ELSEIF((LHAINPUT .GE. 70000) .AND. (LHAINPUT .LE. 79999)) THEN
	  Q2MIN = 1.5D0
	  Q2MAX = 3.5D04
	  XMIN = 5.7D-5
          IF((LHAINPUT .GE. 70050) .AND. (LHAINPUT .LE. 70050)) THEN
           LHASET = 70050
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000ms.LHgrid'
          ELSEIF((LHAINPUT .GE. 70051) .AND. (LHAINPUT .LE. 70070)) THEN
           LHASET = 70050
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000msE.LHgrid'
          ELSEIF((LHAINPUT .GE. 70150) .AND. (LHAINPUT .LE. 70150)) THEN
           LHASET = 70150
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000dis.LHgrid'
          ELSEIF((LHAINPUT .GE. 70151) .AND. (LHAINPUT .LE. 70170)) THEN
           LHASET = 70150
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000disE.LHgrid'
          ELSEIF((LHAINPUT .GE. 70250) .AND. (LHAINPUT .LE. 70250)) THEN
           LHASET = 70250
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000lo.LHgrid'
          ELSEIF((LHAINPUT .GE. 70251) .AND. (LHAINPUT .LE. 70270)) THEN
           LHASET = 70250
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000loE.LHgrid'
c tempoararily removed on returning to original H!2000 files
c          ELSEIF((LHAINPUT .GE. 70350) .AND. (LHAINPUT .LE. 70350)) THEN
c           LHASET = 70350
c           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000lo2.LHgrid'
c          ELSEIF((LHAINPUT .GE. 70351) .AND. (LHAINPUT .LE. 70370)) THEN
c           LHASET = 70350
c           LHANAME=LHAPATH(1:LHAPATHLEN)//'/H12000lo2E.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...GRV Family
        ELSEIF((LHAINPUT .GE. 80000) .AND. (LHAINPUT .LE. 89999)) THEN
	  Q2MIN = 0.8D0
	  Q2MAX = 2.0D06
	  XMIN = 1.0D-9
          IF((LHAINPUT .GE. 80050) .AND. (LHAINPUT .LE. 80051)) THEN
           LHASET = 80050
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GRV98nlo.LHgrid'
          ELSEIF((LHAINPUT .GE. 80060) .AND. (LHAINPUT .LE. 80060)) THEN
           LHASET = 80060
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GRV98lo.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...
C...Pions
C...
C...OW-PI Family
        ELSEIF((LHAINPUT .GE. 210) .AND. (LHAINPUT .LE. 212)) THEN
	  Q2MIN = 4.0D0
	  Q2MAX = 2.0D03
	  XMIN = 5.0D-03
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 210) .AND. (LHAINPUT .LE. 212)) THEN
           LHASET = 210
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/OWPI.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...SMRS-PI Family
        ELSEIF((LHAINPUT .GE. 230) .AND. (LHAINPUT .LE. 233)) THEN
	  Q2MIN = 5.0D0
	  Q2MAX = 1.31D06
	  XMIN = 1.0D-05
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 230) .AND. (LHAINPUT .LE. 233)) THEN
           LHASET = 230
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/SMRSPI.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...GRV-PI Family
        ELSEIF((LHAINPUT .GE. 250) .AND. (LHAINPUT .LE. 252)) THEN
	  Q2MAX = 1.00D06
	  XMIN = 1.0D-05
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 250) .AND. (LHAINPUT .LE. 251)) THEN
	   Q2MIN = 3.0D-1
           LHASET = 250
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GRVPI1.LHgrid'
          ELSEIF((LHAINPUT .GE. 252) .AND. (LHAINPUT .LE. 252)) THEN
	   Q2MIN = 2.5D-1
           LHASET = 252
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GRVPI0.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...ABFKW-PI Family
        ELSEIF((LHAINPUT .GE. 260) .AND. (LHAINPUT .LE. 263)) THEN
	  Q2MIN = 2.0D0
	  Q2MAX = 1.00D08
	  XMIN = 1.0D-03
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 260) .AND. (LHAINPUT .LE. 263)) THEN
           LHASET = 260
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/ABFKWPI.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...
C...Photons
C...
C...DO-G Family
        ELSEIF((LHAINPUT .GE. 310) .AND. (LHAINPUT .LE. 312)) THEN
	  Q2MIN = 1.0D01
	  Q2MAX = 1.00D04
	  XMIN = 1.0D-05
	  XMAX = 0.9D0
          IF((LHAINPUT .GE. 310) .AND. (LHAINPUT .LE. 311)) THEN
           LHASET = 310
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/DOG0.LHgrid'
          ELSEIF((LHAINPUT .GE. 312) .AND. (LHAINPUT .LE. 312)) THEN
           LHASET = 312
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/DOG1.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...DG-G Family
        ELSEIF((LHAINPUT .GE. 320) .AND. (LHAINPUT .LE. 324)) THEN
	  XMIN = 1.0D-05
	  XMAX = 0.9998D0
          LHASET = 320
          IF((LHAINPUT .GE. 320) .AND. (LHAINPUT .LE. 321)) THEN
	   Q2MIN = 1.0D0
	   Q2MAX = 1.0D04
c           LHASET = 320
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/DGG.LHgrid'
          ELSEIF((LHAINPUT .GE. 322) .AND. (LHAINPUT .LE. 322)) THEN
	   Q2MIN = 1.0D0
	   Q2MAX = 5.0D01
c           LHASET = 322
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/DGG.LHgrid'
          ELSEIF((LHAINPUT .GE. 323) .AND. (LHAINPUT .LE. 323)) THEN
	   Q2MIN = 2.0D1
	   Q2MAX = 5.0D02
c           LHASET = 323
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/DGG.LHgrid'
          ELSEIF((LHAINPUT .GE. 324) .AND. (LHAINPUT .LE. 324)) THEN
	   Q2MIN = 2.0D2
	   Q2MAX = 1.0D04
c           LHASET = 324
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/DGG.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...LAC/GAL-G Family
        ELSEIF((LHAINPUT .GE. 330) .AND. (LHAINPUT .LE. 334)) THEN
	  Q2MIN = 4.0D00
	  Q2MAX = 1.0D05
	  XMIN = 1.0D-04
	  XMAX = 0.9998D0
          LHASET = 330
          IF((LHAINPUT .GE. 330) .AND. (LHAINPUT .LE. 332)) THEN
c           LHASET = 330
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/LACG.LHgrid'
          ELSEIF((LHAINPUT .GE. 333) .AND. (LHAINPUT .LE. 333)) THEN
	   Q2MIN = 1.0D00
c           LHASET = 333
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/LACG.LHgrid'
          ELSEIF((LHAINPUT .GE. 334) .AND. (LHAINPUT .LE. 334)) THEN
	   Q2MIN = 4.0D00
c           LHASET = 334
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/LACG.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...GSG/GSG96-G Family
        ELSEIF((LHAINPUT .GE. 340) .AND. (LHAINPUT .LE. 345)) THEN
	  Q2MIN = 5.3D00
	  Q2MAX = 1.0D08
	  XMIN = 5.0D-04
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 340) .AND. (LHAINPUT .LE. 341)) THEN
           LHASET = 340
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GSG1.LHgrid'
          ELSEIF((LHAINPUT .GE. 342) .AND. (LHAINPUT .LE. 343)) THEN
           LHASET = 342
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GSG0.LHgrid'
          ELSEIF((LHAINPUT .GE. 344) .AND. (LHAINPUT .LE. 344)) THEN
           LHASET = 344
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GSG961.LHgrid'
          ELSEIF((LHAINPUT .GE. 345) .AND. (LHAINPUT .LE. 345)) THEN
           LHASET = 345
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GSG960.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...GRV-G Family
        ELSEIF((LHAINPUT .GE. 350) .AND. (LHAINPUT .LE. 354)) THEN
	  Q2MIN = 3.0D-1
	  Q2MAX = 1.0D06
	  XMIN = 1.0D-05
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 350) .AND. (LHAINPUT .LE. 352)) THEN
           LHASET = 350
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GRVG1.LHgrid'
          ELSEIF((LHAINPUT .GE. 353) .AND. (LHAINPUT .LE. 353)) THEN
	   Q2MIN = 2.5D-1
           LHASET = 353
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GRVG0.LHgrid'
          ELSEIF((LHAINPUT .GE. 354) .AND. (LHAINPUT .LE. 354)) THEN
	   Q2MIN = 6.0D-1
	   Q2MAX = 5.0D04
           LHASET = 354
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/GRVG0.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...ACFGP-G Family
        ELSEIF((LHAINPUT .GE. 360) .AND. (LHAINPUT .LE. 363)) THEN
	  Q2MIN = 2.0D00
	  Q2MAX = 5.5D05
	  XMIN = 1.37D-03
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 360) .AND. (LHAINPUT .LE. 363)) THEN
           LHASET = 360
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/ACFGPG.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...WHIT-G Family
        ELSEIF((LHAINPUT .GE. 380) .AND. (LHAINPUT .LE. 386)) THEN
	  Q2MIN = 4.0D00
	  Q2MAX = 2.5D03
	  XMIN = 1.0D-03
	  XMAX = 0.9998D0
          IF((LHAINPUT .GE. 380) .AND. (LHAINPUT .LE. 386)) THEN
           LHASET = 380
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/WHITG.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...SAS-G Family
        ELSEIF((LHAINPUT .GE. 390) .AND. (LHAINPUT .LE. 398)) THEN
	  Q2MAX = 5.0D04
	  XMIN = 1.0D-05
	  XMAX = 0.9998D0
          LHASET = 390
          IF((LHAINPUT .GE. 390) .AND. (LHAINPUT .LE. 392)) THEN
	   Q2MIN = 3.6D-1
c           LHASET = 390
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/SASG.LHgrid'
          ELSEIF((LHAINPUT .GE. 393) .AND. (LHAINPUT .LE. 394)) THEN
	   Q2MIN = 4.0D00
c           LHASET = 393
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/SASG.LHgrid'
          ELSEIF((LHAINPUT .GE. 395) .AND. (LHAINPUT .LE. 396)) THEN
	   Q2MIN = 3.6D-1
c           LHASET = 395
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/SASG.LHgrid'
          ELSEIF((LHAINPUT .GE. 397) .AND. (LHAINPUT .LE. 398)) THEN
	   Q2MIN = 4.0D00
c           LHASET = 397
           LHANAME=LHAPATH(1:LHAPATHLEN)//'/SASG.LHgrid'
          ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
          ENDIF          
C...Unknown Family ?! Giving up
        ELSE
           WRITE(LHAPRINT,5150)  LHASET
           STOP
        ENDIF

        LHAMEMB=LHAINPUT-LHASET
c....Now work out if we have already called this set/member
        iset = 0
	do j=1,nsets
	  if(lhaname.eq.lhanames(j).and.
     +       lhamemb.eq.lhamembers(j)) then
	     iset = j
          endif
	enddo
	if(iset.eq.0) then
           nsets=nsets+1
	   if(nsets.gt.nmxset) then
             if(LHASILENT.ne.1) then
	       print *,'WARNING:too many sets initialised'
               print *,'overwriting from set 1 again'
             endif
	     nsets = 1
c	     stop
	   endif 
	   iset=nsets
	   lhanames(iset)=lhaname
	   lhanumbers(iset)=lhainput
	   lhamembers(iset)=lhamemb
	   xxmin(iset)=xmin
	   xxmax(iset)=xmax
	   qq2min(iset)=q2min
	   qq2max(iset)=q2max
           CALL INITPDFSETM(iset,LHANAME)
           CALL NUMBERPDFM(iset,LHAALLMEM)
           IF(LHASILENT .NE. 1) THEN
             WRITE(LHAPRINT,5151)
             WRITE(LHAPRINT,5152) LHANAME
             WRITE(LHAPRINT,5153) LHAALLMEM
             WRITE(LHAPRINT,5154)
           ENDIF
           IF ((LHAMEMB.LT.0) .OR. (LHAMEMB.GT.LHAALLMEM)) THEN
             WRITE(LHAPRINT,5155)  LHAMEMB
             WRITE(LHAPRINT,5156)  LHAALLMEM
             STOP
           ENDIF

c        print *,'calling initpdf',lhamemb 
c           print *,'calling initpdfm ',iset,lhaname,lhamemb
c           print *,'LHAGLUE .... initializing set,member ',iset,lhamemb
           CALL INITPDFM(iset,LHAMEMB)
        endif	   
c... the rest is done every time pdfset is called
c           print *,'setting nset to:',iset
           call setnset(iset)
	   call setnmem(iset,lhamemb)
	   xmin = xxmin(iset)
	   xmax = xxmax(iset)
	   q2min=qq2min(iset)
           q2max=qq2max(iset)	   
           call GetLam4M(iset,LHAMEMB,qcdl4)
           call GetLam5M(iset,LHAMEMB,qcdl5)

           QMZ = 91.1876D0
           alphasLHA = alphasPDFM(iset,QMZ)
           IF(LHASILENT .NE. 1) 
     >     WRITE(LHAPRINT,5158) alphasLHA

           IF(LHAPARM(17).EQ.'LHAPDF') THEN
           NPTYPEPDFL = 1      ! Proton PDFs
           NFLPDFL = 4
           QCDLHA4 = QCDL4
           QCDLHA5 = QCDL5
           IF(LHASILENT .NE. 1) 
     >       WRITE(LHAPRINT,5159) QCDL4, QCDL5
           ELSE
             NPTYPEPDFL = 1      ! Proton PDFs
             NFLPDFL = 4
             ALAMBDA = 0.192D0
             QCDLHA4 = ALAMBDA
             QCDLHA5 = ALAMBDA
             IF(PARM(1).EQ.'NPTYPE') THEN        !  PYTHIA
             QCDL4 = ALAMBDA
             QCDL5 = ALAMBDA
           ENDIF
        ENDIF
C...Formats for initialization information.
 5150 FORMAT(1X,'WRONG LHAPDF set number =',I12,' given! STOP EXE!')
 5151 FORMAT(1X,'==============================================')
 5152 FORMAT(1X,'PDFset name ',A80)
 5153 FORMAT(1X,'with ',I10,' members')
 5154 FORMAT(1X,'====  initialized. ===========================')
 5155 FORMAT(1X,'LHAPDF problem => YOU asked for member = ',I10)
 5156 FORMAT(1X,'Valid range is: 0 - ',I10,' Execution stopped.')
 5157 FORMAT(1X,'Number of flavors for PDF is:',I4)
 5158 FORMAT(1X,'Strong coupling at Mz for PDF is:',F9.5)
 5159 FORMAT(1X,'Will use for PYTHIA QCDL4, QCDL5:',2F9.5)

      RETURN
      END
 
c********************************************************************
c -- STRUCTA
c -- copy of PDFLIB to use the eks98 nuclear correction factors

      SUBROUTINE STRUCTA(X,Q,A,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      CALL EKS98(X,Q,A,RUV,RDV,RU,RD,RS,RC,RB,RT,RG)
      CALL STRUCTM(X,Q,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU)
      UPV = RUV*UPV
      DNV = RDV*DNV
      USEA = RU*USEA
      DSEA = RD*DSEA
      STR = RS*STR
      CHM = RC*CHM
      BOT = RB*BOT
      TOP = RT*TOP
      GLU = RG*GLU
      RETURN
      END
      
C*********************************************************************
 
C...STRUCTM
C...Gives parton distributions according to the LHAPDF interface.
C...Two evolution codes used:
C...  EVLCTEQ for CTEQ PDF sets
C...  QCDNUM  for Other PDF sets
C...
C...Author: Dimitri Bourilkov  bourilkov@mailaps.org
C...
C...v4.0  21-Mar-2005  Photon/pion/new p PDFs, updated for LHAPDF v4
C...v3.0  23-Jan-2004
C...
C...interface to LHAPDF library

      SUBROUTINE STRUCTM(X,Q,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU)

C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      include 'parmsetup.inc'
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
C...Interface to LHAPDFLIB.
      include 'pathsetup.inc'
c      CHARACTER*172 LHANAME
      INTEGER LHASET, LHAMEMB
      COMMON/LHAPDF/LHANAME, LHASET, LHAMEMB
      SAVE /LHAPDF/
c...added next 2 lines for structp fix
      integer LHAMEMBERS(nmxset),LHANUMBERS(nmxset)
      common/LHASETS/LHANAMES,LHANUMBERS,LHAMEMBERS,nsets
c
      DOUBLE PRECISION QCDLHA4, QCDLHA5
      INTEGER NFLLHA
      COMMON/LHAPDFR/QCDLHA4, QCDLHA5, NFLLHA
      SAVE /LHAPDFR/
      CHARACTER*20 LHAPARM(20)
      DOUBLE PRECISION LHAVALUE(20)
      COMMON/LHACONTROL/LHAPARM,LHAVALUE
      SAVE/LHACONTROL/
      INTEGER LHAEXTRP
      COMMON/LHAPDFE/LHAEXTRP
      SAVE /LHAPDFE/
      DOUBLE PRECISION XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      COMMON/LHAGLSTA/ XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      SAVE/LHAGLSTA/
C...Interface to PDFLIB.
      COMMON/W50513/XMIN,XMAX,Q2MIN,Q2MAX
      SAVE /W50513/
      DOUBLE PRECISION XMIN,XMAX,Q2MIN,Q2MAX
C...Local variables
      DOUBLE PRECISION UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU
      DOUBLE PRECISION X,Q,F(-6:6)
 
      Q2 = Q**2
C...Statistics
      IF(LHAPARM(16).NE.'NOSTAT') THEN
      TOTNUM = TOTNUM+1.D0
      IF(X .LT. XMIN) XMINNUM = XMINNUM+1.D0
      IF(X .GT. XMAX) XMAXNUM = XMAXNUM+1.D0
      IF(Q2 .LT. Q2MIN) Q2MINNUM = Q2MINNUM+1.D0
      IF(Q2 .GT. Q2MAX) Q2MAXNUM = Q2MAXNUM+1.D0
      ENDIF
 
C...Range of validity e.g. 10^-6 < x < 1, Q2MIN < Q^2 extended by
C...freezing x*f(x,Q2) at borders.
      IF(LHAEXTRP .NE. 1) THEN    ! safe mode == "freeze"
       XIN=MAX(XMIN,MIN(XMAX,X))
       Q=SQRT(MAX(0D0,Q2MIN,MIN(Q2MAX,Q2)))
      ELSE                        ! adventurous mode == OWN RISK !
       XIN=X
      ENDIF

      call getnset(iset)
c      print *,'calling evolvepdfm:',iset
C
C...fix to allow STRUCTM to work for photon PDFs (Herwig does this)
C...set P2 = 0.0d0 and IP2 = 0
      if(LHANUMBERS(iset).ge.300.and.LHANUMBERS(iset).le.399) then  
        P2 = 0.0d0
	IP2 = 0
        CALL EVOLVEPDFPM(iset,XIN,Q,P2,IP2,F)
      else
        CALL EVOLVEPDFM(iset,XIN,Q,F)
      endif
      GLU = F(0)
      DSEA = F(-1)
      DNV = F(1) - DSEA
      USEA = F(-2)
      UPV = F(2) - USEA
      STR = F(3)
      CHM = F(4)
      BOT = F(5)
      TOP = F(6)

      RETURN
      END
 
C*********************************************************************
 
C...STRUCTP
C...Gives parton distributions according to the LHAPDF interface.
C...Used for photons.
C...
C...v4.0  21-Mar-2005  Photon/pion/new p PDFs, updated for LHAPDF v4
C...
C...interface to LHAPDF library

      SUBROUTINE STRUCTP
     > (X,Q2,P2,IP2,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU)

C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      include 'parmsetup.inc'
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
C...Interface to LHAPDFLIB.
      include 'pathsetup.inc'
c      CHARACTER*172 LHANAME
      INTEGER LHASET, LHAMEMB
      COMMON/LHAPDF/LHANAME, LHASET, LHAMEMB
      SAVE /LHAPDF/
      DOUBLE PRECISION QCDLHA4, QCDLHA5
      INTEGER NFLLHA
      COMMON/LHAPDFR/QCDLHA4, QCDLHA5, NFLLHA
      SAVE /LHAPDFR/
      CHARACTER*20 LHAPARM(20)
      DOUBLE PRECISION LHAVALUE(20)
      COMMON/LHACONTROL/LHAPARM,LHAVALUE
      SAVE/LHACONTROL/
      INTEGER LHAEXTRP
      COMMON/LHAPDFE/LHAEXTRP
      SAVE /LHAPDFE/
      DOUBLE PRECISION XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      COMMON/LHAGLSTA/ XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      SAVE/LHAGLSTA/
C...Interface to PDFLIB.
      COMMON/W50513/XMIN,XMAX,Q2MIN,Q2MAX
      SAVE /W50513/
      DOUBLE PRECISION XMIN,XMAX,Q2MIN,Q2MAX
C...Local variables
      DOUBLE PRECISION UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU
      DOUBLE PRECISION X,Q,F(-6:6)
 
C...Statistics
      IF(LHAPARM(16).NE.'NOSTAT') THEN
      TOTNUP = TOTNUP+1.D0
      IF(X .LT. XMIN) XMINNUP = XMINNUP+1.D0
      IF(X .GT. XMAX) XMAXNUP = XMAXNUP+1.D0
      IF(Q2 .LT. Q2MIN) Q2MINNUP = Q2MINNUP+1.D0
      IF(Q2 .GT. Q2MAX) Q2MAXNUP = Q2MAXNUP+1.D0
      ENDIF
 
C...Range of validity e.g. 10^-6 < x < 1, Q2MIN < Q^2 extended by
C...freezing x*f(x,Q2) at borders.
      Q = DSQRT(Q2)
      IF(LHAEXTRP .NE. 1) THEN    ! safe mode == "freeze"
       XIN=MAX(XMIN,MIN(XMAX,X))
       Q=SQRT(MAX(0D0,Q2MIN,MIN(Q2MAX,Q2)))
      ELSE                        ! adventurous mode == OWN RISK !
       XIN=X
      ENDIF
      call getnset(iset)
      CALL EVOLVEPDFPM(iset,XIN,Q,P2,IP2,F)

      GLU = F(0)
      DSEA = F(-1)
      DNV = F(1) - DSEA
      USEA = F(-2)
      UPV = F(2) - USEA
      STR = F(3)
      CHM = F(4)
      BOT = F(5)
      TOP = F(6)

      RETURN
      END
 
C*********************************************************************
 
C...PDFSTA
C...For statistics ON structure functions (under/over-flows)
C...
C...Author: Dimitri Bourilkov  bourilkov@mailaps.org
C...
C...
C...first introduced in v4.0  28-Apr-2005 
C...

      SUBROUTINE PDFSTA

C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
C...Interface to LHAPDFLIB.
      DOUBLE PRECISION XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      COMMON/LHAGLSTA/ XMINNUM,XMAXNUM,Q2MINNUM,Q2MAXNUM,TOTNUM,
     >                 XMINNUP,XMAXNUP,Q2MINNUP,Q2MAXNUP,TOTNUP
      SAVE/LHAGLSTA/

      PRINT *
      PRINT *,'===== PDFSTA statistics for PDF under/over-flows ===='
      PRINT *
      PRINT *,'====== STRUCTM statistics for nucleon/pion PDFs ====='
      PRINT *
      PRINT *,'  total # of calls ',TOTNUM
      IF(TOTNUM .GT. 0.D0) THEN
        PERCBELOW = 100.D0*XMINNUM/TOTNUM
        PERCABOVE = 100.D0*XMAXNUM/TOTNUM
        PRINT *,'  X  below PDF min ',XMINNUM,' or ',PERCBELOW, ' %'
        PRINT *,'  X  above PDF max ',XMAXNUM,' or ',PERCABOVE, ' %'
        PERCBELOW = 100.D0*Q2MINNUM/TOTNUM
        PERCABOVE = 100.D0*Q2MAXNUM/TOTNUM
        PRINT *,'  Q2 below PDF min ',Q2MINNUM,' or ',PERCBELOW, ' %'
        PRINT *,'  Q2 above PDF max ',Q2MAXNUM,' or ',PERCABOVE, ' %'
      ENDIF
      PRINT *
      PRINT *,'========= STRUCTP statistics for photon PDFs ========'
      PRINT *
      PRINT *,'  total # of calls ',TOTNUP
      IF(TOTNUP .GT. 0.D0) THEN
        PERCBELOW = 100.D0*XMINNUP/TOTNUP
        PERCABOVE = 100.D0*XMAXNUP/TOTNUP
        PRINT *,'  X  below PDF min ',XMINNUP,' or ',PERCBELOW, ' %'
        PRINT *,'  X  above PDF max ',XMAXNUP,' or ',PERCABOVE, ' %'
        PERCBELOW = 100.D0*Q2MINNUP/TOTNUP
        PERCABOVE = 100.D0*Q2MAXNUP/TOTNUP
        PRINT *,'  Q2 below PDF min ',Q2MINNUP,' or ',PERCBELOW, ' %'
        PRINT *,'  Q2 above PDF max ',Q2MAXNUP,' or ',PERCABOVE, ' %'
      ENDIF
      PRINT *

      RETURN
      END
**********************************************************************
*
* $Id: lhaglue.F,v 1.1 2008/06/03 18:14:29 yarba Exp $
*
* $Log: lhaglue.F,v $
* Revision 1.1  2008/06/03 18:14:29  yarba
* move some of the fortran code to src, add BuildFile for lib
*
* Revision 1.1  2008/02/04 23:20:40  yarba
* add use of lhapdf, partially restore use of tauola, photos
*
* Revision 1.6  2006/07/19 18:25:14  mkirsano
* author version 5.2.2
*
* Revision 1.7  2005/12/02 14:50:54  whalley
* Changes for new CTEQ code/AB sets
*
* Revision 1.6  2005/10/18 15:35:48  whalley
* fix to allow LHAPATH to be user defined as well as lhapdf-config
*
* Revision 1.5  2005/10/18 11:47:48  whalley
* Change to only set LHAPATH once per run
*
* Revision 1.1.1.2  1996/10/30 08:29:06  cernlib
* Version 7.04
*
* Revision 1.1.1.1  1996/04/12 15:29:26  plothow
* Version 7.01
*
*
      SUBROUTINE PFTOPDG(DX,DSCALE,DXPDF)
C
Cinclude "pdf/expdp.inc"
      double precision
     +       DX,DSCALE,DUPV,DDNV,DUSEA,DDSEA,DSTR,DCHM,DBOT,DTOP,DGL,
     +       DXPDF(-6:6)
C... call STRUCTM in PDFLIB to get flavour content
      CALL STRUCTM(DX,DSCALE,
     +                   DUPV,DDNV,DUSEA,DDSEA,DSTR,DCHM,DBOT,DTOP,DGL)
C... convert flavour convention of PDFLIB to PDG convention
      DXPDF(0) = DGL
      DXPDF(1) = DDNV + DDSEA
      DXPDF(2) = DUPV + DUSEA
      DXPDF(3) = DSTR
      DXPDF(4) = DCHM
      DXPDF(5) = DBOT
      DXPDF(6) = DTOP
      DXPDF(-1) = DDSEA
      DXPDF(-2) = DUSEA
      DXPDF(-3) = DSTR
      DXPDF(-4) = DCHM
      DXPDF(-5) = DBOT
      DXPDF(-6) = DTOP
C
      RETURN
      END
****************************************************************************
      subroutine setPDFpath(pathname)
      implicit real*8 (A-H,O-Z)
      include 'parmsetup.inc'
      character*(*) pathname
      include 'pathsetup.inc'
c      character*132 lhapath
      common/LHAPDFC/lhapath
      character*20 lhaparm(20)
      real*8 lhavalue(20)
      common/LHACONTROL/lhaparm,lhavalue
      lhaparm(20) = 'LHAPATH'
      do j=1,lnblnk(lhapath)
        lhapath(j:j)=''
      enddo
      lhapath = pathname
      return
      end
***********************************************************************  
      subroutine lhaset(lhaparm2,lhavalue2)
      implicit real*8 (a-h,o-z)
      character*20 lhaparm(20),lhaparm2(20)
      real*8 lhavalue(20),lhavalue2(20)
      common/LHACONTROL/lhaparm,lhavalue
      do j=1,20
       lhaparm(j)=lhaparm2(j)
       lhavalue(j)=lhavalue2(j)
      enddo
      return
      end
******************************************************************
      subroutine setlhaparm(lparm)
      implicit real*8 (a-h,o-z)
      character*(*) lparm
      character*20 lhaparm(20)
      real*8 lhavalue(20)
      common/LHACONTROL/lhaparm,lhavalue
      if(lparm.eq.'NOSTAT') then
        lhaparm(16)='NOSTAT'
      else if (lparm.eq.'16') then
        lhaparm(16)=''
      else if (lparm.eq.'LHAPDF') then
        lhaparm(17)='LHAPDF'
      else if (lparm.eq.'17') then
        lhaparm(17)=''
      else if (lparm.eq.'EXTRAPOLATE') then
        lhaparm(18)='EXTRAPOLATE'
      else if (lparm.eq.'18') then
        lhaparm(18)=''
      else if (lparm.eq.'SILENT') then
        lhaparm(19)='SILENT'
      else if (lparm.eq.'LOWKEY') then
        lhaparm(19)='LOWKEY'
      else if (lparm.eq.'19') then
        lhaparm(19)=''
      else
        print *,'WARNING from SetLHAPARM - value',lparm,' 
     & not recognized!'  
      endif	
      return
      end
***************************************************************
