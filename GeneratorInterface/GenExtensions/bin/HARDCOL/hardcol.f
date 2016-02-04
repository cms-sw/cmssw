C...Extract parton-level events according to Les Houches Accord.
C...The Les Houches Event File produced here can then be used  
C...as input for hadron-level event simulation, also in Pythia8.

C...Note: you need to create two temporary files for MSTP(161) and MSTP(162). 
C...The final call to PYLHEF will pack them into a standard-compliant 
C...Les Houches Event File on your unit MSTP(163), and erase the two
C...temporary files (unless you set MSTP(164)=1).

C...IMPORTANT: the PYLHEF routine attached below is necessary if you run
C...with PYTHIA versions up to and including 6.403. Starting with 6.404
C...this routine is already part of the standard distribution, so you
C...should remove the copy in this file.

C...Double precision and integer declarations.
      SUBROUTINE HARDCOL
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)

C...Three PYTHIA functios retunr integers, so need declaring.	
      INTEGER PYK,PYCHGE,PYCOMP

C...EXTERNAL statement links PYDATA on most machines.
      EXTERNAL PYDATA

C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)

c...pythia common block.
      PARAMETER (MAXNUP=500)
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/

      PARAMETER (MAXPUP=100)
      INTEGER PDFGUP,PDFSUP,LPRUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
      SAVE /HEPRUP/

c...user process event common block.
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYDATR/MRPY(6),RRPY(100)
      COMMON/HCLPAR/ECM,NEV


C...Temporary files for initialization/event output.
      MSTP(161)=21
      OPEN(21,FILE='HARDCOL.init',STATUS='unknown')
      MSTP(162)=22
      OPEN(22,FILE='HARDCOL.evnt',STATUS='unknown')

C...Final Les Houches Event File, obtained by combining above two.
      MSTP(163)=23
      OPEN(23,FILE='HARDCOL.lhe',STATUS='unknown')

c*********************************************
c... HARDCOL
c*********************************************
      CALL SETPARAMETERS
      CALL read_hcs_file

C...Initialize.
      CALL EVNTINIT

C...Fills the HEPRUP commonblock with info on incoming beams and allowed
C...processes, and optionally stores that information on file.
c      CALL HARDCOL_PYUPIN
c      CALL PYUPIN

C...Event loop. List first few events.
      DO 200 IEV=1,NEV
        CALL PYUPEV
        IF(IEV.LE.2) THEN
           CALL PYLIST(2)
	   CALL PYLIST(7)
	ENDIF
 200  CONTINUE

C...Final statistics.
      CALL PYSTAT(1)
      CALL PYUPIN

C...Produce final Les Houches Event File.
      CALL PYLHEF

      END

C*********************************************************************

C...Combine the two old-style Pythia initialization and event files
C...into a single Les Houches Event File.
 
      SUBROUTINE PYLHEF
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
C...PYTHIA commonblock: only used to provide read/write units and version.
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      SAVE /PYPARS/
 
C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
      SAVE /HEPRUP/
 
C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/

C...Lines to read in assumed never longer than 200 characters. 
      PARAMETER (MAXLEN=200)
      CHARACTER*(MAXLEN) STRING

C...Format for reading lines.
      CHARACTER*6 STRFMT
      STRFMT='(A000)'
      WRITE(STRFMT(3:5),'(I3)') MAXLEN

C...Rewind initialization and event files. 
      REWIND MSTP(161)
      REWIND MSTP(162)

C...Write header info.
      WRITE(MSTP(163),'(A)') '<LesHouchesEvents version="1.0">'
      WRITE(MSTP(163),'(A)') '<!--'
      WRITE(MSTP(163),'(A,I1,A1,I3)') 'File generated with PYTHIA ',
     &MSTP(181),'.',MSTP(182)
      WRITE(MSTP(163),'(A)') '-->'       

C...Read first line of initialization info and get number of processes.
      READ(MSTP(161),'(A)',END=300,ERR=300) STRING                  
      READ(STRING,*,ERR=300) IDBMUP(1),IDBMUP(2),EBMUP(1),
     &EBMUP(2),PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),IDWTUP,NPRUP

C...Copy initialization lines, omitting trailing blanks. 
C...Embed in <init> ... </init> block.
      WRITE(MSTP(163),'(A)') '<init>' 
      DO 120 IPR=0,NPRUP
        IF(IPR.GT.0) READ(MSTP(161),'(A)',END=300,ERR=300) STRING
        LEN=MAXLEN+1  
  110   LEN=LEN-1
        IF(LEN.GT.1.AND.STRING(LEN:LEN).EQ.' ') GOTO 110
        WRITE(MSTP(163),'(A)',ERR=300) STRING(1:LEN)
  120 CONTINUE
      WRITE(MSTP(163),'(A)') '</init>' 

C...Begin event loop.
  200 CONTINUE

C...Read first line of event info and get number of particles.
      READ(MSTP(162),'(A)',END=280,ERR=300) STRING                  
      READ(STRING,*,ERR=300) NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP

C...Copy event lines, omitting trailing blanks. 
C...Embed in <event> ... </event> block.
      WRITE(MSTP(163),'(A)') '<event>' 
      DO 220 I=0,NUP
        IF(I.GT.0) READ(MSTP(162),'(A)',END=300,ERR=300) STRING
        LEN=MAXLEN+1  
  210   LEN=LEN-1
        IF(LEN.GT.1.AND.STRING(LEN:LEN).EQ.' ') GOTO 210
        WRITE(MSTP(163),'(A)',ERR=300) STRING(1:LEN)
  220 CONTINUE
      WRITE(MSTP(163),'(A)') '</event>' 

C...Loop back to look for next event.
      GOTO 200

C...Successfully reached end of event loop: write closing tag
C...and remove temporary intermediate files (unless asked not to).
  280 WRITE(MSTP(163),'(A)') '</LesHouchesEvents>' 
      IF(MSTP(164).EQ.1) RETURN
      CLOSE(MSTP(161),ERR=300,STATUS='DELETE')
      CLOSE(MSTP(162),ERR=300,STATUS='DELETE')
      RETURN

C...Error exit.
  300 WRITE(*,*) ' PYLHEF file joining failed!'

      RETURN
      END
