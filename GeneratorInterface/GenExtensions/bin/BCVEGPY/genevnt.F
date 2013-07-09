cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c upinit----initial parameters for pythia.                           c
c upevnt----call event process and possible error messages.          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...       bc in color-singlet and color-octet states.               c
c copyright (c) z.x zhang, chafik driouich, paula eerola and x.g. wu c
c reference: comput.phys.commun. 159,192(2004);                      c
c reference: comput.phys.commun. 174,241(2006);                      c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      subroutine upinit
 
c...double precision and integer declarations.
      implicit double precision(a-h, o-z)
      implicit integer(i-n)

c...selection of hard scattering subprocesses.
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      save /pypars/
 
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
      SAVE /HEPRUP/
 
c...user process event common block.
      parameter (maxnup=500)
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/

c...extra commonblock to transfer run info.
	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/histcol/inx

c...the user own transfer of information.
      double complex colmat,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	common/counter/ibcstate,nev
	logical generate
	common/genefull/generate
	common/vegcross/vegsec,vegerr,iveggrade

c...parameters transformtion.
      common/funtrans/nq2,npdfu
	common/usertran/ishower,idpp
c...transform some variables
      common/loggrade/ievntdis,igenerate,ivegasopen,igrade

C...Lines to read in assumed never longer than 200 characters. 
      PARAMETER (MAXLEN=200)



c...set up incoming beams. tevotran
      if(npdfu.eq.1) then
	   idbmup(1) = 2212
         idbmup(2) = -2212
	end if

c...set up incoming beams. lhc
	if(npdfu.eq.2) then
	   idbmup(1) = 2212
	   idbmup(2) = 2212
	end if

      ebmup(1)  = 0.5d0*ecm
      ebmup(2)  = 0.5d0*ecm

c...set up the external process.
      idwtup   = idpp
      nprup    = 1
      lprup(1) = 1001
	idprup   = lprup(1)

c...set up g+g --> b_c + b + c~ : maximum cross section in pb. 
c...using the default xmaxup(1)=0 to make pyevnt accept almost 
c...all the upevnt events. crossmax is the maximum differential
c...cross-section.       
	if(idwtup.eq.1) then
	  if(generate) then
         if(ivegasopen.eq.1) then
	    xmaxup(1)=crossmax
	   else
	    write(*,'(a)') 
     &	 'warning: here should input a maximum differential cross-sec'
	    write(*,'(a)')
     &	 '!stop here! input a proper value in (subroutine upinit)!!!!'
	stop 'or running vegas to get the correct value!program stop!'
c...      xmaxup(1)=100! the value added here! find the value in old runs.
c...note: this value should be the one obtained under the same condition.
	   end if
	  else
	     xmaxup(1)=0.0d0
	  end if
	end if

c...the value of xsecup(1) can be given arbitrarily. all of them are
c...only used for pystat(1) to produce a sensible cross-section table.
c...the actural total cross-sec of xsecup is (appcross).
	if (idwtup.eq.3) then
	  xsecup(1)=vegsec*1.0d+3  ! vegas value (nb) for initialization.
	  xmaxup(1)=crossmax       ! maximum differential cross-section
	end if

      return
      end
 
c**************************************************************************

      subroutine upevnt
 
c...all real arithmetic in double precision.
      implicit double precision(a-h, o-z)
      implicit integer(i-n)

C...PYTHIA commonblock: only used to provide read unit MSTP(162).
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      SAVE /PYPARS/


c...user process event common block.

      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/

      parameter (maxpup=100)
      integer pdfgup,pdfsup,lprup
      common/heprup/idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &idwtup,nprup,xsecup(maxpup),xerrup(maxpup),xmaxup(maxpup),
     &lprup(maxpup)
      save /heprup/

C...Lines to read in assumed never longer than 200 characters. 
      PARAMETER (MAXLEN=200)

#include "invegas.h"
#include "bcvegpy_set_par.inc"

       common/grade/xi(NVEGBIN,10)

	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/counter/ibcstate,nev
	logical generate
	common/genefull/generate

c...get the approximate total corss-section.
      common/totcross/appcross

c...to get the subprocess cross-section.
      common/subopen/subfactor,subenergy,isubonly

c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode
      common/mixevnt/xbcsec(8),imix,imixtype

      dimension x(10),ia(10)

cc#include "invegas.h"

c...if to get mixed results, then one should redefine grade
c...(the sampling importance function), the old grade files
c...must be exist and be generated under the same condition.
      if(imix.eq.1 .and. isubonly.eq.0) call initmixgrade

c...call the respective routine to generate event.
      if(idprup.eq.1001) then
         xnd  =NVEGBIN*1.0d0
         if(isubonly.eq.0) then
            ndim =7
         else  
            ndim =5
         end if

c...using the generated grade to generate the events points.
         call generand(ndim,xnd,x,ia,wgt)
         call phpoint(x,wt)
         if(wt.lt.1.0d-16) then
            xwgtup=0.0d0
         else
            xwgtup=totfun(x,wt)*wgt
         end if
         
         if(idwtup.eq.1.and.generate) then
* a dirty trick: limiting the diff xsection to current xwgtup
* the current xwgtup is high enough, so the total xsection 
* does not deviate much. this trick is from yu gouz's program
            if(xwgtup.gt.xmaxup(1)) then
               xwgtup = xmaxup(1)*0.9999999d0
            end if
         end if

c...calculate approximate crossection.
         appcross=appcross+xwgtup/nev*1.0d-3 !nb

c...record the maximum differential cross-section.
         if(xwgtup.gt.crossmax) then
            crossmax=xwgtup
         end if
c...gluon-gluon fusion. for all the s- and p- wave states.
         if(iqqbar.eq.0) then
            call bcpythia(21)
         end if
c...  quark-antiquark annihilation, only for color-singlet s-wave.
	   if(iqqbar.eq.1 .and. (ibcstate.eq.1.or.ibcstate.eq.2)) then
	     call bcpythia(iqcode)
	   end if
      else
         write(*,*) 'fatal error! unknown process',idprup
         stop
      end if

      return
      end
      
c**************************************************************************

C...bcvegpy_write_lhe
C...write the Bc event in the format needed for the
C...Les Houches event record.
 
      SUBROUTINE bcvegpy_write_lhe
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYCTAG/NCT,MCT(4000,2)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      COMMON/PYINT4/MWID(500),WIDS(500,5)
      SAVE /PYJETS/,/PYCTAG/,/PYDAT1/,/PYDAT2/,/PYDAT3/,/PYPARS/,
     &/PYINT1/,/PYINT2/,/PYINT4/
 
C...HEPEUP for output.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/
      
c IDWTUP = 3 requires XWGTUP = 1
      XWGTUP_wlhe = 1

C...Optionally write out event to disk. Minimal size for time/spin fields.
      IF(MSTP(162).GT.0) THEN
        WRITE(MSTP(162),5200) NUP,IDPRUP,XWGTUP_wlhe,SCALUP,AQEDUP
     +   ,AQCDUP
        DO 190 I=1,NUP
          IF(VTIMUP(I).EQ.0D0) THEN
            WRITE(MSTP(162),5300) IDUP(I),ISTUP(I),MOTHUP(1,I),
     &      MOTHUP(2,I),ICOLUP(1,I),ICOLUP(2,I),(PUP(J,I),J=1,5),
     &      ' 0. 9.'
          ELSE
            WRITE(MSTP(162),5400) IDUP(I),ISTUP(I),MOTHUP(1,I),
     &      MOTHUP(2,I),ICOLUP(1,I),ICOLUP(2,I),(PUP(J,I),J=1,5),
     &      VTIMUP(I),' 9.'
          ENDIF
  190   CONTINUE

C...Optional extra line with parton-density information.
        IF(MSTP(165).GE.1) WRITE(MSTP(162),5500) MSTI(15),MSTI(16),
     &  PARI(33),PARI(34),PARI(23),PARI(29),PARI(30) 
      ENDIF
 
C...Print formats.

 5200 FORMAT(1P,2I6,4E14.6)
 5300 FORMAT(1P,I8,5I5,5E18.10,A6)
 5400 FORMAT(1P,I8,5I5,5E18.10,E12.4,A3)
 5500 FORMAT(1P,'#pdf ',2I5,5E18.10)
 
      RETURN
      END
      
C*********************************************************************
 
C...bcvegpy_PYUPIN
C...Fills the HEPRUP commonblock with info on incoming beams and allowed
C...processes, and optionally stores that information on file.
 
      SUBROUTINE bcvegpy_PYUPIN
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      SAVE /PYJETS/,/PYSUBS/,/PYPARS/,/PYINT5/
 
C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
      SAVE /HEPRUP/
 
C...Store info on incoming beams.
c      IDBMUP(1)=K(1,2)
c      IDBMUP(2)=K(2,2)
c      EBMUP(1)=P(1,4)
c      EBMUP(2)=P(2,4)
      PDFGUP(1)=0
      PDFGUP(2)=0
      PDFSUP(1)=MSTP(51)
      PDFSUP(2)=MSTP(51)
 
C...Event weighting strategy.
      IDWTUP=3
 
C...Info on individual processes.
c      NPRUP=0
c      DO 100 ISUB=1,500
c        IF(MSUB(ISUB).EQ.1) THEN
c          NPRUP=NPRUP+1
c          XSECUP(NPRUP)=1D9*XSEC(ISUB,3)
c          XERRUP(NPRUP)=XSECUP(NPRUP)/SQRT(MAX(1D0,DBLE(NGEN(ISUB,3))))
c          XMAXUP(NPRUP)=1D0
c          LPRUP(NPRUP)=ISUB
c        ENDIF
c  100 CONTINUE
 
C...Write info to file.
      IF(MSTP(161).GT.0) THEN
        WRITE(MSTP(161),5100) IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     &  PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),IDWTUP,NPRUP
        DO 110 IPR=1,NPRUP
          WRITE(MSTP(161),5200) XSECUP(IPR),XERRUP(IPR),XMAXUP(IPR),
     &    LPRUP(IPR)
  110   CONTINUE
      ENDIF
 
C...Formats for printout.
 5100 FORMAT(1P,2I8,2E14.6,6I6)
 5200 FORMAT(1P,3E14.6,I6)
 
      RETURN
      END      
