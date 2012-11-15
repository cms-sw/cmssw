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
 
      integer maxpup
      parameter (maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common/heprup/idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &idwtup,nprup,xsecup(maxpup),xerrup(maxpup),xmaxup(maxpup),
     &lprup(maxpup)
      save /heprup/
 
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
	common/mixevnt2/xbcsum,ibclimit

c...parameters transformtion.
      common/funtrans/nq2,npdfu
	common/usertran/ishower,idpp
c...transform some variables
      common/loggrade/ievntdis,igenerate,ivegasopen,igrade


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
	  xsecup(1)=xbcsum*1.0d+3  ! vegas value (nb) for initialization.
	  xerrup(1)=vegerr*1.0d+3  ! error value (nb) for initialization
	  xmaxup(1)=crossmax       ! maximum differential cross-section
	end if

      return
      end
 
c**************************************************************************

      subroutine upevnt
 
c...all real arithmetic in double precision.
      implicit double precision(a-h, o-z)
      implicit integer(i-n)

C...pythia commonblock: only used to provide read unit mstp(162).
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      save /pypars/


c...user process event common block.

      integer maxnup
      parameter (maxnup=500)
      integer nup,idprup,idup,istup,mothup,icolup
      double precision xwgtup,scalup,aqedup,aqcdup,pup,vtimup,spinup
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/

      parameter (maxpup=100)
      integer pdfgup,pdfsup,lprup
      common/heprup/idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &idwtup,nprup,xsecup(maxpup),xerrup(maxpup),xmaxup(maxpup),
     &lprup(maxpup)
      save /heprup/


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
	logical unwght
	common/unweight/unwght
	common/xxmax/xmax(NVEGBIN,8)
	dimension x(10),ia(10)

cc#include "invegas.h"


c...call the respective routine to generate event.
      if(idprup.eq.1001) then
         xnd  =NVEGBIN*1.0d0
         if(isubonly.eq.0) then
            ndim =7
         else  
            ndim =5
         end if

c...using the generated grade to generate the events points.
	 if(.not.unwght) then
	   call generand(ndim,xnd,x,ia,wgt)
	   call phpoint(x,wt)
	   if(wt.lt.1.0d-16) then
	     xwgtup=0.0d0
	   else
	     xwgtup=totfun(x,wt)*wgt
	   end if
	 else
       call gen(totfun,ndim,xmax,1)
	     xwgtup=1
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
 
c...double precision and integer declarations.
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
 
c...commonblocks.
      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pyctag/nct,mct(4000,2)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      common/pyint1/mint(400),vint(400)
      common/pyint2/iset(500),kfpr(500,2),coef(500,20),icol(40,4,2)
      common/pyint4/mwid(500),wids(500,5)
      save /pyjets/,/pyctag/,/pydat1/,/pydat2/,/pydat3/,/pypars/,
     &/pyint1/,/pyint2/,/pyint4/
 
c...hepeup for output.
      integer maxnup
      parameter (maxnup=500)
      integer nup,idprup,idup,istup,mothup,icolup
      double precision xwgtup,scalup,aqedup,aqcdup,pup,vtimup,spinup
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/
      
c idwtup = 3 requires xwgtup = 1
      xwgtup_wlhe = 1

c...optionally write out event to disk. minimal size for time/spin fields.
      if(mstp(162).gt.0) then
        write(mstp(162),5200) nup,idprup,xwgtup_wlhe,scalup,aqedup
     +   ,aqcdup
        do 190 i=1,nup
          if(vtimup(i).eq.0d0) then
            write(mstp(162),5300) idup(i),istup(i),mothup(1,i),
     &      mothup(2,i),icolup(1,i),icolup(2,i),(pup(j,i),j=1,5),
     &      ' 0. 9.'
          else
            write(mstp(162),5400) idup(i),istup(i),mothup(1,i),
     &      mothup(2,i),icolup(1,i),icolup(2,i),(pup(j,i),j=1,5),
     &      vtimup(i),' 9.'
          endif
  190   continue

c...optional extra line with parton-density information.
        if(mstp(165).ge.1) write(mstp(162),5500) msti(15),msti(16),
     &  pari(33),pari(34),pari(23),pari(29),pari(30) 
      endif
 
c...print formats.

 5200 format(1p,2i6,4e14.6)
 5300 format(1p,i8,5i5,5e18.10,a6)
 5400 format(1p,i8,5i5,5e18.10,e12.4,a3)
 5500 format(1p,'#pdf ',2i5,5e18.10)
 
      return
      end
      
c*********************************************************************
 
c...bcvegpy_pyupin
c...fills the heprup commonblock with info on incoming beams and allowed
c...processes, and optionally stores that information on file.
 
      subroutine bcvegpy_pyupin
 
c...double precision and integer declarations.
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
 
c...commonblocks.
      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      common/pyint5/ngenpd,ngen(0:500,3),xsec(0:500,3)
      save /pyjets/,/pysubs/,/pypars/,/pyint5/
 
c...user process initialization commonblock.
      integer maxpup
      parameter (maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common/heprup/idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &idwtup,nprup,xsecup(maxpup),xerrup(maxpup),xmaxup(maxpup),
     &lprup(maxpup)
      save /heprup/
 
c...store info on incoming beams.
c      idbmup(1)=k(1,2)
c      idbmup(2)=k(2,2)
c      ebmup(1)=p(1,4)
c      ebmup(2)=p(2,4)
      pdfgup(1)=0
      pdfgup(2)=0
      pdfsup(1)=mstp(51)
      pdfsup(2)=mstp(51)
 
c...event weighting strategy.
      idwtup=3
 
c...info on individual processes.
c      nprup=0
c      do 100 isub=1,500
c        if(msub(isub).eq.1) then
c          nprup=nprup+1
c          xsecup(nprup)=1d9*xsec(isub,3)
c          xerrup(nprup)=xsecup(nprup)/sqrt(max(1d0,dble(ngen(isub,3))))
c          xmaxup(nprup)=1d0
c          lprup(nprup)=isub
c        endif
c  100 continue
 
c...write info to file.
      if(mstp(161).gt.0) then
        write(mstp(161),5100) idbmup(1),idbmup(2),ebmup(1),ebmup(2),
     &  pdfgup(1),pdfgup(2),pdfsup(1),pdfsup(2),idwtup,nprup
        do 110 ipr=1,nprup
          write(mstp(161),5200) xsecup(ipr),xerrup(ipr),xmaxup(ipr),
     &    lprup(ipr)
  110   continue
      endif
 
c...formats for printout.
 5100 format(1p,2i8,2e14.6,6i6)
 5200 format(1p,3e14.6,i6)
 
      return
      end      
