C======================================================================C
C                                                                      C
C     QQQ        GGG      SSSS    JJJJJJJ   EEEEEEE   TTTTTTT     I I  C
C    Q   Q      G   G    S    S         J   E            T        I I  C
C   Q     Q    G         S              J   E            T        I I  C
C   Q     Q    G   GGG    SSSS          J   EEEEE        T    ==  I I  C
C   Q   Q Q    G     G        S         J   E            T        I I  C
C    Q   Q      G   G    S    S    J   J    E            T        I I  C
C     QQQQQ      GGG      SSSS      JJJ     EEEEEEE      T        I I  C
C                                                                      C
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
C                  QUARK - GLUON - STRING - JET - II MODEL             C
C                                                                      C
C                HIGH ENERGY HADRON INTERACTION PROGRAM                C
C                                                                      C
C                                  BY                                  C
C                                                                      C
C                           S. OSTAPCHENKO                             C
C                                                                      C
C Institute for Physics, Norwegian University for Science & Tech       C
C D.V. Skobeltsyn Institute of Nuclear Physics, Moscow State UniversityC
C                  e-mail: sergei@tf.phys.ntnu.no                      C
C----------------------------------------------------------------------C
C         Publication to be cited when using this program:             C
C         S. Ostapchenko, Phys. Rev. D 83 (2011) 014018                C
C----------------------------------------------------------------------C
C                        LIST OF MODIFICATIONS                         C
C                                                                      C
C (Any modification of this program has to be approved by the author)  C
C                                                                      C
C 24.01.2005 - beta-version completed (qgsjet-II-01)                   C
C 12.04.2005 - final version (qgsjet-II-02)                            C
C 12.12.2005 - technical update -  version II-03:                      C
C    improved treatment of Pomeron cuts (all "net" cuts included);     C
C    improved treatment of nuclear config. (more consistent diffr.);   C
C    "baryon junction" mechanism included (motivated by RHIC data);    C
C    better parameter calibration, e.g. including RHIC data            C
C 21.02.2006 - some commons enlarged to avoid frequent rejects  D.H.   C
C 26.04.2006 - reduce unnecessary looping in qgsha              D.H.   C
C                                                                      C
C 01.10.2010 - new version  (qgsjet-II-04, not released):              C
C   treating all enhanced diagrams (incuding 'Pomeron loops');         C
C   calibration to LHC data on multiparticle production;               C
C   a number of cosmetic improvements,                                 C
C   e.g. more efficient simulation procedure (a factor of ~10 win)     C
C                                                                      C
C 26.06.2012 - final version (qgsjet-II-04):                           C
C additional parameter retuning applied                                C
C (mainly to TOTEM data on total/elastic pp cross sections);           C
C remnant treatment for pion-hadron/nucleus collisions improved        C
C                                                                      C
C                 last modification:  26.06.2012                       C
C                 Version qgsjet-II-04 (for CONEX)                     C
C                                                                      C
C small corrections to adapt to CORSIKA : 25.07.2012 by T.Pierog       C
C=======================================================================



c=============================================================================
      subroutine qgset
c-----------------------------------------------------------------------------
c common model parameters setting
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      character*7 ty
      character*2 tyq
      parameter(iapmax=208)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr3/  rmin,emax,eev
      common /qgarr6/  pi,bm,amws
      common /qgarr7/  xa(iapmax,3),xb(iapmax,3),b
      common /qgarr8/  wwm,be(4),dc(5),deta,almpt,ptdif,ptndi
      common /qgarr10/ am0,amn,amk,amc,amlamc,amlam,ameta,ammu
      common /qgarr11/ b10
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr21/ dmmin(3),wex(3),dmres(3),wdres(3)
      common /qgarr28/ arr(5)
      common /qgarr26/ factk,fqscal
      common /qgarr41/ ty(6)
      common /qgarr42/ tyq(16)
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /opt/     jopt
      common /qgdebug/ debug
      common /qgsIInex1/xan(iapmax,3),xbn(iapmax,3)  !used to link with nexus
     *,bqgs,bmaxqgs,bmaxnex,bminnex

      moniou=6             !output channel for debugging
      debug=0              !debugging level
                           !(0 - no debugging, 1 - very geheral,
                           !2 - more detailed, 3 - all function calls,
                           !4 - all returned values, 5 - technical)
      if(debug.ge.1)write (moniou,210)

      bqgs=0.d0            !used to link with nexus
      bmaxqgs=0.d0         !used to link with nexus
      bmaxnex=-1.d0        !used to link with nexus
      bminnex=0.d0         !used to link with nexus

      jopt=1               !parameter option

      if(jopt.eq.1)then       !tunable parameters
c soft Pomeron parameters
       dels=.165d0            !overcriticality
       alfp=.135d0            !trajectory slope
       sigs=1.01d0            !soft parton cross section
c coupling to DGLAP
       qt0=3.d0               !q**2 cutoff
       betp=2.2d0             !gluon distribution hardness for soft Pomeron
       dgqq=.16d0             !sea quark/gluon relative weight
c multi-Pomeron vertex parameters
       r3p=.0076d0            !triple-Pomeron coupling (/4/pi)
       g3p=.35d0              !factor for multu-Pomeron couplings
       sgap=exp(1.5d0)        !minimal rap-gap between 3P-vertices
c Pomeron-hadron coupling
       rq(1,1)=1.d0           !pion: vertex slope for 1st diffr. eigenst.
       rq(2,1)=.15d0          !pion: vertex slope for 2nd diffr. eigenst.
       cd(1,1)=1.75d0         !pion: relat. strenth for 1st diffr. eigenst.
       rq(1,2)=2.52d0         !proton: vertex slope for 1st diffr. eigenst.
       rq(2,2)=.2d0           !proton: vertex slope for 2nd diffr. eigenst.
       cd(1,2)=1.58d0         !proton: relat. strenth for 1st diffr. eigenst.
       rq(1,3)=.75d0          !kaon: vertex slope for 1st diffr. eigenst.
       rq(2,3)=.15d0          !kaon: vertex slope for 2nd diffr. eigenst.
       cd(1,3)=1.75d0         !kaon: relat. strenth for 1st diffr. eigenst.

c parameters for soft/hard fragmentation:

       qtf=.15d0              !q**2 cutoff for timelike cascades
       almpt=1.5d0            !string fragmentation parameter
       wwm=1.d0               !switching to 2-particle string decay (threshold)
c leading state exponents
       ahl(1)=0.d0            !pion
       ahl(2)=1.3d0           !proton
       ahl(3)=-0.5            !kaon
c remnant excitation probabilities
       wex(1)=.5d0            !pion
       wex(2)=.4d0            !proton
       wex(3)=.5d0            !kaon
c dc(i) - relative probabilities for qq~(qqq~q~)-pair creation from vacuum
       dc(1)=.077d0           !udu~d~
       dc(2)=.08d0            !ss~
       dc(4)=.4d0             !ss~ (intrinsic)
c be(i) - parameters for pt-distributions
       be(1)=.225d0           !uu~(dd~)
       be(2)=.43d0            !qqq~q~
       be(3)=.48d0            !ss~
       ptdif=.15d0            !diffractive momentum transfer
       ptndi=.19d0            !non-diffractive momentum transfer

c parameters for nuclear spectator part fragmentation:

       rmin=3.35d0    !coupling radius squared (fm^2)
       emax=.11d0     !relative critical energy ( / <E_ex>, <E_ex>~12.5 MeV )
       eev=.25d0      !relative evaporation energy ( / <E_ex>, <E_ex>~12.5 MeV )

      else
       stop'wrong option!!!'
      endif

      do i=1,3         !relative strenth of 2nd diffr. eigenst. [2-CD(1,icz)]
       cd(2,i)=2.d0-cd(1,i)
      enddo

!other parameters and constants:

      spmax=1.d11             !max energy squared for tabulations
      delh=0.25d0             !effective exponent for weighting (technical)
      epsxmn=.01d0            !pt-resolution scale (technical)
      alm=.04d0               !lambda_qcd squared
      factk=1.5d0             !k-factor value
      fqscal=4.d0             !factor for fact. scale (Mf^2=p_t^2/fqscal)
      deta=.11111d0           !ratio of etas production to all pions (1/9)
      dc(3)=.000d0            !to switch off charmed particles set to 0.000
      dc(5)=.0d0              !to switch off charmed particles set to 0.000
c weigts for diffractive eigenstates
      cc(1,1)=.5d0            !pion
      cc(2,1)=.5d0
      cc(1,2)=.5d0            !proton
      cc(2,2)=.5d0
      cc(1,3)=.5d0            !kaon
      cc(2,3)=.5d0
c auxiliary constants
      b10=.43876194d0         !initial value of the pseudorandom sequence
      pi=3.1416d0             !pi-value
      amws=.523d0             !diffusive radius for saxon-wood density
c regge intercepts for the uu~, qqq~q~, us~, uc~ trajectories
      arr(1)=0.5d0            !qq~-trajectory
      arr(2)=-0.5d0           !qqq~q~-trajectory
      arr(3)=0.d0             !us~-trajectory
c lowest resonance masses for low-mass excitations
      dmmin(1)=.76d0          !rho
      dmmin(2)=1.23d0         !delta
      dmmin(3)=.89d0          !K*
c mass and width for resonance contribution to low mass diffraction
      dmres(1)=1.23d0         !pion
      dmres(2)=1.44d0         !proton
      dmres(3)=1.27d0         !kaon
      wdres(1)=.3d0           !pion
      wdres(2)=.3d0           !proton
      wdres(3)=.1d0           !kaon
c proton, kaon, pion, d-meson, lambda, lambda_c, eta masses
      amn=0.93827999
      amk=.496d0
      am0=.14d0
      amc=1.868d0
      amlam=1.116d0
      amlamc=2.27d0
      ameta=.548d0
      ammu=.1057d0
c initial particle classes
      ty(1)='pion   '
      ty(2)='nucleon'
      ty(3)='kaon   '
c parton types
      tyq(1)='DD'
      tyq(2)='UU'
      tyq(3)='C '
      tyq(4)='S '
      tyq(5)='UD'
      tyq(6)='D '
      tyq(7)='U '
      tyq(8)='g '
      tyq(9)='u '
      tyq(10)='d '
      tyq(11)='ud'
      tyq(12)='s '
      tyq(13)='c '
      tyq(14)='uu'
      tyq(15)='dd'
      if(debug.ge.2)write (moniou,202)

210   format(2x,'qgset - common model parameters setting')
202   format(2x,'qgset - end')
      return
      end

c=============================================================================
      subroutine qgaini( DATDIR )
c-----------------------------------------------------------------------------
c common initialization procedure
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      CHARACTER DATDIR*(132)
      real qggamfun
      integer debug
      character *7 ty
      logical lcalc
      parameter(iapmax=208)
      dimension mij(40,40,4),nij(40,40,4),cs1(40,40,160)
     *,evs(40,100,3,2),ixemax(40,3,2),gz0(5),gz1(3)
     *,qfan0(11,14),qfan2(11,11,3),fann(14)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr6/  pi,bm,amws
      common /qgarr10/ am(7),ammu
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr24/ qpomr(11,11,216,12,2)
      common /qgarr25/ ahv(3)
      common /qgarr26/ factk,fqscal
      common /qgarr27/ qlegi(51,11,2,3,7),qfanu(51,11,11,6,2)
     *,qfanc(51,11,11,39,18),qdfan(21,11,11,2,3),qrev(11,11,66,219,2)
      common /qgarr28/ arr(5)
      common /qgarr29/ cstot(40,40,160)
      common /qgarr30/ cs0(40,40,160)
      common /qgarr31/ csborn(40,160)
      common /qgarr33/ fsud(10,2)
      common /qgarr34/ qrt(10,101,2)
      common /qgarr35/ qlegc0(51,10,11,6,8),qlegc(51,10,11,11,30)
      common /qgarr38/ qpomc(11,100,11,11,48)
      common /qgarr39/ qpomi(51,11,15),qpomis(51,11,11,11,9)
      common /qgarr41/ ty(6)
      common /qgarr43/ moniou
      common /qgarr47/ gsect(10,5,6)
      common /qgarr48/ qgsasect(10,6,6)
      common /qgarr51/ epsxmn
      common /qgarr52/ evk(40,40,100,3,2)
c auxiliary common blocks to calculate hadron-nucleus cross-sections
      common /arr1/   trnuc(56),twsnuc(56),twbnuc(56)
      common /arr3/   x1(7),a1(7)
      common /opt/    jopt
      common /qgdebug/debug
      character*500 fnIIdat,fnIIncs                        !used to link with nexus
      common /version/ version                             !used to link with nexus
      common/qgsIIfname/fnIIdat, fnIIncs, ifIIdat, ifIIncs !used to link with nexus
      common/qgsIInfname/ nfnIIdat, nfnIIncs               !used to link with nexus
      common/producetab/ producetables              !used to link with CRMC
      logical producetables

      if(debug.ge.1)write (moniou,210)
      version = 204

c-------------------------------------------------
      write(*,100)
 100  format(' ',
     *           '====================================================',
     *     /,' ','|                                                  |',
     *     /,' ','|         QUARK GLUON STRING JET -II MODEL         |',
     *     /,' ','|                                                  |',
     *     /,' ','|         HADRONIC INTERACTION MONTE CARLO         |',
     *     /,' ','|                        BY                        |',
     *     /,' ','|                 S. OSTAPCHENKO                   |',
     *     /,' ','|                                                  |',
     *     /,' ','|             e-mail: sergei@tf.phys.ntnu.no       |',
     *     /,' ','|                                                  |',
     *     /,' ','|                   Version II-04                  |',
     *     /,' ','|                                                  |',
     *     /,' ','| Publication to be cited when using this program: |',
     *     /,' ','| S.Ostapchenko, PRD 83 (2011) 014018              |',
     *     /,' ','|                                                  |',
     *     /,' ','| last modification:  26.06.2012                   |',
     *     /,' ','|                                                  |',
     *     /,' ','| Any modification has to be approved by the author|',
     *     /,' ','====================================================',
     *     /)


c-----------------------------------------------------------------------------
c normalization of parton density in the soft pomeron
      rr=qggamfun(real(2.d0+betp-dels))/qggamfun(real(1.d0-dels))
     */qggamfun(real(1.d0+betp))/4.d0/pi

      ahv(1)=.383d0+.624d0*dlog(dlog(qt0/.204d0**2)
     */dlog(.26d0/.204d0**2))
      ahv(3)=ahv(1)
      sq=dlog(dlog(qt0/.232d0**2)/dlog(.23d0/.232d0**2))
      ahv(2)=2.997d0+.753d0*sq-.076d0*sq*sq
c valence quark momentum share
      qnorm1=0.d0
      do i=1,7
      do m=1,2
       tp=1.d0-(.5d0+x1(i)*(m-1.5d0))**(2.d0/3.d0)
       xp=1.d0-tp**(1.d0/(1.d0+ahv(1)))
       qnorm1=qnorm1+a1(i)*(qggrv(xp,qt0,1,1)+qggrv(xp,qt0,1,2))
     * /dsqrt(1.d0-tp)
      enddo
      enddo
      qnorm1=qnorm1/(1.d0+ahv(1))/3.d0
      qnorm2=0.d0
      do i=1,7
      do m=1,2
       tp=1.d0-(.5d0+x1(i)*(m-1.5d0))**(2.d0/3.d0)
       xp=1.d0-tp**(1.d0/(1.d0+ahv(2)))
       qnorm2=qnorm2+a1(i)*(qggrv(xp,qt0,2,1)+qggrv(xp,qt0,2,2))
     * /dsqrt(1.d0-tp)
      enddo
      enddo
      qnorm2=qnorm2/(1.d0+ahv(2))/3.d0
c fp(i) - pomeron vertex constant (i=icz)
      fp(2)=(1.d0-qnorm2)*(2.d0+ahl(2))*(1.d0+ahl(2))

      gnorm=0.d0
      seanrm=0.d0
      do i=1,7
      do m=1,2
       xxg=(.5d0+x1(i)*(m-1.5d0))**(1.d0/(1.d0-dels))
       gnorm=gnorm+a1(i)*qgftld(xxg,2)
       seanrm=seanrm+a1(i)*qgftle(xxg,2)
      enddo
      enddo
      gnorm=gnorm/(1.d0-dels)*fp(2)*rr*2.d0*pi
      seanrm=seanrm/(1.d0-dels)*fp(2)*rr*2.d0*pi
      if(debug.ge.1)write (moniou,*)'rr,fp,norm,qnorm2,gnorm,seanrm'
     *,rr,fp(2),qnorm2+gnorm+seanrm,qnorm2,gnorm,seanrm

      do icz=1,3,2
       fp(icz)=(1.d0-qnorm1)*(2.d0+ahl(icz))*(1.d0+ahl(icz))
       gnorm=0.d0
       seanrm=0.d0
       do i=1,7
       do m=1,2
        xxg=(.5d0+x1(i)*(m-1.5d0))**(1.d0/(1.d0-dels))
        gnorm=gnorm+a1(i)*qgftld(xxg,icz)
        seanrm=seanrm+a1(i)*qgftle(xxg,icz)
       enddo
       enddo
       gnorm=gnorm/(1.d0-dels)*fp(icz)*rr*2.d0*pi
       seanrm=seanrm/(1.d0-dels)*fp(icz)*rr*2.d0*pi

       if(debug.ge.1)write (moniou,*)'fp,norm,qnorm1,gnorm,seanrm'
     * ,fp(icz),qnorm1+gnorm+seanrm,qnorm1,gnorm,seanrm
      enddo

      do icz=1,3
       gsoft(icz)=fp(icz)*fp(2)*sigs*4.d0*.0389d0
     * *qggamfun(real(1.d0+dels))**2*qggamfun(real(1.d0+ahl(icz)))
     * *qggamfun(real(1.d0+ahl(2)))/qggamfun(real(2.d0+dels+ahl(icz)))
     * /qggamfun(real(2.d0+dels+ahl(2)))
      enddo

c-----------------------------------------------------------------------------
c reading cross sections from the file
      if(ifIIdat.ne.1)then
       inquire(file=DATDIR(1:INDEX(DATDIR,' ')-1)//'qgsdat-II-04'
     *        ,exist=lcalc)
      else
       inquire(file=fnIIdat(1:nfnIIdat),exist=lcalc)       !used to link with nexus
      endif
      lzmaUse=0
      if(lcalc)then
         if(ifIIdat.ne.1)then
            open(1,file=DATDIR(1:INDEX(DATDIR,' ')-1)//'qgsdat-II-04'
     *           ,status='old')
         else                   !used to link with nexus
            if (LEN(fnIIdat).gt.6.and.
     *           fnIIdat(nfnIIdat-4:nfnIIdat) .eq. ".lzma") then
               lzmaUse=1
               call LzmaOpenFile(fnIIdat(1:nfnIIdat))
            else
               open(ifIIdat,file=fnIIdat(1:nfnIIdat),status='old')
            endif
         endif

       if (lzmaUse.ne.0) then

          if(debug.ge.0)write (moniou,214) 'qgsdat-II-04.lzma'

          call LzmaFillArray(csborn,size(csborn))
          call LzmaFillArray(cs0,size(cs0))
          call LzmaFillArray(cstot,size(cstot))
          call LzmaFillArray(evk,size(evk))
          call LzmaFillArray(qpomi,size(qpomi))
          call LzmaFillArray(qpomis,size(qpomis))
          call LzmaFillArray(qlegi,size(qlegi))
          call LzmaFillArray(qfanu,size(qfanu))
          call LzmaFillArray(qfanc,size(qfanc))
          call LzmaFillArray(qdfan,size(qdfan))
          call LzmaFillArray(qpomr,size(qpomr))
          call LzmaFillArray(gsect,size(gsect))
          call LzmaFillArray(qlegc0,size(qlegc0))
          call LzmaFillArray(qlegc,size(qlegc))
          call LzmaFillArray(qpomc,size(qpomc))
          call LzmaFillArray(fsud,size(fsud))
          call LzmaFillArray(qrt,size(qrt))
          call LzmaFillArray(qrev,size(qrev))
          call LzmaFillArray(fsud,size(fsud))
          call LzmaFillArray(qrt,size(qrt))
          call LzmaCloseFile()
       else
          if(debug.ge.0)write (moniou,214) 'qgsdat-II-04'
          read (1,*)csborn,cs0,cstot,evk,qpomi,qpomis,qlegi,qfanu,qfanc
     *         ,qdfan,qpomr,gsect,qlegc0,qlegc,qpomc,fsud,qrt,qrev,fsud,
     *         qrt
          close(1)
       endif

       if(debug.ge.0)write (moniou,*)'done'
       goto 10
      elseif(.not.producetables)then
        write(moniou,*) "Missing QGSDAT-II-04 file !"        
        write(moniou,*) "Please correct the defined path ",
     &"or force production ..."
        stop
      endif

c--------------------------------------------------
c qcd evolution and qcd ladder cross sections
c--------------------------------------------------
      if(debug.ge.0)write (moniou,201)
      do i=1,40
      do m=1,3
      do k=1,2
       ixemax(i,m,k)=99
      do j=1,40
      do l=1,100
       evk(i,j,l,m,k)=0.d0
      enddo
      enddo
      enddo
      enddo
      enddo

      n=1
1     n=n+1
      do m=1,3
      do k=1,2
       if(m.ne.3.or.k.ne.1)then
        do i=1,39
         if(ixemax(i,m,k).gt.0)then
          qi=spmax**((i-1)/39.d0)
          qq=qi*(spmax/qi)**(1.d0/39.d0)
          do l=1,99
           if(l.le.37)then
            xx=.1d0/(.1d0*spmax)**((37.d0-l)/36.d0)
           elseif(l.le.69)then
            xx=.1d0+.8d0*(l-37.d0)/32.d0
           else
            xx=1.d0-.1d0*(10.d0*epsxmn)**((l-69.d0)/31.d0)
           endif

           ev=qgev(qi,qq,qq,xx,m,k)/qgfap(xx,m,k)
           if(m.eq.1.and.k.eq.1.or.m.ne.1.and.k.ne.1)then
            evs(i,l,m,k)=dlog(1.d0+ev*4.5d0*qgsudx(qi,m)/qgsudx(qq,m)
     *      /dlog(dlog(qq/alm)/dlog(qi/alm)))
           else
            evs(i,l,m,k)=dlog(1.d0+ev/.3d0*(dlog(epsxmn)+.75d0)
     *      /(qgsudx(qq,1)/qgsudx(qi,1)-qgsudx(qq,2)/qgsudx(qi,2)))
           endif
          enddo
         endif
        enddo
       endif
      enddo
      enddo

      jec=0
      do m=1,3
      do k=1,2
       if(m.ne.3.or.k.ne.1)then
        do i=1,39
         if(ixemax(i,m,k).gt.0)then
          qi=spmax**((i-1)/39.d0)
          qq=qi*(spmax/qi)**(1.d0/39.d0)
          imx=ixemax(i,m,k)
          do l=imx,1,-1
           if(l.le.37)then
            xx=.1d0/(.1d0*spmax)**((37.d0-l)/36.d0)
           elseif(l.le.69)then
            xx=.1d0+.8d0*(l-37.d0)/32.d0
           else
            xx=1.d0-.1d0*(10.d0*epsxmn)**((l-69.d0)/31.d0)
           endif

           if(abs(evs(i,l,m,k)-evk(i,2,l,m,k)).gt.1.d-3)then
            evk(i,2,l,m,k)=evs(i,l,m,k)
            jec=1
           elseif(ixemax(i,m,k).eq.l)then
            ixemax(i,m,k)=l-1
           endif
          enddo
         endif
        enddo
       endif
      enddo
      enddo

      do i=1,39
       qi=spmax**((i-1)/39.d0)
       qj=qi*(spmax/qi)**(1.d0/39.d0)
       qq=qi*(spmax/qi)**(2.d0/39.d0)
       do l=99,1,-1
        if(l.le.37)then
         xx=.1d0/(.1d0*spmax)**((37.d0-l)/36.d0)
        elseif(l.le.69)then
         xx=.1d0+.8d0*(l-37.d0)/32.d0
        else
         xx=1.d0-.1d0*(10.d0*epsxmn)**((l-69.d0)/31.d0)
        endif
        do m=1,3
        do k=1,2
         if(m.ne.3.or.k.ne.1)then
          ev=(qgev(qi,qj,qq,xx,m,k)
     *    +qgevi(qi,qj,xx,m,k)*qgsudx(qq,k)/qgsudx(qj,k)
     *    +qgevi(qj,qq,xx,m,k)*qgsudx(qj,m)/qgsudx(qi,m))/qgfap(xx,m,k)
          if(m.eq.1.and.k.eq.1.or.m.ne.1.and.k.ne.1)then
           evk(i,3,l,m,k)=dlog(ev*4.5d0*qgsudx(qi,m)/qgsudx(qq,m)
     *     /dlog(dlog(qq/alm)/dlog(qi/alm)))
          else
           evk(i,3,l,m,k)=dlog(ev/.3d0*(dlog(epsxmn)+.75d0)
     *     /(qgsudx(qq,1)/qgsudx(qi,1)-qgsudx(qq,2)/qgsudx(qi,2)))
          endif
         endif
        enddo
        enddo
       enddo
      enddo
      if(jec.ne.0)goto 1

      do i=1,39
       qi=spmax**((i-1)/39.d0)
      do j=4,40
       qj=qi*(spmax/qi)**((j-2)/39.d0)
       qq=qi*(spmax/qi)**((j-1)/39.d0)
       do l=99,1,-1
        if(l.le.37)then
         xx=.1d0/(.1d0*spmax)**((37.d0-l)/36.d0)
        elseif(l.le.69)then
         xx=.1d0+.8d0*(l-37.d0)/32.d0
        else
         xx=1.d0-.1d0*(10.d0*epsxmn)**((l-69.d0)/31.d0)
        endif
        do m=1,3
        do k=1,2
         if(m.ne.3.or.k.ne.1)then
          ev=(qgev(qi,qj,qq,xx,m,k)
     *    +qgevi(qi,qj,xx,m,k)*qgsudx(qq,k)/qgsudx(qj,k)
     *    +qgevi(qj,qq,xx,m,k)*qgsudx(qj,m)/qgsudx(qi,m))/qgfap(xx,m,k)
          if(m.eq.1.and.k.eq.1.or.m.ne.1.and.k.ne.1)then
           evk(i,j,l,m,k)=dlog(ev*4.5d0*qgsudx(qi,m)/qgsudx(qq,m)
     *     /dlog(dlog(qq/alm)/dlog(qi/alm)))
          else
           evk(i,j,l,m,k)=dlog(ev/.3d0*(dlog(epsxmn)+.75d0)
     *     /(qgsudx(qq,1)/qgsudx(qi,1)-qgsudx(qq,2)/qgsudx(qi,2)))
          endif
         endif
        enddo
        enddo
       enddo
      enddo
      enddo

c--------------------------------------------------
c qcd ladder cross sections
      do i=1,40
       qi=(spmax/4.d0/fqscal)**((i-1)/39.d0)  !q^2 cutoff for born process
       s2min=qi*4.d0*fqscal          !energy threshold for 2->2 subprocess
      do m=1,2                                !parton types (1-g, 2-q)
      do l=1,2                                !parton types (1-g, 2-q)
       l1=2*l-1
      do k=1,40
       sk=s2min*(spmax/s2min)**((k-1)/39.d0)  !c.m. energy squared
       k1=k+40*(m-1)+80*(l-1)
       csborn(i,k1)=dlog(qgborn(qi,qi,sk,m-1,l1-1)) !born cross-section (2->2)
       if(.not.(csborn(i,k1).ge.0.d0.or.csborn(i,k1).lt.0.d0))stop
      enddo
      enddo
      enddo
      enddo

      do i=1,40
       qi=(spmax/4.d0/fqscal)**((i-1)/39.d0)
      do j=1,40
       qj=qi*(spmax/4.d0/fqscal/qi)**((j-1)/39.d0)
       s2min=qj*4.d0*fqscal
       smin=s2min/(1.d0-epsxmn)
      do m=1,2
      do l=1,2
       l1=2*l-1
       ml=m+2*l-2
      do k=1,40
       sk=s2min*(spmax/s2min)**((k-1)/39.d0)
       k1=k+40*(m-1)+80*(l-1)

       tmin=qj*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qj*fqscal/sk)))
       sjtot=qgjett(qi,qj,sk,m-1,l-1)
       sjord1=qgjeto(qi,qj,sk,m-1,l-1)
       sjord2=qgjeto(qj,qi,sk,l-1,m-1)
       born=qgborn(qi,qj,sk,m-1,l1-1)
       if(k.eq.1.or.j.eq.40.or.i.eq.40.or.sk.le.smin)then
        cstot(i,j,k1)=dlog(born)
        cs0(i,j,k1)=cstot(i,j,k1)
       else
        cstot(i,j,k1)=dlog(born+(sjtot+sjord1+sjord2)
     *  /(1.d0/tmin-2.d0/sk))
        cs0(i,j,k1)=dlog(born+sjord1/(1.d0/tmin-2.d0/sk))
       endif
       if(.not.(cstot(i,j,k1).ge.0.d0.or.cstot(i,j,k1).lt.0.d0))stop
       if(.not.(cs0(i,j,k1).ge.0.d0.or.cs0(i,j,k1).lt.0.d0))stop
      enddo
      enddo
      enddo
      enddo
      enddo
      goto 3

c--------------------------------------------------
c alternative calculation (not used)
      do i=1,40
       qi=(spmax/4.d0/fqscal)**((i-1)/39.d0)
      do j=1,40
       qj=qi*(spmax/4.d0/fqscal/qi)**((j-1)/39.d0)
       s2min=qj*4.d0*fqscal
      do m=1,2
      do l=1,2
       l1=2*l-1
       ml=m+2*l-2
      do k=1,40
       sk=s2min*(spmax/s2min)**((k-1)/39.d0)
       k1=k+40*(m-1)+80*(l-1)
       cstot(i,j,k1)=dlog(qgborn(qi,qj,sk,m-1,l1-1))
       cs0(i,j,k1)=cstot(i,j,k1)
       mij(i,j,ml)=2
       nij(i,j,ml)=2
      enddo
      enddo
      enddo
      enddo
      enddo

      n=2                             !number of ladder rungs considered
2     if(debug.ge.1)write (moniou,202)n,mij(1,1,1),nij(1,1,1)
      do i=1,39
       qi=(spmax/4.d0/fqscal)**((i-1)/39.d0)       !q^2 for upper parton
      do j=1,39
       qj=qi*(spmax/4.d0/fqscal/qi)**((j-1)/39.d0) !q^2 for downer parton
       s2min=qj*4.d0*fqscal                !energy threshold for 2->2 subprocess
       smin=s2min/(1.d0-epsxmn)            !energy threshold for 2->3 subprocess
      do m=1,2                                     !parton types (1-g, 2-q)
      do l=1,2                                     !parton types (1-g, 2-q)
       l1=2*l-1
       ml=m+2*l-2
       kmin=nij(i,j,ml)                  !lowest energy bin for another rung
       if(kmin.le.40)then
        do k=kmin,40
         sk=s2min*(spmax/s2min)**((k-1)/39.d0)
         if(sk.le.smin)then
          nij(i,j,ml)=nij(i,j,ml)+1
         else
          k1=k+40*(m-1)+80*(l-1)
          tmin=qj*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qj*fqscal/sk)))
          cs1(i,j,k1)=dlog(qgjet1(qi,qj,sk,s2min,m,l)
     *    /(1.d0/tmin-2.d0/sk)+qgborn(qi,qj,sk,m-1,l1-1))
         endif
        enddo
       endif
      enddo
      enddo
      enddo
      enddo

      do i=1,39
       qi=(spmax/4.d0/fqscal)**((i-1)/39.d0)
      do j=1,39
       qj=qi*(spmax/4.d0/fqscal/qi)**((j-1)/39.d0)
       s2min=qj*4.d0*fqscal
      do m=1,2
      do l=1,2
       ml=m+2*l-2
       kmin=nij(i,j,ml)
       if(kmin.le.40)then
        do k=40,kmin,-1
         sk=s2min*(spmax/s2min)**((k-1)/39.d0)
         tmin=qj*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qj*fqscal/sk)))
         k1=k+40*(m-1)+80*(l-1)
         if(abs(cs1(i,j,k1)-cs0(i,j,k1)).gt.1.d-2)then
          cs0(i,j,k1)=cs1(i,j,k1)
         elseif(k.eq.nij(i,j,ml))then
          nij(i,j,ml)=nij(i,j,ml)+1
         endif
        enddo
       endif
      enddo
      enddo
      enddo
      enddo

      do i=1,39
       qi=(spmax/4.d0/fqscal)**((i-1)/39.d0)
      do j=1,39
       qj=qi*(spmax/4.d0/fqscal/qi)**((j-1)/39.d0)
       s2min=qj*4.d0*fqscal         !min energy squared for 2->2 subprocess
       smin=s2min/(1.d0-epsxmn)     !min energy squared for 2->3 subprocess
      do m=1,2
      do l=1,2
       ml=m+2*l-2
       kmin=mij(i,j,ml)             !min energy bin for more ladder rungs
       if(kmin.le.40)then
        do k=kmin,40
         sk=s2min*(spmax/s2min)**((k-1)/39.d0)
         if(sk.le.smin)then
          mij(i,j,ml)=mij(i,j,ml)+1
         else
          k1=k+40*(m-1)+80*(l-1)
          tmin=qj*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qj*fqscal/sk)))
          cs1(i,j,k1)=dlog((qgjet(qi,qj,sk,s2min,m,l)
     *    +qgjit1(qj,qi,sk,l,m))/(1.d0/tmin-2.d0/sk))
         endif
        enddo
       endif
      enddo
      enddo
      enddo
      enddo

c--------------------------------------------------
c check convergence
      do i=1,39
       qi=(spmax/4.d0/fqscal)**((i-1)/39.d0)
      do j=1,39
       qj=qi*(spmax/4.d0/fqscal/qi)**((j-1)/39.d0)
       s2min=qj*4.d0*fqscal
      do m=1,2
      do l=1,2
       ml=m+2*l-2
       kmin=mij(i,j,ml)             !min energy bin for more ladder rungs
       if(kmin.le.40)then
        do k=40,kmin,-1
         sk=s2min*(spmax/s2min)**((k-1)/39.d0)
         tmin=qj*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qj*fqscal/sk)))
         k1=k+40*(m-1)+80*(l-1)
         if(abs(cs1(i,j,k1)-cstot(i,j,k1)).gt.1.d-2)then
          cstot(i,j,k1)=cs1(i,j,k1)
         elseif(k.eq.mij(i,j,ml))then
          mij(i,j,ml)=mij(i,j,ml)+1
         endif
        enddo
       endif
      enddo
      enddo
      enddo
      enddo

      n=n+1                         !one more rung
      do i=1,39
      do j=1,39
      do l=1,4
       if(mij(i,j,l).le.40.or.nij(i,j,l).le.40)goto 2
      enddo
      enddo
      enddo

3     if(debug.ge.2)write (moniou,205)
c-------------------------------------------------
c itermediate Pomeron
      if(debug.ge.1)write (moniou,210)
      s2min=4.d0*fqscal*qt0
      do iy=1,51
       sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
       rp=alfp*log(sy)*4.d0*.0389d0
      do iz=1,11
       if(iz.gt.6)then
        z=.2d0*(iz-6)
        b=sqrt(-log(z)*rp)
       else
        b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
        z=exp(-b*b/rp)
       endif

       qpomi(iy,iz,1)=dlog(qgpint(sy,b*b)
     * /sy**dels/sigs/z*rp/4.d0/.0389d0+1.d0)
      enddo
      enddo

c-------------------------------------------------
c loop contribution
      do iy=1,51
       sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
       rp=alfp*log(sy)*4.d0*.0389d0
      do iz=1,11
       if(iz.gt.6)then
        z=.2d0*(iz-6)
        b=sqrt(-log(z)*rp)
       else
        b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
        z=exp(-b*b/rp)
       endif
      do iqq=2,4
       qpomi(iy,iz,iqq)=qpomi(iy,iz,1)
      enddo
      enddo
      enddo

      do iy=2,51
       sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
       rp=alfp*log(sy)*4.d0*.0389d0
       do iz=1,11
       do iqq=2,4
        qpomi(iy,iz,iqq)=qpomi(iy-1,iz,iqq)
       enddo
       enddo
       n=0
4      n=n+1
       nrep=0
       do iz=1,11
        if(iz.gt.6)then
         z=.2d0*(iz-6)
         b=sqrt(-log(z)*rp)
        else
         b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
         z=exp(-b*b/rp)
        endif
        call qgloop(sy,b*b,fann,1)
        do iqq=1,3
         if(fann(iqq).gt.0.d0)then
          qfan0(iz,iqq)=dlog(fann(iqq)/z/sigs/sy**dels*rp/g3p**2
     *    /4.d0/.0389d0)
         elseif(iy.gt.2)then
          qfan0(iz,iqq)=min(2.d0*qpomi(iy-1,iz,iqq+1)
     *    -qpomi(iy-2,iz,iqq+1),qpomi(iy-1,iz,iqq+1))
         else
          stop'loop<0: iy=2'
         endif
         if(qfan0(iz,iqq).lt.-20.d0)then
          qfan0(iz,iqq)=-20.d0
         endif
         if(abs(qfan0(iz,iqq)-qpomi(iy,iz,iqq+1)).gt.1.d-3)nrep=1
        enddo
       enddo
       do iz=1,11
        if(iz.gt.6)then
         z=.2d0*(iz-6)
         b=sqrt(-log(z)*rp)
        else
         b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
         z=exp(-b*b/rp)
        endif
       do iqq=2,4
        qpomi(iy,iz,iqq)=qfan0(iz,iqq-1)
        if(.not.(qpomi(iy,iz,iqq).le.0.d0
     *  .or.qpomi(iy,iz,iqq).gt.0.d0))stop'qpom-nan'
       enddo
       enddo
       if(nrep.eq.1.and.n.lt.100)goto 4
      enddo

c-------------------------------------------------
c cut loops
      do iy=1,51
      do iz=1,11
       do iqq=5,7
        qpomi(iy,iz,iqq)=qpomi(iy,iz,iqq-3)
       enddo
       qpomi(iy,iz,8)=qpomi(iy,iz,2)
       do iqq=9,10
        qpomi(iy,iz,iqq)=qpomi(iy,iz,iqq-7)
        qpomi(iy,iz,iqq+2)=qpomi(iy,iz,iqq-7)
       enddo
       do iqq=13,15
        qpomi(iy,iz,iqq)=qpomi(iy,iz,iqq-11)
       enddo
      enddo
      enddo

      do iy=2,51
       sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
       rp=alfp*log(sy)*4.d0*.0389d0
       do iz=1,11
       do iqq=5,15
        qpomi(iy,iz,iqq)=qpomi(iy-1,iz,iqq)
       enddo
       enddo
       n=0
5      n=n+1
       nrep=0
       do iz=1,11
        if(iz.gt.6)then
         z=.2d0*(iz-6)
         b=sqrt(-log(z)*rp)
        else
         b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
         z=exp(-b*b/rp)
        endif
        call qgloop(sy,b*b,fann,2)
        do iqq=4,14
         if(fann(iqq).gt.0.d0)then
          qfan0(iz,iqq)=dlog(fann(iqq)/z/sigs/sy**dels*rp/g3p**2
     *    /4.d0/.0389d0)
         elseif(iy.gt.2)then
          qfan0(iz,iqq)=min(2.d0*qpomi(iy-1,iz,iqq+1)
     *    -qpomi(iy-2,iz,iqq+1),qpomi(iy-1,iz,iqq+1))
         else
          stop'loop<0: iy=2'
         endif
         if(qfan0(iz,iqq).lt.-20.d0)then
          qfan0(iz,iqq)=-20.d0
         endif
         if(abs(qfan0(iz,iqq)-qpomi(iy,iz,iqq+1)).gt.1.d-3)nrep=1
        enddo
       enddo
       do iz=1,11
        if(iz.gt.6)then
         z=.2d0*(iz-6)
         b=sqrt(-log(z)*rp)
        else
         b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
         z=exp(-b*b/rp)
        endif
       do iqq=5,15
        qpomi(iy,iz,iqq)=qfan0(iz,iqq-1)
        if(.not.(qpomi(iy,iz,iqq).le.0.d0
     *  .or.qpomi(iy,iz,iqq).gt.0.d0))stop'qpomi-nan'
       enddo
       enddo
       if(nrep.eq.1.and.n.lt.50)goto 5
      enddo

c-------------------------------------------------
c cut loops with proj/targ screening corrections
      do iv=1,11
       vvx=dble(iv-1)/10.d0
      do iv1=1,11
       vvxt=dble(iv1-1)/10.d0

       do iz=1,11
        do iqq=1,8
         qpomis(1,iz,iv,iv1,iqq)=0.d0
        enddo
        qpomis(1,iz,iv,iv1,1)=qpomi(1,iz,1)
        qpomis(1,iz,iv,iv1,4)=qpomi(1,iz,1)
       enddo

       do iy=2,51
        sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
        rp=alfp*log(sy)*4.d0*.0389d0
        do iz=1,11
        do iqq=1,8
         qpomis(iy,iz,iv,iv1,iqq)=qpomis(iy-1,iz,iv,iv1,iqq)
        enddo
        enddo

        n=0
6       n=n+1
        nrep=0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          b=sqrt(-log(z)*rp)
         else
          b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
          z=exp(-b*b/rp)
         endif
         call qgloos(sy,b*b,vvx,vvxt,fann)
         vi=qgpini(sy,b*b,0.d0,0.d0,2)
         vic=min(vi,qgpini(sy,b*b,0.d0,0.d0,8))
         vicng=min(vic,qgpini(sy,b*b,0.d0,0.d0,11))
         do iqq=1,8
          if(fann(iqq).gt.0.d0)then
           if(iqq.eq.1.or.iqq.eq.4)then
            qfan0(iz,iqq)=dlog(fann(iqq)/z/sigs/sy**dels*rp/g3p**2
     *      /4.d0/.0389d0)
           elseif(iqq.eq.3)then
            qfan0(iz,iqq)=dlog(fann(iqq)/(.5d0*max(0.d0,1.d0
     *      -exp(-2.d0*vic)*(1.d0+2.d0*vic))+vicng*exp(-2.d0*vic)))
           elseif(iqq.gt.6)then
            qfan0(iz,iqq)=dlog(fann(iqq)*2.d0/((1.d0-exp(-vi))**2
     *      +(exp(2.d0*(vi-vic))-1.d0)*exp(-2.d0*vi)))
           else
            qfan0(iz,iqq)=dlog(fann(iqq)/(1.d0-exp(-vi)))
           endif
          elseif(iy.gt.2)then
           qfan0(iz,iqq)=min(2.d0*qpomis(iy-1,iz,iv,iv1,iqq)
     *     -qpomis(iy-2,iz,iv,iv1,iqq),qpomis(iy-1,iz,iv,iv1,iqq))
          else
           qfan0(iz,iqq)=qpomis(iy-1,iz,iv,iv1,iqq)
          endif
          if(iqq.gt.5)qfan0(iz,iqq)=min(qfan0(iz,iqq),qfan0(iz,iqq-1))
          qfan0(iz,iqq)=max(qfan0(iz,iqq),-20.d0)
          if(abs(qfan0(iz,iqq)-qpomis(iy,iz,iv,iv1,iqq)).gt.1.d-3)
     *    nrep=1
         enddo
        enddo

        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          b=sqrt(-log(z)*rp)
         else
          b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
          z=exp(-b*b/rp)
         endif
         do iqq=1,8
          qpomis(iy,iz,iv,iv1,iqq)=qfan0(iz,iqq)
          if(iqq.eq.1.or.iqq.eq.4)then
           dpx=exp(qpomis(iy,iz,iv,iv1,iqq))*g3p**2*sigs
     *     *sy**dels*z/rp*4.d0*.0389d0
          else
           dpx=exp(qpomis(iy,iz,iv,iv1,iqq))
          endif
         enddo
        enddo
        if(nrep.eq.1.and.n.lt.50)goto 6
       enddo
      enddo
      enddo

c-------------------------------------------------
c integrated Pomeron leg eikonals
      if(debug.ge.1)write (moniou,212)
      do icz=1,3
      do icdp=1,2
       if(cd(icdp,icz).ne.0.d0)then
        do iy=1,51
         sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
         rp=(rq(icdp,icz)+alfp*log(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          b=sqrt(-log(z)*rp)
         else
          b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
          z=exp(-b*b/rp)
         endif

         qxl=qgleg(sy,b*b,icdp,icz)
         qlegi(iy,iz,icdp,icz,1)=log(qxl/z)
        enddo
        enddo
       endif
      enddo
      enddo

c-------------------------------------------------
c loop-legs
      do icz=1,3
      do icdp=1,2
       if(cd(icdp,icz).ne.0.d0)then
        do iy=1,51
         sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
         rp=(rq(icdp,icz)+alfp*log(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          b=sqrt(-log(z)*rp)
         else
          b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
          z=exp(-b*b/rp)
         endif
         if(iy.eq.1)then
          do iqq=2,7
           qlegi(iy,iz,icdp,icz,iqq)=qlegi(iy,iz,icdp,icz,1)
          enddo
         else
          call qglool(sy,b*b,icdp,icz,fann)
          do iqq=2,7
           if(fann(iqq-1).gt.0.d0)then
            qlegi(iy,iz,icdp,icz,iqq)=log(fann(iqq-1)/z)
           else
            qlegi(iy,iz,icdp,icz,iqq)=2.d0*qlegi(iy-1,iz,icdp,icz,iqq)
     *      -qlegi(iy-2,iz,icdp,icz,iqq)
           endif
           qlegi(iy,iz,icdp,icz,iqq)=max(qlegi(iy,iz,icdp,icz,iqq)
     *     ,-20.d0)
           if(.not.(qlegi(iy,iz,icdp,icz,iqq).le.0.d0
     *     .or.qlegi(iy,iz,icdp,icz,iqq).gt.0.d0))stop'leg-nan'
          enddo
         endif
        enddo
        enddo
       endif
      enddo
      enddo

c-------------------------------------------------
c uncut fan-contributions
      if(debug.ge.1)write (moniou,213)
      do icz=1,3
      do iv=1,11
       vvx=dble(iv-1)/10.d0
      do icdp=1,2
       if(cd(icdp,icz).ne.0.d0)then
        do iy=1,51
        do iz=1,11
        do iqq=1,2
         qfanu(iy,iz,iv,icdp+2*(icz-1),iqq)=qlegi(iy,iz,icdp,icz,iqq+1)
        enddo
        enddo
        enddo

        do iy=2,51
         sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
         rp=(rq(icdp,icz)+alfp*dlog(sy))*4.d0*.0389d0
         do iz=1,11
         do iqq=1,2
          qfanu(iy,iz,iv,icdp+2*(icz-1),iqq)
     *    =qfanu(iy-1,iz,iv,icdp+2*(icz-1),iqq)
         enddo
         enddo

         n=1
7        n=n+1
         nrep=0
         do iz=1,11
          if(iz.gt.6)then
           z=.2d0*dble(iz-6)
           b=dsqrt(-dlog(z)*rp)
          else
           b=dsqrt(-rp*(dlog(0.2d0)+2.d0*dble(iz-7)))
           z=dexp(-b*b/rp)
          endif
          call qgfan(sy,b*b,vvx,icdp,icz,fann)
          do iqq=1,2
           if(fann(iqq).gt.0.d0)then
            qfan0(iz,iqq)=dlog(fann(iqq)/z)
           else
            qfan0(iz,iqq)=min(qfanu(iy-1,iz,iv,icdp+2*(icz-1),iqq)
     *      ,2.d0*qfanu(iy-1,iz,iv,icdp+2*(icz-1),iqq)
     *      -qfanu(iy-2,iz,iv,icdp+2*(icz-1),iqq))
           endif
           qfan0(iz,iqq)=max(qfan0(iz,iqq),-20.d0)
           if(abs(qfan0(iz,iqq)-qfanu(iy,iz,iv,icdp+2*(icz-1),iqq))
     *     .gt.1.d-3)nrep=1
          enddo
         enddo

         do iz=1,11
         do iqq=1,2
          qfanu(iy,iz,iv,icdp+2*(icz-1),iqq)=qfan0(iz,iqq)
         enddo
         enddo
         if(nrep.eq.1)goto 7

         do iz=1,11
         do iqq=1,2
          if(iz.gt.6)then
           z=.2d0*dble(iz-6)
           b=dsqrt(-dlog(z)*rp)
          else
           b=dsqrt(-rp*(dlog(0.2d0)+2.d0*dble(iz-7)))
           z=dexp(-b*b/rp)
          endif
          if(.not.(qfanu(iy,iz,iv,icdp+2*(icz-1),iqq).le.0.d0
     *    .or.qfanu(iy,iz,iv,icdp+2*(icz-1),iqq).gt.0.d0))stop'fan-nn'
         enddo
         enddo
        enddo
       endif
      enddo
      enddo
      enddo

c-------------------------------------------------
c cut fan contributions
      if(debug.ge.1)write (moniou,215)
      do icz=1,3                                !hadron class
      do icdp=1,2                                 !diffractive eigenstate
       if(cd(icdp,icz).ne.0.d0)then
c vvx,vvxp,vvxpl - screening corrections from targ. and nuclear proj. fans
        do iv=1,11
         vvx=dble(iv-1)/10.d0
        do iv1=1,1+5*(icz-1)*(3-icz)
         vvxp=dble(iv1-1)/5.d0
        do iv2=1,1+5*(icz-1)*(3-icz)
         vvxpl=vvx*dble(iv2-1)/5.d0
        do iy=1,51                                !initialization
        do iz=1,11
         do iqq=1,9
          qfanc(iy,iz,iv,icz+(icz-1)*(3-icz)*(iv1+1+6*(iv2-1)),icdp
     *    +2*(iqq-1))=qfanu(iy,iz,iv,icdp+2*(icz-1),1)
         enddo
        enddo
        enddo

        do iy=2,51
         sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
         rp=(rq(icdp,icz)+alfp*dlog(sy))*4.d0*.0389d0
         do iz=1,11
         do iqq=1,9
          qfanc(iy,iz,iv,icz+(icz-1)*(3-icz)*(iv1+1+6*(iv2-1)),icdp
     *    +2*(iqq-1))=qfanc(iy-1,iz,iv,icz+(icz-1)*(3-icz)*(iv1+1
     *    +6*(iv2-1)),icdp+2*(iqq-1))
         enddo
         enddo

         n=1
8        n=n+1                          !number of t-channel iterations
         nrep=0
         do iz=1,11
          if(iz.gt.6)then
           z=.2d0*dble(iz-6)
           b=dsqrt(-dlog(z)*rp)
          else
           b=dsqrt(-rp*(dlog(0.2d0)+2.d0*dble(iz-7)))
           z=dexp(-b*b/rp)
          endif
          call qgfanc(sy,b*b,vvx,vvxp,vvxpl,icdp,icz,fann)
          fann(7)=min(fann(7),fann(8))
          do iqq=1,9
           if(fann(iqq).gt.0.d0)then
            qfan0(iz,iqq)=dlog(fann(iqq)/z)
           else
            qfan0(iz,iqq)=min(2.d0*qfanc(iy-1,iz,iv,icz
     *      +(icz-1)*(3-icz)*(iv1+1+6*(iv2-1)),icdp+2*(iqq-1))
     *      -qfanc(iy-2,iz,iv,icz+(icz-1)*(3-icz)*(iv1+1+6*(iv2-1))
     *      ,icdp+2*(iqq-1)),qfanc(iy-1,iz,iv,icz
     *      +(icz-1)*(3-icz)*(iv1+1+6*(iv2-1)),icdp+2*(iqq-1)))
           endif
           qfan0(iz,iqq)=max(qfan0(iz,iqq),-20.d0)
          enddo
         enddo

         do iz=1,11
         do iqq=1,9
          if(abs(qfan0(iz,iqq)-qfanc(iy,iz,iv,icz+(icz-1)*(3-icz)
     *    *(iv1+1+6*(iv2-1)),icdp+2*(iqq-1))).gt.1.d-3)nrep=1
          qfanc(iy,iz,iv,icz+(icz-1)*(3-icz)*(iv1+1+6*(iv2-1))
     *    ,icdp+2*(iqq-1))=qfan0(iz,iqq)
         enddo
         enddo
         if(nrep.eq.1.and.n.lt.50)goto 8

         do iz=1,11
          if(iz.gt.6)then
           z=.2d0*dble(iz-6)
           b=dsqrt(-dlog(z)*rp)
          else
           b=dsqrt(-rp*(dlog(0.2d0)+2.d0*dble(iz-7)))
           z=dexp(-b*b/rp)
          endif
         do iqq=1,9
          if(.not.(qfanc(iy,iz,iv,icz+(icz-1)*(3-icz)*(iv1+1+6*(iv2-1))
     *    ,icdp+2*(iqq-1)).le.0.d0.or.qfanc(iy,iz,iv,icz+(icz-1)
     *    *(3-icz)*(iv1+1+6*(iv2-1)),icdp+2*(iqq-1)).gt.0.d0))
     *    stop'fanc-nan'
         enddo
         enddo
        enddo
        enddo
        enddo
        enddo
       endif
      enddo
      enddo

c-------------------------------------------------
c zigzag fans
      do icz=1,3                                  !hadron class
      do icdp=1,2                                 !diffractive eigenstate
       if(cd(icdp,icz).ne.0.d0)then
        do iy=1,11
         sy=sgap**2*(spmax/sgap**2)**((iy-1)/10.d0)
         rp=(rq(icdp,icz)+alfp*dlog(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*dble(iz-6)
          bb=-dlog(z)*rp
         else
          bb=-rp*(dlog(0.2d0)+2.d0*dble(iz-7))
          z=dexp(-bb/rp)
         endif
        do iv=1,11
         vvxt0=dble(iv-1)/10.d0
        do iv1=1,6
         vvxt=vvxt0+(1.d0-vvxt0)*dble(iv1-1)/5.d0
        do iv2=1,1+5*(icz-1)*(3-icz)
         vvxpt=dble(iv2-1)/5.d0
        do iv3=1,1+5*(icz-1)*(3-icz)
         vvxp0=vvxpt*dble(iv3-1)/5.d0
        do iv4=1,1+5*(icz-1)*(3-icz)
         vvxpl=dble(iv4-1)/5.d0

         dfan=qgrev(sy,bb,vvxt0,vvxt,vvxpt,vvxp0,vvxpl,icdp,icz)
         if(dfan.gt.0.d0)then
          qrev(iy,iz,iv+11*(iv1-1),icz+(icz-1)*(3-icz)
     *    *(iv2+1+6*(iv3-1)+36*(iv4-1)),icdp)=dlog(dfan/z)
         else
          qrev(iy,iz,iv+11*(iv1-1),icz+(icz-1)*(3-icz)*(iv2+1
     *    +6*(iv3-1)+36*(iv4-1)),icdp)=2.d0*qrev(iy-1,iz,iv+11*(iv1-1)
     *    ,icz+(icz-1)*(3-icz)*(iv2+1+6*(iv3-1)+36*(iv4-1)),icdp)
     *    -qrev(iy-2,iz,iv+11*(iv1-1),icz+(icz-1)*(3-icz)*(iv2+1
     *    +6*(iv3-1)+36*(iv4-1)),icdp)
         endif
         qrev(iy,iz,iv+11*(iv1-1),icz+(icz-1)*(3-icz)*(iv2+1+6*(iv3-1)
     *   +36*(iv4-1)),icdp)=max(qrev(iy,iz,iv+11*(iv1-1),icz
     *   +(icz-1)*(3-icz)*(iv2+1+6*(iv3-1)+36*(iv4-1)),icdp),-20.d0)

         if(.not.(qrev(iy,iz,iv+11*(iv1-1),icz+(icz-1)*(3-icz)
     *   *(iv2+1+6*(iv3-1)+36*(iv4-1)),icdp).le.0.d0.or.qrev(iy,iz
     *   ,iv+11*(iv1-1),icz+(icz-1)*(3-icz)*(iv2+1+6*(iv3-1)
     *   +36*(iv4-1)),icdp).gt.0.d0))stop'fanc-nan'
        enddo
        enddo
        enddo
        enddo
        enddo
        enddo
        enddo
       endif
      enddo
      enddo

c-------------------------------------------------
c diffractive fans
      icz=2
      do icdp=1,2
       if(cd(icdp,icz).ne.0.d0)then
        do iy=1,21
         xpomr=(1.d5/sgap**2)**(-dble(iy-1)/20.d0)/sgap**2
          rp=(rq(icdp,icz)-alfp*dlog(xpomr))*2.d0*.0389d0
         if(iy.gt.1)then
          do iy1=1,11
          do iz=1,11
          do iqq=1,3
           qdfan(iy,iy1,iz,icdp,iqq)=qdfan(iy-1,iy1,iz,icdp,iqq)
          enddo
          enddo
          enddo
         endif

         n=0
9        n=n+1
         nrep=0
         do iy1=1,11
          xpomr1=(xpomr*sgap**2)**(dble(11-iy1)/10.d0)/sgap
         do iz=1,11
          if(iz.gt.6)then
           z=.2d0*dble(iz-6)
           b=dsqrt(-dlog(z)*rp)
          else
           b=dsqrt(-rp*(dlog(0.2d0)+2.d0*dble(iz-7)))
           z=dexp(-b*b/rp)
          endif
          call qgdfan(xpomr,xpomr1,b*b,icdp,fann,n)
          do iqq=1,3
           if(fann(iqq).gt.0.d0)then
            qfan2(iy1,iz,iqq)=dlog(fann(iqq)/z)
           else
            qfan2(iy1,iz,iqq)=qfan2(iy1-1,iz,iqq)
           endif
           if(n.gt.1.and.abs(qfan2(iy1,iz,iqq)
     *     -qdfan(iy,iy1,iz,icdp,iqq)).gt.1.d-3)nrep=1
          enddo
         enddo
         enddo

         do iy1=1,11
         do iz=1,11
         do iqq=1,3
          qdfan(iy,iy1,iz,icdp,iqq)=qfan2(iy1,iz,iqq)
         enddo
         enddo
         enddo
         if((n.eq.1.or.nrep.eq.1).and.iy.gt.1)goto 9

         do iy1=1,11
          xpomr1=(xpomr*sgap**2)**(dble(11-iy1)/10.d0)/sgap
          do iz=1,11
           if(iz.gt.6)then
            z=.2d0*dble(iz-6)
            b=dsqrt(-dlog(z)*rp)
           else
            b=dsqrt(-rp*(dlog(0.2d0)+2.d0*dble(iz-7)))
            z=dexp(-b*b/rp)
           endif
           do iqq=1,3
            if(iqq.ne.3)then
             dpx=dexp(qdfan(iy,iy1,iz,icdp,iqq))*z
            else
             dpx=dexp(qdfan(iy,iy1,iz,icdp,iqq))*z
     *       *dlog(xpomr1/xpomr/sgap)
            endif
            if(.not.(qdfan(iy,iy1,iz,icdp,iqq).le.0.d0
     *      .or.qdfan(iy,iy1,iz,icdp,iqq).gt.0.d0))stop'qdfan-nan'
           enddo
          enddo
         enddo
        enddo
       endif
      enddo

c-------------------------------------------------
c integrated Pomeron eikonals
      do icz=1,3
      do icdp=1,2
      do icdt=1,2
       if(cd(icdp,icz).ne.0.d0.and.cd(icdt,2).ne.0.d0)then
        do iy=1,11
         e0n=10.d0**iy
         sy=2.d0*e0n*am(2)+am(2)**2+am(icz)**2
         rp=(rq(icdp,icz)+rq(icdt,2)+alfp*log(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          b=sqrt(-log(z)*rp)
         else
          b=sqrt(-rp*(log(0.2d0)+2.d0*(iz-7)))
          z=exp(-b*b/rp)
         endif

         vsoft=gsoft(icz)*sy**dels/rp*cd(icdp,icz)*cd(icdt,2)
         vgg=qgfsh(sy,b*b,icdp,icdt,icz,0)
         vqg=qgfsh(sy,b*b,icdp,icdt,icz,1)
         vgq=qgfsh(sy,b*b,icdp,icdt,icz,2)
         vqq=qghard(sy,b*b,icdp,icdt,icz)

         qxp=vsoft*z+vgg+vqg+vgq+vqq
         do iv=1,6
          vvx=(iv-1)/5.d0
         do iv1=1,1+5*(icz-1)*(3-icz)
          vvxp=(iv1-1)/5.d0
         do iv2=1,6
          vvxt=(iv2-1)/5.d0

          v3p=qg3pom(sy,b,vvx,vvxp,vvxt,icdp,icdt,icz)
          v1p=qgpcut(sy,b,vvx,vvxp,vvxt,icdp,icdt,icz)
          qxp1=qxp+v3p
          qxpc=qxp+v1p
          if(qxp1.gt.0.d0)then
           qpomr(iy,iz,iv+6*(iv1-1)+36*(iv2-1)
     *     ,icdp+2*(icdt-1)+4*(icz-1),1)=log(qxp1/z)
          else
           qpomr(iy,iz,iv+6*(iv1-1)+36*(iv2-1),icdp+2*(icdt-1)
     *     +4*(icz-1),1)=min(2.d0*qpomr(iy-1,iz,iv+6*(iv1-1)
     *     +36*(iv2-1),icdp+2*(icdt-1)+4*(icz-1),1)-qpomr(iy-2,iz
     *     ,iv+6*(iv1-1)+36*(iv2-1),icdp+2*(icdt-1)+4*(icz-1),1)
     *     ,qpomr(iy-1,iz,iv+6*(iv1-1)+36*(iv2-1)
     *     ,icdp+2*(icdt-1)+4*(icz-1),1))
          endif
          if(qxpc.gt.0.d0)then
           qpomr(iy,iz,iv+6*(iv1-1)+36*(iv2-1)
     *     ,icdp+2*(icdt-1)+4*(icz-1),2)=log(qxpc/z)
          else
           qpomr(iy,iz,iv+6*(iv1-1)+36*(iv2-1),icdp+2*(icdt-1)
     *     +4*(icz-1),2)=min(2.d0*qpomr(iy-1,iz,iv+6*(iv1-1)
     *     +36*(iv2-1),icdp+2*(icdt-1)+4*(icz-1),2)-qpomr(iy-2,iz
     *     ,iv+6*(iv1-1)+36*(iv2-1),icdp+2*(icdt-1)+4*(icz-1),2)
     *     ,qpomr(iy-1,iz,iv+6*(iv1-1)+36*(iv2-1)
     *     ,icdp+2*(icdt-1)+4*(icz-1),2))
          endif

          do iqq=1,2
           qpomr(iy,iz,iv+6*(iv1-1)+36*(iv2-1),icdp+2*(icdt-1)
     *     +4*(icz-1),iqq)=max(qpomr(iy,iz,iv+6*(iv1-1)+36*(iv2-1)
     *     ,icdp+2*(icdt-1)+4*(icz-1),iqq),-20.d0)

           if(.not.(qpomr(iy,iz,iv+6*(iv1-1)+36*(iv2-1),icdp+2*(icdt-1)
     *     +4*(icz-1),iqq).le.0.d0.or.qpomr(iy,iz,iv+6*(iv1-1)
     *     +36*(iv2-1),icdp+2*(icdt-1)+4*(icz-1),iqq).gt.0.d0))
     *     stop'qpomr-nan'
          enddo
         enddo
         enddo
         enddo
        enddo
        enddo
       endif
      enddo
      enddo
      enddo

c-------------------------------------------------
c interaction cross sections
      ia(1)=1
      do iy=1,10
       e0n=10.d0**iy                               !interaction energy
       scm=2.d0*e0n*am(2)+am(2)**2+am(icz)**2

       do iiz=1,3
        icz=iiz                                    !hadron class
        rp=(rq(1,icz)+rq(1,2)+alfp*log(scm))*4.d0*.0389d0 !slope (in fm^2)
        g0=pi*rp*10.d0                  !factor for cross-sections (in mb)

        do iia=1,6
         if(iia.le.4)then
          ia(2)=4**(iia-1)                         !target mass number
         elseif(iia.eq.5)then
          ia(2)=14
         else
          ia(2)=40
         endif
         if(debug.ge.1)write (moniou,206)e0n,ty(icz),ia(2)
c-------------------------------------------------
c nuclear densities
         if(ia(2).lt.10)then                       !light nuclei - gaussian
          rnuc(2)=.9d0*float(ia(2))**.3333         !nuclear radius
          wsnuc(2)=amws                            !not used
          wbnuc(2)=0.d0                            !not used
         elseif(ia(2).le.56)then                   !3-parameter Fermi
          rnuc(2)=trnuc(ia(2))                     !nuclear radius
          wsnuc(2)=twsnuc(ia(2))                   !diffuseness
          wbnuc(2)=twbnuc(ia(2))                   !'wine-bottle' parameter
         else                                      !2-parameter Fermi
c rnuc - wood-saxon density radius (fit to the data of murthy et al.)
          rnuc(2)=1.19d0*dble(ia(2))**(1.d0/3.d0)
     *    -1.38d0*dble(ia(2))**(-1.d0/3.d0)        !nuclear radius
          wsnuc(2)=amws                            !diffuseness
          wbnuc(2)=0.d0                            !not used
         endif

         if(ia(2).eq.1)then               !hadron-proton interaction
          call qgfz(0.d0,gz0,0,0)
          gtot=gz0(1)                     !total cross-section
          gin=(gz0(2)+gz0(3)+gz0(4))*.5d0 !inelastic cross section
          bel=gz0(5)                      !elastic scattering slope
          gel=gtot-gin                    !elastic cross section
          gdp=gz0(3)*.5d0     !projectile low mass diffr. (+double LMD)
          gdt=gz0(4)*.5d0                 !target low mass diffraction
          if(iy.le.10)gsect(iy,icz,iia)=log(gin)

          if(debug.ge.1)write (moniou,225)gtot,gin,gel,gdp,gdt,bel
         else                             !hadron-nucleus interaction
          bm=rnuc(2)+dlog(29.d0)*wsnuc(2) !for numerical integration
          anorm=qganrm(rnuc(2),wsnuc(2),wbnuc(2))*rp !density normalization
          call qggau(gz1)                 !integration over b<bm
          call qggau1(gz1)                !integration over b>bm
          gin=gz1(1)+gz1(2)+gz1(3)        !inelastic cross section
          if(iy.le.10)gsect(iy,icz,iia)=log(gin*10.d0)

          if(debug.ge.1)write (moniou,224)
     *    gin*10.d0,gz1(3)*10.d0,gz1(2)*10.d0
         endif
         if(.not.(gsect(iy,icz,iia).le.0.d0
     *   .or.gsect(iy,icz,iia).gt.0.d0))stop'qpomr-nan'
        enddo
       enddo
      enddo

c-------------------------------------------------
c cut Pomeron leg eikonals
      if(debug.ge.1)write (moniou,223)
      do icz=1,3                                    !hadron class
      do icdp=1,2
       if(cd(icdp,icz).ne.0.d0)then
        do iy=1,51
         sy=sgap**2*(spmax/sgap**2)**((iy-1)/50.d0)
         rp=(rq(icdp,icz)+alfp*log(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          bb=-log(z)*rp                             !impact parameter^2
         else
          bb=-rp*(log(0.2d0)+2.d0*(iz-7))
          z=exp(-bb/rp)
         endif
        do ix=1,10
         if(ix.le.5)then
          xp=.2d0*(5.d0*sgap/sy)**((6-ix)/5.d0)     !Pomeron LC+ momentum
         else
          xp=.2d0*(ix-5)
         endif
         sys=xp*sy

         vs=qgls(sys,xp,bb,icdp,icz)
         vg=qglsh(sys,xp,bb,icdp,icz,0)
         if(xp.lt..99d0)then
          vq=qglsh(sys,xp,bb,icdp,icz,1)
     *    /dsqrt(xp)*(1.d0-xp)**(ahv(icz)-ahl(icz))
         else
          vq=0.d0
         endif
         qlegc0(iy,ix,iz,icdp+2*(icz-1),1)=dlog((vs+vg+vq)/vs)
         qlegc0(iy,ix,iz,icdp+2*(icz-1),2)=dlog((vs+vg)/vs)
        enddo
        enddo
        enddo
       endif
      enddo
      enddo

      do icz=1,3                                    !hadron class
      do icdp=1,2
       if(cd(icdp,icz).ne.0.d0)then
        do iy=1,51
         sy=sgap**2*(spmax/sgap**2)**((iy-1)/50.d0)
         rp=(rq(icdp,icz)+alfp*log(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          bb=-log(z)*rp                             !impact parameter^2
         else
          bb=-rp*(log(0.2d0)+2.d0*(iz-7))
          z=exp(-bb/rp)
         endif
        do ix=1,10
         if(ix.le.5)then
          xp=.2d0*(5.d0*sgap/sy)**((6-ix)/5.d0)     !Pomeron LC+ momentum
         else
          xp=.2d0*(ix-5)
         endif
         sys=xp*sy

         do iqq=1,3
          call qgloolc(sys,xp,bb,icdp,icz,iqq,fann(2*iqq-1)
     *    ,fann(2*iqq))
         enddo
         do iqq=1,6
          if(fann(iqq).gt.0.d0)then
           qlegc0(iy,ix,iz,icdp+2*(icz-1),iqq+2)
     *     =dlog(fann(iqq)/qgls(sys,xp,bb,icdp,icz))
          else
           qlegc0(iy,ix,iz,icdp+2*(icz-1),iqq+2)
     *     =min(2.d0*qlegc0(iy-1,ix,iz,icdp+2*(icz-1),iqq+2)
     *     -qlegc0(iy-2,ix,iz,icdp+2*(icz-1),iqq+2)
     *     ,qlegc0(iy-1,ix,iz,icdp+2*(icz-1),iqq+2))
          endif
          qlegc0(iy,ix,iz,icdp+2*(icz-1),iqq+2)
     *     =max(qlegc0(iy,ix,iz,icdp+2*(icz-1),iqq+2),-20.d0)
         enddo
        enddo
        enddo
        enddo
       endif
      enddo
      enddo

      do icz=1,3                                    !hadron class
      do icdp=1,2                                   !diffr. eigenstate
       if(cd(icdp,icz).ne.0.d0)then
        do iv=1,11
         vvx=dble(iv-1)/10.d0
         do iy=1,51                                 !initialization
         do ix=1,10
         do iz=1,11
          do iqq=1,3
           qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *     =qlegc0(iy,ix,iz,icdp+2*(icz-1),2*iqq+1)
          enddo
         enddo
         enddo
         enddo

         do iy=2,51
          sy=sgap**2*(spmax/sgap**2)**((iy-1)/50.d0)
          rp=(rq(icdp,icz)+alfp*log(sy))*4.d0*.0389d0

          do ix=1,10
          do iz=1,11
          do iqq=1,3
           qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *     =qlegc(iy-1,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
          enddo
          enddo
          enddo

          n=1
43        n=n+1                          !number of t-channel iterations
          nrep=0
          do iz=1,11
           if(iz.gt.6)then
            z=.2d0*(iz-6)
            bb=-log(z)*rp                           !impact parameter^2
           else
            bb=-rp*(log(0.2d0)+2.d0*(iz-7))
            z=exp(-bb/rp)
           endif
          do ix=1,10
           if(ix.le.5)then
            xp=.2d0*(5.d0*sgap/sy)**((6-ix)/5.d0)   !Pomeron LC+ momentum
           else
            xp=.2d0*(ix-5)
           endif
           sys=xp*sy

           do iqq=1,3
            fann(iqq)=qglscr(sys,xp,bb,vvx,icdp,icz,iqq)
            if(fann(iqq).gt.0.d0)then
             qfan2(ix,iz,iqq)=dlog(fann(iqq)/qgls(sys,xp,bb,icdp,icz))
            elseif(iy.gt.2)then
             qfan2(ix,iz,iqq)
     *       =min(2.d0*qlegc(iy-1,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *       -qlegc(iy-2,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *       ,qlegc(iy-1,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1)))
            else
             qfan2(ix,iz,iqq)
     *       =qlegc(iy-1,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
            endif
            qfan2(ix,iz,iqq)=max(qfan2(ix,iz,iqq),-20.d0)
            if(abs(qfan2(ix,iz,iqq)-qlegc(iy,ix,iv,iz
     *      ,icdp+2*(icz-1)+6*(iqq-1))).gt.1.d-3)nrep=1
           enddo
          enddo
          enddo

          do iz=1,11
           if(iz.gt.6)then
            z=.2d0*(iz-6)
            bb=-log(z)*rp                           !impact parameter
           else
            bb=-rp*(log(0.2d0)+2.d0*(iz-7))
            z=exp(-bb/rp)
           endif
          do ix=1,10
           if(ix.le.5)then
            xp=.2d0*(5.d0*sgap/sy)**((6-ix)/5.d0)   !Pomeron LC+ momentum
           else
            xp=.2d0*(ix-5)
           endif
           sys=xp*sy

          do iqq=1,3
           qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))=qfan2(ix,iz,iqq)

           if(.not.(qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1)).le.0.d0
     *     .or.qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1)).gt.0.d0))
     *     stop'qlegc-nan'
          enddo
          enddo
          enddo
          if(nrep.eq.1.and.n.lt.50)goto 43
         enddo
        enddo
       endif
      enddo
      enddo

c soft pre-evolution
      do icz=1,3                                    !hadron class
      do icdp=1,2                                   !diffr. eigenstate
       if(cd(icdp,icz).ne.0.d0)then
        do iv=1,11
         vvx=dble(iv-1)/10.d0
        do iy=1,51
         sy=sgap*(spmax/sgap)**((iy-1)/50.d0)
         rp=(rq(icdp,icz)+alfp*log(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          bb=-log(z)*rp                             !impact parameter
         else
          bb=-rp*(log(0.2d0)+2.d0*(iz-7))
          z=exp(-bb/rp)
         endif
        do ix=1,10
         if(ix.le.5)then
          xp=.2d0*(sgap/sy)**((6-ix)/5.d0)          !Pomeron LC+ momentum
         else
          xp=.2d0*(ix-5)
         endif
         sys=xp*sy

         if(iy.eq.1)then
          qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+18)=0.d0
          qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+24)=0.d0
         else
          do iqq=4,5
           fann(iqq)=qglh(sys,xp,bb,vvx,icdp,icz,iqq-4)
           if(fann(iqq).gt.0.d0)then
            qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *      =dlog(fann(iqq))
           elseif(iy.gt.2)then
            qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *      =min(2.d0*qlegc(iy-1,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *      -qlegc(iy-2,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *      ,qlegc(iy-1,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1)))
           else
            qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *      =qlegc(iy-1,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
           endif
           qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1))
     *     =max(qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1)),-20.d0)

           if(.not.(qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1)).le.0.d0
     *     .or.qlegc(iy,ix,iv,iz,icdp+2*(icz-1)+6*(iqq-1)).gt.0.d0))
     *     stop'qlegc-nan'
          enddo
         endif
        enddo
        enddo
        enddo
        enddo
       endif
      enddo
      enddo

c-------------------------------------------------
c cut Pomeron eikonals
      if(debug.ge.1)write (moniou,226)
      do icz=1,3                                    !proj. class
      do icdp=1,2
      do icdt=1,2
       if(cd(icdp,icz).ne.0.d0.and.cd(icdt,2).ne.0.d0)then
        do iy=1,11
         sy=sgap**2*(spmax/sgap**2)**((iy-1)/10.d0)
         rp=(rq(icdp,icz)+rq(icdt,2)+alfp*log(sy))*4.d0*.0389d0
        do iz=1,11
         if(iz.gt.6)then
          z=.2d0*(iz-6)
          bb=-log(z)*rp                             !impact parameter^2
         else
          bb=-rp*(log(0.2d0)+2.d0*(iz-7))
          z=exp(-bb/rp)
         endif
        do iv=1,11
         vvx=(iv-1)/10.d0                           !relative scr. strenth

        do ix1=1,10
         if(ix1.le.5)then
          xp=.2d0*(5.d0*sgap/sy)**((6-ix1)/5.d0)    !Pomeron LC+ momentum
         else
          xp=.2d0*(ix1-5)
         endif
        do ix2=1,10
         if(ix2.le.5)then
          xm=.2d0*(sgap/sy/xp)**((6-ix2)/5.d0)      !Pomeron LC- momentum
         else
          xm=.2d0*(ix2-5)
         endif
         sys=xp*xm*sy

         do iqq=1,4
          vv=qgcutp(sys,xp,xm,bb,vvx,icdp,icdt,icz,iqq)
          if(vv.gt.0.d0)then
           qpomc(iy,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)+4*(icz-1)
     *     +12*(iqq-1))=dlog(vv/z)
          elseif(iy.gt.2)then
           qpomc(iy,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)+4*(icz-1)
     *     +12*(iqq-1))=min(2.d0*qpomc(iy-1,ix1+10*(ix2-1),iz,iv
     *     ,icdp+2*(icdt-1)+4*(icz-1)+12*(iqq-1))-qpomc(iy-2
     *     ,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)+4*(icz-1)+12*(iqq-1))
     *     ,qpomc(iy-1,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)+4*(icz-1)
     *     +12*(iqq-1)))
          else
           qpomc(iy,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)+4*(icz-1)
     *     +12*(iqq-1))=qpomc(iy-1,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)
     *     +4*(icz-1)+12*(iqq-1))
          endif
           qpomc(iy,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)+4*(icz-1)
     *     +12*(iqq-1))=max(qpomc(iy,ix1+10*(ix2-1),iz,iv,icdp
     *     +2*(icdt-1)+4*(icz-1)+12*(iqq-1)),-20.d0)

          if(.not.(qpomc(iy,ix1+10*(ix2-1),iz,iv,icdp+2*(icdt-1)+4*(icz
     *    -1)+12*(iqq-1)).le.0.d0.or.qpomc(iy,ix1+10*(ix2-1),iz,iv,icdp
     *    +2*(icdt-1)+4*(icz-1)+12*(iqq-1)).gt.0.d0))stop'qpomc-nan'
         enddo
        enddo
        enddo
        enddo
        enddo
        enddo
       endif
      enddo
      enddo
      enddo

c-----------------------------------------------------------------------------
c timelike Sudakov formfactor
      if(debug.ge.1)write (moniou,221)
      do m=1,2                     !parton type (1-g, 2-q)
       fsud(1,m)=0.d0
      do k=2,10
       qmax=qtf*4.d0**(1.d0+k)     !effective virtuality (qt**2/z**2/(1-z)**2)
       fsud(k,m)=qgsudt(qmax,m)
      enddo
      enddo
c-----------------------------------------------------------------------------
c effective virtuality (used for inversion in timelike branching)
      if(debug.ge.1)write (moniou,222)
      do m=1,2                     !parton type (1-g, 2-q)
      do k=1,10
       qlmax=1.38629d0*(k-1)
       qrt(k,1,m)=0.d0
       qrt(k,101,m)=qlmax
      do i=1,99                    !bins in Sudakov formfactor
       if(k.eq.1)then
        qrt(k,i+1,m)=0.d0
       else
        qrt(k,i+1,m)=qgroot(qlmax,.01d0*i,m)
       endif
      enddo
      enddo
      enddo

c-----------------------------------------------------------------------------
c writing cross sections to the file
      if(debug.ge.1)write (moniou,220)
      if(ifIIdat.ne.1)then
       open(1,file=DATDIR(1:INDEX(DATDIR,' ')-1)//'qgsdat-II-04'
     * ,status='unknown')
      else                                              !used to link with nexus
       open(ifIIdat,file=fnIIdat(1:nfnIIdat),status='unknown')
      endif
      write (1,*)csborn,cs0,cstot,evk,qpomi,qpomis,qlegi,qfanu,qfanc
     *,qdfan,qpomr,gsect,qlegc0,qlegc,qpomc,fsud,qrt,qrev,fsud,qrt
      close(1)

10    continue
c-----------------------------------------------------------------------------
c nuclear cross sections
      if(ifIIncs.ne.2)then
       inquire(file=DATDIR(1:INDEX(DATDIR,' ')-1)//'sectnu-II-04'
     * ,exist=lcalc)
      else                                                  !ctp
       inquire(file=fnIIncs(1:nfnIIncs),exist=lcalc)
      endif

      if(lcalc)then
       if(debug.ge.0)write (moniou,207)
       if(ifIIncs.ne.2)then
        open(2,file=DATDIR(1:INDEX(DATDIR,' ')-1)//'sectnu-II-04'
     *  ,status='old')
       else                                                  !ctp
        open(ifIIncs,file=fnIIncs(1:nfnIIncs),status='old')
       endif
       read (2,*)qgsasect
       close(2)

      elseif(.not.producetables)then
        write(moniou,*) "Missing sectnu-II-04 file !"        
        write(moniou,*) "Please correct the defined path ",
     &"or force production ..."
        stop

      else
       niter=5000                   !number of iterations
       do ie=1,10
        e0n=10.d0**ie               !interaction energy (per nucleon)
       do iia1=1,6
        iap=2**iia1                 !proj. mass number
       do iia2=1,6
        if(iia2.le.4)then
         iat=4**(iia2-1)            !targ. mass number
        elseif(iia2.eq.5)then
         iat=14
        else
         iat=40
        endif
        if(debug.ge.1)write (moniou,208)e0n,iap,iat

        call qgini(e0n,2,iap,iat)
        call qgcrossc(niter,gtot,gprod,gabs,gdd,gqel,gcoh)
        if(debug.ge.1)write (moniou,209)gtot,gprod,gabs,gdd,gqel,gcoh
        qgsasect(ie,iia1,iia2)=log(gprod)
       enddo
       enddo
       enddo
       if(ifIIncs.ne.2)then
        open(2,file=DATDIR(1:INDEX(DATDIR,' ')-1)//'sectnu-II-04'
     *  ,status='unknown')
       else                                                  !ctp
        open(ifIIncs,file=fnIIncs(1:nfnIIncs),status='unknown')
       endif
       write (2,*)qgsasect
       close(2)
      endif

      if(debug.ge.3)write (moniou,218)
201   format(2x,'qgaini: hard cross sections calculation')
202   format(2x,'qgaini: number of rungs considered:',i2
     */4x,'starting energy bin for ordered and general ladders:',3i4)
205   format(2x,'qgaini: pretabulation of the interaction eikonals')
206   format(2x,'qgaini: initial particle energy:',e10.3,2x
     *,'its type:',a7,2x,'target mass number:',i2)
207   format(2x,'qgaini: nuclear cross sections readout from the file'
     *,' sectnu-II-04')
208   format(2x,'qgaini: initial nucleus energy:',e10.3,2x
     *,'projectile mass:',i2,2x,'target mass:',i2)
209   format(2x,'gtot',d10.3,'  gprod',d10.3,'  gabs',d10.3
     */2x,'gdd',d10.3,'  gqel',d10.3,' gcoh',d10.3)
210   format(2x,'qgaini - main initialization procedure')
212   format(2x,'qgaini: integrated Pomeron leg eikonals')
213   format(2x,'qgaini: integrated fan contributions')
214   format(2x,'qgaini: cross sections readout from the file: ', A,2x)
c     *,' qgsdat-II-04')
215   format(2x,'qgaini: integrated cut fan contributions')
c216   format(2x,'qgaini: integrated cut Pomeron eikonals')
218   format(2x,'qgaini - end')
220   format(2x,'qgaini: cross sections are written to the file'
     *,' qgsdat-II-04')
221   format(2x,'qgaini: timelike Sudakov formfactor')
222   format(2x,'qgaini: effective virtuality for inversion')
223   format(2x,'qgaini: cut Pomeron leg eikonals')
224   format(2x,'qgaini: hadron-nucleus cross sections:'
     */4x,'gin=',e10.3,2x,'gdifr_targ=',e10.3,2x
     *,'gdifr_proj=',e10.3)
225   format(2x,'qgaini: hadron-proton cross sections:'
     */4x,'gtot=',e10.3,2x,'gin=',e10.3,2x,'gel=',e10.3/4x
     *,'gdifrp=',e10.3,2x,'gdifrt=',e10.3,2x,'b_el=',e10.3,2x)
226   format(2x,'qgaini: cut Pomeron eikonals (semi-hard)')
      return
      end

c=============================================================================
      subroutine qgini(e0n,icp0,iap,iat)
c-----------------------------------------------------------------------------
c additional initialization procedure
c e0n  - interaction energy (per hadron/nucleon),
c icp0 - hadron type (+-1 - pi+-, +-2 - p(p~), +-3 - n(n~),
c                     +-4 - K+-, +-5 - K_l/s),
c iap  - projectile mass number (1 - for a hadron),
c iat  - target mass number
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr4/  ey0(3)
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr6/  pi,bm,amws
      common /qgarr7/  xa(iapmax,3),xb(iapmax,3),b
      common /qgarr10/ am(7),ammu
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr43/ moniou
      common /arr1/    trnuc(56),twsnuc(56),twbnuc(56)
      common /qgdebug/ debug
      common /qgsIInex1/xan(iapmax,3),xbn(iapmax,3)  !used to link with nexus
     *,bqgs,bmaxqgs,bmaxnex,bminnex

      if(debug.ge.1)write (moniou,201)icp0,iap,iat,e0n
      icp=icp0
      ia(1)=iap
      ia(2)=iat

      icz=iabs(icp)/2+1  !!!!!particle class (1 - pion, 2 - nucleon, 3 - kaon)

      scm=2.d0*e0n*am(2)+am(2)**2+am(icz)**2   !c.m. energy squared
      ey0(1)=dsqrt(scm)/(e0n+am(2)+dsqrt(e0n-am(icz))
     **dsqrt(e0n+am(icz)))                     !Lorentz boost to lab. frame
      ey0(2)=1.d0
      ey0(3)=1.d0
      wp0=dsqrt(scm)                           !initial LC+ mometum
      wm0=wp0                                  !initial LC- mometum

c-------------------------------------------------
c nuclear radii and weights for nuclear configurations - procedure qggea
      do i=1,2
       if(ia(i).lt.10.and.ia(i).ne.1)then      !gaussian density
        rnuc(i)=.9d0*float(ia(i))**.3333       !nuclear radius
        if(ia(i).eq.2)rnuc(i)=3.16d0
c rnuc -> rnuc * a / (a-1) - to use van-hove method (in qggea)
        rnuc(i)=rnuc(i)*dsqrt(2.d0*ia(i)/(ia(i)-1.d0))
                           !rnuc -> rnuc*a/(a-1) - to use Van-Hove method
      elseif(ia(i).ne.1)then
        if(ia(i).le.56)then                    !3-parameter Fermi
         rnuc(i)=trnuc(ia(i))                  !nuclear radius
         wsnuc(i)=twsnuc(ia(i))                !diffuseness
         wbnuc(i)=twbnuc(ia(i))                !'wine-bottle' parameter
        else
         rnuc(i)=1.19*float(ia(i))**(1./3.)-1.38*float(ia(i))**(-1./3.)
         wsnuc(i)=amws                         !diffuseness
         wbnuc(i)=0.d0
        endif
        cr1(i)=1.d0+3.d0/rnuc(i)*wsnuc(i)+6.d0/(rnuc(i)/wsnuc(i))**2
     *  +6.d0/(rnuc(i)/wsnuc(i))**3
        cr2(i)=3.d0/rnuc(i)*wsnuc(i)
        cr3(i)=3.d0/rnuc(i)*wsnuc(i)+6.d0/(rnuc(i)/wsnuc(i))**2
       endif
      enddo

      if(ia(1).ne.1)then                              !primary nucleus
       bm=rnuc(1)+rnuc(2)+5.d0*max(wsnuc(1),wsnuc(2)) !b-cutoff
      elseif(ia(2).ne.1)then                          !hadron-nucleus
       bm=rnuc(2)+5.d0*wsnuc(2)                       !b-cutoff
      else                                            !hadron-proton
       bm=3.d0*dsqrt((rq(1,icz)+rq(1,2)+alfp*log(scm))*4.d0*.0398d0)
      endif

      bmaxqgs=bm                                      !used to link with nexus

      if(debug.ge.3)write (moniou,202)
201   format(2x,'qgini - miniinitialization: particle type icp0=',
     *i2,2x,'projectile mass number iap=',i2/4x,
     *'target mass number iat=',i2,' interaction energy e0n=',e10.3)
202   format(2x,'qgini - end')
      return
      end

c=============================================================================
      double precision function qgpint(sy,bb)
c-----------------------------------------------------------------------------
c qgpint - interm. Pomeron eikonal
c sy  - pomeron mass squared,
c bb  - impact parameter squared
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)sy,bb

      qgpint=0.d0
      s2min=4.d0*fqscal*qt0
      xmin=s2min/sy
      if(xmin.ge.1.d0)return

      xmin=xmin**(delh-dels)
c numerical integration over z1
      do i=1,7
      do m=1,2
       z1=(.5d0*(1.d0+xmin-(2*m-3)*x1(i)*(1.d0-xmin)))
     * **(1.d0/(delh-dels))
       ww=z1*sy
       sjqq=qgjit(qt0,qt0,ww,2,2)  !inclusive qq cross-section
       sjqg=qgjit(qt0,qt0,ww,1,2)  !inclusive qg cross-section
       sjgg=qgjit(qt0,qt0,ww,1,1)  !inclusive gg cross-section

       st2=0.d0
       do j=1,7
       do k=1,2
        xx=.5d0*(1.d0+x1(j)*(2*k-3))
        xp=z1**xx
        xm=z1/xp
        glu1=qgppdi(xp,0)
        sea1=qgppdi(xp,1)
        glu2=qgppdi(xm,0)
        sea2=qgppdi(xm,1)
        st2=st2+a1(j)*(glu1*glu2*sjgg+(glu1*sea2+glu2*sea1)*sjqg
     *  +sea1*sea2*sjqq)
       enddo
       enddo
       rh=-alfp*dlog(z1)
       qgpint=qgpint-a1(i)*dlog(z1)/z1**delh*st2
     * *exp(-bb/(4.d0*.0389d0*rh))/rh
      enddo
      enddo
      qgpint=qgpint*(1.d0-xmin)/(delh-dels)*factk*rr**2/2.d0*pi

      if(debug.ge.3)write (moniou,202)qgpint
201   format(2x,'qgpint - interm. Pomeron eikonal:'
     */4x,'sy=',e10.3,2x,'bb=',e10.3)
202   format(2x,'qgpint=',e10.3)
      return
      end

c------------------------------------------------------------------------
      double precision function qgpini(sy,bb,vvx,vvxt,iqq)
c-----------------------------------------------------------------------
c qgpini - intermediate gg-Pomeron eikonal
c sy   - pomeron mass squared,
c bb   - impact parameter squared,
c vvx  - total / projectile screening factor,
c vvxt - target screening factor
c vvx  - total/projectile screening factor:
c vvx  = 0                                                    (iqq=1,...15)
c vvx  = 1 - exp[-2*sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]  (iqq=16)
c vvx  = 1 + exp[-2*sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]
c          - exp[-2*sum_{i} chi_proj(i)-sum_j chi_targ(j)]
c          - exp[-sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]    (iqq=17 uncut)
c vvx  = 1 - exp[-sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]    (iqq=17,...19)
c vvx  = 1 - exp[-sum_{i} chi_proj(i)]                        (iqq=20,...23)
c vvxt - target screening factor:
c vvxt = 0                                                    (iqq=1,...19)
c vvxt = 1 - exp[-sum_j chi_targ(j)]                          (iqq=20,...23)
c uncut eikonals:
c iqq=0  - single soft Pomeron
c iqq=1  - single Pomeron
c iqq=2  - general loop contribution
c iqq=3  - single Pomeron end on one side
c iqq=4  - single Pomeron ends on both sides
c cut eikonals:
c iqq=5  - single cut Pomeron
c iqq=6  - single cut Pomeron with single end
c iqq=7  - single cut Pomeron with 2 single ends
c iqq=8  - any cuts except the complete rap-gap
c iqq=9  - single cut Pomeron end at one side
c iqq=10 - single cut Pomeron end at one side and single Pomeron on the other
c iqq=11 - no rap-gap at one side
c iqq=12 - no rap-gap at one side and single Pomeron on the other
c iqq=13 - single cut soft Pomeron
c iqq=14 - single cut soft Pomeron with single end
c iqq=15 - single cut soft Pomeron with 2 single ends
c  with proj/targ screening corrections:
c iqq=16 - single cut Pomeron
c iqq=17 - uncut / cut end / loop sequence
c iqq=18 - no rap-gap at the end
c iqq=19 - single cut Pomeron end
c iqq=20 - diffractive cut, Puu
c iqq=21 - diffractive cut, Puu-Puc
c iqq=22 - diffractive cut, Pcc
c iqq=23 - diffractive cut, Pcc+Pcu
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wz(3),wi(3),wj(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr20/ spmax
      common /qgarr26/ factk,fqscal
      common /qgarr39/ qpomi(51,11,15),qpomis(51,11,11,11,9)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qgpini=0.d0
      pinm=0.d0
      s2min=4.d0*fqscal*qt0
      rp=alfp*dlog(sy)*4.d0*.0389d0
      z=exp(-bb/rp)
      if(iqq.le.1.and.(sy.le.s2min.or.iqq.eq.0))goto 1

      yl=log(sy/sgap)/log(spmax/sgap)*50.d0+1.d0
      k=max(1,int(1.00001d0*yl-1.d0))
      k=min(k,49)
      wk(2)=yl-k
      if(yl.le.2.d0)then
       iymax=2
       wk(1)=1.d0-wk(2)
      else
       wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
       wk(1)=1.d0-wk(2)+wk(3)
       wk(2)=wk(2)-2.d0*wk(3)
       iymax=3
      endif

      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-dlog(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      if(iqq.le.15)then
       iqr=iqq
       if(sy.le.sgap**2.and.iqq.le.12)iqr=1
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,iymax
        k2=k+k1-1
        qgpini=qgpini+qpomi(k2,l2,iqr)*wk(k1)*wz(l1)
       enddo
       enddo
       if(zz.lt.1.d0)then
        do k1=1,iymax
         k2=k+k1-1
         pinm=pinm+qpomi(k2,1,iqr)*wk(k1)
        enddo
        qgpini=min(qgpini,pinm)
       endif

      else
       vi=vvx*10.d0+1.d0
       i=max(1,int(vi))
       i=min(i,9)
       wi(2)=vi-i
       wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
       wi(1)=1.d0-wi(2)+wi(3)
       wi(2)=wi(2)-2.d0*wi(3)

       if(iqq.le.19)then
        do i1=1,3
         i2=i+i1-1
        do l1=1,izmax
         l2=jz+l1-1
        do k1=1,iymax
         k2=k+k1-1
         qgpini=qgpini+qpomis(k2,l2,i2,1,iqq-15)*wk(k1)*wz(l1)*wi(i1)
        enddo
        enddo
        enddo
        if(zz.lt.1.d0)then
         do i1=1,3
          i2=i+i1-1
         do k1=1,iymax
          k2=k+k1-1
          pinm=pinm+qpomis(k2,1,i2,1,iqq-15)*wk(k1)*wi(i1)
         enddo
         enddo
         qgpini=min(qgpini,pinm)
        endif

       else
        vj=vvxt*10.d0+1.d0
        j=max(1,int(vj))
        j=min(j,9)
        wj(2)=vj-j
        wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
        wj(1)=1.d0-wj(2)+wj(3)
        wj(2)=wj(2)-2.d0*wj(3)
        jmax=3

        do j1=1,jmax
         j2=j+j1-1
        do i1=1,3
         i2=i+i1-1
        do l1=1,izmax
         l2=jz+l1-1
        do k1=1,iymax
         k2=k+k1-1
         qgpini=qgpini+qpomis(k2,l2,i2,j2,iqq-15)
     *   *wk(k1)*wz(l1)*wi(i1)*wj(j1)
        enddo
        enddo
        enddo
        enddo
        if(zz.lt.1.d0)then
         do j1=1,jmax
          j2=j+j1-1
         do i1=1,3
          i2=i+i1-1
         do k1=1,iymax
          k2=k+k1-1
          pinm=pinm+qpomis(k2,1,i2,j2,iqq-15)*wk(k1)*wi(i1)*wj(j1)
         enddo
         enddo
         enddo
         qgpini=min(qgpini,pinm)
        endif
       endif
      endif
1     qgpini=exp(qgpini)
      if(iqq.le.16.or.iqq.eq.19)qgpini=qgpini
     **sy**dels*sigs*g3p**2*z/rp*4.d0*.0389d0
      return
      end

c=============================================================================
      double precision function qgleg(sy,bb,icdp,icz)
c-----------------------------------------------------------------------------
c qgleg - integrated Pomeron leg eikonal
c sy  - pomeron mass squared,
c bb  - impact parameter squared,
c icz - hadron class
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr25/ ahv(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)sy,bb,icz

      qgleg=0.d0
      if(sy.lt.1.001d0)then
       tmin=1.d0
      else
       tmin=(1.d0-(1.d0-1.d0/sy)**(1.+ahl(icz)))**(1.+dels)
      endif
      if(debug.ge.3)write (moniou,203)tmin
      do i1=1,7
      do m1=1,2
       tp=1.d0-(.5d0*(1.d0+tmin)+x1(i1)*(m1-1.5d0)*(1.d0-tmin))
     * **(1./(1.+dels))
       if(tp.gt.1.d-9)then
        xp=1.d0-tp**(1.d0/(1.d0+ahl(icz)))
       else
        xp=1.d0
       endif

       ws=qgls(xp*sy,xp,bb,icdp,icz)
       wg=qglsh(xp*sy,xp,bb,icdp,icz,0)
       wq=qglsh(xp*sy,xp,bb,icdp,icz,1)/dsqrt(xp)
     * *(1.d0-xp)**(ahv(icz)-ahl(icz))

       qgleg=qgleg+a1(i1)*(ws+wg+wq)/(1.d0-tp)**dels
      enddo
      enddo
      qgleg=qgleg/2.d0/(1.+ahl(icz))/(1.d0+dels)

      if(debug.ge.3)write (moniou,202)qgleg
201   format(2x,'qgleg - Pomeron leg eikonal:'
     */4x,'s=',e10.3,2x,'b^2=',e10.3,2x,'icz=',i1)
202   format(2x,'qgleg=',e10.3)
203   format(2x,'qgleg:',2x,'tmin=',e10.3)
      return
      end

c------------------------------------------------------------------------
      double precision function qglegi(sy,bb,icdp,icz,iqq)
c-----------------------------------------------------------------------
c qglegi - integrated Pomeron leg eikonal
c sy   - pomeron mass squared,
c bb   - impact parameter squared,
c icdp - diffractive state for the hadron,
c icz  - hadron class
c iqq=1 - single leg Pomeron
c iqq=2 - all loops
c iqq=3 - single Pomeron end
c iqq=4 - single cut Pomeron
c iqq=5 - single cut Pomeron with single Pomeron end
c iqq=6 - single cut Pomeron end
c iqq=7 - no rap-gap at the end
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wz(3)
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr27/ qlegi(51,11,2,3,7),qfanu(51,11,11,6,2)
     *,qfanc(51,11,11,39,18),qdfan(21,11,11,2,3),qrev(11,11,66,219,2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qglegi=0.d0
      xlegm=0.d0
      rp=(rq(icdp,icz)+alfp*log(sy))*4.d0*.0389d0
      z=exp(-bb/rp)
      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-dlog(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      yl=log(sy/sgap)/log(spmax/sgap)*50.d0+1.d0
      k=max(1,int(yl))
      k=min(k,49)
      wk(2)=yl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)
      iymax=3

      iqr=iqq
      if(sy.le.sgap**2)iqr=1
      do l1=1,izmax
       l2=jz+l1-1
      do k1=1,iymax
       k2=k+k1-1
       qglegi=qglegi+qlegi(k2,l2,icdp,icz,iqr)*wk(k1)*wz(l1)
      enddo
      enddo
      if(zz.lt.1.d0)then
       do k1=1,iymax
        k2=k+k1-1
        xlegm=xlegm+qlegi(k2,1,icdp,icz,iqr)*wk(k1)
       enddo
       qglegi=min(qglegi,xlegm)
      endif
      qglegi=exp(qglegi)*z
     **(1.d0-(1.d0-(1.d0-1.d0/sy)**(1.+ahl(icz)))**(1.+dels))
      return
      end

c=============================================================================
      double precision function qgls(sy,xp,bb,icdp,icz)
c-----------------------------------------------------------------------------
c qgls - soft pomeron leg eikonal
c sy   - pomeron mass squared,
c xp   - pomeron light cone momentum,
c bb   - impact parameter squared,
c icdp - diffractive state for the connected hadron,
c icz  - hadron class
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)sy,bb,icz

      rp=rq(icdp,icz)+alfp*log(sy/xp)
      qgls=sy**dels*fp(icz)*g3p*sigs/rp*exp(-bb/(4.d0*.0389d0*rp))
     **cd(icdp,icz)

      if(debug.ge.3)write (moniou,202)qgls
201   format(2x,'qgls - soft pomeron leg eikonal:'
     */4x,'sy=',e10.3,2x,'b^2=',e10.3,2x,'icz=',i1)
202   format(2x,'qgls=',e10.3)
      return
      end

c=============================================================================
      double precision function qglsh(sy,xp,bb,icdp,icz,iqq)
c-----------------------------------------------------------------------------
c qglsh - unintegrated Pomeron leg eikonal
c sy  - pomeron mass squared,
c xp  - light cone momentum share,
c bb  - impact parameter squared,
c icz - hadron class
c iqq=0 - gluon/sea quark contribution,
c iqq=1 - valence quark contribution
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr25/ ahv(3)
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)sy,bb,icz

      qglsh=0.d0
      s2min=4.d0*fqscal*qt0
      if(sy.lt.1.001d0*s2min)return

      xmin=(s2min/sy)**(delh-dels)
c numerical integration over zh
      do i1=1,7
      do m1=1,2
       zh=(.5d0*(1.d0+xmin-(2*m1-3)*x1(i1)*(1.d0-xmin)))
     * **(1.d0/(delh-dels))
       ww=zh*sy         !c.m. energy squared for hard interaction
       sjqq=qgjit(qt0,qt0,ww,2,2)
       sjqg=qgjit(qt0,qt0,ww,1,2)
       sjgg=qgjit(qt0,qt0,ww,1,1)
       if(debug.ge.3)write (moniou,203)ww,sjqq+sjqg+sjgg

       if(iqq.eq.0)then
        stg=0.d0
        do i2=1,7
        do m2=1,2
         xx=.5d0*(1.d0+x1(i2)*(2*m2-3))
         xph=zh**xx
         xmh=zh/xph
         glu1=qgppdi(xph,0)
         sea1=qgppdi(xph,1)
         glu2=qgppdi(xmh,0)
         sea2=qgppdi(xmh,1)
         rh=rq(icdp,icz)-alfp*dlog(zh*xp)

         stsum=(glu1*glu2*sjgg+(glu1*sea2+glu2*sea1)*sjqg
     *   +sea1*sea2*sjqq)*exp(-bb/(4.d0*.0389d0*rh))/rh
         stg=stg+a1(i2)*stsum
        enddo
        enddo
        qglsh=qglsh-a1(i1)*dlog(zh)/zh**delh*stg

       elseif(iqq.eq.1)then
        xmh=zh
        glu2=qgppdi(xmh,0)
        sea2=qgppdi(xmh,1)
        rh=rq(icdp,icz)-alfp*dlog(zh)

        stq=(glu2*sjqg+sea2*sjqq)*exp(-bb/(4.d0*.0389d0*rh))/rh
        qglsh=qglsh+a1(i1)/zh**delh*stq
     *  *(qggrv(xp,qt0,icz,1)+qggrv(xp,qt0,icz,2))/dsqrt(xp)
       endif
      enddo
      enddo
      if(iqq.eq.0)then
       qglsh=qglsh*rr**2*(1.d0-xmin)/(delh-dels)*fp(icz)*g3p*factk
     * /2.d0*pi*cd(icdp,icz)
      elseif(iqq.eq.1)then
       qglsh=qglsh*rr*(1.d0-xmin)/(delh-dels)*g3p*factk/4.d0
     * *cd(icdp,icz)
      endif

      if(debug.ge.3)write (moniou,202)qglsh
201   format(2x,'qglsh - unintegrated Pomeron leg eikonal:'
     */4x,'s=',e10.3,2x,'b^2=',e10.3,2x,'icz=',i1)
202   format(2x,'qglsh=',e10.3)
203   format(2x,'qglsh:',2x,'s_hard=',e10.3,2x,'sigma_hard=',e10.3)
      return
      end

c------------------------------------------------------------------------
      subroutine qgloop(sy,bb,fann,jj)
c-----------------------------------------------------------------------
c qgloop - intermediate Pomeron eikonal with loops
c sy   - pomeron mass squared,
c bb   - impact parameter squared,
c jj=1 - uncut loops (iqq=1,...3)
c jj=2 - cut loops (iqq=4,...11)
c iqq=1  - general loop contribution
c iqq=2  - single Pomeron end on one side
c iqq=3  - single Pomeron ends on both sides
c iqq=4  - single cut Pomeron
c iqq=5  - single cut Pomeron with single end
c iqq=6  - single cut Pomeron with 2 single ends
c iqq=7  - any cuts except the complete rap-gap
c iqq=8  - single cut Pomeron at one side
c iqq=9  - single cut Pomeron at one side and single Pomeron on the other
c iqq=10 - no rap-gap at one side
c iqq=11 - no rap-gap at one side and single Pomeron on the other
c iqq=12 - single cut soft Pomeron
c iqq=13 - single cut soft Pomeron with single end
c iqq=14 - single cut soft Pomeron with 2 single ends
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension fann(14)
      common /qgarr6/  pi,bm,amws
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      do iqq=1,14
       fann(iqq)=0.d0
      enddo
      if(sy.le.sgap**2)goto 1
      do ix1=1,7
      do mx1=1,2
       xpomr=(sy/sgap**2)**(-.5d0-x1(ix1)*(mx1-1.5d0))/sgap
       rp=-alfp*log(xpomr)*4.d0*.0389d0
       rp1=alfp*log(xpomr*sy)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
      do ix2=1,7
      do mx2=1,2
       z=.5d0+x1(ix2)*(mx2-1.5d0)
       bb0=-rp2*log(z)
      do ix3=1,7
      do mx3=1,2
       phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
       bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2
       bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2

       vi=qgpini(xpomr*sy,bb1,0.d0,0.d0,1)
       vit=min(vi,qgpini(xpomr*sy,bb1,0.d0,0.d0,2))
       v1i0=qgpini(1.d0/xpomr,bb2,0.d0,0.d0,4)
       v1i1=min(v1i0,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,3))
       v1i=min(v1i1,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,2))
       if(jj.eq.1)then
        do iqq=1,3
         if(iqq.eq.1)then
          dpx=vi*(min(0.d0,1.d0-exp(-v1i)-v1i)+v1i-v1i1)
     *    +min(0.d0,1.d0-exp(-vit)-vit)*(1.d0-exp(-v1i))
         elseif(iqq.eq.2)then
          dpx=vi*(min(0.d0,1.d0-exp(-v1i)-v1i)+v1i-v1i1)
         elseif(iqq.eq.3)then
          dpx=vi*(v1i1-v1i0)
         else
          dpx=0.d0
         endif
         fann(iqq)=fann(iqq)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
        enddo

       else
        v1ic0=min(v1i0,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,7))
        v1ic1=min(v1ic0,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,6))
        v1ic=min(v1ic1,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,5))
        v1icn=min(v1i,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,8))
        vict=min(vit,qgpini(xpomr*sy,bb1,0.d0,0.d0,5))
        victn=min(vit,qgpini(xpomr*sy,bb1,0.d0,0.d0,8))
        victg=min(victn,qgpini(xpomr*sy,bb1,0.d0,0.d0,11))
        vict1=min(victg,qgpini(xpomr*sy,bb1,0.d0,0.d0,9))

        vis=min(vi,qgpini(xpomr*sy,bb1,0.d0,0.d0,0))
        v1ic0s=min(v1ic0,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,15))
        v1ic1s=min(v1ic0s,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,14))
        v1ics=min(v1ic1s,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,13))
        victs=min(vict,qgpini(xpomr*sy,bb1,0.d0,0.d0,13))
        do iqq=4,14
         if(iqq.eq.4)then
          dpx=vi*(v1ic*exp(-2.d0*v1icn)-v1ic1)
     *    +vict*(exp(-2.d0*victn)-1.d0)*v1ic*exp(-2.d0*v1icn)
         elseif(iqq.eq.5)then
          dpx=vi*(v1ic*exp(-2.d0*v1icn)-v1ic1)
         elseif(iqq.eq.6)then
          dpx=vi*(v1ic1-v1ic0)
         elseif(iqq.eq.7)then
          dpx=vi*(min(0.d0,1.d0-exp(-v1i)-v1i)+v1i-v1i1)
     *    +.5d0*min(0.d0,1.d0-exp(-vit)-vit)*(1.d0-exp(-2.d0*v1icn))
     *    +.5d0*min(0.d0,1.d0-exp(-2.d0*victn)-2.d0*victn)
     *    *max(0.d0,1.d0-exp(-v1i)-.5d0*(1.d0-exp(-2.d0*v1icn)))
         elseif(iqq.eq.8)then
          dpx=vi*(min(0.d0,1.d0-exp(-v1i)-v1i)+v1i-v1i1)
     *    +vict1*(exp(-2.d0*victn)-1.d0)*(1.d0-exp(-v1i))
         elseif(iqq.eq.9)then
          dpx=vi*(v1i1-v1i0)
     *    +vict1*(exp(-2.d0*victn)-1.d0)*v1i1
         elseif(iqq.eq.10)then
          dpx=vi*(min(0.d0,1.d0-exp(-v1i)-v1i)+v1i-v1i1)
     *    +(.5d0*max(0.d0,1.d0-exp(-2.d0*victn)-2.d0*victn
     *    *exp(-2.d0*victn))+victg*(exp(-2.d0*victn)-1.d0))
     *    *(1.d0-exp(-v1i))
         elseif(iqq.eq.11)then
          dpx=vi*(v1i1-v1i0)
     *    +(.5d0*max(0.d0,1.d0-exp(-2.d0*victn)-2.d0*victn
     *    *exp(-2.d0*victn))+victg*(exp(-2.d0*victn)-1.d0))*v1i1
         elseif(iqq.eq.12)then
          dpx=vis*(v1ics*exp(-2.d0*v1icn)-v1ic1s)
     *    +victs*(exp(-2.d0*victn)-1.d0)*v1ics*exp(-2.d0*v1icn)
         elseif(iqq.eq.13)then
          dpx=vis*(v1ics*exp(-2.d0*v1icn)-v1ic1s)
         elseif(iqq.eq.14)then
          dpx=vis*(v1ic1s-v1ic0s)
         else
          dpx=0.d0
         endif
         fann(iqq)=fann(iqq)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
        enddo
       endif
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
1     dpin=qgpini(sy,bb,0.d0,0.d0,1)
      do iqq=1,11
       fann(iqq)=fann(iqq)*log(sy/sgap**2)/8.d0*pi*r3p/.0389d0/g3p**3
     * +dpin
      enddo
      dpins=min(dpin,qgpini(sy,bb,0.d0,0.d0,0))
      do iqq=12,14
       fann(iqq)=fann(iqq)*log(sy/sgap**2)/8.d0*pi*r3p/.0389d0/g3p**3
     * +dpins
      enddo
      return
      end

c------------------------------------------------------------------------
      subroutine qgloos(sy,bb,vvx,vvxt,fann)
c-----------------------------------------------------------------------
c qgloos - intermediate Pomeron eikonal with screening corrections
c sy   - pomeron mass squared,
c bb   - impact parameter squared,
c vvx  - total/projectile screening factor:
c vvx  = 1 - exp[-2*sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]  (iqq=1)
c vvx  = 1 + exp[-2*sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]
c          - exp[-2*sum_{i} chi_proj(i)-sum_j chi_targ(j)]
c          - exp[-sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]    (iqq=2 uncut)
c vvx  = 1 - exp[-sum_{i} chi_proj(i)-2*sum_j chi_targ(j)]    (iqq=2,...4)
c vvx  = 1 - exp[-sum_{i} chi_proj(i)]                        (iqq=5,...8)
c vvxt - target screening factor:
c vvxt = 0                                                    (iqq=1,...4)
c vvxt = 1 - exp[-sum_j chi_targ(j)]                          (iqq=5,...8)
c iqq=1  - single cut Pomeron
c iqq=2  - uncut / cut end / loop sequence
c iqq=3  - no rap-gap at the end
c iqq=4  - single cut Pomeron end
c iqq=5  - diffractive cut, Puu
c iqq=6  - diffractive cut, Puu-Puc
c iqq=7  - diffractive cut, Pcc
c iqq=8  - diffractive cut, Pcc+Pcu
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension fann(14)
      common /qgarr6/  pi,bm,amws
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      do iqq=1,8
       fann(iqq)=0.d0
      enddo
      if(sy.le.sgap**2)goto 1

      do ix1=1,7
      do mx1=1,2
       xpomr=(sy/sgap**2)**(-.5d0-x1(ix1)*(mx1-1.5d0))/sgap
       rp=-alfp*log(xpomr)*4.d0*.0389d0
       rp1=alfp*log(xpomr*sy)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
      do ix2=1,7
      do mx2=1,2
       z=.5d0+x1(ix2)*(mx2-1.5d0)
       bb0=-rp2*log(z)
      do ix3=1,7
      do mx3=1,2
       phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
       bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2
       bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2

       vit=qgpini(xpomr*sy,bb1,0.d0,0.d0,2)
       vicn=min(vit,qgpini(xpomr*sy,bb1,0.d0,0.d0,8))
       vicng=min(vicn,qgpini(xpomr*sy,bb1,0.d0,0.d0,11))
       vicpe=min(vicng,qgpini(xpomr*sy,bb1,0.d0,0.d0,9))
       vic1=min(vicpe,qgpini(xpomr*sy,bb1,0.d0,0.d0,5))

       viu=qgpini(1.d0/xpomr,bb2,0.d0,0.d0,2)
       v1icn=min(viu,qgpini(1.d0/xpomr,bb2,0.d0,0.d0,8))
       v1i=qgpini(1.d0/xpomr,bb2,vvx,0.d0,16)*exp(-2.d0*v1icn)
       vi=qgpini(1.d0/xpomr,bb2,vvx,0.d0,17)*(1.d0-exp(-viu))
       vduu=qgpini(1.d0/xpomr,bb2,vvx,vvxt,20)*(1.d0-exp(-viu))
       vduc=max(0.d0,vduu-qgpini(1.d0/xpomr,bb2,vvx,vvxt,21)
     * *(1.d0-exp(-viu)))
       vdcc=qgpini(1.d0/xpomr,bb2,vvx,vvxt,22)*((1.d0-exp(-viu))**2
     * +(exp(2.d0*(viu-v1icn))-1.d0)*exp(-2.d0*viu))/2.d0
       vdcu=max(0.d0,qgpini(1.d0/xpomr,bb2,vvx,vvxt,23)
     * *((1.d0-exp(-viu))**2+(exp(2.d0*(viu-v1icn))-1.d0)
     * *exp(-2.d0*viu))/2.d0-vdcc)

       do iqq=1,8
        if(iqq.eq.1)then       !single cut Pomeron
         dpx=-vvx*v1i*vic1*exp(-2.d0*vicn)
        elseif(iqq.eq.2)then   !uncut / cut end / loop sequence
         dpx=-(1.d0-exp(-vit))*vi*vvx
        elseif(iqq.eq.3)then   !no rap-gap at the end
         dpx=-(.5d0*max(0.d0,1.d0-exp(-2.d0*vicn)*(1.d0+2.d0*vicn))
     *   +vicng*exp(-2.d0*vicn))*vi*vvx
        elseif(iqq.eq.4)then   !single cut Pomeron end
         dpx=-vicpe*exp(-2.d0*vicn)*vi*vvx
        elseif(iqq.eq.5)then   !Puu
         dpx=(1.d0-exp(-vit))
     *   *(vduu*((1.d0-vvx)*(1.d0-vvxt)*(1.d0-vvx*vvxt)-1.d0)
     *   -vdcu*(1.d0-vvx)**2*(1.d0-vvxt)*vvxt)
        elseif(iqq.eq.6)then   !Puu-Puc
         dpx=(1.d0-exp(-vit))
     *   *((vduu-vduc)*((1.d0-vvx)*(1.d0-vvxt)*(1.d0-vvx*vvxt)-1.d0)
     *   -(vdcc+vdcu)*(1.d0-vvx)**2*(1.d0-vvxt)*vvxt)
        elseif(iqq.eq.7)then   !Pcc
         dpx=.5d0*((1.d0-exp(-vit))**2
     *   +(exp(2.d0*(vit-vicn))-1.d0)*exp(-2.d0*vit))
     *   *(vdcc*((1.d0-vvx)**2*(1.d0-vvxt)**2-1.d0)
     *   -vduc*(1.d0-vvx)*(1.d0-vvxt)**2*vvx)
        elseif(iqq.eq.8)then   !Pcc+Pcu
         dpx=.5d0*((1.d0-exp(-vit))**2
     *   +(exp(2.d0*(vit-vicn))-1.d0)*exp(-2.d0*vit))
     *   *((vdcc+vdcu)*((1.d0-vvx)**2*(1.d0-vvxt)**2-1.d0)
     *   +(vduu-vduc)*(1.d0-vvx)*(1.d0-vvxt)**2*vvx)
        else
         dpx=0.d0
        endif
        fann(iqq)=fann(iqq)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
       enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
1     vit=qgpini(sy,bb,0.d0,0.d0,2)
      vicn=min(vit,qgpini(sy,bb,0.d0,0.d0,8))
      vicng=min(vicn,qgpini(sy,bb,0.d0,0.d0,11))
      vicpe=min(vicng,qgpini(sy,bb,0.d0,0.d0,9))
      vic1=min(vicpe,qgpini(sy,bb,0.d0,0.d0,5))
      do iqq=1,8
       fann(iqq)=fann(iqq)*log(sy/sgap**2)/8.d0*pi*r3p/.0389d0/g3p**3
       if(iqq.eq.1)then
        fann(iqq)=fann(iqq)*exp(2.d0*vicn)+vic1
       elseif(iqq.eq.3)then
        fann(iqq)=fann(iqq)+vicng*exp(-2.d0*vicn)
     *  +.5d0*max(0.d0,1.d0-exp(-2.d0*vicn)*(1.d0+2.d0*vicn))
       elseif(iqq.eq.4)then
        fann(iqq)=fann(iqq)*exp(2.d0*vicn)+vicpe
       elseif(iqq.lt.7)then
        fann(iqq)=fann(iqq)+(1.d0-exp(-vit))
       else
        fann(iqq)=fann(iqq)+.5d0*((1.d0-exp(-vit))**2
     *  +(exp(2.d0*(vit-vicn))-1.d0)*exp(-2.d0*vit))
       endif
      enddo
      return
      end

c------------------------------------------------------------------------
      subroutine qglool(sy,bb,icdp,icz,fann)
c-----------------------------------------------------------------------
c qglool - integrated Pomeron leg eikonal with loops
c sy   - pomeron mass squared,
c bb   - impact parameter squared,
c icz  - hadron class
c iqq=1 - all
c iqq=2 - single Pomeron end
c iqq=3 - single cut Pomeron
c iqq=4 - single cut Pomeron with single Pomeron end
c iqq=5 - single cut Pomeron end
c iqq=6 - no rap-gap at the end
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension fann(14)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      do iqq=1,6
       fann(iqq)=0.d0
      enddo
      if(sy.le.sgap**2)goto 1

      do ix1=1,7
      do mx1=1,2
       xpomr=(sy/sgap**2)**(-.5d0-x1(ix1)*(mx1-1.5d0))/sgap
       rp=(rq(icdp,icz)-alfp*log(xpomr))*4.d0*.0389d0
       rp1=alfp*log(xpomr*sy)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
      do ix2=1,7
      do mx2=1,2
       z=.5d0+x1(ix2)*(mx2-1.5d0)
       bb0=-rp2*log(z)
      do ix3=1,7
      do mx3=1,2
       phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
       bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2
       bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2

       vpl=qglegi(1.d0/xpomr,bb2,icdp,icz,1)
       v1i0=qgpini(xpomr*sy,bb1,0.d0,0.d0,4)
       v1i1=min(v1i0,qgpini(xpomr*sy,bb1,0.d0,0.d0,3))
       v1i=min(v1i1,qgpini(xpomr*sy,bb1,0.d0,0.d0,2))
       v1ic0=min(v1i0,qgpini(xpomr*sy,bb1,0.d0,0.d0,7))
       v1ic1=min(v1ic0,qgpini(xpomr*sy,bb1,0.d0,0.d0,6))
       v1ic=min(v1ic1,qgpini(xpomr*sy,bb1,0.d0,0.d0,5))
       v1icn=min(v1i,qgpini(xpomr*sy,bb1,0.d0,0.d0,8))
       vicn0=min(v1i1,qgpini(xpomr*sy,bb1,0.d0,0.d0,12))
       vicn=min(vicn0,qgpini(xpomr*sy,bb1,0.d0,0.d0,11))
       vic0=min(vicn0,qgpini(xpomr*sy,bb1,0.d0,0.d0,10))
       vic1=min(vic0,qgpini(xpomr*sy,bb1,0.d0,0.d0,9))
       vicn=min(vicn,v1icn)
       vic1=min(vicn,vic1)
       do iqq=1,6
        if(iqq.eq.1)then
         dpx=vpl*(min(0.d0,1.d0-exp(-v1i)-v1i)+v1i-v1i1)
        elseif(iqq.eq.2)then
         dpx=vpl*(v1i1-v1i0)
        elseif(iqq.eq.3)then
         dpx=vpl*(v1ic*exp(-2.d0*v1icn)-v1ic1)
        elseif(iqq.eq.4)then
         dpx=vpl*(v1ic1-v1ic0)
        elseif(iqq.eq.5)then
         dpx=vpl*(vic1*exp(-2.d0*v1icn)-vic0)
        elseif(iqq.eq.6)then
         dpx=vpl*(.5d0*max(0.d0,1.d0-exp(-2.d0*v1icn)-2.d0*v1icn
     *   *exp(-2.d0*v1icn))+vicn*exp(-2.d0*v1icn)-vicn0)
        else
         dpx=0.d0
        endif
        fann(iqq)=fann(iqq)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
       enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
1     dlool=qglegi(sy,bb,icdp,icz,1)
      do iqq=1,6
       fann(iqq)=(fann(iqq)*log(sy/sgap**2)/8.d0*pi*r3p/.0389d0/g3p**3
     * +dlool)/(1.d0-(1.d0-(1.d0-1.d0/sy)**(1.+ahl(icz)))**(1.+dels))
      enddo
      return
      end

c------------------------------------------------------------------------
      double precision function qgrev(sy,bb,vvxt0,vvxt,vvxpt,vvxp0
     *,vvxpl,icdp,icz)
c-----------------------------------------------------------------------
c qgrev - zigzag contribution
c sy    - c.m. energy squared,
c bb    - impact parameter squared,
c icdp  - diffractive state for the projectile,
c icz   - hadron class,
c vvxt0 = 1 - exp[-sum_j chi^(3)_targ(j)]
c vvxt  = 1 - exp[-sum_j chi_targ(j)]
c vvxpt = 1 - exp[-sum_{i>I} chi^(6)_proj(i)]
c vvxp0 = 1 - exp[-sum_{i>I} chi^(3)_proj(i)]
c vvxpl = 1 - exp[-sum_{i<I} chi_proj(i)]
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qgrev=0.d0
      if(sy.lt..999d0*sgap**2)return

      do ix1=1,7
      do mx1=1,2
       xpomr=(sy/sgap**2)**(-.5d0-x1(ix1)*(mx1-1.5d0))/sgap
       rp=(rq(icdp,icz)-alfp*log(xpomr))*4.d0*.0389d0
       rp1=alfp*log(xpomr*sy)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
      do ix2=1,7
      do mx2=1,2
       z=.5d0+x1(ix2)*(mx2-1.5d0)
       bb0=-rp2*log(z)
      do ix3=1,7
      do mx3=1,2
       phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
       bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2
       bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2

       vvx=1.d0-(1.d0-vvxt)*(1.d0-vvxpl)
       vpf=qgfani(1.d0/xpomr,bb2,vvx,0.d0,0.d0,icdp,icz,1)

       viu=qgpini(xpomr*sy,bb1,0.d0,0.d0,2)
       viloop=(1.d0-exp(-viu))
       vim=2.d0*min(viu,qgpini(xpomr*sy,bb1,0.d0,0.d0,8))

       if(vvxt.eq.0.d0)then
        vvxpin=1.d0-(1.d0-vvxp0)*(1.d0-vvxpl)*exp(-vpf)
        vvxtin=0.d0
        vi=max(0.d0,qgpini(xpomr*sy,bb1,vvxpin,vvxtin,21)*viloop
     *  -qgpini(xpomr*sy,bb1,vvxpin,vvxtin,23)
     *  *(viloop**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))/2.d0)

        dpx=vi*(1.d0-exp(-vpf))
       else
        vpf0=min(vpf,qgfani(1.d0/xpomr,bb2,vvx,vvxp0,vvxpl,icdp,icz,3))
        vpft=max(vpf,qgfani(1.d0/xpomr,bb2,vvx,vvxpt,vvxpl,icdp,icz,6))
        vvxpin=1.d0-(1.d0-vvxp0)*(1.d0-vvxpl)*exp(-vpf0)
        vvxtin=vvxt0
        vi=max(0.d0,qgpini(xpomr*sy,bb1,vvxpin,vvxtin,21)*viloop
     *  -qgpini(xpomr*sy,bb1,vvxpin,vvxtin,23)
     *  *(viloop**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))/2.d0)
        if(vvxpt.eq.1.d0)then
         dpx=vi*(1.d0-exp(-vpft))
        else
         dpx=vi*(1.d0-exp(-vpft)+((1.d0-vvxt)**2*(max(0.d0
     *   ,1.d0-exp(-vpft)*(1.d0+vpft))-max(0.d0,1.d0-exp(-vpf0)
     *   *(1.d0+vpf0))*(1.d0-vvxp0)/(1.d0-vvxpt))
     *   +vpft*((1.d0-vvxt)**2*exp(-vpft)-exp(-vpf0)*(1.d0-vvxpl)
     *   *(1.d0-vvxp0)/(1.d0-vvxpt)*(1.d0-vvxt0)**2)
     *   -vpf0*exp(-vpf0)*(1.d0-vvxp0)/(1.d0-vvxpt)*((1.d0-vvxt)**2
     *   -(1.d0-vvxpl)*(1.d0-vvxt0)**2))/(1.d0-(1.d0-vvxt)**2))
         if(dpx.le.0.d0)dpx=vi*(1.d0-exp(-vpft))
        endif
       endif

       qgrev=qgrev+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      qgrev=qgrev/8.d0*pi*r3p/.0389d0/g3p**3
      if(.not.(qgrev.gt.0.d0.and.qgrev.lt.1.d10))stop'qgrev=NAN'
      return
      end

c------------------------------------------------------------------------
      double precision function qgrevi(sy,bb,vvxt0,vvxt,vvxpt,vvxp0
     *,vvxpl,icdp,icz)
c-----------------------------------------------------------------------
c qgrevi - zigzag contribution (interpolation)
c sy    - c.m. energy squared,
c bb    - impact parameter squared,
c icdp  - diffractive state for the projectile,
c icz   - hadron class,
c vvxt0 = 1 - exp[-sum_j chi^(3)_targ(j)]
c vvxt  = 1 - exp[-sum_j chi_targ(j)
c vvxpt = 1 - exp[-sum_{i>I} chi^(6)_proj(i)]
c vvxp0 = 1 - exp[-sum_{i>I} chi^(3)_proj(i)]
c vvxpl = 1 - exp[-sum_{i<I} chi_proj(i)]
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wz(3),wj(3),wi(3),wm2(3),wm3(3),wm4(3)
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr27/ qlegi(51,11,2,3,7),qfanu(51,11,11,6,2)
     *,qfanc(51,11,11,39,18),qdfan(21,11,11,2,3),qrev(11,11,66,219,2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qgrevi=0.d0
      revm=0.d0
      if(sy.le.sgap**2)return

      rp=(rq(icdp,icz)+alfp*dlog(sy))*4.d0*.0389d0
      z=dexp(-bb/rp)
      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-dlog(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      yl=dlog(sy/sgap**2)/dlog(spmax/sgap**2)*10.d0+1.d0
      k=max(1,int(1.00001d0*yl-1.d0))
      k=min(k,9)
      wk(2)=yl-k
      if(yl.le.2.d0)then
       iymax=2
       wk(1)=1.d0-wk(2)
      else
       wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
       wk(1)=1.d0-wk(2)+wk(3)
       wk(2)=wk(2)-2.d0*wk(3)
       iymax=3
      endif

      if(vvxt0.gt..99d0)then
       j=11
       wj(1)=1.d0
       ivmax=1
       i=1
       wi(1)=1.d0
       iv1max=1
      else
       vl=max(1.d0,vvxt0*10.d0+1.d0)
       j=min(int(vl),9)
       wj(2)=vl-dble(j)
       wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
       wj(1)=1.d0-wj(2)+wj(3)
       wj(2)=wj(2)-2.d0*wj(3)
       ivmax=3

       vl1=max(1.d0,(vvxt-vvxt0)/(1.d0-vvxt0)*5.d0+1.d0)
       i=min(int(vl1),4)
       wi(2)=vl1-dble(i)
       wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
       wi(1)=1.d0-wi(2)+wi(3)
       wi(2)=wi(2)-2.d0*wi(3)
       iv1max=3
      endif

      if(icz.ne.2.or.vvxpt+vvxp0+vvxpl.eq.0.d0)then !hadron (no proj. nucl. corr.)
       ll=icz+(icz-1)*(3-icz)*2
       do i1=1,iv1max
        i2=i+i1-2
       do j1=1,ivmax
        j2=j+j1-1
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,iymax
        k2=k+k1-1
        qgrevi=qgrevi+qrev(k2,l2,j2+11*i2,ll,icdp)
     *  *wk(k1)*wz(l1)*wj(j1)*wi(i1)
       enddo
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do i1=1,iv1max
         i2=i+i1-2
        do j1=1,ivmax
         j2=j+j1-1
        do k1=1,iymax
         k2=k+k1-1
         revm=revm+qrev(k2,1,j2+11*i2,ll,icdp)*wk(k1)*wj(j1)*wi(i1)
        enddo
        enddo
        enddo
        qgrevi=min(qgrevi,revm)
       endif

      else
       vm2=max(1.d0,vvxpt*5.d0+1.d0)
       m2=min(int(vm2),5)
       wm2(2)=vm2-dble(m2)
       wm2(1)=1.d0-wm2(2)
       im2max=2

       if(vvxpt.lt.1.d-2)then
        m3=1
        wm3(1)=1.d0
        im3max=1
       else
        vm3=max(1.d0,vvxp0/vvxpt*5.d0+1.d0)
        m3=min(int(vm3),5)
        wm3(2)=vm3-dble(m3)
        wm3(1)=1.d0-wm3(2)
        im3max=2
       endif

       vm4=max(1.d0,vvxpl*5.d0+1.d0)
       m4=min(int(vm4),5)
       wm4(2)=vm4-dble(m4)
       wm4(1)=1.d0-wm4(2)
       im4max=2

       do mn4=1,im4max
       do mn3=1,im3max
       do mn2=1,im2max
        mn=icz+m2+mn2+6*(m3+mn3-2)+36*(m4+mn4-2)
       do i1=1,iv1max
        i2=i+i1-2
       do j1=1,ivmax
        j2=j+j1-1
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,iymax
        k2=k+k1-1
        qgrevi=qgrevi+qrev(k2,l2,j2+11*i2,mn,icdp)
     *  *wk(k1)*wz(l1)*wj(j1)*wi(i1)*wm2(mn2)*wm3(mn3)*wm4(mn4)
       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do mn4=1,im4max
        do mn3=1,im3max
        do mn2=1,im2max
         mn=icz+m2+mn2+6*(m3+mn3-2)+36*(m4+mn4-2)
        do i1=1,iv1max
         i2=i+i1-2
        do j1=1,ivmax
         j2=j+j1-1
        do k1=1,iymax
         k2=k+k1-1
         revm=revm+qrev(k2,1,j2+11*i2,mn,icdp)
     *   *wk(k1)*wj(j1)*wi(i1)*wm2(mn2)*wm3(mn3)*wm4(mn4)
        enddo
        enddo
        enddo
        enddo
        enddo
        enddo
        qgrevi=min(qgrevi,revm)
       endif
      endif
      qgrevi=dexp(qgrevi)*z*dlog(sy/sgap**2)
     **(1.d0-(1.d0-vvxt)**2)*(1.d0-vvxpt)
      return
      end

c------------------------------------------------------------------------
      subroutine qgfan(sy,bb,vvx,icdp,icz,fann)
c-----------------------------------------------------------------------
c qgfan - integrated fan-contributions
c sy    - c.m. energy squared,
c bb    - impact parameter squared,
c icdp  - diffractive state for the projectile,
c icz   - hadron class
c vvx  = 1 - exp[-sum_j chi_targ(j) - sum_{i<I} chi_proj(i)]
c iqq=1  - general fan with loops
c iqq=2  - general fan with single pomeron end
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension fann(14)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      do iqq=1,2
       fann(iqq)=0.d0
      enddo
      if(sy.le.sgap**2)goto 1

      do ix1=1,7
      do mx1=1,2
       xpomr1=(sy/sgap**2)**(-.5d0-x1(ix1)*(mx1-1.5d0))/sgap
       rp=(rq(icdp,icz)-alfp*log(xpomr1))*4.d0*.0389d0
       rp1=alfp*log(xpomr1*sy)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
       do ix2=1,7
       do mx2=1,2
        z=.5d0+x1(ix2)*(mx2-1.5d0)
        bb0=-rp2*log(z)
       do ix3=1,7
       do mx3=1,2
        phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
        bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2
        bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2

        vpf1=qgfani(1.d0/xpomr1,bb2,vvx,0.d0,0.d0,icdp,icz,2)
        vpf=min(vpf1,qgfani(1.d0/xpomr1,bb2,vvx,0.d0,0.d0,icdp,icz,1))
        v1i1=qgpini(xpomr1*sy,bb1,0.d0,0.d0,3)
        v1i=min(v1i1,qgpini(xpomr1*sy,bb1,0.d0,0.d0,2))
        do iqq=1,2
         if(iqq.eq.1)then
          dpx=(1.d0-exp(-v1i))*(min(0.d0,1.d0-exp(-vpf)-vpf)
     *    *(1.d0-vvx)-vpf*vvx)
         else
          dpx=v1i1*(min(0.d0,1.d0-exp(-vpf)-vpf)*(1.d0-vvx)-vpf*vvx)
         endif
         fann(iqq)=fann(iqq)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
        enddo
       enddo
       enddo
       enddo
       enddo
      enddo
      enddo
1     continue
      do iqq=1,2
       fann(iqq)=(fann(iqq)*dlog(sy/sgap**2)/8.d0*pi*r3p/.0389d0/g3p**3
     * +qglegi(sy,bb,icdp,icz,iqq+1))
     * /(1.d0-(1.d0-(1.d0-1.d0/sy)**(1.+ahl(icz)))**(1.+dels))
      enddo
      return
      end

c------------------------------------------------------------------------
      subroutine qgfanc(sy,bb,vvx,vvxp,vvxpl,icdp,icz,fann)
c-----------------------------------------------------------------------
c qgfan - cut fan-contributions
c sy    - c.m. energy squared,
c bb    - impact parameter squared,
c icdp  - diffractive state for the projectile,
c icz   - hadron class,
c vvx   = 1 - exp[-sum_j chi_targ(j) - sum_{i<I} chi_proj(i)]
c vvxp  = 1 - exp[-sum_{i>I} chi^(3)_proj(i)] (iqq=1,2,3)
c vvxp  = 1 - exp[-sum_{i>I} chi^(6)_proj(i)] (iqq=4)
c vvxp  = 1 - exp[-sum_{i>I} chi_proj(i)]     (iqq=5-9)
c vvxpl = 1 - exp[-sum_{i<I} chi_proj(i)]
c iqq=1 - cut handle fan
c iqq=2 - no rap-gap at the end
c iqq=3 - single cut Pomeron end
c iqq=4 - total fan-like contribution
c iqq=5 - leg-like cut
c iqq=6 - leg-like cut with cut handle
c iqq=7 - single Pomeron cut
c iqq=8 - leg-like cut with single cut Pomeron end
c iqq=9 - leg-like cut without a rap-gap at the end
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension fann(14)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      do iqq=1,9
       fann(iqq)=0.d0
      enddo
      if(sy.le.sgap**2)goto 1

      if(vvx.gt..999d0)then
       vvxs=0.d0
      else
       vvxs=(1.d0-vvx)**2/(1.d0-vvxpl)
      endif

      do ix1=1,7
      do mx1=1,2
       xpomr1=(sy/sgap**2)**(-.5d0-x1(ix1)*(mx1-1.5d0))/sgap
       rp=(rq(icdp,icz)-alfp*log(xpomr1))*4.d0*.0389d0
       rp1=alfp*log(xpomr1*sy)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
       do ix2=1,7
       do mx2=1,2
        z=.5d0+x1(ix2)*(mx2-1.5d0)
        bb0=-rp2*log(z)
       do ix3=1,7
       do mx3=1,2
        phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
        bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2
        bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2

        vi=qgpini(xpomr1*sy,bb1,0.d0,0.d0,2)
        vicn=min(vi,qgpini(xpomr1*sy,bb1,0.d0,0.d0,8))
        vicgap=min(vicn,qgpini(xpomr1*sy,bb1,0.d0,0.d0,11))
        vic1p=min(vicgap,qgpini(xpomr1*sy,bb1,0.d0,0.d0,9))
        vic1=min(vic1p,qgpini(xpomr1*sy,bb1,0.d0,0.d0,5))

        vpf=qgfani(1.d0/xpomr1,bb2,vvx,0.d0,0.d0,icdp,icz,1)
        vpfc0=min(vpf
     *  ,qgfani(1.d0/xpomr1,bb2,vvx,vvxp,vvxpl,icdp,icz,3))
        vpfct=max(vpf
     *  ,qgfani(1.d0/xpomr1,bb2,vvx,vvxp,vvxpl,icdp,icz,6))
        vpf1p=min(vpf
     *  ,qgfani(1.d0/xpomr1,bb2,vvx,vvxp,vvxpl,icdp,icz,7))
        vpf1p0=min(vpf1p
     *  ,qgfani(1.d0/xpomr1,bb2,vvx,vvxp,vvxpl,icdp,icz,8))
        vpfc1=min(vpf1p0
     *  ,qgfani(1.d0/xpomr1,bb2,vvx,vvxp,vvxpl,icdp,icz,9))
        do iqq=1,9
         if(iqq.eq.1)then      !cut handle
          dpx=(1.d0-exp(-vi))
     *    *(vvxs*(min(0.d0,1.d0-exp(-vpfc0)-vpfc0)
     *    +vvxp*(exp(-vpfc0)-exp(-vpf)))+vpfc0*(vvxs-1.d0))
         elseif(iqq.eq.2)then  !no rap-gap at the end
          dpx=(.5d0*max(0.d0,1.d0-exp(-2.d0*vicn)*(1.d0+2.d0*vicn))
     *    +vicgap*exp(-2.d0*vicn))
     *    *(vvxs*(min(0.d0,1.d0-exp(-vpfc0)-vpfc0)
     *    +vvxp*(exp(-vpfc0)-exp(-vpf)))+vpfc0*(vvxs-1.d0))
         elseif(iqq.eq.3)then  !single cut Pomeron end
          dpx=vic1p*exp(-2.d0*vicn)
     *    *(vvxs*(min(0.d0,1.d0-exp(-vpfc0)-vpfc0)
     *    +vvxp*(exp(-vpfc0)-exp(-vpf)))+vpfc0*(vvxs-1.d0))
         elseif(iqq.eq.4)then  !total fan-like contribution
          dpx=(1.d0-exp(-vi))
     *    *((1.d0-vvxpl)*(min(0.d0,1.d0-exp(-vpfct)-vpfct)
     *    +vvxp*(exp(-vpfct)-exp(-vpf)))-vpfct*vvxpl)
         elseif(iqq.eq.5)then  !leg-like cut
          dpx=(1.d0-exp(-vi))*vpf1p
     *    *((1.d0-vvx)*(1.d0-vvxpl)*(1.d0-vvxp)**2*exp(-2.d0*vpf)-1.d0)
         elseif(iqq.eq.6)then  !leg-like cut with cut handle
          dpx=(1.d0-exp(-vi))
     *    *(vpf1p0*((1.d0-vvx)**2*(1.d0-vvxp)**2*exp(-2.d0*vpf)-1.d0)
     *    -(vpf1p-vpf1p0)*vvxs*(1.d0-vvxp)*exp(-vpf)
     *    *(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpf)))
         elseif(iqq.eq.7)then  !single Pomeron cut
          dpx=vic1*exp(-2.d0*vicn)
     *    *vpfc1*((1.d0-vvx)**2*(1.d0-vvxp)**2*exp(-2.d0*vpf)-1.d0)
         elseif(iqq.eq.8)then  !leg-like cut with single cut Pomeron end
          dpx=vic1p*exp(-2.d0*vicn)
     *    *(vpf1p0*((1.d0-vvx)**2*(1.d0-vvxp)**2*exp(-2.d0*vpf)-1.d0)
     *    -(vpf1p-vpf1p0)*vvxs*(1.d0-vvxp)*exp(-vpf)
     *    *(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpf)))
         elseif(iqq.eq.9)then  !leg-like cut without a rap-gap at the end
          dpx=(.5d0*max(0.d0,1.d0-exp(-2.d0*vicn)*(1.d0+2.d0*vicn))
     *    +vicgap*exp(-2.d0*vicn))
     *    *(vpf1p0*((1.d0-vvx)**2*(1.d0-vvxp)**2*exp(-2.d0*vpf)-1.d0)
     *    -(vpf1p-vpf1p0)*vvxs*(1.d0-vvxp)*exp(-vpf)
     *    *(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpf)))
         else
          dpx=0.d0
         endif
         fann(iqq)=fann(iqq)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
        enddo
       enddo
       enddo
       enddo
       enddo
      enddo
      enddo
1     continue
      dfan=qglegi(sy,bb,icdp,icz,2)
      dfangap=min(dfan,qglegi(sy,bb,icdp,icz,7))
      dfan1p=min(dfangap,qglegi(sy,bb,icdp,icz,6))
      dfanc1=min(dfan1p,qglegi(sy,bb,icdp,icz,4))
      do iqq=1,9
       fann(iqq)=fann(iqq)*dlog(sy/sgap**2)/8.d0*pi*r3p/.0389d0/g3p**3
       if(iqq.eq.2.or.iqq.eq.9)then
        fann(iqq)=fann(iqq)+dfangap
       elseif(iqq.eq.3.or.iqq.eq.8)then
        fann(iqq)=fann(iqq)+dfan1p
       elseif(iqq.eq.7)then
        fann(iqq)=fann(iqq)+dfanc1
       else
        fann(iqq)=fann(iqq)+dfan
       endif
       fann(iqq)=fann(iqq)
     * /(1.d0-(1.d0-(1.d0-1.d0/sy)**(1.+ahl(icz)))**(1.+dels))
      enddo
      return
      end

c------------------------------------------------------------------------
      double precision function qgfani(sy,bb,vvx,vvxp,vvxpl
     *,icdp,icz,iqq)
c-----------------------------------------------------------------------
c qgfani - integrated fan-contributions
c sy   - c.m. energy squared,
c bb   - impact parameter squared,
c icdp - diffractive state for the projectile,
c icz  - hadron class,
c vvx   = 1 - exp[-sum_j chi_targ(j) - sum_{i<I} chi_proj(i)]
c vvxp=vvxpl=0                                (iqq=1,2)
c vvxp  = 1 - exp[-sum_{i>I} chi^(3)_proj(i)] (iqq=3,4,5)
c vvxp  = 1 - exp[-sum_{i>I} chi^(6)_proj(i)] (iqq=6)
c vvxp  = 1 - exp[-sum_{i>I} chi_proj(i)]     (iqq=7-11)
c vvxpl = 1 - exp[-sum_{i<I} chi_proj(i)]
c uncut fans:
c iqq=1  - general fan with loops
c iqq=2  - general fan with single pomeron end
c cut fans:
c iqq=3  - cut handle fan
c iqq=4  - no rap-gap at the end
c iqq=5  - single cut Pomeron end
c iqq=6  - total fan-like contribution
c iqq=7  - leg-like cut
c iqq=8  - leg-like cut with cut handle
c iqq=9  - single Pomeron cut
c iqq=10 - leg-like cut with single cut Pomeron end
c iqq=11 - leg-like cut without a rap-gap at the end
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wz(3),wj(3),wi(3),wn(3)
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr27/ qlegi(51,11,2,3,7),qfanu(51,11,11,6,2)
     *,qfanc(51,11,11,39,18),qdfan(21,11,11,2,3),qrev(11,11,66,219,2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qgfani=0.d0
      fanm=0.d0

      if(sy.le.sgap**2)then
       qgfani=qglegi(sy,bb,icdp,icz,1)
       return
      endif

      rp=(rq(icdp,icz)+alfp*dlog(sy))*4.d0*.0389d0
      z=dexp(-bb/rp)
      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-dlog(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      yl=dlog(sy/sgap)/dlog(spmax/sgap)*50.d0+1.d0
      k=max(1,int(1.00001d0*yl-1.d0))
      k=min(k,49)
      wk(2)=yl-k
      if(yl.le.2.d0)then
       iymax=2
       wk(1)=1.d0-wk(2)
      else
       wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
       wk(1)=1.d0-wk(2)+wk(3)
       wk(2)=wk(2)-2.d0*wk(3)
       iymax=3
      endif

      vl=max(1.d0,vvx*10.d0+1.d0)
      if(vvx.eq.0.d0)then
       ivmax=1
       j=1
       wj(1)=1.d0
      else
       j=min(int(vl),9)
       wj(2)=vl-dble(j)
       wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
       wj(1)=1.d0-wj(2)+wj(3)
       wj(2)=wj(2)-2.d0*wj(3)
       ivmax=3
      endif

      if(iqq.le.2)then
       ii=icdp+2*(icz-1)
       do j1=1,ivmax
        j2=j+j1-1
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,iymax
        k2=k+k1-1
        qgfani=qgfani+qfanu(k2,l2,j2,ii,iqq)
     *  *wk(k1)*wz(l1)*wj(j1)
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do j1=1,ivmax
         j2=j+j1-1
        do k1=1,iymax
         k2=k+k1-1
         fanm=fanm+qfanu(k2,1,j2,ii,iqq)*wk(k1)*wj(j1)
        enddo
        enddo
        qgfani=min(qgfani,fanm)
       endif

      elseif(icz.ne.2.or.vvxp+vvxpl.eq.0.d0)then  !hadron (no proj. nucl. corr.)
       ii=icdp+2*(iqq-3)
       ll=icz+(icz-1)*(3-icz)*2
       do j1=1,ivmax
        j2=j+j1-1
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,iymax
        k2=k+k1-1
        qgfani=qgfani+qfanc(k2,l2,j2,ll,ii)*wk(k1)*wz(l1)*wj(j1)
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do j1=1,ivmax
         j2=j+j1-1
        do k1=1,iymax
         k2=k+k1-1
         fanm=fanm+qfanc(k2,1,j2,ll,ii)*wk(k1)*wj(j1)
        enddo
        enddo
        qgfani=min(qgfani,fanm)
       endif

      else
       iv1max=2
       vl1=max(1.d0,vvxp*5.d0+1.d0)
       i=min(int(vl1),5)
       wi(2)=vl1-i
       wi(1)=1.d0-wi(2)

       if(vvx.lt..01d0)then                 !weak (no) screening
        iv2max=1
         n=1
        wn(1)=1.d0
       else                                    !nuclear effects
        iv2max=2
        vl2=max(1.d0,vvxpl/vvx*5.d0+1.d0)
        n=min(int(vl2),5)
        wn(2)=vl2-n
        wn(1)=1.d0-wn(2)
       endif

       ii=icdp+2*(iqq-3)
       do n1=1,iv2max
        n2=n+n1-2
       do i1=1,iv1max
        i2=i+i1+2
       do j1=1,ivmax
        j2=j+j1-1
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,iymax
        k2=k+k1-1
        qgfani=qgfani+qfanc(k2,l2,j2,i2+6*n2,ii)
     *  *wk(k1)*wz(l1)*wj(j1)*wi(i1)*wn(n1)
       enddo
       enddo
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do n1=1,iv2max
         n2=n+n1-2
        do i1=1,iv1max
         i2=i+i1+2
        do j1=1,ivmax
         j2=j+j1-1
        do k1=1,iymax
         k2=k+k1-1
         fanm=fanm+qfanc(k2,1,j2,i2+6*n2,ii)
     *   *wk(k1)*wj(j1)*wi(i1)*wn(n1)
        enddo
        enddo
        enddo
        enddo
        qgfani=min(qgfani,fanm)
       endif
      endif
      qgfani=dexp(qgfani)*z
     **(1.d0-(1.d0-(1.d0-1.d0/sy)**(1.+ahl(icz)))**(1.+dels))
      return
      end

c------------------------------------------------------------------------
      subroutine qgdfan(xpomr,xpomr1,bb,icdp,fann,nn)
c-----------------------------------------------------------------------
c qgdfan - diffractive fans
c xpomr - pomeron lc momentum,
c xpomr1 - rapgap,
c bb    - impact parameter squared,
c icdp - diffractive state for the projectile,
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension fann(14),dps(3)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      icz=2
      do iqq=1,3
       fann(iqq)=0.d0
      enddo

      rp=(rq(icdp,icz)-alfp*log(xpomr1))*2.d0*.0389d0
      rp1=alfp*log(xpomr1/xpomr)*4.d0*.0389d0
      rp2=rp*rp1/(rp+rp1)
      do ix2=1,7
      do mx2=1,2
       z=.5d0+x1(ix2)*(mx2-1.5d0)
       bb0=-rp2*log(z)
      do ix3=1,7
      do mx3=1,2
       phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
       bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2
       bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2

       vpf=qgfani(1.d0/xpomr1,bb2,0.d0,0.d0,0.d0,icdp,icz,1)
       v1i1=qgpini(xpomr1/xpomr,bb1,0.d0,0.d0,3)
       v1i=min(v1i1,qgpini(xpomr1/xpomr,bb1,0.d0,0.d0,2))

       do iqq=1,2
        if(iqq.eq.1)then
         dpx=(1.d0-exp(-v1i))*(1.d0-exp(-vpf))**2
        else
         dpx=v1i1*(1.d0-exp(-vpf))**2
        endif
        fann(iqq)=fann(iqq)+a1(ix2)*a1(ix3)*dpx/z*rp2
       enddo
      enddo
      enddo
      enddo
      enddo

      do ix1=1,7
      do mx1=1,2
       xpomr2=xpomr1*(xpomr/xpomr1*sgap)**(.5d0+x1(ix1)*(mx1-1.5d0))
       rp=(rq(icdp,icz)-alfp*log(xpomr2))*2.d0*.0389d0
       rp1=alfp*log(xpomr2/xpomr)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
       do ix2=1,7
       do mx2=1,2
        z=.5d0+x1(ix2)*(mx2-1.5d0)
        bb0=-rp2*log(z)
       do ix3=1,7
       do mx3=1,2
        phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
        bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2
        bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2

        vpf=qgfani(1.d0/xpomr2,bb2,0.d0,0.d0,0.d0,icdp,icz,1)
        v1i=qgpini(xpomr2/xpomr,bb1,0.d0,0.d0,2)
        dpx=(1.d0-exp(-v1i))*(1.d0-exp(-vpf))**2/2.d0
        fann(3)=fann(3)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
       enddo
       enddo
       enddo
       enddo
      enddo
      enddo
      do iqq=1,3
       fann(iqq)=fann(iqq)*(r3p*pi/.0389d0)/g3p**3/8.d0
      enddo

      if(nn.gt.1.and.xpomr1/xpomr.gt.sgap**2)then
       do iqq=1,3
        dps(iqq)=0.d0
       enddo
       do ix1=1,7
       do mx1=1,2
        xpomr2=xpomr1/sgap*(xpomr/xpomr1*sgap**2)
     *  **(.5d0+x1(ix1)*(mx1-1.5d0))
        rp=(rq(icdp,icz)-alfp*log(xpomr2))*2.d0*.0389d0
        rp1=alfp*log(xpomr2/xpomr)*4.d0*.0389d0
        rp2=rp*rp1/(rp+rp1)
        do ix2=1,7
        do mx2=1,2
         z=.5d0+x1(ix2)*(mx2-1.5d0)
         bb0=-rp2*log(z)
        do ix3=1,7
        do mx3=1,2
         phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
          bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     *   +bb0*sin(phi)**2
         bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     *   +bb0*sin(phi)**2

         vpf=qgfani(1.d0/xpomr2,bb2,0.d0,0.d0,0.d0,icdp,icz,1)
         v1i1=qgpini(xpomr2/xpomr,bb1,0.d0,0.d0,3)
         v1i=min(v1i1,qgpini(xpomr2/xpomr,bb1,0.d0,0.d0,2))
         vpdf=qgdfani(xpomr2,xpomr1,bb2,icdp,1)
         vpdfi=qgdfani(xpomr2,xpomr1,bb2,icdp,3)
         do iqq=1,3
          if(iqq.eq.1)then
           dpx=(1.d0-exp(-v1i))*vpdf*(exp(2.d0*(vpdfi-vpf))-1.d0)
          elseif(iqq.eq.2)then
           dpx=v1i1*vpdf*(exp(2.d0*(vpdfi-vpf))-1.d0)
          elseif(iqq.eq.3)then
           dpx=(1.d0-exp(-v1i))*((exp(2.d0*vpdfi)-1.d0)*exp(-2.d0*vpf)
     *     -2.d0*vpdfi)/2.d0/dlog(xpomr1/xpomr/sgap)
          endif
          dps(iqq)=dps(iqq)+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
         enddo
        enddo
        enddo
        enddo
        enddo
       enddo
       enddo
       do iqq=1,3
        fann(iqq)=fann(iqq)+dps(iqq)*dlog(xpomr1/xpomr/sgap**2)
     *  *(r3p*pi/.0389d0)/g3p**3/8.d0
       enddo
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgdfani(xpomr,xpomr1,bb,icdp,iqq)
c-----------------------------------------------------------------------
c qgfani - integrated fan-contributions
c xpomr - pomeron lc momentum,
c xpomr1 - rapgap,
c bb    - impact parameter squared,
c icdp - diffractive state for the projectile,
c icz  - hadron class
c iqq=1 - total unintegrated,
c iqq=2 - single end unintegrated,
c iqq=3 - total integrated
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wz(3),wj(3)
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr27/ qlegi(51,11,2,3,7),qfanu(51,11,11,6,2)
     *,qfanc(51,11,11,39,18),qdfan(21,11,11,2,3),qrev(11,11,66,219,2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qgdfani=0.d0
      dfanm=0.d0
      if(xpomr*sgap**2.gt.1.d0)return

      icz=2
      rp=(rq(icdp,icz)-alfp*dlog(xpomr))*2.d0*.0389d0
      z=dexp(-bb/rp)
      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-dlog(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      if(xpomr*sgap**2.gt..9999d0)then
       k=1
       j=1
       wk(1)=1.d0
       wj(1)=1.d0
       iymax=1
       iy1max=1
      else
       yl=-dlog(xpomr*sgap**2)/dlog(1.d5/sgap**2)*20.d0+1.d0
       k=max(1,int(1.00001d0*yl-1.d0))
       k=min(k,19)
       wk(2)=yl-k
       if(yl.le.2.d0)then
        iymax=2
        wk(1)=1.d0-wk(2)
       else
        wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
        wk(1)=1.d0-wk(2)+wk(3)
        wk(2)=wk(2)-2.d0*wk(3)
        iymax=3
       endif

       yl1=11.d0-dlog(xpomr1*sgap)/dlog(xpomr*sgap**2)*10.d0
       j=max(1,int(yl1))
       j=min(j,9)
       wj(2)=yl1-dble(j)
       wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
       wj(1)=1.d0-wj(2)+wj(3)
       wj(2)=wj(2)-2.d0*wj(3)
       iy1max=3
      endif

      do l1=1,izmax
       l2=jz+l1-1
      do j1=1,iy1max
       j2=j+j1-1
      do k1=1,iymax
       k2=k+k1-1
       qgdfani=qgdfani+qdfan(k2,j2,l2,icdp,iqq)
     * *wk(k1)*wz(l1)*wj(j1)
      enddo
      enddo
      enddo
      if(zz.lt.1.d0)then
       do j1=1,iy1max
        j2=j+j1-1
       do k1=1,iymax
        k2=k+k1-1
        dfanm=dfanm+qdfan(k2,j2,1,icdp,iqq)*wk(k1)*wj(j1)
       enddo
       enddo
       qgdfani=min(qgdfani,dfanm)
      endif
      qgdfani=dexp(qgdfani)*z
      if(iqq.eq.3)qgdfani=qgdfani*max(0.d0,dlog(xpomr1/xpomr/sgap))
      return
      end

c=============================================================================
      double precision function qg3pom(sy,b,vvx,vvxp,vvxt
     *,icdp,icdt,icz)
c-----------------------------------------------------------------------
c qg3pom - integrated 3p-contributions to the interaction eikonal
c sy   - pomeron mass squared,
c b    - impact parameter,
c icdp - diffractive state for the projectile,
c icdt - diffractive state for the target,
c icz  - hadron class
c vvx  = 1 - exp[-sum_{j<J} chi_targ(j) - sum_{i<I} chi_proj(i)]
c vvxp = 1 - exp[-sum_{i>I} chi_proj(i)]
c vvxt = 1 - exp[-sum_{j>J} chi_targ(j)]
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgdebug/  debug
      common /qgarr43/ moniou
      common /arr3/   x1(7),a1(7)

      qg3pom=0.d0
      if(sy.le.sgap**2)return

      do ix1=1,7
      do mx1=1,2
       xpomr1=(sy/sgap**2)**(-(.5+x1(ix1)*(mx1-1.5)))/sgap
       rp1=(rq(icdp,icz)-alfp*log(xpomr1))*4.d0*.0389d0
       rp2=(rq(icdt,2)+alfp*log(xpomr1*sy))*4.d0*.0389d0
       rp=rp1*rp2/(rp1+rp2)
      do ib1=1,7
      do mb1=1,2
       z=.5d0+x1(ib1)*(mb1-1.5d0)
       bb0=-rp*dlog(z)
      do ib2=1,7
      do mb2=1,2
       phi=pi*(.5d0+x1(ib2)*(mb2-1.5d0))
       bb1=(b*rp1/(rp1+rp2)+dsqrt(bb0)*cos(phi))**2+bb0*sin(phi)**2
       bb2=(b*rp2/(rp1+rp2)-dsqrt(bb0)*cos(phi))**2+bb0*sin(phi)**2

       v1p0=qglegi(1.d0/xpomr1,bb1,icdp,icz,1)
       v1t0=qglegi(xpomr1*sy,bb2,icdt,2,1)
       v1p1=min(v1p0,qglegi(1.d0/xpomr1,bb1,icdp,icz,3))
       v1t1=min(v1t0,qglegi(xpomr1*sy,bb2,icdt,2,3))
       v1p=min(v1p1,qglegi(1.d0/xpomr1,bb1,icdp,icz,2))
       v1t=min(v1t1,qglegi(xpomr1*sy,bb2,icdt,2,2))

       vpf0=min(v1p,qgfani(1.d0/xpomr1,bb1
     * ,1.d0-(1.d0-vvx)*(1.d0-vvxt),0.d0,0.d0,icdp,icz,1))
       vtf0=min(v1t,qgfani(xpomr1*sy,bb2
     * ,1.d0-(1.d0-vvx)*(1.d0-vvxp),0.d0,0.d0,icdt,2,1))

       n=1
1      n=n+1
       vpf=qgfani(1.d0/xpomr1,bb1
     * ,1.d0-(1.d0-vvx)*(1.d0-vvxt)*exp(-vtf0),0.d0,0.d0,icdp,icz,1)
       vtf=qgfani(xpomr1*sy,bb2
     * ,1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpf0),0.d0,0.d0,icdt,2,1)
       if(abs(1.d0-vpf/vpf0)+abs(1.d0-vtf/vtf0).gt.1.d-2.and.n.lt.100)
     * then
        vpf0=vpf
        vtf0=vtf
        goto 1
       endif

       dpx=(1.d0-vvx)*(min(0.d0,1.d0-exp(-vpf)-vpf)
     * *min(0.d0,1.d0-exp(-vtf)-vtf)
     * +vpf*min(0.d0,1.d0-exp(-vtf)-vtf)
     * +vtf*min(0.d0,1.d0-exp(-vpf)-vpf))-vvx*vpf*vtf
     * -.5d0*(vtf-v1t)*(min(0.d0,1.d0-exp(-vpf)-vpf)
     * *(1.d0-vvx)*(1.d0-vvxt)*exp(-vtf)
     * -vpf*(1.d0-(1.d0-vvx)*(1.d0-vvxt)*exp(-vtf)))
     * -.5d0*(vpf-v1p)*(min(0.d0,1.d0-exp(-vtf)-vtf)
     * *(1.d0-vvx)*(1.d0-vvxp)*exp(-vpf)
     * -vtf*(1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpf)))
     * +.5d0*(v1t-v1t1)*v1p0+.5d0*(v1p-v1p1)*v1t0
       dpx=min(1.d0,dpx)

       qg3pom=qg3pom+a1(ib1)*a1(ib2)*a1(ix1)/z*rp*dpx
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      qg3pom=qg3pom/8.d0*log(sy/sgap**2)*(r3p*pi/.0389d0)/g3p**3
      return
      end

c------------------------------------------------------------------------
      double precision function qgpcut(sy,b,vvx,vvxp,vvxt
     *,icdp,icdt,icz)
c-----------------------------------------------------------------------
c qglool - integrated Pomeron leg eikonal with loops
c sy   - pomeron mass squared,
c bb   - impact parameter squared,
c vvx  = 1 - exp[-sum_{j<J} chi_targ(j) - sum_{i<I} chi_proj(i)]
c vvxp = 1 - exp[-sum_{i>I} chi_proj(i)]
c vvxt = 1 - exp[-sum_{j>J} chi_targ(j)]
c icz  - hadron class
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qgpcut=0.d0
      if(sy.le.sgap**2)return

      do ix1=1,7
      do mx1=1,2
       xpomr1=(sy/sgap**2)**(-(.5+x1(ix1)*(mx1-1.5)))/sgap
       rp1=(rq(icdp,icz)-alfp*log(xpomr1))*4.d0*.0389d0
       rp2=(rq(icdt,2)+alfp*log(xpomr1*sy))*4.d0*.0389d0
       rp=rp1*rp2/(rp1+rp2)
      do ib1=1,7
      do mb1=1,2
       z=.5d0+x1(ib1)*(mb1-1.5d0)
       bb0=-rp*dlog(z)
      do ib2=1,7
      do mb2=1,2
       phi=pi*(.5d0+x1(ib2)*(mb2-1.5d0))
       bb1=(b*rp1/(rp1+rp2)+dsqrt(bb0)*cos(phi))**2+bb0*sin(phi)**2
       bb2=(b*rp2/(rp1+rp2)-dsqrt(bb0)*cos(phi))**2+bb0*sin(phi)**2

       vpl=qglegi(1.d0/xpomr1,bb1,icdp,icz,1)
       vtl=qglegi(xpomr1*sy,bb2,icdt,2,1)
       vpf0=qgfani(1.d0/xpomr1,bb1,1.d0-(1.d0-vvx)*(1.d0-vvxt)
     * ,0.d0,0.d0,icdp,icz,1)
       vtf0=qgfani(xpomr1*sy,bb2,1.d0-(1.d0-vvx)*(1.d0-vvxp)
     * ,0.d0,0.d0,icdt,2,1)

       n=1
1      n=n+1
       vpf=qgfani(1.d0/xpomr1,bb1,1.d0-(1.d0-vvx)*(1.d0-vvxt)
     * *exp(-vtf0),0.d0,0.d0,icdp,icz,1)
       vtf=qgfani(xpomr1*sy,bb2,1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpf0)
     * ,0.d0,0.d0,icdt,2,1)
       if(abs(1.d0-vpf/vpf0)+abs(1.d0-vtf/vtf0).gt.1.d-2.and.n.lt.100)
     * then
        vpf0=vpf
        vtf0=vtf
        goto 1
       endif

       vpls=qgfani(1.d0/xpomr1,bb1,1.d0-(1.d0-vvx)*(1.d0-vvxt)
     * *exp(-vtf),vvxp,0.d0,icdp,icz,9)
       vtls=qgfani(xpomr1*sy,bb2,1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpf)
     * ,vvxt,0.d0,icdt,2,9)
       vploop0=qglegi(1.d0/xpomr1,bb1,icdp,icz,5)
       vploop=min(vploop0,qglegi(1.d0/xpomr1,bb1,icdp,icz,4))
       vtloop0=qglegi(xpomr1*sy,bb2,icdt,2,5)
       vtloop=min(vtloop0,qglegi(xpomr1*sy,bb2,icdt,2,4))

       dpx=(vpls*vtloop+vtls*vploop)*(((1.d0-vvx)*(1.d0-vvxp)
     * *(1.d0-vvxt))**2*exp(-2.d0*vpf-2.d0*vtf)-1.d0)
     * +vpl*(vtloop-vtloop0)+vtl*(vploop-vploop0)

       qgpcut=qgpcut+a1(ib1)*a1(ib2)*a1(ix1)/z*rp*dpx
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      qgpcut=qgpcut/16.d0*log(sy/sgap**2)*(r3p*pi/.0389d0)/g3p**3
      return
      end

c------------------------------------------------------------------------
      double precision function qgpomi(sy,bb,vvx,vvxp,vvxt
     *,icdp,icdt,icz,iqq)
c-----------------------------------------------------------------------
c qgpomi - integrated  eikonal contributions
c sy   - pomeron mass squared,
c bb   - impact parameter squared,
c icdp - diffractive state for the projectile,
c icdt - diffractive state for the target,
c icz  - projectile class
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wz(3),wi(3),wj(3),wm(3)
      common /qgarr10/ am(7),ammu
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr24/ qpomr(11,11,216,12,2)
      common /qgdebug/  debug
      common /qgarr43/ moniou

      qgpomi=0.d0
      pomm=0.d0
      if(cd(icdp,icz).eq.0.d0.or.cd(icdt,2).eq.0.d0)return

      rp=(rq(icdp,icz)+rq(icdt,2)+alfp*log(sy))*4.d0*.0389d0
      z=exp(-bb/rp)
      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-log(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      yl=dlog10((sy-am(2)**2-am(icz)**2)/2.d0/am(2))
      k=max(1,int(yl))
      k=min(k,9)
      wk(2)=yl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)

      ml=icdp+2*(icdt-1)+4*(icz-1)
      if(vvx+vvxp+vvxt.eq.0.d0)then  !hadron-proton (no nucl. screening)
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,3
        k2=k+k1-1
        qgpomi=qgpomi+qpomr(k2,l2,1,ml,iqq)*wk(k1)*wz(l1)
       enddo
       enddo
       if(zz.lt.1.d0)then
        do k1=1,3
         k2=k+k1-1
         pomm=pomm+qpomr(k2,1,1,ml,iqq)*wk(k1)
        enddo
        qgpomi=min(qgpomi,pomm)
       endif
      else
       vl=max(1.d0,vvx*5.d0+1.d0)
       j=min(int(vl),4)
       wj(2)=vl-j
       wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
       wj(1)=1.d0-wj(2)+wj(3)
       wj(2)=wj(2)-2.d0*wj(3)

       if(icz.ne.2.or.vvxp.eq.0.d0)then   !hadron-nucleus (no proj. nucl. scr.)
        i1max=1
        i=1
        wi(1)=1.d0
       else
        i1max=3
        vl1=max(1.d0,vvxp*5.d0+1.d0)
        i=min(int(vl1),4)
        wi(2)=vl1-i
        wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
        wi(1)=1.d0-wi(2)+wi(3)
        wi(2)=wi(2)-2.d0*wi(3)
       endif

       vl2=max(1.d0,vvxt*5.d0+1.d0)
       m=min(int(vl2),4)
       wm(2)=vl2-m
       wm(3)=wm(2)*(wm(2)-1.d0)*.5d0
       wm(1)=1.d0-wm(2)+wm(3)
       wm(2)=wm(2)-2.d0*wm(3)

       do m1=1,3
        m2=m+m1-2
       do i1=1,i1max
        i2=i+i1-2
       do j1=1,3
        j2=j+j1-1
        mij=j2+6*i2+36*m2
       do l1=1,izmax
        l2=jz+l1-1
       do k1=1,3
        k2=k+k1-1
        qgpomi=qgpomi+qpomr(k2,l2,mij,ml,iqq)
     *  *wk(k1)*wz(l1)*wj(j1)*wi(i1)*wm(m1)
       enddo
       enddo
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do m1=1,3
         m2=m+m1-2
        do i1=1,i1max
         i2=i+i1-2
        do j1=1,3
         j2=j+j1-1
         mij=j2+6*i2+36*m2
        do k1=1,3
         k2=k+k1-1
         pomm=pomm+qpomr(k2,1,mij,ml,iqq)*wk(k1)*wj(j1)*wi(i1)*wm(m1)
        enddo
        enddo
        enddo
        enddo
        qgpomi=min(qgpomi,pomm)
       endif
      endif
      qgpomi=exp(qgpomi)*z
      return
      end

c------------------------------------------------------------------------
      double precision function qgppdi(xp,iqq)
c-----------------------------------------------------------------------
c qgppdi - parton distributions in the Pomeron
c xp    - parton LC momentum share,
c iqq=0 - gluon
c iqq=1 - sea quark
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr20/ spmax
      common /qgarr43/ moniou
      common /qgdebug/  debug

c... initialize
      qgppdi=0.d0
      if(debug.ge.3)write (moniou,201)xp,iqq
      if(xp.ge..9999999d0)then
       qgppdi=0.d0
      else
       if(iqq.eq.0)then                             !gluon
        qgppdi=(1.d0-xp)**betp*(1.d0-dgqq)
       elseif(iqq.eq.1)then                         !quark
        qgppdi=qgftlf(xp)*dgqq
       endif
      endif
      if(debug.ge.4)write (moniou,202)qgppdi

201   format(2x,'qgppdi - parton distr. in the Pomeron (interpol.):'
     */4x,'xp=',e10.3,2x,'iqq=',i1)
202   format(2x,'qgppdi=',e10.3)
      return
      end

c=============================================================================
      double precision function qgvpdf(x,icz)
c-----------------------------------------------------------------------------
c qgvpdf - valence quark structure function
c x   - Feinman x,
c icz - hadron class
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr25/ ahv(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qgvpdf=(qggrv(x,qt0,icz,1)+qggrv(x,qt0,icz,2))*(1.d0-x)**ahv(icz)
      return
      end

c=============================================================================
      double precision function qgspdf(x,icz)
c-----------------------------------------------------------------------------
c qgspdf - sea quark structure function
c x   - Feinman x,
c icz - hadron class
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qgspdf=0.d0
      if(x*sgap.ge.1.d0)goto 1

      do icdp=1,2
       rp=(rq(icdp,icz)-alfp*log(x))*2.d0*.0389d0
       if(cd(icdp,icz).ne.0.d0)then
        dps=0.d0
        do ix=1,7
        do mx=1,2
         xpomr=(x*sgap)**(.5d0+x1(ix)*(mx-1.5d0))/sgap
        do ib=1,7
        do mb=1,2
         z=.5d0+x1(ib)*(mb-1.5d0)
         bb=-rp*log(z)

         v1p1=qgfani(1.d0/xpomr,bb,0.d0,0.d0,0.d0,icdp,icz,2)
         v1p=min(v1p1,qgfani(1.d0/xpomr,bb,0.d0,0.d0,0.d0,icdp,icz,1))
         dps=dps+a1(ix)*a1(ib)*(min(0.d0,1.d0-exp(-v1p)-v1p)+v1p-v1p1)
     *   *qgftlf(x/xpomr)*(xpomr/x)**dels/z
        enddo
        enddo
        enddo
        enddo
        qgspdf=qgspdf-dps*dlog(x*sgap)*rp/g3p**2*pi*rr*(r3p*pi/.0389d0)
     *  *dgqq*cc(icdp,icz)
       endif
      enddo

1     qgspdf=qgspdf+4.*pi*rr*fp(icz)*qgftle(x,icz)/x**dels
      return
      end

c=============================================================================
      double precision function qggpdf(x,icz)
c-----------------------------------------------------------------------------
c qggpdf - gluon structure function (xg(x,qt0))
c x   - Feinman x,
c icz - hadron class
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qggpdf=0.d0
      if(x*sgap.ge.1.d0)goto 1

      do icdp=1,2
       rp=(rq(icdp,icz)-alfp*log(x))*2.d0*.0389d0
       if(cd(icdp,icz).ne.0.d0)then
        dps=0.d0
        do ix=1,7
        do mx=1,2
         xpomr=(x*sgap)**(.5d0+x1(ix)*(mx-1.5d0))/sgap
        do ib=1,7
        do mb=1,2
         z=.5d0+x1(ib)*(mb-1.5d0)
         bb=-rp*log(z)

         v1p1=qgfani(1.d0/xpomr,bb,0.d0,0.d0,0.d0,icdp,icz,2)
         v1p=min(v1p1,qgfani(1.d0/xpomr,bb,0.d0,0.d0,0.d0,icdp,icz,1))
         dps=dps+a1(ix)*a1(ib)*(min(0.d0,1.d0-exp(-v1p)-v1p)+v1p-v1p1)
     *   *(1.d0-x/xpomr)**betp*(xpomr/x)**dels/z
        enddo
        enddo
        enddo
        enddo
        qggpdf=qggpdf-dps*dlog(x*sgap)*rp/g3p**2*pi*rr*(r3p*pi/.0389d0)
     *  *(1.d0-dgqq)*cc(icdp,icz)
       endif
      enddo

1     qggpdf=qggpdf+4.*pi*rr*fp(icz)*qgftld(x,icz)/x**dels
      return
      end

c=============================================================================
      double precision function qgpdfb(x,bb,icz,jj)
c-----------------------------------------------------------------------------
c qgpdfb - b-dependent parton momentum distributions (xf(x,b,qt0))
c x   - Feinman x,
c icz - hadron class
c jj=0 - g,
c jj=1 - q
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qgpdfb=0.d0
      if(x*sgap.lt.1.d0)then
       do icdp=1,2
        if(cd(icdp,icz).ne.0.d0)then
         dps=0.d0
         do ix=1,7
         do mx=1,2
          xpomr=(x*sgap)**(.5d0+x1(ix)*(mx-1.5d0))/sgap
          rp=(rq(icdp,icz)-alfp*log(xpomr))*2.d0*.0389d0
          rp1=alfp*dlog(xpomr/x)*4.d0*.0389d0
          rp2=rp1*rp/(rp1+rp)
         do ix2=1,7
         do mx2=1,2
          bb0=-rp2*log(.5d0+x1(ix2)*(mx2-1.5d0))
         do ix3=1,7
         do mx3=1,2
          phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
          bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     *    +bb0*sin(phi)**2
          bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     *    +bb0*sin(phi)**2

          if(jj.eq.0)then
           v1i=(1.d0-x/xpomr)**betp*(1.d0-dgqq)
          else
           v1i=qgftlf(x/xpomr)*dgqq
          endif
          v1p1=qgfani(1.d0/xpomr,bb2,0.d0,0.d0,0.d0,icdp,icz,2)
          v1p=min(v1p1,qgfani(1.d0/xpomr,bb2,0.d0,0.d0,0.d0,icdp,icz,1))

          dps=dps+a1(ix)*a1(ix2)*a1(ix3)*v1i
     *    *(min(0.d0,1.d0-exp(-v1p)-v1p)+v1p-v1p1)
     *    *(xpomr/x)**dels*rp/(rp1+rp)*exp(bb2/rp-bb/(rp1+rp))
         enddo
         enddo
         enddo
         enddo
         enddo
         enddo
         qgpdfb=qgpdfb-dps*dlog(x*sgap)*pi*rr*r3p/g3p**2/.0389d0/2.d0
     *   *cc(icdp,icz)
        endif
       enddo

       do icdp=1,2
        rp=(rq(icdp,icz)-alfp*dlog(x))*4.d0*.0389d0
        if(jj.eq.0)then
         qgpdfb=qgpdfb+4.d0*rr*fp(icz)*qgftld(x,icz)/x**dels
     *   /rp*exp(-bb/rp)*cc(icdp,icz)
        else
         qgpdfb=qgpdfb+4.d0*rr*fp(icz)*qgftle(x,icz)/x**dels
     *   /rp*exp(-bb/rp)*cc(icdp,icz)
        endif
       enddo
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgpdfi(x,bb,icz,jj)
c-----------------------------------------------------------------------
c qgpdfi - b-dependent parton momentum distributions
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wz(3)
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr20/ spmax
      common /qgarr53/ qpdfb(51,11,3,2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qgpdfi=0.d0
      rp=(rq(1,icz)-alfp*dlog(x))*4.d0*.0389d0
      if(rp.le.1.d-10)then
       z=1.d0
      else
       z=exp(-bb/rp)
      endif
      if(z.lt..2d0*exp(-10.d0))then
       izmax=2
       jz=1
       wz(2)=5.d0*z*exp(10.d0)
       wz(1)=1.d0-wz(2)
      else
       if(z.gt..2d0)then
        zz=5.d0*z+6.d0
       else
        zz=(-bb/rp-log(0.2d0))/2.d0+7.d0
       endif
       jz=min(9,int(zz))
       jz=max(2,jz)
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      yl=-dlog(x)/log(spmax)*50.d0+1.d0
      k=max(1,int(yl))
      k=min(k,49)
      wk(2)=yl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)

      do j1=1,izmax
       j2=jz+j1-1
      do k1=1,3
       k2=k+k1-1
       qgpdfi=qgpdfi+qpdfb(k2,j2,icz,jj+1)*wk(k1)*wz(j1)
      enddo
      enddo
      qgpdfi=exp(qgpdfi)*z*4.d0*rr*fp(icz)/x**dels/rp
      if(jj.eq.0)then
       qgpdfi=qgpdfi*qgftld(x,icz)
      else
       qgpdfi=qgpdfi*qgftle(x,icz)
      endif
      return
      end

c=============================================================================
      double precision function qgdgdf(x,xpomr,icz,jj)
c-----------------------------------------------------------------------------
c qgdgdf - diffractive gluon pdf xpomr*g_d^3(x,xpomr,qt0)
c x   - Feinman x,
c icz - hadron class
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qgdgdf=0.d0
      do icdp=1,2
      if(cd(icdp,icz).ne.0.d0)then
       dps=0.d0
       if(jj.eq.1)then
        rp=(rq(icdp,icz)-alfp*log(xpomr))*2.d0*.0389d0
        do ib=1,7
        do mb=1,2
         z=.5d0+x1(ib)*(mb-1.5d0)
         bb=-rp*log(z)

         v1p=qgfani(1.d0/xpomr,bb,0.d0,0.d0,0.d0,icdp,icz,1)
         dps=dps+a1(ib)*(1.d0-exp(-v1p))**2/z
        enddo
        enddo
        dps=dps*rp*pi*rr*(r3p*pi/.0389d0)*cc(icdp,icz)/g3p**2
     *  *(1.d0-x/xpomr)**betp*(1.d0-dgqq)*(xpomr/x)**dels

       elseif(jj.eq.2.and.xpomr/x.gt.sgap)then
        do ix1=1,7
        do mx1=1,2
         xpomr1=(x/xpomr*sgap)**(.5d0+x1(ix1)*(mx1-1.5d0))*xpomr/sgap
         rp=(rq(icdp,icz)-alfp*log(xpomr1))*2.d0*.0389d0
         do ib=1,7
         do mb=1,2
          z=.5d0+x1(ib)*(mb-1.5d0)
          bb=-rp*log(z)

          vpf=qgfani(1.d0/xpomr1,bb,0.d0,0.d0,0.d0,icdp,icz,1)
          vpdf1=qgdfani(xpomr1,xpomr,bb,icdp,2)
          vpdf=min(vpdf1,qgdfani(xpomr1,xpomr,bb,icdp,1))
          vpdfi=qgdfani(xpomr1,xpomr,bb,icdp,3)
           dpx=vpdf*exp(2.d0*vpdfi-2.d0*vpf)-vpdf1

          dps=dps+a1(ix1)*a1(ib)*dpx/z*rp
     *    *(1.d0-x/xpomr1)**betp*(xpomr1/x)**dels
         enddo
         enddo
        enddo
        enddo
        dps=dps*rr*pi*(r3p*pi/.0389d0)*dlog(xpomr/x/sgap)/g3p**2
     *  *(1.d0-dgqq)*cc(icdp,icz)
       endif
       qgdgdf=qgdgdf+dps
      endif
      enddo
      return
      end

c=============================================================================
      double precision function qgdpdf(x,xpomr,icz,jj)
c-----------------------------------------------------------------------------
c qgdpdf - diffractive structure function
c x   - Feinman x,
c icz - hadron class
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qgdpdf=0.d0
      do icdp=1,2
      if(cd(icdp,icz).ne.0.d0)then
       dps=0.d0
       if(jj.eq.1)then
        rp=(rq(icdp,icz)-alfp*log(xpomr))*2.d0*.0389d0
        do ib=1,7
        do mb=1,2
         z=.5d0+x1(ib)*(mb-1.5d0)
         bb=-rp*log(z)

         v1p=qgfani(1.d0/xpomr,bb,0.d0,0.d0,0.d0,icdp,icz,1)
         dps=dps+a1(ib)*(1.d0-exp(-v1p))**2/z
        enddo
        enddo
        dps=dps*rp*pi*rr*(r3p*pi/.0389d0)*cc(icdp,icz)/g3p**2
     *  *qgftlf(x/xpomr)*dgqq*(xpomr/x)**dels

       elseif(jj.eq.2.and.xpomr/x.gt.sgap)then
        do ix1=1,7
        do mx1=1,2
         xpomr1=(x/xpomr*sgap)**(.5d0+x1(ix1)*(mx1-1.5d0))*xpomr/sgap
         rp=(rq(icdp,icz)-alfp*log(xpomr1))*2.d0*.0389d0
         do ib=1,7
         do mb=1,2
          z=.5d0+x1(ib)*(mb-1.5d0)
          bb=-rp*log(z)

          vpf=qgfani(1.d0/xpomr1,bb,0.d0,0.d0,0.d0,icdp,icz,1)
          vpdf1=qgdfani(xpomr1,xpomr,bb,icdp,2)
          vpdf=min(vpdf1,qgdfani(xpomr1,xpomr,bb,icdp,1))
          vpdfi=qgdfani(xpomr1,xpomr,bb,icdp,3)
           dpx=vpdf*exp(2.d0*vpdfi-2.d0*vpf)-vpdf1

          dps=dps+a1(ix1)*a1(ib)*dpx/z*rp
     *    *qgftlf(x/xpomr1)*(xpomr1/x)**dels
         enddo
         enddo
        enddo
        enddo
        dps=dps*rr*pi*(r3p*pi/.0389d0)*dlog(xpomr/x/sgap)/g3p**2
     *  *dgqq*cc(icdp,icz)
       endif
       qgdpdf=qgdpdf+dps
      endif
      enddo
      qgdpdf=qgdpdf/4.5d0
      return
      end

c=============================================================================
      double precision function qgfsh(sy,bb,icdp,icdt,icz,iqq)
c-----------------------------------------------------------------------------
c qgfsh - semihard interaction eikonal
c sy  - pomeron mass squared,
c bb  - impact parameter squared,
c icz - hadron class
c iqq - type of the hard interaction (0-gg, 1-q_vg, 2-gq_v)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr25/ ahv(3)
      common /qgarr26/ factk,fqscal
      common /arr3/   x1(7),a1(7)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)sy,bb,iqq,icz

      qgfsh=0.d0
      s2min=4.d0*fqscal*qt0
      xmin=s2min/sy
      if(xmin.ge.1.d0)return
      xmin=xmin**(delh-dels)
      if(iqq.eq.1)then
       icv=icz
       icq=2
      elseif(iqq.eq.2)then
       icv=2
       icq=icz
      endif
      if(debug.ge.3)write (moniou,205)xmin,iqq

c numerical integration over z1
      do i=1,7
      do m=1,2
       z1=(.5d0*(1.d0+xmin-(2*m-3)*x1(i)*(1.d0-xmin)))
     * **(1.d0/(delh-dels))
       ww=z1*sy
       sjqq=qgjit(qt0,qt0,ww,2,2)
       sjqg=qgjit(qt0,qt0,ww,1,2)
       sjgg=qgjit(qt0,qt0,ww,1,1)
       if(debug.ge.3)write (moniou,203)ww,sjqq+sjqg+sjgg

       if(iqq.eq.0)then
        st2=0.d0
        do j=1,7
        do k=1,2
         xx=.5d0*(1.d0+x1(j)*(2*k-3))
         xp=z1**xx
         xm=z1/xp
         glu1=qgftld(xp,icz)
         sea1=qgftle(xp,icz)
         glu2=qgftld(xm,2)
         sea2=qgftle(xm,2)
         st2=st2+a1(j)*(glu1*glu2*sjgg+(glu1*sea2+glu2*sea1)*sjqg
     *   +sea1*sea2*sjqq)
        enddo
        enddo
        rh=rq(icdp,icz)+rq(icdt,2)-alfp*dlog(z1)
        qgfsh=qgfsh-a1(i)*dlog(z1)/z1**delh*st2
     *  *exp(-bb/(4.d0*.0389d0*rh))/rh

       else
        st2=0.d0
        alh=.5d0+dels
        xam=z1**alh

        do j=1,7
        do k=1,2
         xp=(.5d0*(1.d0+xam+x1(j)*(2*k-3)*(1.d0-xam)))**(1.d0/alh)
         xm=z1/xp
         glu=qgftld(xm,icq)
         sea=qgftle(xm,icq)
         rh=rq(icdp,icz)+rq(icdt,2)-alfp*dlog(xm)

         fst=(glu*sjqg+sea*sjqq)*(1.d0-xp)**ahv(icv)
     *   *(qggrv(xp,qt0,icv,1)+qggrv(xp,qt0,icv,2))/dsqrt(xp)
     *   *exp(-bb/(4.d0*.0389d0*rh))/rh
         st2=st2+a1(j)*fst
        enddo
        enddo
        st2=st2*(1.d0-xam)/alh
        qgfsh=qgfsh+a1(i)/z1**delh*st2
       endif
      enddo
      enddo

      if(iqq.eq.0)then
       qgfsh=qgfsh*rr**2*(1.d0-xmin)/(delh-dels)*fp(icz)*fp(2)*factk
     * /2.d0*pi*cd(icdp,icz)*cd(icdt,2)
      else
       qgfsh=qgfsh*rr*fp(icq)*(1.d0-xmin)/(delh-dels)*factk/8.d0
     * *cd(icdp,icz)*cd(icdt,2)
      endif

      if(debug.ge.3)write (moniou,202)qgfsh
201   format(2x,'qgfsh - semihard interaction eikonal:'
     */4x,'sy=',e10.3,2x,'bb=',e10.3,2x,'iqq=',i1,2x,'icz=',i1)
202   format(2x,'qgfsh=',e10.3)
203   format(2x,'qgfsh:',2x,'s_hard=',e10.3,2x,'sigma_hard=',e10.3)
205   format(2x,'qgfsh:',2x,'xmin=',e10.3,2x,'iqq=',i3)
      return
      end

c=============================================================================
      double precision function qgftld(z,icz)
c-----------------------------------------------------------------------------
c qgftld - auxilliary function for semihard eikonals calculation -
c (proportional to gluon sf: g(z)*z^(1+dels)) -
c integration over semihard block light cone momentum share x
c z - x-cutoff from below,
c icz - type of the hadron to which the semihard block is connected
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/    debug
      common /arr3/     x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)z,icz

      qgftld=0.d0
      xpmin=z**(1.d0+dels)
      do i1=1,7
      do m1=1,2
       tp=1.d0-(1.d0-xpmin)*(.5d0+x1(i1)*(m1-1.5d0))
     * **(1.d0/(1.d0+ahl(icz)))
       xp=tp**(1.d0/(1.d0+dels))
       qgftld=qgftld+a1(i1)*((1.d0-xp)/(1.d0-tp))**ahl(icz)
     * *(1.d0-z/xp)**betp
      enddo
      enddo
      qgftld=qgftld*.5d0*(1.d0-xpmin)**(ahl(icz)+1.d0)
     */(ahl(icz)+1.d0)/(1.d0+dels)*(1.d0-dgqq)

      if(debug.ge.3)write (moniou,202)qgftld
201   format(2x,'qgftld:',2x,'z=',e10.3,2x,'icz=',i1)
202   format(2x,'qgftld=',e10.3)
      return
      end

c------------------------------------------------------------------------
      double precision function qgftle(z,icz)
c-----------------------------------------------------------------------
c qgftle - auxilliary function for semihard eikonals calculation
c (proportional to sea quark sf: q_s(z)*z^(1+dels)) -
c integration over semihard pomeron light cone momentum share x
c z - light cone x of the quark,
c icz - type of the hadron to which the semihard block is connected
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)z,icz

      qgftle=0.d0
      xpmin=z**(1.d0+dels)
      do i1=1,7
      do m1=1,2
       tp=1.d0-(1.d0-xpmin)*(.5d0+x1(i1)*(m1-1.5d0))
     * **(1.d0/(1.d0+ahl(icz)))
       xp=tp**(1.d0/(1.d0+dels))
       qgftle=qgftle+a1(i1)*((1.d0-xp)/(1.d0-tp))**ahl(icz)
     * *qgftlf(z/xp)
      enddo
      enddo
      qgftle=qgftle*.5d0*(1.d0-xpmin)**(ahl(icz)+1.d0)
     */(ahl(icz)+1.d0)/(1.d0+dels)*dgqq

      if(debug.ge.3)write (moniou,202)qgftle
201   format(2x,'qgftle:',2x,'z=',e10.3,2x,'icz=',i1)
202   format(2x,'qgftle=',e10.3)
      return
      end

c------------------------------------------------------------------------
      double precision function qgftlf(zz)
c-----------------------------------------------------------------------
c qgftlf - auxilliary function for semihard eikonals calculation
c zz - ratio of the quark and pomeron light cone x (zz=x_G/x_P)
c integration over quark to gluon light cone momentum ratio (z=x/x_G):
c qgftlf=int(dz) z^dels * (1-zz/z)^betp * P_qG(z)
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)zz
201   format(2x,'qgftlf:',2x,'zz=',e10.3)

      qgftlf=0.d0
      zmin=zz**(1.d0+dels)
      do i=1,7
      do m=1,2
        z=(.5d0*(1.d0+zmin+(2*m-3)*x1(i)*(1.d0-zmin)))**(1.d0/
     *  (1.d0+dels))
        qgftlf=qgftlf+a1(i)*max(1.d-9,(1.d0-zz/z))**betp
     *  *(z**2+(1.d0-z)**2)
      enddo
      enddo
      qgftlf=qgftlf*1.5d0*(1.d0-zmin)/(1.d0+dels)   !1.5=naflav/2 at Q0

      if(debug.ge.3)write (moniou,202)qgftlf
202   format(2x,'qgftlf=',e10.3)
      return
      end

c=============================================================================
      subroutine qgfz(b,gz,iddp1,iddp2)
c----------------------------------------------------------------------------
c hadron-hadron and hadron-nucleus cross sections calculation
c----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      dimension gz(5),wt1(3),wt2(3)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr43/ moniou
      common /arr3/   x1(7),a1(7)
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)b,iddp1,iddp2
      do l=1,5
       gz(l)=0.d0
      enddo
      rp=(rq(1,icz)+rq(1,2)+alfp*log(scm))*4.d0*.0389d0
      g0=0.d0
      if(ia(2).eq.1.and.iddp1.eq.0.and.iddp2.eq.0)then
       g0=pi*rp*10.d0                     !normalization factor (in mb)
       bm=2.d0*dsqrt(rp)                  !impact parameter for exp. fall-down
      endif

      do i1=1,7
      do m=1,2
       z=.5d0+x1(i1)*(m-1.5d0)
       bb1=rp*z
       bb2=rp*(1.d0-dlog(z))

       do l=1,3
        wt1(l)=0.d0
        wt2(l)=0.d0
       enddo

       if(ia(2).eq.1)then
        do idd1=1,2
        do idd2=1,2
         vv1=exp(-qgpomi(scm,bb1,0.d0,0.d0,0.d0,idd1,idd2,icz,1))
         vv2=exp(-qgpomi(scm,bb2,0.d0,0.d0,0.d0,idd1,idd2,icz,1))

         do l=1,2
          wt1(l)=wt1(l)+cc(idd1,icz)*cc(idd2,2)*vv1**l
          wt2(l)=wt2(l)+cc(idd1,icz)*cc(idd2,2)*vv2**l
         enddo
         do idd3=1,2
          wt1(3)=wt1(3)+cc(idd1,icz)*cc(idd2,2)*cc(idd3,icz)*vv1
     *    *exp(-qgpomi(scm,bb1,0.d0,0.d0,0.d0,idd3,idd2,icz,1))
          wt2(3)=wt2(3)+cc(idd1,icz)*cc(idd2,2)*cc(idd3,icz)*vv2
     *    *exp(-qgpomi(scm,bb2,0.d0,0.d0,0.d0,idd3,idd2,icz,1))
         enddo
        enddo
        enddo
        do l=1,2
         gz(l)=gz(l)+a1(i1)*((1.d0-wt1(l))+(1.d0-wt2(l))/z)
        enddo
        gz(3)=gz(3)+a1(i1)*((wt1(2)-wt1(3))+(wt2(2)-wt2(3))/z)
        gz(4)=gz(4)+a1(i1)*((wt1(3)-wt1(1)**2)+(wt2(3)-wt2(1)**2)/z)
        gz(5)=gz(5)+a1(i1)*((1.d0-wt1(1))*z+(1.d0-wt2(1))/z*(1.-log(z)))

       else
        do idd1=1,2
        do idd2=1,2
         vv1=exp(-qgpomi(scm,bb1,0.d0,0.d0,0.d0,iddp1,idd1,icz,1)
     *   -qgpomi(scm,bb1,0.d0,0.d0,0.d0,iddp2,idd2,icz,1))
         vv2=exp(-qgpomi(scm,bb2,0.d0,0.d0,0.d0,iddp1,idd1,icz,1)
     *   -qgpomi(scm,bb2,0.d0,0.d0,0.d0,iddp2,idd2,icz,1))

         if(idd1.eq.idd2)then
          wt1(1)=wt1(1)+cc(idd1,2)*vv1
          wt2(1)=wt2(1)+cc(idd1,2)*vv2
         endif
         wt1(2)=wt1(2)+cc(idd1,2)*cc(idd2,2)*vv1
         wt2(2)=wt2(2)+cc(idd1,2)*cc(idd2,2)*vv2
        enddo
        enddo
        cg1=qgrot(b,dsqrt(bb1))
        cg2=qgrot(b,dsqrt(bb2))
        do l=1,2
         gz(l)=gz(l)+a1(i1)*(cg1*(1.d0-wt1(l))+cg2*(1.d0-wt2(l))/z)
        enddo
       endif
      enddo
      enddo
      if(ia(2).eq.1.and.iddp1.eq.0.and.iddp2.eq.0)then     !hadron-proton
       do l=1,5
        gz(l)=gz(l)*g0
       enddo
       gz(5)=gz(5)/gz(1)*(rq(1,icz)+rq(1,2)+alfp*log(scm))*2.d0
      endif

      if(debug.ge.2)write (moniou,203)gz
      if(debug.ge.3)write (moniou,202)
201   format(2x,'qgfz - hadronic cross-sections calculation'
     */4x,'b=',e10.3,2x,'iddp=',2i3)
202   format(2x,'qgfz - end')
203   format(2x,'qgfz: gz=',5e10.3)
      return
      end

c=============================================================================
      double precision function qghard(sy,bb,icdp,icdt,icz)
c-----------------------------------------------------------------------------
c qghard - hard quark-quark interaction cross-section
c s - energy squared for the interaction (hadron-hadron),
c icz - type of the primaty hadron (nucleon)
c----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr25/ ahv(3)
      common /qgarr26/ factk,fqscal
      common /arr3/   x1(7),a1(7)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)sy,icz

      qghard=0.d0
      s2min=4.d0*fqscal*qt0
      xmin=s2min/sy
      if(xmin.ge.1.d0)return
      xmin=xmin**(delh+.5d0)

c numerical integration over z1
      do i=1,7
      do m=1,2
       z1=(.5d0*(1.d0+xmin-(2*m-3)*x1(i)*(1.d0-xmin)))
     * **(1.d0/(delh+.5d0))

       st2=0.d0
       do j=1,7
       do k=1,2
        xx=.5d0*(1.d0+x1(j)*(2*k-3))
        xp=z1**xx
        xm=z1/xp
        st2=st2+a1(j)*(1.d0-xp)**ahv(icz)*(1.d0-xm)**ahv(2)
     *  *(qggrv(xp,qt0,icz,1)+qggrv(xp,qt0,icz,2))
     *  *(qggrv(xm,qt0,2,1)+qggrv(xm,qt0,2,2))/dsqrt(z1)
       enddo
       enddo
       sj=qgjit(qt0,qt0,z1*sy,2,2)
       st2=-st2*dlog(z1)*sj
       if(debug.ge.3)write (moniou,203)z1*sy,sj

       qghard=qghard+a1(i)/z1**delh*st2
      enddo
      enddo
      qghard=qghard*(1.d0-xmin)/(.5d0+delh)*.25d0*factk
      rh=rq(icdp,icz)+rq(icdt,2)
      qghard=qghard/(8.d0*pi*rh)*exp(-bb/(4.d0*.0389d0*rh))
     **cd(icdp,icz)*cd(icdt,2)

      if(debug.ge.2)write (moniou,202)qghard
201   format(2x,'qghard - hard quark-quark interaction eikonal:'
     */2x,'s=',e10.3,2x,'icz=',i1)
202   format(2x,'qghard=',e10.3)
203   format(2x,'qghard:',2x,'s_hard=',e10.3,2x,'sigma_hard=',e10.3)
      return
      end

c=============================================================================
      subroutine qgbdef(bba,bbb,xxa,yya,xxb,yyb,xxp,yyp,jb)
c-----------------------------------------------------------------------
c qgbdef - defines coordinates (xxp,yyp) of a multi-pomeron vertex
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)

      xx=xxa-xxb
      yy=yya-yyb
      bb=xx**2+yy**2
      if(bb.lt.1.d-5)then
       xxp=xxb+dsqrt(bba)
       yyp=yyb
      elseif(abs(yy).lt.1.d-8)then
       xxp=(bba-bbb+xxb**2-xxa**2)/2.d0/(xxb-xxa)
       yyp=yyb+(2*jb-3)*dsqrt(max(0.d0,bbb-(xxb-xxp)**2))
      else
       bbd=bb+bbb-bba
       discr=max(0.d0,4.d0*bb*bbb-bbd**2)
       xxp=(xx*bbd+(2*jb-3)*abs(yy)*dsqrt(discr))/2.d0/bb
       yyp=(bbd-2.d0*xx*xxp)/2.d0/yy
       xxp=xxp+xxb
       yyp=yyp+yyb
      endif
      return
      end

c=============================================================================
      subroutine qgv(x,y,xb,vin,vdd,vabs)
c xxv - eikonal dependent factor for hadron-nucleus interaction
c (used for total and diffractive hadron-nucleus cross-sections calculation)
c----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      dimension xb(iapmax,3),vabs(2)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)x,y

      vin=0.d0
      vdd=0.d0
      do iddp1=1,2
       dv=0.d0
       do m=1,ia(2)
        bb=(x-xb(m,1))**2+(y-xb(m,2))**2
        dv=dv+qgpomi(scm,bb,0.d0,0.d0,0.d0,iddp1,iddt(m),icz,1)
       enddo
       dv=exp(-dv)
       vabs(iddp1)=1.d0-dv**2       !1-exp(-2 * chi_i)
       vdd=vdd+cc(iddp1,icz)*dv**2  !sum_i cc(i) exp(-2 * chi_i)
       vin=vin+cc(iddp1,icz)*dv     !sum_i cc(i) exp(-chi_i)
      enddo
      vin=1.d0-vin**2               !1-sum_ij cc(i) cc(j) exp(-chi_i-chi_j)
      vdd=vdd+vin-1.d0
          !sum_i cc(i) exp(-2*chi_i) - sum_ij cc(i) cc(j) exp(-chi_i-chi_j)

      if(debug.ge.3)write (moniou,202)vin,vdd,vabs
201   format(2x,'qgv - eikonal factor: nucleon coordinates x='
     *  ,e10.3,2x,'y=',e10.3)
202   format(2x,'vin=',e10.3,2x,'vdd=',e10.3,2x,'vabs=',2e10.3)
      return
      end


c=============================================================================
      subroutine qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt
     *  ,vvxpl,vvxtl,ip,it)
c-----------------------------------------------------------------------
c qgfdf - configuration of fan contributions (cut and uncut fans)
c xxp, yyp -  coordinates of the multi-Pomeron vertex,
c xpomr    - LC momentum share of the multi-Pomeron vertex,
c ip       - proj. index,
c it       - targ. index
c vvx   = 1 - exp[-sum_{j<J} chi_targ(j) - sum_{i<I} chi_proj(i)]
c vvxp  = 1 - exp[-sum_{i>I} chi_proj(i)]
c vvxt  = 1 - exp[-sum_{j>J} chi_targ(j)]
c vvxpl = 1 - exp[-sum_{i<I} chi_proj(i)]
c vvxtl = 1 - exp[-sum_{j<J} chi_targ(j)]
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      dimension vpac(iapmax),vtac(iapmax)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr7/  xa(iapmax,3),xb(iapmax,3),b
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr43/ moniou
      common /qgarr46/ iconab(iapmax,iapmax),icona(iapmax)
     *,iconb(iapmax)
      common /qgdebug/   debug

      if(debug.ge.3)write (moniou,201)xxp,yyp,xpomr,ip,it

      vvx=0.d0
      vvxp=0.d0
      vvxt=0.d0
      vvxpl=0.d0
      vvxtl=0.d0
      if(scm.le.sgap**2)return

      sumup0=0.d0                      !proj. fans without targ. screening
      do ipp=1,ia(1)
       if(iconab(ipp,it).eq.0)then    !no connection
                                      !(nucleon too far from the vertex)
        vpac(ipp)=0.d0
       else
        bbp=(xa(ipp,1)+b-xxp)**2+(xa(ipp,2)-yyp)**2
        vpac(ipp)=qgfani(1.d0/xpomr,bbp,1.d0-exp(-sumup0),0.d0,0.d0
     *  ,iddp(ipp),icz,1)
        sumup0=sumup0+vpac(ipp)
       endif
      enddo

      sumut0=0.d0                      !targ. fans without proj. screening
      do itt=1,ia(2)
       if(iconab(ip,itt).eq.0)then     !no connection
        vtac(itt)=0.d0
       else
        bbt=(xb(itt,1)-xxp)**2+(xb(itt,2)-yyp)**2
        vtac(itt)=qgfani(xpomr*scm,bbt,1.d0-exp(-sumut0),0.d0,0.d0
     *  ,iddt(itt),2,1)
        sumut0=sumut0+vtac(itt)
       endif
      enddo

      nn=0
1     nn=nn+1
      sumup=0.d0                       !proj. fans with targ. screening
      do ipp=1,ia(1)
       if(iconab(ipp,it).eq.0)then    !no connection
        vpac(ipp)=0.d0
       else
        bbp=(xa(ipp,1)+b-xxp)**2+(xa(ipp,2)-yyp)**2
        vpac(ipp)=qgfani(1.d0/xpomr,bbp,1.d0-exp(-sumup-sumut0)
     *  ,0.d0,0.d0,iddp(ipp),icz,1)
        sumup=sumup+vpac(ipp)
       endif
      enddo

      sumut=0.d0                      !targ. uncut fans with proj. screening
      do itt=1,ia(2)
       if(iconab(ip,itt).eq.0)then
        vtac(itt)=0.d0
       else
        bbt=(xb(itt,1)-xxp)**2+(xb(itt,2)-yyp)**2
        vtac(itt)=qgfani(xpomr*scm,bbt,1.d0-exp(-sumut-sumup0)
     *  ,0.d0,0.d0,iddt(itt),2,1)
        sumut=sumut+vtac(itt)
       endif
      enddo

      if((abs(sumup-sumup0).gt..01d0.or.abs(sumut-sumut0).gt..01d0)
     *.and.nn.lt.100)then
       sumup0=sumup
       sumut0=sumut
       goto 1
      endif

      if(ia(1).gt.1)then
       do ipp=1,ia(1)
        if(ipp.lt.ip)then
         vvxpl=vvxpl+vpac(ipp)
        elseif(ipp.gt.ip)then
         vvxp=vvxp+vpac(ipp)
        endif
       enddo
      endif

      if(ia(2).gt.1)then
       do itt=1,ia(2)
        if(itt.lt.it)then
         vvxtl=vvxtl+vtac(itt)
        elseif(itt.gt.it)then
         vvxt=vvxt+vtac(itt)
        endif
       enddo
      endif
      vvx=1.d0-exp(-vvxpl-vvxtl)
      vvxp=1.d0-exp(-vvxp)
      vvxpl=1.d0-exp(-vvxpl)
      vvxt=1.d0-exp(-vvxt)
      vvxtl=1.d0-exp(-vvxtl)
      if(debug.ge.4)write (moniou,202)

201   format(2x,'qgfdf - configuration of fan contributions:'
     */2x,'xxp=',e10.3,2x,'yyp=',e10.3,2x,'xpomr=',e10.3
     *,2x,'ip=',i3,2x,'it=',i3)
202   format(2x,'qgfdf - end')
      return
      end

c=============================================================================
      subroutine qgconf
c-----------------------------------------------------------------------------
c interaction (cut Pomeron) configuration:
c b - impact parameter,
c xa(1-iap,3), xb(1-iat,3) - proj. and targ. nucleon coordinates,
c iddp(1-iap), iddt(1-iat) - proj. and targ. nucleon diffractive eigenstates,
c icona(1-iap) - connection for proj. nucleons (0 if too far from the target),
c iconab(1-iap,1-iat) - connection for proj.-targ. nucleons (0 if too far from
c each other),
c nwp, nwt - numbers of wounded proj. and targ. nucleons (inelastic or diff.),
c iwp(1-iap), iwt(1-iat) - indexes for wounded proj. and targ. nucleons
c (0 - intact, 1 - inel., 2,3 - diffr., -1 - recoiled from diffraction),
c ncola(1-iap), ncolb(1-iat) - index for inel.-wounded proj. and targ. nucleons,
c nbpom  - total number of Pomeron blocks,
c ias(k) (ibs(k)) - index of the proj. (targ.) nucleon for k-th Pomeron block,
c bbpom(k) - squared impact parameter (between proj. and targ.) for k-th block,
c vvxpom(k) - relative strenth of A-screening corrections for k-th block,
c nqs(k) - number of single Pomerons in k-th block (without cut 3P-vertexes),
c npompr(k) - number of proj. leg Pomerons in k-th block,
c npomtg(k) - number of targ. leg Pomerons in k-th block,
c npomin(k) - number of interm. Pomerons (between 2 3P-vertexes) in k-th block,
c xpopin(n,k) - LC momentum of the upper 3P-vertex for n-th interm. Pomeron
c in k-th block,
c xpomin(n,k) - LC momentum of the lower 3P-vertex for n-th interm. Pomeron
c in k-th block,
c nnpr(i,k) - proj. participant index for i-th single Pomeron in k-th block,
c nntg(i,k) - targ. participant index for i-th single Pomeron in k-th block,
c ilpr(i,k) - proj. index for i-th proj. leg Pomeron in k-th block,
c iltg(i,k) - proj. index for i-th targ. leg Pomeron in k-th block,
c lnpr(i,k) - proj. participant index for i-th proj. leg Pomeron in k-th block,
c lntg(i,k) - targ. participant index for i-th targ. leg Pomeron in k-th block,
c lqa(ip) - number of cut Pomerons connected to ip-th proj. nucleon (hadron),
c lqb(it) - number of cut Pomerons connected to it-th targ. nucleon (hadron),
c nbpi(n,ip) - block index for n-th Pomeron connected to ip-th proj. nucleon,
c nbti(n,it) - block index for n-th Pomeron connected to it-th targ. nucleon,
c idnpi(n,ip) - type of n-th Pomeron (0 - single, 1 - leg) connected to ip-th
c proj. nucleon,
c idnti(n,it) - type of n-th Pomeron (0 - single, 1 - leg) connected to it-th
c targ. nucleon,
c nppi(n,ip) - index in the block of n-th Pomeron connected to ip-th proj.
c nucleon (for single Pomerons),
c npti(n,it) - index in the block of n-th Pomeron connected to it-th targ.
c nucleon (for single Pomerons),
c nlpi(n,ip) - index in the block of n-th Pomeron connected to ip-th proj.
c nucleon (for leg Pomerons),
c nlti(n,it) - index in the block of n-th Pomeron connected to it-th targ.
c nucleon (for leg Pomerons),
c iprcn(ip) - index of the recoiled targ. nucleon for ip-th proj. nucleon
c (undergoing diffraction),
c itgcn(it) - index of the recoiled proj. nucleon for it-th targ. nucleon
c (undergoing diffraction),
c bpompr(n,ip) - squared impact parameter for n-th leg Pomeron connected
c to ip-th proj. nucleon,
c bpomtg(n,it) - squared impact parameter for n-th leg Pomeron connected
c to it-th targ. nucleon,
c vvxpr(n,ip) - relative strenth of A-screening corrections for n-th leg
c Pomeron connected to ip-th proj. nucleon,
c vvxtg(n,it) - relative strenth of A-screening corrections for n-th leg
c Pomeron connected to it-th targ. nucleon,
c xpompr(n,ip) - LC momentum of the 3P-vertex for n-th leg Pomeron connected
c to ip-th proj. nucleon,
c xpomtg(n,it) - LC momentum of the 3P-vertex for n-th leg Pomeron connected
c to it-th targ. nucleon
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208,npbmax=1000,npnmax=900,npmax=900,legmax=900)
      dimension xas(iapmax,3),vabs(2),vabsi(2,iapmax),wdifi(iapmax)
     *,vpac(iapmax),vtac(iapmax),xpomip(npmax),xpomim(npmax)
     *,vvxim(npmax),bpomim(npmax),xpompi(legmax),xpomti(legmax)
     *,vvxpi(legmax),vvxti(legmax),bpompi(legmax),bpomti(legmax)
     *,ipompi(legmax),ipomti(legmax),ncola(iapmax),ncolb(iapmax)
     *,wdp(2,iapmax),wdt(2,iapmax),wabs(2,2),xrapmin(100),xrapmax(100)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr4/  ey0(3)
      common /qgarr6/  pi,bm,amws
      common /qgarr7/  xa(iapmax,3),xb(iapmax,3),b
      common /qgarr9/  iwp(iapmax),iwt(iapmax),lqa(iapmax),lqb(iapmax)
     *,iprcn(iapmax),itgcn(iapmax),ias(npbmax),ibs(npbmax),nqs(npbmax)
     *,npompr(npbmax),npomtg(npbmax),npomin(npbmax),nnpr(npmax,npbmax)
     *,nntg(npmax,npbmax),ilpr(legmax,npbmax),iltg(legmax,npbmax)
     *,lnpr(legmax,npbmax),lntg(legmax,npbmax)
     *,nbpi(npnmax,iapmax),nbti(npnmax,iapmax),idnpi(npnmax,iapmax)
     *,idnti(npnmax,iapmax),nppi(npnmax,iapmax),npti(npnmax,iapmax)
     *,nlpi(npnmax,iapmax),nlti(npnmax,iapmax)
      common /qgarr10/ am(7),ammu
      common /qgarr11/ b10
      common /qgarr12/ nsp
      common /qgarr13/ nsf,iaf(iapmax)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr23/ bbpom(npbmax),vvxpom(npbmax)
     *,bpompr(npnmax,iapmax),bpomtg(npnmax,iapmax)
     *,vvxpr(npnmax,iapmax),vvxtg(npnmax,iapmax)
     *,xpompr(npnmax,iapmax),xpomtg(npnmax,iapmax)
     *,xpopin(npmax,npbmax),xpomin(npmax,npbmax),vvxin(npmax,npbmax)
     *,bpomin(npmax,npbmax)
      common /qgarr43/ moniou
      common /qgarr46/ iconab(iapmax,iapmax),icona(iapmax)
     *,iconb(iapmax)
      common /qgarr55/ nwt,nwp       !N of wounded targ.(proj.) nucleons
      common /qgarr56/ nspec,nspect  !N of spectators targ.(proj.) nucleons
      common /qgdebug/  debug
      common /qgsIInex1/xan(iapmax,3),xbn(iapmax,3) !used to link with nexus
     *,bqgs,bmaxqgs,bmaxnex,bminnex
      common/jdiff/bdiff,jdiff     !for external use: impact parameter
                                   !for diffraction, diffraction type
ctp from epos
      integer ng1evt,ng2evt,ikoevt
      real    rglevt,sglevt,eglevt,fglevt,typevt
      common/c2evt/ng1evt,ng2evt,rglevt,sglevt,eglevt,fglevt,ikoevt
     *,typevt            !in epos.inc

      external qgran

      if(debug.ge.1)write (moniou,201)
      nsp=0
      nsf=0
      nsp0=nsp

c initialization
1     continue
      do i=1,ia(1)
       iddp(i)=1+int(qgran(b10)+cc(2,icz)) !diffractive eigenstates for proj.
      enddo
      do i=1,ia(2)
       iddt(i)=1+int(qgran(b10)+cc(2,2))   !diffractive eigenstates for targ.
      enddo

c-------------------------------------------------
c squared impact parameter is sampled uniformly (b**2<bm**2)
      b=bm*dsqrt(qgran(b10))
      if(debug.ge.1)write (moniou,202)b

      if(bmaxnex.ge.0.d0)then              !used to link with nexus
       b1=bminnex
       b2=min(bm,bmaxnex)
       if(b1.gt.b2)stop'bmin > bmax in qgsjet'
       b=dsqrt(b1*b1+(b2*b2-b1*b1)*qgran(b10))
       bqgs=b
      endif

c-------------------------------------------------
c nuclear configurations
      if(debug.ge.1)write (moniou,203)
      if(ia(1).gt.1)then          !projectile nucleon coordinates
       call qggea(ia(1),xa,1)     !xa(n,i), i=1,2,3 - x,y,z for n-th nucleon
      else
       do i=1,3
        xa(1,i)=0.d0              !projectile hadron
       enddo
      endif
      if(ia(2).gt.1)then          !target nucleon coordinates
       call qggea(ia(2),xb,2)     !xb(n,i), i=1,2,3 - x,y,z for n-th nucleon
      else
       do i=1,3
        xb(1,i)=0.d0              !target proton
       enddo
      endif

c-------------------------------------------------
c check connections
      if(debug.ge.1)write (moniou,204)
      do it=1,ia(2)
       iconb(it)=0
      enddo

      do ip=1,ia(1)
       icdp=iddp(ip)
       icona(ip)=0
       do it=1,ia(2)
        icdt=iddt(it)
        bbp=(xa(ip,1)+b-xb(it,1))**2+(xa(ip,2)-xb(it,2))**2
        vv1p=qgpomi(scm,bbp,0.d0,0.d0,0.d0,icdp,icdt,icz,1)
        if(vv1p.gt.1.d-3)then
         if(debug.ge.2)write (moniou,205)ip,it
         iconab(ip,it)=1
         icona(ip)=1
         iconb(it)=1
         if(debug.ge.2)write (moniou,206)ip
         if(debug.ge.2)write (moniou,207)it
        else
         iconab(ip,it)=0
        endif
       enddo
      enddo

      nrej=0
2     nrej=nrej+1
      if(debug.ge.2)write (moniou,208)nrej
      if(nrej.gt.10)then
       if(debug.ge.1)write (moniou,209)
       goto 1
      endif
      nsp=nsp0
      nbpom=0
      nwp=0
      nwt=0
      do i=1,ia(1)
       lqa(i)=0
       iwp(i)=0
       ncola(i)=0
       wdp(1,i)=0.d0
       wdp(2,i)=0.d0
      enddo
      do i=1,ia(2)
       lqb(i)=0
       iwt(i)=0
       ncolb(i)=0
       wdt(1,i)=0.d0
       wdt(2,i)=0.d0
      enddo
      nqs(1)=0
      npomin(1)=0
      npompr(1)=0
      npomtg(1)=0

c-------------------------------------------------
c Pomeron configuration
      if(debug.ge.1)write (moniou,210)
      do 4 ip=1,ia(1)             !loop over all projectile nucleons
       if(debug.ge.2)write (moniou,211)ip
       if(icona(ip).eq.0)goto 4
       x=xa(ip,1)+b               !proj. x is shifted by the impact parameter b
       y=xa(ip,2)
       icdp=iddp(ip)              !diffr. eigenstate for ip

       do 3 it=1,ia(2)            !loop over all target nucleons
        if(debug.ge.2)write (moniou,212)it
        if(iconab(ip,it).eq.0)goto 3
        icdt=iddt(it)                         !diffr. eigenstate for it
        bbp=(x-xb(it,1))**2+(y-xb(it,2))**2   !distance squared between ip, it

c calculate nuclear screening factors for "middle point" -> eikonals
        xpomr=1.d0/dsqrt(scm)
        xxp=.5d0*(x+xb(it,1))
        yyp=.5d0*(y+xb(it,2))
        call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *  ,ip,it)
        vv=qgpomi(scm,bbp,vvx,vvxp,vvxt,icdp,icdt,icz,1)        !total eikonal
        vv1p=min(vv,qgpomi(scm,bbp,vvx,vvxp,vvxt,icdp,icdt,icz,2)) !1P-eikonal
        if(debug.ge.2)write (moniou,213)vv,vv1p

        if(qgran(b10).gt.1.d0-exp(-2.d0*vv))goto 3 !1.-exp(-2*vv) - probability
                                                   !for inelastic interaction
        iwt(it)=1
        iwp(ip)=1
        ncola(ip)=ncola(ip)+1                   !N of binary collisions for ip
        ncolb(it)=ncolb(it)+1                   !N of binary collisions for it

        n=npgen(2.d0*vv,1,50) !number of elem. inter. for (ip-it) collision
        nbpom=nbpom+1         !new Pomeron block
        if(nbpom.gt.npbmax)then
         goto 2
        endif
        ias(nbpom)=ip         !proj. index for current elementary interaction
        ibs(nbpom)=it         !targ. index for current elementary interaction
        bbpom(nbpom)=bbp      !distance squared between ip, it
        vvxpom(nbpom)=1.d0-(1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt)
        if(debug.ge.2)write (moniou,214)nbpom,ip,it,n

        nqs(nbpom)=0
        npomin(nbpom)=0
        npompr(nbpom)=0
        npomtg(nbpom)=0
        do i=1,n
         if(qgran(b10).lt.vv1p/vv.or.scm.le.sgap**2)then  !single Pomeron
          if(debug.ge.2)write (moniou,215)i
          np=nqs(nbpom)+1
          if(np.gt.legmax)then
           goto 2
          endif
          nqs(nbpom)=np                  !update Pomeron number in the block
          l0=lqa(ip)+1
          if(l0.gt.npnmax)then
           goto 2
          endif
          lqa(ip)=l0                     !update number of connections for proj.
          nnpr(np,nbpom)=l0              !index for connected proj. participant
          nbpi(l0,ip)=nbpom
          idnpi(l0,ip)=0
          nppi(l0,ip)=np
          l0=lqb(it)+1
          if(l0.gt.npnmax)then
           goto 2
          endif
          lqb(it)=l0
          nntg(np,nbpom)=l0              !index for connected targ. participant
          nbti(l0,it)=nbpom
          idnti(l0,it)=0
          npti(l0,it)=np

         else                            !multi-Pomeron vertex
          if(debug.ge.2)write (moniou,219)
          call qg3pdf(vvxpi,vvxti,xpompi,xpomti,bpompi,bpomti,xpomip
     *    ,xpomim,vvxim,bpomim,npompi,npomti,npin,ipompi,ipomti
     *    ,wdp,wdt,ip,it,iret)
          if(iret.ne.0)goto 2

          if(npin.ne.0)then
           if(debug.ge.2)write (moniou,220)i,npin
           npomin(nbpom)=npomin(nbpom)+npin
           if(npomin(nbpom).gt.npmax)then
            goto 2
           endif
           do l=1,npin
            l1=npomin(nbpom)+l-npin
            xpopin(l1,nbpom)=xpomip(l)
            xpomin(l1,nbpom)=xpomim(l)
            vvxin(l1,nbpom)=vvxim(l)
            bpomin(l1,nbpom)=bpomim(l)
           enddo
          endif
          if(npompi.ne.0)then
           if(debug.ge.2)write (moniou,221)i,npompi
           do m=1,npompi
            np=npompr(nbpom)+1
            if(np.gt.legmax)then
             goto 2
            endif
            npompr(nbpom)=np
            ipp=ipompi(m)
            iwp(ipp)=1
            ilpr(np,nbpom)=ipp
            l0=lqa(ipp)+1
            if(l0.gt.npnmax)then
             goto 2
            endif
            lqa(ipp)=l0
            lnpr(np,nbpom)=l0
            nbpi(l0,ipp)=nbpom
            idnpi(l0,ipp)=1
            nlpi(l0,ipp)=np
            vvxpr(l0,ipp)=vvxpi(m)
            xpompr(l0,ipp)=1.d0/xpompi(m)/scm
            bpompr(l0,ipp)=bpompi(m)
           enddo
          endif
          if(npomti.ne.0)then
           if(debug.ge.2)write (moniou,222)i,npomti
           do m=1,npomti
            np=npomtg(nbpom)+1
            if(np.gt.legmax)then
             goto 2
            endif
            npomtg(nbpom)=np
            itt=ipomti(m)
            iwt(itt)=1
            iltg(np,nbpom)=itt
            l0=lqb(itt)+1
            if(l0.gt.npnmax)then
             goto 2
            endif
            lqb(itt)=l0
            lntg(np,nbpom)=l0
            nbti(l0,itt)=nbpom
            idnti(l0,itt)=1
            nlti(l0,itt)=np
            vvxtg(l0,itt)=vvxti(m)
            xpomtg(l0,itt)=xpomti(m)
            bpomtg(l0,itt)=bpomti(m)
           enddo
          endif
         endif
        enddo                   !end of Pomeron loop
3      continue                 !end of it-loop
4     continue                  !end of ip-loop

c-------------------------------------------------
c   diffraction (hadron-hadron case)
      if(ia(1).eq.1.and.ia(2).eq.1.and.iwp(1).eq.0.and.iwt(1).eq.0)then
       wel=0.d0
       winel=0.d0
       do icdp=1,2
       do icdt=1,2
        vv=qgpomi(scm,b*b,0.d0,0.d0,0.d0,icdp,icdt,icz,1)   !total eikonal
        wabs(icdp,icdt)=exp(-vv)
        wel=wel+cc(icdp,icz)*cc(icdt,2)*wabs(icdp,icdt)
        winel=winel+cc(icdp,icz)*cc(icdt,2)*wabs(icdp,icdt)**2
       enddo
       enddo
       if(qgran(b10).le.wel**2/winel)then
        if(debug.ge.1)write (moniou,231)
        goto 1
       endif

       wdifp=cc(1,icz)*cc(2,icz)*(cc(1,2)**2*(wabs(1,1)-wabs(2,1))**2
     * +cc(2,2)**2*(wabs(1,2)-wabs(2,2))**2+2.d0*cc(1,2)*cc(2,2)
     * *(wabs(1,1)-wabs(2,1))*(wabs(1,2)-wabs(2,2)))
       wdift=cc(1,2)*cc(2,2)*(cc(1,icz)**2*(wabs(1,1)-wabs(1,2))**2
     * +cc(2,icz)**2*(wabs(2,1)-wabs(2,2))**2+2.d0*cc(1,icz)*cc(2,icz)
     * *(wabs(1,1)-wabs(1,2))*(wabs(2,1)-wabs(2,2)))
       wdifd=cc(1,icz)*cc(2,icz)*cc(1,2)*cc(2,2)
     * *(wabs(1,1)+wabs(2,2)-wabs(1,2)-wabs(2,1))**2
       aks=(wdifp+wdift+wdifd)*qgran(b10)
       if(aks.lt.wdifp)then
        nwp=nwp+1
        iwp(1)=2
        iprcn(1)=1
        iwt(1)=-1
       elseif(aks.lt.wdifp+wdift)then
        nwt=nwt+1
        iwt(1)=2
        itgcn(1)=1
        iwp(1)=-1
       else
        nwp=nwp+1
        nwt=nwt+1
        iwp(1)=2
        iwt(1)=2
        iprcn(1)=1
        itgcn(1)=1
       endif
       goto 9
      endif

c-------------------------------------------------
c   diffraction (hadron-nucleus & nucleus-nucleus)
      do ip=1,ia(1)             !loop over all projectile nucleons
       x=xa(ip,1)+b             !proj. x is shifted by b
       y=xa(ip,2)
       if(iwp(ip).ne.0)then
        nwp=nwp+1               !one more wounded proj. nucleon
        if(lqa(ip).eq.0.and.(wdp(1,ip).ne.0.d0.or.wdp(2,ip).ne.0.d0))
     *  then
         icdps=iddp(ip)
         xpomr=1.d0/dsqrt(scm)
         do it=1,ia(2)
          if(iconab(ip,it).ne.0)then
            bbp=(x-xb(it,1))**2+(y-xb(it,2))**2
           xxp=.5d0*(x+xb(it,1))
           yyp=.5d0*(y+xb(it,2))
           icdt=iddt(it)
           do icdp=1,2
            iddp(ip)=icdp
            call qgfdf(xxp,yyp,xpomr,vpac,vtac
     *      ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,it)
            vv=qgpomi(scm,bbp,vvx,vvxp,vvxt,icdp,icdt,icz,1)   !total eikonal
            wdp(icdp,ip)=wdp(icdp,ip)*exp(-vv)
           enddo
          endif
         enddo
         iddp(ip)=icdps
         wdifr=cc(1,icz)*cc(2,icz)*(wdp(1,ip)-wdp(2,ip))**2
     *   /(cc(1,icz)*wdp(1,ip)**2+cc(2,icz)*wdp(2,ip)**2)
         if(qgran(b10).lt.wdifr)iwp(ip)=3                     !LMD excitation
        endif

       elseif(icona(ip).ne.0)then
        if(debug.ge.2)write (moniou,223)ip
        vabs(1)=0.d0
        vabs(2)=0.d0
        icdps=iddp(ip)
        do it=1,ia(2)
          bbp=(x-xb(it,1))**2+(y-xb(it,2))**2
         icdt=iddt(it)
         do icdp=1,2
          if(iconab(ip,it).eq.0)then
           vabsi(icdp,it)=0.d0
          else
           iddp(ip)=icdp
           xpomr=1.d0/dsqrt(scm)
           xxp=.5d0*(x+xb(it,1))
           yyp=.5d0*(y+xb(it,2))
           call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *     ,ip,it)
           vv=qgpomi(scm,bbp,vvx,vvxp,vvxt,icdp,icdt,icz,1)   !total eikonal
           vabsi(icdp,it)=vv
           vabs(icdp)=vabs(icdp)+vv
          endif
         enddo
        enddo
        iddp(ip)=icdps
        wdifr=cc(1,icz)*cc(2,icz)*(exp(-vabs(1))-exp(-vabs(2)))**2
     *  /(cc(1,icz)*exp(-2.d0*vabs(1))+cc(2,icz)*exp(-2.d0*vabs(2)))

        if(qgran(b10).lt.wdifr)then       !projectile diffraction
         wdift=0.d0
         do it=1,ia(2)
          if(iwt(it).ne.-1)then
           wdifi(it)=cc(1,icz)*cc(2,icz)*(exp(-vabsi(1,it))
     *     -exp(-vabsi(2,it)))**2/(cc(1,icz)*exp(-2.d0*vabsi(1,it))
     *     +cc(2,icz)*exp(-2.d0*vabsi(2,it)))
           wdift=wdift+wdifi(it)
          else
           wdifi(it)=0.d0
          endif
         enddo
         if(wdift.ne.0.d0)then
          nwp=nwp+1
          iwp(ip)=2
          aks=qgran(b10)*wdift
          do it=1,ia(2)
           aks=aks-wdifi(it)
           if(aks.lt.0.d0)goto 5
          enddo
5          continue
          iprcn(ip)=it
          if(iwt(it).eq.0)iwt(it)=-1
          if(debug.ge.2)write (moniou,224)ip,it
         endif
        endif
       endif
      enddo                            !end of ip-loop

      do 8 it=1,ia(2)                     !check target diffraction
       if(iwt(it).gt.0)then
        nwt=nwt+1                         !one more wounded targ. nucleon
        if(lqb(it).eq.0.and.(wdt(1,it).ne.0.d0.or.wdt(2,it).ne.0.d0))
     *  then
         icdts=iddt(it)
         xpomr=1.d0/dsqrt(scm)
         do ip=1,ia(1)
          if(iconab(ip,it).ne.0)then
           bbp=(xa(ip,1)+b-xb(it,1))**2+(xa(ip,2)-xb(it,2))**2
           xxp=.5d0*(xa(ip,1)+b+xb(it,1))
           yyp=.5d0*(xa(ip,2)+xb(it,2))
           icdp=iddp(ip)
           do icdt=1,2
            iddt(it)=icdt
            call qgfdf(xxp,yyp,xpomr,vpac,vtac
     *      ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,it)
            vv=qgpomi(scm,bbp,vvx,vvxp,vvxt,icdp,icdt,icz,1)   !total eikonal
             wdt(icdt,it)=wdt(icdt,it)*exp(-vv)
           enddo
          endif
         enddo
         iddt(it)=icdts
         wdifr=cc(1,2)*cc(2,2)*(wdt(1,it)-wdt(2,it))**2
     *   /(cc(1,2)*wdt(1,it)**2+cc(2,2)*wdt(2,it)**2)
         if(qgran(b10).lt.wdifr)iwt(it)=3
        endif

       elseif(iconb(it).ne.0)then
        if(debug.ge.2)write (moniou,225)it
        vabs(1)=0.d0
        vabs(2)=0.d0
        icdts=iddt(it)
        do ip=1,ia(1)
         bbp=(xa(ip,1)+b-xb(it,1))**2+(xa(ip,2)-xb(it,2))**2
         icdp=iddp(ip)
         do icdt=1,2
          if(iconab(ip,it).eq.0)then
           vabsi(icdt,ip)=0.d0
          else
           iddt(it)=icdt
           xpomr=1.d0/dsqrt(scm)
           xxp=.5d0*(xa(ip,1)+b+xb(it,1))
           yyp=.5d0*(xa(ip,2)+xb(it,2))
           call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *     ,ip,it)
           vv=qgpomi(scm,bbp,vvx,vvxp,vvxt,icdp,icdt,icz,1)   !total eikonal
           vabsi(icdt,ip)=vv
           vabs(icdt)=vabs(icdt)+vv
          endif
         enddo
        enddo
        iddt(it)=icdts
        wdifr=cc(1,2)*cc(2,2)*(exp(-vabs(1))-exp(-vabs(2)))**2
     *  /(cc(1,2)*exp(-2.d0*vabs(1))+cc(2,2)*exp(-2.d0*vabs(2)))

        if(qgran(b10).lt.wdifr)then       !target diffraction
         wdift=0.d0
         do ip=1,ia(1)
          if(iwp(ip).eq.-1)then
           wdifi(ip)=0.d0
          else
           if(iwp(ip).eq.2)then
            itt=iprcn(ip)
            if(itt.eq.it)goto 7
            if(iwt(itt).eq.2)then
             wdifi(ip)=0.d0
             goto 6
            endif
           endif
           wdifi(ip)=cc(1,2)*cc(2,2)*(exp(-vabsi(1,ip))
     *     -exp(-vabsi(2,ip)))**2/(cc(1,2)*exp(-2.d0*vabsi(1,ip))
     *     +cc(2,2)*exp(-2.d0*vabsi(2,ip)))
          endif
6          wdift=wdift+wdifi(ip)
         enddo
         if(wdift.eq.0.d0)goto 8
         nwt=nwt+1
         iwt(it)=2
         aks=qgran(b10)*wdift
         do ip=1,ia(1)
          aks=aks-wdifi(ip)
          if(aks.lt.0.d0)goto 7
         enddo
7         continue
         itgcn(it)=ip
         if(debug.ge.2)write (moniou,226)it,ip
         if(iwp(ip).eq.0)then
          iwp(ip)=-1
         elseif(iwp(ip).eq.2)then
          itt=iprcn(ip)
          iprcn(ip)=it
          if(itt.ne.it.and.iwt(itt).eq.-1)iwt(itt)=0
         endif
        endif
       endif
8     continue

c check diffractive cross sections (hadron-proton only)
9     jdiff=0                             !non-diffractive
      if(ia(1).eq.1.and.ia(2).eq.1.and.(nwp.ne.0.or.nwt.ne.0)
     *.and.nqs(1).eq.0)then
       if(lqa(1).eq.0.and.lqb(1).eq.0)then
        if(nbpom.eq.0.or.npomin(1).eq.0)then
         if(iwp(1).ge.2.and.iwt(1).lt.2)then
          jdiff=6                         !SD(LM)-proj
         elseif(iwp(1).lt.2.and.iwt(1).ge.2)then
          jdiff=7                         !SD(LM)-targ
         elseif(iwp(1).ge.2.and.iwt(1).ge.2)then
          jdiff=8                         !DD(LM)
         else
          goto 12
         endif
        else
         if(iwp(1).lt.2.and.iwt(1).lt.2)then
          jdiff=9                         !CD(DPE)
         else
          jdiff=10                        !CD+LMD
         endif
        endif
       elseif(lqa(1).gt.0.and.lqb(1).eq.0.and.iwt(1).lt.2)then
        jdiff=1                          !SD(HM)-proj
       elseif(lqa(1).eq.0.and.lqb(1).gt.0.and.iwp(1).lt.2)then
        jdiff=2                          !SD(HM)-targ
       elseif(lqa(1).gt.0.and.lqb(1).eq.0.and.iwt(1).ge.2)then
        jdiff=3                          !DD(LHM)-proj
       elseif(lqa(1).eq.0.and.lqb(1).gt.0.and.iwp(1).ge.2)then
        jdiff=4                          !DD(LHM)-targ

       elseif(lqa(1).gt.0.and.lqb(1).gt.0)then
        if(npompr(1).eq.0)stop'problem with npompr!!!'
        xrapmax(1)=1.d0
        do i=1,npompr(1)
         xrapmax(1)=min(xrapmax(1),1.d0/xpompr(i,1)/scm)
        enddo
        if(npomtg(1).eq.0)stop'problem with npomtg!!!'
        xrapmin(1)=1.d0/scm
        do i=1,npomtg(1)
         xrapmin(1)=max(xrapmin(1),xpomtg(i,1))
        enddo
        if(xrapmin(1).gt..999d0*xrapmax(1))goto 12
        nraps=1
        irap=1
11      if(nraps.gt.90)stop'nraps>90'
        if(npomin(1).gt.0)then
         do i=1,npomin(1)
          if(xpomin(i,1).lt..999d0*xrapmax(irap)
     *    .and.xpopin(i,1).gt.1.001d0*xrapmin(irap))then
           if(xpomin(i,1).lt.1.001d0*xrapmin(irap)
     *     .and.xpopin(i,1).gt..999d0*xrapmax(irap))then
            nraps=nraps-1
            if(nraps.eq.0)goto 12
            irap=irap-1
            goto 11
           elseif(xpopin(i,1).gt..999d0*xrapmax(irap))then
            xrapmax(irap)=xpomin(i,1)
            if(xrapmin(irap).gt..999d0*xrapmax(irap))then
             nraps=nraps-1
             if(nraps.eq.0)goto 12
             irap=irap-1
             goto 11
            endif
           elseif(xpomin(i,1).lt.1.001d0*xrapmin(irap))then
            xrapmin(irap)=xpopin(i,1)
            if(xrapmin(irap).gt..999d0*xrapmax(irap))then
             nraps=nraps-1
             if(nraps.eq.0)goto 12
             irap=irap-1
             goto 11
            endif
           else
            xrapmin(irap+1)=xrapmin(irap)
            xrapmin(irap)=xpopin(i,1)
            xrapmax(irap+1)=xpomin(i,1)
            if(xrapmin(irap).lt..999d0*xrapmax(irap)
     *      .and.xrapmin(irap+1).lt..999d0*xrapmax(irap+1))then
             irap=irap+1
             nraps=nraps+1
             goto 11
            elseif(xrapmin(irap).lt..999d0*xrapmax(irap))then
             goto 11
            elseif(xrapmin(irap+1).lt..999d0*xrapmax(irap+1))then
             xrapmin(irap)=xrapmin(irap+1)
             xrapmax(irap)=xrapmax(irap+1)
             goto 11
            else
             nraps=nraps-1
             if(nraps.eq.0)goto 12
             irap=irap-1
             goto 11
            endif
           endif
          endif
         enddo                           !end of npin-loop
        endif
        jdiff=5                          !DD(HM)
       endif
      endif                              !end of diffr. check
12    bdiff=b

ctp define collision type
      typevt=0                      !no interaction
      if(ia(1).eq.1.and.ia(2).eq.1.and.(nwp.gt.0.or.nwt.gt.0))then !only for h-h
       if(jdiff.eq.0)then                                  !ND (no rap-gaps)
        typevt=1
       elseif(jdiff.eq.8.or.jdiff.eq.10.or.
     *       (jdiff.gt.2.and.jdiff.lt.6))then !DD + (CD+LMD)
        typevt=2                           
       elseif(jdiff.eq.1.or.jdiff.eq.6)then                  !SD pro
        typevt=4  
       elseif(jdiff.eq.2.or.jdiff.eq.7)then                  !SD tar
        typevt=-4  
       elseif(jdiff.eq.9)then                                !CD
        typevt=3
       else
        stop'problem with typevt!'
       endif
      endif


c form projectile spectator part
      if(debug.ge.1)write (moniou,227)
      nspec=0
      do ip=1,ia(1)
       if(iwp(ip).eq.0)then
        if(debug.ge.2)write (moniou,228)ip
        nspec=nspec+1
        do l=1,3
         xas(nspec,l)=xa(ip,l)
        enddo
       endif
      enddo

      nspect=0
      do it=1,ia(2)
       if(iwt(it).eq.0)nspect=nspect+1
      enddo

c inelastic interaction: energy sharing and particle production
      if(nwp.ne.0.or.nwt.ne.0)then
       if(ia(1).eq.nspec.or.ia(2).eq.nspect)stop'ia(1)=nspec!!!'
       if(debug.ge.1)write (moniou,229)

       call qgsha(nbpom,ncola,ncolb,iret)
       if(iret.ne.0)goto 1
       if(nsp.le.nsp0+2)then
        if(debug.ge.1)write (moniou,230)
        goto 1
       endif
      else                                 !no interaction
       if(debug.ge.1)write (moniou,231)
       goto 1
      endif
      if(debug.ge.1)write (moniou,232)nsp

c fragmentation of the projectile spectator part
      if(debug.ge.1)write (moniou,233)
      call qgfrgm(nspec,xas)
      if(debug.ge.1)write (moniou,234)nsf
      if(debug.ge.1)write (moniou,235)

201   format(2x,'qgconf - configuration of the interaction')
202   format(2x,'qgconf: impact parameter b=',e10.3,' fm')
203   format(2x,'qgconf: nuclear configurations')
204   format(2x,'qgconf: check connections')
205   format(2x,'qgconf: ',i3,'-th proj. nucleon may interact with '
     *,i3,'-th target nucleon')
206   format(2x,'qgconf: ',i3,'-th projectile nucleon may interact')
207   format(2x,'qgconf: ',i3,'-th target nucleon may interact')
208   format(2x,'qgconf: ',i3,'-th rejection,'
     *,' redo Pomeron configuration')
209   format(2x,'qgconf: too many rejections,'
     *,' redo nuclear configuartions')
210   format(2x,'qgconf: Pomeron configuration')
211   format(2x,'qgconf: check ',i3,'-th projectile nucleon')
212   format(2x,'qgconf: interaction with ',i3,'-th target nucleon?')
213   format(2x,'qgconf: eikonals - total: ',e10.3,2x,'single: ',e10.3)
214   format(2x,'qgconf: ',i4,'-th Pomeron block connected to ',i3
     *,'-th proj. nucleon and'/4x,i3,'-th targ. nucleon;'
     *,' number of element. processes in the block: ',i3)
215   format(2x,'qgconf: ',i3
     *,'-th process in the block is single cut Pomeron')
219   format(2x,'qgconf: configuration of multi-Pomeron vertexes')
220   format(2x,'qgconf: ',i3,'-th process in the block contains '
     *,i3,' interm. Pomerons')
221   format(2x,'qgconf: ',i3,'-th process in the block contains '
     *,i3,' proj. legs')
222   format(2x,'qgconf: ',i3,'-th process in the block contains '
     *,i3,' targ. legs')
223   format(2x,'qgconf: check diffraction for ',i3,'-th proj. nucleon')
224   format(2x,'qgconf: diffr. of ',i3,'-th proj. nucleon,'
     *,' recoil of ',i3,'-th targ. nucleon')
225   format(2x,'qgconf: check diffraction for ',i3,'-th targ. nucleon')
226   format(2x,'qgconf: diffr. of ',i3,'-th targ. nucleon,'
     *,' recoil of ',i3,'-th proj. nucleon')
227   format(2x,'qgconf: projectile spectator part')
228   format(2x,'qgconf: ',i3,'-th proj. nucleon stays idle')
229   format(2x,'qgconf: inelastic interaction: energy sharing'
     *,' and particle production')
230   format(2x,'qgconf: no particle produced - rejection')
231   format(2x,'qgconf: no interaction - rejection')
232   format(2x,'qgconf: ',i5,' particles have been produced')
233   format(2x,'qgconf: fragmentation of the proj. spectator part')
234   format(2x,'qgconf: ',i3,' proj. fragments have been produced')
235   format(2x,'qgconf - end')
      return
      end

c=============================================================================
      subroutine qg3pdf(vvxpi,vvxti,xpompi,xpomti,bpompi,bpomti
     *,xpomip,xpomim,vvxim,bpomim,nppr,nptg,npin,ipompi,ipomti
     *,wdp,wdt,ip,it,iret)
c-----------------------------------------------------------------------
c qg3pdf - configuration for multi-Pomeron/diffractive contributions
c ip,it - indexes of proj. and targ. nucleons for current collision
c to determine:
c nppr - number of proj. leg Pomerons in the process,
c nptg - number of targ. leg Pomerons in the process,
c npin - number of interm. Pomerons (between 2 3P-vertexes) in the process,
c xpomip(i) - LC momentum of the upper 3P-vertex for i-th interm. Pomeron
c in the process,
c xpomim(i) - LC momentum of the lower 3P-vertex for i-th interm. Pomeron
c in the process,
c ipompi(i) - proj. index for i-th proj. leg Pomeron in the process,
c ipomti(i) - proj. index for i-th targ. leg Pomeron in the process,
c bpompi(i) - squared impact param. for i-th proj. leg Pomeron in the process,
c bpomti(i) - squared impact param. for i-th targ. leg Pomeron in the process,
c vvxpi(i) - relative strenth of scr. corrections for i-th proj. leg Pomeron,
c vvxti(i) - relative strenth of scr. corrections for i-th targ. leg Pomeron,
c xpompi(i) - LC momentum of the 3P-vertex for i-th proj. leg Pomeron,
c xpomti(i) - LC momentum of the 3P-vertex for i-th targ. leg Pomeron
c iret=1 - reject configuration
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208,npbmax=1000,npnmax=900,npmax=900
     *,levmax=20,legmax=900)
      dimension vpac(iapmax),vtac(iapmax)
     *,vpac0(iapmax),vtac0(iapmax),vpact(iapmax),vtact(iapmax)
     *,xpomip(npmax),xpomim(npmax),vvxim(npmax),bpomim(npmax)
     *,xpompi(legmax),xpomti(legmax)
     *,vvxpi(legmax),vvxti(legmax),bpompi(legmax),bpomti(legmax)
     *,ipompi(legmax),ipomti(legmax),ippr0(legmax),iptg0(legmax)
     *,nppm(levmax),ippm(legmax,levmax),ii(levmax),xpomm(levmax)
     *,wgpm(levmax),xxm(levmax),yym(levmax)
     *,itypr0(legmax),itytg0(legmax),itypm(legmax,levmax),vv(12)
     *,wdp(2,iapmax),wdt(2,iapmax)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr6/  pi,bm,amws
      common /qgarr7/  xa(iapmax,3),xb(iapmax,3),b
      common /qgarr9/  iwp(iapmax),iwt(iapmax),lqa(iapmax),lqb(iapmax)
     *,iprcn(iapmax),itgcn(iapmax),ias(npbmax),ibs(npbmax),nqs(npbmax)
     *,npompr(npbmax),npomtg(npbmax),npomin(npbmax),nnpr(npmax,npbmax)
     *,nntg(npmax,npbmax),ilpr(legmax,npbmax),iltg(legmax,npbmax)
     *,lnpr(legmax,npbmax),lntg(legmax,npbmax)
     *,nbpi(npnmax,iapmax),nbti(npnmax,iapmax),idnpi(npnmax,iapmax)
     *,idnti(npnmax,iapmax),nppi(npnmax,iapmax),npti(npnmax,iapmax)
     *,nlpi(npnmax,iapmax),nlti(npnmax,iapmax)
      common /qgarr11/ b10
      common /qgarr12/ nsp
      common /qgarr13/ nsf,iaf(iapmax)
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr23/ bbpom(npbmax),vvxpom(npbmax)
     *,bpompr(npnmax,iapmax),bpomtg(npnmax,iapmax)
     *,vvxpr(npnmax,iapmax),vvxtg(npnmax,iapmax)
     *,xpompr(npnmax,iapmax),xpomtg(npnmax,iapmax)
     *,xpopin(npmax,npbmax),xpomin(npmax,npbmax),vvxin(npmax,npbmax)
     *,bpomin(npmax,npbmax)
      common /qgarr43/ moniou
      common /qgarr46/ iconab(iapmax,iapmax),icona(iapmax)
     *,iconb(iapmax)
      common /qgdebug/  debug
      external qgran

      if(debug.ge.2)write (moniou,201)ip,it

      if(scm.le.sgap**2)stop'qg3pdf: scm<sgap**2!'
      iret=0
      vpacng=0.d0
      vtacng=0.d0
      vpacpe=0.d0
      vtacpe=0.d0
      vimp=0.d0
      viuc=0.d0
      viuu=0.d0
      vip=0.d0
      vicc=0.d0
      vicu=0.d0
c normalization of rejection function
      xpomr=1.d0/dsqrt(scm)
      bpt=dsqrt((xa(ip,1)+b-xb(it,1))**2+(xa(ip,2)-xb(it,2))**2)
      rp1=(rq(iddp(ip),icz)-alfp*dlog(xpomr))*4.d0*.0389d0
      rp2=(rq(iddt(it),2)+alfp*dlog(xpomr*scm))*4.d0*.0389d0
      rp0=rp1*rp2/(rp1+rp2)
      bbpr=(bpt*rp1/(rp1+rp2))**2
      bbtg=(bpt*rp2/(rp1+rp2))**2
      call qgbdef(bbpr,bbtg,xa(ip,1)+b,xa(ip,2),xb(it,1),xb(it,2)
     *,xxp,yyp,1)

      rpmax=max(rq(iddp(ip),icz),rq(iddt(it),2))*4.d0*.0389d0
      rpmin=min(rq(iddp(ip),icz),rq(iddt(it),2))*4.d0*.0389d0
      if(rpmax.eq.rpmin)then
       rpmax=rpmax+alfp*dlog(scm)*2.d0*.0389d0
       rpmin=rpmin+alfp*dlog(scm)*2.d0*.0389d0
      else
       rpmin=rpmin+alfp*dlog(scm/sgap)*4.d0*.0389d0
      endif
      rp0=rpmax*rpmin/(rpmax+rpmin)

      call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *,ip,it)
      vvxts=1.d0-(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
      vpl=qglegi(1.d0/xpomr,bbpr,iddp(ip),icz,2)
      vplc=min(vpl
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,7))
      vplc0=min(vplc
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,8))
      vplcpe=min(vplc0
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,10))
      vplcp=min(vplcpe
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,9))

      vvxps=1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
      vtl=qglegi(xpomr*scm,bbtg,iddt(it),2,2)
      vtlc=min(vtl
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,7))
      vtlc0=min(vtlc
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,8))
      vtlcpe=min(vtlc0
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,10))
      vtlcp=min(vtlcpe
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,9))

      sumcp0=0.d0
      sumup=0.d0
      do i=1,ia(1)
       sumup=sumup+vpac(i)
      enddo
      vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
      do i=1,ia(1)-ip+1
       ipp=ia(1)-i+1
       bbp=(xa(ipp,1)+b-xxp)**2+(xa(ipp,2)-yyp)**2
       sumup=sumup-vpac(ipp)
       vpac0(ipp)=min(vpac(ipp)
     * ,qgfani(1.d0/xpomr,bbp,1.d0-vvxs*exp(-sumup)
     * ,1.d0-exp(-sumcp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
       if(ipp.gt.ip)sumcp0=sumcp0+vpac0(ipp)
      enddo
      sumct0=0.d0
      sumut=0.d0
      do i=1,ia(2)
       sumut=sumut+vtac(i)
      enddo
      vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
      do i=1,ia(2)-it+1
       itt=ia(2)-i+1
       bbt=(xb(itt,1)-xxp)**2+(xb(itt,2)-yyp)**2
       sumut=sumut-vtac(itt)
       vtac0(itt)=min(vtac(itt)
     * ,qgfani(xpomr*scm,bbt,1.d0-vvxs*exp(-sumut)
     * ,1.d0-exp(-sumct0),1.d0-exp(-sumut),iddt(itt),2,3))
       if(itt.gt.it)sumct0=sumct0+vtac0(itt)
      enddo
      vvxp0=1.d0-exp(-sumcp0)
      vvxt0=1.d0-exp(-sumct0)

c weights for vertex contributions:
c vv(1): >1 proj. legs and >1 targ. legs
      vv(1)=(max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **(max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvx)**2
     *-2.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))
     **(1.d0-vvxp0)+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))*exp(-vpac(ip))
     **(max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvx)*(1.d0-vvxtl)
     *-2.d0*(max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))*exp(-vtac(it))
     **(1.d0-vvx)*(1.d0-vvxpl)
c vv(2): 0 proj. legs and 0 targ. legs
      vv(2)=((1.d0-exp(-vpac(ip)))**2*(1.d0-vvxpl)
     *+2.d0*(1.d0-exp(-vpac(ip)))*vvxpl)
     **((1.d0-exp(-vtac(it)))**2*(1.d0-vvxtl)
     *+2.d0*(1.d0-exp(-vtac(it)))*vvxtl)*(1.d0-vvx)
c vv(3): 0 proj. legs and >1 targ. legs
      vv(3)=((1.d0-exp(-vpac(ip)))**2*(1.d0-vvxpl)
     *+2.d0*(1.d0-exp(-vpac(ip)))*vvxpl)*(1.d0-vvx)
     **((max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvxtl)
     *-2.d0*(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0
     *-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))*exp(-vtac(it)))
c vv(4): >1 proj. legs and 0 targ. legs
      vv(4)=((max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **(1.d0-vvxpl)
     *-2.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))
     **(1.d0-vvxp0)+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))*exp(-vpac(ip)))
     **((1.d0-exp(-vtac(it)))**2*(1.d0-vvxtl)
     *+2.d0*(1.d0-exp(-vtac(it)))*vvxtl)*(1.d0-vvx)
c vv(5): 0 proj. legs and >1 targ. (handle) legs
      vv(5)=4.d0*(1.d0-exp(-vpac(ip)))*(1.d0-vvx)
     **(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))*exp(-vtac(it))
      if(xpomr*scm.lt.1.1d0*sgap**2)vv(5)=0.d0
c vv(6): >1 proj. (handle) legs and 0 targ. legs
      vv(6)=4.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))*(1.d0-vvxp0)
     *+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))*exp(-vpac(ip))
     **(1.d0-exp(-vtac(it)))*(1.d0-vvx)
      if(xpomr*sgap**2.gt..9d0)vv(6)=0.d0
c vv(7): >1 proj. legs and 1 targ. leg
      vv(7)=(max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **((vtac0(it)+vtlc0)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *-(vtac(it)+vtlc-vtac0(it)-vtlc0)
     **(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(it))
     **(1.d0-vvx)*(1.d0-vvxpl)*(1.d0-vvxt)
     *-2.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))
     **(1.d0-vvxp0)+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))
     **(vtac(it)+vtlc)*exp(-vpac(ip)-2.d0*vtac(it))
     **(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)
c vv(8): 1 proj. leg and >1 targ. legs
      vv(8)=((vpac0(ip)+vplc0)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *-(vpac(ip)+vplc-vpac0(ip)-vplc0)
     **(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ip))
     **(max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxtl)
     *-2.d0*(vpac(ip)+vplc)*exp(-2.d0*vpac(ip)-vtac(it))
     **(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))
     **(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)
c vv(9): 0 proj. legs and 1 targ. leg
      vv(9)=((1.d0-exp(-vpac(ip)))**2*(1.d0-vvxpl)
     *+2.d0*(1.d0-exp(-vpac(ip)))*vvxpl)*(1.d0-vvx)*(1.d0-vvxt)
     **((vtac0(it)+vtlc0)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *-(vtac(it)+vtlc-vtac0(it)-vtlc0)
     **(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(it))
c vv(10): 1 proj. leg and 0 targ. legs
      vv(10)=((vpac0(ip)+vplc0)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *-(vpac(ip)+vplc-vpac0(ip)-vplc0)
     **(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ip))
     **((1.d0-exp(-vtac(it)))**2*(1.d0-vvxtl)
     *+2.d0*(1.d0-exp(-vtac(it)))*vvxtl)*(1.d0-vvx)*(1.d0-vvxp)
c vv(11): 1 cut proj. leg and 1 targ. leg
      vv(11)=2.d0*vplcp*((vtlc0-vtlcpe)
     **exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *-(vtlc-vtlc0)*(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))
     **exp(-2.d0*vpac(ip)-vtac(it))
     **(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)*(1.d0-vvxt)
      if(xpomr*scm.lt.1.1d0*sgap**2)vv(11)=0.d0
c vv(12): 1 proj. leg and 1 cut targ. leg
      vv(12)=2.d0*vtlcp*((vplc0-vplcpe)
     **exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *-(vplc-vplc0)*(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))
     **exp(-2.d0*vtac(it)-vpac(ip))
     **(1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt)**2*(1.d0-vvxtl)
      if(xpomr*sgap**2.gt..9d0)vv(12)=0.d0

      gb0=0.d0
      do i=1,12
       gb0=gb0+max(0.d0,vv(i))/4.d0
      enddo

      if(gb0.le.0.d0)then      !so170712
       if(debug.ge.3)write (moniou,202)
       iret=1
       goto 31
      endif
      if(debug.ge.3)write (moniou,203)gb0

1     continue
      xpomr=(scm/sgap**2)**(-qgran(b10))/sgap   !proposed LC momentum for 3P-vertex
      rp1=(rq(iddp(ip),icz)-alfp*dlog(xpomr))*4.d0*.0389d0
      rp2=(rq(iddt(it),2)+alfp*dlog(xpomr*scm))*4.d0*.0389d0
      rp=rp1*rp2/(rp1+rp2)
      z=qgran(b10)
      phi=pi*qgran(b10)
      b0=dsqrt(-rp*dlog(z))
      bbpr=(bpt*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
      bbtg=(bpt*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2
      call qgbdef(bbpr,bbtg,xa(ip,1)+b,xa(ip,2),xb(it,1),xb(it,2)
     *,xxp,yyp,int(1.5d0+qgran(b10)))   !determine coordinates for the vertex

      call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *,ip,it)
      vvxts=1.d0-(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
      vpl=qglegi(1.d0/xpomr,bbpr,iddp(ip),icz,2)
      vplc=min(vpl
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,7))
      vplc0=min(vplc
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,8))
      vplcpe=min(vplc0
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,10))
      vplcp=min(vplcpe
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,9))

      vvxps=1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
      vtl=qglegi(xpomr*scm,bbtg,iddt(it),2,2)
      vtlc=min(vtl
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,7))
      vtlc0=min(vtlc
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,8))
      vtlcpe=min(vtlc0
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,10))
      vtlcp=min(vtlcpe
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,9))

      sumcp0=0.d0
      sumup=0.d0
      do i=1,ia(1)
       sumup=sumup+vpac(i)
      enddo
      vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
      do i=1,ia(1)-ip+1
       ipp=ia(1)-i+1
       bbp=(xa(ipp,1)+b-xxp)**2+(xa(ipp,2)-yyp)**2
       sumup=sumup-vpac(ipp)
       vpac0(ipp)=min(vpac(ipp)
     * ,qgfani(1.d0/xpomr,bbp,1.d0-vvxs*exp(-sumup)
     * ,1.d0-exp(-sumcp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
       if(ipp.gt.ip)sumcp0=sumcp0+vpac0(ipp)
      enddo
      sumct0=0.d0
      sumut=0.d0
      do i=1,ia(2)
       sumut=sumut+vtac(i)
      enddo
      vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
      do i=1,ia(2)-it+1
       itt=ia(2)-i+1
       bbt=(xb(itt,1)-xxp)**2+(xb(itt,2)-yyp)**2
       sumut=sumut-vtac(itt)
       vtac0(itt)=min(vtac(itt)
     * ,qgfani(xpomr*scm,bbt,1.d0-vvxs*exp(-sumut)
     * ,1.d0-exp(-sumct0),1.d0-exp(-sumut),iddt(itt),2,3))
       if(itt.gt.it)sumct0=sumct0+vtac0(itt)
      enddo
      vvxp0=1.d0-exp(-sumcp0)
      vvxt0=1.d0-exp(-sumct0)

c weights for vertex contributions:
c vv(1): >1 proj. legs and >1 targ. legs
      vv(1)=(max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **(max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvx)**2
     *-2.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))
     **(1.d0-vvxp0)+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))*exp(-vpac(ip))
     **(max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvx)*(1.d0-vvxtl)
     *-2.d0*(max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))*exp(-vtac(it))
     **(1.d0-vvx)*(1.d0-vvxpl)
c vv(2): 0 proj. legs and 0 targ. legs
      vv(2)=((1.d0-exp(-vpac(ip)))**2*(1.d0-vvxpl)
     *+2.d0*(1.d0-exp(-vpac(ip)))*vvxpl)
     **((1.d0-exp(-vtac(it)))**2*(1.d0-vvxtl)
     *+2.d0*(1.d0-exp(-vtac(it)))*vvxtl)*(1.d0-vvx)
c vv(3): 0 proj. legs and >1 targ. legs
      vv(3)=((1.d0-exp(-vpac(ip)))**2*(1.d0-vvxpl)
     *+2.d0*(1.d0-exp(-vpac(ip)))*vvxpl)*(1.d0-vvx)
     **((max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvxtl)
     *-2.d0*(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0
     *-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))*exp(-vtac(it)))
c vv(4): >1 proj. legs and 0 targ. legs
      vv(4)=((max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **(1.d0-vvxpl)
     *-2.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))
     **(1.d0-vvxp0)+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))*exp(-vpac(ip)))
     **((1.d0-exp(-vtac(it)))**2*(1.d0-vvxtl)
     *+2.d0*(1.d0-exp(-vtac(it)))*vvxtl)*(1.d0-vvx)
c vv(5): 0 proj. legs and >1 targ. (handle) legs
      vv(5)=4.d0*(1.d0-exp(-vpac(ip)))*(1.d0-vvx)
     **(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))*exp(-vtac(it))
      if(xpomr*scm.le.sgap**2)vv(5)=0.d0
c vv(6): >1 proj. (handle) legs and 0 targ. legs
      vv(6)=4.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))*(1.d0-vvxp0)
     *+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))*exp(-vpac(ip))
     **(1.d0-exp(-vtac(it)))*(1.d0-vvx)
      if(xpomr*sgap**2.ge.1.d0)vv(6)=0.d0
c vv(7): >1 proj. legs and 1 targ. leg
      vv(7)=(max(0.d0,1.d0-exp(-2.d0*vpac(ip))*(1.d0+2.d0*vpac(ip)))
     *+2.d0*vpac(ip)*exp(-2.d0*vpac(ip))*(1.d0-(1.d0-vvxp)**2))
     **((vtac0(it)+vtlc0)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *-(vtac(it)+vtlc-vtac0(it)-vtlc0)
     **(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(it))
     **(1.d0-vvx)*(1.d0-vvxpl)*(1.d0-vvxt)
     *-2.d0*(max(0.d0,exp(vpac(ip)-vpac0(ip))-1.d0
     *-(vpac(ip)-vpac0(ip)))
     **(1.d0-vvxp0)+(vpac(ip)-vpac0(ip))*(vvxp-vvxp0))
     **(vtac(it)+vtlc)*exp(-vpac(ip)-2.d0*vtac(it))
     **(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)
c vv(8): 1 proj. leg and >1 targ. legs
      vv(8)=((vpac0(ip)+vplc0)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *-(vpac(ip)+vplc-vpac0(ip)-vplc0)
     **(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ip))
     **(max(0.d0,1.d0-exp(-2.d0*vtac(it))*(1.d0+2.d0*vtac(it)))
     *+2.d0*vtac(it)*exp(-2.d0*vtac(it))*(1.d0-(1.d0-vvxt)**2))
     **(1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxtl)
     *-2.d0*(vpac(ip)+vplc)*exp(-2.d0*vpac(ip)-vtac(it))
     **(max(0.d0,exp(vtac(it)-vtac0(it))-1.d0-(vtac(it)-vtac0(it)))
     **(1.d0-vvxt0)+(vtac(it)-vtac0(it))*(vvxt-vvxt0))
     **(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)
c vv(9): 0 proj. legs and 1 targ. leg
      vv(9)=((1.d0-exp(-vpac(ip)))**2*(1.d0-vvxpl)
     *+2.d0*(1.d0-exp(-vpac(ip)))*vvxpl)*(1.d0-vvx)*(1.d0-vvxt)
     **((vtac0(it)+vtlc0)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *-(vtac(it)+vtlc-vtac0(it)-vtlc0)
     **(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(it))
c vv(10): 1 proj. leg and 0 targ. legs
      vv(10)=((vpac0(ip)+vplc0)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *-(vpac(ip)+vplc-vpac0(ip)-vplc0)
     **(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ip))
     **((1.d0-exp(-vtac(it)))**2*(1.d0-vvxtl)
     *+2.d0*(1.d0-exp(-vtac(it)))*vvxtl)*(1.d0-vvx)*(1.d0-vvxp)
c vv(11): 1 cut proj. leg and 1 targ. leg
      vv(11)=2.d0*vplcp*((vtlc0-vtlcpe)
     **exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *-(vtlc-vtlc0)*(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))
     **exp(-2.d0*vpac(ip)-vtac(it))
     **(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)*(1.d0-vvxt)
      if(xpomr*scm.lt.1.1d0*sgap**2)vv(11)=0.d0
c vv(12): 1 proj. leg and 1 cut targ. leg
      vv(12)=2.d0*vtlcp*((vplc0-vplcpe)
     **exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *-(vplc-vplc0)*(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))
     **exp(-2.d0*vtac(it)-vpac(ip))
     **(1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt)**2*(1.d0-vvxtl)
      if(xpomr*sgap**2.gt..9d0)vv(12)=0.d0

      gb=0.d0
      do i=1,12
       vv(i)=max(0.d0,vv(i))
       gb=gb+vv(i)/4.d0
      enddo
      gb=gb/gb0/z*rp/rp0  /max(2.d0,dlog10(scm)-1.d0)  /2.
      if(debug.ge.5)write (moniou,204)xpomr,bbpr,bbtg,gb

      if(qgran(b10).gt.gb)goto 1
      if(debug.ge.3)write (moniou,205)xpomr,bbpr,bbtg,xxp,yyp

      vplcng=min(vplc0
     *,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp,vvxpl,iddp(ip),icz,11))
      vtlcng=min(vtlc0
     *,qgfani(xpomr*scm,bbtg,vvxps,vvxt,vvxtl,iddt(it),2,11))

      sumcpt=0.d0
      sumcp0=0.d0
      sumup=0.d0
      vvxp0l=0.d0
      do i=1,ia(1)
       sumup=sumup+vpac(i)
      enddo
      vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
      do i=1,ia(1)
       ipp=ia(1)-i+1
       bbp=(xa(ipp,1)+b-xxp)**2+(xa(ipp,2)-yyp)**2
       sumup=sumup-vpac(ipp)
       if(ipp.ge.ip)vpact(ipp)=max(vpac(ipp)
     * ,qgfani(1.d0/xpomr,bbp,1.d0-vvxs*exp(-sumup)
     * ,1.d0-exp(-sumcpt),1.d0-exp(-sumup),iddp(ipp),icz,6))
       vpac0(ipp)=min(vpac(ipp)
     * ,qgfani(1.d0/xpomr,bbp,1.d0-vvxs*exp(-sumup)
     * ,1.d0-exp(-sumcp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
       if(ipp.gt.ip)then
        sumcpt=sumcpt+vpact(ipp)
       elseif(ipp.lt.ip)then
        vvxp0l=vvxp0l+vpac0(ipp)
       endif
       sumcp0=sumcp0+vpac0(ipp)
      enddo
      sumctt=0.d0
      sumct0=0.d0
      sumut=0.d0
      vvxt0l=0.d0
      do i=1,ia(2)
       sumut=sumut+vtac(i)
      enddo
      vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
      do i=1,ia(2)
       itt=ia(2)-i+1
       bbt=(xb(itt,1)-xxp)**2+(xb(itt,2)-yyp)**2
       sumut=sumut-vtac(itt)
       if(itt.ge.it)vtact(itt)=max(vtac(itt)
     * ,qgfani(xpomr*scm,bbt,1.d0-vvxs*exp(-sumut)
     * ,1.d0-exp(-sumctt),1.d0-exp(-sumut),iddt(itt),2,6))
       vtac0(itt)=min(vtac(itt)
     * ,qgfani(xpomr*scm,bbt,1.d0-vvxs*exp(-sumut)
     * ,1.d0-exp(-sumct0),1.d0-exp(-sumut),iddt(itt),2,3))
       if(itt.gt.it)then
        sumctt=sumctt+vtact(itt)
       elseif(itt.lt.it)then
        vvxt0l=vvxt0l+vtac0(itt)
       endif
       sumct0=sumct0+vtac0(itt)
      enddo
      vvxpt=1.d0-exp(-sumcpt)
      vvxtt=1.d0-exp(-sumctt)
      vvxp0l=1.d0-exp(-vvxp0l)
      vvxt0l=1.d0-exp(-vvxt0l)

      vvt=0.d0
      do i=1,12
       vvt=vvt+vv(i)
      enddo
      if(.not.(vvt.gt.0.d0))stop'vvt<0'

      aks=qgran(b10)*vvt
      do jt=1,12
       aks=aks-vv(jt)
       if(aks.lt.0.d0)goto 2
      enddo
      stop'jt>12!'

2     continue
      if(xpomr*scm.gt.sgap**2)then
       wzgp=-2.d0*(1.d0-exp(-2.d0*vpac(ip)))*(1.d0-vvxpl)**2
     * *(max(0.d0,1.d0-exp(-vtact(it))*(1.d0+vtact(it)))*(1.d0-vvxtt)
     * -max(0.d0,1.d0-exp(-vtac0(it))*(1.d0+vtac0(it)))*(1.d0-vvxt0)
     * +vtact(it)*exp(-vtact(it))*(1.d0-vvxtt
     * -exp(vtact(it)-vtac(it))*(1.d0-vvxtl)*(1.d0-vvxt))
     * -vtac0(it)*exp(-vtac0(it))*(1.d0-vvxt0
     * -exp(vtac0(it)-vtac(it))*(1.d0-vvxtl)*(1.d0-vvxt)))
      else
       wzgp=0.d0
      endif
      if(xpomr*sgap**2.lt.1.d0)then
       wzgt=-2.d0*(1.d0-exp(-2.d0*vtac(it)))*(1.d0-vvxtl)**2
     * *(max(0.d0,1.d0-exp(-vpact(ip))*(1.d0+vpact(ip)))*(1.d0-vvxpt)
     * -max(0.d0,1.d0-exp(-vpac0(ip))*(1.d0+vpac0(ip)))*(1.d0-vvxp0)
     * +vpact(ip)*exp(-vpact(ip))*(1.d0-vvxpt
     * -exp(vpact(ip)-vpac(ip))*(1.d0-vvxpl)*(1.d0-vvxp))
     * -vpac0(ip)*exp(-vpac0(ip))*(1.d0-vvxp0
     * -exp(vpac0(ip)-vpac(ip))*(1.d0-vvxpl)*(1.d0-vvxp)))
      else
       wzgt=0.d0
      endif

      nppr0=0
      nptg0=0
      npprh0=0
      nptgh0=0
      wgpr0=0.d0
      wgtg0=0.d0
      if(jt.eq.1.or.jt.eq.4.or.jt.eq.7)then
       ntry=0
3      ntry=ntry+1
       npprh0=0
       if(ip.eq.ia(1).or.ntry.gt.100)then
        nppr0=npgen(2.d0*vpac(ip),2,20)
        do i=1,nppr0
         if(qgran(b10).le.vpac0(ip)/vpac(ip).or.xpomr*sgap**2.ge.1.d0)
     *   then
          itypr0(i)=0
         else
          npprh0=npprh0+1
          itypr0(i)=1
         endif
         ippr0(i)=ip
        enddo
        wh=(vpac(ip)/vpac0(ip)-1.d0)/nppr0
       else
        nppr0=npgen(2.d0*vpac(ip),1,20)
        do i=1,nppr0
         if(qgran(b10).le.vpac0(ip)/vpac(ip).or.xpomr*sgap**2.ge.1.d0)
     *   then
          itypr0(i)=0
         else
          npprh0=npprh0+1
          itypr0(i)=1
         endif
         ippr0(i)=ip
        enddo
        wh=(vpac(ip)/vpac0(ip)-1.d0)/nppr0
        do ipp=ip+1,ia(1)
         ninc=npgen(2.d0*vpac(ipp),0,20)
         if(ninc.ne.0)then
          nppr0=nppr0+ninc
          nh0=npprh0
          if(nppr0.gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nppr0-ninc+1,nppr0
           if(qgran(b10).le.vpac0(ipp)/vpac(ipp)
     *     .or.xpomr*sgap**2.ge.1.d0)then
            itypr0(i)=0
           else
            npprh0=npprh0+1
            itypr0(i)=1
           endif
           ippr0(i)=ipp
          enddo
          if(ninc.gt.npprh0-nh0)wh=(vpac(ipp)/vpac0(ipp)-1.d0)/ninc
         endif
        enddo
        if(nppr0.eq.1)goto 3
       endif
       if(nppr0.le.npprh0+1)then
        if(jt.ne.7)then
         wh0=1.d0-exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     *   /(1.d0-vvxp)/(1.d0-vvxpl)
        else
         wh0=1.d0-exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     *   /(1.d0-vvxp)/(1.d0-vvxpl)
     *   *(vtac(it)+vtlc)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *   /((vtac0(it)+vtlc0)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     *   -(vtac(it)+vtlc-vtac0(it)-vtlc0)
     *   *(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))
        endif
        if(nppr0.eq.npprh0.and.wh0.lt.0.d0
     *  .or.nppr0.eq.npprh0+1.and.qgran(b10).gt.1.d0+wh*wh0)goto 3
       endif
      endif

      if(jt.eq.1.or.jt.eq.3.or.jt.eq.8)then
       ntry=0
4      ntry=ntry+1
       nptgh0=0
       if(it.eq.ia(2).or.ntry.gt.100)then
        nptg0=npgen(2.d0*vtac(it),2,20)
        do i=1,nptg0
         if(qgran(b10).le.vtac0(it)/vtac(it).or.xpomr*scm.le.sgap**2)
     *   then
          itytg0(i)=0
         else
          nptgh0=nptgh0+1
          itytg0(i)=1
         endif
         iptg0(i)=it
        enddo
        wh=(vtac(it)/vtac0(it)-1.d0)/nptg0
       else
        nptg0=npgen(2.d0*vtac(it),1,20)
        do i=1,nptg0
         if(qgran(b10).le.vtac0(it)/vtac(it).or.xpomr*scm.le.sgap**2)
     *   then
          itytg0(i)=0
         else
          nptgh0=nptgh0+1
          itytg0(i)=1
         endif
         iptg0(i)=it
        enddo
        wh=(vtac(it)/vtac0(it)-1.d0)/nptg0
        do itt=it+1,ia(2)
         ninc=npgen(2.d0*vtac(itt),0,20)
         if(ninc.ne.0)then
          nptg0=nptg0+ninc
          nh0=nptgh0
          if(nptg0.gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nptg0-ninc+1,nptg0
           if(qgran(b10).le.vtac0(itt)/vtac(itt)
     *     .or.xpomr*scm.le.sgap**2) then
            itytg0(i)=0
           else
            nptgh0=nptgh0+1
            itytg0(i)=1
           endif
           iptg0(i)=itt
          enddo
          if(ninc.gt.nptgh0-nh0)wh=(vtac(itt)/vtac0(itt)-1.d0)/ninc
         endif
        enddo
        if(nptg0.eq.1)goto 4
       endif
       if(nptg0.le.nptgh0+1)then
        if(jt.ne.8)then
         wh0=1.d0-exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))
     *   /(1.d0-vvxt)/(1.d0-vvxtl)
        else
         wh0=1.d0-exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))
     *   /(1.d0-vvxt)/(1.d0-vvxtl)
     *   *(vpac(ip)+vplc)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *   /((vpac0(ip)+vplc0)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     *   -(vpac(ip)+vplc-vpac0(ip)-vplc0)
     *   *(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))
        endif
        if(nptg0.eq.nptgh0.and.wh0.lt.0.d0
     *  .or.nptg0.eq.nptgh0+1.and.qgran(b10).gt.1.d0+wh*wh0)goto 4
       endif
      endif

      if(jt.eq.6)then
       ntry=0
5      ntry=ntry+1
       if(ip.eq.ia(1).or.ntry.gt.100)then
        nppr0=npgen(vpac(ip)-vpac0(ip),2,20)
        do i=1,nppr0
         itypr0(i)=1
         ippr0(i)=ip
        enddo
       else
        nppr0=npgen(vpac(ip)-vpac0(ip),1,20)
        do i=1,nppr0
         itypr0(i)=1
         ippr0(i)=ip
        enddo
        do ipp=ip+1,ia(1)
         ninc=npgen(vpac(ipp)-vpac0(ipp),0,20)
         if(ninc.ne.0)then
          nppr0=nppr0+ninc
          if(nppr0.gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nppr0-ninc+1,nppr0
           itypr0(i)=1
           ippr0(i)=ipp
          enddo
         endif
        enddo
        if(nppr0.eq.1)goto 5
       endif
      endif

      if(jt.eq.5)then
       ntry=0
6      ntry=ntry+1
       if(it.eq.ia(2).or.ntry.gt.100)then
        nptg0=npgen(vtac(it)-vtac0(it),2,20)
        do i=1,nptg0
         itytg0(i)=1
         iptg0(i)=it
        enddo
       else
        nptg0=npgen(vtac(it)-vtac0(it),1,20)
        do i=1,nptg0
         itytg0(i)=1
         iptg0(i)=it
        enddo
        do itt=it+1,ia(2)
         ninc=npgen(vtac(itt)-vtac0(itt),0,20)
         if(ninc.ne.0)then
          nptg0=nptg0+ninc
          if(nptg0.gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nptg0-ninc+1,nptg0
           itytg0(i)=1
           iptg0(i)=itt
          enddo
         endif
        enddo
        if(nptg0.eq.1)goto 6
       endif
      endif

      gbt=1.d0
      if((jt.eq.1.and.nptgh0.lt.nptg0.or.jt.eq.4)
     *.and.npprh0.eq.nppr0)then
       gbt=1.d0-exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     * /(1.d0-vvxp)/(1.d0-vvxpl)
      elseif((jt.eq.1.and.npprh0.lt.nppr0.or.jt.eq.3)
     *.and.nptgh0.eq.nptg0)then
       gbt=1.d0-exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))
     * /(1.d0-vvxt)/(1.d0-vvxtl)
      elseif(jt.eq.1.and.nptgh0.eq.nptg0.and.npprh0.eq.nppr0)then
       gbt=1.d0-exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     * /(1.d0-vvxp)/(1.d0-vvxpl)
     * -exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))/(1.d0-vvxt)/(1.d0-vvxtl)
      elseif(jt.eq.7.and.npprh0.eq.nppr0)then
       gbt=1.d0-exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     * /(1.d0-vvxp)/(1.d0-vvxpl)
     * *(vtac(it)+vtlc)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     * /((vtac0(it)+vtlc0)*exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     * -(vtac(it)+vtlc-vtac0(it)-vtlc0)
     * *(1.d0-exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)))
      elseif(jt.eq.8.and.nptgh0.eq.nptg0)then
       gbt=1.d0-exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))
     * /(1.d0-vvxt)/(1.d0-vvxtl)
     * *(vpac(ip)+vplc)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     * /((vpac0(ip)+vplc0)*exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     * -(vpac(ip)+vplc-vpac0(ip)-vplc0)
     * *(1.d0-exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)))
      endif
      if(qgran(b10).gt.gbt)goto 2

c less important part of 'zigzag' cuts - commented out (sub-per cent effect)
c      if((jt.eq.1.or.jt.eq.8)
c     *  .and.qgran(b10).lt.max(0.d0,wzgp/(vv(1)+vv(8))))nppr0=0
c      if((jt.eq.1.or.jt.eq.7)
c     *  .and.qgran(b10).lt.max(0.d0,wzgt/(vv(1)+vv(7))))nptg0=0

      if(jt.eq.7.or.jt.eq.9.or.jt.eq.11.or.jt.eq.12)then
       nptg0=1
       iptg0(1)=it
      endif
      if(jt.eq.8.or.jt.eq.10.or.jt.eq.11.or.jt.eq.12)then
       nppr0=1
       ippr0(1)=ip
      endif

      if(jt.eq.8.and.nptgh0.lt.nptg0.or.jt.eq.10)then !'fan' from cut vertex
       vpacng=min(vpac0(ip)
     * ,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp0,vvxpl,iddp(ip),icz,4))

       factor=exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
       wng=(vpacng+vplcng)*factor
       wgap=max(0.d0,(vpac0(ip)+vplc0)*factor
     * -(vpac(ip)+vplc-vpac0(ip)-vplc0)*(1.d0-factor)-wng)
       if(qgran(b10).ge.wgap/(wgap+wng).or.xpomr*sgap**2.gt..9d0)then
        if(qgran(b10).lt.vpacng/(vpacng+vplcng)
     *  .and.xpomr*sgap**2.lt..9d0)then
         itypr0(1)=2            !cut 'fan' (no gap at the end)
        else
         itypr0(1)=4            !cut 'leg' (no gap at the end)
        endif
       else
        wfg=max(0.d0,(vpac0(ip)-vpacng)*factor
     *         -(vpac(ip)-vpac0(ip))*(1.d0-factor))
        wlg=max(0.d0,(vplc0-vplcng)*factor-(vplc-vplc0)*(1.d0-factor))
        if(qgran(b10).lt.wfg/(wfg+wlg))then
         itypr0(1)=3            !cut 'fan' (gap at the end)
        else
         itypr0(1)=5            !cut 'leg' (gap at the end)
        endif
        wgpr0=(1.d0-factor)/factor
       endif

      elseif(jt.eq.8.and.nptgh0.eq.nptg0)then !'fan' from cut/uncut vertex
       vpacng=min(vpac0(ip)
     * ,qgfani(1.d0/xpomr,bbpr,vvxts,vvxp0,vvxpl,iddp(ip),icz,4))

       factor=exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
       wng=(vpacng+vplcng)*factor*(1.d0-exp(vtac(it)
     * +(1.d0-nptg0)*dlog(2.d0))/(1.d0-vvxt)/(1.d0-vvxtl))
       wgap=max(0.d0,(vpac0(ip)+vplc0)*factor
     * -(vpac(ip)+vplc-vpac0(ip)-vplc0)*(1.d0-factor)
     * -exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))/(1.d0-vvxt)/(1.d0-vvxtl)
     * *(vpac(ip)+vplc)*factor-wng)
       if(qgran(b10).ge.wgap/(wgap+wng).or.xpomr*sgap**2.gt..9d0)then
        if(qgran(b10).lt.vpacng/(vpacng+vplcng)
     *  .and.xpomr*sgap**2.lt..9d0)then
         itypr0(1)=2            !cut 'fan' (no gap at the end)
        else
         itypr0(1)=4            !cut 'leg' (no gap at the end)
        endif
       else
        wfg=max(0.d0,(vpac0(ip)-vpacng)*factor
     *         -(vpac(ip)-vpac0(ip))*(1.d0-factor)
     *  -exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))
     *  /(1.d0-vvxt)/(1.d0-vvxtl)*(vpac(ip)-vpacng)*factor)
        wlg=max(0.d0,(vplc0-vplcng)*factor-(vplc-vplc0)*(1.d0-factor)
     *  -exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))
     *  /(1.d0-vvxt)/(1.d0-vvxtl)*(vplc-vplcng)*factor)
        if(qgran(b10).lt.wfg/(wfg+wlg))then
         itypr0(1)=3            !cut 'fan' (gap at the end)
        else
         itypr0(1)=5            !cut 'leg' (gap at the end)
        endif
        wgpr0=1.d0/factor/(1.d0-exp(vtac(it)+(1.d0-nptg0)*dlog(2.d0))
     *  /(1.d0-vvxt)/(1.d0-vvxtl))-1.d0
       endif

      elseif(jt.eq.11)then
       itypr0(1)=6
      elseif(jt.eq.12)then
       factor=exp(-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
       wng=max(0.d0,vplcng-vplcpe)*factor
     * /((vplc0-vplcpe)*factor-(vplc-vplc0)*(1.d0-factor))
       if(qgran(b10).le.wng)then
        itypr0(1)=7            !cut 'leg' (>1 cut Poms at the end)
       else
        itypr0(1)=5            !cut 'leg' (gap at the end)
        wgpr0=(1.d0-factor)/factor
       endif
      endif

      if(jt.eq.7.and.npprh0.lt.nppr0.or.jt.eq.9)then !'fan' from cut vertex
       vtacng=min(vtac0(it)
     * ,qgfani(xpomr*scm,bbtg,vvxps,vvxt0,vvxtl,iddt(it),2,4))

       factor=exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
       wng=(vtacng+vtlcng)*factor
       wgap=max(0.d0,(vtac0(it)+vtlc0)*factor
     * -(vtac(it)+vtlc-vtac0(it)-vtlc0)*(1.d0-factor)-wng)
       if(qgran(b10).ge.wgap/(wgap+wng)
     * .or.xpomr*scm.lt.1.1d0*sgap**2)then
        if(qgran(b10).lt.vtacng/(vtacng+vtlcng)
     *  .and.xpomr*scm.gt.1.1d0*sgap**2)then
         itytg0(1)=2            !cut 'fan' (no gap at the end)
        else
         itytg0(1)=4            !cut 'leg' (no gap at the end)
        endif
       else
        wfg=max(0.d0,(vtac0(it)-vtacng)*factor
     *         -(vtac(it)-vtac0(it))*(1.d0-factor))
        wlg=max(0.d0,(vtlc0-vtlcng)*factor-(vtlc-vtlc0)*(1.d0-factor))
        if(qgran(b10).lt.wfg/(wfg+wlg))then
         itytg0(1)=3            !cut 'fan' (gap at the end)
        else
         itytg0(1)=5            !cut 'leg' (gap at the end)
        endif
        wgtg0=(1.d0-factor)/factor
       endif

      elseif(jt.eq.7.and.npprh0.eq.nppr0)then !'fan' from cut/uncut vertex
       vtacng=min(vtac0(it)
     * ,qgfani(xpomr*scm,bbtg,vvxps,vvxt0,vvxtl,iddt(it),2,4))

       factor=exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
       wng=(vtacng+vtlcng)*factor*(1.d0-exp(vpac(ip)
     * +(1.d0-nppr0)*dlog(2.d0))/(1.d0-vvxp)/(1.d0-vvxpl))
       wgap=max(0.d0,(vtac0(it)+vtlc0)*factor
     * -(vtac(it)+vtlc-vtac0(it)-vtlc0)*(1.d0-factor)
     * -exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))/(1.d0-vvxp)/(1.d0-vvxpl)
     * *(vtac(it)+vtlc)*factor-wng)
       if(qgran(b10).ge.wgap/(wgap+wng)
     * .or.xpomr*scm.lt.1.1d0*sgap**2)then
        if(qgran(b10).lt.vtacng/(vtacng+vtlcng)
     *  .and.xpomr*scm.gt.1.1d0*sgap**2)then
         itytg0(1)=2            !cut 'fan' (no gap at the end)
        else
         itytg0(1)=4            !cut 'leg' (no gap at the end)
        endif
       else
        wfg=max(0.d0,(vtac0(it)-vtacng)*factor
     *         -(vtac(it)-vtac0(it))*(1.d0-factor)
     *  -exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     *  /(1.d0-vvxp)/(1.d0-vvxpl)*(vtac(it)-vtacng)*factor)
        wlg=max(0.d0,(vtlc0-vtlcng)*factor-(vtlc-vtlc0)*(1.d0-factor)
     *  -exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     *  /(1.d0-vvxp)/(1.d0-vvxpl)*(vtlc-vtlcng)*factor)
        if(qgran(b10).lt.wfg/(wfg+wlg))then
         itytg0(1)=3            !cut 'fan' (gap at the end)
        else
         itytg0(1)=5            !cut 'leg' (gap at the end)
        endif
        wgtg0=1.d0/factor/(1.d0-exp(vpac(ip)+(1.d0-nppr0)*dlog(2.d0))
     *  /(1.d0-vvxp)/(1.d0-vvxpl))-1.d0
       endif

      elseif(jt.eq.12)then
       itytg0(1)=6
      elseif(jt.eq.11)then
       factor=exp(-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
       wng=max(0.d0,vtlcng-vtlcpe)*factor
     * /((vtlc0-vtlcpe)*factor-(vtlc-vtlc0)*(1.d0-factor))
       if(qgran(b10).le.wng)then
        itytg0(1)=7            !cut 'leg' (>1 cut Poms at the end)
       else
        itytg0(1)=5            !cut 'leg' (gap at the end)
        wgtg0=(1.d0-factor)/factor
       endif
      endif
      if(debug.ge.3)write (moniou,206)nppr0,nptg0

      nppr=0
      nptg=0
      npin=0

      if(nppr0.eq.1.and.itypr0(1).eq.6)then     !single cut Pomeron
       nppr=1
       xpompi(nppr)=xpomr
       vvxpi(nppr)=1.d0-(1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt)
     * *exp(-vtac(it))
       ipompi(nppr)=ip
       bpompi(nppr)=bbpr
       if(debug.ge.4)write (moniou,209)nppr,ip,bbpr,xpompi(nppr)
     * ,vvxpi(nppr)
       nppr0=0
      endif
      if(nptg0.eq.1.and.itytg0(1).eq.6)then     !single cut Pomeron
       nptg=1
       xpomti(nptg)=xpomr
       vvxti(nptg)=1.d0-(1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt)
     * *exp(-vpac(ip))
       ipomti(nptg)=it
       bpomti(nptg)=bbtg
       if(debug.ge.4)write (moniou,217)nptg,it,bbtg,xpomti(nptg)
     * ,vvxti(nptg)
       nptg0=0
      endif

      vvxps=vvxp
      vvxpls=vvxpl
      vvxp0s=vvxp0
      if(nppr0.ne.0)then
       i=0
7      i=i+1
       ityp=itypr0(i)
       if(ityp.eq.0.or.ityp.eq.2.or.ityp.eq.4)then
        ipp=ippr0(i)
        bbp=(xa(ipp,1)+b-xxp)**2+(xa(ipp,2)-yyp)**2
        vvxp=0.d0
        vvxpl=0.d0
        vvxp0=0.d0
        if(ia(1).gt.1)then
         do l=1,ia(1)
          if(l.lt.ipp)then
           vvxpl=vvxpl+vpac(l)
          elseif(l.gt.ipp)then
           vvxp=vvxp+vpac(l)
           vvxp0=vvxp0+vpac0(l)
          endif
         enddo
        endif
        vvxp=1.d0-exp(-vvxp)
        vvxpl=1.d0-exp(-vvxpl)
        vvxp0=1.d0-exp(-vvxp0)
        vvxts=1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*(1.d0-vvxpl)*exp(-vtac(it))
        if(ityp.ne.4)then
         vpacng=min(vpac0(ipp)
     *   ,qgfani(1.d0/xpomr,bbp,vvxts,vvxp0,vvxpl,iddp(ipp),icz,4))
         vpacpe=min(vpacng
     *   ,qgfani(1.d0/xpomr,bbp,vvxts,vvxp0,vvxpl,iddp(ipp),icz,5))
         vplcp=min(vpacpe
     *   ,qgfani(1.d0/xpomr,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,9))
        else
         vplcng=min(vpac0(ipp)
     *   ,qgfani(1.d0/xpomr,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,11))
         vplcpe=min(vplcng
     *   ,qgfani(1.d0/xpomr,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,10))
         vplcp=min(vplcpe
     *   ,qgfani(1.d0/xpomr,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,9))
        endif

        if(ityp.eq.0)then
         aks=qgran(b10)*vpac0(ipp)
         if(aks.le.vplcp.or.xpomr*sgap**2.gt..9d0)then
          itypr0(i)=6        !single cut Pomeron
         elseif(aks.lt.vpacpe)then
          itypr0(i)=-1       !'fan' (cut Pomeron end)
         elseif(aks.lt.vpacng)then
          itypr0(i)=2        !'fan' (>1 cut Poms at the end)
         endif
        elseif(ityp.eq.2)then
         aks=qgran(b10)*vpacng
         if(aks.le.vplcp.or.xpomr*sgap**2.gt..9d0)then
          itypr0(i)=6        !single cut Pomeron
         elseif(aks.lt.vpacpe)then
          itypr0(i)=-1       !'fan' (cut Pomeron end)
         endif
        elseif(ityp.eq.4)then
         aks=qgran(b10)*vplcng
         if(aks.le.vplcp.or.xpomr*sgap**2.gt..9d0)then
          itypr0(i)=6        !single cut Pomeron
         elseif(aks.gt.vplcpe.or.xpomr*sgap**3.gt..9d0)then
          itypr0(i)=7        !'leg' (>1 cut Poms at the end)
         endif
        endif

        if(itypr0(i).eq.6)then        !single cut Pomeron
         nppr=nppr+1
         xpompi(nppr)=xpomr
         vvxpi(nppr)=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*(1.d0-vvxt)
     *   *(1.d0-vvxtl)*exp(-vtac(it))
         ipompi(nppr)=ipp
         bpompi(nppr)=bbp
         if(debug.ge.4)write (moniou,209)nppr,ipp,bbp,xpompi(nppr)
     *   ,vvxpi(nppr)
         nppr0=nppr0-1
         if(nppr0.ge.i)then
          do l=i,nppr0
           ippr0(l)=ippr0(l+1)
           itypr0(l)=itypr0(l+1)
          enddo
         endif
         i=i-1
        endif
       endif
       if(i.lt.nppr0)goto 7
      endif

      vvxp=vvxps
      vvxpl=vvxpls
      vvxp0=vvxp0s
      vvxts=vvxt
      vvxtls=vvxtl
      vvxt0s=vvxt0
      if(nptg0.ne.0)then
       i=0
8      i=i+1
       ityt=itytg0(i)
       if(ityt.eq.0.or.ityt.eq.2.or.ityt.eq.4)then
        itt=iptg0(i)
        bbt=(xb(itt,1)-xxp)**2+(xb(itt,2)-yyp)**2
        vvxt=0.d0
        vvxtl=0.d0
        vvxt0=0.d0
        if(ia(2).gt.1)then
         do l=1,ia(2)
          if(l.lt.itt)then
           vvxtl=vvxtl+vtac(l)
          elseif(l.gt.itt)then
           vvxt=vvxt+vtac(l)
           vvxt0=vvxt0+vtac0(l)
          endif
         enddo
        endif
        vvxt=1.d0-exp(-vvxt)
        vvxtl=1.d0-exp(-vvxtl)
        vvxt0=1.d0-exp(-vvxt0)
        vvxps=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*(1.d0-vvxtl)*exp(-vpac(ip))
        if(ityt.ne.4)then
         vtacng=min(vtac0(itt)
     *   ,qgfani(xpomr*scm,bbt,vvxps,vvxt0,vvxtl,iddt(itt),2,4))
         vtacpe=min(vtacng
     *   ,qgfani(xpomr*scm,bbt,vvxps,vvxt0,vvxtl,iddt(itt),2,5))
         vtlcp=min(vtacpe
     *   ,qgfani(xpomr*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,9))
        else
         vtlcng=min(vtac0(itt)
     *   ,qgfani(xpomr*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,11))
         vtlcpe=min(vtlcng
     *   ,qgfani(xpomr*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,10))
         vtlcp=min(vtlcpe
     *   ,qgfani(xpomr*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,9))
        endif

        if(ityt.eq.0)then
         aks=qgran(b10)*vtac0(itt)
         if(aks.le.vtlcp.or.xpomr*scm.lt.1.1d0*sgap**2)then
          itytg0(i)=6        !single cut Pomeron
         elseif(aks.lt.vtacpe)then
          itytg0(i)=-1       !'fan' (cut Pomeron end)
         elseif(aks.lt.vtacng)then
          itytg0(i)=2        !'fan' (>1 cut Poms at the end)
         endif
        elseif(ityt.eq.2)then
         aks=qgran(b10)*vtacng
         if(aks.le.vtlcp.or.xpomr*scm.lt.1.1d0*sgap**2)then
          itytg0(i)=6        !single cut Pomeron
         elseif(aks.lt.vtacpe)then
          itytg0(i)=-1       !'fan' (cut Pomeron end)
         endif
        elseif(ityt.eq.4)then
         aks=qgran(b10)*vtlcng
         if(aks.le.vtlcp.or.xpomr*scm.lt.1.1d0*sgap**2)then
          itytg0(i)=6
         elseif(aks.gt.vtlcpe.or.xpomr*scm.lt.1.1d0*sgap**3)then
          itytg0(i)=7        !'leg' (>1 cut Poms at the end)
         endif
        endif

        if(itytg0(i).eq.6)then        !single cut Pomeron
         nptg=nptg+1
         xpomti(nptg)=xpomr
         vvxti(nptg)=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*(1.d0-vvxt)
     *   *(1.d0-vvxtl)*exp(-vpac(ip))
         ipomti(nptg)=itt
         bpomti(nptg)=bbt
         if(debug.ge.4)write (moniou,217)nptg,itt,bbt,xpomti(nptg)
     *   ,vvxti(nptg)
         nptg0=nptg0-1
         if(nptg0.ge.i)then
          do l=i,nptg0
           iptg0(l)=iptg0(l+1)
           itytg0(l)=itytg0(l+1)
          enddo
         endif
         i=i-1
        endif
       endif
       if(i.lt.nptg0)goto 8
      endif
      vvxt=vvxts
      vvxtl=vvxtls
      vvxt0=vvxt0s

      if((jt-1)*(jt-4)*(jt-7).eq.0.and.xpomr*sgap**2.lt..9d0)then
       vvxts=1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
       vvxt0s=1.d0-(1.d0-vvxt0)*(1.d0-vvxt0l)*exp(-vtac0(it))
       vvxs=((1.d0-vvxp)*(1.d0-vvxpl))**2*exp(-2.d0*vpac(ip))
       vvx0s=((1.d0-vvxp0)*(1.d0-vvxp0l))**2*exp(-2.d0*vpac0(ip))

       wzzp=2.d0*qgrevi(1.d0/xpomr,bbpr,vvxt0s,vvxts
     * ,vvxpt,vvxp0,vvxpl,iddp(ip),icz)
     * *((1.d0-exp(-vtact(it)))*(1.d0-vvxtt)*(1.d0-vvxs)
     * +vvxs*(max(0.d0,1.d0-exp(-vtact(it))*(1.d0+vtact(it)))
     * *(1.d0-vvxtt)
     * -max(0.d0,1.d0-exp(-vtac0(it))*(1.d0+vtac0(it)))*(1.d0-vvxt0))
     * +vtact(it)*exp(-vtact(it))*((1.d0-vvxtt)*vvxs
     * -exp(vtact(it)-vtac0(it))*(1.d0-vvxt0)*(1.d0-vvxt0l)*vvx0s)
     * -vtac0(it)*exp(-vtac0(it))*(1.d0-vvxt0)
     * *(vvxs-vvx0s+vvxt0l*vvx0s))
       wzzp=max(0.d0,wzzp)
       nzzp=npgen(wzzp/(vv(1)+vv(4)+vv(7)),0,50)
      else
       nzzp=0
      endif

      if((jt-1)*(jt-3)*(jt-8).eq.0.and.xpomr*scm.gt.1.1d0*sgap**2)then
       vvxps=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
       vvxp0s=1.d0-(1.d0-vvxp0)*(1.d0-vvxp0l)*exp(-vpac0(ip))
       vvxs=((1.d0-vvxt)*(1.d0-vvxtl))**2*exp(-2.d0*vtac(it))
       vvx0s=((1.d0-vvxt0)*(1.d0-vvxt0l))**2*exp(-2.d0*vtac0(it))
       wzzt=2.d0*qgrevi(xpomr*scm,bbtg,vvxp0s,vvxps
     * ,vvxtt,vvxt0,vvxtl,iddt(it),2)
     * *((1.d0-exp(-vpact(ip)))*(1.d0-vvxpt)*(1.d0-vvxs)
     * +vvxs*(max(0.d0,1.d0-exp(-vpact(ip))*(1.d0+vpact(ip)))
     * *(1.d0-vvxpt)
     * -max(0.d0,1.d0-exp(-vpac0(ip))*(1.d0+vpac0(ip)))*(1.d0-vvxp0))
     * +vpact(ip)*exp(-vpact(ip))*((1.d0-vvxpt)*vvxs
     * -exp(vpact(ip)-vpac0(ip))*(1.d0-vvxp0)*(1.d0-vvxp0l)*vvx0s)
     * -vpac0(ip)*exp(-vpac0(ip))*(1.d0-vvxp0)
     * *(vvxs-vvx0s+vvxp0l*vvx0s))
       wzzt=max(0.d0,wzzt)
       nzzt=npgen(wzzt/(vv(1)+vv(3)+vv(8)),0,50)
      else
       nzzt=0
      endif

      if(nzzp.ne.0)then
       bpm=(xa(ip,1)+b-xxp)**2+(xa(ip,2)-yyp)**2
       xpomr0=min(dsqrt(xpomr),1.d0/sgap)
       xpomr0=max(xpomr0,xpomr*sgap)
       rp1=(rq(iddp(ip),icz)-alfp*dlog(xpomr0))*4.d0*.0389d0
       rp2=alfp*dlog(xpomr0/xpomr)*4.d0*.0389d0
       rp0=rp1*rp2/(rp1+rp2)
       bbp=bpm*(rp1/(rp1+rp2))**2
       bbi=bpm*(rp2/(rp1+rp2))**2
       call qgbdef(bbp,bbi,xa(ip,1)+b,xa(ip,2),xxp,yyp,xxp0,yyp0,1)
       call qgfdf(xxp0,yyp0,xpomr0,vpac,vtac,vvx,vvxp,vvxt
     * ,vvxpl,vvxtl,ip,it)

       sumcp0=0.d0
       sumcpt=0.d0
       sumup=0.d0
       vvxp0=0.d0
       vvxp0l=0.d0
       do i=1,ia(1)
        sumup=sumup+vpac(i)
       enddo
       vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
       do i=1,ia(1)
        ipp=ia(1)-i+1
        bbpi=(xa(ipp,1)+b-xxp0)**2+(xa(ipp,2)-yyp0)**2
        sumup=sumup-vpac(ipp)
        vpac0(ipp)=min(vpac(ipp)
     *  ,qgfani(1.d0/xpomr0,bbpi,1.d0-vvxs*exp(-sumup)
     *  ,1.d0-exp(-sumcp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
        if(ipp.ge.ip)vpact(ipp)=max(vpac(ipp)
     *  ,qgfani(1.d0/xpomr0,bbpi,1.d0-vvxs*exp(-sumup)
     *  ,1.d0-exp(-sumcpt),1.d0-exp(-sumup),iddp(ipp),icz,6))
        if(ipp.gt.ip)then
         vvxp0=vvxp0+vpac0(ipp)
         sumcpt=sumcpt+vpact(ipp)
        elseif(ipp.lt.ip)then
         vvxp0l=vvxp0l+vpac0(ipp)
        endif
        sumcp0=sumcp0+vpac0(ipp)
       enddo
       vvxpt=1.d0-exp(-sumcpt)
       vvxp0=1.d0-exp(-vvxp0)
       vvxp0l=1.d0-exp(-vvxp0l)

       sumut=0.d0
       sumct0=0.d0
       vvxt0=0.d0
       vvxt0l=0.d0
       do i=1,ia(2)
        sumut=sumut+vtac(i)
       enddo
       vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
       do i=1,ia(2)
        itt=ia(2)-i+1
        bbti=(xb(itt,1)-xxp0)**2+(xb(itt,2)-yyp0)**2
        sumut=sumut-vtac(itt)
        vtac0(itt)=min(vtac(itt)
     *  ,qgfani(xpomr0*scm,bbti,1.d0-vvxs*exp(-sumut)
     *  ,1.d0-exp(-sumct0),1.d0-exp(-sumut),iddt(itt),2,3))
        if(itt.gt.it)then
         vvxt0=vvxt0+vtac0(itt)
        elseif(itt.lt.it)then
        vvxt0l=vvxt0l+vtac0(itt)
        endif
        sumct0=sumct0+vtac0(itt)
       enddo
       vvxt0=1.d0-exp(-vvxt0)
       vvxt0l=1.d0-exp(-vvxt0l)

       viu=qgpini(xpomr0/xpomr,bbi,0.d0,0.d0,2)
       vim=2.d0*min(viu,qgpini(xpomr0/xpomr,bbi,0.d0,0.d0,8))
       vvxpin=1.d0-(1.d0-vvxp0)*(1.d0-vvxp0l)*exp(-vpac0(ip))
       vvxtin=1.d0-(1.d0-vvxt0)*(1.d0-vvxt0l)*exp(-vtac0(it))
       vi=qgpini(xpomr0/xpomr,bbi,vvxpin,vvxtin,21)*(1.d0-exp(-viu))
     * -qgpini(xpomr0/xpomr,bbi,vvxpin,vvxtin,23)*((1.d0-exp(-viu))**2
     * +(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))/2.d0

       vvx0s=(1.d0-vvxtin)**2
       vvxs=((1.d0-vvxt)*(1.d0-vvxtl))**2*exp(-2.d0*vtac(it))

       gb0=vi                        *15.
     * *((1.d0-exp(-vpact(ip)))*(1.d0-vvxpt)*(1.d0-vvxs)
     * +vvxs*(max(0.d0,1.d0-exp(-vpact(ip))*(1.d0+vpact(ip)))
     * *(1.d0-vvxpt)
     * -max(0.d0,1.d0-exp(-vpac0(ip))*(1.d0+vpac0(ip)))*(1.d0-vvxp0))
     * +vpact(ip)*exp(-vpact(ip))*((1.d0-vvxpt)*vvxs
     * -exp(vpact(ip)-vpac0(ip))*(1.d0-vvxp0)*(1.d0-vvxp0l)*vvx0s)
     * -vpac0(ip)*exp(-vpac0(ip))*(1.d0-vvxp0)
     * *(vvxs-vvx0s+vvxp0l*vvx0s))

       do in=1,nzzp
        nrej=0
32      xpomri=(xpomr*sgap**2)**qgran(b10)/sgap
        rp1=(rq(iddp(ip),icz)-alfp*dlog(xpomri))*4.d0*.0389d0
        rp2=alfp*dlog(xpomri/xpomr)*4.d0*.0389d0
        rp=rp1*rp2/(rp1+rp2)
        z=qgran(b10)
        phi=pi*qgran(b10)
        b0=dsqrt(-rp*dlog(z))
        bbp=(dsqrt(bpm)*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
        bbi=(dsqrt(bpm)*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2
        call qgbdef(bbp,bbi,xa(ip,1)+b,xa(ip,2),xxp,yyp
     *  ,xxi,yyi,int(1.5d0+qgran(b10)))   !coordinates for the vertex
        call qgfdf(xxi,yyi,xpomri,vpac,vtac
     *  ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,it)

        sumcp0=0.d0
        sumcpt=0.d0
        sumup=0.d0
        vvxp0=0.d0
        vvxp0l=0.d0
        do i=1,ia(1)
         sumup=sumup+vpac(i)
        enddo
        vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
        do i=1,ia(1)
         ipp=ia(1)-i+1
         bbpi=(xa(ipp,1)+b-xxi)**2+(xa(ipp,2)-yyi)**2
         sumup=sumup-vpac(ipp)
         vpac0(ipp)=min(vpac(ipp)
     *   ,qgfani(1.d0/xpomri,bbpi,1.d0-vvxs*exp(-sumup)
     *   ,1.d0-exp(-sumcp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
         if(ipp.ge.ip)vpact(ipp)=max(vpac(ipp)
     *   ,qgfani(1.d0/xpomri,bbpi,1.d0-vvxs*exp(-sumup)
     *   ,1.d0-exp(-sumcpt),1.d0-exp(-sumup),iddp(ipp),icz,6))
         if(ipp.gt.ip)then
          vvxp0=vvxp0+vpac0(ipp)
          sumcpt=sumcpt+vpact(ipp)
         elseif(ipp.lt.ip)then
          vvxp0l=vvxp0l+vpac0(ipp)
         endif
         sumcp0=sumcp0+vpac0(ipp)
        enddo
        vvxpt=1.d0-exp(-sumcpt)
        vvxp0=1.d0-exp(-vvxp0)
        vvxp0l=1.d0-exp(-vvxp0l)

        sumut=0.d0
        sumct0=0.d0
        vvxt0=0.d0
        vvxt0l=0.d0
        do i=1,ia(2)
         sumut=sumut+vtac(i)
        enddo
        vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
        do i=1,ia(2)
         itt=ia(2)-i+1
         bbti=(xb(itt,1)-xxi)**2+(xb(itt,2)-yyi)**2
         sumut=sumut-vtac(itt)
         vtac0(itt)=min(vtac(itt)
     *   ,qgfani(xpomri*scm,bbti,1.d0-vvxs*exp(-sumut)
     *   ,1.d0-exp(-sumct0),1.d0-exp(-sumut),iddt(itt),2,3))
         if(itt.gt.it)then
          vvxt0=vvxt0+vtac0(itt)
         elseif(itt.lt.it)then
          vvxt0l=vvxt0l+vtac0(itt)
         endif
         sumct0=sumct0+vtac0(itt)
        enddo
        vvxt0=1.d0-exp(-vvxt0)
        vvxt0l=1.d0-exp(-vvxt0l)

        viu=qgpini(xpomri/xpomr,bbi,0.d0,0.d0,2)
        vim=2.d0*min(viu,qgpini(xpomri/xpomr,bbi,0.d0,0.d0,8))
        vvxpin=1.d0-(1.d0-vvxp0)*(1.d0-vvxp0l)*exp(-vpac0(ip))
        vvxtin=1.d0-(1.d0-vvxt0)*(1.d0-vvxt0l)*exp(-vtac0(it))
        vi=qgpini(xpomri/xpomr,bbi,vvxpin,vvxtin,21)*(1.d0-exp(-viu))
     *  -qgpini(xpomri/xpomr,bbi,vvxpin,vvxtin,23)*((1.d0-exp(-viu))**2
     *  +(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))/2.d0

        vvx0s=(1.d0-vvxtin)**2
        vvxs=((1.d0-vvxt)*(1.d0-vvxtl))**2*exp(-2.d0*vtac(it))

        gb=vi
     *  *((1.d0-exp(-vpact(ip)))*(1.d0-vvxpt)*(1.d0-vvxs)
     *  +vvxs*(max(0.d0,1.d0-exp(-vpact(ip))*(1.d0+vpact(ip)))
     *  *(1.d0-vvxpt)
     *  -max(0.d0,1.d0-exp(-vpac0(ip))*(1.d0+vpac0(ip)))*(1.d0-vvxp0))
     *  +vpact(ip)*exp(-vpact(ip))*((1.d0-vvxpt)*vvxs
     *  -exp(vpact(ip)-vpac0(ip))*(1.d0-vvxp0)*(1.d0-vvxp0l)*vvx0s)
     *  -vpac0(ip)*exp(-vpac0(ip))*(1.d0-vvxp0)
     *  *(vvxs-vvx0s+vvxp0l*vvx0s))

        gb=gb/gb0/z*rp/rp0
        nrej=nrej+1
        if(qgran(b10).gt.gb.and.nrej.lt.10000)goto 32

        vi1p=qgpini(xpomri/xpomr,bbi,1.d0-(1.d0-vvxpin)**2*vvx0s
     *  ,0.d0,16)*exp(-vim)
        vimp=max(0.d0,(1.d0-exp(-vim)*(1.d0+vim)))/2.d0

        if(qgran(b10).le.(vi1p+vimp)/vi
     *  .or.xpomri/xpomr.lt.1.1d0*sgap**2)then
         if(qgran(b10).le.vi1p/(vi1p+vimp))then   !single cut Pomeron
          npin=npin+1
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          xpomim(npin)=1.d0/xpomr/scm
          xpomip(npin)=xpomri
          vvxim(npin)=1.d0-(1.d0-vvxpin)**2*vvx0s
          bpomim(npin)=bbi
          if(debug.ge.4)write (moniou,211)npin,xpomip(npin)
     *    ,xpomim(npin),vvxim(npin),bpomim(npin)
         else                                     !more than 1 cut Pomeron
          ninc=npgen(vim,2,20)
          npin=npin+ninc
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          do i=npin-ninc+1,npin
           xpomim(i)=1.d0/xpomr/scm
           xpomip(i)=xpomri
           vvxim(i)=0.d0
           bpomim(i)=bbi
           if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *     ,vvxim(i),bpomim(i)
          enddo
         endif

        else                                      !additional vertices
         xpomz0=dsqrt(xpomr*xpomri)
         rp0=alfp*dlog(xpomri/xpomr)*.0389d0
         xxz0=.5d0*(xxp+xxi)
         yyz0=.5d0*(yyp+yyi)
         bbzp=.25d0*bbi
         bbzt=bbzp
         call qgfdf(xxz0,yyz0,xpomz0,vpac,vtac,vvx,vvxp,vvxt
     *   ,vvxpl,vvxtl,ip,it)

         vvxp0=0.d0
         sumup=0.d0
         do i=1,ia(1)
          sumup=sumup+vpac(i)
         enddo
         vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
         do i=1,ia(1)
          ipp=ia(1)-i+1
          bbpi=(xa(ipp,1)+b-xxz0)**2+(xa(ipp,2)-yyz0)**2
          sumup=sumup-vpac(ipp)
          vpac0(ipp)=min(vpac(ipp)
     *    ,qgfani(1.d0/xpomz0,bbpi,1.d0-vvxs*exp(-sumup)
     *    ,1.d0-exp(-vvxp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
          vvxp0=vvxp0+vpac0(ipp)
         enddo
         vvxp0=1.d0-exp(-vvxp0)

         sumut=0.d0
         vvxt0=0.d0
         do i=1,ia(2)
          sumut=sumut+vtac(i)
         enddo
         vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
         do i=1,ia(2)
          itt=ia(2)-i+1
          bbti=(xb(itt,1)-xxz0)**2+(xb(itt,2)-yyz0)**2
          sumut=sumut-vtac(itt)
          vtac0(itt)=min(vtac(itt)
     *    ,qgfani(xpomz0*scm,bbti,1.d0-vvxs*exp(-sumut)
     *    ,1.d0-exp(-vvxt0),1.d0-exp(-sumut),iddt(itt),2,3))
          vvxt0=vvxt0+vtac0(itt)
         enddo
         vvxt0=1.d0-exp(-vvxt0)

         viu=qgpini(xpomri/xpomz0,bbzp,0.d0,0.d0,2)
         vilu=1.d0-exp(-viu)
         vimu=2.d0*min(viu,qgpini(xpomri/xpomz0,bbzp,0.d0,0.d0,8))
         vimpu=max(0.d0,(1.d0-exp(-vimu)*(1.d0+vimu)))/2.d0
         vid=qgpini(xpomz0/xpomr,bbzt,0.d0,0.d0,2)
         vild=1.d0-exp(-vid)
         vimd=2.d0*min(vid,qgpini(xpomz0/xpomr,bbzt,0.d0,0.d0,8))
         vimpd=max(0.d0,(1.d0-exp(-vimd)*(1.d0+vimd)))/2.d0

         vi1pu=qgpini(xpomri/xpomz0,bbzp
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimu)
         vguu=qgpini(xpomri/xpomz0,bbzp,vvxp0,vvxt0,21)*vilu      !uu+uc
         vgcu=qgpini(xpomri/xpomz0,bbzp,vvxp0,vvxt0,23)
     *   *(vilu**2+(exp(2.d0*viu-vimu)-1.d0)*exp(-2.d0*viu))/2.d0 !cc+cu
         vi1pd=qgpini(xpomz0/xpomr,bbzt
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimd)
         vgud=qgpini(xpomz0/xpomr,bbzt,vvxt0,vvxp0,21)*vild       !uu+uc
         vgcd=qgpini(xpomz0/xpomr,bbzt,vvxt0,vvxp0,23)
     *   *(vild**2+(exp(2.d0*vid-vimd)-1.d0)*exp(-2.d0*vid))/2.d0 !cc+cu

         gbz0=(vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd+vimpu*vgcd
     *   +vgcu*vimpd+vi1pu*vgcd+vgcu*vi1pd)*(1.d0-vvxp0)*(1.d0-vvxt0)
     *   +(vimpu+vi1pu)*vgud*(1.d0-vvxp0)*vvxt0
     *   +(vimpd+vi1pd)*vguu*(1.d0-vvxt0)*vvxp0

         nrej=0
34       xpomz=xpomr*sgap*(xpomri/xpomr/sgap**2)**qgran(b10)
         rpp=alfp*dlog(xpomri/xpomz)*4.d0*.0389d0
         rpt=alfp*dlog(xpomz/xpomr)*4.d0*.0389d0
         rp=rpp*rpt/(rpp+rpt)
         z=qgran(b10)
         phi=pi*qgran(b10)
         b0=dsqrt(-rp*dlog(z))
         bbzp=(dsqrt(bbi)*rpp/(rpp+rpt)+b0*cos(phi))**2
     *   +(b0*sin(phi))**2
         bbzt=(dsqrt(bbi)*rpt/(rpp+rpt)-b0*cos(phi))**2
     *   +(b0*sin(phi))**2
         call qgbdef(bbzp,bbzt,xxi,yyi,xxp,yyp,xxz,yyz
     *   ,int(1.5d0+qgran(b10)))               !coordinates for the vertex
         call qgfdf(xxz,yyz,xpomz,vpac,vtac
     *   ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,it)

         vvxp0=0.d0
         sumup=0.d0
         do i=1,ia(1)
          sumup=sumup+vpac(i)
         enddo
         vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
         do i=1,ia(1)
          ipp=ia(1)-i+1
          bbpi=(xa(ipp,1)+b-xxz)**2+(xa(ipp,2)-yyz)**2
          sumup=sumup-vpac(ipp)
          vpac0(ipp)=min(vpac(ipp)
     *    ,qgfani(1.d0/xpomz,bbpi,1.d0-vvxs*exp(-sumup)
     *    ,1.d0-exp(-vvxp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
          vvxp0=vvxp0+vpac0(ipp)
         enddo
         vvxp0=1.d0-exp(-vvxp0)

         sumut=0.d0
         vvxt0=0.d0
         do i=1,ia(2)
          sumut=sumut+vtac(i)
         enddo
         vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
         do i=1,ia(2)
          itt=ia(2)-i+1
          bbti=(xb(itt,1)-xxz)**2+(xb(itt,2)-yyz)**2
          sumut=sumut-vtac(itt)
          vtac0(itt)=min(vtac(itt)
     *    ,qgfani(xpomz*scm,bbti,1.d0-vvxs*exp(-sumut)
     *    ,1.d0-exp(-vvxt0),1.d0-exp(-sumut),iddt(itt),2,3))
          vvxt0=vvxt0+vtac0(itt)
         enddo
         vvxt0=1.d0-exp(-vvxt0)

         viu=qgpini(xpomri/xpomz,bbzp,0.d0,0.d0,2)
         vilu=1.d0-exp(-viu)
         vimu=2.d0*min(viu,qgpini(xpomri/xpomz,bbzp,0.d0,0.d0,8))
         vimpu=max(0.d0,(1.d0-exp(-vimu)*(1.d0+vimu)))/2.d0
         vid=qgpini(xpomz/xpomr,bbzt,0.d0,0.d0,2)
         vild=1.d0-exp(-vid)
         vimd=2.d0*min(vid,qgpini(xpomz/xpomr,bbzt,0.d0,0.d0,8))
         vimpd=max(0.d0,(1.d0-exp(-vimd)*(1.d0+vimd)))/2.d0

         vi1pu=qgpini(xpomri/xpomz,bbzp
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimu)
         vguu=qgpini(xpomri/xpomz,bbzp,vvxp0,vvxt0,21)*vilu       !uu+uc
         vgcu=qgpini(xpomri/xpomz,bbzp,vvxp0,vvxt0,23)
     *   *(vilu**2+(exp(2.d0*viu-vimu)-1.d0)*exp(-2.d0*viu))/2.d0 !cc+cu
         vi1pd=qgpini(xpomz/xpomr,bbzt
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimd)
         vgud=qgpini(xpomz/xpomr,bbzt,vvxt0,vvxp0,21)*vild        !uu+uc
         vgcd=qgpini(xpomz/xpomr,bbzt,vvxt0,vvxp0,23)
     *   *(vild**2+(exp(2.d0*vid-vimd)-1.d0)*exp(-2.d0*vid))/2.d0 !cc+cu

         vvcc=vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd+vimpu*vgcd+vgcu*vimpd
     *   +vi1pu*vgcd+vgcu*vi1pd
         vvt=vvcc*(1.d0-vvxp0)*(1.d0-vvxt0)
     *   +(vimpu+vi1pu)*vgud*(1.d0-vvxp0)*vvxt0
     *   +(vimpd+vi1pd)*vguu*(1.d0-vvxt0)*vvxp0

         gbz=vvt/gbz0/z*rp/rp0  /1.4d0
         nrej=nrej+1
         if(qgran(b10).gt.gbz.and.nrej.lt.10000)goto 34

         aks=vvt*qgran(b10)
         if(aks.gt.vvcc*(1.d0-vvxp0)*(1.d0-vvxt0)
     *   +(vimpu+vi1pu)*vgud*(1.d0-vvxp0)*vvxt0)then
          jtu=0
          if(qgran(b10).lt.vimpd/(vimpd+vi1pd))then
           jtd=2
          else
           jtd=1
          endif
         elseif(aks.gt.vvcc*(1.d0-vvxp0)*(1.d0-vvxt0))then
          jtd=0
          if(qgran(b10).lt.vimpu/(vimpu+vi1pu))then
           jtu=2
          else
           jtu=1
          endif
         else
          aks=vvcc*qgran(b10)
          if(aks.lt.vimpu*vimpd)then
           jtu=2
           jtd=2
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd)then
           jtu=2
           jtd=1
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd)then
           jtu=1
           jtd=2
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd
     *    +vimpu*vgcd)then
           jtu=2
           jtd=0
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd
     *    +vimpu*vgcd+vgcu*vimpd)then
           jtu=0
           jtd=2
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd
     *    +vimpu*vgcd+vgcu*vimpd+vi1pu*vgcd)then
           jtu=1
           jtd=0
          else
           jtu=0
           jtd=1
          endif
         endif

         if(jtu.eq.1)then                         !single cut Pomeron
          npin=npin+1
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          xpomim(npin)=1.d0/xpomz/scm
          xpomip(npin)=xpomri
          vvxim(npin)=1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2
          bpomim(npin)=bbzp
          if(debug.ge.4)write (moniou,211)npin,xpomip(npin)
     *    ,xpomim(npin),vvxim(npin),bpomim(npin)
         elseif(jtu.eq.2)then                     !more than 1 cut Pomeron
          ninc=npgen(vimu,2,20)
          npin=npin+ninc
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          do i=npin-ninc+1,npin
           xpomim(i)=1.d0/xpomz/scm
           xpomip(i)=xpomri
           vvxim(i)=0.d0
           bpomim(i)=bbzp
           if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *     ,vvxim(i),bpomim(i)
          enddo
         endif

         if(jtd.eq.1)then                         !single cut Pomeron
          npin=npin+1
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          xpomim(npin)=1.d0/xpomr/scm
          xpomip(npin)=xpomz
          vvxim(npin)=1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2
          bpomim(npin)=bbzt
          if(debug.ge.4)write (moniou,211)npin,xpomip(npin)
     *    ,xpomim(npin),vvxim(npin),bpomim(npin)
         elseif(jtu.eq.2)then                     !more than 1 cut Pomeron
          ninc=npgen(vimd,2,20)
          npin=npin+ninc
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          do i=npin-ninc+1,npin
           xpomim(i)=1.d0/xpomr/scm
           xpomip(i)=xpomz
           vvxim(i)=0.d0
           bpomim(i)=bbzt
           if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *     ,vvxim(i),bpomim(i)
          enddo
         endif
        endif
       enddo          !end of the zigzag-loop
      endif           !nzzp.ne.0

      if(nzzt.ne.0)then
       btm=(xb(it,1)-xxp)**2+(xb(it,2)-yyp)**2
       xpomr0=max(dsqrt(xpomr/scm),sgap/scm)
       xpomr0=min(xpomr0,xpomr/sgap)
       rp1=(rq(iddt(it),2)+alfp*dlog(xpomr0*scm))*4.d0*.0389d0
       rp2=alfp*dlog(xpomr/xpomr0)*4.d0*.0389d0
       rp0=rp1*rp2/(rp1+rp2)
       bbt=btm*(rp1/(rp1+rp2))**2
       bbi=btm*(rp2/(rp1+rp2))**2
       call qgbdef(bbt,bbi,xb(it,1),xb(it,2),xxp,yyp,xxp0,yyp0,1)
       call qgfdf(xxp0,yyp0,xpomr0,vpac,vtac
     * ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,it)

       sumct0=0.d0
       sumctt=0.d0
       sumut=0.d0
       vvxt0=0.d0
       vvxt0l=0.d0
       do i=1,ia(2)
        sumut=sumut+vtac(i)
       enddo
       vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
       do i=1,ia(2)
        itt=ia(2)-i+1
        bbti=(xb(itt,1)-xxp0)**2+(xb(itt,2)-yyp0)**2
        sumut=sumut-vtac(itt)
        vtac0(itt)=min(vtac(itt)
     *  ,qgfani(xpomr0*scm,bbti,1.d0-vvxs*exp(-sumut)
     *  ,1.d0-exp(-sumct0),1.d0-exp(-sumut),iddt(itt),2,3))
        if(itt.ge.it)vtact(itt)=max(vtac(itt)
     *  ,qgfani(xpomr0*scm,bbti,1.d0-vvxs*exp(-sumut)
     *  ,1.d0-exp(-sumctt),1.d0-exp(-sumut),iddt(itt),2,6))
        if(itt.gt.it)then
         vvxt0=vvxt0+vtac0(itt)
         sumctt=sumctt+vtact(itt)
        elseif(itt.lt.it)then
         vvxt0l=vvxt0l+vtac0(itt)
        endif
        sumct0=sumct0+vtac0(itt)
       enddo
       vvxtt=1.d0-exp(-sumctt)
       vvxt0=1.d0-exp(-vvxt0)
       vvxt0l=1.d0-exp(-vvxt0l)

       sumcp0=0.d0
       sumup=0.d0
       vvxp0=0.d0
       vvxp0l=0.d0
       do i=1,ia(1)
        sumup=sumup+vpac(i)
       enddo
       vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
       do i=1,ia(1)
        ipp=ia(1)-i+1
        bbpi=(xa(ipp,1)+b-xxp0)**2+(xa(ipp,2)-yyp0)**2
        sumup=sumup-vpac(ipp)
        vpac0(ipp)=min(vpac(ipp)
     *  ,qgfani(1.d0/xpomr0,bbpi,1.d0-vvxs*exp(-sumup)
     *  ,1.d0-exp(-sumcp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
        if(ipp.gt.ip)then
         vvxp0=vvxp0+vpac0(ipp)
        elseif(ipp.lt.ip)then
         vvxp0l=vvxp0l+vpac0(ipp)
        endif
        sumcp0=sumcp0+vpac0(ipp)
       enddo
       vvxp0=1.d0-exp(-vvxp0)
       vvxp0l=1.d0-exp(-vvxp0l)

       viu=qgpini(xpomr/xpomr0,bbi,0.d0,0.d0,2)
       vim=2.d0*min(viu,qgpini(xpomr/xpomr0,bbi,0.d0,0.d0,8))
       vvxpin=1.d0-(1.d0-vvxp0)*(1.d0-vvxp0l)*exp(-vpac0(ip))
       vvxtin=1.d0-(1.d0-vvxt0)*(1.d0-vvxt0l)*exp(-vtac0(it))
       vi=qgpini(xpomr/xpomr0,bbi,vvxtin,vvxpin,21)*(1.d0-exp(-viu))
     * -qgpini(xpomr/xpomr0,bbi,vvxtin,vvxpin,23)*((1.d0-exp(-viu))**2
     * +(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))/2.d0

       vvx0s=(1.d0-vvxpin)**2
       vvxs=((1.d0-vvxp)*(1.d0-vvxpl))**2*exp(-2.d0*vpac(ip))

       gb0=vi                      *15.
     * *((1.d0-exp(-vtact(it)))*(1.d0-vvxtt)*(1.d0-vvxs)
     * +vvxs*(max(0.d0,1.d0-exp(-vtact(it))*(1.d0+vtact(it)))
     * *(1.d0-vvxtt)
     * -max(0.d0,1.d0-exp(-vtac0(it))*(1.d0+vtac0(it)))*(1.d0-vvxt0))
     * +vtact(it)*exp(-vtact(it))*((1.d0-vvxtt)*vvxs
     * -exp(vtact(it)-vtac0(it))*(1.d0-vvxt0)*(1.d0-vvxt0l)*vvx0s)
     * -vtac0(it)*exp(-vtac0(it))*(1.d0-vvxt0)
     * *(vvxs-vvx0s+vvxt0l*vvx0s))

       do in=1,nzzt
        nrej=0
33      xpomri=xpomr/sgap/(xpomr*scm/sgap**2)**qgran(b10)
        rp1=(rq(iddt(it),2)+alfp*dlog(xpomri*scm))*4.d0*.0389d0
        rp2=alfp*dlog(xpomr/xpomri)*4.d0*.0389d0
        rp=rp1*rp2/(rp1+rp2)
        z=qgran(b10)
        phi=pi*qgran(b10)
        b0=dsqrt(-rp*dlog(z))
        bbt=(dsqrt(btm)*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
        bbi=(dsqrt(btm)*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2
        call qgbdef(bbt,bbi,xb(it,1),xb(it,2),xxp,yyp,xxi,yyi
     *  ,int(1.5d0+qgran(b10)))   !coordinates for the vertex
        call qgfdf(xxi,yyi,xpomri,vpac,vtac
     *  ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,it)

        sumct0=0.d0
        sumctt=0.d0
        sumut=0.d0
        vvxt0=0.d0
        vvxt0l=0.d0
        do i=1,ia(2)
         sumut=sumut+vtac(i)
        enddo
        vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
        do i=1,ia(2)
         itt=ia(2)-i+1
         bbti=(xb(itt,1)-xxi)**2+(xb(itt,2)-yyi)**2
         sumut=sumut-vtac(itt)
         vtac0(itt)=min(vtac(itt)
     *   ,qgfani(xpomri*scm,bbti,1.d0-vvxs*exp(-sumut)
     *   ,1.d0-exp(-sumct0),1.d0-exp(-sumut),iddt(itt),2,3))
         if(itt.ge.it)vtact(itt)=max(vtac(itt)
     *   ,qgfani(xpomri*scm,bbti,1.d0-vvxs*exp(-sumut)
     *   ,1.d0-exp(-sumctt),1.d0-exp(-sumut),iddt(itt),2,6))
         if(itt.gt.it)then
          vvxt0=vvxt0+vtac0(itt)
          sumctt=sumctt+vtact(itt)
         elseif(itt.lt.it)then
          vvxt0l=vvxt0l+vtac0(itt)
         endif
         sumct0=sumct0+vtac0(itt)
        enddo
        vvxtt=1.d0-exp(-sumctt)
        vvxt0=1.d0-exp(-vvxt0)
        vvxt0l=1.d0-exp(-vvxt0l)

        sumcp0=0.d0
        sumup=0.d0
        vvxp0=0.d0
        vvxp0l=0.d0
        do i=1,ia(1)
         sumup=sumup+vpac(i)
        enddo
        vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
        do i=1,ia(1)
         ipp=ia(1)-i+1
         bbpi=(xa(ipp,1)+b-xxi)**2+(xa(ipp,2)-yyi)**2
         sumup=sumup-vpac(ipp)
         vpac0(ipp)=min(vpac(ipp)
     *   ,qgfani(1.d0/xpomri,bbpi,1.d0-vvxs*exp(-sumup)
     *   ,1.d0-exp(-sumcp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
         if(ipp.gt.ip)then
          vvxp0=vvxp0+vpac0(ipp)
         elseif(ipp.lt.ip)then
          vvxp0l=vvxp0l+vpac0(ipp)
         endif
         sumcp0=sumcp0+vpac0(ipp)
        enddo
        vvxp0=1.d0-exp(-vvxp0)
        vvxp0l=1.d0-exp(-vvxp0l)

        viu=qgpini(xpomr/xpomri,bbi,0.d0,0.d0,2)
        vim=2.d0*min(viu,qgpini(xpomr/xpomri,bbi,0.d0,0.d0,8))
        vvxpin=1.d0-(1.d0-vvxp0)*(1.d0-vvxp0l)*exp(-vpac0(ip))
        vvxtin=1.d0-(1.d0-vvxt0)*(1.d0-vvxt0l)*exp(-vtac0(it))
        vi=qgpini(xpomr/xpomri,bbi,vvxtin,vvxpin,21)*(1.d0-exp(-viu))
     *  -qgpini(xpomr/xpomri,bbi,vvxtin,vvxpin,23)*((1.d0-exp(-viu))**2
     *  +(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))/2.d0

        vvx0s=(1.d0-vvxpin)**2
        vvxs=((1.d0-vvxp)*(1.d0-vvxpl))**2*exp(-2.d0*vpac(ip))

        gb=vi
     *  *((1.d0-exp(-vtact(it)))*(1.d0-vvxtt)*(1.d0-vvxs)
     *  +vvxs*(max(0.d0,1.d0-exp(-vtact(it))*(1.d0+vtact(it)))
     *  *(1.d0-vvxtt)
     *  -max(0.d0,1.d0-exp(-vtac0(it))*(1.d0+vtac0(it)))*(1.d0-vvxt0))
     *  +vtact(it)*exp(-vtact(it))*((1.d0-vvxtt)*vvxs
     *  -exp(vtact(it)-vtac0(it))*(1.d0-vvxt0)*(1.d0-vvxt0l)*vvx0s)
     *  -vtac0(it)*exp(-vtac0(it))*(1.d0-vvxt0)
     *  *(vvxs-vvx0s+vvxt0l*vvx0s))

        gb=gb/gb0/z*rp/rp0
        nrej=nrej+1
        if(qgran(b10).gt.gb.and.nrej.lt.10000)goto 33

        vi1p=qgpini(xpomr/xpomri,bbi,1.d0-(1.d0-vvxtin)**2*vvx0s
     *  ,0.d0,16)*exp(-vim)
        vimp=max(0.d0,(1.d0-exp(-vim)*(1.d0+vim)))/2.d0

        if(qgran(b10).le.(vi1p+vimp)/vi
     *  .or.xpomr/xpomri.lt.1.1d0*sgap**2)then
         if(qgran(b10).le.vi1p/(vi1p+vimp))then   !single cut Pomeron
          npin=npin+1
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          xpomim(npin)=1.d0/xpomri/scm
          xpomip(npin)=xpomr
          vvxim(npin)=1.d0-(1.d0-vvxtin)**2*vvx0s
          bpomim(npin)=bbi
          if(debug.ge.4)write (moniou,211)npin,xpomip(npin)
     *    ,xpomim(npin),vvxim(npin),bpomim(npin)
         else                                     !more than 1 cut pomeron
          ninc=npgen(vim,2,20)
          npin=npin+ninc
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          do i=npin-ninc+1,npin
           xpomim(i)=1.d0/xpomri/scm
           xpomip(i)=xpomr
           vvxim(i)=0.d0
           bpomim(i)=bbi
           if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *     ,vvxim(i),bpomim(i)
          enddo
         endif

        else                                      !additional vertices
         xpomz0=dsqrt(xpomr*xpomri)
         rp0=alfp*dlog(xpomr/xpomri)*.0389d0
         xxz0=.5d0*(xxp+xxi)
         yyz0=.5d0*(yyp+yyi)
         bbzp=.25d0*bbi
         bbzt=bbzp
         call qgfdf(xxz0,yyz0,xpomz0,vpac,vtac,vvx,vvxp,vvxt
     *   ,vvxpl,vvxtl,ip,it)

         vvxp0=0.d0
         sumup=0.d0
         do i=1,ia(1)
          sumup=sumup+vpac(i)
         enddo
         vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
         do i=1,ia(1)
          ipp=ia(1)-i+1
          bbpi=(xa(ipp,1)+b-xxz0)**2+(xa(ipp,2)-yyz0)**2
          sumup=sumup-vpac(ipp)
          vpac0(ipp)=min(vpac(ipp)
     *    ,qgfani(1.d0/xpomz0,bbpi,1.d0-vvxs*exp(-sumup)
     *    ,1.d0-exp(-vvxp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
          vvxp0=vvxp0+vpac0(ipp)
         enddo
         vvxp0=1.d0-exp(-vvxp0)

         sumut=0.d0
         vvxt0=0.d0
         do i=1,ia(2)
          sumut=sumut+vtac(i)
         enddo
         vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
         do i=1,ia(2)
          itt=ia(2)-i+1
          bbti=(xb(itt,1)-xxz0)**2+(xb(itt,2)-yyz0)**2
          sumut=sumut-vtac(itt)
          vtac0(itt)=min(vtac(itt)
     *    ,qgfani(xpomz0*scm,bbti,1.d0-vvxs*exp(-sumut)
     *    ,1.d0-exp(-vvxt0),1.d0-exp(-sumut),iddt(itt),2,3))
          vvxt0=vvxt0+vtac0(itt)
         enddo
         vvxt0=1.d0-exp(-vvxt0)

         viu=qgpini(xpomr/xpomz0,bbzp,0.d0,0.d0,2)
         vilu=1.d0-exp(-viu)
         vimu=2.d0*min(viu,qgpini(xpomr/xpomz0,bbzp,0.d0,0.d0,8))
         vimpu=max(0.d0,(1.d0-exp(-vimu)*(1.d0+vimu)))/2.d0
         vid=qgpini(xpomz0/xpomri,bbzt,0.d0,0.d0,2)
         vild=1.d0-exp(-vid)
         vimd=2.d0*min(vid,qgpini(xpomz0/xpomri,bbzt,0.d0,0.d0,8))
         vimpd=max(0.d0,(1.d0-exp(-vimd)*(1.d0+vimd)))/2.d0

         vi1pu=qgpini(xpomr/xpomz0,bbzp
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimu)
         vguu=qgpini(xpomr/xpomz0,bbzp,vvxp0,vvxt0,21)*vilu       !uu+uc
         vgcu=qgpini(xpomr/xpomz0,bbzp,vvxp0,vvxt0,23)
     *   *(vilu**2+(exp(2.d0*viu-vimu)-1.d0)*exp(-2.d0*viu))/2.d0 !cc+cu
         vi1pd=qgpini(xpomz0/xpomri,bbzt
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimd)
         vgud=qgpini(xpomz0/xpomri,bbzt,vvxt0,vvxp0,21)*vild      !uu+uc
         vgcd=qgpini(xpomz0/xpomri,bbzt,vvxt0,vvxp0,23)
     *   *(vild**2+(exp(2.d0*vid-vimd)-1.d0)*exp(-2.d0*vid))/2.d0 !cc+cu

         gbz0=(vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd+vimpu*vgcd
     *   +vgcu*vimpd+vi1pu*vgcd+vgcu*vi1pd)*(1.d0-vvxp0)*(1.d0-vvxt0)
     *   +(vimpu+vi1pu)*vgud*(1.d0-vvxp0)*vvxt0
     *   +(vimpd+vi1pd)*vguu*(1.d0-vvxt0)*vvxp0

         nrej=0
35       xpomz=xpomri*sgap*(xpomr/xpomri/sgap**2)**qgran(b10)
         rpt=alfp*dlog(xpomz/xpomri)*4.d0*.0389d0
         rpp=alfp*dlog(xpomr/xpomz)*4.d0*.0389d0
         rp=rpp*rpt/(rpp+rpt)
         z=qgran(b10)
         phi=pi*qgran(b10)
         b0=dsqrt(-rp*dlog(z))
         bbzt=(dsqrt(bbi)*rpt/(rpp+rpt)-b0*cos(phi))**2
     *   +(b0*sin(phi))**2
         bbzp=(dsqrt(bbi)*rpp/(rpp+rpt)+b0*cos(phi))**2
     *   +(b0*sin(phi))**2
         call qgbdef(bbzt,bbzp,xxi,yyi,xxp,yyp,xxz,yyz
     *   ,int(1.5d0+qgran(b10)))               !coordinates for the vertex
         call qgfdf(xxz,yyz,xpomz,vpac,vtac
     *   ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,it)

         vvxp0=0.d0
         sumup=0.d0
         do i=1,ia(1)
          sumup=sumup+vpac(i)
         enddo
         vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
         do i=1,ia(1)
          ipp=ia(1)-i+1
          bbpi=(xa(ipp,1)+b-xxz)**2+(xa(ipp,2)-yyz)**2
          sumup=sumup-vpac(ipp)
          vpac0(ipp)=min(vpac(ipp)
     *    ,qgfani(1.d0/xpomz,bbpi,1.d0-vvxs*exp(-sumup)
     *    ,1.d0-exp(-vvxp0),1.d0-exp(-sumup),iddp(ipp),icz,3))
          vvxp0=vvxp0+vpac0(ipp)
         enddo
         vvxp0=1.d0-exp(-vvxp0)

         sumut=0.d0
         vvxt0=0.d0
         do i=1,ia(2)
          sumut=sumut+vtac(i)
         enddo
         vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
         do i=1,ia(2)
          itt=ia(2)-i+1
          bbti=(xb(itt,1)-xxz)**2+(xb(itt,2)-yyz)**2
          sumut=sumut-vtac(itt)
          vtac0(itt)=min(vtac(itt)
     *    ,qgfani(xpomz*scm,bbti,1.d0-vvxs*exp(-sumut)
     *    ,1.d0-exp(-vvxt0),1.d0-exp(-sumut),iddt(itt),2,3))
          vvxt0=vvxt0+vtac0(itt)
         enddo
         vvxt0=1.d0-exp(-vvxt0)

         viu=qgpini(xpomr/xpomz,bbzp,0.d0,0.d0,2)
         vilu=1.d0-exp(-viu)
         vimu=2.d0*min(viu,qgpini(xpomr/xpomz,bbzp,0.d0,0.d0,8))
         vimpu=max(0.d0,(1.d0-exp(-vimu)*(1.d0+vimu)))/2.d0
         vid=qgpini(xpomz/xpomri,bbzt,0.d0,0.d0,2)
         vild=1.d0-exp(-vid)
         vimd=2.d0*min(vid,qgpini(xpomz/xpomri,bbzt,0.d0,0.d0,8))
         vimpd=max(0.d0,(1.d0-exp(-vimd)*(1.d0+vimd)))/2.d0

         vi1pu=qgpini(xpomr/xpomz,bbzp
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimu)
         vguu=qgpini(xpomr/xpomz,bbzp,vvxp0,vvxt0,21)*vilu        !uu+uc
         vgcu=qgpini(xpomr/xpomz,bbzp,vvxp0,vvxt0,23)
     *   *(vilu**2+(exp(2.d0*viu-vimu)-1.d0)*exp(-2.d0*viu))/2.d0 !cc+cu
         vi1pd=qgpini(xpomz/xpomri,bbzt
     *   ,1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2,0.d0,16)*exp(-vimd)
         vgud=qgpini(xpomz/xpomri,bbzt,vvxt0,vvxp0,21)*vild       !uu+uc
         vgcd=qgpini(xpomz/xpomri,bbzt,vvxt0,vvxp0,23)
     *   *(vild**2+(exp(2.d0*vid-vimd)-1.d0)*exp(-2.d0*vid))/2.d0 !cc+cu

         vvcc=vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd+vimpu*vgcd+vgcu*vimpd
     *   +vi1pu*vgcd+vgcu*vi1pd
         vvt=vvcc*(1.d0-vvxp0)*(1.d0-vvxt0)
     *   +(vimpu+vi1pu)*vgud*(1.d0-vvxp0)*vvxt0
     *   +(vimpd+vi1pd)*vguu*(1.d0-vvxt0)*vvxp0

         gbz=vvt/gbz0/z*rp/rp0    /1.4d0
         nrej=nrej+1
         if(qgran(b10).gt.gbz.and.nrej.lt.10000)goto 35

         aks=vvt*qgran(b10)
         if(aks.gt.vvcc*(1.d0-vvxp0)*(1.d0-vvxt0)
     *   +(vimpu+vi1pu)*vgud*(1.d0-vvxp0)*vvxt0)then
          jtu=0
          if(qgran(b10).lt.vimpd/(vimpd+vi1pd))then
           jtd=2
          else
           jtd=1
          endif
         elseif(aks.gt.vvcc*(1.d0-vvxp0)*(1.d0-vvxt0))then
          jtd=0
          if(qgran(b10).lt.vimpu/(vimpu+vi1pu))then
           jtu=2
          else
           jtu=1
          endif
         else
          aks=vvcc*qgran(b10)
          if(aks.lt.vimpu*vimpd)then
           jtu=2
           jtd=2
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd)then
           jtu=2
           jtd=1
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd)then
           jtu=1
           jtd=2
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd
     *    +vimpu*vgcd)then
           jtu=2
           jtd=0
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd
     *    +vimpu*vgcd+vgcu*vimpd)then
           jtu=0
           jtd=2
          elseif(aks.lt.vimpu*vimpd+vimpu*vi1pd+vi1pu*vimpd
     *    +vimpu*vgcd+vgcu*vimpd+vi1pu*vgcd)then
           jtu=1
           jtd=0
          else
           jtu=0
           jtd=1
          endif
         endif

         if(jtu.eq.1)then                         !single cut Pomeron
          npin=npin+1
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          xpomim(npin)=1.d0/xpomz/scm
          xpomip(npin)=xpomr
          vvxim(npin)=1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2
          bpomim(npin)=bbzp
          if(debug.ge.4)write (moniou,211)npin,xpomip(npin)
     *    ,xpomim(npin),vvxim(npin),bpomim(npin)
         elseif(jtu.eq.2)then                     !more than 1 cut Pomeron
          ninc=npgen(vimu,2,20)
          npin=npin+ninc
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          do i=npin-ninc+1,npin
           xpomim(i)=1.d0/xpomz/scm
           xpomip(i)=xpomr
           vvxim(i)=0.d0
           bpomim(i)=bbzp
           if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *     ,vvxim(i),bpomim(i)
          enddo
         endif

         if(jtd.eq.1)then                         !single cut Pomeron
          npin=npin+1
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          xpomim(npin)=1.d0/xpomri/scm
          xpomip(npin)=xpomz
          vvxim(npin)=1.d0-((1.d0-vvxp0)*(1.d0-vvxt0))**2
          bpomim(npin)=bbzt
          if(debug.ge.4)write (moniou,211)npin,xpomip(npin)
     *    ,xpomim(npin),vvxim(npin),bpomim(npin)
         elseif(jtu.eq.2)then                     !more than 1 cut Pomeron
          ninc=npgen(vimd,2,20)
          npin=npin+ninc
          if(npin.gt.npmax)then
           iret=1
           goto 31
          endif
          do i=npin-ninc+1,npin
           xpomim(i)=1.d0/xpomri/scm
           xpomip(i)=xpomz
           vvxim(i)=0.d0
           bpomim(i)=bbzt
           if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *     ,vvxim(i),bpomim(i)
          enddo
         endif
        endif
       enddo          !end of the zigzag-loop
      endif           !nzzt.ne.0

      call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *,ip,it)
      if((jt.eq.2.or.jt.eq.3.or.jt.eq.9)
     *.and.qgran(b10).lt.(1.d0-exp(-vpac(ip)))*(1.d0-vvxpl)
     */((1.d0-exp(-vpac(ip)))*(1.d0-vvxpl)+2.d0*vvxpl))then
       icdps=iddp(ip)
       do icdp=1,2
        iddp(ip)=icdp
        call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *  ,ip,it)
        wdp(icdp,ip)=(1.d0-exp(-vpac(ip)))*(1.d0-vvxpl)
       enddo
       iddp(ip)=icdps
      endif
      call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *,ip,it)
      if((jt.eq.2.or.jt.eq.4.or.jt.eq.10)
     *.and.qgran(b10).lt.(1.d0-exp(-vtac(it)))*(1.d0-vvxtl)
     */((1.d0-exp(-vtac(it)))*(1.d0-vvxtl)+2.d0*vvxtl))then
       icdts=iddt(it)
       do icdt=1,2
        iddt(it)=icdt
        call qgfdf(xxp,yyp,xpomr,vpac,vtac,vvx,vvxp,vvxt,vvxpl,vvxtl
     *  ,ip,it)
        wdt(icdt,it)=(1.d0-exp(-vtac(it)))*(1.d0-vvxtl)
       enddo
       iddt(it)=icdts
      endif

      if(nppr0.eq.0)goto 20

c projectile 'fans'
      m=0
      nppm(1)=nppr0
      xpomm(1)=xpomr
      wgpm(1)=wgpr0
      xxm(1)=xxp
      yym(1)=yyp
      do i=1,nppr0
       ippm(i,1)=ippr0(i)
       itypm(i,1)=itypr0(i)
      enddo

9     m=m+1                                 !next level multi-Pomeron vertex
      if(m.gt.levmax)then
       iret=1
       goto 31
      endif
      ii(m)=0
10    ii(m)=ii(m)+1                         !next cut fan in the vertex
      if(ii(m).gt.nppm(m))then              !all fans at the level considered
       m=m-1                                !one level down
       if(m.eq.0)goto 20                    !all proj. fans considered
       goto 10
      endif
      l=ii(m)
      ipp=ippm(l,m)                         !proj. index for the leg
      itypom=itypm(l,m)                     !type of the cut
      bpm=(xa(ipp,1)+b-xxm(m))**2+(xa(ipp,2)-yym(m))**2      !b^2 for the leg
      if(debug.ge.4)write (moniou,208)ii(m),m,ipp,bpm
      if(xpomm(m)*sgap**2.gt.1.d0)stop'xpomm(m)*sgap**2>1!'
      if(itypom.eq.4.and.xpomm(m)*sgap**3.gt.1.d0)
     *stop'4:xpomm(m)*sgap**3>1!'

      if(debug.ge.4)write (moniou,210)m
      xpomr0=min(dsqrt(xpomm(m)),1.d0/sgap)
      xpomr0=max(xpomr0,xpomm(m)*sgap)
      if(itypom.eq.4)xpomr0=min(xpomr0,dsqrt(xpomm(m)/sgap))
      rp1=(rq(iddp(ipp),icz)-alfp*dlog(xpomr0))*4.d0*.0389d0
      rp2=alfp*dlog(xpomr0/xpomm(m))*4.d0*.0389d0
      rp0=rp1*rp2/(rp1+rp2)
      bbp=bpm*(rp1/(rp1+rp2))**2
      bbi=bpm*(rp2/(rp1+rp2))**2
      call qgbdef(bbp,bbi,xa(ipp,1)+b,xa(ipp,2),xxm(m),yym(m)
     *,xxp0,yyp0,1)

      call qgfdf(xxp0,yyp0,xpomr0,vpac,vtac,vvx,vvxp,vvxt
     *,vvxpl,vvxtl,ipp,it)
      vvxts=1.d0-(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
      viu=qgpini(xpomr0/xpomm(m),bbi,0.d0,0.d0,2)
      vim=2.d0*min(viu,qgpini(xpomr0/xpomm(m),bbi,0.d0,0.d0,8))
      if(itypom.eq.-1.or.itypom.eq.4)then         !single cut Pomeron at the end
       vvxi=1.d0-((1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt))**2
     * *exp(-2.d0*vpac(ipp)-2.d0*vtac(it))
       vip=qgpini(xpomr0/xpomm(m),bbi,vvxi,0.d0,16)*exp(-vim)
      elseif(itypom.eq.2.or.itypom.eq.7)then       !>1 cut Poms at the end
       vimp=max(0.d0,1.d0-exp(-vim)*(1.d0+vim))
      else                                         !rap-gap
       vvxpin=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ipp))
       vvxtin=1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
       viuu=qgpini(xpomr0/xpomm(m),bbi,vvxpin,vvxtin,20)
     * *(1.d0-exp(-viu))
       viuc=max(0.d0,viuu
     * -qgpini(xpomr0/xpomm(m),bbi,vvxpin,vvxtin,21)*(1.d0-exp(-viu)))
       vicc=qgpini(xpomr0/xpomm(m),bbi,vvxpin,vvxtin,22)*.5d0
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
       vicu=max(0.d0,qgpini(xpomr0/xpomm(m),bbi,vvxpin,vvxtin,23)*.5d0
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
     * -vicc)
      endif

      if(itypom.le.3)then
       sumup=0.d0
       vvxp0=0.d0
       do i=1,ia(1)
        sumup=sumup+vpac(i)
       enddo
       vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
       do i=1,ia(1)-ipp+1
        ipi=ia(1)-i+1
        bbl=(xa(ipi,1)+b-xxp0)**2+(xa(ipi,2)-yyp0)**2
        sumup=sumup-vpac(ipi)
        vpac0(ipi)=min(vpac(ipi)
     *  ,qgfani(1.d0/xpomr0,bbl,1.d0-vvxs*exp(-sumup)
     *  ,1.d0-exp(-vvxp0),1.d0-exp(-sumup),iddp(ipi),icz,3))
        if(ipi.gt.ipp)vvxp0=vvxp0+vpac0(ipi)
       enddo
       vvxp0=1.d0-exp(-vvxp0)
       vpacng=min(vpac0(ipp)
     * ,qgfani(1.d0/xpomr0,bbp,vvxts,vvxp0,vvxpl,iddp(ipp),icz,4))
       vpacpe=min(vpacng
     * ,qgfani(1.d0/xpomr0,bbp,vvxts,vvxp0,vvxpl,iddp(ipp),icz,5))
      else
       vplc=qgfani(1.d0/xpomr0,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,7)
       vplc0=min(vplc
     * ,qgfani(1.d0/xpomr0,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,8))
       vplcng=min(vplc0
     * ,qgfani(1.d0/xpomr0,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,11))
       vplcpe=min(vplcng
     * ,qgfani(1.d0/xpomr0,bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,10))
      endif

      if(itypom.eq.-1)then          !'fan' (single cut Pomeron at the end)
       gb0=vip*((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*((vpac0(ipp)-vpacpe)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)-(vpac(ipp)-vpac0(ipp))*(1.d0-exp(-vpac(ipp))
     * *(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ipp))*(1.d0-vvxp))
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)*exp(-2.d0*vtac(it))
       gb0=gb0*40.d0
      elseif(itypom.eq.0)then      !'fan' (cut loop at the end - rapgap)
       gb0=((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl))
     * *(vicc*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * -vicu*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))))
     * *(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
     * -2.d0*vicu*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp)-vtac(it))
     * *(1.d0-vvx)*(1.d0-vvxt)
      elseif(itypom.eq.1)then      !'fan' (uncut end - rapgap)
       gb0=((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl))
     * *(viuc*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * +viuu*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))))
     * *(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
     * +2.d0*viuu*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp)-vtac(it))
     * *(1.d0-vvx)*(1.d0-vvxt)
      elseif(itypom.eq.2)then        !'fan' (>1 cut Poms at the end)
       gb0=vimp*((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*(vpac0(ipp)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)-(vpac(ipp)-vpac0(ipp))*(1.d0-exp(-vpac(ipp))
     * *(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ipp))*(1.d0-vvxp))
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)*exp(-2.d0*vtac(it))
      elseif(itypom.eq.3)then      !'fan' (cut/uncut end - rapgap)
       gb0=((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl))
     * *(vicc-wgpm(m)*viuc)*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * *(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
      elseif(itypom.eq.4)then          !'leg' (single cut Pomeron at the end)
       gb0=vip*((vplc0-vplcpe)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl))*exp(-vpac(ipp)-2.d0*vtac(it))*(1.d0-vvxp)
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)
       if(gb0.le.0.d0)then
        gb0=vip*vplc0*.01d0*exp(-vpac(ipp))*(1.d0-vvxp)
     *  *(1.d0-vvxpl)*exp(-vpac(ipp)-2.d0*vtac(it))*(1.d0-vvxp)
     *  *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)
       endif
      elseif(itypom.eq.5)then      !'leg' (cut/uncut end - rapgap)
       gb0=vplcng*exp(-2.d0*vpac(ipp)-vtac(it))
     * *(1.d0-vvxp)**2*(1.d0-vvxpl)*(1.d0-vvx)*(1.d0-vvxt)
     * *(vicc-wgpm(m)*viuc)*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
      elseif(itypom.eq.7)then      !'leg' (>1 cut Poms at the end)
       gb0=vimp*(vplc0*exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)
     * -(vplc-vplc0)*(1.d0-exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)))
     * *exp(-vpac(ipp)-2.d0*vtac(it))*(1.d0-vvxp)
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)
      endif
      if(gb0.le.0.d0)then      !so170712
       iret=1
       goto 31
      endif
      nrej=0

11    xpomm(m+1)=(xpomm(m)*sgap**2)**qgran(b10)/sgap
      if(itypom.eq.4)xpomm(m+1)=(xpomm(m)*sgap**3)**qgran(b10)/sgap**2
      rp1=(rq(iddp(ipp),icz)-alfp*dlog(xpomm(m+1)))*4.d0*.0389d0
      rp2=alfp*dlog(xpomm(m+1)/xpomm(m))*4.d0*.0389d0
      rp=rp1*rp2/(rp1+rp2)
      z=qgran(b10)
      phi=pi*qgran(b10)
      b0=dsqrt(-rp*dlog(z))
      bbp=(dsqrt(bpm)*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
      bbi=(dsqrt(bpm)*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2
      call qgbdef(bbp,bbi,xa(ipp,1)+b,xa(ipp,2),xxm(m),yym(m)
     *,xxm(m+1),yym(m+1),int(1.5d0+qgran(b10)))   !coordinates for the vertex

      call qgfdf(xxm(m+1),yym(m+1),xpomm(m+1),vpac,vtac
     *,vvx,vvxp,vvxt,vvxpl,vvxtl,ipp,it)
      vvxts=1.d0-(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
      viu=qgpini(xpomm(m+1)/xpomm(m),bbi,0.d0,0.d0,2)
      vim=2.d0*min(viu,qgpini(xpomm(m+1)/xpomm(m),bbi,0.d0,0.d0,8))
      if(itypom.eq.-1.or.itypom.eq.4)then         !single cut Pomeron at the end
       vvxi=1.d0-((1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt))**2
     * *exp(-2.d0*vpac(ipp)-2.d0*vtac(it))
       vip=qgpini(xpomm(m+1)/xpomm(m),bbi,vvxi,0.d0,16)*exp(-vim)
      elseif(itypom.eq.2.or.itypom.eq.7)then       !>1 cut Poms at the end
       vimp=max(0.d0,1.d0-exp(-vim)*(1.d0+vim))
      else                                         !rap-gap
       vvxpin=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ipp))
       vvxtin=1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
       viuu=qgpini(xpomm(m+1)/xpomm(m),bbi,vvxpin,vvxtin,20)
     * *(1.d0-exp(-viu))
       viuc=max(0.d0,viuu-qgpini(xpomm(m+1)/xpomm(m),bbi
     * ,vvxpin,vvxtin,21)*(1.d0-exp(-viu)))
       vicc=qgpini(xpomm(m+1)/xpomm(m),bbi,vvxpin,vvxtin,22)*.5d0
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
       vicu=max(0.d0,qgpini(xpomm(m+1)/xpomm(m),bbi,vvxpin,vvxtin,23)
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
     * /2.d0-vicc)
      endif

      if(itypom.le.3)then
       sumup=0.d0
       vvxp0=0.d0
       do i=1,ia(1)
        sumup=sumup+vpac(i)
       enddo
       vvxs=(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
       do i=1,ia(1)-ipp+1
        ipi=ia(1)-i+1
        bbl=(xa(ipi,1)+b-xxm(m+1))**2+(xa(ipi,2)-yym(m+1))**2
        sumup=sumup-vpac(ipi)
        vpac0(ipi)=min(vpac(ipi)
     *  ,qgfani(1.d0/xpomm(m+1),bbl,1.d0-vvxs*exp(-sumup)
     *  ,1.d0-exp(-vvxp0),1.d0-exp(-sumup),iddp(ipi),icz,3))
        if(ipi.gt.ipp)vvxp0=vvxp0+vpac0(ipi)
       enddo
       vvxp0=1.d0-exp(-vvxp0)

       vpacng=min(vpac0(ipp)
     * ,qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp0,vvxpl,iddp(ipp),icz,4))
       vpacpe=min(vpacng
     * ,qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp0,vvxpl,iddp(ipp),icz,5))
      else
       vplc=qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp,vvxpl,iddp(ipp)
     * ,icz,7)
       vplc0=min(vplc
     * ,qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,8))
       vplcng=min(vplc0
     * ,qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,11))
       vplcpe=min(vplcng
     * ,qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,10))
      endif

      if(itypom.eq.-1)then          !'fan' (single cut Pomeron at the end)
       gb=vip*((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*((vpac0(ipp)-vpacpe)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)-(vpac(ipp)-vpac0(ipp))*(1.d0-exp(-vpac(ipp))
     * *(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ipp))*(1.d0-vvxp))
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)*exp(-2.d0*vtac(it))
      elseif(itypom.eq.0)then      !'fan' (cut loop at the end - rapgap)
       gb=((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl))
     * *(vicc*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * -vicu*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))))
     * *(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
     * -2.d0*vicu*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp)-vtac(it))
     * *(1.d0-vvx)*(1.d0-vvxt)
      elseif(itypom.eq.1)then      !'fan' (uncut end - rapgap)
       gb=((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl))
     * *(viuc*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * +viuu*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))))
     * *(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
     * +2.d0*viuu*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp)-vtac(it))
     * *(1.d0-vvx)*(1.d0-vvxt)
      elseif(itypom.eq.2)then        !'fan' (>1 cut Poms at the end)
       gb=vimp*((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*(vpac0(ipp)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)-(vpac(ipp)-vpac0(ipp))*(1.d0-exp(-vpac(ipp))
     * *(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ipp))*(1.d0-vvxp))
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)*exp(-2.d0*vtac(it))
      elseif(itypom.eq.3)then      !'fan' (cut/uncut end - rapgap)
       gb=((max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
     * +((1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl)
     * +2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl))
     * *((vicc-wgpm(m)*viuc)*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * -(vicu+wgpm(m)*viuu)*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)
     * *exp(-vtac(it))))*(1.d0-vvx)*(1.d0-vvxt)*exp(-vtac(it))
     * -2.d0*(vicu+wgpm(m)*viuu)*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))
     * -1.d0-(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp)-vtac(it))
     * *(1.d0-vvx)*(1.d0-vvxt)
      elseif(itypom.eq.4)then          !'leg' (single cut Pomeron at the end)
       gb=vip*((vplc0-vplcpe)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)-(vplc-vplc0)*(1.d0-exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)))*exp(-vpac(ipp)-2.d0*vtac(it))*(1.d0-vvxp)
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)
      elseif(itypom.eq.5)then      !'leg' (cut/uncut end - rapgap)
       gb=vplcng*exp(-2.d0*vpac(ipp)-vtac(it))
     * *(1.d0-vvxp)**2*(1.d0-vvxpl)*(1.d0-vvx)*(1.d0-vvxt)
     * *((vicc-wgpm(m)*viuc)*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * -(vicu+wgpm(m)*viuu)*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)
     * *exp(-vtac(it))))
      elseif(itypom.eq.7)then      !'leg' (>1 cut Poms at the end)
       gb=vimp*(vplc0*exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)
     * -(vplc-vplc0)*(1.d0-exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)))
     * *exp(-vpac(ipp)-2.d0*vtac(it))*(1.d0-vvxp)
     * *(1.d0-vvx)*(1.d0-vvxt)**2*(1.d0-vvxtl)
      endif
      gb=gb/gb0/z*rp/rp0  /10.d0
      nrej=nrej+1
      if(qgran(b10).gt.gb.and.nrej.le.1000)goto 11

      if(itypom.eq.-1.or.itypom.eq.4)then  !'single cut Pomeron in the handle
       npin=npin+1
       if(npin.gt.npmax)then
        iret=1
        goto 31
       endif
       xpomim(npin)=1.d0/xpomm(m)/scm
       xpomip(npin)=xpomm(m+1)
       vvxim(npin)=vvxi
       bpomim(npin)=bbi
       if(debug.ge.4)write (moniou,211)npin,xpomip(npin),xpomim(npin)
     * ,vvxim(npin),bpomim(npin)
      elseif(itypom.eq.2.or.itypom.eq.7)then   !>1 cut Pomerons in the handle
       ninc=npgen(vim,2,20)
       npin=npin+ninc
       if(npin.gt.npmax)then
        iret=1
        goto 31
       endif
       do i=npin-ninc+1,npin
        xpomim(i)=1.d0/xpomm(m)/scm
        xpomip(i)=xpomm(m+1)
        vvxim(i)=0.d0
        bpomim(i)=bbi
        if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *  ,vvxim(i),bpomim(i)
       enddo
      endif

      if(itypom.eq.-1)then      !single cut Pomeron in the 'handle'
       vv1=(max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
       vv2=(1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl
       vv3=2.d0*((vpac0(ipp)-vpacpe)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)-(vpac(ipp)-vpac0(ipp))*(1.d0-exp(-vpac(ipp))
     * *(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ipp))*(1.d0-vvxp)
       if(xpomm(m+1)*sgap**2.gt..9d0.or.vv3.lt.0.d0)vv3=0.d0
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif
      elseif(itypom.eq.0)then    !cut 'loop' in the 'handle' (rap-gap)
       vv1=(max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp)-vtac(it))
     * *(1.d0-vvxt)*(1.d0-vvxtl)*(vicc+vicu)
     * /(vicc*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * -vicu*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))))
       vv2=(1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl
       vv3=2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl)
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif
      elseif(itypom.eq.1)then    !uncut 'handle' (rap-gap)
       vv1=(max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
       vv2=(1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl
       vv3=2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2*(1.d0-vvxpl)
       vv4=2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0-(vpac(ipp)
     * -vpac0(ipp)))*(1.d0-vvxp0)+(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))
     * *exp(-vpac(ipp))*viuu/(viuu*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)
     * *exp(-vtac(it)))+viuc*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it)))
       if(xpomm(m+1)*sgap**2.gt..9d0.or.vv4.lt.0.d0)vv4=0.d0
       aks=(vv1+vv2+vv3+vv4)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       elseif(aks.lt.vv1+vv2+vv3)then
        jt=3                     !1 cut fan
       else
        jt=4                     !>1 cut 'handle' fans
       endif
      elseif(itypom.eq.2)then    !>1 cut Pomerons in the 'handle'
       vv1=(max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))*exp(-vpac(ipp))
       vv2=(1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl
       vv3=2.d0*(vpac0(ipp)*exp(-vpac(ipp))*(1.d0-vvxp)
     * *(1.d0-vvxpl)-(vpac(ipp)-vpac0(ipp))*(1.d0-exp(-vpac(ipp))
     * *(1.d0-vvxp)*(1.d0-vvxpl)))*exp(-vpac(ipp))*(1.d0-vvxp)
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif

      elseif(itypom.eq.3)then    !rap-gap in the 'handle'
       vv1=(max(0.d0,1.d0-exp(-2.d0*vpac(ipp))
     * *(1.d0+2.d0*vpac(ipp)))+2.d0*vpac(ipp)*exp(-2.d0*vpac(ipp))
     * *(1.d0-(1.d0-vvxp)**2))*(1.d0-vvxpl)
     * -2.d0*(max(0.d0,exp(vpac(ipp)-vpac0(ipp))-1.d0
     * -(vpac(ipp)-vpac0(ipp)))*(1.d0-vvxp0)
     * +(vpac(ipp)-vpac0(ipp))*(vvxp-vvxp0))
     * *exp(-vpac(ipp)-vtac(it))*(1.d0-vvxt)*(1.d0-vvxtl)
     * *(vicc+vicu+wgpm(m)*(viuu-viuc))
     * /((vicc-wgpm(m)*viuc)*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     * -(vicu+wgpm(m)*viuu)*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)
     * *exp(-vtac(it))))
       vv2=(1.d0-exp(-vpac(ipp)))**2*(1.d0-vvxpl)
     * +2.d0*(1.d0-exp(-vpac(ipp)))*vvxpl
       vv3=2.d0*vpacng*exp(-2.d0*vpac(ipp))*(1.d0-vvxp)**2
     * *(1.d0-vvxpl)
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif
      else
       jt=5                      !cut leg
      endif

      nppm(m+1)=0
      wgpm(m+1)=0.d0
      if(jt.eq.1)then                        !>1 cut fans
       ntry=0
12     ntry=ntry+1
       nphm=0
       if(ipp.eq.ia(1).or.ntry.gt.100)then
        nppm(m+1)=npgen(2.d0*vpac(ipp),2,20)
        do i=1,nppm(m+1)
         if(qgran(b10).le.vpac0(ipp)/vpac(ipp)
     *   .or.xpomm(m+1)*sgap**2.gt..9d0)then
          itypm(i,m+1)=0
         else
          itypm(i,m+1)=1
          nphm=nphm+1
         endif
         ippm(i,m+1)=ipp
        enddo
        wh=(vpac(ipp)/vpac0(ipp)-1.d0)/nppm(m+1)
       else
        nppm(m+1)=npgen(2.d0*vpac(ipp),1,20)
        do i=1,nppm(m+1)
         if(qgran(b10).le.vpac0(ipp)/vpac(ipp)
     *   .or.xpomm(m+1)*sgap**2.gt..9d0)then
          itypm(i,m+1)=0
         else
          itypm(i,m+1)=1
          nphm=nphm+1
         endif
         ippm(i,m+1)=ipp
        enddo
        wh=(vpac(ipp)/vpac0(ipp)-1.d0)/nppm(m+1)
        do ipi=ipp+1,ia(1)
         ninc=npgen(2.d0*vpac(ipi),0,20)
         if(ninc.ne.0)then
          nppm(m+1)=nppm(m+1)+ninc
          nh0=nphm
          if(nppm(m+1).gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nppm(m+1)-ninc+1,nppm(m+1)
           if(qgran(b10).le.vpac0(ipi)/vpac(ipi)
     *     .or.xpomm(m+1)*sgap**2.gt..9d0)then
            itypm(i,m+1)=0
           else
            itypm(i,m+1)=1
            nphm=nphm+1
           endif
           ippm(i,m+1)=ipi
          enddo
          if(ninc.gt.nphm-nh0)wh=(vpac(ipi)/vpac0(ipi)-1.d0)/ninc
         endif
        enddo
        if(nppm(m+1).eq.1)goto 12
       endif

       if(nphm+1.ge.nppm(m+1))then
        if(itypom.eq.-1.or.itypom.eq.1.or.itypom.eq.2)then
         gbt=1.d0-exp(vpac(ipp)+(1.d0-nphm)*dlog(2.d0))
     *   /(1.d0-vvxp)/(1.d0-vvxpl)
        elseif(itypom.eq.0)then
         gbt=1.d0-(vicc+vicu)*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     *   /(vicc*(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     *   -vicu*(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))))
     *   *exp(vpac(ipp)+(1.d0-nphm)*dlog(2.d0))
     *   /(1.d0-vvxp)/(1.d0-vvxpl)
        elseif(itypom.eq.3)then
         gbt=1.d0-(vicc+vicu+wgpm(m)*(viuu-viuc))
     *   *(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))
     *   /((vicc-wgpm(m)*viuc)*(1.d0-vvxt)*(1.d0-vvxtl)
     *   *exp(-vtac(it))-(vicu+wgpm(m)*viuu)
     *   *(1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(it))))
     *   *exp(vpac(ipp)+(1.d0-nphm)*dlog(2.d0))
     *   /(1.d0-vvxp)/(1.d0-vvxpl)
        else
         stop'unknown itypom'
        endif
        if(nphm.eq.nppm(m+1).and.qgran(b10).gt.gbt
     *  .or.nphm+1.eq.nppm(m+1).and.qgran(b10).gt.1.d0+wh*gbt)then
         ntry=0
          goto 12
        endif
       endif

      elseif(jt.eq.4)then                    !>1 cut 'handle' fans
       ntry=0
14     ntry=ntry+1
       if(ipp.eq.ia(1).or.ntry.gt.100)then
        nppm(m+1)=npgen(vpac(ipp)-vpac0(ipp),2,20)
        do i=1,nppm(m+1)
          itypm(i,m+1)=1
         ippm(i,m+1)=ipp
        enddo
       else
        nppm(m+1)=npgen(vpac(ipp)-vpac0(ipp),1,20)
        do i=1,nppm(m+1)
         itypm(i,m+1)=1
         ippm(i,m+1)=ipp
        enddo
        do ipi=ipp+1,ia(1)
         ninc=npgen(vpac(ipi)-vpac0(ipi),0,20)
         if(ninc.ne.0)then
          nppm(m+1)=nppm(m+1)+ninc
          if(nppm(m+1).gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nppm(m+1)-ninc+1,nppm(m+1)
           itypm(i,m+1)=1
           ippm(i,m+1)=ipi
          enddo
         endif
        enddo
        if(nppm(m+1).eq.1)goto 14
       endif

      elseif(jt.eq.3)then                    !1 cut fan
       nppm(m+1)=1
       ippm(1,m+1)=ipp
       if(itypom.eq.-1)then             !single cut Pomeron in the 'handle'
        factor=exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)
        wng=(vpacng-vpacpe)*factor/((vpac0(ipp)-vpacpe)*factor
     *  -(vpac(ipp)-vpac0(ipp))*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0
     *  .or.xpomm(m+1)*sgap**2.gt..9d0)then
         itypm(1,m+1)=2          !>1 cut Pomerons in the 'handle'
        else
         itypm(1,m+1)=3          !rap-gap in the 'handle'
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       elseif(itypom.eq.2)then          !>1 cut Pomerons in the 'handle'
        factor=exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)
        wng=vpacng*factor/(vpac0(ipp)*factor
     *  -(vpac(ipp)-vpac0(ipp))*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0
     *  .or.xpomm(m+1)*sgap**2.gt..9d0)then
         if(qgran(b10).le.vpacpe/vpacng
     *   .or.xpomm(m+1)*sgap**2.gt..9d0)then
          itypm(1,m+1)=-1        !single cut Pomeron in the 'handle'
         else
          itypm(1,m+1)=2         !>1 cut Pomerons in the 'handle'
         endif
        else
         itypm(1,m+1)=3          !rap-gap in the 'handle'
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       else                             !rap-gap in the 'handle'
        if(qgran(b10).le.vpacpe/vpacng
     *  .or.xpomm(m+1)*sgap**2.gt..9d0)then
         itypm(1,m+1)=-1         !single cut Pomeron in the 'handle'
        else
         itypm(1,m+1)=2          !>1 cut Pomerons in the 'handle'
        endif
       endif

       if(itypm(1,m+1).eq.-1)then     !single cut Pomeron in the 'handle'
        vplcp=min(vpacpe
     *  ,qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,9))
        if(qgran(b10).le.vplcp/vpacpe
     *  .or.xpomm(m+1)*sgap**2.gt..9d0)itypm(1,m+1)=6 !single cut Pomeron
       endif

      elseif(jt.eq.5)then                    !cut 'leg'
       nppm(m+1)=1
       ippm(1,m+1)=ipp
       if(itypom.eq.4)then              !single cut Pomeron at the end
        if(xpomm(m+1)*sgap**2.ge.1.d0)stop'=4:xpomm(m+1)*sgap**2>1'
        factor=exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)
        wng=(vplcng-vplcpe)*factor/((vplc0-vplcpe)*factor
     *  -(vplc-vplc0)*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0)then
         itypm(1,m+1)=7          !>1 cut Pomerons at the end
        else
         itypm(1,m+1)=5          !rap-gap at the end
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       elseif(itypom.eq.5)then          !rap-gap at the end (cut or uncut loop)
        if(qgran(b10).le.vplcpe/vplcng
     *  .or.xpomm(m+1)*sgap**2.gt..9d0)then
         itypm(1,m+1)=4          !single cut Pomeron at the end
        else
         itypm(1,m+1)=7          !>1 cut Pomerons at the end
        endif
       elseif(itypom.eq.7)then          !>1 cut Pomerons at the end
        factor=exp(-vpac(ipp))*(1.d0-vvxp)*(1.d0-vvxpl)
        wng=vplcng*factor/(vplc0*factor-(vplc-vplc0)*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0
     *  .or.xpomm(m+1)*sgap**2.gt..9d0)then
         if(qgran(b10).le.vplcpe/vplcng
     *   .or.xpomm(m+1)*sgap**2.gt..9d0)then
          itypm(1,m+1)=4         !single cut Pomeron at the end
         else
          itypm(1,m+1)=7         !>1 cut Pomerons at the end
         endif
        else
         itypm(1,m+1)=5          !rap-gap at the end
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       endif

       if(itypm(1,m+1).eq.4)then        !single cut Pomeron at the end
        vplcp=min(vplcpe
     *  ,qgfani(1.d0/xpomm(m+1),bbp,vvxts,vvxp,vvxpl,iddp(ipp),icz,9))
        if(qgran(b10).le.vplcp/vplcpe
     *  .or.xpomm(m+1)*sgap**3.gt..9d0)itypm(1,m+1)=6 !single cut Pomeron
       endif
      endif

      if(nppm(m+1).eq.1.and.itypm(1,m+1).eq.6)then  !record single cut Pomeron
       nppr=nppr+1
       if(nppr.gt.legmax)then
        iret=1
        goto 31
       endif
       xpompi(nppr)=xpomm(m+1)
       vvxpi(nppr)=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*(1.d0-vvxt)
     * *(1.d0-vvxtl)*exp(-vtac(it))
       ipompi(nppr)=ipp
       bpompi(nppr)=bbp
       nppm(m+1)=0
       if(debug.ge.4)write (moniou,209)nppr,ipp,bbp,xpompi(nppr)
     * ,vvxpi(nppr)

      elseif(nppm(m+1).gt.1)then
       i=0
15     i=i+1
       ityp=itypm(i,m+1)
       if(ityp.eq.0)then
        ipi=ippm(i,m+1)
        bbi=(xa(ipi,1)+b-xxm(m+1))**2+(xa(ipi,2)-yym(m+1))**2
        vvxp=0.d0
        vvxpl=0.d0
        vvxp0=0.d0
        if(ia(1).gt.1)then
         do l=1,ia(1)
          if(l.lt.ipi)then
           vvxpl=vvxpl+vpac(l)
          elseif(l.gt.ipi)then
           vvxp=vvxp+vpac(l)
           vvxp0=vvxp0+vpac0(l)
          endif
         enddo
        endif
        vvxp=1.d0-exp(-vvxp)
        vvxpl=1.d0-exp(-vvxpl)
        vvxp0=1.d0-exp(-vvxp0)

        vpacng=min(vpac0(ipi)
     *  ,qgfani(1.d0/xpomm(m+1),bbi,vvxts,vvxp0,vvxpl,iddp(ipi),icz,4))
        vpacpe=min(vpacng
     *  ,qgfani(1.d0/xpomm(m+1),bbi,vvxts,vvxp0,vvxpl,iddp(ipi),icz,5))
        vplcp=min(vpacpe
     *  ,qgfani(1.d0/xpomm(m+1),bbi,vvxts,vvxp,vvxpl,iddp(ipi),icz,9))

        aks=qgran(b10)*vpac0(ipi)
        if(aks.le.vplcp.or.xpomm(m+1)*sgap**2.gt..9d0)then
         itypm(i,m+1)=6          !single cut Pomeron
        elseif(aks.lt.vpacpe)then
         itypm(i,m+1)=-1         !single cut Pomeron in the 'handle'
        elseif(aks.lt.vpacng)then
         itypm(i,m+1)=2          !>1 cut Pomerons in the 'handle'
        endif

        if(itypm(i,m+1).eq.6)then      !record single cut Pomeron
         nppr=nppr+1
         if(nppr.gt.legmax)then
          iret=1
          goto 31
         endif
         xpompi(nppr)=xpomm(m+1)
         vvxpi(nppr)=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*(1.d0-vvxt)
     *   *(1.d0-vvxtl)*exp(-vtac(it))
         ipompi(nppr)=ipi
         bpompi(nppr)=bbi
         if(debug.ge.4)write (moniou,209)nppr,ipi,bbi,xpompi(nppr)
     *   ,vvxpi(nppr)
         nppm(m+1)=nppm(m+1)-1
         if(nppm(m+1).ge.i)then
          do l=i,nppm(m+1)
           ippm(l,m+1)=ippm(l+1,m+1)
           itypm(l,m+1)=itypm(l+1,m+1)
          enddo
         endif
         i=i-1
        endif
       endif
       if(i.lt.nppm(m+1))goto 15
      endif

      if(jt.eq.2.and.qgran(b10).lt.(1.d0-exp(-vpac(ipp)))*(1.d0-vvxpl)
     */((1.d0-exp(-vpac(ipp)))*(1.d0-vvxpl)+2.d0*vvxpl))then
       if(debug.ge.4)write (moniou,212)
       icdps=iddp(ipp)
       do icdp=1,2
        iddp(ipp)=icdp
        call qgfdf(xxm(m+1),yym(m+1),xpomm(m+1),vpac,vtac
     *  ,vvx,vvxp,vvxt,vvxpl,vvxtl,ipp,it)
        wdp(icdp,ipp)=(1.d0-exp(-vpac(ipp)))*(1.d0-vvxpl)
       enddo
       iddp(ipp)=icdps
      endif

      if(nppm(m+1).ne.0)then
       goto 9
      else
       goto 10
      endif

20    continue
      if(debug.ge.3)write (moniou,214)nppr
      if(nptg0.eq.0)goto 31

c target 'fans'
      m=0
      nppm(1)=nptg0
      xpomm(1)=xpomr
      wgpm(1)=wgtg0
      xxm(1)=xxp
      yym(1)=yyp
      do i=1,nptg0
       ippm(i,1)=iptg0(i)
       itypm(i,1)=itytg0(i)
      enddo

21    m=m+1                                   !next level multi-Pomeron vertex
      if(m.gt.levmax)then
       iret=1
       goto 31
      endif
      ii(m)=0
22    ii(m)=ii(m)+1                           !next cut fan in the vertex
      if(ii(m).gt.nppm(m))then                !all fans at the level considered
       m=m-1                                  !one level down
       if(m.eq.0)goto 31                      !all targ. fans considered
       goto 22
      endif
      l=ii(m)
      itt=ippm(l,m)                           !targ. index for the leg
      itypom=itypm(l,m)                       !type of the cut
      btm=(xb(itt,1)-xxm(m))**2+(xb(itt,2)-yym(m))**2  !b^2 for the leg
      if(debug.ge.4)write (moniou,216)ii(m),m,itt,btm
      if(xpomm(m)*scm.lt.sgap**2)stop'xpomm(m)*scm<sgap**2!'

      if(debug.ge.4)write (moniou,210)m
      xpomr0=min(dsqrt(xpomm(m)/scm),xpomm(m)/sgap)
      xpomr0=max(xpomr0,sgap/scm)
      if(itypom.eq.4)xpomr0=max(xpomr0,dsqrt(xpomm(m)*sgap/scm))
      rp1=(rq(iddt(itt),2)+alfp*dlog(xpomr0*scm))*4.d0*.0389d0
      rp2=alfp*dlog(xpomm(m)/xpomr0)*4.d0*.0389d0
      rp0=rp1*rp2/(rp1+rp2)
      bbt=btm*(rp1/(rp1+rp2))**2
      bbi=btm*(rp2/(rp1+rp2))**2
      call qgbdef(bbt,bbi,xb(itt,1),xb(itt,2),xxm(m),yym(m)
     *,xxp0,yyp0,1)

      call qgfdf(xxp0,yyp0,xpomr0,vpac,vtac
     *,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,itt)
      vvxps=1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
      viu=qgpini(xpomm(m)/xpomr0,bbi,0.d0,0.d0,2)
      vim=2.d0*min(viu,qgpini(xpomm(m)/xpomr0,bbi,0.d0,0.d0,8))
      if(itypom.eq.-1.or.itypom.eq.4)then      !single cut Pomeron at the end
       vvxi=1.d0-((1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt))**2
     * *exp(-2.d0*vpac(ip)-2.d0*vtac(itt))
       vip=qgpini(xpomm(m)/xpomr0,bbi,vvxi,0.d0,16)*exp(-vim)
      elseif(itypom.eq.2.or.itypom.eq.7)then   !>1 cut Pomerons at the end
       vimp=max(0.d0,1.d0-exp(-vim)*(1.d0+vim))
      else                                     !rap-gap at the end
       vvxpin=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
       vvxtin=1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(itt))
       viuu=qgpini(xpomm(m)/xpomr0,bbi,vvxtin,vvxpin,20)
     * *(1.d0-exp(-viu))
       viuc=max(0.d0,viuu-qgpini(xpomm(m)/xpomr0,bbi
     * ,vvxtin,vvxpin,21)*(1.d0-exp(-viu)))
       vicc=qgpini(xpomm(m)/xpomr0,bbi,vvxtin,vvxpin,22)*.5d0
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
       vicu=max(0.d0,qgpini(xpomm(m)/xpomr0,bbi,vvxtin,vvxpin,23)*.5d0
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
     * -vicc)
      endif

      if(itypom.le.3)then                         !cut 'fan'
       sumut=0.d0
       vvxt0=0.d0
       do i=1,ia(2)
        sumut=sumut+vtac(i)
       enddo
       vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
       do i=1,ia(2)-itt+1
        iti=ia(2)-i+1
        bbl=(xb(iti,1)-xxp0)**2+(xb(iti,2)-yyp0)**2
        sumut=sumut-vtac(iti)
        vtac0(iti)=min(vtac(iti)
     *  ,qgfani(xpomr0*scm,bbl,1.d0-vvxs*exp(-sumut)
     *  ,1.d0-exp(-vvxt0),1.d0-exp(-sumut),iddt(iti),2,3))
        if(iti.gt.itt)vvxt0=vvxt0+vtac0(iti)
       enddo
       vvxt0=1.d0-exp(-vvxt0)
       vtacng=min(vtac0(itt)
     * ,qgfani(xpomr0*scm,bbt,vvxps,vvxt0,vvxtl,iddt(itt),2,4))
       vtacpe=min(vtacng
     * ,qgfani(xpomr0*scm,bbt,vvxps,vvxt0,vvxtl,iddt(itt),2,5))
      else                                        !cut 'leg'
       vtlc=qgfani(xpomr0*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,7)
       vtlc0=min(vtlc
     * ,qgfani(xpomr0*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,8))
       vtlcng=min(vtlc0
     * ,qgfani(xpomr0*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,11))
       vtlcpe=min(vtlcng
     * ,qgfani(xpomr0*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,10))
      endif

      if(itypom.eq.-1)then         !'fan' (single cut Pomeron at the end)
       gb0=vip*((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*((vtac0(itt)-vtacpe)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)-(vtac(itt)-vtac0(itt))*(1.d0-exp(-vtac(itt))
     * *(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(itt))*(1.d0-vvxt))
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)*exp(-2.d0*vpac(ip))
       gb0=gb0*40.d0
      elseif(itypom.eq.0)then      !'fan' (cut loop at the end - rapgap)
       gb0=((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl))
     * *(vicc*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * -vicu*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))))
     * *(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
     * -2.d0*vicu*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt)-vpac(ip))
     * *(1.d0-vvx)*(1.d0-vvxp)
      elseif(itypom.eq.1)then      !'fan' (uncut end - rapgap)
       gb0=((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl))
     * *(viuc*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * +viuu*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))))
     * *(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
     * +2.d0*viuu*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt)-vpac(ip))
     * *(1.d0-vvx)*(1.d0-vvxp)
      elseif(itypom.eq.2)then      !'fan' (>1 cut Poms at the end)
       gb0=vimp*((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*(vtac0(itt)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)-(vtac(itt)-vtac0(itt))*(1.d0-exp(-vtac(itt))
     * *(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(itt))*(1.d0-vvxt))
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)*exp(-2.d0*vpac(ip))
      elseif(itypom.eq.3)then      !'fan' (cut/uncut end - rapgap)
       gb0=((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl))
     * *(vicc-wgpm(m)*viuc)*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * *(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
      elseif(itypom.eq.4)then      !'leg' (single cut Pomeron at the end)
       gb0=vip*((vtlc0-vtlcpe)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl))*exp(-vtac(itt)-2.d0*vpac(ip))*(1.d0-vvxt)
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)
       if(gb0.eq.0.d0)then
        gb0=vip*vtlc0*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)*exp(-vtac(itt)-2.d0*vpac(ip))*(1.d0-vvxt)
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)  *.01d0
       endif
      elseif(itypom.eq.5)then      !'leg' (cut/uncut end - rapgap)
       gb0=vtlcng*exp(-2.d0*vtac(itt)-vpac(ip))
     * *(1.d0-vvxt)**2*(1.d0-vvxtl)*(1.d0-vvx)*(1.d0-vvxp)
     * *(vicc-wgpm(m)*viuc)*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
      elseif(itypom.eq.7)then      !'leg' (>1 cut Poms at the end)
       gb0=vimp*(vtlc0*exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)
     * -(vtlc-vtlc0)*(1.d0-exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)))
     * *exp(-vtac(itt)-2.d0*vpac(ip))*(1.d0-vvxt)
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)
      endif
      if(gb0.le.0.d0)then      !so170712
       iret=1
       goto 31
      endif
      nrej=0

23    xpomm(m+1)=xpomm(m)/sgap/(xpomm(m)*scm/sgap**2)**qgran(b10)
      if(itypom.eq.4)xpomm(m+1)=xpomm(m)/sgap
     */(xpomm(m)*scm/sgap**3)**qgran(b10)
      rp1=(rq(iddt(itt),2)+alfp*dlog(xpomm(m+1)*scm))*4.d0*.0389d0
      rp2=alfp*dlog(xpomm(m)/xpomm(m+1))*4.d0*.0389d0
      rp=rp1*rp2/(rp1+rp2)
      z=qgran(b10)
      phi=pi*qgran(b10)
      b0=dsqrt(-rp*dlog(z))
      bbt=(dsqrt(btm)*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
      bbi=(dsqrt(btm)*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2
      call qgbdef(bbt,bbi,xb(itt,1),xb(itt,2),xxm(m),yym(m)
     *,xxm(m+1),yym(m+1),int(1.5d0+qgran(b10)))   !coordinates for the vertex

      call qgfdf(xxm(m+1),yym(m+1),xpomm(m+1),vpac,vtac
     *,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,itt)
      vvxps=1.d0-(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
      viu=qgpini(xpomm(m)/xpomm(m+1),bbi,0.d0,0.d0,2)
      vim=2.d0*min(viu,qgpini(xpomm(m)/xpomm(m+1),bbi,0.d0,0.d0,8))
      if(itypom.eq.-1.or.itypom.eq.4)then      !single cut Pomeron at the end
       vvxi=1.d0-((1.d0-vvx)*(1.d0-vvxp)*(1.d0-vvxt))**2
     * *exp(-2.d0*vpac(ip)-2.d0*vtac(itt))
       vip=qgpini(xpomm(m)/xpomm(m+1),bbi,vvxi,0.d0,16)*exp(-vim)
      elseif(itypom.eq.2.or.itypom.eq.7)then   !>1 cut Pomerons at the end
        vimp=max(0.d0,1.d0-exp(-vim)*(1.d0+vim))
      else                                     !rap-gap at the end
       vvxpin=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
       vvxtin=1.d0-(1.d0-vvxt)*(1.d0-vvxtl)*exp(-vtac(itt))
       viuu=qgpini(xpomm(m)/xpomm(m+1),bbi,vvxtin,vvxpin,20)
     * *(1.d0-exp(-viu))
       viuc=max(0.d0,viuu-qgpini(xpomm(m)/xpomm(m+1),bbi
     * ,vvxtin,vvxpin,21)*(1.d0-exp(-viu)))
       vicc=qgpini(xpomm(m)/xpomm(m+1),bbi,vvxtin,vvxpin,22)*.5d0
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
       vicu=max(0.d0,qgpini(xpomm(m)/xpomm(m+1),bbi,vvxtin,vvxpin,23)
     * *((1.d0-exp(-viu))**2+(exp(2.d0*viu-vim)-1.d0)*exp(-2.d0*viu))
     * /2.d0-vicc)
      endif

      if(itypom.le.3)then                         !cut 'fan'
       sumut=0.d0
       vvxt0=0.d0
       do i=1,ia(2)
        sumut=sumut+vtac(i)
       enddo
       vvxs=(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
       do i=1,ia(2)-itt+1
        iti=ia(2)-i+1
        bbl=(xb(iti,1)-xxm(m+1))**2+(xb(iti,2)-yym(m+1))**2
        sumut=sumut-vtac(iti)
        vtac0(iti)=min(vtac(iti)
     *  ,qgfani(xpomm(m+1)*scm,bbl,1.d0-vvxs*exp(-sumut)
     *  ,1.d0-exp(-vvxt0),1.d0-exp(-sumut),iddt(iti),2,3))
        if(iti.gt.itt)vvxt0=vvxt0+vtac0(iti)
       enddo
       vvxt0=1.d0-exp(-vvxt0)

       vtacng=min(vtac0(itt)
     * ,qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt0,vvxtl,iddt(itt),2,4))
       vtacpe=min(vtacng
     * ,qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt0,vvxtl,iddt(itt),2,5))
      else                                        !cut 'leg'
       vtlc=qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,7)
       vtlc0=min(vtlc
     * ,qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,8))
       vtlcng=min(vtlc0
     * ,qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,11))
       vtlcpe=min(vtlcng
     * ,qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,10))
      endif

      if(itypom.eq.-1)then         !'fan' (single cut Pomeron at the end)
       gb=vip*((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*((vtac0(itt)-vtacpe)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)-(vtac(itt)-vtac0(itt))*(1.d0-exp(-vtac(itt))
     * *(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(itt))*(1.d0-vvxt))
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)*exp(-2.d0*vpac(ip))
      elseif(itypom.eq.0)then      !'fan' (cut loop at the end - rapgap)
       gb=((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl))
     * *(vicc*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * -vicu*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))))
     * *(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
     * -2.d0*vicu*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt)-vpac(ip))
     * *(1.d0-vvx)*(1.d0-vvxp)
      elseif(itypom.eq.1)then      !'fan' (uncut end - rapgap)
       gb=((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl))
     * *(viuc*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * +viuu*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))))
     * *(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
     * +2.d0*viuu*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt)-vpac(ip))
     * *(1.d0-vvx)*(1.d0-vvxp)
      elseif(itypom.eq.2)then      !'fan' (>1 cut Poms at the end)
       gb=vimp*((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*(vtac0(itt)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)-(vtac(itt)-vtac0(itt))*(1.d0-exp(-vtac(itt))
     * *(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(itt))*(1.d0-vvxt))
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)*exp(-2.d0*vpac(ip))
      elseif(itypom.eq.3)then      !'fan' (cut/uncut end - rapgap)
       gb=((max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
     * +((1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl)
     * +2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl))
     * *((vicc-wgpm(m)*viuc)*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * -(vicu+wgpm(m)*viuu)*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)
     * *exp(-vpac(ip))))*(1.d0-vvx)*(1.d0-vvxp)*exp(-vpac(ip))
     * -2.d0*(vicu+wgpm(m)*viuu)*(max(0.d0,exp(vtac(itt)-vtac0(itt))
     * -1.d0-(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt)-vpac(ip))
     * *(1.d0-vvx)*(1.d0-vvxp)
      elseif(itypom.eq.4)then      !'leg' (single cut Pomeron at the end)
       gb=vip*((vtlc0-vtlcpe)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)-(vtlc-vtlc0)*(1.d0-exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)))*exp(-vtac(itt)-2.d0*vpac(ip))*(1.d0-vvxt)
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)
      elseif(itypom.eq.5)then      !'leg' (cut/uncut end - rapgap)
       gb=vtlcng*exp(-2.d0*vtac(itt)-vpac(ip))
     * *(1.d0-vvxt)**2*(1.d0-vvxtl)*(1.d0-vvx)*(1.d0-vvxp)
     * *((vicc-wgpm(m)*viuc)*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * -(vicu+wgpm(m)*viuu)*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)
     * *exp(-vpac(ip))))
      elseif(itypom.eq.7)then      !'leg' (>1 cut Poms at the end)
       gb=vimp*(vtlc0*exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)
     * -(vtlc-vtlc0)*(1.d0-exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)))
     * *exp(-vtac(itt)-2.d0*vpac(ip))*(1.d0-vvxt)
     * *(1.d0-vvx)*(1.d0-vvxp)**2*(1.d0-vvxpl)
      endif
      nrej=nrej+1
      gb=gb/gb0/z*rp/rp0  /10.d0
      if(qgran(b10).gt.gb.and.nrej.le.1000)goto 23

      if(itypom.eq.-1.or.itypom.eq.4)then    !'single cut Pomeron in the handle
       npin=npin+1
       if(npin.gt.npmax)then
        iret=1
        goto 31
       endif
       xpomim(npin)=1.d0/xpomm(m+1)/scm
       xpomip(npin)=xpomm(m)
       vvxim(npin)=vvxi
       bpomim(npin)=bbi
       if(debug.ge.4)write (moniou,211)npin,xpomip(npin),xpomim(npin)
     * ,vvxim(npin),bpomim(npin)
      elseif(itypom.eq.2.or.itypom.eq.7)then !>1 cut Pomerons in the handle
       ninc=npgen(vim,2,20)
       npin=npin+ninc
       if(npin.gt.npmax)then
        iret=1
        goto 31
       endif
       do i=npin-ninc+1,npin
        xpomim(i)=1.d0/xpomm(m+1)/scm
        xpomip(i)=xpomm(m)
        vvxim(i)=0.d0
        bpomim(i)=bbi
        if(debug.ge.4)write (moniou,211)i,xpomip(i),xpomim(i)
     *  ,vvxim(i),bpomim(i)
       enddo
      endif

      if(itypom.eq.-1)then      !single cut Pomeron in the 'handle'
       vv1=(max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
       vv2=(1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl
       vv3=2.d0*((vtac0(itt)-vtacpe)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)-(vtac(itt)-vtac0(itt))*(1.d0-exp(-vtac(itt))
     * *(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(itt))*(1.d0-vvxt)
       if(xpomm(m+1)*scm.lt.1.1d0*sgap**2.or.vv3.lt.0.d0)vv3=0.d0
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif
      elseif(itypom.eq.0)then      !cut 'loop' in the 'handle'
       vv1=(max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt)-vpac(ip))
     * *(1.d0-vvxp)*(1.d0-vvxpl)*(vicc+vicu)
     * /(vicc*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * -vicu*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))))
       vv2=(1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl
       vv3=2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl)
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif
      elseif(itypom.eq.1)then    !uncut 'handle' (rap-gap)
       vv1=(max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
       vv2=(1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl
       vv3=2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2*(1.d0-vvxtl)
       vv4=2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0-(vtac(itt)
     * -vtac0(itt)))*(1.d0-vvxt0)+(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))
     * *exp(-vtac(itt))*viuu/(viuu*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)
     * *exp(-vpac(ip)))+viuc*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip)))
       if(xpomm(m+1)*scm.lt.1.1d0*sgap**2.or.vv4.lt.0.d0)vv4=0.d0
       aks=(vv1+vv2+vv3+vv4)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       elseif(aks.lt.vv1+vv2+vv3)then
        jt=3                     !1 cut fan
       else
        jt=4                     !>1 cut 'handle' fans
       endif
      elseif(itypom.eq.2)then    !>1 cut Pomerons in the 'handle'
       vv1=(max(0.d0,1.d0-exp(-2.d0*vtac(itt))
     * *(1.d0+2.d0*vtac(itt)))+2.d0*vtac(itt)*exp(-2.d0*vtac(itt))
     * *(1.d0-(1.d0-vvxt)**2))*(1.d0-vvxtl)
     * -2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)
     * +(vtac(itt)-vtac0(itt))*(vvxt-vvxt0))*exp(-vtac(itt))
       vv2=(1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl
       vv3=2.d0*(vtac0(itt)*exp(-vtac(itt))*(1.d0-vvxt)
     * *(1.d0-vvxtl)-(vtac(itt)-vtac0(itt))*(1.d0-exp(-vtac(itt))
     * *(1.d0-vvxt)*(1.d0-vvxtl)))*exp(-vtac(itt))*(1.d0-vvxt)
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif
      elseif(itypom.eq.3)then    !rap-gap in the 'handle'
       vv1=(max(0.d0,1.d0-exp(-2.d0*vtac(itt))*(1.d0+2.d0*vtac(itt)))
     * +2.d0*vtac(itt)*exp(-2.d0*vtac(itt))*(1.d0-(1.d0-vvxt)**2))
     * *(1.d0-vvxtl)-2.d0*(max(0.d0,exp(vtac(itt)-vtac0(itt))-1.d0
     * -(vtac(itt)-vtac0(itt)))*(1.d0-vvxt0)+(vtac(itt)-vtac0(itt))
     * *(vvxt-vvxt0))*exp(-vtac(itt)-vpac(ip))*(1.d0-vvxp)*(1.d0-vvxpl)
     * *(vicc+vicu+wgpm(m)*(viuu-viuc))
     * /((vicc-wgpm(m)*viuc)*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     * -(vicu+wgpm(m)*viuu)*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)
     * *exp(-vpac(ip))))
       vv2=(1.d0-exp(-vtac(itt)))**2*(1.d0-vvxtl)
     * +2.d0*(1.d0-exp(-vtac(itt)))*vvxtl
       vv3=2.d0*vtacng*exp(-2.d0*vtac(itt))*(1.d0-vvxt)**2
     * *(1.d0-vvxtl)
       aks=(vv1+vv2+vv3)*qgran(b10)
       if(aks.lt.vv1)then
        jt=1                     !>1 cut fans
       elseif(aks.lt.vv1+vv2)then
        jt=2                     !diffr. cut
       else
        jt=3                     !1 cut fan
       endif
      else
       jt=5                      !cut leg
      endif

      nppm(m+1)=0
      wgpm(m+1)=0.d0
      if(jt.eq.1)then                        !>1 cut fans
       ntry=0
24     ntry=ntry+1
       nphm=0
       if(itt.eq.ia(2).or.ntry.gt.100)then
        nppm(m+1)=npgen(2.d0*vtac(itt),2,20)
        do i=1,nppm(m+1)
         if(qgran(b10).le.vtac0(itt)/vtac(itt)
     *   .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
          itypm(i,m+1)=0
         else
          nphm=nphm+1
          itypm(i,m+1)=1
         endif
         ippm(i,m+1)=itt
        enddo
        wh=(vtac(itt)/vtac0(itt)-1.d0)/nppm(m+1)
       else
        nppm(m+1)=npgen(2.d0*vtac(itt),1,20)
        do i=1,nppm(m+1)
         if(qgran(b10).le.vtac0(itt)/vtac(itt)
     *   .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
          itypm(i,m+1)=0
         else
          nphm=nphm+1
          itypm(i,m+1)=1
         endif
         ippm(i,m+1)=itt
        enddo
        wh=(vtac(itt)/vtac0(itt)-1.d0)/nppm(m+1)
        do iti=itt+1,ia(2)
         ninc=npgen(2.d0*vtac(iti),0,20)
         if(ninc.ne.0)then
          nppm(m+1)=nppm(m+1)+ninc
          nh0=nphm
          if(nppm(m+1).gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nppm(m+1)-ninc+1,nppm(m+1)
           if(qgran(b10).le.vtac0(iti)/vtac(iti)
     *     .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
            itypm(i,m+1)=0
           else
            nphm=nphm+1
            itypm(i,m+1)=1
           endif
           ippm(i,m+1)=iti
          enddo
          if(ninc.gt.nphm-nh0)wh=(vtac(iti)/vtac0(iti)-1.d0)/ninc
         endif
        enddo
        if(nppm(m+1).eq.1)goto 24
       endif

       if(nphm+1.ge.nppm(m+1))then
        if(itypom.eq.-1.or.itypom.eq.1.or.itypom.eq.2)then
         gbt=1.d0-exp(vtac(itt)+(1.d0-nphm)*dlog(2.d0))
     *   /(1.d0-vvxt)/(1.d0-vvxtl)
        elseif(itypom.eq.0)then
         gbt=1.d0-(vicc+vicu)*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     *   /(vicc*(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     *   -vicu*(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))))
     *   *exp(vtac(itt)+(1.d0-nphm)*dlog(2.d0))
     *   /(1.d0-vvxt)/(1.d0-vvxtl)
        elseif(itypom.eq.3)then
         gbt=1.d0-(vicc+vicu+wgpm(m)*(viuu-viuc))
     *   *(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))
     *   /((vicc-wgpm(m)*viuc)*(1.d0-vvxp)*(1.d0-vvxpl)
     *   *exp(-vpac(ip))-(vicu+wgpm(m)*viuu)
     *   *(1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*exp(-vpac(ip))))
     *   *exp(vtac(itt)+(1.d0-nphm)*dlog(2.d0))
     *   /(1.d0-vvxt)/(1.d0-vvxtl)
        else
         stop'unknown itypom'
        endif
        if(nphm.eq.nppm(m+1).and.qgran(b10).gt.gbt
     *  .or.nphm+1.eq.nppm(m+1).and.qgran(b10).gt.1.d0+wh*gbt)then
         ntry=0
          goto 24
        endif
       endif

      elseif(jt.eq.4)then                    !>1 cut 'handle' fans
       ntry=0
25     ntry=ntry+1
       if(itt.eq.ia(2).or.ntry.gt.100)then
        nppm(m+1)=npgen(vtac(itt)-vtac0(itt),2,20)
        do i=1,nppm(m+1)
         itypm(i,m+1)=1
         ippm(i,m+1)=itt
        enddo
       else
        nppm(m+1)=npgen(vtac(itt)-vtac0(itt),1,20)
        do i=1,nppm(m+1)
         itypm(i,m+1)=1
         ippm(i,m+1)=itt
        enddo
        do iti=itt+1,ia(2)
         ninc=npgen(vtac(iti)-vtac0(iti),0,20)
         if(ninc.ne.0)then
          nppm(m+1)=nppm(m+1)+ninc
          if(nppm(m+1).gt.legmax)then
           iret=1
           goto 31
          endif
          do i=nppm(m+1)-ninc+1,nppm(m+1)
           itypm(i,m+1)=1
           ippm(i,m+1)=iti
          enddo
         endif
        enddo
        if(nppm(m+1).eq.1)goto 25
       endif

      elseif(jt.eq.3)then                    !1 cut fan
       nppm(m+1)=1
       ippm(1,m+1)=itt
       if(itypom.eq.-1)then             !single cut Pomeron in the 'handle'
        factor=exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)
        wng=(vtacng-vtacpe)*factor/((vtac0(itt)-vtacpe)*factor
     *  -(vtac(itt)-vtac0(itt))*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0
     *  .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
         itypm(1,m+1)=2          !>1 cut Pomerons in the 'handle'
        else
         itypm(1,m+1)=3          !rap-gap in the 'handle'
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       elseif(itypom.eq.2)then          !>1 cut Pomerons in the 'handle'
        factor=exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)
        wng=vtacng*factor/(vtac0(itt)*factor
     *  -(vtac(itt)-vtac0(itt))*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0
     *  .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
         if(qgran(b10).le.vtacpe/vtacng
     *   .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
          itypm(1,m+1)=-1        !single cut Pomeron in the 'handle'
         else
          itypm(1,m+1)=2         !>1 cut Pomerons in the 'handle'
         endif
        else
         itypm(1,m+1)=3          !rap-gap in the 'handle'
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       else                             !rap-gap in the 'handle'
        if(qgran(b10).le.vtacpe/vtacng
     *  .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
         itypm(1,m+1)=-1         !single cut Pomeron in the 'handle'
        else
         itypm(1,m+1)=2          !>1 cut Pomerons in the 'handle'
        endif
       endif

       if(itypm(1,m+1).eq.-1)then     !single cut Pomeron in the 'handle'
        vtlcp=min(vtacpe
     *  ,qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,9))
        if(qgran(b10).le.vtlcp/vtacpe
     *  .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)itypm(1,m+1)=6 !single cut Pomeron
       endif

      elseif(jt.eq.5)then                    !cut 'leg'
       nppm(m+1)=1
       ippm(1,m+1)=itt
       if(itypom.eq.4)then              !single cut Pomeron at the end
        if(xpomm(m+1)*scm.le.sgap**2)stop'=4:xpomm(m+1)*scm<sgap**2'
        factor=exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)
        wng=(vtlcng-vtlcpe)*factor/((vtlc0-vtlcpe)*factor
     *  -(vtlc-vtlc0)*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0)then
         itypm(1,m+1)=7          !>1 cut Pomerons at the end
        else
         itypm(1,m+1)=5          !rap-gap at the end
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       elseif(itypom.eq.5)then          !rap-gap at the end (cut or uncut loop)
        if(qgran(b10).le.vtlcpe/vtlcng
     *  .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
         itypm(1,m+1)=4          !single cut Pomeron at the end
        else
         itypm(1,m+1)=7          !>1 cut Pomerons at the end
        endif
       elseif(itypom.eq.7)then          !>1 cut Pomerons at the end
        factor=exp(-vtac(itt))*(1.d0-vvxt)*(1.d0-vvxtl)
        wng=vtlcng*factor/(vtlc0*factor-(vtlc-vtlc0)*(1.d0-factor))
        if(qgran(b10).le.wng.or.wng.lt.0.d0
     *  .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
         if(qgran(b10).le.vtlcpe/vtlcng
     *   .or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
          itypm(1,m+1)=4         !single cut Pomeron at the end
         else
          itypm(1,m+1)=7         !>1 cut Pomerons at the end
         endif
        else
         itypm(1,m+1)=5          !rap-gap at the end
         wgpm(m+1)=(1.d0-factor)/factor
        endif
       endif

       if(itypm(1,m+1).eq.4)then        !single cut Pomeron at the end
        vtlcp=min(vtlcpe
     *  ,qgfani(xpomm(m+1)*scm,bbt,vvxps,vvxt,vvxtl,iddt(itt),2,9))
        if(qgran(b10).le.vtlcp/vtlcpe
     *  .or.xpomm(m+1)*scm.lt.1.1d0*sgap**3)itypm(1,m+1)=6 !single cut Pomeron
       endif
      endif

      if(nppm(m+1).eq.1.and.itypm(1,m+1).eq.6)then  !record single cut Pomeron
       nptg=nptg+1
       if(nptg.gt.legmax)then
        iret=1
        goto 31
       endif
       xpomti(nptg)=xpomm(m+1)
       vvxti(nptg)=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*(1.d0-vvxt)
     * *(1.d0-vvxtl)*exp(-vpac(ip))
       ipomti(nptg)=itt
       bpomti(nptg)=bbt
       nppm(m+1)=0
       if(debug.ge.4)write (moniou,217)nptg,itt,bbt,xpomti(nptg)
     * ,vvxti(nptg)

      elseif(nppm(m+1).gt.1)then
       i=0
26     i=i+1
       ityp=itypm(i,m+1)
       if(ityp.eq.0)then
        iti=ippm(i,m+1)
        bbi=(xb(iti,1)-xxm(m+1))**2+(xb(iti,2)-yym(m+1))**2
        vvxt=0.d0
        vvxtl=0.d0
        vvxt0=0.d0
        if(ia(2).gt.1)then
         do l=1,ia(2)
          if(l.lt.iti)then
           vvxtl=vvxtl+vtac(l)
          elseif(l.gt.iti)then
           vvxt=vvxt+vtac(l)
           vvxt0=vvxt0+vtac0(l)
          endif
         enddo
        endif
        vvxt=1.d0-exp(-vvxt)
        vvxtl=1.d0-exp(-vvxtl)
        vvxt0=1.d0-exp(-vvxt0)

        vtacng=min(vtac0(iti)
     *  ,qgfani(xpomm(m+1)*scm,bbi,vvxps,vvxt0,vvxtl,iddt(iti),2,4))
        vtacpe=min(vtacng
     *  ,qgfani(xpomm(m+1)*scm,bbi,vvxps,vvxt0,vvxtl,iddt(iti),2,5))
        vtlcp=min(vtacpe
     *  ,qgfani(xpomm(m+1)*scm,bbi,vvxps,vvxt,vvxtl,iddt(iti),2,9))

        aks=qgran(b10)*vtac0(iti)
        if(aks.le.vtlcp.or.xpomm(m+1)*scm.lt.1.1d0*sgap**2)then
         itypm(i,m+1)=6          !single cut Pomeron
        elseif(aks.lt.vtacpe)then
         itypm(i,m+1)=-1         !single cut Pomeron in the 'handle'
        elseif(aks.lt.vtacng)then
         itypm(i,m+1)=2          !>1 cut Pomerons in the 'handle'
        endif

        if(itypm(i,m+1).eq.6)then      !record single cut Pomeron
         nptg=nptg+1
         if(nptg.gt.legmax)then
          iret=1
          goto 31
         endif
         xpomti(nptg)=xpomm(m+1)
         vvxti(nptg)=1.d0-(1.d0-vvxp)*(1.d0-vvxpl)*(1.d0-vvxt)
     *   *(1.d0-vvxtl)*exp(-vpac(ip))
         ipomti(nptg)=iti
         bpomti(nptg)=bbi
         if(debug.ge.4)write (moniou,217)nptg,iti,bbi,xpomti(nptg)
     *   ,vvxti(nptg)
         nppm(m+1)=nppm(m+1)-1
         if(nppm(m+1).ge.i)then
          do l=i,nppm(m+1)
           ippm(l,m+1)=ippm(l+1,m+1)
           itypm(l,m+1)=itypm(l+1,m+1)
          enddo
         endif
         i=i-1
        endif
       endif
       if(i.lt.nppm(m+1))goto 26
      endif

      if(jt.eq.2.and.qgran(b10).lt.(1.d0-exp(-vtac(itt)))*(1.d0-vvxtl)
     */((1.d0-exp(-vtac(itt)))*(1.d0-vvxtl)+2.d0*vvxtl))then
       if(debug.ge.4)write (moniou,212)
       icdts=iddt(itt)
       do icdt=1,2
        iddt(itt)=icdt
        call qgfdf(xxm(m+1),yym(m+1),xpomm(m+1),vpac,vtac
     *  ,vvx,vvxp,vvxt,vvxpl,vvxtl,ip,itt)
        wdt(icdt,itt)=(1.d0-exp(-vtac(itt)))*(1.d0-vvxtl)
       enddo
       iddt(itt)=icdts
      endif

      if(nppm(m+1).ne.0)then
       goto 21
      else
       goto 22
      endif
31    continue
      if(debug.ge.2)write (moniou,219)nppr,nptg,npin,iret

201   format(2x,'qg3pdf - configuration for multi-Pomeron'
     *,'/diffractive contributions'
     */4x,i2,'-th proj. nucleon',2x,i2,'-th targ. nucleon')
202   format(2x,'qg3pdf: problem with initial normalization'
     *,' -> rejection')
203   format(2x,'qg3pdf: normalization of rejection function - ',e10.3)
204   format(2x,'qg3pdf: xpomr=',e10.3,2x,'bbpr=',e10.3,2x,'bbtg=',e10.3
     *,2x,'gb=',e10.3)
205   format(2x,'qg3pdf: xpomr=',e10.3,2x,'bbpr=',e10.3,2x,'bbtg=',e10.3
     *,2x,'xxp=',e10.3,2x,'yyp=',e10.3)
206   format(2x,'qg3pdf: main vertex, nppr0=',i3,2x,'nptg0=',i3)
208   format(2x,'qg3pdf: check',i3,'-th cut fan at ',i2,'-th level,'
     *,' proj. index - ',i3,2x,'b^2=',e10.3)
209   format(2x,'qg3pdf: ',i3,'-th proj. leg, proj. index - ',i3
     *,2x,'b^2=',e10.3,2x,'xpomr=',e10.3,2x,'vvx=',e10.3)
210   format(2x,'qg3pdf: new vertex at ',i3,'-th level')
211   format(2x,'qg3pdf: ',i3,'-th interm. Pomeron'
     */4x,'xpomip=',e10.3,2x,'xpomim=',e10.3
     *,2x,'vvxim=',e10.3,2x,'bpomim=',e10.3)
212   format(2x,'qg3pdf: diffractive cut')
214   format(2x,'qg3pdf: total number of proj. legs - ',i3)
216   format(2x,'qg3pdf: check',i3,'-th cut fan at ',i2,'-th level,'
     *,' targ. index - ',i3,2x,'b^2=',e10.3)
217   format(2x,'qg3pdf: ',i3,'-th targ. leg, targ. index - ',i3
     *,2x,'b^2=',e10.3,2x,'xpomr=',e10.3,2x,'vvx=',e10.3)
219   format(2x,'qg3pdf - end',2x,'number of proj. legs:',i3
     *,2x,'number of targ. legs:',i3
     */4x,'number of interm. Pomerons:',i3,'return flag:',i2)
      return
      end

c------------------------------------------------------------------------
      subroutine qgloolc(sy,xp,bb,icdp,icz,iqq,fan1,fan0)
c-----------------------------------------------------------------------
c qgloolc - unintegrated Pomeron leg eikonal with loops
c sy   - Pomeron mass squared,
c xp   - Pomeron LC momentum,
c bb   - impact parameter squared,
c icz  - hadron class
c iqq=1 - tot
c iqq=2 - soft Pomeron
c iqq=3 - (soft+g)-Pomeron
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      fan0=0.d0
      fan1=0.d0
      if(sy.le.sgap*max(1.d0,xp*sgap))goto 1

      do ix1=1,7
      do mx1=1,2
       xpomr=min(xp,1.d0/sgap)/(sy/sgap/max(1.d0,xp*sgap))
     * **(.5d0+x1(ix1)*(mx1-1.5d0))
       rp=(rq(icdp,icz)-alfp*log(xpomr))*4.d0*.0389d0
       rp1=alfp*log(xpomr*sy/xp)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
      do ix2=1,7
      do mx2=1,2
       z=.5d0+x1(ix2)*(mx2-1.5d0)
       bb0=-rp2*log(z)
      do ix3=1,7
      do mx3=1,2
       phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
       bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2
       bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2

       v1icn=qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,8)
       if(iqq.eq.1)then
        vpl=qglegc(xp/xpomr,xp,bb2,0.d0,icdp,icz,1)
        v1ic0=qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,7)
        v1ic1=min(v1ic0,qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,6))
        v1ic=min(v1ic1,qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,5))
       elseif(iqq.eq.2)then
        vpl=qglegc(xp/xpomr,xp,bb2,0.d0,icdp,icz,0)
        v1ic0=qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,15)
        v1ic1=min(v1ic0,qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,14))
        v1ic=min(v1ic1,qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,13))
       elseif(iqq.eq.3)then
        vpl=qglegc(xp/xpomr,xp,bb2,0.d0,icdp,icz,2)
        v1ic0=qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,7)
        v1ic1=min(v1ic0,qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,6))
        v1ic=min(v1ic1,qgpini(xpomr*sy/xp,bb1,0.d0,0.d0,5))
       else
         vpl=0.d0
         v1ic0=0.d0
         v1ic1=0.d0
         v1ic=0.d0
         stop 'Should no happen in qgloolc !'
       endif
       fan1=fan1+a1(ix1)*a1(ix2)*a1(ix3)/z*rp2
     * *vpl*(v1ic*exp(-2.d0*v1icn)-v1ic1)
       fan0=fan0+a1(ix1)*a1(ix2)*a1(ix3)/z*rp2*vpl*(v1ic1-v1ic0)
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      fan0=fan0/8.d0*pi*r3p/.0389d0/g3p**3
     **dlog(sy/sgap/max(1.d0,xp*sgap))
      fan1=fan1/8.d0*pi*r3p/.0389d0/g3p**3
     **dlog(sy/sgap/max(1.d0,xp*sgap))
1     continue
      if(iqq.eq.1)then
       dleg=qglegc(sy,xp,bb,0.d0,icdp,icz,1)
      elseif(iqq.eq.2)then
       dleg=qglegc(sy,xp,bb,0.d0,icdp,icz,0)
      elseif(iqq.eq.3)then
       dleg=qglegc(sy,xp,bb,0.d0,icdp,icz,2)
      else
       dleg=0.d0
       stop 'Should no happen in qgloolc !'
      endif
      fan0=fan0+dleg
      fan1=fan1+dleg
      return
      end

c------------------------------------------------------------------------
      double precision function qglscr(sy,xp,bb,vvx,icdp,icz,iqq)
c-----------------------------------------------------------------------
c vvx  = 1 - exp[-sum_j chi_targ(j) - sum_{i.ne.I} chi_proj(i)]
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qglscr=0.d0
      if(sy.le.sgap*max(1.d0,xp*sgap))goto 1

      do ix1=1,7
      do mx1=1,2
       xpomr1=min(xp,1.d0/sgap)/(sy/sgap/max(1.d0,xp*sgap))
     * **(.5d0+x1(ix1)*(mx1-1.5d0))
       rp=(rq(icdp,icz)-alfp*log(xpomr1))*4.d0*.0389d0
       rp1=alfp*log(xpomr1*sy/xp)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
       do ix2=1,7
       do mx2=1,2
        z=.5d0+x1(ix2)*(mx2-1.5d0)
        bb0=-rp2*log(z)
       do ix3=1,7
       do mx3=1,2
        phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
        bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2
        bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2

        vicn=qgpini(xpomr1*sy/xp,bb1,0.d0,0.d0,8)
        vpf=qgfani(1.d0/xpomr1,bb2,vvx,0.d0,0.d0,icdp,icz,1)
        if(iqq.eq.1)then
         vpl=qglegc(xp/xpomr1,xp,bb2,vvx,icdp,icz,9)
         vi=qgpini(xpomr1*sy/xp,bb1,0.d0,0.d0,5)
        elseif(iqq.eq.2)then
         vpl=qglegc(xp/xpomr1,xp,bb2,vvx,icdp,icz,10)
         vi=qgpini(xpomr1*sy/xp,bb1,0.d0,0.d0,13)
        elseif(iqq.eq.3)then
         vpl=qglegc(xp/xpomr1,xp,bb2,vvx,icdp,icz,11)
         vi=qgpini(xpomr1*sy/xp,bb1,0.d0,0.d0,5)
        else
         vpl=0.d0
         vi=0.d0
         stop 'Should no happen in qglscr !'
        endif

        dpx=vpl*vi*exp(-2.d0*vicn)
     *  *((1.d0-vvx)**2*exp(-2.d0*vpf)-1.d0)
        qglscr=qglscr+a1(ix1)*a1(ix2)*a1(ix3)*dpx/z*rp2
       enddo
       enddo
       enddo
       enddo
      enddo
      enddo
      qglscr=qglscr/8.d0*pi*r3p/.0389d0/g3p**3
     **dlog(sy/sgap/max(1.d0,xp*sgap))
1     continue
      if(iqq.eq.1)then
       qglscr=qglscr+qglegc(sy,xp,bb,0.d0,icdp,icz,3)
      elseif(iqq.eq.2)then
       qglscr=qglscr+qglegc(sy,xp,bb,0.d0,icdp,icz,5)
      elseif(iqq.eq.3)then
       qglscr=qglscr+qglegc(sy,xp,bb,0.d0,icdp,icz,7)
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qglh(sy,xp,bb,vvx,icdp,icz,iqq)
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/    debug
      common /arr3/     x1(7),a1(7)

      qglh=0.d0
      if(sy.le.max(1.d0,xp*sgap))goto 1

      do ix1=1,7
      do mx1=1,2
       xpomr1=min(xp,1.d0/sgap)/(sy/max(1.d0,xp*sgap))
     * **(.5d0+x1(ix1)*(mx1-1.5d0))
       rp=(rq(icdp,icz)-alfp*log(xpomr1))*4.d0*.0389d0
       rp1=alfp*log(xpomr1*sy/xp)*4.d0*.0389d0
       rp2=rp*rp1/(rp+rp1)
       do ix2=1,7
       do mx2=1,2
        z=.5d0+x1(ix2)*(mx2-1.5d0)
        bb0=-rp2*log(z)
       do ix3=1,7
       do mx3=1,2
        phi=pi*(.5d0+x1(ix3)*(mx3-1.5d0))
        bb1=(dsqrt(bb)*rp1/(rp+rp1)-dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2
        bb2=(dsqrt(bb)*rp/(rp+rp1)+dsqrt(bb0)*cos(phi))**2
     *  +bb0*sin(phi)**2

        vi=qgppdi(xp/xpomr1/sy,iqq)
        vpf=qgfani(1.d0/xpomr1,bb2,vvx,0.d0,0.d0,icdp,icz,1)
        vpl=qglegc(xp/xpomr1,xp,bb2,vvx,icdp,icz,10)

        dpx=vpl*vi*((1.d0-vvx)**2*exp(-2.d0*vpf)-1.d0)
     *  *(xpomr1/xp)**dels*exp(bb2/rp)*rp
        qglh=qglh+a1(ix1)*a1(ix2)*a1(ix3)*dpx
       enddo
       enddo
       enddo
       enddo
      enddo
      enddo
      qglh=qglh/8.d0*pi*r3p/.0389d0/g3p**2*dlog(sy/max(1.d0,xp*sgap))
     */fp(icz)/cd(icdp,icz)/qgppdi(1.d0/sy,iqq)

1     qglh=qglh+1.d0
      return
      end

c------------------------------------------------------------------------
      double precision function qgcutp(sy,xp,xm,bb,vvx
     *,icdp,icdt,icz,iqq)
c-----------------------------------------------------------------------
c qgcutp - unintegrated cut Pomeron eikonal
c sy         - Pomeron mass squared,
c xp,xm      - Pomeron light cone momenta,
c b          - squared impact parameter,
c vvx        - relative strenth of nuclear screening corrections,
c icdp, icdt - proj. and targ. diffractive eigenstates,
c icz        - hadron class
c iqq=1 - total,
c iqq=2 - soft contribution,
c iqq=3  - (soft+gg+gq+qq) contribution
c iqq=4  - (soft+gg+qq) contribution
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr25/ ahv(3)
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      qgcutp=0.d0
      if(sy.le.max(1.d0,xp*sgap)*max(1.d0,xm*sgap))goto 2

      do ix1=1,7
      do mx1=1,2
       xpomr1=xp/max(1.d0,xp*sgap)/(sy/max(1.d0,xp*sgap)
     * /max(1.d0,xm*sgap))**(.5+x1(ix1)*(mx1-1.5))
       rp1=(rq(icdp,icz)-alfp*log(xpomr1))*4.d0*.0389d0
       rp2=(rq(icdt,2)+alfp*log(xpomr1*sy/xp/xm))*4.d0*.0389d0
       rp=rp1*rp2/(rp1+rp2)
      do ib1=1,7
      do mb1=1,2
       z=.5d0+x1(ib1)*(mb1-1.5d0)
       bb0=-rp*dlog(z)
      do ib2=1,7
      do mb2=1,2
       phi=pi*(.5d0+x1(ib2)*(mb2-1.5d0))
       bb1=(dsqrt(bb)*rp1/(rp1+rp2)+dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2
       bb2=(dsqrt(bb)*rp2/(rp1+rp2)-dsqrt(bb0)*cos(phi))**2
     * +bb0*sin(phi)**2

       vpf0=qgfani(1.d0/xpomr1,bb1,vvx,0.d0,0.d0,icdp,icz,1)
       vtf0=qgfani(xpomr1*sy/xp/xm,bb2,vvx,0.d0,0.d0,icdt,2,1)
       n=1
1      n=n+1
       vpf=qgfani(1.d0/xpomr1,bb1,1.d0-(1.d0-vvx)*exp(-vtf0)
     * ,0.d0,0.d0,icdp,icz,1)
       vtf=qgfani(xpomr1*sy/xp/xm,bb2,1.d0-(1.d0-vvx)*exp(-vpf0)
     * ,0.d0,0.d0,icdt,2,1)
       if(abs(1.d0-vpf/vpf0)+abs(1.d0-vtf/vtf0).gt.1.d-2.and.n.le.50)
     * then
        vpf0=vpf
        vtf0=vtf
        goto 1
       endif

       if(iqq.eq.1)then
        vplt=qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,1)
        vtlt=qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,1)
        vpltloop0=min(vplt,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,4))
        vpltloop=min(vpltloop0,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,3))
        vtltloop0=min(vtlt,qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,4))
        vtltloop=min(vtltloop0,qglegc(xpomr1*sy/xp,xm,bb2,0.d0
     *  ,icdt,2,3))
        vpltscr=min(vpltloop,qglegc(xp/xpomr1,xp,bb1
     *  ,1.d0-(1.d0-vvx)*exp(-vtf),icdp,icz,9))
        vtltscr=min(vtltloop,qglegc(xpomr1*sy/xp,xm,bb2
     *  ,1.d0-(1.d0-vvx)*exp(-vpf),icdt,2,9))

        dpx=(vpltscr*vtltloop+vtltscr*vpltloop)
     *  *((1.d0-vvx)**2*exp(-2.d0*vpf-2.d0*vtf)-1.d0)
     *  +vplt*(vtltloop-vtltloop0)+vtlt*(vpltloop-vpltloop0)
       elseif(iqq.eq.2)then
        vpls=qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,0)
        vtls=qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,0)
        vplsloop0=min(vpls,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,6))
        vplsloop=min(vplsloop0,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,5))
        vtlsloop0=min(vtls,qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,6))
        vtlsloop=min(vtlsloop0,qglegc(xpomr1*sy/xp,xm,bb2,0.d0
     *  ,icdt,2,5))
        vplsscr=min(vplsloop,qglegc(xp/xpomr1,xp,bb1
     *  ,1.d0-(1.d0-vvx)*exp(-vtf),icdp,icz,10))
        vtlsscr=min(vtlsloop,qglegc(xpomr1*sy/xp,xm,bb2
     *  ,1.d0-(1.d0-vvx)*exp(-vpf),icdt,2,10))

        dpx=(vplsscr*vtlsloop+vtlsscr*vplsloop)
     *  *((1.d0-vvx)**2*exp(-2.d0*vpf-2.d0*vtf)-1.d0)
     *  +vpls*(vtlsloop-vtlsloop0)+vtls*(vplsloop-vplsloop0)
       elseif(iqq.eq.3)then
        vplq=qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,2)
        vtlt=qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,1)
        vplqloop0=min(vplq,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,8))
        vplqloop=min(vplqloop0
     *  ,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,7))
        vtltloop0=min(vtlt,qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,4))
        vtltloop=min(vtltloop0,qglegc(xpomr1*sy/xp,xm,bb2,0.d0
     *  ,icdt,2,3))
        vplqscr=min(vplqloop,qglegc(xp/xpomr1,xp,bb1
     *  ,1.d0-(1.d0-vvx)*exp(-vtf),icdp,icz,11))
        vtltscr=min(vtltloop,qglegc(xpomr1*sy/xp,xm,bb2
     *  ,1.d0-(1.d0-vvx)*exp(-vpf),icdt,2,9))

        dpx=(vplqscr*vtltloop+vtltscr*vplqloop)
     *  *((1.d0-vvx)**2*exp(-2.d0*vpf-2.d0*vtf)-1.d0)
     *  +vplq*(vtltloop-vtltloop0)+vtlt*(vplqloop-vplqloop0)
       elseif(iqq.eq.4)then
        vplq=qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,2)
        vtlq=qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,2)
        vplqloop0=min(vplq,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,8))
        vplqloop=min(vplqloop0
     *  ,qglegc(xp/xpomr1,xp,bb1,0.d0,icdp,icz,7))
        vtlqloop0=min(vtlq,qglegc(xpomr1*sy/xp,xm,bb2,0.d0,icdt,2,8))
        vtlqloop=min(vtlqloop0,qglegc(xpomr1*sy/xp,xm,bb2,0.d0
     *  ,icdt,2,7))
        vplqscr=min(vplqloop,qglegc(xp/xpomr1,xp,bb1
     *  ,1.d0-(1.d0-vvx)*exp(-vtf),icdp,icz,11))
        vtlqscr=min(vtlqloop,qglegc(xpomr1*sy/xp,xm,bb2
     *  ,1.d0-(1.d0-vvx)*exp(-vpf),icdt,2,11))

        dpx=(vplqscr*vtlqloop+vtlqscr*vplqloop)
     *  *((1.d0-vvx)**2*exp(-2.d0*vpf-2.d0*vtf)-1.d0)
     *  +vplq*(vtlqloop-vtlqloop0)+vtlq*(vplqloop-vplqloop0)
       else
        dpx=0.d0
       endif
       qgcutp=qgcutp+a1(ib1)*a1(ib2)*a1(ix1)/z*rp*dpx
      enddo
      enddo
      enddo
      enddo
      enddo
      enddo
      qgcutp=qgcutp/16.d0*(r3p*pi/.0389d0)/g3p**3
     **dlog(sy/max(1.d0,xp*sgap)/max(1.d0,xm*sgap))

2     continue
      rp=(rq(icdp,icz)+rq(icdt,2)+alfp*log(sy/xp/xm))
      vs=sy**dels*fp(icz)*fp(2)*sigs/rp
     **exp(-bb/rp/4.d0/.0389d0)*cd(icdp,icz)*cd(icdt,2)
      vgg=qgpsh(sy,xp,xm,bb,icdp,icdt,icz,0)
      vqq=qgpomc(sy,xp,xm,bb,0.d0,icdp,icdt,icz,5)
      vqg=qgpsh(sy,xp,xm,bb,icdp,icdt,icz,1)
     */dsqrt(xp)*(1.d0-xp)**(ahv(icz)-ahl(icz))
      vgq=qgpsh(sy,xp,xm,bb,icdp,icdt,icz,2)
     */dsqrt(xm)*(1.d0-xm)**(ahv(2)-ahl(2))
      if(iqq.eq.1)then
       qgcutp=qgcutp+vs+vgg+vqg+vgq+vqq
      elseif(iqq.eq.2)then
       qgcutp=qgcutp+vs
      elseif(iqq.eq.3)then
       qgcutp=qgcutp+vs+vgg+vgq+vqq
      elseif(iqq.eq.4)then
       qgcutp=qgcutp+vs+vgg+vqq
      endif
      return
      end

c=============================================================================
      double precision function qgpsh(sy,xpp,xpm,bb,icdp,icdt,icz,iqq)
c-----------------------------------------------------------------------------
c qgpsh - unintegrated semihard Pomeron eikonal
c sy         - Pomeron mass squared,
c xpp, xpm   - Pomeron LC momenta,
c b          - impact parameter,
c icdp, icdt - proj. and targ. diffractive eigenstates,
c icz        - hadron class,
c iqq        - type of the hard interaction (0-gg, 1-q_vg, 2-gq_v)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /arr3/   x1(7),a1(7)
      common /qgdebug/  debug

      if(debug.ge.3)write (moniou,201)sy,xpp,xpm,b,vvx0,icdp,icdt
     *,icz,iqq
      qgpsh=0.d0
      s2min=4.d0*fqscal*qt0               !energy threshold for hard interaction
      if(s2min/sy.ge.1.d0)then
       if(debug.ge.4)write (moniou,202)qgpsh
       return
      endif

      if(iqq.ne.2)then
       icv=icz
       icq=2
       xp=xpp
       xm=xpm
       icdv=icdp
       icdq=icdt
      else
       icv=2
       icq=icz
       xp=xpm
       xm=xpp
       icdq=icdp
       icdv=icdt
      endif

      xmin=(s2min/sy)**(delh-dels)
      do i=1,7
      do m=1,2
       z1=(.5d0*(1.d0+xmin-(2*m-3)*x1(i)*(1.d0-xmin)))
     * **(1.d0/(delh-dels))
       ww=z1*sy
       sjqq=qgjit(qt0,qt0,ww,2,2)
       sjqg=qgjit(qt0,qt0,ww,1,2)
       sjgg=qgjit(qt0,qt0,ww,1,1)

       if(iqq.eq.0)then                                !gg-Pomeron
        st2=0.d0
        do j=1,7
        do k=1,2
         xx=.5d0*(1.d0+x1(j)*(2*k-3))
         xph=z1**xx
         xmh=z1/xph

         glu1=qgppdi(xph,0)
         sea1=qgppdi(xph,1)
         glu2=qgppdi(xmh,0)
         sea2=qgppdi(xmh,1)
         st2=st2+a1(j)*(glu1*glu2*sjgg+(glu1*sea2+glu2*sea1)*sjqg
     *   +sea1*sea2*sjqq)
        enddo
        enddo
        rh=rq(icdp,icz)+rq(icdt,2)-alfp*dlog(xpp*xpm*z1)
        qgpsh=qgpsh-a1(i)*dlog(z1)/z1**delh*st2
     *  *exp(-bb/rh/4.d0/.0389d0)/rh

       else                                !qg-Pomeron
        xmh=z1
        glu=qgppdi(xmh,0)
        sea=qgppdi(xmh,1)
        rh=rq(icdp,icz)+rq(icdt,2)-alfp*dlog(xm*xmh)

        fst=(glu*sjqg+sea*sjqq)
     *  *(qggrv(xp,qt0,icv,1)+qggrv(xp,qt0,icv,2))/dsqrt(xp)
     *  *exp(-bb/rh/4.d0/.0389d0)/rh
        qgpsh=qgpsh+a1(i)/z1**delh*fst
       endif
      enddo
      enddo
      qgpsh=qgpsh*(1.d0-xmin)/(delh-dels)
      if(iqq.eq.0)then
       qgpsh=qgpsh*rr**2*fp(icz)*fp(2)*factk/2.d0*pi
     * *cd(icdp,icz)*cd(icdt,2)
      else
       qgpsh=qgpsh*rr*fp(icq)*factk/4.d0
     * *cd(icdp,icz)*cd(icdt,2)
      endif
      if(debug.ge.4)write (moniou,202)qgpsh

201   format(2x,'qgpsh - unintegrated semihard Pomeron eikonal:'
     */4x,'sy=',e10.3,2x,'xpp=',e10.3,2x,'xpm=',e10.3,2x,'b=',e10.3
     */4x,'vvx0=',e10.3,2x,'icdp=',i1,2x,'icdt=',i1,2x,'icz=',i1
     *,2x,'iqq=',i1)
202   format(2x,'qgpsh=',e10.3)
      return
      end

c------------------------------------------------------------------------
      double precision function qglegc(sy,xp,bb,vvx,icdp,icz,iqq)
c-----------------------------------------------------------------------
c qglegc - interpolation of cut Pomeron leg eikonal
c sy   - Pomeron mass squared,
c xp   - Pomeron LC momentum,
c bb   - squared impact parameter,
c vvx - relative strenth of screening corrections (0<vvx<1),
c icdp - diffractive eigenstate for the hadron,
c icz  - hadron class
c iqq=0  - soft Pomeron,
c iqq=1  - total Pomeron,
c iqq=2  - (soft+g)-Pomeron,
c iqq=3  - total loop,
c iqq=4  - total loop with single Pomeron end,
c iqq=5  - soft loop,
c iqq=6  - soft loop with single Pomeron end,
c iqq=7  - (soft+g)-loop,
c iqq=8  - (soft+g)-loop with single Pomeron end,
c iqq=9  - total screened,
c iqq=10 - soft screened,
c iqq=11 - (soft+g)-screened
c iqq=12 - g-distribution,
c iqq=13 - q-distribution
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wj(3),wi(3),wz(3)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr25/ ahv(3)
      common /qgarr26/ factk,fqscal
      common /qgarr35/ qlegc0(51,10,11,6,8),qlegc(51,10,11,11,30)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.3)write (moniou,201)sy,xp,bb,vvx,icdp,icz,iqq

      qglegc=0.d0
      clegm=0.d0
      rp=(rq(icdp,icz)+alfp*log(max(1.d0,sy/xp)))*4.d0*.0389d0
      z=exp(-bb/rp)
      if(iqq.eq.0.or.iqq.le.11.and.sy.le.sgap*max(1.d0,xp*sgap)
     *  .or.iqq.gt.11.and.sy.le.max(1.d0,xp*sgap))then
       if(iqq.le.11)then
        qglegc=sy**dels*fp(icz)*sigs*g3p/rp*4.d0*.0389d0*z*cd(icdp,icz)
       else
        qglegc=qgppdi(1.d0/sy,iqq-12)
       endif
       if(debug.ge.4)write (moniou,202)qglegc
       return
      endif

      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-dlog(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      if(iqq.le.11)then
       yl=max(0.d0,dlog(sy/xp/sgap**2)/dlog(spmax/sgap**2))*50.d0+1.d0
      else
       yl=max(0.d0,dlog(sy/xp/sgap)/dlog(spmax/sgap))*50.d0+1.d0
      endif
      k=max(1,int(yl))
      k=min(k,49)
      wk(2)=yl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)
      iymax=3

      if(xp.lt..2d0)then
       if(iqq.le.11)then
        xl=6.d0-5.d0*log(5.d0*xp)/log(5.d0*xp*sgap/sy)
       elseif(sy.gt.1.01d0*xp*sgap)then
        xl=6.d0-5.d0*log(5.d0*xp)/log(xp*sgap/sy)
       else
        xl=1.d0
       endif
      else
       xl=5.d0*xp+5.d0
      endif
      i=min(8,int(xl))
      i=max(1,i)
      if(i.eq.5)i=4
      wi(2)=xl-i
      wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
      wi(1)=1.d0-wi(2)+wi(3)
      wi(2)=wi(2)-2.d0*wi(3)
      ixmax=3

      if(iqq.lt.9)then
       do k1=1,iymax
        k2=k+k1-1
       do i1=1,ixmax
        i2=i+i1-1
       do l1=1,izmax
        l2=jz+l1-1
        qglegc=qglegc+qlegc0(k2,i2,l2,icdp+2*(icz-1),iqq)
     *  *wk(k1)*wi(i1)*wz(l1)
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do k1=1,iymax
         k2=k+k1-1
        do i1=1,ixmax
         i2=i+i1-1
         clegm=clegm+qlegc0(k2,i2,1,icdp+2*(icz-1),iqq)*wk(k1)*wi(i1)
        enddo
        enddo
        qglegc=min(qglegc,clegm)
       endif
      else
       vl=max(1.d0,vvx*10.d0+1.d0)
       if(vl.lt.2.d0)then
        j=1
        wj(2)=vl-j
        wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
        wj(1)=1.d0-wj(2)+wj(3)
        wj(2)=wj(2)-2.d0*wj(3)
        ivmax=3
       else
        j=min(int(vl),10)
        wj(2)=vl-j
        wj(1)=1.d0-wj(2)
        ivmax=2
       endif

       do l1=1,izmax
        l2=jz+l1-1
       do j1=1,ivmax
        j2=j+j1-1
       do i1=1,ixmax
        i2=i+i1-1
       do k1=1,iymax
        k2=k+k1-1
        qglegc=qglegc+qlegc(k2,i2,j2,l2,icdp+2*(icz-1)+6*(iqq-9))
     *  *wk(k1)*wi(i1)*wz(l1)*wj(j1)
       enddo
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do j1=1,ivmax
         j2=j+j1-1
        do i1=1,ixmax
         i2=i+i1-1
        do k1=1,iymax
         k2=k+k1-1
         clegm=clegm+qlegc(k2,i2,j2,1,icdp+2*(icz-1)+6*(iqq-9))
     *   *wk(k1)*wi(i1)*wj(j1)
        enddo
        enddo
        enddo
        qglegc=min(qglegc,clegm)
       endif
      endif
      if(iqq.le.11)then
       qglegc=exp(qglegc)*qgls(sy,xp,bb,icdp,icz)
      else
       qglegc=exp(qglegc)*qgppdi(1.d0/sy,iqq-12)
      endif
      if(debug.ge.4)write (moniou,202)qglegc

201   format(2x,'qglegc - interpolation of Pomeron leg eikonal:'
     */4x,'sy=',e10.3,2x,'xp=',e10.3,2x,'b^2=',e10.3,2x,'vvx=',e10.3
     *,2x,'icdp=',i1,2x,'icz=',i1,2x,'iqq=',i1)
202   format(2x,'qglegc=',e10.3)
      return
      end

c=============================================================================
      double precision function qgpomc(sy,xp,xm,bb,vvx
     *,icdp,icdt,icz,iqq)
c-----------------------------------------------------------------------
c qgpomc - unintegrated cut Pomeron eikonal
c sy         - Pomeron mass squared,
c xp,xm      - Pomeron light cone momenta,
c bb         - squared impact parameter,
c vvx        - relative strenth of nuclear screening corrections,
c icdp, icdt - proj. and targ. diffractive eigenstates,
c icz        - hadron class
c iqq=1 - total,
c iqq=2 - soft contribution,
c iqq=3 - qg contribution
c iqq=4 - gq contribution
c iqq=5 - qq contribution
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wi(3),wj(3),wz(3),wm(3)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr19/ ahl(3)
      common /qgarr20/ spmax
      common /qgarr25/ ahv(3)
      common /qgarr26/ factk,fqscal
      common /qgarr38/ qpomc(11,100,11,11,48)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.3)write (moniou,201)sy,xp,xm,bb,vvx
     *,icdp,icdt,icz,iqq

      qgpomc=0.d0
      pomm=0.d0
      if(iqq.eq.5)then                          !qq contribution
       s2min=4.d0*fqscal*qt0
       if(sy.gt.1.001d0*s2min.and.xp.lt..99d0.and.xm.lt..99d0)then
        sj=qgjit(qt0,qt0,sy,2,2)
        qgpomc=sj*factk*(qggrv(xp,qt0,icz,1)+qggrv(xp,qt0,icz,2))
     *  *(qggrv(xm,qt0,2,1)+qggrv(xm,qt0,2,2))/xp/xm
     *  *(1.d0-xp)**(ahv(icz)-ahl(icz))*(1.d0-xm)**(ahv(2)-ahl(2))
     *  *exp(-bb/(4.d0*.0389d0*(rq(icdp,icz)+rq(icdt,2))))
     *  /(8.d0*pi*(rq(icdp,icz)+rq(icdt,2)))*cd(icdp,icz)*cd(icdt,2)
       endif
       if(debug.ge.4)write (moniou,202)qgpomc
       return
      endif

      rp=(rq(icdp,icz)+rq(icdt,2)+alfp*log(sy/xp/xm))*4.d0*.0389d0
      z=exp(-bb/rp)
      if(sy.le.max(1.d0,xp*sgap)*max(1.d0,xm*sgap)*1.01d0)then
       qgpomc=sy**dels*fp(icz)*fp(2)*sigs*z/rp
     * *4.d0*.0389d0*cd(icdp,icz)*cd(icdt,2)
       return
      endif

      if(z.gt..2d0)then
       zz=5.d0*z+6.d0
      else
       zz=(-bb/rp-dlog(0.2d0))/2.d0+7.d0
      endif
      jz=min(9,int(zz))
      jz=max(1,jz)
      if(zz.lt.1.d0)then
       wz(2)=zz-jz
       wz(1)=1.d0-wz(2)
       izmax=2
      else
       if(jz.eq.6)jz=5
       wz(2)=zz-jz
       wz(3)=wz(2)*(wz(2)-1.d0)*.5d0
       wz(1)=1.d0-wz(2)+wz(3)
       wz(2)=wz(2)-2.d0*wz(3)
       izmax=3
      endif

      yl=max(0.d0,dlog(sy/xp/xm/sgap**2)
     */dlog(spmax/sgap**2))*10.d0+1.d0
      k=max(1,int(yl))
      k=min(k,9)
      wk(2)=yl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)
      iymax=3

      if(xp.lt..2d0)then
       xl1=6.d0-5.d0*log(5.d0*xp)/log(5.d0*sgap*xp*xm/sy)
      else
       xl1=5.d0*xp+5.d0
      endif
      i=min(8,int(xl1))
      i=max(1,i)
      if(i.eq.5)i=4
      wi(2)=xl1-i
      wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
      wi(1)=1.d0-wi(2)+wi(3)
      wi(2)=wi(2)-2.d0*wi(3)
      ix1max=3

      if(sgap/sy*xm.gt..99d0)then
       j=1
       wj(1)=1.d0
       ix2max=1
      else
       if(xm.lt..2d0)then
        xl2=6.d0-5.d0*log(5.d0*xm)/log(sgap/sy*xm)
       else
        xl2=5.d0*xm+5.d0
       endif
       j=min(8,int(xl2))
       j=max(1,j)
       if(j.eq.5)j=4
       wj(2)=xl2-j
       wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
       wj(1)=1.d0-wj(2)+wj(3)
       wj(2)=wj(2)-2.d0*wj(3)
       ix2max=3
      endif

      ml=icdp+2*(icdt-1)+4*(icz-1)+12*(iqq-1)
      if(vvx.eq.0.d0)then                     !hadron-proton collision
       do l1=1,izmax
        l2=jz+l1-1
       do j1=1,ix2max
        j2=j+j1-2
       do i1=1,ix1max
        i2=i+i1-1
       do k1=1,iymax
        k2=k+k1-1
        qgpomc=qgpomc+qpomc(k2,i2+10*j2,l2,1,ml)
     *  *wk(k1)*wi(i1)*wj(j1)*wz(l1)
       enddo
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do j1=1,ix2max
         j2=j+j1-2
        do i1=1,ix1max
         i2=i+i1-1
        do k1=1,iymax
         k2=k+k1-1
         pomm=pomm+qpomc(k2,i2+10*j2,1,1,ml)*wk(k1)*wi(i1)*wj(j1)
        enddo
        enddo
        enddo
        qgpomc=min(qgpomc,pomm)
       endif

      else                                    !hA (AA) collision
       vl=max(1.d0,vvx*10.d0+1.d0)
       if(vl.lt.2.d0)then
        m=1
        wm(2)=vl-m
        wm(3)=wm(2)*(wm(2)-1.d0)*.5d0
        wm(1)=1.d0-wm(2)+wm(3)
        wm(2)=wm(2)-2.d0*wm(3)
        ivmax=3
       else
        m=min(int(vl),10)
        wm(2)=vl-m
        wm(1)=1.d0-wm(2)
        ivmax=2
       endif

       do m1=1,ivmax
        m2=m+m1-1
       do l1=1,izmax
        l2=jz+l1-1
       do j1=1,ix2max
        j2=j+j1-2
       do i1=1,ix1max
        i2=i+i1-1
       do k1=1,iymax
        k2=k+k1-1
        qgpomc=qgpomc+qpomc(k2,i2+10*j2,l2,m2,ml)
     *  *wk(k1)*wi(i1)*wj(j1)*wz(l1)*wm(m1)
       enddo
       enddo
       enddo
       enddo
       enddo
       if(zz.lt.1.d0)then
        do m1=1,ivmax
         m2=m+m1-1
        do j1=1,ix2max
         j2=j+j1-2
        do i1=1,ix1max
         i2=i+i1-1
        do k1=1,iymax
         k2=k+k1-1
         pomm=pomm+qpomc(k2,i2+10*j2,1,m2,ml)
     *   *wk(k1)*wi(i1)*wj(j1)*wm(m1)
        enddo
        enddo
        enddo
        enddo
        qgpomc=min(qgpomc,pomm)
       endif
      endif
      qgpomc=exp(qgpomc)*z
      if(debug.ge.4)write (moniou,202)qgpomc

201   format(2x,'qgpomc - unintegrated cut Pomeron eikonal:'
     */4x,'sy=',e10.3,2x,'xp=',e10.3,2x,'xm=',e10.3,2x,'b^2=',e10.3
     */4x,'vvx=',e10.3,2x,'icdp=',i1,2x,'icdt=',i1,2x,'icz=',i1
     *,2x,'iqq=',i1)
202   format(2x,'qgpomc=',e10.3)
      return
      end

c=============================================================================
      subroutine qgsha(nbpom,ncola,ncolb,iret)
c-----------------------------------------------------------------------------
c qgsha - inelastic interaction (energy sharing and particle production)
c nbpom - number of Pomeron blocks (nucleon(hadron)-nucleon collisions),
c ncola - number of inel.-wounded proj. nucleons,
c ncolb - number of inel.-wounded targ. nucleons
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208,npbmax=1000,npnmax=900,npmax=900
     *,legmax=900,njmax=50000)
      dimension wppr0(iapmax),wmtg0(iapmax),wppr1(iapmax),wmtg1(iapmax)
     *,wppr2(iapmax),wmtg2(iapmax),izp(iapmax),izt(iapmax)
     *,ila(iapmax),ilb(iapmax),lva(iapmax),lvb(iapmax)
     *,lqa0(iapmax),lqb0(iapmax),ncola(iapmax),ncolb(iapmax)
     *,ncola0(iapmax),ncolb0(iapmax)
     *,xpomp0(npnmax,iapmax),xpomt0(npnmax,iapmax)
     *,xpopin0(npmax,npbmax),xpomin0(npmax,npbmax)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr6/  pi,bm,amws
      common /qgarr7/  xa(iapmax,3),xb(iapmax,3),b
      common /qgarr9/  iwp(iapmax),iwt(iapmax),lqa(iapmax),lqb(iapmax)
     *,iprcn(iapmax),itgcn(iapmax),ias(npbmax),ibs(npbmax),nqs(npbmax)
     *,npompr(npbmax),npomtg(npbmax),npomin(npbmax),nnpr(npmax,npbmax)
     *,nntg(npmax,npbmax),ilpr(legmax,npbmax),iltg(legmax,npbmax)
     *,lnpr(legmax,npbmax),lntg(legmax,npbmax)
     *,nbpi(npnmax,iapmax),nbti(npnmax,iapmax),idnpi(npnmax,iapmax)
     *,idnti(npnmax,iapmax),nppi(npnmax,iapmax),npti(npnmax,iapmax)
     *,nlpi(npnmax,iapmax),nlti(npnmax,iapmax)
      common /qgarr11/ b10
      common /qgarr12/ nsp
      common /qgarr13/ nsf,iaf(iapmax)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr23/ bbpom(npbmax),vvxpom(npbmax)
     *,bpompr(npnmax,iapmax),bpomtg(npnmax,iapmax)
     *,vvxpr(npnmax,iapmax),vvxtg(npnmax,iapmax)
     *,xpompr(npnmax,iapmax),xpomtg(npnmax,iapmax)
     *,xpopin(npmax,npbmax),xpomin(npmax,npbmax),vvxin(npmax,npbmax)
     *,bpomin(npmax,npbmax)
      common /qgarr26/ factk,fqscal
      common /qgarr37/ eqj(4,njmax),iqj(njmax),ncj(2,njmax),nj
      common /qgarr40/ xppr(npnmax,iapmax),xmtg(npnmax,iapmax)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      external qgran

      if(debug.ge.1)write (moniou,201)nbpom             !so161205
      nsp0=nsp

      do j=1,ia(1)
       if(lqa(j).ne.0)then
        do i=1,lqa(j)
         if(idnpi(i,j).ne.0)xpomp0(i,j)=xpompr(i,j)
        enddo
       endif
      enddo
      do j=1,ia(2)
       if(lqb(j).ne.0)then
        do i=1,lqb(j)
         if(idnti(i,j).ne.0)xpomt0(i,j)=xpomtg(i,j)
        enddo
       endif
      enddo
      if(nbpom.ne.0)then
       do nb=1,nbpom                            !loop over collisions
        if(npomin(nb).ne.0)then
         do np=1,npomin(nb)         !loop over interm. Pomerons in the collision
          xpopin0(np,nb)=xpopin(np,nb)
          xpomin0(np,nb)=xpomin(np,nb)
         enddo
        endif
       enddo
      endif
      iret=0
      nret=0

1     nsp=nsp0
      nj=0

      if(iret.ne.0)then             !rejection during energy-sharing
       nret=nret+1
       if(nret.gt.100)return        !too many rejections -> redo configuration
      endif

      do j=1,ia(1)
       if(lqa(j).ne.0)then
        do i=1,lqa(j)
         if(idnpi(i,j).ne.0)xpompr(i,j)=xpomp0(i,j)
        enddo
       endif
      enddo
      do j=1,ia(2)
       if(lqb(j).ne.0)then
        do i=1,lqb(j)
         if(idnti(i,j).ne.0)xpomtg(i,j)=xpomt0(i,j)
        enddo
       endif
      enddo
      if(nbpom.ne.0)then
       do nb=1,nbpom                            !loop over collisions
        if(npomin(nb).ne.0)then
         do np=1,npomin(nb)         !loop over interm. Pomerons in the collision
          xpopin(np,nb)=xpopin0(np,nb)
          xpomin(np,nb)=xpomin0(np,nb)
         enddo
        endif
       enddo
      endif

c-------------------------------------------------
c initial nucleon (hadron) types
      if(ia(1).ne.1)then
       do i=1,ia(1)
        izp(i)=int(2.5d0+qgran(b10))   !i-th projectile nucleon type
       enddo
      else
       izp(1)=icp                      !projectile hadron type
      endif
      if(ia(2).ne.1)then
       do i=1,ia(2)
        izt(i)=int(2.5d0+qgran(b10))   !i-th target nucleon type
       enddo
      else
       izt(1)=2                        !target proton
      endif

      do i=1,ia(1)
       lqa0(i)=lqa(i)
       lva(i)=0
       ncola0(i)=ncola(i)
      enddo
      do i=1,ia(2)
       lqb0(i)=lqb(i)
       lvb(i)=0
       ncolb0(i)=ncolb(i)
      enddo

c-------------------------------------------------
c energy-momentum sharing between Pomerons
      if(nbpom.ne.0)then
       if(debug.ge.1)write (moniou,202)
       call qgprox(0)        !initial x-configuration
       gbl0=qgweix(nbpom)    !log-weight for the initial x-configuration
       nrej=0
       nchange=0
       gbnorm=.1d0
       gbhmax=-1000.d0

2      continue
       call qgprox(1)        !proposed x-configuration
       gbl=qgweix(nbpom)     !log-weight for the proposed x-configuration
       gbh=gbl-gbl0-gbnorm   !log of acceptance probability
       gbhmax=max(gbhmax,gbh)

       if(debug.ge.5)write (moniou,203)gbh,nrej,nchange
       if(gbh.lt.-50.d0.or.qgran(b10).gt.exp(gbh))then
        nrej=nrej+1
        if(nrej.gt.100)then               !too many rejections
         nrej=0
         nchange=nchange+1
         gbnorm=gbnorm+gbhmax+.5d0        !new normalization of acceptance
         gbhmax=-1000.d0
         if(debug.ge.4)write (moniou,204)nchange
        endif
        goto 2                            !rejection
       endif
      endif

c-------------------------------------------------
c leading remnant LC momenta
      if(debug.ge.1)write (moniou,205)
      do i=1,ia(1)                        !loop over proj. nucleons
       wppr0(i)=wp0
       wppr1(i)=0.d0
       wppr2(i)=0.d0
       if(lqa(i).ne.0)then
        do l=1,lqa(i)                     !loop over constituent partons
         wppr0(i)=wppr0(i)-wp0*xppr(l,i)  !subtract Pomeron LC momentum
         if(wppr0(i).lt.0.d0)then
          wppr0(i)=0.d0
         endif
        enddo
       endif
      enddo
      do i=1,ia(2)                        !loop over targ. nucleons
       wmtg0(i)=wm0
       wmtg1(i)=0.d0
       wmtg2(i)=0.d0
       if(lqb(i).ne.0)then
        do l=1,lqb(i)                     !loop over constituent partons
         wmtg0(i)=wmtg0(i)-wm0*xmtg(l,i)  !subtract Pomeron LC momentum
         if(wmtg0(i).lt.-1.d-15)stop'w^-<0!!!'
         wmtg0(i)=max(0.d0,wmtg0(i))
        enddo
       endif
      enddo

c-------------------------------------------------
c momentum conservation (correction for 3p-vertexes)
      if(debug.ge.1)write (moniou,206)
      if(nbpom.ne.0)then
       do nb=1,nbpom                            !loop over collisions
        ip=ias(nb)                              !proj. index
        it=ibs(nb)                              !targ. index
        if(nqs(nb).ne.0)then
         do np=1,nqs(nb)             !loop over single Pomerons in the collision
          lnp=nnpr(np,nb)                       !proj. constituent parton index
          lnt=nntg(np,nb)                       !targ. constituent parton index
          wppr1(ip)=wppr1(ip)+xppr(lnp,ip)*wp0  !count Pomeron LC momentum
          wmtg1(it)=wmtg1(it)+xmtg(lnt,it)*wm0  !count Pomeron LC momentum
         enddo
        endif
        if(npomin(nb).ne.0)then
         do np=1,npomin(nb)         !loop over interm. Pomerons in the collision
          xpp=xpopin(np,nb)
          xpm=xpomin(np,nb)
          if(xpp*xpm*scm.gt.1.d0)then
           wppr2(ip)=wppr2(ip)+xpp*wp0          !count Pomeron LC momentum
           wmtg2(it)=wmtg2(it)+xpm*wm0          !count Pomeron LC momentum
          else
           xpopin(np,nb)=0.d0
           xpomin(np,nb)=0.d0
          endif
         enddo
        endif
        if(npompr(nb).ne.0)then
         do np=1,npompr(nb)       !loop over proj. leg Pomerons in the collision
          ipp=ilpr(np,nb)                       !proj. index
          lnp=lnpr(np,nb)                       !proj. constituent parton index
          xpp=xppr(lnp,ipp)
          xpm=xpompr(lnp,ipp)
          if(xpp*xpm*scm.gt.1.d0)then
           wppr1(ipp)=wppr1(ipp)+xpp*wp0        !count Pomeron LC momentum
           wmtg2(it)=wmtg2(it)+xpm*wm0          !count Pomeron LC momentum
          else
           xppr(lnp,ipp)=0.d0
           xpompr(lnp,ipp)=0.d0
          endif
         enddo
        endif
        if(npomtg(nb).ne.0)then
         do np=1,npomtg(nb)       !loop over targ. leg Pomerons in the collision
          itt=iltg(np,nb)                       !targ. index
          lnt=lntg(np,nb)                       !targ. constituent parton index
          xpp=xpomtg(lnt,itt)
          xpm=xmtg(lnt,itt)
          if(xpp*xpm*scm.gt.1.d0)then
           wppr2(ip)=wppr2(ip)+xpp*wp0                !count Pomeron LC momentum
           wmtg1(itt)=wmtg1(itt)+xpm*wm0        !count Pomeron LC momentum
          else
           xmtg(lnt,itt)=0.d0
           xpomtg(lnt,itt)=0.d0
          endif
         enddo
        endif
       enddo
      endif

      do ip=1,ia(1)
       if(wppr1(ip)+wppr2(ip).ne.0.d0)then
        if(lqa(ip).ne.0)then
         do i=1,lqa(ip)
          xppr(i,ip)=xppr(i,ip)*(wp0-wppr0(ip)) !renorm. for const. partons
     *    /(wppr1(ip)+wppr2(ip))
         enddo

         do nb=1,nbpom
          if(ias(nb).eq.ip.and.npomtg(nb)+npomin(nb).ne.0)then
           if(npomin(nb).ne.0)then
            do np=1,npomin(nb)
             xpopin(np,nb)=xpopin(np,nb)*(wp0-wppr0(ip))
     *       /(wppr1(ip)+wppr2(ip))
            enddo
           endif
           if(npomtg(nb).ne.0)then
            do np=1,npomtg(nb)
             itt=iltg(np,nb)
             lnt=lntg(np,nb)
             xpomtg(lnt,itt)=xpomtg(lnt,itt)*(wp0-wppr0(ip))
     *       /(wppr1(ip)+wppr2(ip))
            enddo
           endif
          endif
         enddo

        elseif(wppr2(ip).gt.wp0)then
         wpt=wp0/sgap/2.d0*4.d0**qgran(b10)
         do nb=1,nbpom
          if(ias(nb).eq.ip.and.npomtg(nb)+npomin(nb).ne.0)then
           if(npomin(nb).ne.0)then
            do np=1,npomin(nb)
             xpopin(np,nb)=xpopin(np,nb)*wpt/wppr2(ip)
            enddo
           endif
           if(npomtg(nb).ne.0)then
            do np=1,npomtg(nb)
             itt=iltg(np,nb)
             lnt=lntg(np,nb)
             xpomtg(lnt,itt)=xpomtg(lnt,itt)*wpt/wppr2(ip)
            enddo
           endif
          endif
         enddo
         wppr0(ip)=wp0-wpt
        else
         wppr0(ip)=wp0-wppr2(ip)
        endif
       endif
      enddo

      do it=1,ia(2)
       if(wmtg1(it)+wmtg2(it).ne.0.d0)then
        if(lqb(it).ne.0)then
         do i=1,lqb(it)
          xmtg(i,it)=xmtg(i,it)*(wm0-wmtg0(it))/(wmtg1(it)+wmtg2(it))
         enddo

         do nb=1,nbpom
          if(ibs(nb).eq.it.and.npompr(nb)+npomin(nb).ne.0)then
           if(npomin(nb).ne.0)then
            do np=1,npomin(nb)
             xpomin(np,nb)=xpomin(np,nb)*(wm0-wmtg0(it))
     *       /(wmtg1(it)+wmtg2(it))
            enddo
           endif
           if(npompr(nb).ne.0)then
            do np=1,npompr(nb)
             ipp=ilpr(np,nb)
             lnp=lnpr(np,nb)
             xpompr(lnp,ipp)=xpompr(lnp,ipp)*(wm0-wmtg0(it))
     *       /(wmtg1(it)+wmtg2(it))
            enddo
           endif
          endif
         enddo

        elseif(wmtg2(it).gt.wm0)then
         wmt=wm0/sgap/2.d0*4.d0**qgran(b10)
         do nb=1,nbpom
          if(ibs(nb).eq.it.and.npompr(nb)+npomin(nb).ne.0)then
           if(npomin(nb).ne.0)then
            do np=1,npomin(nb)
             xpomin(np,nb)=xpomin(np,nb)*wmt/wmtg2(it)
            enddo
           endif
           if(npompr(nb).ne.0)then
            do np=1,npompr(nb)
             ipp=ilpr(np,nb)
             lnp=lnpr(np,nb)
             xpompr(lnp,ipp)=xpompr(lnp,ipp)*wmt/wmtg2(it)
            enddo
           endif
          endif
         enddo
         wmtg0(it)=wm0-wmt
        else
         wmtg0(it)=wm0-wmtg2(it)
        endif
       endif
      enddo

c-------------------------------------------------
c treatment of low mass diffraction
      if(debug.ge.1)write (moniou,207)
      do ip=1,ia(1)                        !loop over proj. nucleons
       if(iwp(ip).eq.2)then                !diffraction dissociation
        it=iprcn(ip)
        if(debug.ge.2)write (moniou,208)ip,it
        if(iwt(it).eq.2)then
         call qgdifr(wppr0(ip),wmtg0(it),izp(ip),izt(it),-2,-2,iret)
        elseif(iwt(it).eq.-1)then
         call qgdifr(wppr0(ip),wmtg0(it),izp(ip),izt(it),-2,0,iret)
        elseif(iwt(it).gt.0)then
         call qgdifr(wppr0(ip),wmtg0(it),izp(ip),izt(it),-2,-1,iret)
        else
         stop'wrong connection for diffraction'
        endif
        if(iret.eq.1)goto 1
       endif
      enddo

      do it=1,ia(2)                        !loop over targ. nucleons
       if(iwt(it).eq.2)then                !diffraction dissociation
        ip=itgcn(it)
        if(debug.ge.2)write (moniou,209)it,ip
        if(iwp(ip).eq.-1)then
         call qgdifr(wppr0(ip),wmtg0(it),izp(ip),izt(it),0,-2,iret)
        elseif(iwp(ip).gt.0.and.iwp(ip).ne.2)then
         call qgdifr(wppr0(ip),wmtg0(it),izp(ip),izt(it),-1,-2,iret)
        endif
        if(iret.eq.1)goto 1
       endif
      enddo

c-------------------------------------------------
c particle production for all cut Pomerons
      s2min=4.d0*fqscal*qt0       !threshold energy for a hard process
      if(nbpom.ne.0)then
       if(debug.ge.1)write (moniou,210)
       do npb=1,nbpom                            !loop over collisions
        ip=ias(npb)                              !proj. index
        it=ibs(npb)                              !targ. index
        icdp=iddp(ip)                            !proj. diffr. eigenstate
        icdt=iddt(it)                            !targ. diffr. eigenstate
        bbp=bbpom(npb)                           !b^2 between proj. and targ.
        vvx=vvxpom(npb)                          !nuclear screening factor
        if(debug.ge.1)write (moniou,211)npb,ip,it,bbp,vvx,nqs(npb)
     *  ,npomin(npb),npompr(npb),npomtg(npb)

        if(npomin(npb).ne.0)then
         do n=1,npomin(npb)                      !loop over interm. Pomerons
          wpi=xpopin(n,npb)*wp0                  !LC+ for the Pomeron
          wmi=xpomin(n,npb)*wm0                  !LC- for the Pomeron
          if(debug.ge.2)write (moniou,212)n,wpi,wmi
          if(wpi*wmi.ne.0.d0)then
           ic11=0
           ic12=0
           ic21=0
           ic22=0
           call qgstr(wpi,wmi,wppr0(ip),wmtg0(it)
     *     ,ic11,ic12,ic22,ic21,0,0)             !string hadronization
          endif
         enddo
        endif

        if(nqs(npb).ne.0)then
         do n=1,nqs(npb)                         !loop over single Pomerons
          lnp=nnpr(n,npb)                        !index for proj. constituent
          lnt=nntg(n,npb)                        !index for targ. constituent
          lqa0(ip)=lqa0(ip)-1
          lqb0(it)=lqb0(it)-1
          xpi=xppr(lnp,ip)
          xmi=xmtg(lnt,it)
          wpi=wp0*xpi                            !LC+ for the Pomeron
          wmi=wm0*xmi                            !LC- for the Pomeron
          sy=wpi*wmi
          wtot=qgpomc(sy,xpi,xmi,bbp,vvx,icdp,icdt,icz,1) !total
          wsoft=qgpomc(sy,xpi,xmi,bbp,vvx,icdp,icdt,icz,2)!soft interaction
          wqg=qgpomc(sy,xpi,xmi,bbp,vvx,icdp,icdt,icz,3)  !qg-hard interaction
          wgq=qgpomc(sy,xpi,xmi,bbp,vvx,icdp,icdt,icz,4)  !gq-hard interaction
          wqq=qgpomc(sy,xpi,xmi,bbp,vvx,icdp,icdt,icz,5)  !qq-hard interaction
          aks=qgran(b10)*wtot
          if(debug.ge.2)write (moniou,213)n,wpi,wmi

          if(aks.lt.wsoft.or.sy.lt.2.d0*s2min)then !soft string hadronization
           if(lqa0(ip).eq.0.and.lva(ip).eq.0)then
            call qgixxd(izp(ip),ic11,ic12,icz)
           else
            ic11=0
            ic12=0
           endif
           if(lqb0(it).eq.0.and.lvb(it).eq.0)then
            call qgixxd(izt(it),ic21,ic22,2)
           else
            ic21=0
            ic22=0
           endif
           call qgstr(wpi,wmi,wppr0(ip),wmtg0(it),ic11,ic12,ic22,ic21
     *     ,1,1)
          else            !QCD evolution and hadronization for semi-hard Pomeron
           if(lva(ip).eq.0.and.lvb(it).eq.0.and.aks.lt.wsoft+wqq)then
            iqq=3
            lva(ip)=1
            lvb(it)=1
           elseif(lva(ip).eq.0.and.aks.gt.wqg)then
            iqq=1
            lva(ip)=1
           elseif(lvb(it).eq.0.and.aks.gt.wgq)then
            iqq=2
            lvb(it)=1
           else
            iqq=0
           endif

           call qghot(wpi,wmi,dsqrt(bbp),vvx,nva,nvb,izp(ip),izt(it)
     *     ,icdp,icdt,icz,iqq,0)            !QCD evolution + jet hadronization
           if(iqq.eq.1.or.iqq.eq.3)ila(ip)=nva
           if(iqq.eq.2.or.iqq.eq.3)ilb(it)=nvb
          endif
         enddo
        endif

        if(npompr(npb).ne.0)then
         do l=1,npompr(npb)                 !loop over proj. leg Pomerons
          ipp=ilpr(l,npb)                  !proj. index
          lnp=lnpr(l,npb)                  !index for proj. constituent
          bbpr=bpompr(lnp,ipp)             !b^2 for the Pomeron
          vvxp=vvxpr(lnp,ipp)              !screening factor
          lqa0(ipp)=lqa0(ipp)-1
          xpi=xppr(lnp,ipp)
          xmi=xpompr(lnp,ipp)
          wpi=wp0*xpi                      !LC+ for the Pomeron
          wmi=wm0*xmi                      !LC- for the Pomeron
          sy=wpi*wmi
          if(sy.ne.0.d0)then
           wtot=qglegc(sy,xpi,bbpr,vvxp,iddp(ipp),icz,9)   !total
           wsoft=qglegc(sy,xpi,bbpr,vvxp,iddp(ipp),icz,10) !soft interaction
           wqg=qglegc(sy,xpi,bbpr,vvxp,iddp(ipp),icz,11)   !qg-hard interaction
          else
           wsoft=1.d0
           wtot=1.d0
           wqg=0.d0
          endif
          aks=qgran(b10)*wtot
          if(debug.ge.2)write (moniou,214)l,wpi,wmi

          if(aks.le.wsoft.or.sy.lt.2.d0*s2min)then  !soft string hadronization
           if(lqa0(ipp).eq.0.and.lva(ipp).eq.0.and.sy.ne.0.d0)then
            call qgixxd(izp(ipp),ic11,ic12,icz)
           else
            ic11=0
            ic12=0
           endif
           ic21=0
           ic22=0
           call qgstr(wpi,wmi,wppr0(ipp),wmtg0(it),ic11,ic12,ic22,ic21
     *     ,1,0)

          else        !QCD evolution and hadronization for semi-hard Pomeron
           if(lva(ipp).eq.0.and.aks.gt.wqg)then
            iqq=1
            lva(ipp)=1
           else
            iqq=0
           endif

           call qghot(wpi,wmi,dsqrt(bbpr),vvxp,nva,nvb,izp(ipp),izt(it)
     *     ,iddp(ipp),icdt,icz,iqq,1)         !QCD evolution + jet hadronization
           if(iqq.eq.1)ila(ipp)=nva
          endif
          call qglead(wppr0(ipp),wmtg0(it),lqa(ipp)+1-iwp(ipp)
     *    ,lqb(it)+1-iwt(it),lqa0(ipp)+ncola0(ipp),lqb0(it)+ncolb0(it)
     *    ,lva(ipp),lvb(it),izp(ipp),izt(it),ila(ipp),ilb(it),iret)  !remnants
          if(iret.ne.0)goto 1
         enddo
        endif

        if(npomtg(npb).ne.0)then
         do l=1,npomtg(npb)                !loop over targ. leg Pomerons
          itt=iltg(l,npb)                  !targ. index
          lnt=lntg(l,npb)                  !index for targ. constituent
          bbtg=bpomtg(lnt,itt)             !b^2 for the Pomeron
          vvxt=vvxtg(lnt,itt)              !screening factor
          lqb0(itt)=lqb0(itt)-1
          xmi=xmtg(lnt,itt)
          wmi=wm0*xmi                      !LC- for the Pomeron
          wpi=wp0*xpomtg(lnt,itt)          !LC+ for the Pomeron
          sy=wpi*wmi
          if(sy.ne.0.d0)then
           wtot=qglegc(sy,xmi,bbtg,vvxt,iddt(itt),2,9)  !tot
           wsoft=qglegc(sy,xmi,bbtg,vvxt,iddt(itt),2,10)!soft interaction
           wqg=qglegc(sy,xmi,bbtg,vvxt,iddt(itt),2,11)  !qg-hard interaction
          else
           wtot=1.d0
           wsoft=1.d0
           wqg=0.d0
          endif
          aks=qgran(b10)*wtot
          if(debug.ge.2)write (moniou,215)l,wpi,wmi

          if(aks.le.wsoft.or.sy.lt.2.d0*s2min)then  !soft string hadronization
           ic11=0
           ic12=0
           if(lqb0(itt).eq.0.and.lvb(itt).eq.0.and.sy.ne.0.d0)then
            call qgixxd(izt(itt),ic21,ic22,2)
           else
            ic21=0
            ic22=0
           endif
           call qgstr(wpi,wmi,wppr0(ip),wmtg0(itt),ic11,ic12,ic22,ic21
     *     ,0,1)

          else         !QCD evolution and hadronization for semi-hard Pomeron
           if(lvb(itt).eq.0.and.aks.gt.wqg)then
            iqq=2
            lvb(itt)=1
           else
            iqq=0
           endif

           call qghot(wpi,wmi,dsqrt(bbtg),vvxt,nva,nvb,izp(ip),izt(itt)
     *     ,icdp,iddt(itt),icz,iqq,2)         !QCD evolution + jet hadronization
           if(iqq.eq.2)ilb(itt)=nvb
          endif
          call qglead(wppr0(ip),wmtg0(itt),lqa(ip)+1-iwp(ip),lqb(itt)
     *    +1-iwt(itt),lqa0(ip)+ncola0(ip),lqb0(itt)+ncolb0(itt)
     *    ,lva(ip),lvb(itt),izp(ip),izt(itt),ila(ip),ilb(itt),iret) !remnants
          if(iret.ne.0)goto 1
         enddo
        endif
        ncola0(ip)=ncola0(ip)-1
        ncolb0(it)=ncolb0(it)-1
        call qglead(wppr0(ip),wmtg0(it),lqa(ip)+1-iwp(ip),lqb(it)
     *  +1-iwt(it),lqa0(ip)+ncola0(ip),lqb0(it)+ncolb0(it)
     *  ,lva(ip),lvb(it),izp(ip),izt(it),ila(ip),ilb(it),iret) !remnants
        if(iret.ne.0)goto 1
       enddo                                           !end of collision loop
      endif

      if(nj.ne.0)then                   !arrangement of parton color connections
       if(debug.ge.1)write (moniou,216)nj
       call qgjarr(jfl)
       if(jfl.eq.0)then
        iret=1
        goto 1
       endif
       if(debug.ge.1)write (moniou,217)
       call qgxjet                      !jet hadronization
      endif
      if(debug.ge.1)write (moniou,218)

201   format(2x,'qgsha - inelastic interaction, N of Pomeron blocks:'
     *,i4)
202   format(2x,'qgsha: energy-momentum sharing between Pomerons')
203   format(2x,'qgsha: log of acceptance probability - ',e10.3
     */4x,'N of rejections - ',i4,2x,'N of renorm. - ',i3)
204   format(2x,'qgsha:  new normalization of acceptance,'
     *,' N of renorm. - ',i3)
205   format(2x,'qgsha: leading remnant LC momenta')
206   format(2x,'qgsha: momentum conservation '
     *,'(correction for 3p-vertexes)')
207   format(2x,'qgsha: treatment of low mass diffraction')
208   format(2x,'qgsha: diffraction of ',i3,'-th proj. nucleon,'
     *,' recoil of ',i3,'-th targ. nucleon')
209   format(2x,'qgsha: diffraction of ',i3,'-th targ. nucleon,'
     *,' recoil of ',i3,'-th proj. nucleon')
210   format(2x,'qgsha: particle production for all cut Pomerons')
211   format(2x,'qgsha: ',i4,'-th collision,  proj. index - ',i3,2x
     *,'targ. index - ',i3
     */4x,'b^2=',e10.3,2x,'vvx=',e10.3,2x,'N of single Pomerons - ',i3
     *,2x,' N of interm. Pomerons - ',i3
     */4x,'N of proj. legs - ',i3,2x,'N of targ. legs - ',i3)
212   format(2x,'qgsha: particle production for '
     *,i3,'-th interm. Pomeron'
     */4x,'light cone momenta for the Pomeron:',2e10.3)
213   format(2x,'qgsha: particle production for '
     *,i3,'-th single Pomeron'
     */4x,'light cone momenta for the Pomeron:',2e10.3)
214   format(2x,'qgsha: particle production for '
     *,i3,'-th proj. leg Pomeron'
     */4x,'light cone momenta for the Pomeron:',2e10.3)
215   format(2x,'qgsha: particle production for '
     *,i3,'-th targ. leg Pomeron'
     */4x,'light cone momenta for the Pomeron:',2e10.3)
216   format(2x,'qgsha: arrangement of color connections for '
     *,i5,' final partons')
217   format(2x,'qgsha: jet hadronization')
218   format(2x,'qgsha - end')
      return
      end

c=============================================================================
      subroutine qgprox(imode)
c-------------------------------------------------------------------------
c qgprox - propose Pomeron end LC momenta
c imod = 0 - to define normalization
c imod = 1 - propose values according to x^delf * (1 - sum_i x_i)^ahl
c-------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208,npbmax=1000,npnmax=900,npmax=900,legmax=900)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr6/  pi,bm,amws
      common /qgarr9/  iwp(iapmax),iwt(iapmax),lqa(iapmax),lqb(iapmax)
     *,iprcn(iapmax),itgcn(iapmax),ias(npbmax),ibs(npbmax),nqs(npbmax)
     *,npompr(npbmax),npomtg(npbmax),npomin(npbmax),nnpr(npmax,npbmax)
     *,nntg(npmax,npbmax),ilpr(legmax,npbmax),iltg(legmax,npbmax)
     *,lnpr(legmax,npbmax),lntg(legmax,npbmax)
     *,nbpi(npnmax,iapmax),nbti(npnmax,iapmax),idnpi(npnmax,iapmax)
     *,idnti(npnmax,iapmax),nppi(npnmax,iapmax),npti(npnmax,iapmax)
     *,nlpi(npnmax,iapmax),nlti(npnmax,iapmax)
      common /qgarr11/ b10
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr19/ ahl(3)
      common /qgarr23/ bbpom(npbmax),vvxpom(npbmax)
     *,bpompr(npnmax,iapmax),bpomtg(npnmax,iapmax)
     *,vvxpr(npnmax,iapmax),vvxtg(npnmax,iapmax)
     *,xpompr(npnmax,iapmax),xpomtg(npnmax,iapmax)
     *,xpopin(npmax,npbmax),xpomin(npmax,npbmax),vvxin(npmax,npbmax)
     *,bpomin(npmax,npbmax)
      common /qgarr40/ xppr(npnmax,iapmax),xmtg(npnmax,iapmax)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      external qgran

      if(debug.ge.3)write (moniou,201)imode

      delf=dels
      if(imode.eq.0)then                    !0-configuration (for normalization)
       do ip=1,ia(1)                        !loop over proj. nucleons
        if(lqa(ip).ne.0)then
         do n=1,lqa(ip)                     !loop over proj. constituents
          if(idnpi(n,ip).eq.0)then
           xppr(n,ip)=1.d0/wp0              !LC+ for single Pomeron
          else
           xppr(n,ip)=1.d0/xpompr(n,ip)/scm !LC+ for leg Pomeron
          endif
          enddo
        endif
       enddo
       do it=1,ia(2)                        !loop over targ. nucleons
        if(lqb(it).ne.0)then
         do n=1,lqb(it)                     !loop over targ. constituents
          if(idnti(n,it).eq.0)then
           xmtg(n,it)=1.d0/wm0              !LC- for single Pomeron
          else
           xmtg(n,it)=1.d0/xpomtg(n,it)/scm !LC- for leg Pomeron
          endif
         enddo
        endif
       enddo

      else                                  !proposed configuration
       do ip=1,ia(1)                        !loop over proj. nucleons
        if(lqa(ip).ne.0)then
         xpt=1.d0
         do n=1,lqa(ip)                     !loop over proj. constituents
          nrej=0
          alfl=ahl(icz)+(lqa(ip)-n)*(1.d0+delf)
c          if(icz.eq.2)alfl=alfl-float(lqa(ip)-1)/lqa(ip)  !baryon "junction"
          gb0=(1.d0-.11d0**(1.d0/(1.d0+delf)))**alfl
     *    *exp(alfl*(1.d0+delf)*.11d0)*2.d0
1         continue
c proposal functions are chosen depending on the parameters
c to assure an efficient procedure
          if(delf.ge.0.d0.and.alfl.ge.0.d0
     *    .or.delf.lt.0.d0.and.alfl.le.0.d0)then
           up=1.d0-qgran(b10)**(1.d0/(1.d0+delf))
           if(1.d0-up.lt.1.d-20)goto 1
           tp=1.d0-up**(1.d0/(1.d0+alfl))
           gb=(tp/(1.d0-up))**delf
          elseif(delf.lt.0.d0.and.alfl.gt.0.d0)then
           up=-log(1.d0-qgran(b10)*(1.d0-exp(-alfl*(1.d0+delf))))
     *     /alfl/(1.d0+delf)
           tp=up**(1.d0/(1.d0+delf))
           gb=(1.d0-tp)**alfl*exp(alfl*(1.d0+delf)*up)/gb0
          else
           tp=1.d0-qgran(b10)**(1.d0/(1.d0+alfl))
           gb=tp**delf
          endif
          if(qgran(b10).gt.gb)then
           nrej=nrej+1
           goto 1
          endif
          xppr(n,ip)=tp*xpt                 !proposed LC+ for the constituent
          xpt=xpt-xppr(n,ip)                !LC+ of the remnant
          enddo
        endif
       enddo

       do it=1,ia(2)                        !loop over targ. nucleons
        if(lqb(it).ne.0)then
         xmt=1.d0
         do n=1,lqb(it)                     !loop over targ. constituents
          nrej=0
          alfl=ahl(2)+(lqb(it)-n)*(1.d0+delf)
c     *    -float(lqb(it)-1)/lqb(it)                       !baryon "junction"
          gb0=(1.d0-.11d0**(1.d0/(1.d0+delf)))**alfl
     *    *exp(alfl*(1.d0+delf)*.11d0)*2.d0
2         continue
          if(delf.ge.0.d0.and.alfl.ge.0.d0
     *    .or.delf.lt.0.d0.and.alfl.le.0.d0)then
           up=1.d0-qgran(b10)**(1.d0/(1.d0+delf))
           if(1.d0-up.lt.1.d-20)goto 2
           tp=1.d0-up**(1.d0/(1.d0+alfl))
           gb=(tp/(1.d0-up))**delf
          elseif(delf.lt.0.d0.and.alfl.gt.0.d0)then
           up=-log(1.d0-qgran(b10)*(1.d0-exp(-alfl*(1.d0+delf))))
     *     /alfl/(1.d0+delf)
           tp=up**(1.d0/(1.d0+delf))
           gb=(1.d0-tp)**alfl*exp(alfl*(1.d0+delf)*up)/gb0
          else
           tp=1.d0-qgran(b10)**(1.d0/(1.d0+alfl))
           gb=tp**delf
          endif
          if(qgran(b10).gt.gb)then
           nrej=nrej+1
           goto 2
          endif
          if(qgran(b10).gt.gb)goto 2
          xmtg(n,it)=tp*xmt                 !proposed LC- for the constituent
          xmt=xmt-xmtg(n,it)                !LC- of the remnant
          enddo
        endif
       enddo
      endif
      if(debug.ge.4)write (moniou,202)

201   format(2x,'qgprox - propose Pomeron end LC momenta, imode=',i2)
202   format(2x,'qgprox - end')
      return
      end

c=============================================================================
      double precision function qgweix(nbpom)
c-------------------------------------------------------------------------
c qgweix - log-weight of x-configuration
c imod = 0 - to define normalization
c imod = 1 - propose values according to x^delf * (1 - sum_i x_i)^ahl
c-------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208,npbmax=1000,npnmax=900,npmax=900,legmax=900)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr6/  pi,bm,amws
      common /qgarr9/  iwp(iapmax),iwt(iapmax),lqa(iapmax),lqb(iapmax)
     *,iprcn(iapmax),itgcn(iapmax),ias(npbmax),ibs(npbmax),nqs(npbmax)
     *,npompr(npbmax),npomtg(npbmax),npomin(npbmax),nnpr(npmax,npbmax)
     *,nntg(npmax,npbmax),ilpr(legmax,npbmax),iltg(legmax,npbmax)
     *,lnpr(legmax,npbmax),lntg(legmax,npbmax)
     *,nbpi(npnmax,iapmax),nbti(npnmax,iapmax),idnpi(npnmax,iapmax)
     *,idnti(npnmax,iapmax),nppi(npnmax,iapmax),npti(npnmax,iapmax)
     *,nlpi(npnmax,iapmax),nlti(npnmax,iapmax)
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr23/ bbpom(npbmax),vvxpom(npbmax)
     *,bpompr(npnmax,iapmax),bpomtg(npnmax,iapmax)
     *,vvxpr(npnmax,iapmax),vvxtg(npnmax,iapmax)
     *,xpompr(npnmax,iapmax),xpomtg(npnmax,iapmax)
     *,xpopin(npmax,npbmax),xpomin(npmax,npbmax),vvxin(npmax,npbmax)
     *,bpomin(npmax,npbmax)
      common /qgarr40/ xppr(npnmax,iapmax),xmtg(npnmax,iapmax)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.3)write (moniou,201)nbpom

      delf=dels
      qgweix=0.d0
      do npb=1,nbpom                              !loop over collisions
       ip=ias(npb)                                !proj. index
       it=ibs(npb)                                !targ. index
       icdp=iddp(ip)                              !proj. diffr. eigenstate
       icdt=iddt(it)                              !targ. diffr. eigenstate
       bbp=bbpom(npb)                             !b^2 between proj. and targ.
       vvx=vvxpom(npb)                            !nuclear screening factor
       if(nqs(npb).ne.0)then
        do n=1,nqs(npb)                           !loop over single Pomerons
         lnp=nnpr(n,npb)                          !proj. constituent index
         lnt=nntg(n,npb)                          !targ. constituent index
         xpp=xppr(lnp,ip)                         !LC+ for the Pomeron
         xpm=xmtg(lnt,it)                         !LC- for the Pomeron
         qgweix=qgweix+dlog(qgpomc(scm*xpp*xpm,xpp,xpm,bbp,vvx
     *   ,icdp,icdt,icz,1)/(xpp*xpm)**delf)       !add single Pomeron contrib.
        enddo
       endif
       if(npompr(npb).ne.0)then
        do l=1,npompr(npb)                         !loop over proj. leg Pomerons
         ipp=ilpr(l,npb)                          !proj. index
         lnp=lnpr(l,npb)                          !proj. constituent index
         xpp=xppr(lnp,ipp)                        !LC+ for the Pomeron
         xpomr=1.d0/xpompr(lnp,ipp)/scm           !LC+ for the 3P vertex
         vvxp=vvxpr(lnp,ipp)                      !screening factor
         bbpr=bpompr(lnp,ipp)                          !b^2 for the Pomeron
         qgweix=qgweix+dlog(qglegc(xpp/xpomr,xpp,bbpr,vvxp
     *   ,iddp(ipp),icz,9)/xpp**delf)             !add leg Pomeron contrib.
        enddo
       endif
       if(npomtg(npb).ne.0)then
        do l=1,npomtg(npb)                        !loop over targ. leg Pomerons
         itt=iltg(l,npb)                          !targ. index
         lnt=lntg(l,npb)                          !targ. constituent index
         xpm=xmtg(lnt,itt)                        !LC- for the Pomeron
         xpomr=xpomtg(lnt,itt)                    !LC+ for the 3P vertex
         vvxt=vvxtg(lnt,itt)                      !screening factor
         bbtg=bpomtg(lnt,itt)                          !b^2 for the Pomeron
         qgweix=qgweix+dlog(qglegc(xpomr*scm*xpm,xpm,bbtg,vvxt
     *   ,iddt(itt),2,9)/xpm**delf)               !add leg Pomeron contrib.
        enddo
       endif
      enddo
      if(debug.ge.4)write (moniou,202)qgweix

201   format(2x,'qgweix - log-weight of x-configuration,'
     *,' N of collisions - ',i4)
202   format(2x,'qgweix=',e10.3)
      return
      end

c=============================================================================
      subroutine qghot(wpp,wpm,b,vvx,nva,nvb,izp,izt,icdp,icdt,icz,iqq
     *,jpt)
c---------------------------------------------------------------------------
c qghot - semi-hard process
c wpp,wpm   - LC momenta for the constituent partons,
c b         - impact parameter for the semi-hard Pomeron,
c izp, izt  - types of proj. and targ. remnants,
c icdp,icdt - proj. and targ.  diffractive eigenstates,
c iqq - type of the semi-hard process: 0 - gg, 1 - q_vg, 2 - gq_v, 3 - q_vq_v
c jpt=0 - single Pomeron,
c jpt=1 - proj. leg Pomeron,
c jpt=2 - targ. leg Pomeron
c---------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      character*2 tyq
      parameter(njmax=50000)
      dimension ept(4),ep3(4),ey(3),ebal(4),
     *qmin(2),wp(2),iqc(2),iqp(2),nqc(2),ncc(2,2),
     *qv1(30,50),zv1(30,50),qm1(30,50),iqv1(30,50),
     *ldau1(30,49),lpar1(30,50),
     *qv2(30,50),zv2(30,50),qm2(30,50),iqv2(30,50),
     *ldau2(30,49),lpar2(30,50)
      parameter(iapmax=208,npbmax=1000,npnmax=900,npmax=900,legmax=900)
      common /qgarr2/  scm,wp0,wm0
      common /qgarr6/  pi,bm,amws
      common /qgarr8/  wwm,be(4),dc(5),deta,almpt,ptdif,ptndi
      common /qgarr10/ am(7),ammu
      common /qgarr11/ b10
      common /qgarr12/ nsp
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr37/ eqj(4,njmax),iqj(njmax),ncj(2,njmax),nj
      common /qgarr42/ tyq(16)
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/ debug
      external qgran

      if(debug.ge.1)write (moniou,201)iqq,wpp,wpm,izp,izt,icdp,icdt
     *,icz,jpt,nj

      wwgg=0.d0
      wwqg=0.d0
      wwgq=0.d0
      wwqq=0.d0
      wpi=0.d0
      wmi=0.d0
      sjqg=0.d0
      sjqq=0.d0
      sea1=0.d0
      sea2=0.d0
      glu1=0.d0
      glu2=0.d0
      nj0=nj                       !store number of final partons
      nsp0=nsp                     !store number of final particles

1     sy=wpp*wpm  !energy squared for semi-hard inter. (including preevolution)
      nj=nj0
      nsp=nsp0
      s2min=4.d0*fqscal*qt0       !threshold energy
      if(sy.lt.s2min)stop'qghot: sy<s2min!!!'

      if(iqq.eq.3)then             !q_vq_v-ladder
       wpi=wpp                     !LC+ for the hard interaction
       wmi=wpm                     !LC- for the hard interaction
      else

c-------------------------------------------------
c normalization of acceptance
       xmin=s2min/sy
       iq=(iqq+1)/2+1              !auxilliary type of parton (1 - g, 2 - q(q~))
       sj=qgjit(qt0,qt0,sy,1,iq)   !inclusive parton-parton cross-sections
       if(iqq.eq.0)then
        gb0=-dlog(xmin)*(1.d0-dsqrt(xmin))**(2.d0*betp)*sj
       else
        gb0=(1.d0-xmin)**betp*sj
       endif
       if(jpt.eq.0)then            !single Pomeron
        if(iqq.eq.0)then
         rp0=(rq(icdp,icz)+rq(icdt,2)+alfp*dlog(scm/s2min))
     *   *4.d0*.0389d0
         gb0=gb0/(rq(icdp,icz)+rq(icdt,2)+alfp*dlog(scm/sy))
     *   *exp(-b*b/rp0)
        elseif(iqq.eq.1)then
         rp0=(rq(icdp,icz)+rq(icdt,2)+alfp*dlog(wpp*wm0/s2min))
     *   *4.d0*.0389d0
         gb0=gb0/(rq(icdp,icz)+rq(icdt,2)+alfp*dlog(wm0/wpm))
     *   *exp(-b*b/rp0)
        elseif(iqq.eq.2)then
         rp0=(rq(icdp,icz)+rq(icdt,2)+alfp*dlog(wpm*wp0/s2min))
     *   *4.d0*.0389d0
         gb0=gb0/(rq(icdp,icz)+rq(icdt,2)+alfp*dlog(wp0/wpp))
     *   *exp(-b*b/rp0)
        endif
       elseif(jpt.eq.1)then        !proj. leg Pomeron
        if(iqq.eq.0)then
         rp0=(rq(icdp,icz)+alfp*dlog(wp0*wpm/s2min))*4.d0*.0389d0
         gb0=gb0/(rq(icdp,icz)+alfp*dlog(wp0/wpp))*exp(-b*b/rp0)
        elseif(iqq.eq.1)then
         rp0=(rq(icdp,icz)+alfp*dlog(sy/s2min))*4.d0*.0389d0
         gb0=gb0/rq(icdp,icz)*exp(-b*b/rp0)
        endif
       elseif(jpt.eq.2)then        !targ. leg Pomeron
        if(iqq.eq.0)then
         rp0=(rq(icdt,2)+alfp*dlog(wm0*wpp/s2min))*4.d0*.0389d0
         gb0=gb0/(rq(icdt,2)+alfp*dlog(wm0/wpm))*exp(-b*b/rp0)
        elseif(iqq.eq.2)then
         rp0=(rq(icdt,2)+alfp*dlog(sy/s2min))*4.d0*.0389d0
         gb0=gb0/rq(icdt,2)*exp(-b*b/rp0)
        endif
       endif

c-------------------------------------------------
c sharing of LC momenta between soft preevolution and hard ladder
2      zpm=(1.d0-qgran(b10)*(1.d0-xmin**(delh-dels)))
     * **(1.d0/(delh-dels))
       sjqq=qgjit(qt0,qt0,zpm*sy,2,2)  !inclusive qq cross-section
       sjqg=qgjit(qt0,qt0,zpm*sy,1,2)  !inclusive qg cross-section
       sjgg=qgjit(qt0,qt0,zpm*sy,1,1)  !inclusive gg cross-section

       if(iqq.eq.0)then              !gg-ladder
        xp=zpm**qgran(b10)           !LC+ momentum share
        xm=zpm/xp                    !LC- momentum share
        wpi=wpp*xp                   !LC+ for the hard interaction
        wmi=wpm*xm                   !LC- for the hard interaction
        if(jpt.eq.0)then             !single Pomeron
         rp1=(rq(icdp,icz)+alfp*dlog(wp0/wpi))*4.d0*.0389d0
         rp2=(rq(icdt,2)+alfp*dlog(wm0/wmi))*4.d0*.0389d0
         rp=rp1*rp2/(rp1+rp2)
         z=qgran(b10)
         phi=pi*qgran(b10)
         b0=dsqrt(-rp*dlog(z))
         bb1=(b*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
         bb2=(b*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2

         xpomr=wpi/wp0
         if(xpomr*sgap.ge.1.d0.or.xpomr*scm.le.sgap)then
          vvx1=0.d0
         else
          v1pnu0=qgfani(1.d0/xpomr,bb1,vvx,0.d0,0.d0,icdp,icz,1)
          v1tnu0=qgfani(xpomr*scm,bb2,vvx,0.d0,0.d0,icdt,2,1)
          nn=0
21        nn=nn+1
          vvxt=1.d0-exp(-v1pnu0)*(1.d0-vvx)
          vvxp=1.d0-exp(-v1tnu0)*(1.d0-vvx)
          v1pnu=qgfani(1.d0/xpomr,bb1,vvxp,0.d0,0.d0,icdp,icz,1)
          v1tnu=qgfani(xpomr*scm,bb2,vvxt,0.d0,0.d0,icdt,2,1)
          if((abs(v1pnu0-v1pnu).gt.1.d-1.or.abs(v1tnu0-v1tnu).gt.1.d-1)
     *    .and.nn.lt.100)then
           v1pnu0=v1pnu
           v1tnu0=v1tnu
           goto 21
          endif
          vvx1=1.d0-exp(-v1tnu)*(1.d0-vvx)
         endif

         xpomr=wm0/wmi/scm
         if(xpomr*sgap.ge.1.d0.or.xpomr*scm.le.sgap)then
          vvx2=0.d0
         else
          v1pnu0=qgfani(1.d0/xpomr,bb1,vvx,0.d0,0.d0,icdp,icz,1)
          v1tnu0=qgfani(xpomr*scm,bb2,vvx,0.d0,0.d0,icdt,2,1)
          nn=0
22        nn=nn+1
          vvxt=1.d0-exp(-v1pnu0)*(1.d0-vvx)
          vvxp=1.d0-exp(-v1tnu0)*(1.d0-vvx)
          v1pnu=qgfani(1.d0/xpomr,bb1,vvxp,0.d0,0.d0,icdp,icz,1)
          v1tnu=qgfani(xpomr*scm,bb2,vvxt,0.d0,0.d0,icdt,2,1)
          if((abs(v1pnu0-v1pnu).gt.1.d-1.or.abs(v1tnu0-v1tnu).gt.1.d-1)
     *    .and.nn.lt.100)then
           v1pnu0=v1pnu
           v1tnu0=v1tnu
           goto 22
          endif
          vvx2=1.d0-exp(-v1pnu)*(1.d0-vvx)
         endif

         glu1=qglegc(1.d0/xp,wpp/wp0,bb1,vvx1,icdp,icz,12) !upper gluon PDF
         sea1=qglegc(1.d0/xp,wpp/wp0,bb1,vvx1,icdp,icz,13) !upper quark PDF
         glu2=qglegc(1.d0/xm,wpm/wm0,bb2,vvx2,icdt,2,12)   !lower gluon PDF
         sea2=qglegc(1.d0/xm,wpm/wm0,bb2,vvx2,icdt,2,13)   !lower quark PDF
        elseif(jpt.eq.1)then                         !proj. leg Pomeron
         rp1=(rq(icdp,icz)+alfp*dlog(wp0/wpi))*4.d0*.0389d0
         rp2=-alfp*dlog(xm)*4.d0*.0389d0
         rp=rp1*rp2/(rp1+rp2)
         z=qgran(b10)
         phi=pi*qgran(b10)
         b0=dsqrt(-rp*dlog(z))
         bb1=(b*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
         bb2=(b*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2

         glu1=qglegc(1.d0/xp,wpp/wp0,bb1,vvx,icdp,icz,12) !upper gluon PDF
         sea1=qglegc(1.d0/xp,wpp/wp0,bb1,vvx,icdp,icz,13) !upper quark PDF
         glu2=qgppdi(xm,0)
         sea2=qgppdi(xm,1)
        elseif(jpt.eq.2)then                         !proj. leg Pomeron
         rp1=(rq(icdt,2)+alfp*dlog(wm0/wmi))*4.d0*.0389d0
         rp2=-alfp*dlog(xp)*4.d0*.0389d0
         rp=rp1*rp2/(rp1+rp2)
         z=qgran(b10)
         phi=pi*qgran(b10)
         b0=dsqrt(-rp*dlog(z))
         bb1=(b*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
         bb2=(b*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2

         glu1=qglegc(1.d0/xm,wpm/wm0,bb1,vvx,icdt,2,12) !upper gluon PDF
         sea1=qglegc(1.d0/xm,wpm/wm0,bb1,vvx,icdt,2,13) !upper quark PDF
         glu2=qgppdi(xp,0)
         sea2=qgppdi(xp,1)
        endif
        wwgg=glu1*glu2*sjgg
        wwqg=sea1*glu2*sjqg
        wwgq=glu1*sea2*sjqg
        wwqq=sea1*sea2*sjqq
        gbyj=-dlog(zpm)*(wwgg+wwqg+wwgq+wwqq)
        if(jpt.eq.0)then
         rh=rq(icdp,icz)+rq(icdt,2)-alfp*dlog(zpm*sy/scm)
        elseif(jpt.eq.1)then
         rh=rq(icdp,icz)-alfp*dlog(wpp/wp0*zpm)
        elseif(jpt.eq.2)then
         rh=rq(icdt,2)-alfp*dlog(wpm/wm0*zpm)
        else
         rh=0.d0
         stop 'Should not happen in qghot'
        endif
        gbyj=gbyj/rh*exp(-b*b/(4.d0*.0389d0*rh))

       else                          !q_vg-(gq_v-)ladder
        if(iqq.eq.1)then             !q_vg-ladder
         wpi=wpp
         wmi=wpm*zpm
         xm=zpm
         if(jpt.eq.0)then            !single Pomeron
          rp1=rq(icdp,icz)*4.d0*.0389d0
          rp2=(rq(icdt,2)+alfp*dlog(wm0/wmi))*4.d0*.0389d0
          rp=rp1*rp2/(rp1+rp2)
          z=qgran(b10)
          phi=pi*qgran(b10)
          b0=dsqrt(-rp*dlog(z))
          bb1=(b*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
          bb2=(b*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2

          xpomr=wm0/wmi/scm
          if(xpomr*sgap.ge.1.d0.or.xpomr*scm.le.sgap)then
           vvx2=0.d0
          else
           v1pnu0=qgfani(1.d0/xpomr,bb1,vvx,0.d0,0.d0,icdp,icz,1)
           v1tnu0=qgfani(xpomr*scm,bb2,vvx,0.d0,0.d0,icdt,2,1)
           nn=0
23         nn=nn+1
           vvxt=1.d0-exp(-v1pnu0)*(1.d0-vvx)
           vvxp=1.d0-exp(-v1tnu0)*(1.d0-vvx)
           v1pnu=qgfani(1.d0/xpomr,bb1,vvxp,0.d0,0.d0,icdp,icz,1)
           v1tnu=qgfani(xpomr*scm,bb2,vvxt,0.d0,0.d0,icdt,2,1)
           if((abs(v1pnu0-v1pnu).gt.1.d-1.or.abs(v1tnu0-v1tnu).gt.1.d-1)
     *     .and.nn.lt.100)then
            v1pnu0=v1pnu
            v1tnu0=v1tnu
            goto 23
           endif
           vvx2=1.d0-exp(-v1pnu)*(1.d0-vvx)
          endif

          glu2=qglegc(1.d0/xm,wpm/wm0,bb2,vvx2,icdt,2,12) !upper gluon PDF
          sea2=qglegc(1.d0/xm,wpm/wm0,bb2,vvx2,icdt,2,13) !upper quark PDF
          wwqg=glu2*sjqg
          wwqq=sea2*sjqq
         else                        !leg Pomeron
          wwqg=qgppdi(xm,0)*sjqg
          wwqq=qgppdi(xm,1)*sjqq
         endif
        elseif(iqq.eq.2)then         !gq_v-ladder
         wpi=wpp*zpm
         wmi=wpm
         xp=zpm
         if(jpt.eq.0)then            !single Pomeron
          rp1=(rq(icdp,icz)+alfp*dlog(wp0/wpi))*4.d0*.0389d0
          rp2=rq(icdt,2)*4.d0*.0389d0
          rp=rp1*rp2/(rp1+rp2)
          z=qgran(b10)
          phi=pi*qgran(b10)
          b0=dsqrt(-rp*dlog(z))
          bb1=(b*rp1/(rp1+rp2)+b0*cos(phi))**2+(b0*sin(phi))**2
          bb2=(b*rp2/(rp1+rp2)-b0*cos(phi))**2+(b0*sin(phi))**2

          xpomr=wpi/wp0
          if(xpomr*sgap.ge.1.d0.or.xpomr*scm.le.sgap)then
           vvx1=0.d0
          else
           v1pnu0=qgfani(1.d0/xpomr,bb1,vvx,0.d0,0.d0,icdp,icz,1)
           v1tnu0=qgfani(xpomr*scm,bb2,vvx,0.d0,0.d0,icdt,2,1)
           nn=0
24         nn=nn+1
           vvxt=1.d0-exp(-v1pnu0)*(1.d0-vvx)
           vvxp=1.d0-exp(-v1tnu0)*(1.d0-vvx)
           v1pnu=qgfani(1.d0/xpomr,bb1,vvxp,0.d0,0.d0,icdp,icz,1)
           v1tnu=qgfani(xpomr*scm,bb2,vvxt,0.d0,0.d0,icdt,2,1)
           if((abs(v1pnu0-v1pnu).gt.1.d-1.or.abs(v1tnu0-v1tnu).gt.1.d-1)
     *     .and.nn.lt.100)then
            v1pnu0=v1pnu
            v1tnu0=v1tnu
            goto 24
           endif
           vvx1=1.d0-exp(-v1tnu)*(1.d0-vvx)
          endif

          glu1=qglegc(1.d0/xp,wpp/wp0,bb1,vvx1,icdp,icz,12) !upper gluon PDF
          sea1=qglegc(1.d0/xp,wpp/wp0,bb1,vvx1,icdp,icz,13) !upper quark PDF
          wwqg=glu1*sjqg
          wwqq=sea1*sjqq
         else                        !leg Pomeron
          wwqg=qgppdi(xp,0)*sjqg
          wwqq=qgppdi(xp,1)*sjqq
         endif
        endif
        gbyj=wwqg+wwqq
        if(jpt.eq.0)then
         if(iqq.eq.1)then
          rh=rq(icdp,icz)+rq(icdt,2)-alfp*dlog(wpm/wm0*zpm)
         else
          rh=rq(icdp,icz)+rq(icdt,2)-alfp*dlog(wpp/wp0*zpm)
         endif
        elseif(jpt.eq.1)then
         rh=rq(icdp,icz)-alfp*dlog(zpm)
        elseif(jpt.eq.2)then
         rh=rq(icdt,2)-alfp*dlog(zpm)
        else
         rh=0.d0
         stop 'Should not happen in qghot'
        endif
        gbyj=gbyj/rh*exp(-b*b/(4.d0*.0389d0*rh))
       endif

       gbyj=gbyj/gb0/zpm**delh
       if(qgran(b10).gt.gbyj)goto 2
      endif
      if(debug.ge.2)write (moniou,202)wpi*wmi

11    wpi1=wpi
      wmi1=wmi
      wpq=0.d0
      wmq=0.d0
      nj=nj0                     !initialization for the number of final partons
      rrr=qgran(b10)
      jqq=0                                  !gg-ladder
      if(iqq.eq.1.or.iqq.eq.2)then
       if(rrr.lt.wwqq/(wwqg+wwqq))jqq=1      !q_vq_s-laddder
      elseif(iqq.eq.0)then
       if(rrr.lt.wwqg/(wwgg+wwqg+wwgq+wwqq))then
        jqq=1                                !q_sg-ladder
       elseif(rrr.lt.(wwqg+wwgq)/(wwgg+wwqg+wwgq+wwqq))then
        jqq=2                                !gq_s-ladder
       elseif(rrr.lt.(wwqg+wwgq+wwqq)/(wwgg+wwqg+wwgq+wwqq))then
        jqq=3                                !q_sq_s-ladder
       endif
      endif

c-------------------------------------------------
c parton types for the ladder legs and for the leading jets
c iqc(1) - flavor for the upper quark (0 in case of gluon),
c iqc(2) - the same for the lower one
      if(iqq.ne.0.and.iqq.ne.2)then          !q_v from the proj.
       call qgvdef(izp,ic1,ic2,icz)          !leading state flavor
       iqc(1)=ic1                            !upper leg parton
       nj=nj+1
       if(nj.gt.njmax)stop'increase njmax!!!'
       nva=nj
       iqj(nj)=ic2                           !leading jet parton
       ncc(1,1)=nj                           !color connection with leading jet
       ncc(2,1)=0
      else                                   !g(q_s) from the proj.
       nj=nj+1
       if(nj.gt.njmax)stop'increase njmax!!!'
       if(qgran(b10).lt.dc(2))then
        iqj(nj)=-4
       else
        iqj(nj)=-int(2.d0*qgran(b10)+1.d0)
       endif
       iqj(nj+1)=-iqj(nj)
       wp1=wpp-wpi
       wp2=wp1*qgran(b10)
       wp1=wp1-wp2
       eqj(1,nj)=.5d0*wp1
       eqj(2,nj)=eqj(1,nj)
       eqj(3,nj)=0.d0
       eqj(4,nj)=0.d0
       eqj(1,nj+1)=.5d0*wp2
       eqj(2,nj+1)=eqj(1,nj+1)
       eqj(3,nj+1)=0.d0
       eqj(4,nj+1)=0.d0
       if(jqq.eq.0.or.iqq.eq.0.and.jqq.eq.2)then
        iqc(1)=0
        ncc(1,1)=nj
        ncc(2,1)=nj+1
        nj=nj+1
        if(nj.gt.njmax)stop'increase njmax!!!'
       else
        if(qgran(b10).lt..3333d0)then
         iqc(1)=3*(2.d0*int(.5d0+qgran(b10))-1.d0)
        else
         iqc(1)=int(2.d0*qgran(b10)+1.d0)
     *   *(2.d0*int(.5d0+qgran(b10))-1.d0)
        endif
12      zg=xp+qgran(b10)*(1.d0-xp)           !gluon splitting into qq~
        if(qgran(b10).gt.zg**dels*((1.d0-xp/zg)/ (1.d0-xp))**betp)
     *  goto 12
        xg=xp/zg
        wpq0=wpp*(xg-xp)
        wmq=1.d0/wpq0
        wmi1=wmi1-wmq
        if(wmi1*wpi1.le.s2min)goto 11
        nj=nj+2
        if(nj.gt.njmax)stop'increase njmax!!!'
        iqj(nj)=-iqc(1)
        if(iabs(iqc(1)).eq.3)iqj(nj)=iqj(nj)*4/3
        eqj(1,nj)=.5d0*wmq
        eqj(2,nj)=-.5d0*wmq
        eqj(3,nj)=0.d0
        eqj(4,nj)=0.d0
        if(iqc(1).gt.0)then
         ncj(1,nj)=nj-1
         ncj(1,nj-1)=nj
         ncj(2,nj)=0
         ncj(2,nj-1)=0
         ncc(1,1)=nj-2
         ncc(2,1)=0
        else
         ncj(1,nj)=nj-2
         ncj(1,nj-2)=nj
         ncj(2,nj)=0
         ncj(2,nj-2)=0
         ncc(1,1)=nj-1
         ncc(2,1)=0
        endif
       endif
      endif

      if((iqq-2)*(iqq-3)*(iqq-4).eq.0)then     !q_v from the targ.
       call qgvdef(izt,ic1,ic2,2)              !leading state flavor
       iqc(2)=ic1                              !lower leg parton
       nj=nj+1
       if(nj.gt.njmax)stop'increase njmax!!!'
       nvb=nj
       iqj(nj)=ic2
       ncc(1,2)=nj
       ncc(2,2)=0
      else
       nj=nj+1
       if(nj.gt.njmax)stop'increase njmax!!!'
       if(qgran(b10).lt.dc(2))then
        iqj(nj)=-4
       else
        iqj(nj)=-int(2.d0*qgran(b10)+1.d0)
       endif
       iqj(nj+1)=-iqj(nj)
       wm1=wpm-wmi
       wm2=wm1*qgran(b10)
       wm1=wm1-wm2
       eqj(1,nj)=.5d0*wm1
       eqj(2,nj)=-eqj(1,nj)
       eqj(3,nj)=0.d0
       eqj(4,nj)=0.d0
       eqj(1,nj+1)=.5d0*wm2
       eqj(2,nj+1)=-eqj(1,nj+1)
       eqj(3,nj+1)=0.d0
       eqj(4,nj+1)=0.d0
       if(jqq.eq.0.or.iqq.eq.0.and.jqq.eq.1)then
        iqc(2)=0
        ncc(1,2)=nj
        ncc(2,2)=nj+1
        nj=nj+1
        if(nj.gt.njmax)stop'increase njmax!!!'
       else
        if(qgran(b10).lt..3333d0)then
         iqc(2)=3*(2.d0*int(.5d0+qgran(b10))-1.d0)
        else
         iqc(2)=int(2.d0*qgran(b10)+1.d0)
     *   *(2.d0*int(.5d0+qgran(b10))-1.d0)
        endif
14      zg=xm+qgran(b10)*(1.d0-xm)           !gluon splitting into qq~
        if(qgran(b10).gt.zg**dels*((1.d0-xm/zg)/ (1.d0-xm))**betp)
     *  goto 14
        xg=xm/zg
        wmq0=wpm*(xg-xm)
        wpq=1.d0/wmq0
        wpi1=wpi1-wpq
        if(wmi1*wpi1.le.s2min)goto 11
        nj=nj+2
        if(nj.gt.njmax)stop'increase njmax!!!'
        iqj(nj)=-iqc(2)
        if(iabs(iqc(2)).eq.3)iqj(nj)=iqj(nj)*4/3
        eqj(1,nj)=.5d0*wpq
        eqj(2,nj)=.5d0*wpq
        eqj(3,nj)=0.d0
        eqj(4,nj)=0.d0
        if(iqc(2).gt.0)then
         ncj(1,nj)=nj-1
         ncj(1,nj-1)=nj
         ncj(2,nj)=0
         ncj(2,nj-1)=0
         ncc(1,2)=nj-2
         ncc(2,2)=0
        else
         ncj(1,nj)=nj-2
         ncj(1,nj-2)=nj
         ncj(2,nj)=0
         ncj(2,nj-2)=0
         ncc(1,2)=nj-1
         ncc(2,2)=0
        endif
       endif
      endif

      if(jqq.ne.0)then
       if(iqq.ne.0.or.iqq.eq.0.and.jqq.eq.3)then
        sjqq1=qgjit(qt0,qt0,wpi1*wmi1,2,2)
        gbs=sjqq1/sjqq
       else
        sjqg1=qgjit(qt0,qt0,wpi1*wmi1,1,2)
        gbs=sjqg1/sjqg
       endif
       if(qgran(b10).gt.gbs)goto 11
      endif
      wpi=wpi1
      wmi=wmi1

      ept(1)=.5d0*(wpi+wmi)      !ladder 4-momentum
      ept(2)=.5d0*(wpi-wmi)
      ept(3)=0.d0
      ept(4)=0.d0
      qmin(1)=qt0                !q^2 cutoff for the upper leg
      qmin(2)=qt0                !q^2 cutoff for the downer leg
      qminn=max(qmin(1),qmin(2)) !overall q^2 cutoff
      si=qgnrm(ept)
      jini=1
      jj=int(1.5d0+qgran(b10)) !1st parton at upper (jj=1) or downer (jj=2) leg

3     continue

      aaa=qgnrm(ept)             !ladder mass squared
      if(debug.ge.3)write (moniou,203)si,iqc,ept,aaa

      pt2=ept(3)**2+ept(4)**2
      pt=dsqrt(pt2)
      ww=si+pt2

      iqp(1)=min(1,iabs(iqc(1)))+1
      iqp(2)=min(1,iabs(iqc(2)))+1
      wp(1)=ept(1)+ept(2)                 !LC+ for the ladder
      wp(2)=ept(1)-ept(2)                 !LC- for the ladder
      s2min=4.d0*fqscal*qminn   !minimal energy squared for 2-parton production
      if(jini.eq.1)then                   !general ladder
       sj=qgjit(qmin(jj),qmin(3-jj),si,iqp(jj),iqp(3-jj))   !total ladder contribution
       sj1=qgjit1(qmin(3-jj),qmin(jj),si,iqp(3-jj),iqp(jj)) !one-way ordered
       sjb=qgbit(qmin(1),qmin(2),si,iqp(1),iqp(2))          !born contribution
       aks=qgran(b10)
       if(aks.lt.sjb/sj)then
        goto 6      !born process sampled
       elseif(aks.lt.sj1/sj)then       !change to one-way ordered ladder
        jj=3-jj
        sj=sj1
        jini=0
       endif
      else                                !one-way ordered ladder
       sj=qgjit1(qmin(jj),qmin(3-jj),si,iqp(jj),iqp(3-jj)) !one-way ordered
       sjb=qgbit(qmin(1),qmin(2),si,iqp(1),iqp(2))         !born contribution
       if(qgran(b10).lt.sjb/sj)goto 6      !born process sampled
      endif
      wwmin=(s2min+qmin(jj)+pt2-2.d0*pt*dsqrt(qmin(jj)*epsxmn))
     */(1.d0-epsxmn)           !minimal energy squared for 3-parton production

      if(debug.ge.3)write (moniou,204)s2min,wwmin,sj,sjb

      if(ww.lt.1.1d0*wwmin)goto 6         !energy too low -> born process

      xxx=pt*dsqrt(qmin(jj))/ww
      xmin=(s2min+qmin(jj)+pt2)/ww
      xmin=xmin-2.d0*xxx*(xxx+dsqrt(xxx**2+1.d0-xmin))

      xmax=1.d0-epsxmn
      if(debug.ge.3)write (moniou,205)xmin,xmax

      qqmax=(pt*dsqrt(epsxmn)+dsqrt(max(0.d0,pt2*epsxmn
     *+(1.d0+4.d0*fqscal)*(xmax*ww-pt2))))/(1.d0+4.d0*fqscal)
      qqmin=qmin(jj)        !minimal parton virtuality in the current rung
      if(debug.ge.3)write (moniou,206)qqmin,qqmax

      qm0=qqmin
      xm0=xmax
      s2max=xm0*ww

      if(jini.eq.1)then
       sj0=qgjit(qm0,qmin(3-jj),s2max,1,iqp(3-jj))*qgfap(xm0,iqp(jj),1)
     * +qgjit(qm0,qmin(3-jj),s2max,2,iqp(3-jj))*qgfap(xm0,iqp(jj),2)
      else
       sj0=qgjit1(qm0,qmin(3-jj),s2max,1,iqp(3-jj))
     * *qgfap(xm0,iqp(jj),1)
     * +qgjit1(qm0,qmin(3-jj),s2max,2,iqp(3-jj))*qgfap(xm0,iqp(jj),2)
      endif

      gb0=sj0*qm0*qgalf(qm0/alm)*qgsudx(qm0,iqp(jj)) *4.5d0  !normal. of accept.
      if(xm0.le..5d0)then
       gb0=gb0*xm0**(1.d0-delh)
      else
       gb0=gb0*(1.d0-xm0)*2.d0**delh
      endif
      if(debug.ge.3)write (moniou,208)xm0,xmin,xmax,gb0

      xmin2=max(.5d0,xmin)
      xmin1=xmin**delh
      xmax1=min(xmax,.5d0)**delh
      if(xmin.ge..5d0)then                             !choose proposal function
       djl=1.d0
      elseif(xmax.lt..5d0)then
       djl=0.d0
      else
       djl=1.d0/(1.d0+((2.d0*xmin)**delh-1.d0)/delh
     * /dlog(2.d0*(1.d0-xmax)))
      endif

c-------------------------------------------------
c propose x, q^2
4     continue
      if(qgran(b10).gt.djl)then
       x=(xmin1+qgran(b10)*(xmax1-xmin1))**(1.d0/delh) !parton LC share
      else
       x=1.d0-(1.d0-xmin2)*((1.d0-xmax)/(1.d0-xmin2))**qgran(b10)
      endif
      qq=qqmin/(1.d0+qgran(b10)*(qqmin/qqmax-1.d0))    !parton virtuality
      qt2=qq*(1.d0-x)                                  !parton p_t^2
      if(debug.ge.4)write (moniou,209)qq,qqmin,qqmax,x,qt2

      if(qq.gt.qminn)then                  !update virtuality cutoff
       qmin2=qq
      else
       qmin2=qminn
      endif
      qt=dsqrt(qt2)
      call qgcs(c,s)
      ep3(3)=qt*c                          !final parton p_x, p_y
      ep3(4)=qt*s
      pt2new=(ept(3)-ep3(3))**2+(ept(4)-ep3(4))**2!p_t^2 for the remained ladder
      s2min2=max(s2min,4.d0*fqscal*qmin2)  !new ladder kinematic limit
      s2=x*ww-qt2*x/(1.d0-x)-pt2new        !mass squared for the remained ladder
      if(s2.lt.s2min2)goto 4           !ladder mass below threshold -> rejection

      if(jini.eq.1)then                    !weights for g- and q-legs
       sj1=qgjit(qq,qmin(3-jj),s2,1,iqp(3-jj))*qgfap(x,iqp(jj),1)
       sj2=qgjit(qq,qmin(3-jj),s2,2,iqp(3-jj))*qgfap(x,iqp(jj),2)
      else
       sj1=qgjit1(qq,qmin(3-jj),s2,1,iqp(3-jj))*qgfap(x,iqp(jj),1)
       sj2=qgjit1(qq,qmin(3-jj),s2,2,iqp(3-jj))*qgfap(x,iqp(jj),2)
      endif
      gb7=(sj1+sj2)*qgalf(qq/alm)*qq*qgsudx(qq,iqp(jj))/gb0  /2.d0
                               !acceptance probability for x and q**2 simulation
      if(x.le..5d0)then
       gb7=gb7*x**(1.d0-delh)
      else
       gb7=gb7*(1.d0-x)*2.d0**delh
      endif
      if(debug.ge.4)write (moniou,210)gb7,s2,sj1,sj2,jj,jini
      if(qgran(b10).gt.gb7)goto 4          !rejection

c-------------------------------------------------
c define color flow for the emitted jet; perform final state emission
      nqc(2)=0
      if(qgran(b10).lt.sj1/(sj1+sj2))then         !new gluon-leg ladder
       if(iqc(jj).eq.0)then                       !g -> gg
        jt=1
        jq=int(1.5d0+qgran(b10))
        nqc(1)=ncc(jq,jj)                         !color connection for the jet
        nqc(2)=0
       else                                       !q -> qg
        jt=2
        if(iqc(jj).gt.0)then                      !orientation of color flow
         jq=1
        else
         jq=2
        endif
        nqc(1)=0
        ncc(jq,jj)=ncc(1,jj)                      !color connection for the jet
       endif
       iq1=iqc(jj)                                !jet flavor (type)
       iqc(jj)=0                                  !new ladder leg flavor (type)

      else                                        !new quark-leg ladder
       if(iqc(jj).ne.0)then                       !q -> gq
        iq1=0
        jt=3
        if(iqc(jj).gt.0)then                      !orientation of color flow
         jq=1
        else
         jq=2
        endif
        nqc(1)=ncc(1,jj)                          !color connection for the jet
        nqc(2)=0

       else                                       !g -> qq~
        jq=int(1.5d0+qgran(b10))                  !orientation of color flow
        iq1=int(3.d0*qgran(b10)+1.d0)*(3-2*jq)    !jet flavor (type)
        iqc(jj)=-iq1                              !new ladder leg flavor (type)
        jt=4
        nqc(1)=ncc(jq,jj)                         !color connections for the jet
        ncc(1,jj)=ncc(3-jq,jj)
       endif
      endif
      if(debug.ge.3)write (moniou,211)jt

      call qgcjet(qt2,iq1,qv1,zv1,qm1,iqv1,ldau1,lpar1,jq) !final state emission
      si=x*ww-(qt2+qm1(1,1))*x/(1.d0-x)-pt2new  !mass squared for the new ladder
      if(si.gt.s2min2)then
       iq=min(1,iabs(iqc(jj)))+1
       if(jini.eq.1)then
        gb=qgjit(qq,qmin(3-jj),si,iq,iqp(3-jj))
     *  /qgjit(qq,qmin(3-jj),s2,iq,iqp(3-jj))
       else
        gb=qgjit1(qq,qmin(3-jj),si,iq,iqp(3-jj))
     *  /qgjit1(qq,qmin(3-jj),s2,iq,iqp(3-jj))
       endif
       if(qgran(b10).gt.gb)goto 1        !jet mass correction for the acceptance
      else                                        !below threshold -> rejection
       goto 1
      endif

      wp3=wp(jj)*(1.d0-x)
      wm3=(qt2+qm1(1,1))/wp3
      ep3(1)=.5d0*(wp3+wm3)                       !jet 4-momentum
      ep3(2)=.5d0*(wp3-wm3)*(3-2*jj)
      call qgrec(ep3,nqc,qv1,zv1,qm1,iqv1,ldau1,lpar1,jq)
                               !reconstruction of 4-momenta of all final partons
c-------------------------------------------------
c define color connections for the new ladder
      if(jt.eq.1)then
       if(ncc(1,jj).eq.0.and.ncc(2,jj).eq.0)ncc(3-jq,jj)=nqc(1)
       ncc(jq,jj)=nqc(2)
      elseif(jt.eq.2)then
       ncc(3-jq,jj)=nqc(1)
      elseif(jt.eq.3)then
       ncc(1,jj)=nqc(2)
      elseif(jt.eq.4.and.ncc(1,jj).eq.0.and.ncc(2,jj).eq.0)then
       ncc(1,jj)=nqc(1)
      endif

      if(iabs(iq1).eq.3)then
       iqqq=8+iq1/3*4
      else
       iqqq=8+iq1
      endif
      if(debug.ge.3)write (moniou,212)tyq(iqqq),qt2,ep3
      do i=1,4
       ept(i)=ept(i)-ep3(i)                       !new ladder 4-momentum
      enddo
      qmin(jj)=qq                                 !new virtuality cutoffs
      qminn=qmin2
      goto 3                                      !consider next parton emission

c------------------------------------------------
c born process - last parton pair production in the ladder
6     continue
      if(debug.ge.2)write (moniou,214)si,qminn,iqc
      tmin=qminn*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qminn*fqscal/si)))
      qtmin=tmin*(1.d0-tmin/si)
      if(iqc(1).ne.0.or.iqc(2).ne.0)then
       gb0=tmin**2*qgalf(qtmin/fqscal/alm)**2
     * *qgfbor(si,tmin,iqc(1),iqc(2),1)    *1.1d0
      else
       gb0=.25d0*si**2*qgalf(qtmin/fqscal/alm)**2
     * *qgfbor(si,.5d0*si,iqc(1),iqc(2),1)
      endif
      gb0=gb0*qgsudx(qtmin/fqscal,iqp(1))*qgsudx(qtmin/fqscal,iqp(2))
                                                    !normalization of acceptance
      if(debug.ge.3)write (moniou,215)gb0

7     q2=tmin/(1.d0-qgran(b10)*(1.d0-2.d0*tmin/si))   !proposed q^2
      z=q2/si                                         !parton LC momentum share
      qt2=q2*(1.d0-z)                                 !parton p_t^2
      if(qgran(b10).lt..5d0)then
       jm=2
       tq=si-q2
      else
       jm=1
       tq=q2
      endif
      gb=q2**2*qgalf(qt2/fqscal/alm)**2*qgfbor(si,tq,iqc(1),iqc(2),1)
     **qgsudx(qt2/fqscal,iqp(1))*qgsudx(qt2/fqscal,iqp(2))/gb0
                                                      !acceptance probabilty
      if(debug.ge.4)write (moniou,216)gb,q2,z,qt2
      if(qgran(b10).gt.gb)goto 7                      !rejection

c-------------------------------------------------
c define color connections for the 1st emitted jet
      nqc(2)=0
      if(iqc(1).eq.0.and.iqc(2).eq.0)then             !gg-process
       jq=int(1.5d0+qgran(b10))                       !orientation of color flow
       nqc(1)=ncc(jq,jm)

       if(qgran(b10).lt..5d0)then
        jt=1                                          !gg -> gg
        nqc(2)=0
        njc1=ncc(3-jq,jm)                         !color connections for 1st jet
        njc2=ncc(jq,3-jm)
        if(ncc(1,1).eq.0.and.ncc(2,1).eq.0)then
         if(jm.eq.1)nqc(1)=njc2
        else
         if(iqj(njc1).ne.0)then
          ncj(1,njc1)=njc2
         else
          ncj(jq,njc1)=njc2
         endif
         if(iqj(njc2).ne.0)then
          ncj(1,njc2)=njc1
         else
          ncj(3-jq,njc2)=njc1
         endif
        endif
       else                                 !gg -> gg (inverse color connection)
        jt=2
        nqc(2)=ncc(3-jq,3-jm)
       endif

      elseif(iqc(1)*iqc(2).eq.0)then                  !qg -> qg
       if(iqc(1)+iqc(2).gt.0)then                     !orientation of color flow
        jq=1
       else
        jq=2
       endif
       if(qgran(b10).lt..5d0)then
        if(iqc(jm).eq.0)then
         jt=3
         nqc(1)=ncc(jq,jm)
         nqc(2)=0
         njc1=ncc(3-jq,jm)
         njc2=ncc(1,3-jm)
         if(ncc(1,jm).eq.0.and.ncc(2,jm).eq.0)then
          nqc(1)=njc2
         else
          if(iqj(njc1).ne.0)then
           ncj(1,njc1)=njc2
          else
           ncj(jq,njc1)=njc2
          endif
          if(iqj(njc2).ne.0)then
           ncj(1,njc2)=njc1
          else
           ncj(3-jq,njc2)=njc1
          endif
         endif
        else
         jt=4
         nqc(1)=0
         njc1=ncc(1,jm)
         njc2=ncc(3-jq,3-jm)
         if(njc2.ne.0)then
          if(iqj(njc1).ne.0)then
           ncj(1,njc1)=njc2
          else
           ncj(3-jq,njc1)=njc2
          endif
          if(iqj(njc2).ne.0)then
           ncj(1,njc2)=njc1
          else
           ncj(jq,njc2)=njc1
          endif
         endif
        endif
       else
        if(iqc(jm).eq.0)then
         jt=5
         nqc(2)=ncc(3-jq,jm)
         nqc(1)=ncc(1,3-jm)
        else
         jt=6
         nqc(1)=ncc(jq,3-jm)
        endif
       endif

      elseif(iqc(1)*iqc(2).gt.0)then                  !qq (q~q~) -> qq (q~q~)
       jt=7
       if(iqc(1).gt.0)then
        jq=1
       else
        jq=2
       endif
       nqc(1)=ncc(1,3-jm)
      else                                            !qq~ -> qq~
       jt=8
       if(iqc(jm).gt.0)then
        jq=1
       else
        jq=2
       endif
       nqc(1)=0
       njc1=ncc(1,jm)
       njc2=ncc(1,3-jm)
       if(iqj(njc1).ne.0)then
        ncj(1,njc1)=njc2
       else
        ncj(3-jq,njc1)=njc2
       endif
       if(iqj(njc2).ne.0)then
        ncj(1,njc2)=njc1
       else
        ncj(jq,njc2)=njc1
       endif
      endif
      if(jt.ne.8)then
       jq2=jq
      else
       jq2=3-jq
      endif
      if(debug.ge.3)write (moniou,211)jt
      call qgcjet(qt2,iqc(jm),qv1,zv1,qm1,iqv1,ldau1,lpar1,jq)!final state emis.
      call qgcjet(qt2,iqc(3-jm),qv2,zv2,qm2,iqv2,ldau2,lpar2,jq2)
      amt1=qt2+qm1(1,1)
      amt2=qt2+qm2(1,1)
      if(dsqrt(si).gt.dsqrt(amt1)+dsqrt(amt2))then
       z=qgtwd(si,amt1,amt2)
      else
       if(debug.ge.4)write (moniou,217)dsqrt(si),dsqrt(amt1),dsqrt(amt2)
       goto 1                                      !below threshold -> rejection
      endif

      call qgdeft(si,ept,ey)
      wp3=z*dsqrt(si)
      wm3=(qt2+qm1(1,1))/wp3
      ep3(1)=.5d0*(wp3+wm3)                        !1st jet 4-momentum
      ep3(2)=.5d0*(wp3-wm3)
      qt=dsqrt(qt2)
      call qgcs(c,s)
      ep3(3)=qt*c
      ep3(4)=qt*s

      call qgtran(ep3,ey,1)
      call qgrec(ep3,nqc,qv1,zv1,qm1,iqv1,ldau1,lpar1,jq)
                               !reconstruction of 4-momenta of all final partons
      if(iabs(iqc(jm)).eq.3)then
       iqqq=8+iqc(jm)/3*4
      else
       iqqq=8+iqc(jm)
      endif
      if(debug.ge.3)write (moniou,212)tyq(iqqq),qt2,ep3

      wp3=(1.d0-z)*dsqrt(si)
      wm3=(qt2+qm2(1,1))/wp3
      ep3(1)=.5d0*(wp3+wm3)                        !2nd jet 4-momentum
      ep3(2)=.5d0*(wp3-wm3)
      ep3(3)=-qt*c
      ep3(4)=-qt*s
      call qgtran(ep3,ey,1)

c-------------------------------------------------
c define color connections for the 2nd emitted jet
      if(jt.eq.1)then
       nqc(1)=nqc(2)
       if(ncc(1,3-jm).eq.0.and.ncc(2,3-jm).eq.0)then
        nqc(2)=ncc(3-jq,jm)
       else
        nqc(2)=ncc(3-jq,3-jm)
       endif
      elseif(jt.eq.2)then
       if(ncc(1,1).eq.0.and.ncc(2,1).eq.0)then
        if(jm.eq.1)then
         nqc(2)=nqc(1)
         nqc(1)=ncc(jq,3-jm)
        else
         nqc(1)=nqc(2)
         nqc(2)=ncc(3-jq,jm)
        endif
       else
        nqc(2)=ncc(3-jq,jm)
        nqc(1)=ncc(jq,3-jm)
       endif
      elseif(jt.eq.3)then
       nqc(1)=nqc(2)
      elseif(jt.eq.4)then
       nqc(2)=nqc(1)
       if(ncc(1,1).eq.0.and.ncc(2,1).eq.0)then
        nqc(1)=ncc(1,jm)
       else
        nqc(1)=ncc(jq,3-jm)
       endif
      elseif(jt.eq.5)then
       if(ncc(1,jm).eq.0.and.ncc(2,jm).eq.0)then
        nqc(1)=nqc(2)
       else
        nqc(1)=ncc(jq,jm)
       endif
      elseif(jt.eq.6)then
       if(ncc(1,3-jm).eq.0.and.ncc(2,3-jm).eq.0)then
        nqc(2)=nqc(1)
       else
        nqc(2)=ncc(3-jq,3-jm)
       endif
       nqc(1)=ncc(1,jm)
      elseif(jt.eq.7)then
       nqc(1)=ncc(1,jm)
      endif
      call qgrec(ep3,nqc,qv2,zv2,qm2,iqv2,ldau2,lpar2,jq2)
                               !reconstruction of 4-momenta of all final partons
      if(iabs(iqc(3-jm)).eq.3)then
       iqqq=8+iqc(3-jm)/3*4
      else
       iqqq=8+iqc(3-jm)
      endif
      if(debug.ge.3)write (moniou,212)tyq(iqqq),qt2,ep3

      ebal(1)=.5d0*(wpp+wpm)                          !balans of 4-momentum
      ebal(2)=.5d0*(wpp-wpm)
      ebal(3)=0.d0
      ebal(4)=0.d0
      do i=nj0+1,nj
       if(iqq.eq.0.or.iqq.eq.1.and.i.ne.nva.or.iqq.eq.2
     * .and.i.ne.nvb.or.iqq.eq.3.and.i.ne.nva.and.i.ne.nvb)then
        do j=1,4
         ebal(j)=ebal(j)-eqj(j,i)
        enddo
       endif
      enddo
      if(debug.ge.2)write (moniou,218)nj
      if(debug.ge.5)write (moniou,219)ebal
      if(debug.ge.1)write (moniou,220)

201   format(2x,'qghot - semihard interaction:'/
     *4x,'type of the interaction - ',i2/
     *4x,'initial light cone momenta - ',2e10.3/
     *4x,'remnant types - ',2i3,2x,'diffr. eigenstates - ',2i2/
     *4x,'proj. class - ',i2,2x,'Pomeron type - ',i2/
     *4x,'initial number of final partons - ',i4)
202   format(2x,'qghot: mass squared for parton ladder - ',e10.3)
203   format(2x,'qghot: ',' mass squared for the laddder:',e10.3/
     *4x,'ladder end flavors:',2i3/4x,'ladder 5-momentum: ',5e10.3)
204   format(2x,'qghot: kinematic bounds s2min=',e10.3,
     *2x,'wwmin=',e10.3/4x,'jet cross section sj=',e10.3,
     *2x,'born cross section sjb=',e10.3)
205   format(2x,'qghot: xmin=',e10.3,2x,'xmax=',e10.3)
206   format(2x,'qghot: qqmin=',e10.3,2x,'qqmax=',e10.3)
208   format(2x,'qghot: xm0=',e10.3,2x,'xmin=',e10.3,2x,
     *'xmax=',e10.3,2x,'gb0=',e10.3)
209   format(2x,'qghot: qq=',e10.3,2x,'qqmin=',e10.3,2x,
     *'qqmax=',e10.3,2x,'x=',e10.3,2x,'qt2=',e10.3)
210   format(2x,'qghot: gb7=',e10.3,2x,'s2=',e10.3,2x,'sj1=',e10.3
     *,2x,'sj2=',e10.3,2x,'jj=',i2,2x,'jini=',i2)
211   format(2x,'qghot: colour connection jt=:',i1)
212   format(2x,'qghot: new jet flavor:',a2,
     *' pt squared for the jet:',e10.3/4x,'jet 4-momentum:',4e10.3)
214   format(2x,'qghot - highest virtuality subprocess in the ladder:'/
     *4x,'mass squared for the process:',e10.3/4x,'q^2-cutoff:',e10.3
     *,2x,'iqc=',2i3)
215   format(2x,'qghot - normalization of acceptance:',' gb0=',e10.3)
216   format(2x,'qghot - acceptance probabilty:'/
     *4x,'gb=',e10.3,2x,'q2=',e10.3,2x,'z=',e10.3,2x,'qt2=',e10.3)
217   format(2x,'qghot: ecm=',e10.3,2x,'mt1=',e10.3,2x,'mt2=',e10.3)
218   format(2x,'qghot: total number of jets - ',i4)
219   format(2x,'qghot: 4-momentum balans - ',4e10.3)
220   format(2x,'qghot - end')
      return
      end

c------------------------------------------------------------------------
      function npgen(vv,npmin,npmax)
c-----------------------------------------------------------------------
c npgen -  Poisson distribution
c vv    - average number
c npmin - minimal number
c npmax - maximal number
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr11/ b10
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(npmin.eq.0)then
       aks=qgran(b10)
       vvn=exp(-vv)
       do n=1,npmax
         aks=aks-vvn
        if(aks.lt.0.d0)goto 1
         vvn=vvn*vv/dble(n)
       enddo
      elseif(npmin.eq.1)then
       aks=qgran(b10)*(1.d0-exp(-vv))
       vvn=exp(-vv)
       do n=1,npmax
         vvn=vvn*vv/dble(n)
         aks=aks-vvn
        if(aks.lt.0.d0)goto 2
       enddo
      elseif(npmin.eq.2)then
       aks=qgran(b10)*(1.d0-exp(-vv)*(1.d0+vv))
       vvn=vv*exp(-vv)
       do n=2,npmax
         vvn=vvn*vv/dble(n)
         aks=aks-vvn
        if(aks.lt.0.d0)goto 2
       enddo
      else
       stop'npgen'
      endif
1     n=n-1
2     npgen=n
      return
      end

c=============================================================================
      subroutine qglead(wppr0,wmtg0,lqa,lqb,lqa0,lqb0,lva,lvb
     *,izp,izt,ila,ilb,iret)
c-------------------------------------------------------------------------
c qglead-treatment of leading hadron states
c-------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(njmax=50000)
      common /qgdebug/ debug
      common /qgarr37/ eqj(4,njmax),iqj(njmax),ncj(2,njmax),nj

      iret=0
      if(lqa0.eq.0.and.lqb0.eq.0)then
       if(lva.eq.0.and.lvb.eq.0)then
        call qgdifr(wppr0,wmtg0,izp,izt,lqa,lqb,iret)
       elseif(lva.eq.0)then
        call qgdifr(wppr0,wmtg0,izp,izt,lqa,-1,iret)
       elseif(lvb.eq.0)then
        call qgdifr(wppr0,wmtg0,izp,izt,-1,lqb,iret)
       endif
       if(lva.eq.1)then
        eqj(1,ila)=.5d0*wppr0
        eqj(2,ila)=eqj(1,ila)
        eqj(3,ila)=0.d0
        eqj(4,ila)=0.d0
       endif
       if(lvb.eq.1)then
        eqj(1,ilb)=.5d0*wmtg0
        eqj(2,ilb)=-eqj(1,ilb)
        eqj(3,ilb)=0.d0
        eqj(4,ilb)=0.d0
       endif
      elseif(lqa0.eq.0)then
       if(lva.eq.0)then
        call qgdifr(wppr0,wmtg0,izp,izt,lqa,-1,iret)
       else
        eqj(1,ila)=.5d0*wppr0
        eqj(2,ila)=eqj(1,ila)
        eqj(3,ila)=0.d0
        eqj(4,ila)=0.d0
       endif
      elseif(lqb0.eq.0)then
       if(lvb.eq.0)then
        call qgdifr(wppr0,wmtg0,izp,izt,-1,lqb,iret)
       else
        eqj(1,ilb)=.5d0*wmtg0
        eqj(2,ilb)=-eqj(1,ilb)
        eqj(3,ilb)=0.d0
        eqj(4,ilb)=0.d0
       endif
      endif
      return
      end

c=============================================================================
      double precision function qgbit(qi,qj,s,m,l)
c------------------------------------------------------------------------
c qgbit - born cross-section interpolation
c qi,qj - effective momentum cutoffs for the scattering,
c s - total c.m. energy squared for the scattering,
c m - parton type at current end of the ladder (1 - g, 2 - q)
c l - parton type at opposite end of the ladder (1 - g, 2 - q)
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wi(3),wk(3)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr20/ spmax
      common /qgarr26/ factk,fqscal
      common /qgarr31/ csj(40,160)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)qi,qj,s,m,l
      qgbit=0.d0
      qq=max(qi,qj)
      s2min=qq*4.d0*fqscal
      if(s.le..99d0*s2min)then
       if(debug.ge.3)write (moniou,202)qgbit
       return
      endif

      tmin=qq*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qq*fqscal/s)))
      ml=40*(m-1)+80*(l-1)
      qli=dlog(qq)/dlog(spmax/4.d0/fqscal)*39.d0+1.d0
      sl=dlog(s/s2min)/dlog(spmax/s2min)*39.d0+1.d0
      i=min(38,int(qli))
      k=min(38,int(sl))

      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)
      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
      wi(1)=1.d0-wi(2)+wi(3)
      wi(2)=wi(2)-2.d0*wi(3)
      do k1=1,3
       k2=k+k1-1+ml
      do i1=1,3
       qgbit=qgbit+csj(i+i1-1,k2)*wi(i1)*wk(k1)
      enddo
      enddo
      qgbit=exp(qgbit)*(1.d0/tmin-2.d0/s)
      if(qi.lt.qq)qgbit=qgbit*qgsudx(qq,m)/qgsudx(qi,m)
      if(qj.lt.qq)qgbit=qgbit*qgsudx(qq,l)/qgsudx(qj,l)

      if(debug.ge.3)write (moniou,202)qgbit
201   format(2x,'qgbit: qi=',e10.3,2x,'qj=',e10.3
     *,2x,'s= ',e10.3,2x,'m= ',i1,2x,'l= ',i1)
202   format(2x,'qgbit=',e10.3)
      return
      end

c=============================================================================
      double precision function qgfbor(s,t,iq1,iq2,n)
c---------------------------------------------------------------------------
c qgfbor - integrand for the born cross-section (matrix element squared)
c s - total c.m. energy squared for the scattering,
c t - invariant variable for the scattering abs[(p1-p3)**2],
c iq1 - parton type at current end of the ladder (0 - g, 1,2 - q)
c iq2 - parton type at opposite end of the ladder (0 - g, 1,2 - q)
c---------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)s,t,iq1,iq2

      u=s-t
c... initialize
      qgfbor=0.0
      if(n.eq.1)then
       if(iq1.eq.0.and.iq2.eq.0)then        !gluon-gluon
        qgfbor=(3.d0-t*u/s**2+s*u/t**2+s*t/u**2)*4.5d0
       elseif(iq1*iq2.eq.0)then             !gluon-quark
        qgfbor=(s**2+u**2)/t**2+(s/u+u/s)/2.25d0
       elseif(iq1.eq.iq2)then               !quark-quark (same flavor)
        qgfbor=((s**2+u**2)/t**2+(s**2+t**2)/u**2)/2.25d0
     *  -s**2/t/u/3.375d0
       elseif(iq1+iq2.eq.0)then             !quark-antiquark (same flavor)
        qgfbor=((s**2+u**2)/t**2+(u**2+t**2)/s**2)/2.25d0
     *  +u**2/t/s/3.375d0
       else                                 !quark-antiquark (different flavors)
        qgfbor=(s**2+u**2)/t**2/2.25d0
       endif
      elseif(n.eq.2)then
       if(iq1.eq.0.and.iq2.eq.0)then        !gluon-gluon->quark-antiquark
        qgfbor=.5d0*(t/u+u/t)-1.125d0*(t*t+u*u)/s**2
       elseif(iq1+iq2.eq.0)then             !quark-antiquark->quark-antiquark
        qgfbor=(t*t+u*u)/s**2/1.125d0       !(different flavor)
       else
        qgfbor=0.d0
       endif
      elseif(n.eq.3)then
       if(iq1.ne.0.and.iq1+iq2.eq.0)then    !quark-antiquark->gluon-gluon
        qgfbor=32.d0/27.d0*(t/u+u/t)-(t*t+u*u)/s**2/.375d0
       else
        qgfbor=0.d0
       endif
      endif

      if(debug.ge.2)write (moniou,202)qgfbor
201   format(2x,'qgfbor - hard scattering matrix element squared:'/
     *4x,'s=',e10.3,2x,'|t|=',e10.3,2x,'iq1=',i1,2x,'iq2=',i1)
202   format(2x,'qgfbor=',e10.3)
      return
      end

c=============================================================================
      double precision function qgborn(qi,qj,s,iq1,iq2)
c-----------------------------------------------------------------------------
c qgborn - hard 2->2 parton scattering born cross-section
c s is the c.m. energy square for the scattering process,
c iq1 - parton type at current end of the ladder (0 - g, 1,2 etc. - q)
c iq2 - parton type at opposite end of the ladder (0 - g, 1,2 etc. - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/  debug
      common /arr3/  x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)qi,qj,s,iq1,iq2

      qgborn=0.d0
      qq=max(qi,qj)
      tmin=qq*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qq*fqscal/s)))
      do i=1,7
      do m=1,2
       t=2.d0*tmin/(1.d0+2.d0*tmin/s-x1(i)*(2*m-3)*(1.d0-2.d0*tmin/s))
       qt=t*(1.d0-t/s)

       fb=0.d0
       do n=1,3
        fb=fb+qgfbor(s,t,iq1,iq2,n)+qgfbor(s,s-t,iq1,iq2,n)
       enddo
       fb=fb*qgsudx(qt/fqscal,iabs(iq1)+1)
     * *qgsudx(qt/fqscal,iabs(iq2)+1)

       qgborn=qgborn+a1(i)*fb*qgalf(qt/fqscal/alm)**2*t**2
      enddo
      enddo
      qgborn=qgborn*2.d0*pi**3/s**2

      qgborn=qgborn/qgsudx(qi,iabs(iq1)+1)/qgsudx(qj,iabs(iq2)+1)
      if(iq1.eq.iq2)qgborn=qgborn*.5d0

      if(debug.ge.3)write (moniou,202)qgborn
201   format(2x,'qgborn: qi=',e10.3,2x,'qj=',e10.3,2x,
     *'s= ',e10.3,2x,'iq1= ',i1,2x,'iq2= ',i1)
202   format(2x,'qgborn=',e10.3)
      return
      end

c=============================================================================
      subroutine qgcjet(qq,iq1,qv,zv,qm,iqv,ldau,lpar,jq)
c-----------------------------------------------------------------------------
c final state emission process (all branchings as well as parton masses
c are determined)
c qq - maximal effective momentum transfer for the first branching
c iq1 - initial jet flavour (0 - for gluon)
c qv(i,j) - effective momentum for the branching of the parton in i-th row
c on j-th level (0 - in case of no branching)  - to be determined
c zv(i,j) - z-value for the branching of the parton in i-th row
c on j-th level - to be determined
c qm(i,j) - mass squared for the parton in i-th row
c on j-th level - to be determined
c iqv(i,j) - flavour for the parton in i-th row on j-th level
c - to be determined
c ldau(i,j) - first daughter row for the branching of the parton in i-th row
c on j-th level - to be determined
c lpar(i,j) - the parent row for the parton in i-th row
c on j-th level - to be determined
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension qmax(30,50),iqm(2),lnv(50),
     *qv(30,50),zv(30,50),qm(30,50),iqv(30,50),
     *ldau(30,49),lpar(30,50)
      common /qgarr11/ b10
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)qq,iq1,jq

      do i=2,20
       lnv(i)=0
      enddo
      lnv(1)=1
      qmax(1,1)=qq
      iqv(1,1)=iq1
      nlev=1
      nrow=1

2     qlmax=dlog(qmax(nrow,nlev)/qtf/16.d0)
      iq=min(1,iabs(iqv(nrow,nlev)))+1

      if(qgran(b10).gt.qgsudi(qlmax,iq))then
       q=qgqint(qlmax,qgran(b10),iq)
       z=qgzsim(q,iq)
       ll=lnv(nlev+1)+1
       ldau(nrow,nlev)=ll
       lpar(ll,nlev+1)=nrow
       lpar(ll+1,nlev+1)=nrow
       lnv(nlev+1)=ll+1

       if(iq.ne.1)then
        if((3-2*jq)*iqv(nrow,nlev).gt.0)then
         iqm(1)=0
         iqm(2)=iqv(nrow,nlev)
        else
         iqm(2)=0
         iqm(1)=iqv(nrow,nlev)
         z=1.d0-z
        endif
       else
        wg=qgfap(z,1,1)
        wg=wg/(wg+qgfap(z,1,2))
        if(qgran(b10).lt.wg)then
         iqm(1)=0
         iqm(2)=0
        else
         iqm(1)=int(3.d0*qgran(b10)+1.d0)*(3-2*jq)
         iqm(2)=-iqm(1)
        endif
        if(qgran(b10).lt..5d0)z=1.d0-z
       endif
       qv(nrow,nlev)=q
       zv(nrow,nlev)=z
       nrow=ll
       nlev=nlev+1
       qmax(nrow,nlev)=q*z**2
       qmax(nrow+1,nlev)=q*(1.d0-z)**2
       iqv(nrow,nlev)=iqm(1)
       iqv(nrow+1,nlev)=iqm(2)
       if(debug.ge.3)write (moniou,203)nlev,nrow,q,z
       goto 2
      else
       qv(nrow,nlev)=0.d0
       zv(nrow,nlev)=0.d0
       qm(nrow,nlev)=0.d0
       if(debug.ge.3)write (moniou,204)nlev,nrow
      endif

3     continue
      if(nlev.eq.1)then
       if(debug.ge.3)write (moniou,202)
       return
      endif

      lprow=lpar(nrow,nlev)
      if(ldau(lprow,nlev-1).eq.nrow)then
       nrow=nrow+1
       goto 2
      else
       z=zv(lprow,nlev-1)
       qm(lprow,nlev-1)=z*(1.d0-z)*qv(lprow,nlev-1)
     * +qm(nrow-1,nlev)/z+qm(nrow,nlev)/(1.d0-z)
       nrow=lprow
       nlev=nlev-1
       if(debug.ge.3)write (moniou,205)nlev,nrow,qm(lprow,nlev)
       goto 3
      endif

201   format(2x,'qgcjet: qq=',e10.3,2x,'iq1= ',i1,2x,'jq=',i1)
202   format(2x,'qgcjet - end')
203   format(2x,'qgcjet: new branching at level nlev=',i2,' nrow=',i2
     */4x,' effective momentum q=',e10.3,2x,' z=',e10.3)
204   format(2x,'qgcjet: new final jet at level nlev=',i2,' nrow=',i2)
205   format(2x,'qgcjet: jet mass at level nlev=',i2,' nrow=',i2
     *,' - qm=',e10.3)
      end

c===========================================================================
      subroutine qgcs(c,s)
c---------------------------------------------------------------------------
c c,s - cos and sin generation for uniformly distributed angle 0<fi<2*pi
c---------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr11/ b10
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)
1     s1=2.d0*qgran(b10)-1.d0
      s2=2.d0*qgran(b10)-1.d0
      s3=s1*s1+s2*s2
      if(s3.gt.1.d0)goto 1
      s3=dsqrt(s3)
      c=s1/s3
      s=s2/s3

      if(debug.ge.3)write (moniou,202)c,s
201   format(2x,'qgcs - cos(fi) and sin(fi) are generated',
     *' (0<fi<2*pi)')
202   format(2x,'qgcs: c=',e10.3,2x,'s=',e10.3)
      return
      end

c===========================================================================
      subroutine qgdeft(s,ep,ey)
c---------------------------------------------------------------------------
c determination of the parameters for the lorentz transform to the rest frame
c system for 4-vector ep
c---------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension ey(3),ep(4)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)ep,s

      do i=1,3
       if(ep(i+1).eq.0.d0)then
        ey(i)=1.d0
       else
        wp=ep(1)+ep(i+1)
        wm=ep(1)-ep(i+1)
        if(wm/wp.lt.1.d-8)then
         ww=s
         do l=1,3
          if(l.ne.i)ww=ww+ep(l+1)**2
         enddo
         wm=ww/wp
        endif
        ey(i)=dsqrt(wm/wp)
        ep(1)=wp*ey(i)
        ep(i+1)=0.d0
       endif
      enddo

      if(debug.ge.3)write (moniou,202)ey
201   format(2x,'qgdeft - lorentz boost parameters:'
     */4x,'4-vector ep=',4e10.3/4x,'4-vector squared s=',e10.3)
202   format(2x,'qgdeft: lorentz boost parameters ey(i)=',2x,3e10.3)
      return
      end

c=============================================================================
      subroutine qgdefr(ep,s0x,c0x,s0,c0)
c-----------------------------------------------------------------------------
c determination of the parameters the spacial rotation to the lab. system
c for 4-vector ep
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension ep(4)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)ep

c transverse momentum square for the current parton (ep)
      pt2=ep(3)**2+ep(4)**2
      if(pt2.ne.0.d0)then
       pt=dsqrt(pt2)
c system rotation to get pt=0 - euler angles are determined (c0x = cos theta,
c s0x = sin theta, c0 = cos phi, s0 = sin phi)
       c0x=ep(3)/pt
       s0x=ep(4)/pt
c total momentum for the gluon
       pl=dsqrt(pt2+ep(2)**2)
       s0=pt/pl
       c0=ep(2)/pl
      else
       c0x=1.d0
       s0x=0.d0
       pl=abs(ep(2))
       s0=0.d0
       c0=ep(2)/pl
      endif
      ep(2)=pl
      ep(3)=0.d0
      ep(4)=0.d0

      if(debug.ge.3)write (moniou,202)s0x,c0x,s0,c0,ep
201   format(2x,'qgdefr - spacial rotation parameters'/4x,
     *'4-vector ep=',2x,4(e10.3,1x))
202   format(2x,'qgdefr: spacial rotation parameters'/
     *4x,'s0x=',e10.3,2x,'c0x=',e10.3,2x,'s0=',e10.3,2x,'c0=',e10.3/
     *4x,'rotated 4-vector ep=',4(e10.3,1x))
      return
      end

c=============================================================================
      double precision function qgfap(x,j,l)
c------------------------------------------------------------------------
c qgfap - altarelli-parisi function (multiplied by x)
c x - light cone momentum share value,
c j - type of the parent parton (1-g,2-q)
c l - type of the daughter parton (1-g,2-q)
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)x,j,l

      if(j.eq.1)then
       if(l.eq.1)then
        qgfap=((1.d0-x)/x+x/(1.d0-x)+x*(1.d0-x))*6.d0
       else
        qgfap=(x**2+(1.d0-x)**2)*3.d0
       endif
      else
       if(l.eq.1)then
        qgfap=(1.d0+(1.d0-x)**2)/x/.75d0
       else
        qgfap=(x**2+1.d0)/(1.d0-x)/.75d0
       endif
      endif

      if(debug.ge.3)write (moniou,202)qgfap
201   format(2x,'qgfap - altarelli-parisi function:'
     *,2x,'x=',e10.3,2x,'j=',i1,2x,'l=',i1)
202   format(2x,'qgfap=',e10.3)
      return
      end

c=============================================================================
      subroutine qggea(ia,xa,jj)
c-----------------------------------------------------------------------------
c qggea - nuclear configuration simulation (nucleons positions)
c ia - number of nucleons to be considered
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      dimension xa(iapmax,3)
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr6/  pi,bm,amws
      common /qgarr11/ b10
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)jj,ia

      if(ia.ge.10)then
       do i=1,ia
1       zuk=qgran(b10)*cr1(jj)-1.d0
c        if(zuk)2,2,3
        if(zuk.le.0.d0)then
         tt=rnuc(jj)/wsnuc(jj)*(qgran(b10)**.3333d0-1.d0)
         goto 6
        else
         if(zuk.gt.cr2(jj))goto 4
         tt=-dlog(qgran(b10))
         goto 6
4        if(zuk.gt.cr3(jj))goto 5
         tt=-dlog(qgran(b10))-dlog(qgran(b10))
         goto 6
5        tt=-dlog(qgran(b10))-dlog(qgran(b10))-dlog(qgran(b10))
        endif
6       rim=tt*wsnuc(jj)+rnuc(jj)
        if(qgran(b10).gt.(1.d0+wbnuc(jj)*rim**2/rnuc(jj)**2)
     *  /(1.d0+exp(-abs(tt))))goto 1
        z=rim*(2.d0*qgran(b10)-1.d0)
        rim=dsqrt(rim*rim-z*z)
        xa(i,3)=z
        call qgcs(c,s)
        xa(i,1)=rim*c
        xa(i,2)=rim*s
       enddo
      else
       do l=1,3
        summ=0.d0
        do i=1,ia-1
         j=ia-i
         aks=rnuc(jj)*(qgran(b10)+qgran(b10)+qgran(b10)-1.5d0)
         k=j+1
         xa(k,l)=summ-aks*sqrt(float(j)/k)
         summ=summ+aks/sqrt(float(j*k))
        enddo
        xa(1,l)=summ
       enddo
      endif

      if(debug.ge.3)then
       write (moniou,203)
       do i=1,ia
        write (moniou,204)i,(xa(i,l),l=1,3)
       enddo
       write (moniou,202)
      endif
201   format(2x,'qggea - configuration of the nucleus ',i1,';',2x,
     *'coordinates for ',i2,' nucleons')
202   format(2x,'qggea - end')
203   format(2x,'qggea:  positions of the nucleons')
204   format(2x,'qggea: ',i2,' - ',3(e10.3,1x))
      return
      end

c=============================================================================
      double precision function qgapi(x,j,l)
c-----------------------------------------------------------------------------
c qgapi - integrated altarelli-parisi function
c x - light cone momentum share value,
c j - type of initial parton (1 - g, 2 - q)
c l - type of final parton (1 - g, 2 - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)x,j,l

      if(j.eq.1)then
       if(l.eq.1)then
        qgapi=6.d0*(dlog(x/(1.d0-x))-x**3/3.d0+x**2/2.d0-2.d0*x)
       else
        qgapi=3.d0*(x+x**3/1.5d0-x*x)
       endif
      else
       if(l.eq.1)then
        qgapi=(dlog(x)-x+.25d0*x*x)/.375d0
       else
        z=1.d0-x
        qgapi=-(dlog(z)-z+.25d0*z*z)/.375d0
       endif
      endif

      if(debug.ge.2)write (moniou,202)qgapi
201   format(2x,'qgapi: x=',e10.3,2x,'j= ',i1,2x,'l= ',i1)
202   format(2x,'qgapi=',e10.3)
      return
      end

c=============================================================================
      subroutine qgjarr(jfl)
c-----------------------------------------------------------------------------
c final jets rearrangement according to their colour connections
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(njmax=50000)
      dimension mark(njmax),ept(4)
      common /qgarr10/ am(7),ammu
      common /qgarr36/ epjet(4,njmax),ipjet(njmax),njtot
      common /qgarr37/ eqj(4,njmax),iqj(njmax),ncj(2,njmax),nj
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)nj
      if(debug.ge.2.and.nj.ne.0)then
       do i=1,nj
        write (moniou,203)i,iqj(i),(eqj(l,i),l=1,4)
        if(iqj(i).eq.0)then
         write (moniou,204)ncj(1,i),ncj(2,i)
        else
         ncdum=0
         write (moniou,204)ncj(1,i),ncdum
        endif
       enddo
      endif

      njpar=0
      jfl=0
      do i=1,nj
       mark(i)=1
      enddo
      njtot=0

2     continue
      do ij=1,nj
       if(mark(ij).ne.0.and.iqj(ij).ne.0)goto 4
      enddo
4     continue

      jfirst=1
      if(iabs(iqj(ij)).le.2)then
       am1=am(1)
      elseif(iabs(iqj(ij)).eq.4)then
       am1=am(3)
      else
       am1=am(2)
      endif
      do i=1,4
       ept(i)=0.d0
      enddo

6     mark(ij)=0
      njtot=njtot+1
      ipjet(njtot)=iqj(ij)
      do i=1,4
       ept(i)=ept(i)+eqj(i,ij)
       epjet(i,njtot)=eqj(i,ij)
      enddo

      if(iqj(ij).ne.0)then
       if(jfirst.ne.1)then
        if(iabs(iqj(ij)).le.2)then
         am2=am(1)
        elseif(iabs(iqj(ij)).eq.4)then
         am2=am(3)
        else
         am2=am(2)
        endif
        amj=(am1+am2)**2
        if(amj.gt.qgnrm(ept))then
         if(debug.ge.3)write (moniou,202)jfl
         return
        endif

        if(njtot.lt.nj)then
         goto 2
        else
         jfl=1
         nj=0
         if(debug.ge.3)write (moniou,202)jfl
         return
        endif
       else
        jfirst=0
        njpar=ij
        ij=ncj(1,ij)
        goto 6
       endif
      else
       if(ncj(1,ij).eq.njpar)then
        njdau=ncj(2,ij)
       else
        njdau=ncj(1,ij)
       endif
       njpar=ij
       ij=njdau
       goto 6
      endif

201   format(2x,'qgjarr: total number of jets nj=',i4)
202   format(2x,'qgjarr - end,jfl=',i2)
203   format(2x,'qgjarr: ij=',i3,2x,'iqj=',i2,2x,'eqj=',4e10.3)
204   format(2x,'qgjarr: ncj=',2i3)
      end

c=============================================================================
      double precision function qgjet(q1,q2,s,s2min,j,l)
c-----------------------------------------------------------------------------
c qgjet - inclusive hard cross-section calculation (one more run is added
c to the ladder) - for any ordering
c q1 - effective momentum cutoff for current end of the ladder,
c q2 - effective momentum cutoff for opposide end of the ladder,
c s - total c.m. energy squared for the ladder,
c s2min - minimal c.m. energy squared for born process (above q1 and q2)
c j - parton type at current end of the ladder (1 - g, 2 - q)
c l - parton type at opposite end of the ladder (1 - g, 2 - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)s,q1,q2,s2min,j,l

      qgjet=0.d0
      qmax=s/4.d0/fqscal*(1.d0-epsxmn)
      qmin=q1
      if(debug.ge.3)write (moniou,203)qmin,qmax

      if(qmax.gt.qmin)then
c numerical integration over transverse momentum square;
c gaussian integration is used
       do i=1,7
       do m=1,2
        qi=2.d0*qmin/(1.d0+qmin/qmax+(2*m-3)*x1(i)*(1.d0-qmin/qmax))
        zmax=(1.d0-epsxmn)**delh
        zmin=(max(4.d0*fqscal*qi,s2min)/s)**delh
        fsj=0.d0
        if(debug.ge.3)write (moniou,204)qi,zmin,zmax

        if(zmax.gt.zmin)then
         do i1=1,7
         do m1=1,2
          z=(.5d0*(zmax+zmin+(2*m1-3)*x1(i1)*(zmax-zmin)))**(1.d0/delh)
          s2=z*s

          sj=0.d0
          do k=1,2
           sj=sj+qgjit(qi,q2,s2,k,l)*qgfap(z,j,k)*z
          enddo
          fsj=fsj+a1(i1)*sj/z**delh
         enddo
         enddo
         fsj=fsj*(zmax-zmin)
        endif
        qgjet=qgjet+a1(i)*fsj*qi*qgsudx(qi,j)*qgalf(qi/alm)
       enddo
       enddo
       qgjet=qgjet*(1.d0/qmin-1.d0/qmax)/qgsudx(q1,j)/delh/4.d0
      endif

      if(debug.ge.3)write (moniou,202)qgjet
201   format(2x,'qgjet - unordered ladder cross section:'
     */4x,'s=',e10.3,2x,'q1=',e10.3,2x,'q2=',e10.3,2x,'s2min=',
     *e10.3,2x,'j=',i1,2x,'l=',i1)
202   format(2x,'qgjet=',e10.3)
203   format(2x,'qgjet:',2x,'qmin=',e10.3,2x,'qmax=',e10.3)
204   format(2x,'qgjet:',2x,'qi=',e10.3,2x,'zmin=',e10.3
     *,2x,'zmax=',e10.3)
      return
      end

c=============================================================================
      double precision function qgjet1(q1,q2,s,s2min,j,l)
c-----------------------------------------------------------------------------
c qgjet1 - inclusive hard cross-section calculation (one more run is added
c to the ladder) - for strict ordering
c q1 - effective momentum cutoff for current end of the ladder,
c q2 - effective momentum cutoff for opposide end of the ladder,
c s - total c.m. energy squared for the ladder,
c s2min - minimal c.m. energy squared for born process (above q1 and q2)
c j - parton type at current end of the ladder (1 - g, 2 - q)
c l - parton type at opposite end of the ladder (1 - g, 2 - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/  debug
      common /arr3/   x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)s,q1,q2,s2min,j,l

      qgjet1=0.d0
      qmax=s/4.d0/fqscal*(1.d0-epsxmn)
      qmin=q1
      if(debug.ge.3)write (moniou,203)qmin,qmax

      if(qmax.gt.qmin)then
c numerical integration over transverse momentum square;
c gaussian integration is used
       do i=1,7
       do m=1,2
        qi=2.d0*qmin/(1.d0+qmin/qmax+(2*m-3)*x1(i)*(1.d0-qmin/qmax))
        zmax=(1.d0-epsxmn)**delh
        zmin=(max(4.d0*fqscal*qi,s2min)/s)**delh
        fsj=0.d0
        if(debug.ge.3)write (moniou,204)qi,zmin,zmax

        if(zmax.gt.zmin)then
         do i1=1,7
         do m1=1,2
          z=(.5d0*(zmax+zmin+(2*m1-3)*x1(i1)*(zmax-zmin)))**(1.d0/delh)
          s2=z*s

          sj=0.d0
          do k=1,2
           sj=sj+qgjit1(qi,q2,s2,k,l)*qgfap(z,j,k)*z
          enddo
          fsj=fsj+a1(i1)*sj/z**delh
         enddo
         enddo
         fsj=fsj*(zmax-zmin)
        endif
        qgjet1=qgjet1+a1(i)*fsj*qi*qgsudx(qi,j)*qgalf(qi/alm)
       enddo
       enddo
       qgjet1=qgjet1*(1.d0/qmin-1.d0/qmax)/qgsudx(q1,j)/delh/4.d0
      endif

      if(debug.ge.3)write (moniou,202)qgjet1
201   format(2x,'qgjet1 - strictly ordered ladder cross section:'
     */4x,'s=',e10.3,2x,'q1=',e10.3,2x,'q2=',e10.3,2x,'s2min=',
     *e10.3,2x,'j=',i1,2x,'l=',i1)
202   format(2x,'qgjet1=',e10.3)
203   format(2x,'qgjet1:',2x,'qmin=',e10.3,2x,'qmax=',e10.3)
204   format(2x,'qgjet1:',2x,'qi=',e10.3,2x,'zmin=',e10.3
     *,2x,'zmax=',e10.3)
      return
      end

c=============================================================================
      double precision function qgjit(q1,q2,s,m,l)
c-----------------------------------------------------------------------------
c qgjit - inclusive hard cross-section interpolation - for any ordering
c in the ladder
c q1 - effective momentum cutoff for current end of the ladder,
c q2 - effective momentum cutoff for opposide end of the ladder,
c s - total c.m. energy squared for the ladder,
c m - parton type at current end of the ladder (1 - g, 2 - q)
c l - parton type at opposite end of the ladder (1 - g, 2 - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wi(3),wj(3),wk(3)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr20/ spmax
      common /qgarr26/ factk,fqscal
      common /qgarr29/ csj(40,40,160)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)s,q1,q2,m,l

      qgjit=0.d0
      qq=max(q1,q2)
      s2min=qq*4.d0*fqscal
      if(s.le..99d0*s2min)then
       if(debug.ge.3)write (moniou,202)qgjit
       return
      endif

      if(q1.le.q2)then
       qi=q1
       qj=q2
       ml=40*(m-1)+80*(l-1)
      else
       qi=q2
       qj=q1
       ml=40*(l-1)+80*(m-1)
      endif

      tmin=qq*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qq*fqscal/s)))
      qli=dlog(qi)/dlog(spmax/4.d0/fqscal)*39.d0+1.d0
      if(qi.lt..99d0*spmax/4.d0/fqscal)then
       qlj=dlog(qj/qi)/dlog(spmax/4.d0/fqscal/qi)*39.d0+1.d0
      else
       qlj=1.d0
      endif
      sl=dlog(s/s2min)/dlog(spmax/s2min)*39.d0+1.d0
      i=min(38,int(qli))
      j=min(38,int(qlj))
      k=min(38,int(sl))

      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)
      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
      wi(1)=1.d0-wi(2)+wi(3)
      wi(2)=wi(2)-2.d0*wi(3)
      wj(2)=qlj-j
      wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
      wj(1)=1.d0-wj(2)+wj(3)
      wj(2)=wj(2)-2.d0*wj(3)
      do k1=1,3
       k2=k+k1-1+ml
      do i1=1,3
      do j1=1,3
       qgjit=qgjit+csj(i+i1-1,j+j1-1,k2)*wi(i1)*wj(j1)*wk(k1)
      enddo
      enddo
      enddo
      qgjit=exp(qgjit)*(1.d0/tmin-2.d0/s)

      if(debug.ge.3)write (moniou,202)qgjit
201   format(2x,'qgjit - unordered ladder cross section interpol.:'/4x,
     *'s=',e10.3,2x,'q1=',e10.3,2x,'q2=',e10.3,2x,2x,'m=',i1,2x,'l=',i1)
202   format(2x,'qgjit=',e10.3)
      return
      end

c=============================================================================
      double precision function qgjit1(q1,q2,s,m,l)
c-----------------------------------------------------------------------------
c qgjit1 - inclusive hard cross-section interpolation - for strict ordering
c in the ladder
c q1 - effective momentum cutoff for current end of the ladder,
c q2 - effective momentum cutoff for opposide end of the ladder,
c s - total c.m. energy squared for the ladder,
c m - parton type at current end of the ladder (1 - g, 2 - q)
c l - parton type at opposite end of the ladder (1 - g, 2 - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wi(3),wj(3),wk(3)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr20/ spmax
      common /qgarr26/ factk,fqscal
      common /qgarr30/ csj(40,40,160)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)s,q1,q2,m,l

      qgjit1=0.d0
      qq=max(q1,q2)
      s2min=qq*4.d0*fqscal
      if(s.le.s2min)then
       if(debug.ge.3)write (moniou,202)qgjit1
       return
      endif

      tmin=qq*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qq*fqscal/s)))
      ml=40*(m-1)+80*(l-1)
      qli=dlog(q1)/dlog(spmax/4.d0/fqscal)*39.d0+1.d0
      if(q1.lt..99d0*spmax/4.d0/fqscal)then
       qlj=dlog(qq/q1)/dlog(spmax/4.d0/fqscal/q1)*39.d0+1.d0
      else
       qlj=1.d0
      endif
      sl=dlog(s/s2min)/dlog(spmax/s2min)*39.d0+1.d0
      i=min(38,int(qli))
      j=min(38,int(qlj))
      k=min(38,int(sl))
      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)
      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
      wi(1)=1.d0-wi(2)+wi(3)
      wi(2)=wi(2)-2.d0*wi(3)
      wj(2)=qlj-j
      wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
      wj(1)=1.d0-wj(2)+wj(3)
      wj(2)=wj(2)-2.d0*wj(3)

      do k1=1,3
       k2=k+k1-1+ml
      do i1=1,3
      do j1=1,3
       qgjit1=qgjit1+csj(i+i1-1,j+j1-1,k2)*wi(i1)*wj(j1)*wk(k1)
      enddo
      enddo
      enddo
      qgjit1=exp(qgjit1)*(1.d0/tmin-2.d0/s)
      if(q2.lt.q1)qgjit1=qgjit1*qgsudx(q1,l)/qgsudx(q2,l)

      if(debug.ge.3)write (moniou,202)qgjit1
201   format(2x,'qgjit1 - ordered ladder cross section interpol.:'/4x,
     *'s=',e10.3,2x,'q1=',e10.3,2x,'q2=',e10.3,2x,2x,'m=',i1,2x,'l=',i1)
202   format(2x,'qgjit1=',e10.3)
      return
      end

c=============================================================================
      double precision function qglam(s,a,b)
c-----------------------------------------------------------------------------
c kinematical function for two particle decay - maximal pt-value
c a - first particle mass squared,
c b - second particle mass squared,
c s - two particle invariant mass
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)s,a,b

      qglam=max(0.d0,.25d0/s*(s+a-b)**2-a)

      if(debug.ge.3)write (moniou,202)qglam
201   format(2x,'qglam - kinematical function s=',e10.3,2x,'a='
     *,e10.3,2x,'b=',e10.3)
202   format(2x,'qglam=',e10.3)
      return
      end

c=============================================================================
      double precision function qgnrm(ep)
c-----------------------------------------------------------------------------
c 4-vector squared calculation
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension ep(4)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)ep
      qgnrm=(ep(1)-ep(2))*(ep(1)+ep(2))-ep(3)**2-ep(4)**2

      if(debug.ge.3)write (moniou,202)qgnrm
201   format(2x,'qgnrm - 4-vector squared for ','ep=',4(e10.3,1x))
202   format(2x,'qgnrm=',e10.3)
      return
      end

c===========================================================================
      subroutine qgrec(ep,nqc,qv,zv,qm,iqv,ldau,lpar,jq)
c---------------------------------------------------------------------------
c jet reconstructuring procedure - 4-momenta for all final jets are determ.
c ep(i) - jet 4-momentum
c---------------------------------------------------------------------------
c qv(i,j) - effective momentum for the branching of the parton in i-th row
c on j-th level (0 - in case of no branching)
c zv(i,j) - z-value for the branching of the parton in i-th row
c on j-th level
c qm(i,j) - mass squared for the parton in i-th row
c on j-th level
c iqv(i,j) - flavours for the parton in i-th row on j-th level
c ldau(i,j) - first daughter row for the branching of the parton in i-th row
c on j-th level
c lpar(i,j) - the parent row for the parton in i-th row on j-th level
c----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(njmax=50000)
      dimension ep(4),ep3(4),epv(4,30,50),nqc(2),ncc(2,30,50),
     *qv(30,50),zv(30,50),qm(30,50),iqv(30,50),
     *ldau(30,49),lpar(30,50)
c eqj(i,nj) - 4-momentum for the final jet nj
c iqj(nj) - flavour for the final jet nj
c ncj(m,nj) - colour connections for the final jet nj
      common /qgarr37/ eqj(4,njmax),iqj(njmax),ncj(2,njmax),nj
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)jq,ep,iqv(1,1),nqc

      do i=1,4
       epv(i,1,1)=ep(i)
      enddo
      ncc(1,1,1)=nqc(1)
      if(iqv(1,1).eq.0)ncc(2,1,1)=nqc(2)
      nlev=1
      nrow=1

2     continue
      if(qv(nrow,nlev).eq.0.d0)then
       nj=nj+1
       do i=1,4
        eqj(i,nj)=epv(i,nrow,nlev)
       enddo
       iqj(nj)=iqv(nrow,nlev)
       if(iabs(iqj(nj)).eq.3)iqj(nj)=iqj(nj)*4/3

       if(iqj(nj).ne.0)then
        njc=ncc(1,nrow,nlev)
        if(njc.ne.0)then
         ncj(1,nj)=njc
         iqc=iqj(njc)
         if(iqc.ne.0)then
          ncj(1,njc)=nj
         else
          if(iqj(nj).gt.0)then
           ncj(2,njc)=nj
          else
           ncj(1,njc)=nj
          endif
         endif
        else
         ncc(1,nrow,nlev)=nj
        endif
       else

        do m=1,2
         if(jq.eq.1)then
          m1=m
         else
          m1=3-m
         endif
         njc=ncc(m1,nrow,nlev)
         if(njc.ne.0)then
          ncj(m,nj)=njc
          iqc=iqj(njc)
          if(iqc.ne.0)then
           ncj(1,njc)=nj
          else
           ncj(3-m,njc)=nj
          endif
         else
          ncc(m1,nrow,nlev)=nj
         endif
        enddo
       endif
       if(debug.ge.3)write (moniou,204)
     * nj,nlev,nrow,iqj(nj),(eqj(i,nj),i=1,4)

      else
       do i=1,4
         ep3(i)=epv(i,nrow,nlev)
       enddo
       call qgdefr(ep3,s0x,c0x,s0,c0)
       z=zv(nrow,nlev)
       qt2=(z*(1.d0-z))**2*qv(nrow,nlev)
       ldrow=ldau(nrow,nlev)

       wp0=ep3(1)+ep3(2)
       wpi=z*wp0
       wmi=(qt2+qm(ldrow,nlev+1))/wpi
       ep3(1)=.5d0*(wpi+wmi)
       ep3(2)=.5d0*(wpi-wmi)
       qt=dsqrt(qt2)
       call qgcs(c,s)
       ep3(3)=qt*c
       ep3(4)=qt*s
       call qgrota(ep3,s0x,c0x,s0,c0)
       do i=1,4
        epv(i,ldrow,nlev+1)=ep3(i)
       enddo
       if(debug.ge.3)write (moniou,206)nlev+1,ldrow,ep3

       wpi=(1.d0-z)*wp0
       wmi=(qt2+qm(ldrow+1,nlev+1))/wpi
       ep3(1)=.5d0*(wpi+wmi)
       ep3(2)=.5d0*(wpi-wmi)
       ep3(3)=-qt*c
       ep3(4)=-qt*s
       call qgrota(ep3,s0x,c0x,s0,c0)
       do i=1,4
        epv(i,ldrow+1,nlev+1)=ep3(i)
       enddo
       if(debug.ge.3)write (moniou,206)nlev+1,ldrow+1,ep3

       if(iqv(nrow,nlev).eq.0)then
        if(iqv(ldrow,nlev+1).ne.0)then
         ncc(1,ldrow,nlev+1)=ncc(1,nrow,nlev)
         ncc(1,ldrow+1,nlev+1)=ncc(2,nrow,nlev)
        else
         ncc(1,ldrow,nlev+1)=ncc(1,nrow,nlev)
         ncc(2,ldrow,nlev+1)=0
         ncc(1,ldrow+1,nlev+1)=0
         ncc(2,ldrow+1,nlev+1)=ncc(2,nrow,nlev)
        endif
       else
        if(iqv(ldrow,nlev+1).eq.0)then
         ncc(1,ldrow,nlev+1)=ncc(1,nrow,nlev)
         ncc(2,ldrow,nlev+1)=0
         ncc(1,ldrow+1,nlev+1)=0
        else
         ncc(1,ldrow,nlev+1)=0
         ncc(1,ldrow+1,nlev+1)=0
         ncc(2,ldrow+1,nlev+1)=ncc(1,nrow,nlev)
        endif
       endif

       nrow=ldrow
       nlev=nlev+1
       goto 2
      endif

8     continue
      if(nlev.eq.1)then
       if(nqc(1).eq.0)nqc(1)=ncc(1,1,1)
       if(iqv(1,1).eq.0.and.nqc(2).eq.0)nqc(2)=ncc(2,1,1)
       if(debug.ge.3)write (moniou,202)
       return
      endif

      lprow=lpar(nrow,nlev)
      if(ldau(lprow,nlev-1).eq.nrow)then
       if(iqv(nrow,nlev).eq.0)then
        if(ncc(1,lprow,nlev-1).eq.0)ncc(1,lprow,nlev-1)=ncc(1,nrow,nlev)
        ncc(1,nrow+1,nlev)=ncc(2,nrow,nlev)
       else
        if(iqv(lprow,nlev-1).eq.0)then
         if(ncc(1,lprow,nlev-1).eq.0)
     *   ncc(1,lprow,nlev-1)=ncc(1,nrow,nlev)
        else
         ncc(1,nrow+1,nlev)=ncc(1,nrow,nlev)
        endif
       endif
       nrow=nrow+1
       goto 2
      else
       if(iqv(nrow,nlev).eq.0)then
        if(iqv(lprow,nlev-1).eq.0)then
         if(ncc(2,lprow,nlev-1).eq.0)
     *   ncc(2,lprow,nlev-1)=ncc(2,nrow,nlev)
        else
         if(ncc(1,lprow,nlev-1).eq.0)
     *   ncc(1,lprow,nlev-1)=ncc(2,nrow,nlev)
        endif
       else
        if(iqv(lprow,nlev-1).eq.0.and.ncc(2,lprow,nlev-1).eq.0)
     *  ncc(2,lprow,nlev-1)=ncc(1,nrow,nlev)
       endif
       nrow=lprow
       nlev=nlev-1
       goto 8
      endif

201   format(2x,'qgrec - jet reconstructuring: jq=',i1
     */4x,'jet 4-momentum ep=',4(e10.3,1x)
     */4x,'jet flavor: ',i2,2x,'colour connections: ',2i3)
202   format(2x,'qgrec - end')
204   format(2x,'qgrec: ',i3,'-th final jet at level nlev=',i2,' nrow='
     *,i2/4x,'jet flavor: ',i3,2x,'jet 4-momentum:',4(e10.3,1x))
206   format(2x,'qgrec: jet at level nlev='
     *,i2,' nrow=',i2/4x,'jet 4-momentum:',4(e10.3,1x))
      end

c=============================================================================
      double precision function qgroot(qlmax,g,j)
c-----------------------------------------------------------------------------
c qgroot - effective momentum tabulation for given set of random number
c values and maximal effective momentum qmax values - according to the
c probability of branching: (1 - timelike sudakov formfactor)
c qlmax - ln qmax/16/qtf,
c g - dzeta number (some function of ksi)
c j - type of the parton (1-g,2-q)
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)qlmax,g,j

      ql0=0.d0
      ql1=qlmax
      f0=-g
      f1=1.d0-g
      sud0=-dlog(qgsudi(qlmax,j))

1     ql2=ql1-(ql1-ql0)*f1/(f1-f0)
      if(ql2.lt.0.d0)then
       ql2=0.d0
       f2=-g
      elseif(ql2.gt.qlmax)then
       ql2=qlmax
       f2=1.d0-g
      else
       f2=-dlog(qgsudi(ql2,j))/sud0-g
      endif
      if(abs(f2).gt.1.d-3)then
       if(f1*f2.lt.0.d0)then
        ql0=ql1
        f0=f1
       endif
       ql1=ql2
       f1=f2
       goto 1
      else
       qgroot=ql2
      endif

      if(debug.ge.3)write (moniou,202)qgroot
201   format(2x,'qgqint - branching momentum tabulation:'
     */4x,'qlmax=',e10.3,2x,'g=',e10.3,2x,'j=',i1)
202   format(2x,'qgroot=',e10.3)
      return
      end

c=============================================================================
      subroutine qgrota(ep,s0x,c0x,s0,c0)
c-----------------------------------------------------------------------------
c spacial rotation to the lab. system for 4-vector ep
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension ep(4),ep1(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)ep,s0x,c0x,s0,c0

      ep1(3)=ep(4)
      ep1(2)=ep(2)*s0+ep(3)*c0
      ep1(1)=ep(2)*c0-ep(3)*s0
      ep(2)=ep1(1)
      ep(4)=ep1(2)*s0x+ep1(3)*c0x
      ep(3)=ep1(2)*c0x-ep1(3)*s0x

      if(debug.ge.3)write (moniou,202)ep
201   format(2x,'qgrota - spacial rotation:'/4x,'4-vector ep=',4(e10.3
     *,1x)/4x,'s0x=',e10.3,'c0x=',e10.3,2x,'s0=',e10.3,'c0=',e10.3)
202   format(2x,'qgrota: rotated 4-vector ep=',2x,4e10.3)
      return
      end

c=============================================================================
      double precision function qgqint(qlmax,g,j)
c-----------------------------------------------------------------------------
c qgqint - effective momentum interpolation for given random number g
c and maximal effective momentum qmax
c qlmax - ln qmax/16/qtf,
c g - random number (0<g<1)
c j - type of the parton (1-g,2-q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wi(3),wk(3)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr34/ qrt(10,101,2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)qlmax,g,j

      qli=qlmax/1.38629d0
      sud0=1.d0/qgsudi(qlmax,j)
      sl=100.d0*dlog(1.d0-g*(1.d0-sud0))/dlog(sud0)
      i=int(qli)
      k=int(sl)
      if(k.gt.98)k=98
      wk(2)=sl-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)
      qgqint=0.d0
      if(i.gt.7)i=7
      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
      wi(1)=1.d0-wi(2)+wi(3)
      wi(2)=wi(2)-2.d0*wi(3)
      do k1=1,3
      do i1=1,3
       qgqint=qgqint+qrt(i+i1,k+k1,j)*wi(i1)*wk(k1)
      enddo
      enddo
      if(qgqint.le.0.d0)qgqint=0.d0
      qgqint=16.d0*qtf*exp(qgqint)

      if(debug.ge.3)write (moniou,202)qgqint
201   format(2x,'qgqint - branching momentum interpolation:'
     */4x,'qlmax=',e10.3,2x,'g=',e10.3,2x,'j=',i1)
202   format(2x,'qgqint=',e10.3)
      return
      end

c=============================================================================
      double precision function qgalf(qq)
c-----------------------------------------------------------------------------
c qgalf - alpha_s(qq)/2/pi
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr43/ moniou
      common /qgdebug/  debug

      qgalf=2.d0/9.d0/dlog(qq)
      return
      end

c=============================================================================
      subroutine qgtran(ep,ey,jj)
c-----------------------------------------------------------------------------
c lorentz transform according to parameters ey ( determining lorentz shift
c along the z,x,y-axis respectively (ey(1),ey(2),ey(3)))
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension ey(3),ep(4)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)ep,ey

      if(jj.eq.1)then
c lorentz transform to lab. system according to 1/ey(i) parameters
       do i=1,3
        if(ey(4-i).ne.1.d0)then
         wp=(ep(1)+ep(5-i))/ey(4-i)
         wm=(ep(1)-ep(5-i))*ey(4-i)
         ep(1)=.5d0*(wp+wm)
         ep(5-i)=.5d0*(wp-wm)
        endif
       enddo
      else
c lorentz transform to lab. system according to ey(i) parameters
       do i=1,3
        if(ey(i).ne.1.d0)then
         wp=(ep(1)+ep(i+1))*ey(i)
         wm=(ep(1)-ep(i+1))/ey(i)
         ep(1)=.5d0*(wp+wm)
         ep(i+1)=.5d0*(wp-wm)
        endif
       enddo
      endif

      if(debug.ge.3)write (moniou,202)ep
201   format(2x,'qgtran - lorentz boost for 4-vector'/4x,'ep='
     *,2x,4(e10.3,1x)/4x,'boost parameters ey=',3e10.3)
202   format(2x,'qgtran: transformed 4-vector ep=',2x,4(e10.3,1x))
      return
      end

c=============================================================================
      double precision function qgsudi(qlmax,j)
c-----------------------------------------------------------------------------
c qgsudi - timelike sudakov formfactor interpolation
c qlmax - ln qmax/16/qtf,
c j - type of the parton (1-g,2-q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3)
      common /qgarr33/ fsud(10,2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)j,qlmax

      ql=qlmax/1.38629d0
      if(ql.le.0.d0)then
       qgsudi=1.d0
      else
       k=int(ql)
       if(k.gt.7)k=7
       wk(2)=ql-k
       wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
       wk(1)=1.d0-wk(2)+wk(3)
       wk(2)=wk(2)-2.d0*wk(3)

       qgsudi=0.d0
       do k1=1,3
        qgsudi=qgsudi+fsud(k+k1,j)*wk(k1)
       enddo
       if(qgsudi.le.0.d0)qgsudi=0.d0
       qgsudi=exp(-qgsudi)
      endif

      if(debug.ge.3)write (moniou,202)qgsudi
201   format(2x,'qgsudi - spacelike form factor interpolation:'
     */4x,'parton type j=',i1,2x,'momentum logarithm qlmax=',e10.3)
202   format(2x,'qgsudi=',e10.3)
      return
      end

c=============================================================================
      double precision function qgsudx(q,j)
c-----------------------------------------------------------------------------
c qgsudx - spacelike sudakov formfactor
c q - maximal value of the effective momentum,
c j - type of parton (1 - g, 2 - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)j,q

      if(q.gt.1.d0)then
       qgsudx=dlog(dlog(q/alm)/dlog(1.d0/alm))*(.75d0+dlog(epsxmn))
       if(j.eq.1)then
        qgsudx=exp(qgsudx/.75d0)
       else
        qgsudx=exp(qgsudx*16.d0/27.d0)
       endif
      else
       qgsudx=1.d0
      endif

      if(debug.ge.3)write (moniou,202)qgsudx
201   format(2x,'qgsudx - spacelike form factor: parton type j='
     *,i1,2x,'momentum q=',e10.3)
202   format(2x,'qgsudx=',e10.3)
      return
      end

c=============================================================================
      double precision function qgsudt(qmax,j)
c-----------------------------------------------------------------------------
c qgsudt - timelike sudakov formfactor
c qmax - maximal value of the effective momentum,
c j - type of parton (1 - g, 2 - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common/arr3/x1(7),a1(7)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)j,qmax

      qgsudt=0.d0
      qlmax=dlog(dlog(qmax/16.d0/alm))
      qfl=dlog(dlog(qtf/alm))
c numerical integration over transverse momentum square;
c gaussian integration is used
      do i=1,7
      do m=1,2
       qtl=.5d0*(qlmax+qfl+(2*m-3)*x1(i)*(qlmax-qfl))
       qt=alm*exp(exp(qtl))
       if(qt.ge.qmax/16.d0)qt=qmax/16.0001d0
       zmin=.5d0-dsqrt((.25d0-dsqrt(qt/qmax)))
       zmax=1.d0-zmin

       if(j.eq.1)then
        ap=(qgapi(zmax,1,1)-qgapi(zmin,1,1)+
     *  qgapi(zmax,1,2)-qgapi(zmin,1,2))*.5d0
       else
        ap=qgapi(zmax,2,1)-qgapi(zmin,2,1)
       endif
       qgsudt=qgsudt+a1(i)*ap
      enddo
      enddo
      qgsudt=qgsudt*(qlmax-qfl)/9.d0

      if(debug.ge.3)write (moniou,202)qgsudt
201   format(2x,'qgsudt - timelike form factor: parton type j='
     *,i1,2x,'momentum qmax=',e10.3)
202   format(2x,'qgsudt=',e10.3)
      return
      end

c=============================================================================
      double precision function qgtwd(s,a,b)
c-----------------------------------------------------------------------------
c kinematical function for two particle decay - light cone momentum share
c for the particle of mass squared a,
c b - partner's mass squared,
c s - two particle invariant mass
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)s,a,b

      x=.5d0*(1.d0+(a-b)/s)
      dx=x-dsqrt(a/s)
      if(dx.gt.0.d0)then
       x=x+dsqrt(dx)*dsqrt(x+dsqrt(a/s))
      else
       x=dsqrt(a/s)
      endif
      qgtwd=x

      if(debug.ge.3)write (moniou,202)qgtwd
201   format(2x,'qgtwd: s=',e10.3,2x,'a=',e10.3,2x,'b=',e10.3)
202   format(2x,'qgtwd=',e10.3)
      return
      end

c=============================================================================
      subroutine qgvdef(ich,ic1,ic2,icz)
c-----------------------------------------------------------------------------
c determination of valence quark flavour -
c for valence quark hard scattering
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr11/ b10
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)ich,icz

      is=iabs(ich)/ich
      if(icz.eq.1)then
       ic1=ich*(1-3*int(.5d0+qgran(b10)))
       ic2=-ic1-ich
      elseif(icz.eq.2)then
       if(qgran(b10).gt..33333d0.or.ich.lt.0)then
        ic1=ich-is
        ic2=3*is
       else
        ic1=4*is-ich
        ic2=ich+4*is
       endif
      elseif(icz.eq.3)then
       ic1=ich-3*is
       ic2=-4*is
      elseif(icz.eq.4)then
       ic1=ich-9*is
       ic2=5*is
      endif

      if(debug.ge.3)write (moniou,202)ic1,ic2
201   format(2x,'qgvdef: hadron type ich=',i2,' auxilliary type icz='
     *,i1)
202   format(2x,'qgvdef-end: parton flavors ic1=',i2,
     *'ic2=',i2)
      return
      end

c=============================================================================
      double precision function qgzsim(qq,j)
c-----------------------------------------------------------------------------
c qgzsim - light cone momentum share simulation (for the timelike
c branching)
c qq - effective momentum value,
c j - type of the parent parton (1-g,2-q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr11/ b10
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)qq,j

      zmin=.5d0-dsqrt(.25d0-dsqrt(qtf/qq))
      qlf=dlog(qtf/alm)
1     continue
      if(j.eq.1)then
       qgzsim=.5d0*(2.d0*zmin)**qgran(b10)
       gb=qgzsim*(qgfap(qgzsim,1,1)+qgfap(qgzsim,1,2))/7.5d0
      else
       qgzsim=zmin*((1.d0-zmin)/zmin)**qgran(b10)
       gb=qgzsim*qgfap(qgzsim,2,1)*.375d0
      endif
      qt=qq*(qgzsim*(1.d0-qgzsim))**2
      gb=gb/dlog(qt/alm)*qlf
      if(debug.ge.3)write (moniou,203)qt,gb
      if(qgran(b10).gt.gb)goto 1

      if(debug.ge.3)write (moniou,202)qgzsim
201   format(2x,'qgzsim - z-share simulation: qq=',e10.3,2x,'j=',i1)
202   format(2x,'qgzsim=',e10.3)
203   format(2x,'qgzsim: qt=',e10.3,2x,'gb=',e10.3)
      return
      end

c===========================================================================
      subroutine qgixxd(ich,ic1,ic2,icz)
c---------------------------------------------------------------------------
c determination of parton flavours for valence quark soft interaction
c (charge exchange)
c---------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr8/  wwm,be(4),dc(5),deta,almpt,ptdif,ptndi
      common /qgarr11/ b10
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)ich,icz

      is=iabs(ich)/ich
      if(icz.eq.1)then                      !pion
       ic1=ich*(1-3*int(.5d0+qgran(b10)))
       if(qgran(b10).lt.dc(2))then
        ic2=-4*ic1/iabs(ic1)
        if(iabs(ic1).eq.1)then
         ich1=-5*is
        else
         ich1=4*is
        endif
       else
        ich1=ich*int(.5d0+qgran(b10))
        ic2=-ic1*iabs(ich1)-(ich+ic1)*iabs(ich-ich1)
       endif
      elseif(icz.eq.2)then
c valence quark type simulation ( for proton )
       ic1=int(1.3333d0+qgran(b10))
c leading nucleon type simulation ( flavors combinatorics )
       if(ic1.eq.1)then
        ich1=int(qgran(b10)+.5d0)+2
        ic2=1-ich1
       elseif(qgran(b10).lt..5d0)then
        ich1=2
        ic2=-2
       else
        ich1=7                   !uuu
        ic2=-1
       endif
       if(iabs(ich).eq.3)then    !neutron
        ic1=3-ic1
        ic2=-3-ic2
        if(ich1.eq.7)then
         ich1=8                  !ddd
        else
         ich1=5-ich1
        endif
       endif
       if(ich.lt.0)then
        ic1=-ic1
        ic2=-ic2
        ich1=-ich1
       endif
      elseif(icz.eq.3)then
       ic1=ich-3*is
       ic2=-is*int(1.5d0+qgran(b10))
       ich1=3*is-ic2
      elseif(icz.eq.4)then
       ic1=ich-9*is
       ic2=is*int(1.5d0+qgran(b10))
       ich1=9*is-ic2
      elseif(icz.eq.5)then
       ic1=is*int(1.5d0+qgran(b10))
       ic2=-ic1
       ich1=ich
      else
       ich1=0
       stop 'Should not happen in qgixxd !'
      endif
      ich=ich1

      if(debug.ge.3)write (moniou,202)ic1,ic2,ich
201   format(2x,'qgixxd: hadron type ich=',i2,' auxilliary type icz='
     *,i1)
202   format(2x,'qgixxd-end: parton flavors ic1=',i2,' ic2='
     *,i2,'new hadron type ich=',i2)
      return
      end

c=============================================================================
      subroutine qgdifr(wppr,wmtg,izp,izt,jexpr,jextg,iret)
c-----------------------------------------------------------------------------
c qgdifr - treatment of diffraction dissociation / leading hadron states
c wppr - LC momentum for projectile remnant;
c wptg - LC momentum for target remnant;
c izp  - projectile remnant type;
c izt  - target remnant type;
c jexpr/jextg = -2 - low mass diffraction;
c             = -1 - more collisions to follow;
c             =  0 - no excitation;
c             >  0 - low mass excitation
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension ey(3),ep(4)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr2/  scm,wp0,wm0
      common /qgarr6/  pi,bm,amws
      common /qgarr8/  wwm,be(4),dc(5),deta,almpt,ptdif,ptndi
      common /qgarr10/ am(7),ammu
      common /qgarr11/ b10
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr21/ dmmin(3),wex(3),dmres(3),wdres(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)izp,izt,wppr,wmtg

      iret=0
      jexip=0
      jexit=0
      ddmin1=0.d0
      ddmax1=0.d0
c check if remnants are excited to low mass states
      if(jexpr.eq.-2.or.jexpr.gt.0.and.qgran(b10)
     *.lt.1.d0-(1.d0-wex(icz))**dble(jexpr).and.iabs(izp).lt.7)jexip=1
      if(jextg.eq.-2.or.jextg.gt.0.and.qgran(b10)
     *.lt.1.d0-(1.d0-wex(2))**dble(jextg).and.iabs(izt).lt.7)jexit=1
c add low mass excitations if no particles produced before
      if(wppr.ge.wp0.and.jexpr.gt.0.and.jexip.eq.0.and.iabs(izp).lt.7)
     *jexip=1
      if(wmtg.ge.wm0.and.jextg.gt.0.and.jexit.eq.0.and.iabs(izt).lt.7)
     *jexit=1

      sd0=wppr*wmtg                          !energy squared available
      if(jextg.eq.-1)then                    !more collisions to follow
       dmass2=0.d0
       ddmin2=0.d0
      elseif(jexit.eq.0)then                 !no excitation
       if(iabs(izt).eq.7.or.iabs(izt).eq.8)then  !delta++/-
        dmass2=dmmin(2)
       else
        dmass2=am(2)
       endif
       ddmin2=dmass2
      else                                   !low mass excitation
       ddmin2=dmmin(2)
       if(jextg.eq.-2)ddmin2=dmres(2)        !low mass diffraction
      endif
      if(jexpr.eq.-1)then                    !more collisions to follow
       dmass1=0.d0
      elseif(jexip.eq.0)then                 !no excitation
       if(iabs(izp).eq.7.or.iabs(izp).eq.8)then  !delta++/-
        dmass1=dmmin(2)
       elseif(izp.eq.0)then                      !rho0
        dmass1=dmmin(1)
        izp=-10
       else
        dmass1=am(icz)
       endif
      else                                   !low mass excitation
       ddmin1=dmmin(icz)
       if(jexpr.eq.-2)ddmin1=dmres(icz)        !low mass diffraction
       ddmax1=dsqrt(sd0)-ddmin2
      endif


c determine mass for projectile excited remnant
      if(jexip.eq.1)then
       if(jexpr.ne.-2)then                   !low mass excitation (dM/M^2)
        if(ddmax1.gt.ddmin1)then
         dmass1=ddmin1/(1.d0-qgran(b10)*(1.d0-ddmin1/ddmax1))
        else
         dmass1=ddmin1
        endif
       else                                  !low mass diffraction (res. + PPR)
        ddmin=dmmin(icz)+am(1)
        ddmax=min(ddmax1,dmres(icz)+.5d0*wdres(icz))
        ddmax=max(ddmax,ddmin)
        wres=1.d0/(1.d0+.5d0*(1.d0+2.d0*dmres(icz)/wdres(icz))
     *  *(1.d0-(dmres(icz)+.5d0*wdres(icz))
     *  /max(ddmax1,dmres(icz)+.5d0*wdres(icz)))
     *  /(.25d0*pi+atan(2.d0*(dmres(icz)-ddmin)/wdres(icz))))
        if(qgran(b10).gt.wres)then           !PPR contribution
         dmass1=ddmax/(1.d0-qgran(b10)*(1.d0-ddmax/ddmax1))
        else                                 !resonance contribution
         dmass1=dmres(icz)+.5d0*wdres(icz)
     *   *tan(atan(2.d0*(ddmax-dmres(icz))/wdres(icz))
     *   -qgran(b10)*(atan(2.d0*(ddmax-dmres(icz))/wdres(icz))
     *   +atan(2.d0*(dmres(icz)-ddmin)/wdres(icz))))
         jexip=0
         izp=izp+10*izp/iabs(izp)
        endif
       endif
      endif

c determine mass for target excited remnant
      if(jexit.eq.1)then
       ddmax2=dsqrt(sd0)-dmass1
       if(jextg.ne.-2)then                   !low mass excitation (dM/M^2)
        if(ddmax2.gt.ddmin2)then
         dmass2=ddmin2/(1.d0-qgran(b10)*(1.d0-ddmin2/ddmax2))
        else                                  !low mass diffraction
         dmass2=ddmin2
        endif
       else                                  !low mass diffraction (res. + PPR)
        ddmin=dmmin(2)+am(1)
        ddmax=min(ddmax2,dmres(2)+.5d0*wdres(2))
        ddmax=max(ddmax,ddmin)
        wres=1.d0/(1.d0+.5d0*(1.d0+2.d0*dmres(2)/wdres(2))
     *  *(1.d0-(dmres(2)+.5d0*wdres(2))/max(ddmax2,dmres(2)+.5d0
     *  *wdres(2)))/(.25d0*pi+atan(2.d0*(dmres(2)-ddmin)/wdres(2))))
        if(qgran(b10).gt.wres)then           !PPR contribution
         dmass2=ddmax/(1.d0-qgran(b10)*(1.d0-ddmax/ddmax2))
        else                                 !resonance contribution
         dmass2=dmres(2)+.5d0*wdres(2)*tan(atan(2.d0*(ddmax-dmres(2))
     *   /wdres(2))-qgran(b10)*(atan(2.d0*(ddmax-dmres(2))/wdres(2))
     *   +atan(2.d0*(dmres(2)-ddmin)/wdres(2))))
         izt=izt+10*izt/iabs(izt)
         jexit=0
        endif
       endif
      endif

      wpp=wppr
      wpm=wmtg
      if(sd0.lt.(dmass1+dmass2)**2)then
       iret=1
       return
      endif
      dmass1=dmass1**2
      dmass2=dmass2**2

      if(jexpr.ne.-1.and.jextg.ne.-1)then
       ptmax=max(0.d0,qglam(sd0,dmass1,dmass2))
       if(jexpr.eq.-2.or.jextg.eq.-2)then
        ptmean=ptdif
       else
        ptmean=ptndi*dsqrt(dble(max(jexpr,jextg)))
       endif
       if(ptmax.lt.ptmean**2)then
1       pti=ptmax*qgran(b10)
        if(qgran(b10).gt.exp(-dsqrt(pti)/ptmean))goto 1
       else
2       pti=(ptmean*dlog(qgran(b10)*qgran(b10)))**2
        if(pti.gt.ptmax)goto 2
       endif
      else
       pti=0.d0
      endif
      amt1=dmass1+pti
      amt2=dmass2+pti
      wpd1=wpp*qgtwd(sd0,amt1,amt2)
      if(wpd1.gt.0.d0)then
       wmd1=amt1/wpd1
      else
       wmd1=0.d0
      endif
      wmd2=wpm-wmd1
      if(wmd2.gt.0.d0)then
       wpd2=amt2/wmd2
      else
       wpd2=0.d0
      endif
      pt=dsqrt(pti)
      call qgcs(c,s)

      if(jexpr.eq.-1)then
       wppr=wpd1
       if(wmd1.ne.0.d0)stop'wmd1.ne.0!!!'
      else
       ep(1)=.5d0*(wpd1+wmd1)
       ep(2)=.5d0*(wpd1-wmd1)
       ep(3)=pt*c
       ep(4)=pt*s
       wppr=0.d0
       if(jexip.eq.0)then
        call qgreg(ep,izp)
       else
        is=0
        if(izp.ne.0)is=iabs(izp)/izp
        if(icz.eq.1)then
         if(iabs(izp).ge.4)then
          ic2=-4*is
          ic1=izp-3*is
         elseif(izp.ne.0)then
          ic1=izp*(1-3*int(.5d0+qgran(b10)))
          ic2=-izp-ic1
         else
          ic1=int(1.5d0+qgran(b10))*(2*int(.5d0+qgran(b10))-1)
          ic2=-ic1
         endif
        elseif(icz.eq.2)then
         if(qgran(b10).gt..33333d0)then
          ic1=3*is
          ic2=izp-is
         else
          ic1=izp+4*is
          ic2=4*is-izp
         endif
        elseif(icz.eq.3)then
         ic1=-4*is
         ic2=izp-3*is
        endif
        call qgdeft(dmass1,ep,ey)
        call qggene(dsqrt(dmass1),dsqrt(dmass1),ey
     *  ,0.d0,1.d0,0.d0,1.d0,ic1,ic2)
       endif
      endif

      if(jextg.eq.-1)then
       wmtg=wmd2
       if(wpd2.ne.0.d0)stop'wpd2.ne.0!!!'
      else
       ep(1)=.5d0*(wpd2+wmd2)
       ep(2)=.5d0*(wpd2-wmd2)
       ep(3)=-pt*c
       ep(4)=-pt*s
       wmtg=0.d0
       if(jexit.eq.0)then
        call qgreg(ep,izt)
       else
        is=iabs(izt)/izt
        if(qgran(b10).gt..33333d0)then
         ic1=3*is
         ic2=izt-is
        else
         ic1=izt+4*is
         ic2=4*is-izt
        endif
        call qgdeft(dmass2,ep,ey)
        call qggene(dsqrt(dmass2),dsqrt(dmass2),ey
     *  ,0.d0,1.d0,0.d0,1.d0,ic2,ic1)
       endif
      endif

      if(debug.ge.3)write (moniou,202)
201   format(2x,'qgdifr - leading clusters hadronization:'
     */4x,'cluster types izp=',i2,2x,
     *'izt=',i2/4x,'available light cone momenta: wppr=',e10.3,
     *' wmtg=',e10.3)
202   format(2x,'qgdifr - end')
      return
      end

c=============================================================================
      subroutine qgfau(b,gz)
c-----------------------------------------------------------------------------
c integrands for hadron-hadron and hadron-nucleus cross-sections calculation
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      dimension gz(3),gz0(5)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)b

      do l=1,3
       gz(l)=0.d0
      enddo

      ab=float(ia(2))
      do iddp1=1,2
      do iddp2=1,2
       call qgfz(b,gz0,iddp1,iddp2)
       if(iddp1.eq.iddp2)gz(1)=gz(1)+(1.d0-gz0(1)*anorm)**ab
     * *cc(iddp1,icz)
       do l=2,3
        gz(l)=gz(l)+(1.d0-gz0(l-1)*anorm)**ab
     *  *cc(iddp1,icz)*cc(iddp2,icz)
       enddo
      enddo
      enddo

      gz(3)=gz(2)-gz(3)
      gz(2)=gz(1)-gz(2)
      gz(1)=1.d0-gz(1)

      if(debug.ge.2)write (moniou,203)gz
      if(debug.ge.3)write (moniou,202)
201   format(2x,'qgfau - integrands for hadron-hadron and hadron'
     *,'-nucleus cross-sections calculation'/4x,'b=',e10.3)
202   format(2x,'qgfau - end')
203   format(2x,'qgfau: gz=',3e10.3)
      return
      end

c=============================================================================
      subroutine qgfrag(sa,na,rc)
c-----------------------------------------------------------------------------
c connected nucleon clasters extraction - used for the nuclear spectator part
c multifragmentation
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(iapmax=208)
      dimension sa(iapmax,3)
      common /qgarr13/ nsf,iaf(iapmax)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)na
      if(debug.ge.3)then
       write (moniou,203)
       do i=1,na
        write (moniou,204)(sa(i,l),l=1,3)
       enddo
      endif

      ni=1
      ng=1
      j=0
1     j=j+1
      j1=ni+1

      do 4 i=j1,na
       ri=0.d0
       do m=1,3
        ri=ri+(sa(j,m)-sa(i,m))**2
       enddo
       if(ri.gt.rc)goto 4

       ni=ni+1
       ng=ng+1
       if(i.eq.ni)goto 4
       do m=1,3
        s0=sa(ni,m)
        sa(ni,m)=sa(i,m)
        sa(i,m)=s0
       enddo
4     continue

      if(j.lt.ni.and.na-ni.gt.0)goto 1
      nsf=nsf+1
      iaf(nsf)=ng
      if(debug.ge.3)write (moniou,206)nsf,iaf(nsf)

      ng=1
      j=ni
      ni=ni+1
      if(na.eq.ni)then
       nsf=nsf+1
       iaf(nsf)=1
       if(debug.ge.3)write (moniou,206)nsf,iaf(nsf)
      elseif(na.gt.ni)then
       goto 1
      endif

      if(debug.ge.3)write (moniou,202)
201   format(2x,'qgfrag-multifragmentation: nucleus mass number: na='
     *,i2)
202   format(2x,'qgfrag - end')
203   format(2x,'nucleons coordinates:')
204   format(2x,3e10.3)
206   format(2x,'qgfrag: fragment n',i2,2x,'fragment mass - ',i2)
      return
      end

c=============================================================================
      subroutine qgfrgm(ns,xa)
c-----------------------------------------------------------------------------
c fragmentation of the spectator part of the nucleus
c xa - array for spectator nucleons positions
c ns - total number of spectators
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      parameter(iapmax=208)
      dimension xa(iapmax,3)
      integer debug
      common /qgarr1/  ia(2),icz,icp
      common /qgarr3/  rmin,emax,eev
      common /qgarr11/ b10
c nsf - number of secondary fragments;
c iaf(i) - mass of the i-th fragment
      common /qgarr13/ nsf,iaf(iapmax)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)ns

      nsf=0
      if(ns.eq.0)then                  !no fragments
       return
      elseif(ns.eq.1)then              !single spectator nucleon recorded
       nsf=nsf+1
       iaf(nsf)=1
       if(debug.ge.3)write (moniou,205)
       return
      endif

      eex=0.d0                         !excitation energy for spectator part
           !sum of excitations due to wounded nucleons (including diffractive)
      do i=1,ia(1)-ns
c partial excitation according to f(e) ~ 1/sqrt(e) * exp(-e/(2*<e>))
       eex=eex+(qgran(b10)+qgran(b10)+qgran(b10)+
     * qgran(b10)+qgran(b10)-2.5d0)**2*2.4d0
      enddo
      if(debug.ge.3)write (moniou,203)eex

      if(eex/ns.gt.emax)then    !if eex>emax -> multifragmentation
       call qgfrag(xa,ns,rmin)  !multifragmentation (percolation algorithm)
      else                      !otherwise eveporation
       nf=npgen(eex/eev,0,ns-1) !number of eveporated nucleons (mean=eex/eev)
       nsf=nsf+1
       iaf(nsf)=ns-nf           !recording of the fragment produced
       if(debug.ge.3)write (moniou,206)iaf(nsf)

       nal=nf/4                 !number of evapotared alphas (taken as nf/4)
       if(nal.ne.0)then
        do i=1,nal              !recording the evaporated alphas
         nsf=nsf+1
         iaf(nsf)=4
        enddo
       endif
       nf=nf-4*nal

       if(nf.ne.0)then
        do i=1,nf               !recording the evaporated nucleons
         nsf=nsf+1
         iaf(nsf)=1
        enddo
       endif
       if(debug.ge.3)write (moniou,204)nf,nal
      endif
c6     continue

      if(debug.ge.3)write (moniou,202)
201   format(2x,'qgfrgm: number of spectators: ns=',i2)
202   format(2x,'qgfrgm - end')
203   format(2x,'qgfrgm: excitation energy: eex=',e10.3)
204   format(2x,'qgfrgm - evaporation: number of nucleons nf='
     *,i2,'number of alphas nal=',i2)
205   format(2x,'qgfrgm - single spectator')
206   format(2x,'qgfrgm - evaporation: mass number of the fragment:',i2)
      return
      end

c=============================================================================
      subroutine qggau(gz)
c-----------------------------------------------------------------------------
c impact parameter integration for impact parameters <bm -
c for hadron-hadron and hadron-nucleus cross-sections calculation
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension gz(3),gz0(3)
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr6/  pi,bm,amws
      common /arr3/   x1(7),a1(7)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)

      do i=1,3
       gz(i)=0.d0
      enddo
      do i=1,7
      do m=1,2
       b=bm*dsqrt(.5d0+x1(i)*(m-1.5d0))
       call qgfau(b,gz0)
       do l=1,3
        gz(l)=gz(l)+gz0(l)*a1(i)
       enddo
      enddo
      enddo

      do l=1,3
       gz(l)=gz(l)*bm**2*pi*.5d0
      enddo

      if(debug.ge.3)write (moniou,202)
201   format(2x,'qggau - nuclear cross-sections calculation')
202   format(2x,'qggau - end')
      return
      end

c=============================================================================
      subroutine qggau1(gz)
c-----------------------------------------------------------------------------
c impact parameter integration for impact parameters >bm -
c for hadron-hadron and hadron-nucleus cross-sections calculation
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension gz(3),gz0(3)
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr6/  pi,bm,amws
      common /qgarr43/ moniou
      common /arr3/   x1(7),a1(7)
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)

      do i=1,7
      do m=1,2
       b=bm-wsnuc(2)*dlog(.5d0+x1(i)*(m-1.5d0))
       call qgfau(b,gz0)
       do l=1,3
        gz(l)=gz(l)+gz0(l)*a1(i)*exp((b-bm)/wsnuc(2))*b*pi*wsnuc(2)
       enddo
      enddo
      enddo

      if(debug.ge.3)write (moniou,202)
201   format(2x,'qggau1 - nuclear cross-sections calculation')
202   format(2x,'qggau1 - end')
      return
      end

c=============================================================================
      double precision function qganrm(rnuc,wsnuc,wbnuc)
c-----------------------------------------------------------------------------
c impact parameter integration for impact parameters <bm -
c for hadron-hadron and hadron-nucleus cross-sections calculation
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /arr3/   x1(7),a1(7)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)

      qganrm=0.d0
      do i=1,7
      do m=1,2
       r=rnuc*(.5d0+x1(i)*(m-1.5d0))**(1.d0/3.d0)
       quq=(r-rnuc)/wsnuc
       if(quq.lt.1.d80)qganrm=qganrm+a1(i)/(1.d0+exp(quq))
     * *(1.d0+wbnuc*(r/rnuc)**2)
      enddo
      enddo
      qganrm=qganrm*rnuc**3*pi/1.5d0

      dnrm=0.d0
      do i=1,7
      do m=1,2
       t=.5d0+x1(i)*(m-1.5d0)
       r=rnuc-wsnuc*log(t)
       dnrm=dnrm+a1(i)/(1.d0+t)*r*r
     * *(1.d0+wbnuc*(r/rnuc)**2)
      enddo
      enddo
      qganrm=1.d0/(qganrm+dnrm*2.d0*pi*wsnuc)

      if(debug.ge.3)write (moniou,202)qganrm
201   format(2x,'qganrm - nuclear density normalization')
202   format(2x,'qganrm=',e10.3)
      return
      end

c=============================================================================
      subroutine qggene(wp0,wm0,ey0,s0x,c0x,s0,c0,ic1,ic2)
c-----------------------------------------------------------------------------
c to simulate the fragmentation of the string into secondary hadrons
c the algorithm conserves energy-momentum;
c wp0, wm0 are initial longitudinal momenta ( e+p, e-p ) of the quarks
c at the ends of the string; ic1, ic2 - their types
c the following partons types are used: 1 - u, -1 - U, 2 - d, -2 - D,
c 3 - ud, -3 - UD, 4 - s, -4 - S, 6 - uu, -6 - UU, 7 - dd, -7 - DD,
c 8 - us, -8 - US
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      character *2 tyq
      dimension wp(2),ic(2),ept(4),ep(4),ey(3),ey0(3)
c wp(1), wp(2) - current longitudinal momenta of the partons at the string
c ends, ic(1), ic(2) - their types
      common /qgarr8/  wwm,bep,ben,bek,bec,dc(5),deta,almpt,ptdif
     *,ptndi
      common /qgarr10/ am0,amn,amk,amc,amlamc,amlam,ameta,ammu
      common /qgarr11/ b10
      common /qgarr19/ ahl(3)
      common /qgarr28/ arr(5)
      common /qgarr42/ tyq(16)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      external qgran

      if(debug.ge.2)write (moniou,201)tyq(8+ic1),tyq(8+ic2)
     *,wp0,wm0,ey0,s0x,c0x,s0,c0

      ww=wp0*wm0                              !mass squared for the string
      ept(1)=.5d0*(wp0+wm0)                   !4-momentum for the string
      ept(2)=.5d0*(wp0-wm0)
      ept(3)=0.d0
      ept(4)=0.d0

      if(iabs(ic1).eq.5.or.iabs(ic2).eq.5.or.iabs(ic1).gt.8
     *.or.iabs(ic2).gt.8)stop'qggene: problem with parton types'

      ic(1)=ic1                               !parton types at string ends
      ic(2)=ic2

1     sww=dsqrt(ww)
      call qgdeft(ww,ept,ey)                  !boost to c.m.  for the string
      j=int(2.d0*qgran(b10))+1                !choose string end to start

      if(debug.ge.3)then
       iqt=8+ic(j)
       write (moniou,203)j,tyq(iqt),ww
      endif

      iab=iabs(ic(j))
      is=ic(j)/iab
      if(iab.eq.8)then
       iab=6
      elseif(iab.gt.5)then
       iab=3
      endif
      iaj=iabs(ic(3-j))
      if(iaj.eq.8)then
       iaj=6
      elseif(iaj.gt.5)then
       iaj=3
      endif
      if(iab.eq.5)stop'no charm anymore!'

      if(iaj.eq.3)then
       restm=amn
      elseif(iaj.eq.4)then
       restm=amk
      elseif(iaj.eq.5)then
       stop'no charm anymore!'
      elseif(iaj.eq.6)then
       restm=amlam
      else
       restm=am0
      endif

      if(iab.le.2.and.sww.gt.restm+2.d0*am0+wwm
     *.or.iab.eq.3.and.sww.gt.restm+am0+amn+wwm
     *.or.iab.eq.4.and.sww.gt.restm+am0+amk+wwm
     *.or.iab.eq.6.and.sww.gt.restm+am0+amlam+wwm)then !more than 2 particles
       if(iab.le.2)then                                !light quark string end
        if(iab.eq.2.and.iabs(ic(3-j)).ne.7
     *  .and.sww.gt.restm+2.d0*amlam.and.qgran(b10).lt.dc(1)*dc(2))then
c lambda generation
         restm=(restm+amlam)**2
         bet=ben
         ami=amlam**2
         alf=almpt-arr(2)+arr(1)-arr(3)
         blf=1.d0-arr(2)-arr(3)
         ic0=6*is                                      !(anti-)lambda
         ic(j)=-8*is                                   !US(us)
        elseif(sww.gt.restm+2.d0*amn.and.qgran(b10).lt.dc(1))then
c nucleon generation
         restm=(restm+amn)**2
         bet=ben
         ami=amn**2
         alf=almpt-arr(2)
         blf=1.d0-arr(1)-arr(2)
         ic0=ic(j)+is
         ic(j)=-3*is
        elseif(sww.gt.restm+2.d0*amk.and.qgran(b10).lt.dc(2))then
c kaon generation
         restm=(restm+amk)**2
         bet=bek
         ami=amk**2
         alf=almpt-arr(3)
         blf=1.d0-arr(1)-arr(3)
         ic0=ic(j)+3*is
         ic(j)=4*is
        elseif(sww.gt.restm+ameta+am0.and.qgran(b10).lt.deta)then
c eta generation
         restm=(restm+am0)**2
         bet=bek
         ami=ameta**2
         alf=almpt-arr(1)
         blf=1.d0-2.d0*arr(1)
         ic0=10
        else
c pion generation
         restm=(restm+am0)**2
         bet=bep
         ami=am0**2
         alf=almpt-arr(1)
         blf=1.d0-2.d0*arr(1)
         if(qgran(b10).lt..3333d0)then
          ic0=0
         else
          ic0=3*is-2*ic(j)
          ic(j)=3*is-ic(j)
         endif
        endif

       elseif(iab.eq.3)then
        if(sww.gt.restm+amk+amlam.and.qgran(b10).lt.dc(4)
     *  .and.iabs(ic(j)).eq.3)then
c lambda generation
         restm=(restm+amk)**2
         bet=bek
         ami=amlam**2
         alf=almpt-arr(3)
         blf=1.d0-arr(2)-arr(3)
         ic0=6*is
         ic(j)=-4*is
        else
c nucleon generation
         restm=(restm+am0)**2
         bet=ben
         ami=amn**2
         alf=almpt-arr(1)
         blf=1.d0-arr(1)-arr(2)
         if(iabs(ic(j)).eq.3)then
          ic0=is*int(2.5d0+qgran(b10))
          ic(j)=is-ic0
         else
          ic0=ic(j)-4*is
          ic(j)=ic0-4*is
         endif
        endif

       elseif(iab.eq.4)then
        if(sww.gt.restm+amn+amlam.and.qgran(b10).lt.dc(1))then
c lambda generation
         restm=(restm+amn)**2
         bet=ben
         ami=amlam**2
         alf=almpt-arr(2)
         blf=1.d0-arr(2)-arr(3)
         ic0=6*is
         ic(j)=-3*is
        else
c kaon generation
         restm=(restm+am0)**2
         bet=bep
         ami=amk**2
         alf=almpt-arr(1)
         blf=1.d0-arr(1)-arr(3)
         ic(j)=is*int(1.5d0+qgran(b10))
         ic0=-3*is-ic(j)
        endif

       elseif(iab.eq.6)then
c lambda generation
        restm=(restm+am0)**2
        bet=bep
        ami=amlam**2
        alf=almpt-arr(1)
        blf=1.d0-arr(2)-arr(3)
        ic0=6*is
        ic(j)=-2*is
       endif

       ptmax=qglam(ww,restm,ami)
       if(ptmax.lt.0.)ptmax=0.

       if(ptmax.lt.bet**2)then
2       pti=ptmax*qgran(b10)
        if(qgran(b10).gt.exp(-dsqrt(pti)/bet))goto 2
       else
3       pti=(bet*dlog(qgran(b10)*qgran(b10)))**2
        if(pti.gt.ptmax)goto 3
       endif

       amt=ami+pti
       restm1=restm+pti
       zmin=1.d0-qgtwd(ww,restm1,amt)
       zmax=qgtwd(ww,amt,restm1)

       z1=(1.d0-zmax)**alf
       z2=(1.d0-zmin)**alf
4      z=1.-(z1+(z2-z1)*qgran(b10))**(1./alf)
       if(qgran(b10).gt.(z/zmax)**blf)goto 4
       wp(j)=z*sww
       wp(3-j)=amt/wp(j)
       ep(1)=.5d0*(wp(1)+wp(2))
       ep(2)=.5d0*(wp(1)-wp(2))
       pti=dsqrt(pti)
       call qgcs(c,s)
       ep(3)=pti*c
       ep(4)=pti*s
       ept(1)=sww-ep(1)
       do i=2,4
        ept(i)=-ep(i)
       enddo
       ww=qgnrm(ept)
       if(ww.lt.restm)goto 4

       call qgtran(ep,ey,1)
       call qgtran(ept,ey,1)
       if(s0x.ne.0.d0.or.s0.ne.0.d0)then
        call qgrota(ep,s0x,c0x,s0,c0)
       endif
       if(ey0(1)*ey0(2)*ey0(3).ne.1.d0)then
        call qgtran(ep,ey0,1)
       endif
       call qgreg(ep,ic0)

      else
       ami2=restm**2
       bet=bep
       if(iab.eq.6.or.iaj.eq.6)then
        if(iab.eq.6)then
         ami=amlam**2
         ic(j)=6*is
         if(iaj.eq.6)then
          ic(3-j)=-6*is
         elseif(iaj.eq.4)then
          ic(3-j)=-5*is
         elseif(iaj.le.2)then
          ic(3-j)=2*is-ic(3-j)
         else
          if(iabs(ic(3-j)).eq.3)then
           ic(3-j)=-3*is
          elseif(iabs(ic(3-j)).eq.6)then
           ic(3-j)=-2*is
          else
           stop'wrong parton types'
          endif
         endif
        elseif(iab.eq.4)then
         ami=amk**2
         ic(j)=-5*is
         ic(3-j)=6*is
        elseif(iab.le.2)then
         ami=am0**2
         ic(j)=2*is-ic(j)
         ic(3-j)=6*is
        else
         ami=amn**2
         ic(3-j)=-6*is
         if(iabs(ic(j)).eq.3)then
          ic(j)=3*is
         elseif(iabs(ic(j)).eq.6)then
          ic(j)=2*is
         else
          stop'wrong parton types'
         endif
        endif

       elseif(iab.le.2.and.iaj.le.2)then
        if(sww.gt.2.d0*amk.and.qgran(b10).lt.dc(2))then
         bet=bek
         ami=amk**2
         ami2=ami
         ic(j)=ic(j)+3*is
         ic(3-j)=ic(3-j)-3*is
        else
         ami=am0**2
         ic0=-ic(1)-ic(2)
         if(ic0.ne.0)then
          ic(j)=ic0*int(.5d0+qgran(b10))
          ic(3-j)=ic0-ic(j)
         else
          if(qgran(b10).lt..2d0)then
           ic(j)=0
           ic(3-j)=0
          else
           ic(j)=3*is-2*ic(j)
           ic(3-j)=-ic(j)
          endif
         endif
        endif

       elseif(iab.eq.3.or.iaj.eq.3)then
        if(iab.eq.3)then
         ami=amn**2
         if(iabs(ic(j)).eq.3)then
          if(iaj.eq.3)then
           if(iabs(ic(3-j)).eq.3)then
            if(sww.gt.2.d0*amlam.and.qgran(b10).lt.dc(4))then
             bet=bek
             ami=amlam**2
             ami2=ami
             ic(j)=6*is
             ic(3-j)=-6*is
            else
             ic(j)=is*int(2.5d0+qgran(b10))
             ic(3-j)=-ic(j)
            endif
           else
            ic(3-j)=ic(3-j)+4*is
            ic(j)=5*is+ic(3-j)
           endif
          elseif(iaj.lt.3)then
           if(sww.gt.amlam+amk.and.qgran(b10).lt.dc(4))then
            bet=bek
            ami=amlam**2
            ami2=amk**2
            ic(j)=6*is
            ic(3-j)=ic(3-j)+3*is
           else
            if(qgran(b10).lt..3333d0)then
             ic(j)=ic(3-j)+is
             ic(3-j)=0
            else
             ic(j)=is*(4-iaj)
             ic(3-j)=is*(3-2*iaj)
            endif
           endif
          elseif(iaj.eq.4)then
           ic(j)=is*int(2.5d0+qgran(b10))
           ic(3-j)=-ic(j)-2*is
          endif
         else
          if(iabs(ic(3-j)).gt.4)stop'qggene: problem with parton types'
          ic(j)=ic(j)-4*is
          ic0=ic(j)-4*is
          if(iaj.eq.3)then
           ic(3-j)=ic0-is
          elseif(iaj.lt.3)then
           ic(3-j)=-ic(3-j)-ic0
          elseif(iaj.eq.4)then
           ic(3-j)=ic0-3*is
          endif
         endif
        else
         if(iabs(ic(3-j)).eq.3)then
          if(iab.lt.3)then
           if(sww.gt.amlam+amk.and.qgran(b10).lt.dc(4))then
            bet=bek
            ami2=amlam**2
            ami=amk**2
            ic(j)=ic(j)+3*is
            ic(3-j)=6*is
           else
            ami=am0**2
            if(qgran(b10).lt..3333d0)then
             ic(3-j)=ic(j)+is
             ic(j)=0
            else
             ic(3-j)=is*(4-iab)
             ic(j)=is*(3-2*iab)
            endif
           endif
          elseif(iab.eq.4)then
           ami=amk**2
           ic(3-j)=is*int(2.5d0+qgran(b10))
           ic(j)=-ic(3-j)-2*is
          endif
         else
          ic(3-j)=ic(3-j)-4*is
          ic0=ic(3-j)-4*is
          if(iab.lt.3)then
           ami=am0**2
           ic(j)=-ic0-ic(j)
          elseif(iab.eq.4)then
           ami=amk**2
           ic(j)=ic0-3*is
          endif
         endif
        endif
       elseif(iab.eq.4.or.iaj.eq.4)then
        if(iab.eq.4)then
         ami=amk**2
         if(iaj.eq.4)then
          ic(j)=-is*int(4.5d0+qgran(b10))
          ic(3-j)=-ic(j)
         else
          ic0=ic(3-j)+int(.6667d0+qgran(b10))*(-3*is-2*ic(3-j))
          ic(j)=ic0-3*is
          ic(3-j)=ic0-ic(3-j)
         endif
        else
         ami=am0**2
         ic0=ic(j)+int(.6667d0+qgran(b10))*(3*is-2*ic(j))
         ic(j)=ic0-ic(j)
         ic(3-j)=ic0+3*is
        endif
       endif

       ptmax=qglam(ww,ami2,ami)
       if(ptmax.lt.0.)ptmax=0.
       if(ptmax.lt.bet**2)then
5       pti=ptmax*qgran(b10)
        if(qgran(b10).gt.exp(-dsqrt(pti)/bet))goto 5
       else
6       pti=(bet*dlog(qgran(b10)*qgran(b10)))**2
        if(pti.gt.ptmax)goto 6
       endif
       amt1=ami+pti
       amt2=ami2+pti
       z=qgtwd(ww,amt1,amt2)
       wp(j)=z*sww
       wp(3-j)=amt1/wp(j)
       ep(1)=.5d0*(wp(1)+wp(2))
       ep(2)=.5d0*(wp(1)-wp(2))
       pti=dsqrt(pti)
       call qgcs(c,s)
       ep(3)=pti*c
       ep(4)=pti*s
       ept(1)=sww-ep(1)
       do i=2,4
        ept(i)=-ep(i)
       enddo
       call qgtran(ep,ey,1)
       call qgtran(ept,ey,1)
       if(s0x.ne.0.d0.or.s0.ne.0.d0)then
        call qgrota(ep,s0x,c0x,s0,c0)
        call qgrota(ept,s0x,c0x,s0,c0)
       endif
       if(ey0(1)*ey0(2)*ey0(3).ne.1.d0)then
        call qgtran(ep,ey0,1)
        call qgtran(ept,ey0,1)
       endif

       call qgreg(ep,ic(j))
       call qgreg(ept,ic(3-j))
       if(debug.ge.3)write (moniou,202)
       return
      endif
      goto 1

201   format(2x,'qggene: parton flavors at the ends of the string:'
     *,2x,a2,2x,a2/4x,'light cone momenta of the string: ',e10.3
     *,2x,e10.3/4x,'ey0=',3e10.3/4x,'s0x=',e10.3,2x,'c0x=',e10.3
     *,2x,'s0=',e10.3,2x,'c0=',e10.3)
202   format(2x,'qggene - end')
203   format(2x,'qggene: current parton flavor at the end '
     *,i1,' of the string: ',a2/4x,' string mass: ',e10.3)
      end

c=============================================================================
      subroutine qgxjet
c-----------------------------------------------------------------------------
c procedure for jet hadronization
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(njmax=50000)
      dimension ep(4),ept(4),ept1(4),ey(3)
     *,epj(4,2,2*njmax),ipj(2,2*njmax)
      common /qgarr8/  wwm,be(4),dc(5),deta,almpt,ptdif,ptndi
      common /qgarr10/ am(7),ammu
      common /qgarr11/ b10
      common /qgarr36/ epjet(4,njmax),ipjet(njmax),njtot
      common /qgarr43/ moniou
      common /qgdebug/  debug
      external qgran

      if(debug.ge.2)write (moniou,201)njtot
201   format(2x,'qgxjet: total number of jets njtot=',i4)

      nj0=1
      njet0=0
      nrej=0

1     njet=njet0
      do i=1,4
       ept(i)=epjet(i,nj0)
       epj(i,1,njet+1)=ept(i)
      enddo
      iq1=ipjet(nj0)
      ipj(1,njet+1)=iq1

      if(iabs(iq1).le.2)then
       am1=am(1)
       if(iq1.gt.0)then
        jq=1
       else
        jq=2
       endif
      elseif(iabs(iq1).eq.4)then
       am1=am(3)
       if(iq1.gt.0)then
        jq=1
       else
        jq=2
       endif
      else
       am1=am(2)
       if(iq1.gt.0)then
        jq=2
       else
        jq=1
       endif
      endif

      ij=nj0
2     ij=ij+1
      njet=njet+1
      iq2=ipjet(ij)

      if(iq2.eq.0)then
       aks=qgran(b10)
       do i=1,4
        epi=epjet(i,ij)*aks
        epj(i,2,njet)=epi
        ept(i)=ept(i)+epi
       enddo
       if(qgran(b10).lt.dc(2))then
        ipj(2,njet)=4*(2*jq-3)
        amj=am(3)
       else
        ipj(2,njet)=int(1.5d0+qgran(b10))*(2*jq-3)
        amj=am(1)
       endif

       if(qgnrm(ept).gt.(am1+amj)**2)then
        if(debug.ge.3)write (moniou,211)njet,ipj(1,njet),ipj(2,njet)
     *  ,qgnrm(ept),ept

        ipj(1,njet+1)=-ipj(2,njet)
        do i=1,4
         ept(i)=epjet(i,ij)-epj(i,2,njet)
         epj(i,1,njet+1)=ept(i)
        enddo
        am1=amj
        goto 2
       elseif(nrej.lt.100000)then
        nrej=nrej+1
        goto 1
       else
3       continue
        do i=1,4
         ept(i)=epjet(i,ij)+epjet(i,ij-1)+epjet(i,ij+1)
         ep(i)=epjet(i,ij-1)
         ept1(i)=ept(i)
        enddo
        ww=qgnrm(ept1)
        if(ww.le.0.)then
         if(ij.gt.nj0+1)then
          ij=ij-1
          goto 3
         else
          ij=ij+1
          goto 3
         endif
        endif
        ipjet(ij)=ipjet(ij+1)
        sww=sqrt(ww)
        call qgdeft(ww,ept1,ey)
        call qgtran(ep,ey,-1)
        call qgdefr(ep,s0x,c0x,s0,c0)
        ep(1)=.5d0*sww
        ep(2)=.5d0*sww
        ep(3)=0.d0
        ep(4)=0.d0
        call qgrota(ep,s0x,c0x,s0,c0)
        call qgtran(ep,ey,1)
        do i=1,4
         epjet(i,ij-1)=ep(i)
         epjet(i,ij)=ept(i)-ep(i)
        enddo

        if(njtot.gt.ij+1)then
         do j=ij+1,njtot-1
          ipjet(j)=ipjet(j+1)
         do i=1,4
          epjet(i,j)=epjet(i,j+1)
         enddo
         enddo
        endif
        nrej=0
        njtot=njtot-1
        goto 1
       endif

      else
       ipj(2,njet)=iq2
       do i=1,4
        epi=epjet(i,ij)
        epj(i,2,njet)=epi
        ept(i)=ept(i)+epi
       enddo

       if(iabs(iq2).le.2)then
        am2=am(1)
       elseif(iabs(iq2).eq.4)then
        am2=am(3)
       else
        am2=am(2)
       endif

       if(qgnrm(ept).gt.(am1+am2)**2)then
        if(debug.ge.3)write (moniou,211)njet,ipj(1,njet),ipj(2,njet)
     *  ,qgnrm(ept),ept

        nj0=ij+1
        njet0=njet
        nrej=0
        if(ij.lt.njtot)then
         goto 1
        else
         goto 5
        endif
       elseif(nrej.lt.100000)then
        nrej=nrej+1
        goto 1
       else
4       continue
        do i=1,4
         ept(i)=epjet(i,ij)+epjet(i,ij-1)+epjet(i,ij-2)
         ep(i)=epjet(i,ij-2)
         ept1(i)=ept(i)
        enddo
        ww=qgnrm(ept1)
        if(ww.lt.0.d0)then
         ij=ij-1
         goto 4
        endif
        ipjet(ij-1)=ipjet(ij)
        sww=sqrt(ww)
        call qgdeft(ww,ept1,ey)
        call qgtran(ep,ey,-1)
        call qgdefr(ep,s0x,c0x,s0,c0)
        ep(1)=.5d0*sww
        ep(2)=.5d0*sww
        ep(3)=0.d0
        ep(4)=0.d0
        call qgrota(ep,s0x,c0x,s0,c0)
        call qgtran(ep,ey,1)
        do i=1,4
         epjet(i,ij-2)=ep(i)
         epjet(i,ij-1)=ept(i)-ep(i)
        enddo

        if(ij.lt.njtot)then
         do j=ij,njtot-1
          ipjet(j)=ipjet(j+1)
         do i=1,4
          epjet(i,j)=epjet(i,j+1)
         enddo
         enddo
        endif

        nrej=0
        njtot=njtot-1
        goto 1
       endif
      endif

5     continue
      do ij=1,njet
       do i=1,4
        ep(i)=epj(i,1,ij)
        ept(i)=ep(i)+epj(i,2,ij)
       enddo
c invariant mass squared for the jet
       ww=qgnrm(ept)

       if(debug.ge.3)write (moniou,208)
     * ij,njet,ww,ipj(1,ij),ipj(2,ij)

       sww=dsqrt(ww)
       call qgdeft(ww,ept,ey)
       call qgtran(ep,ey,-1)
       call qgdefr(ep,s0x,c0x,s0,c0)
       call qggene(sww,sww,ey,s0x,c0x,s0,c0,ipj(1,ij),ipj(2,ij))
      enddo

      if(debug.ge.3)write (moniou,202)
202   format(2x,'qgxjet - end')
208   format(2x,'qgxjet: ij=',i2,2x,'njet=',i3,2x,'ww=',e10.3
     *,2x,'ic=',2i3)
211   format(2x,'qgxjet: njet=',i3,2x,'ic=',2i2,2x,'mass=',e10.3
     *,2x,'ep=',4e10.3)
      return
      end

c=============================================================================
      double precision function qgrot(b,s)
c-----------------------------------------------------------------------------
c convolution of nuclear profile functions (axial angle integration)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /arr8/  x2(4),a2
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)b,s

      qgrot=0.d0
      do i=1,4
       sb1=b**2+s**2-2.*b*s*(2.*x2(i)-1.)
       sb2=b**2+s**2-2.*b*s*(1.-2.*x2(i))
       qgrot=qgrot+(qgt(sb1)+qgt(sb2))
      enddo
      qgrot=qgrot*a2

      if(debug.ge.2)write (moniou,202)qgrot
201   format(2x,'qgrot - axial angle integration of the ',
     *'nuclear profile function'/4x,
     *'impact parameter b=',e10.3,2x,'nucleon coordinate s=',e10.3)
202   format(2x,'qgrot=',e10.3)
      return
      end

c=============================================================================
      subroutine qgstr(wpi0,wmi0,wp0,wm0,ic10,ic120,ic210,ic20,jp,jt)
c-----------------------------------------------------------------------------
c fragmentation process for the pomeron ( quarks and antiquarks types at the
c ends of the two strings are determined, energy-momentum is shared
c between them and strings fragmentation is simulated )
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension ey(3)
      common /qgarr6/  pi,bm,amws
      common /qgarr8/  wwm,be(4),dc(5),deta,almpt,ptdif,ptndi
      common /qgarr10/ am(7),ammu
      common /qgarr11/ b10
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)wpi0,wmi0,wp0,wm0

      do i=1,3
       ey(i)=1.d0
      enddo
      wpi=wpi0
      wmi=wmi0
c quark-antiquark types (1 - u, 2 - d, -1 - u~, -2 - d~); s- and d- quarks are
c taken into consideration at the fragmentation step
      if(ic10.eq.0)then
       if(qgran(b10).lt.dc(2))then
        ic1=4
        ic12=-4
       else
        ic1=int(1.5+qgran(b10))
        ic12=-ic1
       endif
      elseif(ic10.gt.0)then
       ic1=ic10
       ic12=ic120
      else
       ic1=ic120
       ic12=ic10
      endif

      if(ic20.eq.0)then
       if(qgran(b10).lt.dc(2))then
        ic2=4
        ic21=-4
       else
        ic2=int(1.5+qgran(b10))
        ic21=-ic2
       endif
      elseif(ic20.gt.0)then
       ic2=ic20
       ic21=ic210
      else
       ic2=ic210
       ic21=ic20
      endif

c longitudinal momenta for the strings
      if(jp.eq.0)then
       wp1=wpi*cos(pi*qgran(b10))**2
      else
1      xp=.5d0*qgran(b10)**2
       if(qgran(b10).gt.(2.d0*(1.d0-xp))**(-.5d0))goto 1
       wp1=wpi*xp
       if(qgran(b10).lt..5d0)wp1=wpi-wp1
      endif
      if(jt.eq.0)then
       wm1=wmi*cos(pi*qgran(b10))**2
      else
2      xm=.5d0*qgran(b10)**2
       if(qgran(b10).gt.(2.d0*(1.d0-xm))**(-.5d0))goto 2
       wm1=wmi*xm
       if(qgran(b10).lt..5d0)wm1=wmi-wm1
      endif
      wpi=wpi-wp1
      wmi=wmi-wm1
c string masses
      sm1=wp1*wm1
      sm2=wpi*wmi

c mass thresholds
      if(iabs(ic1).le.2)then
       am1=am(1)
      elseif(iabs(ic1).eq.3)then
       am1=am(2)
      elseif(iabs(ic1).eq.4)then
       am1=am(3)
      else
       am1=0.d0
       stop 'should not happen in qgstr 1 !'
      endif
      if(iabs(ic2).le.2)then
       am2=am(1)
      elseif(iabs(ic2).eq.3)then
       am2=am(2)
      elseif(iabs(ic2).eq.4)then
       am2=am(3)
      else
       am2=0.d0
       stop 'should not happen in qgstr 2 !'
      endif
      if(iabs(ic12).le.2)then
       am12=am(1)
      elseif(iabs(ic12).eq.3)then
       am12=am(2)
      elseif(iabs(ic12).eq.4)then
       am12=am(3)
      else
       am12=0.d0
       stop 'should not happen in qgstr 3 !'
      endif
      if(iabs(ic21).le.2)then
       am21=am(1)
      elseif(iabs(ic21).eq.3)then
       am21=am(2)
      elseif(iabs(ic21).eq.4)then
       am21=am(3)
      else
       am21=0.d0
       stop 'should not happen in qgstr 4 !'
      endif

c too short strings are neglected (energy is given to partner string
c or to the hadron (nucleon) to which the pomeron is connected)
      if(sm1.gt.am1+am21.and.sm2.gt.am2+am12)then
c strings fragmentation is simulated - gener
       call qggene(wp1,wm1,ey,0.d0,1.d0,0.d0,1.d0,ic1,ic21)
       call qggene(wpi,wmi,ey,0.d0,1.d0,0.d0,1.d0,ic12,ic2)
      elseif((wpi+wp1)*(wmi+wm1).gt.am1+am21)then
       call qggene(wp1+wpi,wm1+wmi,ey,0.d0,1.d0,0.d0,1.d0,ic1,ic21)
      elseif((wpi+wp1)*(wmi+wm1).gt.am2+am12)then
       call qggene(wp1+wpi,wm1+wmi,ey,0.d0,1.d0,0.d0,1.d0,ic12,ic2)
      else
       wp0=wp0+wp1+wpi
       wm0=wm0+wm1+wmi
      endif

      if(debug.ge.3)write (moniou,202)wp0,wm0
201   format(2x,'qgstr: wpi0=',e10.3,2x,'wmi0=',e10.3
     *,2x,'wp0=',e10.3,2x,'wm0=',e10.3)
202   format(2x,'qgstr - returned light cone momenta:'
     *,2x,'wp0=',e10.3,2x,'wm0=',e10.3)
      return
      end

c===========================================================================
      double precision function qgt(b)
c---------------------------------------------------------------------------
c nuclear profile function value at impact parameter squared b
c---------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr6/  pi,bm,amws
      common /arr3/   x1(7),a1(7)
      common /qgarr43/ moniou
      common /qgdebug/  debug

      if(debug.ge.2)write (moniou,201)b

      qgt=0.
      zm=rnuc(2)**2-b
      if(zm.gt.4.*b)then
       zm=dsqrt(zm)
      else
       zm=2.*dsqrt(b)
      endif

      do i=1,7
      do m=1,2
       z1=zm*(.5d0+x1(i)*(m-1.5d0))
       r=dsqrt(b+z1**2)
       quq=(r-rnuc(2))/wsnuc(2)
       if (quq.lt.85.)qgt=qgt+a1(i)/(1.+exp(quq))
     * *(1.d0+wbnuc(2)*(r/rnuc(2))**2)
      enddo
      enddo
      qgt=qgt*zm*0.5d0

      dt=0.
      do i=1,7
      do m=1,2
       z1=zm-wsnuc(2)*log(.5d0+x1(i)*(m-1.5d0))
       r=dsqrt(b+z1**2)
       quq=(r-rnuc(2)-z1+zm)/wsnuc(2)
       if (quq.lt.85.)dt=dt+a1(i)/(exp((zm-z1)/wsnuc(2))+exp(quq))
     * *(1.d0+wbnuc(2)*(r/rnuc(2))**2)
      enddo
      enddo
      qgt=qgt+dt*wsnuc(2)/2.d0

      if(debug.ge.3)write (moniou,202)qgt
201   format(2x,'qgt - nuclear profile function value at impact'
     *,' parameter squared b=',e10.3)
202   format(2x,'qgt=',e10.3)
      return
      end

c=============================================================================
      block data qgdata
c-----------------------------------------------------------------------------
c constants for numerical integration (gaussian weights)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /arr1/ trnuc(56),twsnuc(56),twbnuc(56)
      common /arr3/ x1(7),a1(7)
      common /arr4/ x4(2),a4(2)
      common /arr5/ x5(2),a5(2)
      common /arr8/ x2(4),a2
      common /arr9/ x9(3),a9(3)
      data x1/.9862838d0,.9284349d0,.8272013d0,.6872929d0,.5152486d0,
     *.3191124d0,.1080549d0/
      data a1/.03511946d0,.08015809d0,.1215186d0,.1572032d0,
     *.1855384d0,.2051985d0,.2152639d0/
      data x2/.00960736d0,.0842652d0,.222215d0,.402455d0/
      data a2/.392699d0/
      data x4/ 0.339981,0.861136/
      data a4/ 0.652145,0.347855/
      data x5/.585786d0,3.41421d0/
      data a5/.853553d0,.146447d0/
      data x9/.93247d0,.661209d0,.238619d0/
      data a9/.171324d0,.360762d0,.467914d0/
      data trnuc/0.69d0,1.71d0,1.53d0,1.37d0,1.37d0,2.09d0,1.95d0
     *,1.95d0,2.06d0,1.76d0,1.67d0,1.74d0,1.66d0,2.57d0,2.334d0
     *,2.608d0,2.201d0,2.331d0,2.58d0,2.791d0,2.791d0,2.782d0,2.74d0
     *,3.192d0,3.22d0,3.05d0,3.07d0,3.34d0,3.338d0,3.252d0
     *,3.369d0,3.244d0,3.244d0,3.313d0,3.476d0,3.54d0,3.554d0
     *,3.554d0,3.743d0,3.73d0,3.744d0,3.759d0,3.774d0,3.788d0
     *,3.802d0,3.815d0,3.829d0,3.843d0,3.855d0,3.941d0
     *,3.94d0,3.984d0,4.d0,4.074d0,3.89d0,4.111d0/
      data twsnuc/0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0
     *,0.55d0,0.55d0,0.56d0,0.56d0,0.5052d0,0.498d0,0.513d0
     *,0.55d0,0.55d0,0.567d0,0.698d0,0.698d0,0.549d0,0.55d0
     *,0.604d0,0.58d0,0.523d0,0.519d0,0.58d0,0.547d0,0.553d0
     *,0.582d0,0.55d0,0.55d0,0.7d0,0.599d0,0.507d0,0.588d0
     *,0.588d0,0.585d0,0.62d0,0.55d0,0.55d0,0.55d0,0.55d0
     *,0.55d0,0.55d0,0.55d0,0.588d0,0.588d0
     *,0.566d0,0.505d0,0.542d0,0.557d0,0.536d0,0.567d0,0.558d0/
      data twbnuc/0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0
     *,0.d0,0.d0,0.d0,0.d0,-0.18d0,0.139d0,-0.051d0,0.d0,0.d0
     *,0.d0,-0.168d0,0.d0,0.d0,0.d0,-0.249d0,-0.236d0,0.d0,0.d0
     *,0.233d0,-0.203d0,-0.078d0,-0.173d0,0.d0,0.d0,0.d0,-0.1d0
     *,0.d0,-0.13d0,-0.13d0,-0.201d0,-0.19d0,0.d0,0.d0,0.d0,0.d0
     *,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0
     *,0.d0,0.d0/
      end

c-----------------------------------------------------------------------
      real function qggamfun(x)
c-----------------------------------------------------------------------
c     gamma fctn
c-----------------------------------------------------------------------
      dimension c(13)
      data c
     1/ 0.00053 96989 58808, 0.00261 93072 82746, 0.02044 96308 23590,
     2  0.07309 48364 14370, 0.27964 36915 78538, 0.55338 76923 85769,
     3  0.99999 99999 99998,-0.00083 27247 08684, 0.00469 86580 79622,
     4  0.02252 38347 47260,-0.17044 79328 74746,-0.05681 03350 86194,
     5  1.13060 33572 86556/
      qggamfun=0
      z=x
      if(x .gt. 0.0) goto1
      if(x .eq. aint(x)) goto5
      z=1.0-z
    1 f=1.0/z
      if(z .le. 1.0) goto4
      f=1.0
    2 continue
      if(z .lt. 2.0) goto3
      z=z-1.0
      f=f*z
      goto2
    3 z=z-1.0
    4 qggamfun=
     1 f*((((((c(1)*z+c(2))*z+c(3))*z+c(4))*z+c(5))*z+c(6))*z+c(7))/
     2   ((((((c(8)*z+c(9))*z+c(10))*z+c(11))*z+c(12))*z+c(13))*z+1.0)
      if(x .gt. 0.0) return
      qggamfun=3.141592653589793/(sin(3.141592653589793*x)*qggamfun)
      return
    5 write(*,10)x
   10 format(1x,'argument of gamma fctn = ',e20.5)
      stop
      end

c-------------------------------------------------------------------------------
      subroutine qgcrossc(niter,gtot,gprod,gabs,gdd,gqel,gcoh)
c-------------------------------------------------------------------------------
c nucleus-nucleus (nucleus-hydrogen) interaction cross sections
c gtot  - total cross section
c gprod - production cross section (projectile diffraction included)
c gabs  - cut pomerons cross section
c gdd   - projectile diffraction cross section
c gqel  - quasielastic (projectile nucleon knock-out) cross section
c gcoh  - coherent (elastic with respect to the projectile) cross section
c (target diffraction is not treated explicitely and contributes to
c gdd, gqel, gcoh).
c-------------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      parameter(iapmax=208)
      dimension wabs(28),wdd(28),wqel(28),wcoh(28)
     *,wprod(28),b0(28),ai(28),xa(iapmax,3),xb(iapmax,3)
      common /qgarr1/  ia(2),icz,icp
      common /qgarr5/  rnuc(2),wsnuc(2),wbnuc(2),anorm
     *,cr1(2),cr2(2),cr3(2)
      common /qgarr6/  pi,bm,amws
      common /qgarr11/ b10
      common /qgarr16/ cc(2,3),iddp(iapmax),iddt(iapmax)
      common /arr3/   x1(7),a1(7)
      EXTERNAL qgran

      e1=exp(-1.d0)

      do i=1,7
       b0(15-i)=bm*sqrt((1.d0+x1(i))/2.d0)
       b0(i)=bm*sqrt((1.d0-x1(i))/2.d0)
       ai(i)=a1(i)*bm**2*5.d0*pi
       ai(15-i)=ai(i)
      enddo

      do i=1,7
       tp=(1.d0+x1(i))/2.d0
       tm=(1.d0-x1(i))/2.d0
       b0(14+i)=bm-log(tp)*max(wsnuc(1),wsnuc(2))
       b0(29-i)=bm-log(tm)*max(wsnuc(1),wsnuc(2))
       ai(14+i)=a1(i)*b0(14+i)/tp*10.d0*max(wsnuc(1),wsnuc(2))*pi
       ai(29-i)=a1(i)*b0(29-i)/tm*10.d0*max(wsnuc(1),wsnuc(2))*pi
      enddo

      do i=1,28
       wabs(i)=0.
       wdd(i)=0.
       wqel(i)=0.
       wcoh(i)=0.
      enddo

      do nc=1,niter
       do i=1,ia(2)
        iddt(i)=1+int(qgran(b10)+cc(2,2))
       enddo

       if(ia(1).eq.1)then
        xa(1,1)=0.d0
        xa(1,2)=0.d0
        xa(1,3)=0.d0
       else
        call qggea(ia(1),xa,1)
       endif
       if(ia(2).eq.1)then
        xb(1,1)=0.d0
        xb(1,2)=0.d0
        xb(1,3)=0.d0
       else
        call qggea(ia(2),xb,2)
       endif

       do i=1,28
        call qggcr(b0(i),gabs,gdd,gqel,gcoh,xa,xb,ia(1))
        wabs(i)=wabs(i)+gabs
        wdd(i)=wdd(i)+gdd
        wqel(i)=wqel(i)+gqel
        wcoh(i)=wcoh(i)+gcoh
       enddo
      enddo

      gabs=0.
      gdd=0.
      gqel=0.
      gcoh=0.
      do i=1,28
       wabs(i)=wabs(i)/niter
       wdd(i)=wdd(i)/niter
       wqel(i)=wqel(i)/niter
       wcoh(i)=wcoh(i)/niter
       wprod(i)=wabs(i)+wdd(i)
       gabs=gabs+ai(i)*wabs(i)
       gdd=gdd+ai(i)*wdd(i)
       gqel=gqel+ai(i)*wqel(i)
       gcoh=gcoh+ai(i)*wcoh(i)
      enddo
      gprod=gabs+gdd
      gtot=gprod+gqel+gcoh
      return
      end

c-------------------------------------------------------------------------------
      subroutine qggcr(b,gabs,gdd,gqel,gcoh,xa,xb,ia)
c-------------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      parameter(iapmax=208)
      dimension xa(iapmax,3),xb(iapmax,3),vabs(2)

      gabs=1.
      gdd=1.
      gqel=1.
      gcoh=1.
      do n=1,ia
       call qgv(xa(n,1)+b,xa(n,2),xb,vin,vdd,vabs)
       gabs=gabs*(vdd-vin+1.d0)          !prod_n^A [sum_i c_i exp(-2chi_i(n))]
       gdd=gdd*(1.-vin)                  !prod_n^A [sum_i c_i exp(-chi_i(n))]^2
       gqel=gqel*(2.d0*dsqrt(1.d0-vin)-1.d0)
                                       !prod_n^A [sum_i c_i exp(-chi_i(n)) - 1]
       gcoh=gcoh*dsqrt(1.d0-vin)
      enddo
      gcoh=1.-2.*gcoh+gqel
      gqel=gdd-gqel
      gdd=gabs-gdd
      gabs=1.-gabs
      return
      end

c-------------------------------------------------------------------------------
      double precision function qgsect(e0n,icz,iap,iat)
c-------------------------------------------------------------------------------
c qgsect - hadron-nucleus (hadron-nucleus) particle production cross section
c e0n - lab. energy per projectile nucleon (hadron),
c icz - hadron class,
c iap - projectile mass number (1=<iap<=iapmax),
c iat - target mass number     (1=<iat<=iapmax)
c-------------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      dimension wk(3),wa(3),wb(3)
      common /qgarr47/ gsect(10,5,6)
      common /qgarr48/ qgsasect(10,6,6)
      common /qgarr43/ moniou
      common /qgdebug/    debug

      if(debug.ge.3)write (moniou,201)e0n,icz,iap,iat
      qgsect=0.d0
      ye=dlog10(e0n)
      if(ye.lt.1.d0)ye=1.d0
      je=int(ye)
      if(je.gt.8)je=8

      wk(2)=ye-je
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)

      yb=iat
      yb=dlog(yb)/1.38629d0+1.d0
      jb=min(int(yb),2)
      wb(2)=yb-jb
      wb(3)=wb(2)*(wb(2)-1.d0)*.5d0
      wb(1)=1.d0-wb(2)+wb(3)
      wb(2)=wb(2)-2.d0*wb(3)

      if(iap.eq.1)then
       if(iat.eq.14)then
        do i=1,3
         qgsect=qgsect+gsect(je+i-1,icz,5)*wk(i)
        enddo
       elseif(iat.eq.40)then
        do i=1,3
         qgsect=qgsect+gsect(je+i-1,icz,6)*wk(i)
        enddo
       else
        do i=1,3
        do l=1,3
         qgsect=qgsect+gsect(je+i-1,icz,jb+l-1)*wk(i)*wb(l)
        enddo
        enddo
       endif
      else
       ya=iap
       ya=dlog(ya/2.d0)/.69315d0+1.d0
       ja=min(int(ya),4)
       wa(2)=ya-ja
       wa(3)=wa(2)*(wa(2)-1.d0)*.5d0
       wa(1)=1.d0-wa(2)+wa(3)
       wa(2)=wa(2)-2.d0*wa(3)
       if(iat.eq.14)then
        do i=1,3
        do m=1,3
         qgsect=qgsect+qgsasect(je+i-1,ja+m-1,5)*wk(i)*wa(m)
        enddo
        enddo
       elseif(iat.eq.40)then
        do i=1,3
        do m=1,3
         qgsect=qgsect+qgsasect(je+i-1,ja+m-1,6)*wk(i)*wa(m)
        enddo
        enddo
       else
        do i=1,3
        do m=1,3
        do l=1,3
         qgsect=qgsect+qgsasect(je+i-1,ja+m-1,jb+l-1)*wk(i)*wa(m)*wb(l)
        enddo
        enddo
        enddo
       endif
      endif
      qgsect=exp(qgsect)
      if(debug.ge.4)write (moniou,202)

201   format(2x,'qgsect - nucleus-nucleus production cross section'
     */4x,'lab. energy per nucleon - ',e10.3,2x,'hadron class - ',i2
     */4x,'proj. mass N - ',i3,2x,'targ. mass N - ',i3)
202   format(2x,'qgsect=',e10.3)
      return
      end

c=============================================================================
      subroutine qgreg(ep0,ic)
c-----------------------------------------------------------------------
c qgreg - registration of produced hadron
c ep0 - 4-momentum,
c ic  - hadron type
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      parameter(nptmax=95000)
      dimension ep(4),ep0(4),ep1(4),ep2(4),ep3(4)
      common /qgarr4/  ey0(3)
      common /qgarr10/ am0,amn,amk,amc,amlamc,amlam,ameta,ammu
      common /qgarr11/ b10
      common /qgarr12/ nsh
      common /qgarr14/ esp(4,nptmax),ich(nptmax)
      common /qgarr21/ dmmin(3),wex(3),dmres(3),wdres(3)
      common /qgarr43/ moniou
      common /qgdebug/  debug
      external qgran

      if(debug.ge.3)write (moniou,201)ic,ep0,nsh
      nsh=nsh+1
      
      nstprev = nsh

      if(nsh.gt.nptmax)stop'increase nptmax!!!'
      iab=iabs(ic)
      do i=1,4
       ep(i)=ep0(i)
      enddo
      
c       call qgtran(ep,ey0,1)
      
      if(iab.eq.7.or.iab.eq.8)then         !delta++(-)
       call qgdec2(ep,ep1,ep2,dmmin(2)**2,amn**2,am0**2)
       ich(nsh)=ic-5*ic/iab
       do i=1,4
        esp(i,nsh)=ep1(i)
        ep(i)=ep2(i)
       enddo
       nsh=nsh+1
       ich(nsh)=15*ic/iab-2*ic

ctp      elseif(iab.eq.-10)then                   !rho0 -> pi+ + pi-
ctp       call qgdec2(ep,ep1,ep2,dmmin(1)**2,am0**2,am0**2)
ctp       ich(nsh)=2*int(.5d0+qgran(b10))-1
ctp       do i=1,4
ctp        esp(i,nsh)=ep1(i)
ctp        ep(i)=ep2(i)
ctp       enddo
ctp       nsh=nsh+1
ctp       ich(nsh)=-ich(nsh-1)

      elseif(iab.eq.11)then                  !pi* -> rho + pi
       am2=qgnrm(ep)
       call qgdec2(ep,ep1,ep2,am2,dmmin(1)**2,am0**2)
ctp       call qgdec2(ep1,ep3,ep,dmmin(1)**2,am0**2,am0**2)
       if(qgran(b10).lt..5d0)then  !rho0 + pi+/-
        ich(nsh)=-10
        ich(nsh+1)=ic/iab
ctp        ich(nsh+1)=2*int(.5d0+qgran(b10))-1
ctp        ich(nsh+2)=-ich(nsh+1)
        do i=1,4
          esp(i,nsh)=ep1(i)
          ep(i)=ep2(i)
        enddo
        nsh=nsh+1
       else      !rho+/- + pi0 -> pi+/- + 2 pi0
        call qgdec2(ep1,ep3,ep,dmmin(1)**2,am0**2,am0**2)
        ich(nsh)=0
        ich(nsh+1)=ic/iab
        ich(nsh+2)=0
        do i=1,4
          esp(i,nsh)=ep2(i)
          esp(i,nsh+1)=ep3(i)
        enddo
        nsh=nsh+2
       endif
ctp       do i=1,4
ctp        esp(i,nsh)=ep2(i)
ctp        esp(i,nsh+1)=ep3(i)
ctp       enddo
ctp       nsh=nsh+2

      elseif(iab.eq.12.or.iab.eq.13)then       !N*
       am2=qgnrm(ep)
       if(6.d0*qgran(b10).lt.1.d0)then         !delta + pi
        call qgdec2(ep,ep1,ep2,am2,dmmin(2)**2,am0**2)
        call qgdec2(ep1,ep3,ep,dmmin(2)**2,amn**2,am0**2)
        ich(nsh)=2*ic-25*ic/iab
        ich(nsh+1)=ic-10*ic/iab
        ich(nsh+2)=-ich(nsh)
        do i=1,4
         esp(i,nsh)=ep2(i)
         esp(i,nsh+1)=ep3(i)
        enddo
        nsh=nsh+2
       else                                    !N + pi
        call qgdec2(ep,ep1,ep2,am2,amn**2,am0**2)
        do i=1,4
         esp(i,nsh)=ep1(i)
         ep(i)=ep2(i)
        enddo
        if(qgran(b10).lt..4d0)then
         ich(nsh)=ic-10*ic/iab
         ich(nsh+1)=0
        else
         ich(nsh)=15*ic/iab-ic
         ich(nsh+1)=25*ic/iab-2*ic
        endif
        nsh=nsh+1
       endif

      elseif(iab.eq.14.or.iab.eq.15)then       !K1
       am2=qgnrm(ep)
       if(dsqrt(am2).gt.dmmin(1)+amk)then      !rho + K
        call qgdec2(ep,ep1,ep2,am2,dmmin(1)**2,amk**2)
ctp        call qgdec2(ep1,ep3,ep,dmmin(1)**2,am0**2,am0**2)
        if(3.d0*qgran(b10).lt.1.d0)then  !rho0
         ich(nsh)=ic-10*ic/iab
         ich(nsh+1)=-10
c         ich(nsh+1)=2*int(.5d0+qgran(b10))-1
c         ich(nsh+2)=-ich(nsh+1)
         do i=1,4
           esp(i,nsh)=ep2(i)
           ep(i)=ep1(i)
         enddo
         nsh=nsh+1
        else                             !rho+/-
         call qgdec2(ep1,ep3,ep,dmmin(1)**2,am0**2,am0**2)
         ich(nsh)=19*ic/iab-ic
         ich(nsh+1)=29*ic/iab-2*ic
         ich(nsh+2)=0
         do i=1,4
          esp(i,nsh)=ep2(i)
          esp(i,nsh+1)=ep3(i)
         enddo
         nsh=nsh+2
        endif
       else                                    !K* + pi
        call qgdec2(ep,ep1,ep2,am2,dmmin(3)**2,am0**2)
        call qgdec2(ep1,ep3,ep,dmmin(3)**2,amk**2,am0**2)
        if(3.d0*qgran(b10).lt.1.d0)then
         ich(nsh)=0
         if(3.d0*qgran(b10).lt.1.d0)then
          ich(nsh+1)=ic-10*ic/iab
          ich(nsh+2)=0
         else
          ich(nsh+1)=19*ic/iab-ic
          ich(nsh+2)=29*ic/iab-2*ic
         endif
        else
         ich(nsh)=29*ic/iab-2*ic
         if(3.d0*qgran(b10).lt.1.d0)then
          ich(nsh+1)=19*ic/iab-ic
          ich(nsh+2)=0
         else
          ich(nsh+1)=ic-10*ic/iab
          ich(nsh+2)=2*ic-29*ic/iab
         endif
        endif
        do i=1,4
         esp(i,nsh)=ep2(i)
         esp(i,nsh+1)=ep3(i)
        enddo
        nsh=nsh+2
       endif
ctp       do i=1,4
ctp        esp(i,nsh)=ep2(i)
ctp        esp(i,nsh+1)=ep3(i)
ctp       enddo
ctp       nsh=nsh+2

      elseif(iab.eq.5)then                     !K0,K0~
       ich(nsh)=10*int(.5d0+qgran(b10))-5

c      elseif(iab.eq.6)then                !lambda decay (switch on in CONEX!)
c       ic2=-ic/iab*int(.64d0+qgran(b10))
c       ic1=3*ic/iab+ic2
c       call qgdec2(ep,ep1,ep2,amlam**2,amn**2,am0**2)
c       do i=1,4
c        esp(i,nsh)=ep1(i)
c        ep(i)=ep2(i)
c       enddo
c       ich(nsh)=ic1
c       ich(nsh+1)=ic2
c       nsh=nsh+1

      else
       ich(nsh)=ic
      endif

      do i=1,4
       esp(i,nsh)=ep(i)
      enddo

      do n=nstprev,nsh
        do i=1,4
          ep(i)=esp(i,n)
        enddo
        call qgtran(ep,ey0,1)
        do i=1,4
          esp(i,n)=ep(i)
        enddo
      enddo

      if(debug.ge.4)write (moniou,202)

201   format(2x,'qgreg: ic=',i2,2x,'c.m. 4-momentum:',2x,4(e10.3,1x)/
     * 4x,'number of particles in the storage: ',i5)
202   format(2x,'qgreg - end')
      return
      end

c-----------------------------------------------------------------------------
      subroutine qgdec2(ep,ep1,ep2,ww,a,b)
c two particle decay
      implicit double precision (a-h,o-z)
      integer debug
      dimension ep(4),ep1(4),ep2(4),ey(3)
      common /qgarr11/ b10
      common /qgarr43/ moniou
      common /qgdebug/  debug
      EXTERNAL qgran

      if(debug.ge.2)write (moniou,201)ep,ww,a,b
201   format(2x,'qgdec2: 4-momentum:',2x,4(e10.3,1x)
     */4x,'ww=',e10.3,2x,'a=',e10.3,2x,'b=',e10.3)

      pl=qglam(ww,a,b)
      ep1(1)=dsqrt(pl+a)
      ep2(1)=dsqrt(pl+b)
      pl=dsqrt(pl)
      cosz=2.d0*qgran(b10)-1.d0
      pt=pl*dsqrt(1.d0-cosz**2)
      ep1(2)=pl*cosz
      call qgcs(c,s)
      ep1(3)=pt*c
      ep1(4)=pt*s
      do i=2,4
       ep2(i)=-ep1(i)
      enddo
      call qgdeft(ww,ep,ey)
      call qgtran(ep1,ey,1)
      call qgtran(ep2,ey,1)
      if(debug.ge.3)write (moniou,203)
203   format(2x,'qgdec2 - end')
      return
      end

c------------------------------------------------------------------------
      double precision function qggrv(x,qqs,icq,iq)
c------------------------------------------------------------------------
c qggrv - GRV structure functions
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr25/ ahv(3)

      qggrv=0.
      if(x.gt..99999d0.and.(qqs.ne.qt0.or.iq.ne.1.and.iq.ne.2))return

      if(icq.eq.2)then
       sq=dlog(dlog(qqs/.232d0**2)/dlog(.23d0/.232d0**2))
       if(iq.eq.0)then                                 !gluon
        alg=.524d0
        betg=1.088d0
        aag=1.742d0-.93d0*sq
        bbg=-.399d0*sq**2
        ag=7.486d0-2.185d0*sq
        bg=16.69d0-22.74d0*sq+5.779d0*sq*sq
        cg=-25.59d0+29.71d0*sq-7.296d0*sq*sq
        dg=2.792d0+2.215d0*sq+.422d0*sq*sq-.104d0*sq*sq*sq
        eg=.807d0+2.005d0*sq
        eeg=3.841d0+.361d0*sq
        qggrv=(1.d0-x)**dg*(x**aag*(ag+bg*x+cg*x**2)*log(1.d0/x)**bbg
     *  +sq**alg*exp(-eg+sqrt(eeg*sq**betg*log(1.d0/x))))
       elseif(iq.eq.1.or.iq.eq.2)then                  !u_v or d_v
        aau=.59d0-.024d0*sq
        bbu=.131d0+.063d0*sq
        auu=2.284d0+.802d0*sq+.055d0*sq*sq
        au=-.449d0-.138d0*sq-.076d0*sq*sq
        bu=.213d0+2.669d0*sq-.728d0*sq*sq
        cu=8.854d0-9.135d0*sq+1.979d0*sq*sq
        du=2.997d0+.753d0*sq-.076d0*sq*sq
        uv=auu*x**aau*(1.d0+au*x**bbu+bu*x+cu*x**1.5d0)
        if(qqs.ne.qt0)uv=uv*(1.d0-x)**du

        aad=.376d0
        bbd=.486d0+.062d0*sq
        add=.371d0+.083d0*sq+.039d0*sq*sq
        ad=-.509d0+3.31d0*sq-1.248d0*sq*sq
        bd=12.41d0-10.52d0*sq+2.267d0*sq*sq
        ccd=6.373d0-6.208d0*sq+1.418d0*sq*sq
        dd=3.691d0+.799d0*sq-.071d0*sq*sq
        dv=add*x**aad*(1.d0+ad*x**bbd+bd*x+ccd*x**1.5d0)
        if(qqs.ne.qt0)then
         dv=dv*(1.d0-x)**dd
        elseif(x.gt..99999d0)then
         dv=0.d0
        else
         dv=dv*(1.d0-x)**(dd-ahv(2))
        endif
        if(iq.eq.1)then                              !u_v
         qggrv=uv
        elseif(iq.eq.2)then                          !d_v
         qggrv=dv
        endif

       elseif(iq.eq.-3)then                           !s_sea
        als=.914
        bets=.577
        aas=1.798-.596*sq
        as=-5.548+3.669*sqrt(sq)-.616*sq
        bs=18.92-16.73*sqrt(sq)+5.168*sq
        ds=6.379-.35*sq+.142*sq*sq
        es=3.981+1.638*sq
        ees=6.402
        qggrv=(1.-x)**ds*sq**als/log(1./x)**aas*(1.+as*sqrt(x)
     *  +bs*x)*exp(-es+sqrt(ees*sq**bets*log(1./x)))
       elseif(iabs(iq).lt.3)then                      !u_sea or d_sea
        aadel=.409-.005*sq
        bbdel=.799+.071*sq
        addel=.082+.014*sq+.008*sq*sq
        adel=-38.07+36.13*sq-.656*sq*sq
        bdel=90.31-74.15*sq+7.645*sq*sq
        ccdel=0.
        ddel=7.486+1.217*sq-.159*sq*sq
        delv=addel*x**aadel*(1.-x)**ddel
     *  *(1.+adel*x**bbdel+bdel*x+ccdel*x**1.5)

        alud=1.451
        betud=.271
        aaud=.41-.232*sq
        bbud=.534-.457*sq
        aud=.89-.14*sq
        bud=-.981
        cud=.32+.683*sq
        dud=4.752+1.164*sq+.286*sq*sq
        eud=4.119+1.713*sq
        eeud=.682+2.978*sq
        udsea=(1.-x)**dud*(x**aaud*(aud+bud*x+cud*x**2)
     *  *log(1./x)**bbud+sq**alud*exp(-eud+sqrt(eeud*sq**betud
     *  *log(1./x))))

        if(iq.eq.-1)then                           !u_sea
         qggrv=(udsea-delv)/2.
        elseif(iq.eq.-2)then                       !d_sea
         qggrv=(udsea+delv)/2.
        endif
       else
        qggrv=0.
       endif

      elseif(icq.eq.1.or.icq.eq.3)then
       sq=dlog(dlog(qqs/.204d0**2)/dlog(.26d0/.204d0**2))
       if(iq.eq.1.or.iq.eq.2)then
        aapi=.517-.02*sq
        api=-.037-.578*sq
        bpi=.241+.251*sq
        dpi=.383+.624*sq
        anorm=1.212+.498*sq+.009*sq**2
        qggrv=.5*anorm*x**aapi*(1.+api*sqrt(x)+bpi*x)
        if(qqs.ne.qt0)qggrv=qggrv*(1.d0-x)**dpi
       elseif(iq.eq.0)then
          alfpi=.504
          betpi=.226
          aapi=2.251-1.339*sqrt(sq)
          api=2.668-1.265*sq+.156*sq**2
          bbpi=0.
          bpi=-1.839+.386*sq
          cpi=-1.014+.92*sq-.101*sq**2
          dpi=-.077+1.466*sq
          epi=1.245+1.833*sq
          eppi=.51+3.844*sq
          qggrv=(1.-x)**dpi*(x**aapi*(api+bpi*sqrt(x)+cpi*x)*
     *    log(1./x)**bbpi+sq**alfpi*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x))))
        elseif(iq.eq.-3)then
          alfpi=.823
          betpi=.65
          aapi=1.036-.709*sq
          api=-1.245+.713*sq
          bpi=5.58-1.281*sq
          dpi=2.746-.191*sq
          epi=5.101+1.294*sq
          eppi=4.854-.437*sq
          qggrv=sq**alfpi/log(1./x)**aapi*(1.-x)**dpi*
     *    (1.+api*sqrt(x)+bpi*x)*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x)))
        elseif(iabs(iq).lt.3)then
          alfpi=1.147
          betpi=1.241
          aapi=.309-.134*sqrt(sq)
          api=.219-.054*sq
          bbpi=.893-.264*sqrt(sq)
          bpi=-.593+.24*sq
          cpi=1.1-.452*sq
          dpi=3.526+.491*sq
          epi=4.521+1.583*sq
          eppi=3.102
          qggrv=(1.-x)**dpi*(x**aapi*(api+bpi*sqrt(x)+cpi*x)*
     *    log(1./x)**bbpi+sq**alfpi*
     *    exp(-epi+sqrt(eppi*sq**betpi*log(1./x))))
        else
          qggrv=0.
        endif
      else
       qggrv=0.
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgev(q1,qj,qq,xx,j,l)
c------------------------------------------------------------------------
c qgev - PDF evolution
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr51/ epsxmn
      common /arr3/   x1(7),a1(7)

      qgev=0.d0
      zmax=1.d0-epsxmn
      zmin=xx/zmax
      if(zmin.ge.zmax)return

      if(qj.eq.qq)then
       do i1=1,7
       do m1=1,2
        qi=q1*(qq/q1)**(.5d0+x1(i1)*(m1-1.5d0))

        fz1=0.d0
        fz2=0.d0
        fz3=0.d0
        zmin1=max(.2d0,zmin)
        zmax1=min(.2d0,zmax)
        zmax1=min(5.d0*xx,zmax1)
        zmax2=min(zmin1,zmax)
        zmin2=max(zmax1,zmin)

        if(zmax1.gt.zmin)then
         do i=1,7
         do m=1,2
          z=xx+(zmin-xx)*((zmax1-xx)/(zmin-xx))**(.5d0+(m-1.5d0)*x1(i))
          do k=1,2
           if(j.ne.3.or.k.ne.1)then
            fz1=fz1+a1(i)*qgevi(q1,qi,xx/z,j,k)*qgfap(z,k,l)*(1.d0-xx/z)
           endif
          enddo
         enddo
         enddo
         fz1=fz1*dlog((zmax1-xx)/(zmin-xx))
        endif
        if(zmin1.lt.zmax)then
         do i=1,7
         do m=1,2
          z=1.d0-(1.d0-zmax)*((1.d0-zmin1)/(1.d0-zmax))
     *    **(.5d0+x1(i)*(m-1.5d0))
          do k=1,2
           if(j.ne.3.or.k.ne.1)then
            fz2=fz2+a1(i)*qgevi(q1,qi,xx/z,j,k)*qgfap(z,k,l)
     *      *(1.d0/z-1.d0)
           endif
          enddo
         enddo
         enddo
         fz2=fz2*dlog((1.d0-zmin1)/(1.d0-zmax))
        endif
        if(zmax2.gt.zmin2)then
         do i=1,7
         do m=1,2
          z=zmin2*(zmax2/zmin2)**(.5d0+x1(i)*(m-1.5d0))
          do k=1,2
           if(j.ne.3.or.k.ne.1)then
            fz3=fz3+a1(i)*qgevi(q1,qi,xx/z,j,k)*qgfap(z,k,l)
           endif
          enddo
         enddo
         enddo
         fz3=fz3*dlog(zmax2/zmin2)
        endif
        qgev=qgev+a1(i1)*(fz1+fz2+fz3)/qgsudx(qi,l)*qgalf(qi/alm)
       enddo
       enddo
       qgev=qgev*dlog(qq/q1)/4.d0*qgsudx(qq,l)

      else
       fz1=0.d0
       fz2=0.d0
       fz3=0.d0
       zmin1=max(.2d0,zmin)
       zmax1=min(.2d0,zmax)
       zmax1=min(5.d0*xx,zmax1)
       zmax2=min(zmin1,zmax)
       zmin2=max(zmax1,zmin)

       if(zmax1.gt.zmin)then
        do i=1,7
        do m=1,2
         z=xx+(zmin-xx)*((zmax1-xx)/(zmin-xx))**(.5d0+(m-1.5d0)*x1(i))
         do k=1,2
          if(j.ne.3)then
           fz1=fz1+a1(i)*qgevi(q1,qj,xx/z,j,k)*qgevi(qj,qq,z,k,l)
     *     *(1.d0-xx/z)
          elseif(k.ne.1)then
           fz1=fz1+a1(i)*qgevi(q1,qj,xx/z,3,2)*qgevi(qj,qq,z,3,2)
     *     *(1.d0-xx/z)
          endif
         enddo
        enddo
        enddo
        fz1=fz1*dlog((zmax1-xx)/(zmin-xx))
       endif
       if(zmin1.lt.zmax)then
        do i=1,7
        do m=1,2
         z=1.d0-(1.d0-zmax)*((1.d0-zmin1)/(1.d0-zmax))
     *   **(.5d0+x1(i)*(m-1.5d0))
         do k=1,2
          if(j.ne.3)then
           fz2=fz2+a1(i)*qgevi(q1,qj,xx/z,j,k)*qgevi(qj,qq,z,k,l)
     *     *(1.d0/z-1.d0)
          elseif(k.ne.1)then
           fz2=fz2+a1(i)*qgevi(q1,qj,xx/z,3,2)*qgevi(qj,qq,z,3,2)
     *     *(1.d0/z-1.d0)
          endif
         enddo
        enddo
        enddo
        fz2=fz2*dlog((1.d0-zmin1)/(1.d0-zmax))
       endif
       if(zmax2.gt.zmin2)then
        do i=1,7
        do m=1,2
         z=zmin2*(zmax2/zmin2)**(.5d0+x1(i)*(m-1.5d0))
         do k=1,2
          if(j.ne.3)then
           fz2=fz2+a1(i)*qgevi(q1,qj,xx/z,j,k)*qgevi(qj,qq,z,k,l)
          elseif(k.ne.1)then
           fz2=fz2+a1(i)*qgevi(q1,qj,xx/z,3,2)*qgevi(qj,qq,z,3,2)
          endif
         enddo
        enddo
        enddo
        fz3=fz3*dlog(zmax2/zmin2)
       endif
       qgev=(fz1+fz2+fz3)/2.d0
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgevi(q1,qq,xx,m,l)
c------------------------------------------------------------------------
c qgevi - PDF evolution - interpolation
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      dimension wi(3),wj(3),wk(3)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr20/ spmax
      common /qgarr51/ epsxmn
      common /qgarr52/ evk(40,40,100,3,2)

      qgevi=0.d0
      if(q1.ge..9999d0*spmax)goto 1

      if(xx.le..1d0)then
       yx=37.d0-dlog(.1d0/xx)/dlog(.1d0*spmax)*36.d0
       k=max(1,int(yx))
       k=min(k,35)
      elseif(xx.le..9d0)then
       yx=(xx-.1d0)*40.d0+37.d0
       k=max(37,int(yx))
       k=min(k,67)
      else
       yx=dlog(10.d0*(1.d0-xx))/log(10.d0*epsxmn)*31.d0+69.d0
       k=max(69,int(yx))
       k=min(k,98)
      endif
      wk(2)=yx-k
      wk(3)=wk(2)*(wk(2)-1.d0)*.5d0
      wk(1)=1.d0-wk(2)+wk(3)
      wk(2)=wk(2)-2.d0*wk(3)

      qli=log(q1)/dlog(spmax)*39.d0+1.d0
      qlj=log(qq/q1)/dlog(spmax/q1)*39.d0+1.d0
      i=max(1,int(1.0001d0*qli))
      i=min(i,38)
      wi(2)=qli-i
      wi(3)=wi(2)*(wi(2)-1.d0)*.5d0
      wi(1)=1.d0-wi(2)+wi(3)
      wi(2)=wi(2)-2.d0*wi(3)

      j=max(1,int(1.0001d0*qlj))
      j=min(j,38)
      wj(2)=qlj-j
      wj(3)=wj(2)*(wj(2)-1.d0)*.5d0
      wj(1)=1.d0-wj(2)+wj(3)
      wj(2)=wj(2)-2.d0*wj(3)

      do i1=1,3
      do j1=1,3
      do k1=1,3
       k2=k+k1-1
       qgevi=qgevi+evk(i+i1-1,j+j1-1,k2,m,l)*wi(i1)*wj(j1)*wk(k1)
      enddo
      enddo
      enddo
1     qgevi=exp(qgevi)*qgfap(xx,m,l)
      if(m.eq.1.and.l.eq.1.or.m.ne.1.and.l.ne.1)then
       qgevi=qgevi/4.5d0/qgsudx(q1,m)*qgsudx(qq,m)
     * *dlog(dlog(qq/alm)/dlog(q1/alm))
      else
       qgevi=qgevi*.3d0/(dlog(epsxmn)+.75d0)
     * *(qgsudx(qq,1)/qgsudx(q1,1)-qgsudx(qq,2)/qgsudx(q1,2))
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgpdf(xx,qq,icz,jj)
c-----------------------------------------------------------------------
c qgpdf - parton distribution function for proton
c qq  - virtuality scale,
c xx  - light cone x,
c icz - hadron type,
c jj  - parton type (0 - gluon, 1 - u_v, 2 - d_v, -1 - q_sea)
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr25/ ahv(3)
      common /qgarr51/ epsxmn
      common /arr3/   x1(7),a1(7)

      if(jj.eq.0)then
       qgpdf=qggpdf(xx,icz)
      elseif(jj.eq.1.or.jj.eq.2)then
       qgpdf=qggrv(xx,qt0,icz,jj)*(1.d0-xx)**ahv(icz)
      else
       qgpdf=qgspdf(xx,icz)
      endif
      qgpdf=qgpdf*qgsudx(qq,iabs(jj)+1)/qgsudx(qt0,iabs(jj)+1)

      xmin=xx/(1.d0-epsxmn)
      if(xmin.lt.1.d0.and.qq.gt.qt0)then
       dpd1=0.d0
       dpd2=0.d0
       xm=max(xmin,.3d0)
       do i=1,7         !numerical integration over zx
       do m=1,2
        zx=1.d0-(1.d0-xm)*(.5d0+(m-1.5d0)*x1(i))**.25d0
        z=xx/zx

        gl=qggpdf(zx,icz)
        uv=qggrv(zx,qt0,icz,1)*(1.d0-zx)**ahv(icz)
        dv=qggrv(zx,qt0,icz,2)*(1.d0-zx)**ahv(icz)
        sea=qgspdf(zx,icz)
        if(jj.eq.0)then
         fz=qgevi(qt0,qq,z,1,1)*gl+qgevi(qt0,qq,z,2,1)*(uv+dv+sea)
        elseif(jj.eq.1)then
         fz=qgevi(qt0,qq,z,3,2)*uv
        elseif(jj.eq.2)then
         fz=qgevi(qt0,qq,z,3,2)*dv
        else
         akns=qgevi(qt0,qq,z,3,2)              !nonsinglet contribution
         aks=(qgevi(qt0,qq,z,2,2)-akns)        !singlet contribution
         fz=(qgevi(qt0,qq,z,1,2)*gl+aks*(uv+dv+sea)+akns*sea)
        endif
        dpd1=dpd1+a1(i)*fz/zx**2/(1.d0-zx)**3
       enddo
       enddo
       dpd1=dpd1*(1.d0-xm)**4/8.d0*xx

       if(xm.gt.xmin)then
        do i=1,7         !numerical integration
        do m=1,2
         zx=xx+(xm-xx)*((xmin-xx)/(xm-xx))**(.5d0-(m-1.5d0)*x1(i))
         z=xx/zx

         gl=qggpdf(zx,icz)
         uv=qggrv(zx,qt0,icz,1)*(1.d0-zx)**ahv(icz)
         dv=qggrv(zx,qt0,icz,2)*(1.d0-zx)**ahv(icz)
         sea=qgspdf(zx,icz)
         if(jj.eq.0)then
          fz=qgevi(qt0,qq,z,1,1)*gl+qgevi(qt0,qq,z,2,1)*(uv+dv+sea)
         elseif(jj.eq.1)then
          fz=qgevi(qt0,qq,z,3,2)*uv
         elseif(jj.eq.2)then
          fz=qgevi(qt0,qq,z,3,2)*dv
         else
          akns=qgevi(qt0,qq,z,3,2)              !nonsinglet contribution
          aks=(qgevi(qt0,qq,z,2,2)-akns)        !singlet contribution
          fz=(qgevi(qt0,qq,z,1,2)*gl+aks*(uv+dv+sea)+akns*sea)
         endif
         dpd2=dpd2+a1(i)*fz*(1.d0-xx/zx)/zx
        enddo
        enddo
        dpd2=dpd2*dlog((xm-xx)/(xmin-xx))*.5d0*xx
       endif
       qgpdf=qgpdf+dpd2+dpd1
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgpdfd(xx,xpomr,qq,icz)
c-----------------------------------------------------------------------
c qgpdfd - diffractive sf f2_d^(3)
c qq    - virtuality scale,
c xx    - parton light cone x,
c xpomr - pomeron lc x,
c icz   - hadron type
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /qgarr6/  pi,bm,amws
      common /qgarr15/ fp(3),rq(2,3),cd(2,3),gsoft(3)
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr25/ ahv(3)
      common /qgarr51/ epsxmn
      common /arr3/   x1(7),a1(7)

      qgpdfd=(qgdpdf(xx,xpomr,icz,1)+qgdpdf(xx,xpomr,icz,2))
     **qgsudx(qq,2)/qgsudx(qt0,2)
      xmin=xx/(1.d0-epsxmn)
      if(xmin.lt.xpomr.and.qq.gt.qt0)then
       dpd1=0.d0
       dpd2=0.d0
       xm=max(xmin,.3d0)
       if(xm.lt.xpomr)then
        do i=1,7         !numerical integration over zx
        do m=1,2
         zx=1.d0-(1.d0-xm)*(1.d0-(.5d0+(m-1.5d0)*x1(i))
     *   *(1.d0-((1.d0-xpomr)/(1.d0-xm))**4))**.25d0
         z=xx/zx

         glu=(qgdgdf(zx,xpomr,icz,1)+qgdgdf(zx,xpomr,icz,2))/4.5d0
         sea=qgdpdf(zx,xpomr,icz,1)+qgdpdf(zx,xpomr,icz,2)
         fz=qgevi(qt0,qq,z,1,2)*glu+qgevi(qt0,qq,z,2,2)*sea
         dpd1=dpd1+a1(i)*fz/zx**2/(1.d0-zx)**3
        enddo
        enddo
        dpd1=dpd1*((1.d0-xm)**4-(1.d0-xpomr)**4)/8.d0*xx
       endif

       xm=min(xm,xpomr)
       if(xm.gt.xmin)then
        do i=1,7         !numerical integration
        do m=1,2
         zx=xx+(xm-xx)*((xmin-xx)/(xm-xx))**(.5d0-(m-1.5d0)*x1(i))
         z=xx/zx

         glu=(qgdgdf(zx,xpomr,icz,1)+qgdgdf(zx,xpomr,icz,2))/4.5d0
         sea=qgdpdf(zx,xpomr,icz,1)+qgdpdf(zx,xpomr,icz,2)
         fz=qgevi(qt0,qq,z,1,2)*glu+qgevi(qt0,qq,z,2,2)*sea
         dpd2=dpd2+a1(i)*fz*(1.d0-xx/zx)/zx
        enddo
        enddo
        dpd2=dpd2*dlog((xm-xx)/(xmin-xx))*.5d0*xx
       endif
       qgpdfd=qgpdfd+dpd2+dpd1
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgf2c(xx,qq,icz)
c-----------------------------------------------------------------------
c qgf2c - c-quark contribution to f2
c qq  - virtuality scale,
c xx  - light cone x,
c icz - hadron type,
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /arr3/   x1(7),a1(7)

      qgf2c=0.d0
      qcmass=1.3d0
      s2min=4.*qcmass**2+qq
      xmin=s2min*xx/qq

      if(xmin.lt.1.d0)then
       do i=1,7          !numerical integration over z1
       do m=1,2
        z1=xmin**(.5d0+x1(i)*(m-1.5d0))
        sdc=qgdbor(qq,xx/z1,qcmass**2)
        glu=qgpdf(z1,s2min-qq,icz,0)
        qgf2c=qgf2c+a1(i)*sdc*glu
       enddo
       enddo
       qgf2c=-qgf2c*dlog(xmin)*.5d0
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgf2cd(xx,xpomr,qq,icz)
c-----------------------------------------------------------------------
c qgf2cd - c-quark contribution to diffractive sf
c qq  - virtuality scale,
c xx  - light cone x,
c icz - hadron type,
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /arr3/   x1(7),a1(7)

      qgf2cd=0.d0
      qcmass=1.3d0
      s2min=4.*qcmass**2+qq
      xmin=s2min*xx/qq

      if(xmin.lt.xpomr)then
       do i=1,7          !numerical integration over z1
       do m=1,2
        z1=xpomr*(xmin/xpomr)**(.5d0+x1(i)*(m-1.5d0))
        sdc=qgdbor(qq,xx/z1,qcmass**2)
        glu=qgdgdf(z1,xpomr,icz,1)+qgdgdf(z1,xpomr,icz,2)
        qgf2cd=qgf2cd+a1(i)*sdc*glu
       enddo
       enddo
       qgf2cd=qgf2cd*dlog(xpomr/xmin)*.5d0
      endif
      return
      end

c------------------------------------------------------------------------
      double precision function qgdbor(qq,zz,q2mass)
c-----------------------------------------------------------------------
c qgdbor - DIS c-quark cross-section
c qq      - photon virtuality
c s=2(pq) - s_true + qq,
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common /qgarr18/ alm,qt0,qtf,betp,dgqq

      qgdbor=0.
      qtq=4.d0*q2mass*zz/qq/(1.d0-zz)
      if(qtq.ge.1.d0)return
      bet=dsqrt(1.d0-qtq)

      qgdbor=qgalf(4.d0*q2mass/alm)/2.25d0*zz
     **(dlog((1.d0+bet)/(1.d0-bet))*(1.d0-2.d0*zz*(1.d0-zz)
     *-8.d0*(zz*q2mass/qq)**2+4.d0*zz*(1.d0-3.d0*zz)*q2mass/qq)
     *+bet*(-1.d0-4.d0*zz*(1.d0-zz)*q2mass/qq+8.d0*zz*(1.d0-zz)))
      return
      end

c=============================================================================
      double precision function qgjeto(qi,qj,s,iq1,iq2)
c-----------------------------------------------------------------------------
c qgjeto - hard 2->2 parton scattering born cross-section
c s is the c.m. energy square for the scattering process,
c iq1 - parton type at current end of the ladder (0 - g, 1,2 etc. - q)
c iq2 - parton type at opposite end of the ladder (0 - g, 1,2 etc. - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/    debug
      common /arr3/     x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)qi,qj,s,iq1,iq2

      qgjeto=0.d0
      qq=max(qi,qj)

      zmin=qq*fqscal*4.d0/s
      zmax=1.d0-epsxmn
      if(zmin.ge.zmax)return

      dpx1=0.d0
      zmin1=min(.2d0,1.d0-zmin)
      do i1=1,7
      do m1=1,2
       z=1.d0-epsxmn*(zmin1/epsxmn)**(.5d0+x1(i1)*(m1-1.5d0))

       si=z*s
       fb=qgjeti(qi,qj,si,z,1.d0,iq1,iq2,1)
       dpx1=dpx1+a1(i1)*fb*(1.d0-z)
      enddo
      enddo
      dpx1=dpx1*dlog(zmin1/epsxmn)

      dpx2=0.d0
      if(zmin.lt..8d0)then
       zmin1=zmin**(-delh)
       zmax1=.8d0**(-delh)
       do i1=1,7
       do m1=1,2
        z=(.5d0*(zmax1+zmin1+(zmax1-zmin1)*x1(i1)*(2*m1-3)))
     *  **(-1.d0/delh)

        si=z*s
        fb=qgjeti(qi,qj,si,z,1.d0,iq1,iq2,1)
        dpx2=dpx2+a1(i1)*fb*z**(1.d0+delh)
       enddo
       enddo
       dpx2=dpx2*(zmin1-zmax1)/delh
      endif
      qgjeto=(dpx1+dpx2)/qgsudx(qj,iabs(iq2)+1)*pi**3

      if(debug.ge.3)write (moniou,202)qgjeto
201   format(2x,'qgjeto: qi=',e10.3,2x,'qj=',e10.3,2x,
     *'s= ',e10.3,2x,'iq1= ',i1,2x,'iq2= ',i1)
202   format(2x,'qgjeto=',e10.3)
      return
      end

c=============================================================================
      double precision function qgjett(qi,qj,s,iq1,iq2)
c-----------------------------------------------------------------------------
c qgjett - hard 2->2 parton scattering born cross-section
c s is the c.m. energy square for the scattering process,
c iq1 - parton type at current end of the ladder (0 - g, 1,2 etc. - q)
c iq2 - parton type at opposite end of the ladder (0 - g, 1,2 etc. - q)
c-----------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr17/ dels,alfp,sigs,rr,r3p,g3p,delh,sgap
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/    debug
      common /arr3/     x1(7),a1(7)

      if(debug.ge.2)write (moniou,201)qi,qj,s,iq1,iq2

      qgjett=0.d0
      qq=max(qi,qj)

      zmin=qq*fqscal*4.d0/s
      zmax=(1.d0-epsxmn)**2
      if(zmin.ge.zmax)return
      zmin1=zmin**(-delh)
      zmax1=zmax**(-delh)
      do i1=1,7
      do m1=1,2
       z=(.5d0*(zmax1+zmin1+(zmax1-zmin1)*x1(i1)*(2*m1-3)))
     * **(-1.d0/delh)

       si=z*s
       fb1=0.d0
       zmin2=min(.2d0,1.d0-dsqrt(z))
       do i2=1,7
       do m2=1,2
        z1=1.d0-epsxmn*(zmin2/epsxmn)**(.5d0+x1(i2)*(m2-1.5d0))
        z2=z/z1

        fb1=fb1+a1(i2)*(qgjeti(qi,qj,si,z1,z2,iq1,iq2,2)
     *  +qgjeti(qi,qj,si,z2,z1,iq1,iq2,2))*(1.d0/z1-1.d0)
       enddo
       enddo
       fb1=fb1*dlog(zmin2/epsxmn)

       fb2=0.d0
       if(z.lt..64d0)then
        do i2=1,7
        do m2=1,2
         z1=.8d0*(dsqrt(z)/.8d0)**(.5d0+x1(i2)*(m2-1.5d0))
          z2=z/z1

         fb2=fb2+a1(i2)*(qgjeti(qi,qj,si,z1,z2,iq1,iq2,2)
     *   +qgjeti(qi,qj,si,z2,z1,iq1,iq2,2))
        enddo
        enddo
        fb2=fb2*dlog(.64d0/z)/2.d0
       endif

       qgjett=qgjett+a1(i1)*(fb1+fb2)*z**(1.d0+delh)
      enddo
      enddo
      qgjett=qgjett*(zmin1-zmax1)/delh*pi**3/2.d0

      if(debug.ge.3)write (moniou,202)qgjett
201   format(2x,'qgjett: qi=',e10.3,2x,'qj=',e10.3,2x,
     *'s= ',e10.3,2x,'iq1= ',i1,2x,'iq2= ',i1)
202   format(2x,'qgjett=',e10.3)
      return
      end

c=============================================================================
      double precision function qgjeti(qi,qj,si,z1,z2,iq1,iq2,jj)
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/    debug
      common /arr3/     x1(7),a1(7)

      qgjeti=0.d0
      qq=max(qi,qj)
      tmin=qq*fqscal/(.5d0+dsqrt(max(0.d0,.25d0-qq*fqscal/si)))
      if(tmin.ge.si/2.d0)return
      do i=1,7
      do m=1,2
       t=2.d0*tmin/(1.d0+2.d0*tmin/si
     *   -x1(i)*(2*m-3)*(1.d0-2.d0*tmin/si))
       qt=t*(1.d0-t/si)

       fb=0.d0
       if(jj.eq.1)then
        do iql=1,2
         iq=2*iql-2
         dfb=0.d0
         do n=1,3
          dfb=dfb+qgfbor(si,t,iq,iq2,n)+qgfbor(si,si-t,iq,iq2,n)
         enddo
         if(iq.eq.iq2)dfb=dfb/2.d0
         fb=fb+dfb*qgevi(qi,qt/fqscal,z1,iabs(iq1)+1,iql)
        enddo
        fb=fb*qgsudx(qt/fqscal,iabs(iq2)+1)
       else
        do iql=1,2
         iq=2*iql-2
        do iqr=1,2
         dfb=0.d0
         do n=1,3
          dfb=dfb+qgfbor(si,t,iq,iqr-1,n)+qgfbor(si,si-t,iq,iqr-1,n)
         enddo
         if(iq.eq.iqr-1)dfb=dfb/2.d0
         fb=fb+dfb*qgevi(qi,qt/fqscal,z1,iabs(iq1)+1,iql)
     *   *qgevi(qj,qt/fqscal,z2,iabs(iq2)+1,iqr)
        enddo
        enddo
       endif

       qgjeti=qgjeti+a1(i)*fb*qgalf(qt/fqscal/alm)**2*t**2
      enddo
      enddo
      qgjeti=qgjeti*(1.d0/tmin-2.d0/si)/si**2
      return
      end

c=============================================================================
      double precision function qgptj(s,pt,y0,sigin)
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr6/  pi,bm,amws
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgarr51/ epsxmn
      common /qgdebug/    debug
      common /arr3/     x1(7),a1(7)

      qgptj=0.d0
      zmin=4.d0*pt**2/s
      xt=2.d0*pt*exp(y0)/dsqrt(s)
      zmax=min(1.d0,xt**2/(2.d0*xt-zmin))
      if(zmax.le.zmin)return

      qq=pt**2/fqscal
      do i1=1,7
      do m1=1,2
       z=zmax*(zmin/zmax)**(.5d0+x1(i1)*(m1-1.5d0))
       si=z*s
       t=2.d0*pt**2/(1.d0+dsqrt(max(0.d0,1.d0-zmin/z)))

       xmax=min(1.d0,xt/(1.d0+dsqrt(max(0.d0,1.d0-zmin/z))))
       xmin=max(z,xmax*exp(-2.d0*y0))
       do i2=1,7
       do m2=1,2
        xp=xmax*(xmin/xmax)**(.5d0+x1(i2)*(m2-1.5d0))
        xm=z/xp

        glu1=qgpdf(xp,qq,2,0)
        glu2=qgpdf(xm,qq,2,0)
        seav2=qgpdf(xm,qq,2,-1)+qgpdf(xm,qq,2,1)+qgpdf(xm,qq,2,2)

        qgptj=qgptj+a1(i1)*a1(i2)*(qgptjb(si,pt**2,t,1)*glu1*glu2
     *  +qgptjb(si,pt**2,t,2)*glu1*seav2)
     *  *dlog(xmax/xmin)/(1.d0-2.d0*t/si)
       enddo
       enddo
      enddo
      enddo
      qgptj=qgptj*dlog(zmax/zmin)*pi**3*.39d0/sigin  *2.  !2 jets
      return
      end

c=============================================================================
      double precision function qgptjb(si,qt,t,jj)
      implicit double precision (a-h,o-z)
      integer debug
      common /qgarr18/ alm,qt0,qtf,betp,dgqq
      common /qgarr26/ factk,fqscal
      common /qgarr43/ moniou
      common /qgdebug/    debug

      if(jj.eq.1)then
       qgptjb=qgfbor(si,t,0,0,1)
      else       !if(jj.eq.2)then
       qgptjb=qgfbor(si,t,0,1,1)
      endif
      qgptjb=qgptjb*qgalf(qt/fqscal/alm)**2/si**2
      return
      end
