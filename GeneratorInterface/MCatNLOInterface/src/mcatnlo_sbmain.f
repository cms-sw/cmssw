      SUBROUTINE SBMAIN
      implicit none
      include 'svbcblks.h'
      real * 8 value(20),xmass(-5:21)
      real * 8 
     #  xm0,xpdflam4,xpdflam5,tmp,xren,xfh,xrenmc,xfhmc,ecm,xicut,roh,
     #  delta,deltas,deltac,dtot,avtot,ac1,ac2,xtotal,ytotal,av3a,
     #  av3nega,d3a,d3nega,ctime,pi,tmas,etacut,av3b,av3negb,d3b,
     #  d3negb,wgtaev,wgtbev,evfrac,evprcfrac,xares,yares,xbres,ybres,
     #  al_gfun,be_gfun,ccc_gfun,dummy,xmone,xicutss,gammax,
     #  xmlow,xmupp,bwxmdpl,bwxmdmn,pfcwon,pfcwmsb,pfczon,pfczmsb,
     #  xalfaem
      integer 
     #  ih1,ih2,ndns1,ndns2,iseld,nlf,ncl3,mx_of_evta,mx_of_evtb,
     #  loproc,maproc,iproc,iinput,iverbose,ichkmom,
     #  ibswrite,itmpih,itmpndns,idpdfset,ipdfih,ipdfgroup,ipdfndns,
     #  ifk88istrl,ifk88ih,ifk88ndns,maxevt,it1,it2,ifuntype,
     #  ndim,nwild,itd1,itd2,ibscall,iwgtnorm,iseed0,ifxdaem,
     #  iseed,maxtrials,mode,lo,isubttype,iprespl,itmp,iprdct,
     #  ndiff,ntotal,i,ionshell,iwrong,iwrong1,nevts,ntrls,
     #  iunita,iunitb,iunit,ioutput,itot,ii,iwidth,inloscale,
     #  imcscale,ifk88seed,izero,ione
      character * 2 scheme
      character * 4 part1,part2
      character * 20 parm(20),gname
      character * 80 fname,fnamea,fnameb,fname1,fnamev
      character * 80 pref,prefn,prefev,prefnev
      character * 70 strin,strout,lhapdf
      logical evgen
      external ssb5afw,ssb5azw,ssb5bfw,ssb5bzw
      parameter (pi=3.14159265358979312D0)
      parameter (xmone=-1.d0)
      parameter (izero=0)
      parameter (ione=1)
c
c common /strfsb0/ is only in strfsb:
c ndns = pdf type
c ih1,ih2 = beam type (0=(p+n)/2, 1=p, -1=pbar, 2=n, -2=nbar)
      common/strfsb0/ih1,ih2,ndns1,ndns2
c quark and gluon masses, used by Herwig. PDF labeling convention
      common/parmass/xmass
c al_gfun, be_gfun, ccc_gfun are the parameters entering gfun
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
c iwgtnorm=0 for weight=+1/-1, iwgtnorm=1 otherwise
      common/ciwgtnorm/iwgtnorm
c wgtaev and wgtbev are the norms of weights for H and S events respectively
      common/cwgtev/wgtaev,wgtbev
c iprespl=0 ==> preserves rapidity
c iprespl=1 ==> preserves longitudinal momentum
      common/ciprespl/iprespl
c ifxdaem=0 ==> uses running alpha_EM(M^2)
c ifxdaem=1 ==> uses alpha_EM=1/137.0359895
c ifxdaem=2 ==> uses running alpha_EM(M0^2)
c Options 0 and 2 are equivalent for zero-width bosons. Option 2 is
c what results by letting width-->0 in dilepton production. For gamma
c production, ifxdaem=2 is converted to ifxdaem=0
      common/cifxdaem/ifxdaem
c ichkmom=0 --> enables checks on kinematics
      common/cichkmom/ichkmom
c----------------------------------------------------------
c Variables that control the integrations
c
      common/cisubttype/isubttype
      common/parsub/xicut,delta
      common/xisave/xicutss
      common/pmerge/deltas,deltac
c etacut is the maximum allowed for [2*kt(gluon)/sqrt(shat)]^2
      common/cetacut/etacut
      integer nsamp
      common/samp/nsamp
c Prefactors for W and Z cross sections in the on-shell and MSbar schemes
      common/xsecpref/pfcwon,pfcwmsb,pfczon,pfczmsb
c----------------------------------------------------------
c inloscale controls the reference scale in the NLO computation
      common/cinloscale/inloscale
c imcscale controls the reference scale in the MC subtraction terms
      common/cimcscale/imcscale
c----------------------------------------------------------
c Process: iprdct=1396,1397,1497,1498 for gamma,Z,W+,W-
      common/ciprdct/iprdct
c Subprocesses: 'qq', 'qg', corresponding to jproc=jproc0=2,3
c In the integration routines, loproc<=jproc<=maproc
      common/cwchproc/loproc,maproc
c Number of failures in flavour determination
      common/ciwrong/iwrong,iwrong1
c----------------------------------------------------------
c ifuntype=1 for ssb5a, ifuntype=2 for ssb5b
      common/cifuntype/ifuntype
c Flag to put partons on shell, according to Herwig list of masses
      common/cionshell/ionshell
c----------------------------------------------------------
c Common blocks for general MC@NLO routines
c common block for internal rnd number generation, independent of bases
      common/cifk88seed/ifk88seed
c common block fk88ipdfs is filled by our interface to MLMPDF
      common/fk88ipdfs/ifk88ih,ifk88ndns
c common block w50511 and w50512 are filled by PDFLIB 
      common/w50511/ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,tmas
      common/w50512/xpdflam4,xpdflam5
C

CCCC  ================================================================
CCCC  added by Fabian Stoeckli (fabian.stoeckli@cern.ch)
CCCC  in order to control bases output
      integer basesoutput
      character*100 stfilename
      common/fstbases/ basesoutput, stfilename
CCCC  ================================================================


CCCC ====================================================================
CCCC  added by fabian stoeckli (fabian.stoeckli@cern.ch)
CCCC  12.2.2007
CCCC  common block and variabes added for CMSSW interface
      character * 20 mmgname
      character * 2 mmscheme
      character * 4 mmpart1,mmpart2
      real * 8 mmecm,mmxren,mmxfh,mmxrenmc,mmxfhmc,mmxmh0,mmgah,mmxmt,
     % mmgammax,mmxmhl,mmxmhu,mmxmass1,mmxmass2,mmxmass3,mmxmass4,
     % mmxmass5,mmxmass21,mmxlam,mmxm0,mmgav,mmxm0v,mmtwidth
      real * 8 mmxwm,mmxzm,mmxww,mmxzw,mmv1gammax,mmv1massinf,
     % mmv1masssup
      real * 8 mmv2gammax,mmv2massinf,mmv2masssup,mmvud,mmvus,mmvub,
     % mmvcd,mmvcs,mmvcb
      real * 8 mmvtd,mmvts,mmvtb,mmxlamherw
      integer mmmaxevt,mmidpdfset,mmiwgtnorm,mmiseed,mmibornex,
     % mmit1,mmit2
      integer mmivcode,mmil1code,mmil2code,mmaemrun,mmiproc

cccc common block for the input parameters
      common/params/mmecm,mmxren,mmxfh,mmxrenmc,mmxfhmc,mmxmh0,mmgah,
     % mmxmt,mmgammax,mmxmhl,mmxmhu,mmxmass1,mmxmass2,mmxmass3,mmxmass4,
     % mmxmass5,mmxmass21,mmxlam,mmxm0,mmgav,mmxm0v,mmtwidth,
     % mmxwm,mmxzm,mmxww,mmxzw,
     % mmv1gammax,mmv1massinf,mmv1masssup,mmv2gammax,mmv2massinf,
     % mmv2masssup,mmvud,mmvus,mmvub,
     % mmvcd,mmvcs,mmvcb,mmvtd,mmvts,mmvtb,mmxlamherw,
     % mmmaxevt,mmidpdfset,mmiwgtnorm,
     % mmiseed,mmibornex,mmit1,mmit2,mmivcode,mmil1code,mmil2code,
     % mmaemrun,mmiproc,mmgname,mmscheme,
     % mmpart1,mmpart2

      integer mmdecide
      common/decpar/mmdecide
CCC =================================================================

C------------------------------------------------------------------------
C                             START                                     -
C------------------------------------------------------------------------
c iinput=1 ==> all inputs are given by the user
      iinput=2
      mmdecide=6
c iverbose=1 ==> writes more on standard output
      iverbose=0
c ichkmom=0 ==> enables checks on kinematics
      ichkmom=1
c initialization of internal randon number generation
      ifk88seed=1
c if linked to PDFLIB, these quantities stay negative
      ifk88ih=-100
      ifk88ndns=-100
c forces the code to get Lambda value if not obtained from PDFLIB/MLMPDF
      xpdflam5=-1.d0
C Set system dependent parameters
      call sysdep
c----- vegas prints nothing
c      call nopr(0)
c Bases writes data file
      ibswrite=1
c-----
c Open the file collecting all the input parameter. This file is meant 
c to be converted in a command file in a subsequent run
      open(unit=11,file='svblog',status=newver)
c
CCC =================================================================
CCC   added by fabian stoeckli
CCC   read string in from file
      open(unit=33,file=stfilename,status='old')
CCC =================================================================

      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     # 'Enter prefix for name of BASES files'
         read (*,*) pref
      else
         read(33,*) pref
      endif
      write(11,*) ''''//pref(1:ifk88istrl(pref))//'''',
     # '  ! prefix for BASES files'
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     # 'Enter prefix for name of event files'
         read (*,*) prefev
      else
         read(33,*) prefev
      endif
      close(33)
      write(11,*) ''''//prefev(1:ifk88istrl(prefev))//'''',
     # '  ! prefix for event files'
c----------------------------------------------------------
c Parameters of the run
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     # 'Enter pair ECM(GeV),fren[NLO],ffact[NLO],fren[MC],ffact[MC]'
      write(*,*)' fren=mu_ren/mu0'
      write(*,*)' ffact=mu_fac/mu0'
      write(*,*)' mu_ren=renormalization scale'
      write(*,*)' mu_fac=factorization scale'
      write(*,*)' mu0=reference scale'
         read(*,*) ecm,xren,xfh,xrenmc,xfhmc
      else
         ecm=mmecm
         xren=mmxren
         xfh=mmxfh
         xrenmc=mmxrenmc
         xfhmc=mmxfhmc
      endif
      write(11,'(5(1x,d10.4),1x,a)') ecm,xren,xfh,xrenmc,xfhmc
     #     ,'! Ecm, fren, ffact, frenmc, ffactmc'
      sh = ecm**2
c Will allow more flexibility in future versions
      xfh = xren
      xrenmc = xren
      xfhmc = xfh
      xren2 = xren**2
      xf2h1 = xfh**2
      xf2h2 = xfh**2
      xren2mc = xrenmc**2
      xf2h1mc = xfhmc**2
      xf2h2mc = xfhmc**2
c---------------------------------------------------------------
c Select process
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter -(1)1396 for gamma production'
      write(*,*)'      -(1)1397 for Z production'
      write(*,*)'      -(1)1497 for W+ production'
      write(*,*)'      -(1)1498 for W- production'
         read(*,*) itmp
      else
         itmp=mmiproc
      endif
      iprdct=mod(-itmp,10000)
      if(iprdct.ne.1396.and.iprdct.ne.1397 .and.
     #   iprdct.ne.1497.and.iprdct.ne.1498) then
         write(*,*)'Process not implemented'
         stop
      endif
      write(11,'(1x,i6,27x,a)') -iprdct,
     #  '! -1396/97/1497/98=gamma/Z/W+/W-'
c Inizialization of physical parameters
      call setpsb()
c Vector boson mass and width
      if(iprdct.ne.1396)then
        if(iinput.lt.2) then
        write(*,*)' '
        write(*,*)'Enter vector boson mass (GeV), its width (GeV),'
        write(*,*)' and GammaX; the boson mass is chosen in the range'
        write(*,*)'      M0-GammaX*width < M_V < M0+GammaX*width'
           read(*,*)xm0,gah,gammax
        else
           xm0=mmxm0
           gah=mmgah
           gammax=mmv1gammax
        endif
        write(11,'(3(1x,d10.4),1x,a)') xm0,gah,gammax,
     #       '! M_V, Ga_V, GammaX'
        xm02 = xm0**2
        if(gah.ge.0.001d0)then
           iwidth=1
        else
           iwidth=0
           gah=0.d0
        endif
      else
        if(iinput.lt.2) then
        write(*,*)' '
        write(*,*)'Enter M_gamma(min), M_gamma(max)'
           read(*,*)xmlow,xmupp
        else
           xmlow=mmv1massinf
           xmupp=mmv1masssup
        endif
        write(11,'(2(1x,d10.4),12x,a)') xmlow,xmupp,
     #    '! M_gamma(min), M_gamma(max)'
        if(xmupp-xmlow.ge.0.001d0)then
          iwidth=1
          gah=xmupp-xmlow
        else
          iwidth=-1
          gah=-1.d0
          write(*,*)'   '
          write(*,*)'Zero cross section'
          stop
        endif
        gammax=-1.d0
        xm0=(xmupp+xmlow)/2.d0
        xm02=xm0**2
      endif
c Quark and gluon masses (must be consistent with Herwig)
      do i=-5,21
        xmass(i)=0.d0
      enddo
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter d, u, s, c, b, glu (Herwig) masses'
      write(*,*)'Enter d, u, s, c, b, glu (Herwig) masses'
         read(*,*)xmass(1),xmass(2),xmass(3),xmass(4),xmass(5),xmass(21)
      else
         xmass(1)=mmxmass1
         xmass(2)=mmxmass2
         xmass(3)=mmxmass3
         xmass(4)=mmxmass4
         xmass(5)=mmxmass5
         xmass(21)=mmxmass21
      endif
      write(11,'(6(1x,d10.4),1x,a)') xmass(1),xmass(2),xmass(3),
     #  xmass(4),xmass(5),xmass(21),'! quark and gluon masses'
      do i=-5,-1
        xmass(i)=xmass(-i)
      enddo
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)
     #  'Enter beam type for beam1 and beam2 (p, pbar, n, nbar):'
         read(*,*) part1,part2
      else
         part1=mmpart1
         part2=mmpart2
      endif
      write(11,'(1x,a,2x,a,19x,a)') ''''//part1//'''',
     #  ''''//part2//'''','! hadron types'
      strin=part1
      call fk88low_to_upp(strin,strout)
      part1=strout
      strin=part2
      call fk88low_to_upp(strin,strout)
      part2=strout
      if( (part1.ne.'P   ' .and. part1.ne.'PBAR' .and. 
     #     part1.ne.'N   ' .and. part1.ne.'NBAR') .or.
     #    (part2.ne.'P   ' .and. part2.ne.'PBAR' .and. 
     #     part2.ne.'N   ' .and. part2.ne.'NBAR') )then
        write(*,*)'This code only works for hadronic collisions'
        stop
      endif
      call getihpart(part1,itmpih)
      ih1=itmpih
      call getihpart(part2,itmpih)
      ih2=itmpih
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter group name and id number for PDF set'
         read(*,*)gname,idpdfset
      else
         gname=mmgname
         write(*,*) gname
         idpdfset=mmidpdfset
      endif
      write(11,'(1x,a,1x,i6,21x,a)') 
     # ''''//gname(1:ifk88istrl(gname))//'''',
     # idpdfset,'! PDF group and id number'
      strin=gname
      call fk88low_to_upp(strin,strout)
      if(strout.eq.'LHAPDF'.or.strout.eq.'LHAEXT')then
        lhapdf='FREEZE'
        if(strout.eq.'LHAEXT')lhapdf='EXTRAPOLATE'
        call setlhacblk(lhapdf)
        parm(1)='DEFAULT'
      else
        lhapdf='NOTLKD'
        parm(1)=gname
      endif
      value(1)=idpdfset
      call pdfset(parm,value)
      if(ipdfih.ne.1)then
        write(*,*)'PDFLIB could not understand the input'
        write(*,*)'Hadron type:',ipdfih
        stop
      endif
      if(ifk88ih.eq.-100.and.ifk88ndns.eq.-100)then
        if(lhapdf.eq.'NOTLKD')then
c the code is linked to PDFLIB; get the MLM pdf id number from
c ipdfih, ipdfgroup, and ipdfndns returned by PDFLIB in /w50511/
          call pdftomlm(ipdfih,ipdfgroup,ipdfndns,itmpih,itmpndns)
          ndns1=itmpndns
          ndns2=itmpndns
        elseif(lhapdf.eq.'FREEZE'.or.lhapdf.eq.'EXTRAPOLATE')then
c the code is linked to LHAPDF, which doesn't fill /w50511/
          call pdftomlm(ione,izero,idpdfset,itmpih,itmpndns)
          ndns1=itmpndns
          ndns2=itmpndns
        else
          write(*,*) 'Unknown lhapdf value: ',lhapdf
          stop
        endif
      else
c the code is linked to the interface to MLMPDF
        ndns1=ifk88ndns
        ndns2=ifk88ndns
      endif
c
      scheme='**'
      xlam=xpdflam5
c
      if(xlam.gt.0) then
         if(iinput.lt.2) then            
         write(*,*)' '
         write(*,*) 'Enter Lambda_QCD_5, < 0 for default'
         write(*,*) 'Enter Lambda_QCD_5, < 0 for default'
            read(*,*) tmp
         else
            tmp=mmxlam
         endif
         write(11,'(1x,d10.4,23x,a)') tmp,'! Lambda_5, < 0 for default'
         if(tmp.gt.0) xlam=tmp
      else
         dowhile(xlam.le.0)
            if(iinput.lt.2) then
            write(*,*)' '
            write(*,*)'Enter Lambda_5_2loop'
               read(*,*) xlam
            else
               xlam=mmxlam
            endif
c            if (xlam.le.0) call prntsf
         enddo
         write(11,'(1x,d10.4,23x,a)') xlam,'! lambda'
      endif
C      write(*,*) 'Lambda_5=',xlam,' GeV'
c
 22   if(scheme.ne.'DI'.and.scheme.ne.'MS') then
         if(iinput.lt.2) then
         write(*,*)' '
         write(*,'(1x,a)') 'Enter scheme: ''DI'' or ''MS'''
            read(*,*) scheme
         else
            scheme=mmscheme
         endif
         if(scheme.ne.'DI'.and.scheme.ne.'MS') then
c            call prntsf
            goto 22
         endif
         write(11,'(1x,a,29x,a)') ''''//scheme//'''','! scheme'
      endif
C      write(*,*) 'Scheme=',scheme
      schhad1=scheme
      schhad2=scheme
c-----------------------------------------------------------------
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter alpha and beta for the function G'
        write(*,*)' Defaults are: alpha=1, beta=0.1'
        write(*,*)' Allowed ranges: alpha>=1, 0<beta<=1'
        read(*,*) al_gfun,be_gfun
        ccc_gfun=0.d0
        write(11,'(3(1x,d10.4),1x,a)') al_gfun,be_gfun,ccc_gfun,
     #    '! alpha, beta, c'
      else
        al_gfun=1.d0
        be_gfun=0.1d0
        ccc_gfun=0.d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)
     # 'Enter accuracies for grid setup and for integral evaluation'
        read(*,*)ac1,ac2
        write(11,'(2(2x,d10.4),10x,a)') ac1,ac2,'! ac1,ac2'
      else
        ac1=0.2d0
        ac2=0.05d0
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'For the computation of alpha_S in NLO terms'
        write(*,*)'Enter 1 to set mu_0^2=M_V^2+pt_V^2'
        write(*,*)'      2 to set mu_0=M_V'
        write(*,*)'      3 to set mu_0=max(M_V,pt_V)'
        write(*,*)'      4 to set mu_0=max(M_V-pt,pt)'
        write(*,*)' The default is 1'
        read(*,*) inloscale
        write(11,'(1(1x,i8),25x,a)') inloscale,
     #    '! 1->mu_0=mh+pt, 2->mu_0=mh'
      else
        inloscale=1
      endif
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'For the computation of alpha_S in MC terms'
        write(*,*)'Enter 1 to set mu_0^2=M_V^2+pt_V^2'
        write(*,*)'      2 to set mu_0=M_V'
        write(*,*)'      3 to set mu_0=max(M_V,pt_V)'
        write(*,*)'      4 to set mu_0=max(M_V-pt,pt)'
        write(*,*)' The default is 4'
        read(*,*) imcscale
        write(11,'(1(1x,i8),25x,a)') imcscale,
     #    '! 1->mu_0=mh+pt, 2->mu_0=mh'
      else
        imcscale=5
      endif
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter the maximum number of events to generate;'
      write(*,*)'enter 0 to skip the event generation step'
         read(*,*)maxevt
      else
         maxevt=mmmaxevt
      endif
      write(11,'(1(1x,i8),25x,a)') maxevt,'! number of events'
      evgen=.true.
      if(maxevt.eq.0)then
        evgen=.false.
        maxevt=100000
      endif
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter 0 to have +1/-1 event weights'
      write(*,*)'      1 to normalize the weights, in such a way that'
      write(*,*)'        their sum is equal to the total rate'
         read(*,*)iwgtnorm
      else
         iwgtnorm=mmiwgtnorm
      endif
      write(11,'(1(1x,i8),25x,a)') iwgtnorm,
     #  '! 0 => wgt=+1/-1, 1 otherwise'
c iseed0 is the seed for the integration step, iseed is the seed
c for the event generation step
      iseed0=12345
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter the seed for random numbers; it will be used'
      write(*,*)'to generate events. Enter 0 for default'
         read(*,*)iseed
      else
         iseed=mmiseed
      endif
      write(11,'(1(1x,i8),25x,a)') iseed,'! seed for rnd numbers'
      if(iseed.lt.0)then
        stop
      elseif(iseed.eq.0)then
        iseed=iseed0
      endif
c Here, assume that the unweighting efficiency is larger than 10%
      maxtrials=10*maxevt
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to use standard subtraction'
        write(*,*)'      1 to use zeta subtraction'
        read(*,*)isubttype
        write(11,'(1(1x,i8),25x,a)') isubttype,
     #    '! 0=subt, 1=zeta subt'
      else
        isubttype=0
      endif
      if(iinput.eq.1)then
        if(isubttype.eq.0)then
          write(*,*)' '
          write(*,*)'enter xicut and delta (defaults: 1, 2)'
          read(*,*)xicut,delta
          write(11,'(2(2x,d10.4),10x,a)') xicut,delta,'! xicut,delta'
        else
          write(*,*)' '
          write(*,*)'Enter zi ( [ 2*kt(gluon)/sqrt(shat) ]^2 < zi )'
          read(*,*) etacut
          write(11,'(1x,d10.4,23x,a)') etacut,'! zi'
          xicut = 1.d0
          delta = 1.d0
        endif
      else
        if(isubttype.eq.0)then
          xicut = 1.d0
          delta = 2.d0
        else
          etacut = 1.d0
          xicut = 1.d0
          delta = 1.d0
        endif
      endif
      xicutss = xicut
      deltas = 0
      deltac = 0
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to preserve rapidity'
        write(*,*)'      1 to preserve longitudinal momentum'
        read(*,*)iprespl
        write(11,'(1(1x,i8),25x,a)') iprespl,'! 0=y, 1=k_3 preserved'
      else
        iprespl=0
      endif
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter 0 to use running alpha_EM'
      write(*,*)'      1 to use alpha_EM=1/137.0359895'
      write(*,*)'      2 to use alpha_EM(M0)'
         read(*,*)ifxdaem
      else
         ifxdaem=mmaemrun
      endif
      write(11,'(1(1x,i8),25x,a)') ifxdaem,
     #  '! 0=aem(M), 1=Thompson, 2=aem(M0)'
      if(iprdct.eq.1396.and.ifxdaem.eq.2)ifxdaem=0
      if(ifxdaem.ne.0.and.ifxdaem.ne.1.and.ifxdaem.ne.2)then
        write(*,*)'No such option for alpha_em'
        stop
      endif
      if(ifxdaem.eq.0)ze2=0.d0
c---------------------------------------------------------------
c Initialize parameters, such as labelling for parton processes
      call parsetpsb()
c Select partonic subprocess
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*) 'Enter 2 for qq, 3 for qg, 0 for all'
        read(*,*) iproc
        write(11,'(1x,i2,31x,a)') iproc,'! 2=qq, 3=qg, 0=all'
      else
        iproc=0
      endif
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to leave the partons massless'
        write(*,*)'      2 to put partons on mass shell'
        read(*,*) ionshell
        write(11,'(1x,i1,32x,a)') 
     #      ionshell,'! 0=massless, 2=massive partons'
      else
        ionshell=2
      endif
      if(ionshell.ne.0.and.ionshell.ne.2) then
        write(*,*) 'Error: enter 0 or 2.'
        stop
      endif
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter number of iterations'
      write(*,*)'for grid setup and for integral evaluation;'
      write(*,*)'set either one to 0 to skip the integration step'
         read(*,*) it1,it2
      else
         it1=mmit1
         it2=mmit2
      endif
      write(11,'(2(1x,i4),24x,a)') it1,it2,'! itmx1,itmx2'
      iseld=1
      if(it1.eq.0.or.it2.eq.0)iseld=0
c---------------------------------------------------------------
c Integration parameters
c
      if(iinput.eq.1)then
        if(iseld.eq.1)then
          write(*,*)' '
          write(*,*)
     #     'Enter number of calls for bases (<0 for default)'
          read(*,*)ncl3
          if(ncl3.lt.0)ncl3=120000
          write(11,'(1x,i9,24x,a)')ncl3,'! # of calls for bases'
        endif
      else
        ncl3=120000
      endif
c---- close logfile
      close(11)
c----------------------------------------------------------------
c  *********************  START INTEGRATION *********************
c----------------------------------------------------------------
      ifuntype=0
      loproc = 2
      maproc = 3
      if(iproc.ne.0) then
        loproc=iproc
        maproc=iproc
      endif
      prefn = pref
      prefnev = prefev
c tau generated according to a flat distribution in (1/tau)**nsamp
      nsamp = 1
c
      if(iwidth.eq.1)then
        ndim=4
        nwild=4
        xm2=-1.d10
        if(gammax.ge.0)then
          xmlow2=(max(0.d0,xm0-gammax*gah))**2
          xmupp2=(xm0+gammax*gah)**2
        else
          xmlow2=xmlow**2
          xmupp2=xmupp**2
        endif
        xmlow2=max(100.d0,xmlow2)
        xmupp2=min(sh*0.95,xmupp2)
        bwxmdpl=xmupp2-xm02
        bwxmdmn=xm02-xmlow2
        if(bwxmdpl.lt.0.d0.or.bwxmdmn.lt.0.d0)then
          write(*,*)'Error in the computation of BW bounds'
          stop
        endif
        bwfmpl=atan(bwxmdpl/(xm0*gah))
        bwfmmn=atan(bwxmdmn/(xm0*gah))
        bwdelf=(bwfmpl+bwfmmn)/pi
        if(ifxdaem.eq.2)ze2=4*pi*xalfaem(xm02)
      else
        if(iprdct.eq.1396)then
          write(*,*)'   '
          write(*,*)'For gamma production, a non-zero width range'
          write(*,*)'must be entered'
          stop
        endif
        ndim=3
        nwild=3
        xm2=xm02
        roh=xm2/sh
        if(ifxdaem.eq.0.or.ifxdaem.eq.2)ze2=4*pi*xalfaem(xm2)
        if(xicut.gt.(1-roh))xicut=1-roh
        bwfmpl=0.d0
        bwfmmn=0.d0
        bwdelf=0.d0
      endif
c Perform the integration step
      if(iseld.eq.1)then
        xtotal=0.d0
        ytotal=0.d0
        dtot=0.d0
        xares=0.d0
        yares=0.d0
        xbres=0.d0
        ybres=0.d0
        mx_of_evta=0
        mx_of_evtb=0
c
        ifuntype=1
        call fk88strcat(prefn,'_a',fnamea)
        if(iwidth.eq.1)then
          call run_bases(ssb5afw,fnamea,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        else
          call run_bases(ssb5azw,fnamea,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        endif
        if(basesoutput.gt.0) then
        write(*,*)'   '
        write(*,*)'|integral[a]|:',av3a,' +- ',d3a
        write(*,*)' integral[a] :',av3nega,' +- ',d3nega
        endif
        xares=av3a
        yares=av3nega
        xtotal=xtotal+xares
        ytotal=ytotal+yares
        dtot=dtot+d3nega**2
c
        ifuntype=2
        call fk88strcat(prefn,'_b',fnameb)
        if(iwidth.eq.1)then
          call run_bases(ssb5bfw,fnameb,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        else
          call run_bases(ssb5bzw,fnameb,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        endif
        if(basesoutput.gt.0) then
        write(*,*)'   '
        write(*,*)'|integral[b]|:',av3b,' +- ',d3b
        write(*,*)' integral[b] :',av3negb,' +- ',d3negb
        endif
        xbres=av3b
        ybres=av3negb
        xtotal=xtotal+xbres
        ytotal=ytotal+ybres
        dtot=dtot+d3negb**2
c
        avtot=ytotal
        dtot=sqrt(dtot)
           call fk88strcat(prefn,'.integrals',fname)
           open(unit=21,file=fname,
     #          form='formatted',status='unknown')
           write(21,240)xares
           write(21,240)xbres
           write(21,240)yares
           write(21,240)ybres
           close(21)
 240       format(1x,d14.8)
      endif
c Sanity check
      if( isubttype.eq.1 .and. iwidth.eq.0 .and.
     #   (abs(xicut-1+roh).gt.1.d-7.or.delta.ne.1.d0) )then
        write(*,*)'Fatal error: xicut, delta=',xicut,delta
        stop
      endif
      if(iseld.eq.0)then
c Read integrals from disk only if the integration step has been skipped
        call fk88strcat(prefn,'.integrals',fname)
        open(unit=21,file=fname,
     #       form='formatted',status='old')
        read(21,240)xares
        read(21,240)xbres
        read(21,240)yares
        read(21,240)ybres
        close(21)
      endif
c
c Generates events when evgen=.true.; if evgen=.false., maxevt=100000 in
c order to estimate the number of negative weights
      if(maxevt.ne.0)then
        ntotal=0
        xtotal=0.d0
        ytotal=0.d0
        xtotal=xtotal+xares+xbres
        ytotal=ytotal+yares+ybres
        avtot=ytotal
c For future upgrades, define the weights of H and S events; this is 
c necessary when the relative number of H and S events is not generated
c according to total rates
        if(iwgtnorm.eq.0)then
          wgtaev=1.d0
          wgtbev=1.d0
        else
          wgtaev=xtotal/dfloat(maxevt)
          wgtbev=xtotal/dfloat(maxevt)
        endif
        mx_of_evta=int(maxevt*xares/xtotal)
        mx_of_evtb=int(maxevt*xbres/xtotal)
        ntotal=ntotal+mx_of_evta+mx_of_evtb
        ndiff=maxevt-ntotal
        if(ndiff.gt.0)mx_of_evta=mx_of_evta+ndiff
        if(ndiff.lt.0)then
          write(6,*)'Fatal error:',maxevt,ntotal
          stop
        endif
        if(evgen)then
        if(basesoutput.gt.0) then
          write(*,*)'  '
          write(*,*)
     #  'The following number of events will be generated'
          write(*,*)'# events[a]:',mx_of_evta
          write(*,*)'# events[b]:',mx_of_evtb
          write(*,*)'  '
          write(*,*)
     #  'Estimated fractions of events with negative weights'
        endif
        endif
        evfrac=0.d0
        evprcfrac=(xares-yares)/
     #            (xares+yares)
        evprcfrac=evprcfrac/(1+evprcfrac)
        evfrac=evfrac+evprcfrac*mx_of_evta
        if(basesoutput.gt.0) then
        write(*,*)'Events[a]: w<0/all:',evprcfrac
        endif
        evprcfrac=(xbres-ybres)/
     #            (xbres+ybres)
        evprcfrac=evprcfrac/(1+evprcfrac)
        if(basesoutput.gt.0) then
        write(*,*)'Events[b]: w<0/all:',evprcfrac
        endif
        evfrac=evfrac+evprcfrac*mx_of_evtb
        evfrac=evfrac/dfloat(maxevt)
        if(basesoutput.gt.0) then
        write(*,*)'Events[all]: w<0/all:',evfrac
        endif
c
        if(.not.evgen)goto 111
        fname=prefnev
        call fk88strcat(fname,'_a.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evta
        close(22)
        call fk88strcat(fname,'_b.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='unknown')
        write(22,250)mx_of_evtb
        close(22)
c
        fname=prefn
        fnamev=prefnev
c
        iwrong=0
        iwrong1=0
        ifuntype=1
        call fk88strcat(fname,'_a',fnamea)
        call fk88strcat(fnamev,'_a.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        if(iwidth.eq.1)then
          call run_spring(ssb5afw,fnamea,mx_of_evta,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        else
          call run_spring(ssb5azw,fnamea,mx_of_evta,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        endif
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[a]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
        endif
c
        iwrong=0
        iwrong1=0
        ifuntype=2
        call fk88strcat(fname,'_b',fnameb)
        call fk88strcat(fnamev,'_b.events',fname1)
        open(unit=22,file=fname1,
     #       form='formatted',status='old')
        call toend(22)
        if(iwidth.eq.1)then
          call run_spring(ssb5bfw,fnameb,mx_of_evtb,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        else
          call run_spring(ssb5bzw,fnameb,mx_of_evtb,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        endif
        close(22)
        if(iverbose.eq.1)then
          write(*,*)'   '
          write(*,*)'Events[b]'
          write(*,*)'Trials:',ntrls
          write(*,*)'Events generated:',nevts
          write(*,*)'Unlike sign events(1):',iwrong
          write(*,*)'Unlike sign events(2):',iwrong1
          write(*,*)'Unlike sign(1)/all events:',
     #              iwrong/dfloat(nevts)
          write(*,*)'Unlike sign(2)/all events:',
     #              iwrong1/dfloat(nevts)
        endif
c write a single event file
        iunita=21
        call fk88strcat(prefnev,'_a.events',fname1)
        open(unit=iunita,file=fname1,form='formatted',status='old')
        read(iunita,250)mx_of_evta
        iunitb=22
        call fk88strcat(prefnev,'_b.events',fname1)
        open(unit=iunitb,file=fname1,form='formatted',status='old')
        read(iunitb,250)mx_of_evtb
c
        call fk88strcat(prefnev,'.events',fname1)
        ioutput=30
        open(unit=ioutput,file=fname1,form='formatted',
     #       status='unknown')
c Write all the quantities which identify the run
        write(ioutput,801)
     #    ecm,xren,xfh,xrenmc,xfhmc,
     #    '--> CM energy, muR/mu0[NLO], muF/mu0[NLO], '//
     #    'muR/mu0[MC], muF/mu0[MC]'
        write(ioutput,802)iprdct,'--> 1396/97/1497/98=gamma/Z/W+/W-'
        if(iprdct.ne.1396)then
          write(ioutput,813)xm0,gah,gammax,'--> M_V, Ga_V, GammaX'
        else
          write(ioutput,803)xmlow,xmupp,
     #      '--> M_gamma(min), M_gamma(max)'
        endif
        write(ioutput,804)xmass(1),xmass(2),
     #                    xmass(3),xmass(4),
     #                    xmass(5),xmass(21),
     #                    '--> quark and gluon masses'
        write(ioutput,805)part1,part2,'--> colliding particles'
        write(ioutput,806)gname(1:8),idpdfset,
     #    '--> PDF group and id number'
        write(ioutput,807)xlam,scheme,'--> Lambda_5, scheme'
        write(ioutput,811)'P,M','--> Format of v3.1 and higher'
        write(ioutput,250)maxevt
        itot=maxevt
        do ii=1,maxevt
          call whichosb(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
          call retrieve_evensb(iunit,ii,dummy)
          call store_evensb(ioutput,xmone)
        enddo
        call crosschesb(itot,mx_of_evta,mx_of_evtb)
        close(iunita)
        close(iunitb)
        close(ioutput)
 111    continue
      endif
      if(basesoutput.gt.0) then
      if(iprdct.eq.1397)then
        write(*,*) '   '
        write(*,*)'Prefactors in MSbar and on-shell schemes'
        write(*,*)'MSbar is used in the present computation'
        write(*,*)'MSbar:    pi*aem(MZ)/(sw^2*cw^2) =',pfczmsb
        write(*,*)'On-shell: sqrt(2)*GF*MZ^2        =',pfczon
      elseif(iprdct.eq.1497.or.iprdct.eq.1498) then
        write(*,*) '   '
        write(*,*)'Prefactors in MSbar and on-shell schemes'
        write(*,*)'MSbar is used in the present computation'
        write(*,*)'MSbar:    pi*aem(MW)/sw^2 =',pfcwmsb
        write(*,*)'On-shell: sqrt(2)*GF*MW^2 =',pfcwon
      endif
      write(*,*) '   '
      write(*,*) 'Total for fully inclusive'
      write(*,200)ih1,ih2,ndns1,ndns2,nl,xlam
      write(*,201) 'tot'
      write(*,300)ecm,xm0,gah,xfh,xren,avtot,dtot
      endif
 200  format(' had1=',i2,'  had2=',i2,'  strf1=',i6,'  strf2=',i6,
     #  '  nl=',i2,'  lambda5=',d10.4)
 201  format(' ecm or ebeam  mass      width     xf   xr   ',a,
     # '        err    ')
 300  format((1x,1pd9.3),4x,2(1x,1pd9.3),2(1x,0pf4.2),
     # 2(1x,0pd10.4))
 250  format(1x,i8)
 801  format(5(1x,d10.4),1x,a)
 802  format(1x,i6,1x,a)
 803  format(2(1x,d10.4),1x,a)
 804  format(6(1x,d10.4),1x,a)
 805  format(2(1x,a4),1x,a)
 806  format(1x,a8,1x,i6,1x,a)
 807  format(1x,d10.4,1x,a2,1x,a)
 811  format(1x,a3,1x,a)
 813  format(3(1x,d10.4),1x,a)
      end


C      subroutine getset(str,ndns,ih)
C      implicit real * 8 (a-h,o-z)
C      character * (*) str
C 2    write(*,*) str
C      write(*,*)
C     # '   (< 0 for a display of the features of the various sets'
C      read(*,*) ndns
C      if(ndns.lt.0) then
C        call prntsf
C        go to 2
C      endif
C      end


C      subroutine toend(iunit)
C      ios = 0    
C      dowhile(ios.eq.0)
C         read(unit=iunit,fmt='(1x)',iostat=ios)
C      enddo                        
C      end


C      subroutine getihpart(part,ih)
Cc Converts particle naming conventions, for Herwig to MLM
C      implicit real * 8 (a-h,o-z)
C      character * 4 part
Cc
C      ih=-100
C      if(part.eq.'P   ')then
C        ih=1
C      elseif(part.eq.'PBAR')then
C        ih=-1
C      elseif(part.eq.'N   ')then
C        ih=2
C      elseif(part.eq.'NBAR')then
C        ih=-2
C      elseif(part.eq.'GAMA')then
C        ih=4
C      elseif(part.eq.'E-  ')then
C        ih=5
C      else
C        write(*,*)'Error in getihpart'
C        write(*,*)'No such particle in MLM:',part
C        stop
C      endif
C      return
C      end


      subroutine strfsb(x1,x2,sf)
c Return parton densities through the matrix
c  sf(idr,jproc,itype), with the following conventions:
c  idr=1 -> qqbar, qg
c  idr=2 -> 0    , qbarg
c  idr=3 -> qbarq, gq
c  idr=4 -> 0    , gqbar
c  jproc=2,3 -> qqbar, qg processes respectively
c  itype -> identifies the individual contribution to a given jproc
      implicit none
      real*4 fh1x1(-5:5),fh2x2(-5:5),smuf2h1,smuf2h2
      real * 8 pi,x1,x2,sf(4,2:3,6),coupling
      integer ih1,ih2,ndns1,ndns2,ii,jproc,itype,iprdct
      parameter(pi=3.14159265358979312D0)
      include 'svbcblks.h'
      common/strfsb0/ih1,ih2,ndns1,ndns2
      common/ciprdct/iprdct
      real*8 zel(1:6),zel2(1:6)
      real * 8 zqqcoup(1:6),ckm2(1:6,1:6)
      common/charges/zel,zel2
      common/czqqcoup/zqqcoup
      common/cckm2/ckm2
      integer ipdfscale
      common/cipdfscale/ipdfscale
c ipdfscale=1 --> use NLO factorization scale
c ipdfscale=2 --> use MC factorization scale
c
      do ii=1,4
        do jproc=2,3
          do itype=1,6
            sf(ii,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      if(ipdfscale.eq.1)then
        smuf2h1=sngl(xmuf2h1)
        smuf2h2=sngl(xmuf2h2)
      elseif(ipdfscale.eq.2)then
        smuf2h1=sngl(xmumcf2h1)
        smuf2h2=sngl(xmumcf2h2)
      else
        write(*,*)'Fatal error in strfsb: unknown ipdfscale',ipdfscale
        stop
      endif
c
      call mlmpdf(ndns1,ih1,smuf2h1,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smuf2h2,sngl(x2),fh2x2,5)
c
c jproc=2
      if(iprdct.eq.1396.or.iprdct.eq.1397)then
        do ii=1,nl
          if(iprdct.eq.1396)then
            coupling=ze2**2*zel2(ii)
          else
            coupling=ze2*zqqcoup(ii)
          endif
          sf(1,2,ii)=coupling * dble(fh1x1( ii) * fh2x2(-ii))
          sf(3,2,ii)=coupling * dble(fh1x1(-ii) * fh2x2( ii))
        enddo
      elseif(iprdct.eq.1497)then
        sf(1,2,1)=ze2*ckm2(1,2)*dble(fh1x1( 1) * fh2x2(-2)) 
        sf(1,2,2)=ze2*ckm2(1,3)*dble(fh1x1( 1) * fh2x2(-3)) 
        sf(1,2,3)=ze2*ckm2(1,5)*dble(fh1x1( 1) * fh2x2(-5)) 
        sf(1,2,4)=ze2*ckm2(4,2)*dble(fh1x1( 4) * fh2x2(-2)) 
        sf(1,2,5)=ze2*ckm2(4,3)*dble(fh1x1( 4) * fh2x2(-3)) 
        sf(1,2,6)=ze2*ckm2(4,5)*dble(fh1x1( 4) * fh2x2(-5)) 
        sf(3,2,1)=ze2*ckm2(1,2)*dble(fh1x1(-2) * fh2x2( 1)) 
        sf(3,2,2)=ze2*ckm2(1,3)*dble(fh1x1(-3) * fh2x2( 1)) 
        sf(3,2,3)=ze2*ckm2(1,5)*dble(fh1x1(-5) * fh2x2( 1)) 
        sf(3,2,4)=ze2*ckm2(4,2)*dble(fh1x1(-2) * fh2x2( 4)) 
        sf(3,2,5)=ze2*ckm2(4,3)*dble(fh1x1(-3) * fh2x2( 4)) 
        sf(3,2,6)=ze2*ckm2(4,5)*dble(fh1x1(-5) * fh2x2( 4)) 
      elseif(iprdct.eq.1498)then
        sf(1,2,1)=ze2*ckm2(1,2)*dble(fh1x1( 2) * fh2x2(-1)) 
        sf(1,2,2)=ze2*ckm2(4,2)*dble(fh1x1( 2) * fh2x2(-4)) 
        sf(1,2,3)=ze2*ckm2(1,3)*dble(fh1x1( 3) * fh2x2(-1)) 
        sf(1,2,4)=ze2*ckm2(4,3)*dble(fh1x1( 3) * fh2x2(-4)) 
        sf(1,2,5)=ze2*ckm2(1,5)*dble(fh1x1( 5) * fh2x2(-1)) 
        sf(1,2,6)=ze2*ckm2(4,5)*dble(fh1x1( 5) * fh2x2(-4)) 
        sf(3,2,1)=ze2*ckm2(1,2)*dble(fh1x1(-1) * fh2x2( 2)) 
        sf(3,2,2)=ze2*ckm2(4,2)*dble(fh1x1(-4) * fh2x2( 2)) 
        sf(3,2,3)=ze2*ckm2(1,3)*dble(fh1x1(-1) * fh2x2( 3)) 
        sf(3,2,4)=ze2*ckm2(4,3)*dble(fh1x1(-4) * fh2x2( 3)) 
        sf(3,2,5)=ze2*ckm2(1,5)*dble(fh1x1(-1) * fh2x2( 5)) 
        sf(3,2,6)=ze2*ckm2(4,5)*dble(fh1x1(-4) * fh2x2( 5)) 
      else
        write(*,*)'Error in strfsb: unknown IPROC',iprdct
        stop
      endif
c jproc=3
      if(iprdct.eq.1396.or.iprdct.eq.1397)then
        do ii=1,nl
          if(iprdct.eq.1396)then
            coupling=ze2**2*zel2(ii)
          else
            coupling=ze2*zqqcoup(ii)
          endif
          sf(1,3,ii)=coupling * dble(fh1x1( ii) * fh2x2(  0))
          sf(2,3,ii)=coupling * dble(fh1x1(-ii) * fh2x2(  0))
          sf(3,3,ii)=coupling * dble(fh1x1(  0) * fh2x2( ii))
          sf(4,3,ii)=coupling * dble(fh1x1(  0) * fh2x2(-ii))
        enddo
      elseif(iprdct.eq.1497)then
        sf(1,3,1)=ze2*ckm2(1,2)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,2)=ze2*ckm2(1,3)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,3)=ze2*ckm2(1,5)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,4)=ze2*ckm2(4,2)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(1,3,5)=ze2*ckm2(4,3)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(1,3,6)=ze2*ckm2(4,5)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(2,3,1)=ze2*ckm2(1,2)*dble(fh1x1(-2) * fh2x2( 0)) 
        sf(2,3,2)=ze2*ckm2(4,2)*dble(fh1x1(-2) * fh2x2( 0)) 
        sf(2,3,3)=ze2*ckm2(1,3)*dble(fh1x1(-3) * fh2x2( 0)) 
        sf(2,3,4)=ze2*ckm2(4,3)*dble(fh1x1(-3) * fh2x2( 0)) 
        sf(2,3,5)=ze2*ckm2(1,5)*dble(fh1x1(-5) * fh2x2( 0)) 
        sf(2,3,6)=ze2*ckm2(4,5)*dble(fh1x1(-5) * fh2x2( 0)) 
        sf(3,3,1)=ze2*ckm2(1,2)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,2)=ze2*ckm2(1,3)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,3)=ze2*ckm2(1,5)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,4)=ze2*ckm2(4,2)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(3,3,5)=ze2*ckm2(4,3)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(3,3,6)=ze2*ckm2(4,5)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(4,3,1)=ze2*ckm2(1,2)*dble(fh1x1( 0) * fh2x2(-2)) 
        sf(4,3,2)=ze2*ckm2(4,2)*dble(fh1x1( 0) * fh2x2(-2)) 
        sf(4,3,3)=ze2*ckm2(1,3)*dble(fh1x1( 0) * fh2x2(-3)) 
        sf(4,3,4)=ze2*ckm2(4,3)*dble(fh1x1( 0) * fh2x2(-3)) 
        sf(4,3,5)=ze2*ckm2(1,5)*dble(fh1x1( 0) * fh2x2(-5)) 
        sf(4,3,6)=ze2*ckm2(4,5)*dble(fh1x1( 0) * fh2x2(-5)) 
      elseif(iprdct.eq.1498)then
        sf(1,3,1)=ze2*ckm2(1,2)*dble(fh1x1( 2) * fh2x2( 0)) 
        sf(1,3,2)=ze2*ckm2(4,2)*dble(fh1x1( 2) * fh2x2( 0)) 
        sf(1,3,3)=ze2*ckm2(1,3)*dble(fh1x1( 3) * fh2x2( 0)) 
        sf(1,3,4)=ze2*ckm2(4,3)*dble(fh1x1( 3) * fh2x2( 0)) 
        sf(1,3,5)=ze2*ckm2(1,5)*dble(fh1x1( 5) * fh2x2( 0)) 
        sf(1,3,6)=ze2*ckm2(4,5)*dble(fh1x1( 5) * fh2x2( 0)) 
        sf(2,3,1)=ze2*ckm2(1,2)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,2)=ze2*ckm2(1,3)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,3)=ze2*ckm2(1,5)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,4)=ze2*ckm2(4,2)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(2,3,5)=ze2*ckm2(4,3)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(2,3,6)=ze2*ckm2(4,5)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(3,3,1)=ze2*ckm2(1,2)*dble(fh1x1( 0) * fh2x2( 2)) 
        sf(3,3,2)=ze2*ckm2(4,2)*dble(fh1x1( 0) * fh2x2( 2)) 
        sf(3,3,3)=ze2*ckm2(1,3)*dble(fh1x1( 0) * fh2x2( 3)) 
        sf(3,3,4)=ze2*ckm2(4,3)*dble(fh1x1( 0) * fh2x2( 3)) 
        sf(3,3,5)=ze2*ckm2(1,5)*dble(fh1x1( 0) * fh2x2( 5)) 
        sf(3,3,6)=ze2*ckm2(4,5)*dble(fh1x1( 0) * fh2x2( 5)) 
        sf(4,3,1)=ze2*ckm2(1,2)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,2)=ze2*ckm2(1,3)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,3)=ze2*ckm2(1,5)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,4)=ze2*ckm2(4,2)*dble(fh1x1( 0) * fh2x2(-4)) 
        sf(4,3,5)=ze2*ckm2(4,3)*dble(fh1x1( 0) * fh2x2(-4)) 
        sf(4,3,6)=ze2*ckm2(4,5)*dble(fh1x1( 0) * fh2x2(-4)) 
      else
        write(*,*)'Error in strfsb: unknown IPROC',iprdct
        stop
      endif
c
      return
      end

c
c
c Begin of cross section routines. In the case of boson finite width,
c the cross section is obtained with the formula
c       sigma=\int dM^2 BW(M,M0,Ga) sigma_zw(M^2)
c sigma_zw(M^2) being the cross section relevant to zero width computed
c at a mass value of M, and 
c       BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c is the normalized (i.e., with integral equal to 1) Breit-Wigner function,
c which reduces to delta(M^2-M0^2) for Ga --> 0.
c The integration range in M^2 is controlled by gammax, xmlow, and xmupp
c
c
      function ssb5azw(xx)
c H events
      implicit none
      real * 8 ssb5azw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(3)
      include 'svbcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,tsb5a
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi
c
      xjac = 1
      roh = xm2/sh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(2)**2
      xjac = xjac * xx(2) * 2
      xii = zzz*(1-roh)
      xjac = xjac * (1-roh)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(3)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
c
      yi   = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      tau = roh/(1-xii)
      omega = sqrt( (2-xii*(1+yi))/(2-xii*(1-yi)) )
      romxi = sqrt(1-xii)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(omega*romxi))
        ymin=  log(tau)/2 - log(omega/romxi)
      else
        xxa1 = (2-xii-yi*xii)/2.d0
        xxa2 = (2-xii+yi*xii)/2.d0
        xxc = (1-(1-xii)*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(1)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
      s = sh * tau
      ssb5azw = tsb5a(s,xii,yi,xjac)
      return
      end


      function ssb5afw(xx)
c H events
      implicit none
      real * 8 ssb5afw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(4)
      include 'svbcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 extrafact
      common/cextra/extrafact
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,xbwmasb2,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,tsb5a,xalfaem
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi, xx(4) --> xm2
c
      xjac = 1
c 
      if(iprdct.eq.1396)then
        ttt=xx(4)**2
        xjac=xjac*2*xx(4)
        xm2=xmlow2+(xmupp2-xmlow2)*ttt
        xjac=xjac*(xmupp2-xmlow2)
c Insert the coupling and propagator in the jacobian
        xjac=xjac*extrafact/xm2
      elseif(iprdct.eq.1397 .or.
     #       iprdct.eq.1497.or.iprdct.eq.1498)then
c The following change of variable includes the Breit-Wigner function
        xm2=xbwmasb2(xx(4))
        xjac=bwdelf*xjac
      else
        write(*,*)'Error in ssb5afw: unknown IPROC',iprdct
        stop
      endif
      roh=xm2/sh
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xm2)
      xicut=xicutss
      if(xicut.gt.(1-roh))xicut=1-roh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(2)**2
      xjac = xjac * xx(2) * 2
      xii = zzz*(1-roh)
      xjac = xjac * (1-roh)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(3)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
c
      yi   = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      tau = roh/(1-xii)
      omega = sqrt( (2-xii*(1+yi))/(2-xii*(1-yi)) )
      romxi = sqrt(1-xii)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(omega*romxi))
        ymin=  log(tau)/2 - log(omega/romxi)
      else
        xxa1 = (2-xii-yi*xii)/2.d0
        xxa2 = (2-xii+yi*xii)/2.d0
        xxc = (1-(1-xii)*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(1)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
      s = sh * tau
      ssb5afw = tsb5a(s,xii,yi,xjac)
      return
      end


      function tsb5a(xs,xxii,xyi,xjac)
      implicit none
      real * 8 tsb5a,tsb5as,tsb5az,xs,xxii,xyi,s,xii,yi,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      s = xs
      xii = xxii
      yi = xyi
      if(isubttype.eq.0)then
        tmp=tsb5as(s,xii,yi,xjac)
      elseif(isubttype.eq.1)then
        tmp=tsb5az(s,xii,yi,xjac)
      else
        write(*,*)'Fatal error in tsb5a:',isubttype
        stop
      endif
      tsb5a=tmp
      return
      end


      function tsb5as(xs,xxii,xyi,xjac)
c H events
      implicit none
      real * 8 tsb5as,xs,xxii,xyi,xjac
      real * 8 pi,pi2,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      include 'svbcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 sf(4,2:3,6),vv(4,2:3,6),vvs(4,2:3,6)
      real * 8 res(4),xmcsec(4)
      common/cvv/vv
      common/cvvs/vvs
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 s,xii,yi,x,sx,xphspb,xphsp_ns,xphsp_s,x1,x2,t,u,
     # xnorm,zg2_nlo,zgmu2_nsb,zg2_mc,zgmu2_sb,
     # ffunval5,ffunction5,ytmp,zhwfct,zherw_spl,x1t,x1soft,
     # x2t,x2soft,x1x2j,x1x2jac,zherw_smn,xsum,dummy,xint,xitmp
      integer jproc,loproc,maproc,jproc0,ipdfscale,i,itype,
     # iret,i2b,itoosoftkin
      common/cjproc/jproc0
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      xii = xxii
      yi = xyi
      x = 1-xii
      sx = x*s
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphspb = xjac*2*pi/(x*sh)
      xphsp_ns = xjac * 2*pi/(64*pi2) * xm2/(x**2*sh)
      xphsp_s = xjac * 2*pi/(64*pi2) * xm2/(x*sh)
c
      do jproc=2,3
        do i=1,4
          do itype=1,6
            vv(i,jproc,itype)=0.d0
            vvs(i,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(x1.lt.1.and.x2.lt.1)then
        call invsb(xm2,s,xii,yi,t,u)
        xnorm = xphsp_ns/xii*( 1/(1-yi) + 1/(1+yi) )
        zg2_nlo = zgmu2_nsb()
        ipdfscale=1
        call strfsb(x1,x2,sf)
        do jproc=loproc,maproc
          jproc0=jproc
          call f2bosb(s,xii,yi,t,u,res)
          do i=1,4
            do itype=1,6
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*res(i)*zg2_nlo*xnorm
            enddo
          enddo
        enddo
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
        ffunval5 = ffunction5(x,yi)
        if(ffunval5.ne.0.d0)then
          ipdfscale=2
          zg2_mc = zgmu2_sb()
          ytmp=1.d0
          zhwfct=zherw_spl(x,yi)
          x1t=x1soft(x1,x2,x,yi)/zhwfct
          x2t=x2soft(x1,x2,x,yi)
          if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            call strfsb(x1t,x2t,sf)
            do jproc=loproc,maproc
              jproc0=jproc
              call xmcsusb(jproc,s,xii,yi,ytmp,xmcsec)
              do i=1,4
                do itype=1,6
                  vv(i,jproc,itype)=vv(i,jproc,itype)-
     #              sf(i,jproc,itype)*xmcsec(i)*x1x2j*zg2_mc*xnorm
                enddo
              enddo
            enddo
          endif
          ytmp=-1.d0
          zhwfct=zherw_smn(x,yi)
          x1t=x1soft(x1,x2,x,yi)
          x2t=x2soft(x1,x2,x,yi)/zhwfct
          if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            call strfsb(x1t,x2t,sf)
            do jproc=loproc,maproc
              jproc0=jproc
              call xmcsusb(jproc,s,xii,yi,ytmp,xmcsec)
              do i=1,4
                do itype=1,6
                  vv(i,jproc,itype)=vv(i,jproc,itype)-
     #              sf(i,jproc,itype)*xmcsec(i)*x1x2j*zg2_mc*xnorm
                enddo
              enddo
            enddo
          endif
        endif
      endif
c
      call checksb(xsum,dummy,iret)
      if(iret.eq.1)then
        call invsb(xm2,s,xii,yi,t,u)
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkin()
        if(i2b.eq.1)then
          xitmp=0.d0
          ytmp=1.d0
          call invsb(xm2,sx,xitmp,ytmp,t,u)
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tsb5as=abs(xint)
c
      return
      end


      function tsb5az(xs,xxii,xyi,xjac)
      implicit real*8 (a-z)
      tsb5az=0
      return
      end


      function ssb5bzw(xx)
c S events
      implicit none
      real * 8 ssb5bzw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(3)
      include 'svbcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,tsb5b
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi
c
      xjac = 1
      roh = xm2/sh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(2)**2
      xjac = xjac * xx(2) * 2
      xii = zzz*(1-roh)
      xjac = xjac * (1-roh)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(3)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
c
      yi   = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      tau = roh/(1-xii)
      omega = sqrt( (2-xii*(1+yi))/(2-xii*(1-yi)) )
      romxi = sqrt(1-xii)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(omega*romxi))
        ymin=  log(tau)/2 - log(omega/romxi)
      else
        xxa1 = (2-xii-yi*xii)/2.d0
        xxa2 = (2-xii+yi*xii)/2.d0
        xxc = (1-(1-xii)*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(1)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
      s = sh * tau
      ssb5bzw = tsb5b(s,xii,yi,xjac)
      return
      end


      function ssb5bfw(xx)
c S events
      implicit none
      real * 8 ssb5bfw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(4)
      include 'svbcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 extrafact
      common/cextra/extrafact
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      integer ifxdaem
      common/cifxdaem/ifxdaem
      integer iprdct
      common/ciprdct/iprdct
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,xbwmasb2,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,tsb5b,xalfaem
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi, xx(4) --> xm2
c
      xjac = 1
c 
      if(iprdct.eq.1396)then
        ttt=xx(4)**2
        xjac=xjac*2*xx(4)
        xm2=xmlow2+(xmupp2-xmlow2)*ttt
        xjac=xjac*(xmupp2-xmlow2)
c Insert the coupling and propagator in the jacobian
        xjac=xjac*extrafact/xm2
      elseif(iprdct.eq.1397 .or.
     #       iprdct.eq.1497.or.iprdct.eq.1498)then
c The following change of variable includes the Breit-Wigner function
        xm2=xbwmasb2(xx(4))
        xjac=bwdelf*xjac
      else
        write(*,*)'Error in ssb5bfw: unknown IPROC',iprdct
        stop
      endif
      roh=xm2/sh
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xm2)
      xicut=xicutss
      if(xicut.gt.(1-roh))xicut=1-roh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(2)**2
      xjac = xjac * xx(2) * 2
      xii = zzz*(1-roh)
      xjac = xjac * (1-roh)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(3)
      xjac = xjac * 2
      ttt = tiny+(1-tiny)*zzz**2
      xjac = xjac * 2 * abs(zzz)
      if(zzz.gt.0) then
         th = ttt * pi/2
      else
         th = pi-ttt*pi/2
      endif
      xjac = xjac * pi/2
c
      yi   = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      tau = roh/(1-xii)
      omega = sqrt( (2-xii*(1+yi))/(2-xii*(1-yi)) )
      romxi = sqrt(1-xii)
      if(iprespl.eq.0)then
        ymax= -log(tau)/2 + log(1/(omega*romxi))
        ymin=  log(tau)/2 - log(omega/romxi)
      else
        xxa1 = (2-xii-yi*xii)/2.d0
        xxa2 = (2-xii+yi*xii)/2.d0
        xxc = (1-(1-xii)*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(1)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
      s = sh * tau
      ssb5bfw = tsb5b(s,xii,yi,xjac)
      return
      end


      function tsb5b(xs,xxii,xyi,xjac)
      implicit none
      real * 8 tsb5b,tsb5bs,tsb5bz,xs,xxii,xyi,s,xii,yi,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      s = xs
      xii = xxii
      yi = xyi
      if(isubttype.eq.0)then
        tmp=tsb5bs(s,xii,yi,xjac)
      elseif(isubttype.eq.1)then
        tmp=tsb5bz(s,xii,yi,xjac)
      else
        write(*,*)'Fatal error in tsb5b:',isubttype
        stop
      endif
      tsb5b=tmp
      return
      end


      function tsb5bs(xs,xxii,xyi,xjac)
c S events
      implicit none
      real * 8 tsb5bs,xs,xxii,xyi,xjac
      real * 8 pi,pi2,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      include 'svbcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 sf(4,2:3,6),vv(4,2:3,6),vvs(4,2:3,6)
      real * 8 res(4),resc(4),xmcsec(4)
      common/cvv/vv
      common/cvvs/vvs
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 xints(4,2:3),xborn(4,2:3),xcs(4,2:3),xsv(4,2:3)
      real * 8 s,xii,yi,x,sx,xphspb,xphsp_ns,xphsp_s,x1,x2,t,u,
     # xnorm,zg2_nlo,zgmu2_nsb,zg2_mc,zgmu2_sb,
     # ffunval5,ffunction5,ytmp,zhwfct,zherw_spl,x1t,x1soft,
     # x2t,x2soft,x1x2j,x1x2jac,zherw_smn,xsum,dummy,xint,xitmp,
     # xintcp,xlmude,xnormc,xcplus,xintcm,xcminus,xintffs,xnormb,
     # xnormsv,xintcps,xintcms
      integer jproc,loproc,maproc,jproc0,ipdfscale,i,itype,
     # iret
      common/cjproc/jproc0
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      xii = xxii
      yi = xyi
      x = 1-xii
      sx = x*s
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphspb = xjac*2*pi/(x*sh)
      xphsp_ns = xjac * 2*pi/(64*pi2) * xm2/(x**2*sh)
      xphsp_s = xjac * 2*pi/(64*pi2) * xm2/(x*sh)
c
      do jproc=2,3
        do i=1,4
          do itype=1,6
            vv(i,jproc,itype)=0.d0
            vvs(i,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
      ffunval5 = ffunction5(x,yi)
      if(x1.lt.1.and.x2.lt.1.and.ffunval5.ne.0.d0)then
        call invsb(xm2,s,xii,yi,t,u)
        xnorm = xphsp_ns/xii*( 1/(1-yi) + 1/(1+yi) )
        ipdfscale=2
        zg2_mc = zgmu2_sb()
        ytmp=1.d0
        zhwfct=zherw_spl(x,yi)
        x1t=x1soft(x1,x2,x,yi)/zhwfct
        x2t=x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          call strfsb(x1t,x2t,sf)
          do jproc=loproc,maproc
            jproc0=jproc
            call xmcsusb(jproc,s,xii,yi,ytmp,xmcsec)
            do i=1,4
              do itype=1,6
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*xmcsec(i)*x1x2j*zg2_mc*xnorm
              enddo
            enddo
          enddo
        endif
        ytmp=-1.d0
        zhwfct=zherw_smn(x,yi)
        x1t=x1soft(x1,x2,x,yi)
        x2t=x2soft(x1,x2,x,yi)/zhwfct
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          call strfsb(x1t,x2t,sf)
          do jproc=loproc,maproc
            jproc0=jproc
            call xmcsusb(jproc,s,xii,yi,ytmp,xmcsec)
            do i=1,4
              do itype=1,6
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*xmcsec(i)*x1x2j*zg2_mc*xnorm
              enddo
            enddo
          enddo
        endif
      endif
c
c Counter-event :
c
      ipdfscale=1
      if(yi.gt.1-delta) then
         ytmp = 1.d0
         x1t = x1soft(x1,x2,x,yi)/x
         x2t = x2soft(x1,x2,x,yi)
         if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jac(x1,x2,x,yi)/x
            call invsb(xm2,s,xii,ytmp,t,u) 
            zg2_nlo = zgmu2_nsb()
            call strfsb(x1t,x2t,sf)
            xnorm = x1x2j * zg2_nlo * xphsp_ns
            do jproc=loproc,maproc
              jproc0=jproc
              call f2bosb(s,xii,ytmp,t,u,res)
              xintcp = xnorm * 1.d0/xii*( - 1/(1-yi) )
c Adding the collinear contribution
              xlmude = log(s/xmuf2h1)+log(delta/2)
              xnormc = x1x2j * zg2_nlo * xphspb /
     #                 (8*pi2 * delta)
              xcplus = xnormc / xii
              call f2b_cosb(s,xii,xii,ytmp,xlmude,resc)
              do i=1,4
                do itype=1,6
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*(xintcp*res(i)+xcplus*resc(i))
                enddo
              enddo
            enddo
         endif
      endif
c
      if(yi.lt.-1+delta) then
         ytmp = -1.d0
         x1t = x1soft(x1,x2,x,yi)
         x2t = x2soft(x1,x2,x,yi)/x
         if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jac(x1,x2,x,yi)/x
            call invsb(xm2,s,xii,ytmp,t,u) 
            zg2_nlo = zgmu2_nsb()
            call strfsb(x1t,x2t,sf)
            xnorm = x1x2j * zg2_nlo * xphsp_ns
            do jproc=loproc,maproc
              jproc0=jproc
              call f2bosb(s,xii,ytmp,t,u,res)
              xintcm = xnorm * 1.d0/xii*( - 1/(1+yi) )
c Adding the collinear contribution
              xlmude = log(s/xmuf2h2)+log(delta/2)
              xnormc = x1x2j * zg2_nlo * xphspb /
     #                 (8*pi2 * delta)
              xcminus = xnormc / xii
              call f2b_cosb(s,xii,xii,ytmp,xlmude,resc)
              do i=1,4
                do itype=1,6
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*(xintcm*res(i)+xcminus*resc(i))
                enddo
              enddo
            enddo
         endif
      endif
c
c     Soft part of the counter-event:
c
      if(xii.lt.xicut) then
         xitmp = 0.d0
         x1t = x1soft(x1,x2,x,yi)
         x2t = x2soft(x1,x2,x,yi)
         if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jac(x1,x2,x,yi)
            call invsb(xm2,sx,xitmp,yi,t,u)
            zg2_nlo = zgmu2_nsb()
            xnorm = x1x2j * zg2_nlo * xphsp_s
            do jproc=loproc,maproc
              jproc0=jproc
              do i=1,4
                xints(i,jproc) = 0.d0
                xborn(i,jproc) = 0.d0
                xcs(i,jproc) = 0.d0
                xsv(i,jproc) = 0.d0
              enddo
              call f2bosb(sx,xitmp,yi,t,u,res)
              xintffs = - xnorm * 1.d0/xii*
     #                  ( 1/(1-yi) + 1/(1+yi) )
              do i=1,4
                xints(i,jproc) = xintffs*res(i)
              enddo
c Adding the Born term
              xnormb = x1x2j * xphspb /(2*xicut)
              call f1sbrn(sx,jproc0,'bo',res)
              do i=1,4
                xborn(i,jproc) = xnormb*res(i)
              enddo
c Adding the soft-virtual contribution
              xnormsv = x1x2j * zg2_nlo * xphspb / 
     #                 (8*pi2 * 2*xicut)
              call fosb(sx,jproc0,res)
              do i=1,4
                xsv(i,jproc) = xnormsv*res(i)
              enddo
            enddo
            if(yi.gt.1-delta) then
               ytmp = 1.d0
               call invsb(xm2,sx,xitmp,ytmp,t,u)
               zg2_nlo = zgmu2_nsb()
               xnorm = x1x2j * zg2_nlo * xphsp_s
               do jproc=loproc,maproc
                 jproc0=jproc
                 call f2bosb(sx,xitmp,ytmp,t,u,res)
                 xintcps = - xnorm * 1.d0/xii*( - 1/(1-yi) )
                 do i=1,4
                   xints(i,jproc) = xints(i,jproc)+xintcps*res(i)
                 enddo
c Adding the collinear contribution
                 xlmude = log(sx/xmuf2h1)+log(delta/2)
                 xnormc = x1x2j * zg2_nlo * xphspb /
     #                    (8*pi2 * delta)
                 call f2b_cosb(sx,xii,xitmp,ytmp,xlmude,res)
                 do i=1,4
                   xcs(i,jproc) = xcs(i,jproc) - xnormc / xii * res(i)
                 enddo
               enddo
            endif
            if(yi.lt.-1+delta) then
               ytmp = -1.d0
               call invsb(xm2,sx,xitmp,ytmp,t,u)
               zg2_nlo = zgmu2_nsb()
               xnorm = x1x2j * zg2_nlo * xphsp_s
               do jproc=loproc,maproc
                 jproc0=jproc
                 call f2bosb(sx,xitmp,ytmp,t,u,res)
                 xintcms = - xnorm * 1.d0/xii*( - 1/(1+yi) )
                 do i=1,4
                   xints(i,jproc) = xints(i,jproc)+xintcms*res(i)
                 enddo
c Adding the collinear contribution
                 xlmude = log(sx/xmuf2h2)+log(delta/2)
                 xnormc = x1x2j * zg2_nlo * xphspb /
     #                    (8*pi2 * delta)
                 call f2b_cosb(sx,xii,xitmp,ytmp,xlmude,res)
                 do i=1,4
                   xcs(i,jproc) = xcs(i,jproc) - xnormc / xii * res(i)
                 enddo
               enddo
            endif
            call strfsb(x1t,x2t,sf)
            do jproc=loproc,maproc
              jproc0=jproc
              do i=1,4
                do itype=1,6
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*( xints(i,jproc)+xsv(i,jproc)+
     #                                  xborn(i,jproc)+xcs(i,jproc) )
                enddo
              enddo
            enddo
         endif
      endif
c
      call checksb(xsum,dummy,iret)
      if(iret.eq.1)then
        xitmp = 0.d0
        ytmp = 1.d0
        call invsb(xm2,sx,xitmp,ytmp,t,u)
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)
        ycm = 0.5d0*log(x1t/x2t)
        tau=x*tau
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tsb5bs=abs(xint)
c
      return
      end


      function tsb5bz(xs,xxii,xyi,xjac)
      implicit real*8 (a-z)
      tsb5bz=0
      return
      end
c
c
c End of cross section routines
c
c
c
c
c Begin of utility functions for zeta subtraction
c
c
C      function svn(ro)
C      implicit none
C      real*8 svn,ro,tmp,be4,ybar,etacut
C      common/cetacut/etacut
Cc
C      tmp=0.d0
C      if(ro.lt.1.d0-sqrt(etacut))then
C        be4=(1-ro)**2
C        ybar=sqrt(1-etacut/be4)
C        tmp=-(1-ro)*ybar+sqrt(etacut)*asin(ybar)
C      endif
C      svn=tmp
C      return
C      end


C      function f1fun(ro)
C      implicit real * 8 (a-z)
C      common/cetacut/etacut
Cc
C      tmp=0.d0
C      if(ro.lt.1.d0-sqrt(etacut))then
C        be4=(1-ro)**2
C        ybar=sqrt(1-etacut/be4)
C        tmp=log((1+ybar)/(1-ybar))*( log(etacut/be4)
C     #        -log(1-ybar**2)/2.d0-log(2.d0) )
C     #     +ddilog((1+ybar)/2.d0)-ddilog((1-ybar)/2.d0) 
C        tmp=tmp/4.d0
C      endif
C      f1fun=tmp
C      return
C      end


C      function bdelta(x)
C      implicit none
C      real*8 bdelta,x,tmp,etacut
C      common/cetacut/etacut
Cc
C      tmp=0.d0
C      if(x.lt.1.d0-dsqrt(etacut))tmp=sqrt(1-etacut/(1-x)**2)
C      bdelta=tmp
C      return
C      end
c
c
c End of utility functions for zeta subtraction
c
c
c
c
c Begin of event-generation routines
c
c
      subroutine sprfsb()
c This routine is called by run_spring; the entry is dummy, all the 
c parameters must be passed through common blocks
      implicit none
      integer iunit
      parameter (iunit=22)
      real*8 xone
      parameter (xone=1.d0)
      real*8 ycm,tau
      common/x1x2/ycm,tau
      integer i0,jproc0,itype0
      common/cidproc/i0,jproc0,itype0
      integer iret
      real*8 ycm0
c
      call xosb(iret)
      if(iret.eq.1)then
        if(i0.lt.1.or.i0.gt.4)then
          write(*,*)'Fatal error in sprfsb'
          stop
        endif
        ycm0=ycm
        call getx1x2(tau,ycm0)
        call getmsb(tau,ycm0)
        call store_evensb(iunit,xone)
      endif
      return
      end


C      subroutine getx1x2(tau,ycm)
C      implicit none
C      real*8 tau,ycm,x1,x2,stau,ey
C      common/cx1x2/x1,x2
Cc
C      stau=sqrt(tau)
C      ey=exp(ycm)
C      x1=stau*ey
C      x2=stau/ey
C      return
C      end


      subroutine getmsb(xtau,xycm)
      implicit none
      real*8 xtau,xycm
      include 'svbcblks.h'
      real*8 pi
      parameter (pi=3.14159265358979312D0)
      integer i
      real*8 tau,ycm,theta,cth,sth,fk88random,sqsh,ycmnew
      real*8 x1,x2
      common/cx1x2/x1,x2
      real*8 xmom_cm(4,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_lb(4,4)
      common/cxmomlb/xmom_lb
      real*8 xmom_prime(4,4)
      common/cxmomprime/xmom_prime
      integer ionshell
      common/cionshell/ionshell
      integer ichkmom
      common/cichkmom/ichkmom
      integer ifk88seed
      common/cifk88seed/ifk88seed
c
      tau=xtau
      ycm=xycm
      call getx1x2(tau,ycm)
c perform a random rotation in the transverse plane
      theta=2*pi*fk88random(ifk88seed)
      cth=cos(theta)
      sth=sin(theta)
      do i=3,4
        call transrot(cth,sth,xmom_cm(i,1),xmom_cm(i,2))
      enddo
      if(ichkmom.eq.0)call checkmsb(xmom_cm,sh,0.d0,3)
c determine colour connections
c      call getcolconn()
c put partons on Herwig mass shell
      if(ionshell.eq.0)then
c keep the parton massless
        sqsh=sqrt(sh)
        xmom_lb(1,1)=0.d0
        xmom_lb(1,2)=0.d0
        xmom_lb(1,3)=x1*sqsh/2.d0
        xmom_lb(1,4)=x1*sqsh/2.d0
        xmom_lb(2,1)=0.d0
        xmom_lb(2,2)=0.d0
        xmom_lb(2,3)=-x2*sqsh/2.d0
        xmom_lb(2,4)=x2*sqsh/2.d0
        do i=3,4
          call boost(-ycm,
     #         xmom_cm(i,1),xmom_cm(i,2),
     #         xmom_cm(i,3),xmom_cm(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      else
c put the partons on Herwig mass shell
        call put_on_shesb(ycm,ycmnew)
        do i=1,4
          call boost(-ycmnew,
     #         xmom_prime(i,1),xmom_prime(i,2),
     #         xmom_prime(i,3),xmom_prime(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      endif
      if(ichkmom.eq.0)call checkmsb(xmom_lb,sh,-ycmnew,2)
      call momnewformsb()
      return
      end


      subroutine momnewformsb()
c Replaces the energy with the mass in the fourth component of xmom_lb,
c to comply with the new format of the event file. Must be called as the
c last step before storing events on the temporary event files.
c If the energy is zero, the fourth component is left unchanged,
c since the LH interface uses it to distinguish between S and H events.
      implicit none
      real*8 xmom_lb(4,4)
      common/cxmomlb/xmom_lb
      real*8 xmss(1:4)
      common/procmass/xmss
      integer i
c
      do i=1,4
        if(xmom_lb(i,4).ne.0.d0)xmom_lb(i,4)=xmss(i)
      enddo
      return
      end


C      subroutine boost(y,a1,a2,a3,a4,b1,b2,b3,b4)
C      implicit none
C      real*8 y,a1,a2,a3,a4,b1,b2,b3,b4
Cc
C      b1=a1
C      b2=a2
C      b3=a3*cosh(y)-a4*sinh(y)
C      b4=a4*cosh(y)-a3*sinh(y)
C      return
C      end


C      subroutine transrot(cth,sth,xpt1,xpt2)
C      implicit none
C      real*8 cth,sth,xpt1,xpt2,pt1,pt2
Cc
C      pt1=xpt1
C      pt2=xpt2
C      xpt1=pt1*cth+pt2*sth
C      xpt2=-pt1*sth+pt2*cth
C      return
C      end


      subroutine put_on_shesb(ycm,ycmnew)
      implicit none
      include 'svbcblks.h'
      integer i1b,i,j
      real*8 xmss(1:4)
      real*8 ycm,ycmnew,shat,delta_thrs,xpt2,efin,xkp2prime_norm2,
     # xkplon2,xkplon
      parameter (delta_thrs=0.5d-3)
      common/procmass/xmss
      real*8 xmass(-5:21)
      common/parmass/xmass
c x1 and x2 are the Bjorken variables; x1 is relevant to the parton
c coming from the left
      real*8 x1,x2
      common/cx1x2/x1,x2
c xmom_cm(i,j) is the j component of the four vector of the particle # i,
c given in the partonic CM frame. j=4 is the energy. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is the Higgs.
c Momentum conservation is (1+2)-(3+4)=0.
      real*8 xmom_cm(4,4)
      common/cxmomcm/xmom_cm
c new momenta (put on shell) are stored here
      real*8 xmom_prime(4,4)
      common/cxmomprime/xmom_prime
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ionshell
      common/cionshell/ionshell
      integer ichkmom
      common/cichkmom/ichkmom
c 
      if(ionshell.ne.2)then
        write(6,*)'This option is not supported for this process'
        stop
      endif
      ycmnew = ycm
      xmss(1) = xmass(ip1)
      xmss(2) = xmass(ip2)
      xmss(3) = xmass(ip3)
      xmss(4) = sqrt(xm2)
      shat=(xmom_cm(1,4)+xmom_cm(2,4))**2
c i1b=0 --> 2-body kinematics; i1b=1 --> 1-body kinematics
      i1b = 0
      if(xmom_cm(3,4).lt.1.d-14)i1b=1
      if(i1b.eq.0)then
c Since transverse momenta are kept fixed, check whether shat is larger
c than the transverse mass. If not, enlarge it
        xpt2=xmom_cm(3,1)**2+xmom_cm(3,2)**2
        efin=sqrt(xmss(3)**2+xpt2)+
     #       sqrt(xmss(4)**2+xpt2)
        if((efin+delta_thrs).gt.sqrt(shat))then
          shat=(efin+delta_thrs)**2
          xmom_cm(1,3)=sqrt(shat)/2.d0
          xmom_cm(2,3)=-xmom_cm(1,3)
        endif
      endif
      do i=1,4
        do j=1,2
          xmom_prime(i,j)=xmom_cm(i,j)
        enddo
      enddo
      call getxmsb(shat,ycm,
     #             xmom_cm(1,3),xmss(1),
     #             xmom_cm(2,3),xmss(2),
     #             xmom_prime(1,3),xmom_prime(2,3))
      if(i1b.eq.0)then
        xkp2prime_norm2=( shat-2*(xmss(3)**2+xmss(4)**2)+
     #                    (xmss(3)**2-xmss(4)**2)**2/shat )/4.d0
        xkplon2=xkp2prime_norm2-xmom_prime(3,1)**2-xmom_prime(3,2)**2
        xkplon=sqrt(xkplon2)
        xmom_prime(3,3)=sign(1.d0,xmom_cm(3,3))*xkplon
        xmom_prime(4,3)=sign(1.d0,xmom_cm(4,3))*xkplon
      else
        xmom_prime(3,3)=0.d0
        xmom_prime(4,3)=0.d0
      endif
      do i=1,4
        if(i.eq.3.and.i1b.eq.1)then
          xmom_prime(i,4)=0.d0
        else
          call getenergy(xmom_prime(i,1),xmom_prime(i,2),
     #                   xmom_prime(i,3),xmss(i),xmom_prime(i,4))
        endif
      enddo
      if(ichkmom.eq.0)call checkmsb(xmom_prime,shat,0.d0,4)
      return
      end


      subroutine getxmsb(shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23)
c After putting the momenta on shell, the two incoming partons may
c travel in the same direction. This routine prevents this to happen,
c redefining Herwig masses if necessary
      implicit none
      real*8 shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23
      real*8 tiny,fact,sqs,xm1s,xm2s,xkp2prime_norm2,xkp2prime_norm,
     #  ytmp,e1,e2,p13p,p23p,s1p,s2p,xif,sol
      integer iflag,idone,ileg
      parameter (fact=0.98d0)
      parameter (tiny=1.d-6)
c
      sqs=sqrt(shat)
      xm1s=xm1
      xm2s=xm2
      ytmp=-ycm
      idone=0
 100  continue
      xkp2prime_norm2=( shat-2*(xm1**2+xm2**2)+
     #                  (xm1**2-xm2**2)**2/shat )/4.d0
      xkp2prime_norm=sqrt(xkp2prime_norm2)
      if(sign(1.d0,p13cm).ne.1.d0.or.sign(1.d0,p23cm).ne.-1.d0)then
        write(*,*)'Error # 0 in getxmsb'
        stop
      endif
      p13=xkp2prime_norm
      p23=-xkp2prime_norm
      e1=sqrt(p13**2+xm1**2)
      e2=sqrt(p23**2+xm2**2)
      p13p=p13*cosh(ytmp)-e1*sinh(ytmp)
      p23p=p23*cosh(ytmp)-e2*sinh(ytmp)
      s1p=sign(1.d0,p13p)
      s2p=sign(1.d0,p23p)
      iflag=0
      if(s1p.eq.1.d0 .and. s2p.eq.-1.d0)then
        iflag=1
      elseif(s1p.eq.-1.d0 .and. s2p.eq.-1.d0)then
        if(ytmp.lt.0.d0)then
          write(*,*)'Wrong y sign, # 1'
          stop
        endif
        ileg=1
        xif=xm2**2/shat
      elseif(s1p.eq.1.d0 .and. s2p.eq.1.d0)then
        if(ytmp.gt.0.d0)then
          write(*,*)'Wrong y sign, # 2'
          stop
        endif
        ileg=2
        xif=xm1**2/shat
      else
        write(*,*)'Error # 1 in getxmsb'
        stop
      endif
      if(iflag.eq.0)then
        sol=xif+cosh(2*ytmp)-
     #      sqrt(2.d0)*cosh(ytmp)*sqrt(cosh(2*ytmp)-1+2*xif)
        if(sol.le.0.d0.or.idone.eq.1)then
c The procedure failed; pass the massless event to Herwig, and let Herwig
c deal with it
          xm1=0.d0
          xm2=0.d0
          p13=sqs/2.d0
          p23=-sqs/2.d0
          return
        endif
        if(ileg.eq.1)then
          xm1=fact*sqrt(sol*shat)
          if(xm1.gt.xm1s)then
            write(*,*)'Mass # 1 too large in getxmsb'
            stop
          endif
        elseif(ileg.eq.2)then
          xm2=fact*sqrt(sol*shat)
          if(xm2.gt.xm2s)then
            write(*,*)'Mass # 2 too large in getxmsb'
            stop
          endif
        else
          write(*,*)'Error # 2 in getxmsb'
          stop
        endif
        idone=1
        goto 100
      endif
      return
      end


C      subroutine fillvec(p1,p2,p3,p4,ytmp)
C      implicit none
C      real*8 p1,p2,p3,p4,ytmp(1:4)
Cc
C      ytmp(1)=p1
C      ytmp(2)=p2
C      ytmp(3)=p3
C      ytmp(4)=p4
C      return
C      end


C      subroutine getvec(ytmp,p1,p2,p3,p4)
C      implicit none
C      real*8 ytmp(1:4),p1,p2,p3,p4
Cc
C      p1=ytmp(1)
C      p2=ytmp(2)
C      p3=ytmp(3)
C      p4=ytmp(4)
C      return
C      end

Cc-----------------------------------------------------------------------
C      subroutine xhwulob(ps,ps5,pi,pi5,pf,pf5)
Cc     transforms pi (given in rest frame of ps) into pf (in lab)
Cc     n.b. p(1,2,3,4,5) = (px,py,pz,e,m)
Cc-----------------------------------------------------------------------
C      real*8 pf4,fn,ps(4),ps5,pi(4),pi5,pf(4),pf5
C      if (ps(4).eq.ps5) then
C        pf(1)= pi(1)
C        pf(2)= pi(2)
C        pf(3)= pi(3)
C        pf(4)= pi(4)
C      else
C        pf4  = (pi(1)*ps(1)+pi(2)*ps(2)
C     &         +pi(3)*ps(3)+pi(4)*ps(4))/ps5
C        fn   = (pf4+pi(4)) / (ps(4)+ps5)
C        pf(1)= pi(1) + fn*ps(1)
C        pf(2)= pi(2) + fn*ps(2)
C        pf(3)= pi(3) + fn*ps(3)
C        pf(4)= pf4
C      end if
C      pf5= pi5
C      end

Cc-----------------------------------------------------------------------
C      subroutine xhwulof(ps,ps5,pi,pi5,pf,pf5)
Cc     transforms pi (given in lab) into pf (in rest frame of ps)
Cc     n.b. p(1,2,3,4,5) = (px,py,pz,e,m)
Cc-----------------------------------------------------------------------
C      real*8 pf4,fn,ps(4),ps5,pi(4),pi5,pf(4),pf5
C      if (ps(4).eq.ps5) then
C        pf(1)= pi(1)
C        pf(2)= pi(2)
C        pf(3)= pi(3)
C        pf(4)= pi(4)
C      else
C        pf4  = (pi(4)*ps(4)-pi(3)*ps(3)
C     &         -pi(2)*ps(2)-pi(1)*ps(1))/ps5
C        fn   = (pf4+pi(4)) / (ps(4)+ps5)
C        pf(1)= pi(1) - fn*ps(1)
C        pf(2)= pi(2) - fn*ps(2)
C        pf(3)= pi(3) - fn*ps(3)
C        pf(4)= pf4
C      end if
C      pf5= pi5
C      end


C      subroutine getenergy(p1,p2,p3,xm,en)
C      implicit none
C      real*8 p1,p2,p3,xm,en
Cc
C      en=sqrt(p1**2+p2**2+p3**2+xm**2)
C      return
C      end


C      function dotprod(p1,p2,p3,p4,q1,q2,q3,q4)
C      implicit none
C      real*8 dotprod,p1,p2,p3,p4,q1,q2,q3,q4
Cc
C      dotprod=p4*q4-p1*q1-p2*q2-p3*q3
C      return
C      end


C      function xnorm_3(p)
Cc Evaluates the norm of the spatial component of a four-momentum
Cc The result is positive by definition, regardless of the 4-metric
C      implicit none
C      real*8 xnorm_3,p(1:4),tmp
Cc
C      tmp=p(1)*p(1)+p(2)*p(2)+p(3)*p(3)
C      xnorm_3=sqrt(tmp)
C      return
C      end


C      subroutine vecsum(p,pfact,q,qfact,r)
Cc Weighted sum of the four-vectors p and q. The result is r
C      implicit none
C      integer i
C      real*8 p(1:4),q(1:4),r(1:4),pfact,qfact
Cc
C      do i=1,4
C        r(i)=pfact*p(i)+qfact*q(i)
C      enddo
C      return
C      end


      subroutine xosb(iret)
c This routine is called by sprfsb; it determines, on statistical
c basis, which partonic process has been generated.
c It also counts the number of unlike sign events (iwrong), and the number
c of these events (iwrong1) for which the relative difference between
c unlike signs exceeds 5%. If all the entries of vv are equal to zero,
c iret is set equal to 0 (by checksb), and no operation is performed
      implicit none
      integer iret,iretvv,iretvvs,iproc,iproclo,iprocma,i,itype,
     #  iwh,iflag,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o
      real*8 wwx(4,2:3,6),xsum,xsumabs,xsumvvs,xsumabsvvs,xstsign,
     #  xg,wh,rmax,fk88random
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      integer ifuntype
      common/cifuntype/ifuntype
      real*8 vv(4,2:3,6)
      common/cvv/vv
      real*8 vvs(4,2:3,6)
      common/cvvs/vvs
      integer iwrong,iwrong1
      common/ciwrong/iwrong,iwrong1
      integer i0,jproc0,itype0
      common/cidproc/i0,jproc0,itype0
      integer ivbhpro(4,2:3,6)
      common/civbhpro/ivbhpro
      integer idp1(4,2:3,6),idp2(4,2:3,6),idp3(4,2:3,6)
      common/cidpart/idp1,idp2,idp3
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer nl
      common/nl/nl
      integer ifk88seed
      common/cifk88seed/ifk88seed
c
      i0=0
      jproc0=0
      itype0=0
      iret=0
      call checksb(xsum,xsumabs,iretvv)
      call checksbs(xsumvvs,xsumabsvvs,iretvvs)
      if(iretvv.eq.0.and.iretvvs.eq.1)then
        write(6,*)'Fatal error in xosb:',iretvv,iretvvs
        stop
      endif
      if(iretvv.eq.1)then
        iret=iretvv
        if(ifuntype.eq.1)then
          iproclo=loproc
          iprocma=maproc
        elseif(ifuntype.eq.2)then
          iproclo=loproc
          iprocma=maproc
        else
          write(*,*)'Fatal error in xosb: ifuntype=',ifuntype
          stop
        endif
        if(iretvvs.eq.1)then
          xsum=xsumvvs
          xsumabs=xsumabsvvs
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,6
                wwx(i,iproc,itype)=vvs(i,iproc,itype)
              enddo
            enddo
          enddo
        else
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,6
                wwx(i,iproc,itype)=vv(i,iproc,itype)
              enddo
            enddo
          enddo
        endif
        xstsign=sign(1.d0,xsum)
        xg=fk88random(ifk88seed)
        wh=0.d0
        iwh=0
        iflag=0
        rmax=0.d0
        do iproc=iproclo,iprocma
          do i=1,4
            do itype=1,6
              if(iwh.eq.0)then
                wh=wh+abs(wwx(i,iproc,itype))/xsumabs
                if(wh.gt.xg)then
                  i0=i
                  jproc0=iproc
                  itype0=itype
                  iwh=1
                endif
              endif
              if(wwx(i,iproc,itype).ne.0.d0)then
                if(xstsign.ne.sign(1.d0,wwx(i,iproc,itype)))then
                  if(iflag.eq.0)then
                    iwrong=iwrong+1
                    iflag=1
                  endif
                  rmax=max(rmax,abs(wwx(i,iproc,itype)))
                endif
              endif
            enddo
          enddo
        enddo
        if(iflag.eq.1)then
          if(xsum.ne.0.d0)rmax=rmax/xsum
          if(rmax.gt.0.05d0.or.xsum.eq.0.d0)iwrong1=iwrong1+1
        endif
        if(i0.eq.0.or.jproc0.eq.0.or.itype0.eq.0)then
          write(*,*)'Fatal error in xosb',i0,jproc0,itype0
          stop
        endif
        ihpro=ivbhpro(i0,jproc0,itype0)
        i1=idp1(i0,jproc0,itype0)
        i2=idp2(i0,jproc0,itype0)
        i3=idp3(i0,jproc0,itype0)
        call parcrossisb(jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o)
        i1hpro=i1hproo
        ip1=ip1o
        ip2=ip2o
        ip3=ip3o
      endif
      return
      end


      subroutine parcrossisb(jproc0,ihpro,i1,i2,i3,
     #                       i1hproo,ip1o,ip2o,ip3o)
      implicit none
      integer jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o,
     # iallzero,ihprotrans(402:406)
      parameter (iallzero=1)
      integer ifuntype
      common/cifuntype/ifuntype
      data ihprotrans/401,0,403,403,401/
c
      if( (ifuntype.eq.1) .or. (ifuntype.eq.2.and.jproc0.eq.2) )then
        i1hproo=ihpro
        ip1o=i1
        ip2o=i2
        ip3o=i3
      elseif(ifuntype.eq.2.and.jproc0.eq.3)then
        if(ihpro.eq.401.or.ihpro.eq.403.or.ihpro.eq.407)then
          write(*,*)'Error #1 in parcrossisb:',ihpro,i1,i2,i3
          stop
        endif
        i1hproo=ihprotrans(ihpro)
        if(i1.eq.21)then
          ip1o=-i3
          ip2o=i2
          ip3o=i1
        elseif(i2.eq.21)then
          ip1o=i1
          ip2o=-i3
          ip3o=i2
        endif
      else
        write(*,*)'parcrossisb: do not know what to do'
        write(*,*)ifuntype,jproc0
        stop
      endif
      call parcheckfsb(i1hproo,ip1o,ip2o,ip3o,iallzero)
      return
      end


      subroutine checksb(xsum,xsumabs,iret)
c iret=0 -> all vv entries are equal to zero
c iret=1 -> there is at least one entry which is not zero
c xsum is the sum of all the entries of vv
c xsumabs is the sum of the absolute value of all the entries of vv
      implicit none
      integer jproc,iret,i,itype
      integer nl
      common/nl/nl
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      real * 8 vv(4,2:3,6)
      common/cvv/vv
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=loproc,maproc
        do i=1,4
          do itype=1,6
            if(vv(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vv(i,jproc,itype)
            xsumabs=xsumabs+abs(vv(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      subroutine checksbs(xsum,xsumabs,iret)
c identical to checksb, except for the fact that works on vvs instead of vv,
c and jproc is not fixed
      implicit none
      integer jproc,iret,i,itype
      integer nl
      common/nl/nl
      real * 8 vvs(4,2:3,6)
      common/cvvs/vvs
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=2,2
        do i=1,4
          do itype=1,6
            if(vvs(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vvs(i,jproc,itype)
            xsumabs=xsumabs+abs(vvs(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


C      function itoosoftkin()
Cc Returns 1 when a three-body kinematics can be safely approximated
Cc with a two-body kinematics. It is useful when three-body NLO configurations
Cc are obtained, which cannot be produced through showering
C      implicit none
C      integer itoosoftkin,itmp
Cc
C      itmp=0
C      itoosoftkin=itmp
C      return
C      end
c
c
c End of event-generation routines
c
c
c
c
c Begin of phase-space routines
c
c
      subroutine invsb(xm2,s,xii,yi,t,u)
      implicit none
      real * 8 xm2,s,xii,yi,t,u,sqs,syi,sqso2
      real * 8 pth,ptp
      common/perpen/pth,ptp
      real*8 xmom_cm(4,4)
      common/cxmomcm/xmom_cm
      integer ichkmom
      common/cichkmom/ichkmom
c
      t=-s/2.d0*xii*(1-yi)
      u=-s/2.d0*xii*(1+yi)
      sqs=sqrt(s)
      syi=sqrt(1-yi**2)
      sqso2=sqs/2.d0
      xmom_cm(1,1)=0.d0
      xmom_cm(1,2)=0.d0
      xmom_cm(1,3)=sqso2
      xmom_cm(1,4)=sqso2
      xmom_cm(2,1)=0.d0
      xmom_cm(2,2)=0.d0
      xmom_cm(2,3)=-sqso2
      xmom_cm(2,4)=sqso2
      xmom_cm(3,1)=sqso2*xii*syi
      xmom_cm(3,2)=0.d0
      xmom_cm(3,3)=sqso2*xii*yi
      xmom_cm(3,4)=sqso2*xii
      xmom_cm(4,1)=-sqso2*xii*syi
      xmom_cm(4,2)=0.d0
      xmom_cm(4,3)=-sqso2*xii*yi
      xmom_cm(4,4)=sqso2*(2-xii)
      ptp=xmom_cm(3,1)
      pth=xmom_cm(3,1)
      return
      end


      function xbwmasb2(t)
c Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
c and the mass range (implicit in bwdelf and bwfmmn). This function
c is the inverse of F(M^2), where
c   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
c   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c and therefore eats up the Breit-Wigner when changing integration 
c variable M^2 --> t
      implicit none
      real*8 xbwmasb2,t
      include 'svbcblks.h'
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      xbwmasb2=xm02+xm0*gah*tan(pi*bwdelf*t-bwfmmn)
      return
      end


      subroutine checkmsb(xmom,smax,ybst,iflag)
      implicit none
      real * 8 xmom(4,4)
      real * 8 smax,ybst,xpmax
      real*8 x1,x2
      common/cx1x2/x1,x2
      real * 8 tiny,vtiny,xsum(4),xsuma(4),xsign,xrat(4)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-5)
      integer iflag,i,j,jj,jflag,jeflag
c
      jflag=0
      jeflag=0
      xpmax=sqrt(smax)/2.d0*(1+vtiny)
      do i=1,4
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=1,4
          if(i.ne.4.and.xmom(j,i).gt.xpmax)jeflag=1
          xsign=1.d0
          if(j.eq.1.or.j.eq.2)xsign=-1.d0
          xsum(i)=xsum(i)+xmom(j,i)*xsign
          xsuma(i)=xsuma(i)+abs(xmom(j,i))
        enddo
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved'
          write(*,*)'iflag,i=',iflag,i
          write(*,*)'smax,y=',smax,ybst
          write(*,*)'x1,x2=',x1,x2
          do j=1,4
            write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=1,4)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=1,4)
        stop
      endif
      if(jeflag.eq.1)then
        write(*,*)'Momentum component larger than sqrt(s)/2'
        write(*,*)'iflag=',iflag
        write(*,*)'s,pmax,y=',smax,xpmax,ybst
        write(*,*)'x1,x2=',x1,x2
        do j=1,4
          write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
        enddo
        stop
      endif
      return
      end
c
c
c End of phase-space routines
c
c
C      FUNCTION FK88RANDOM(SEED)
C*     -----------------
C* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
C* Use seed = 1 as first value.
C*
C      IMPLICIT INTEGER(A-Z)
C      DOUBLE PRECISION MINV,FK88RANDOM
C      SAVE
C      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
C      PARAMETER(MINV=0.46566128752458d-09)
C      HI = SEED/Q
C      LO = MOD(SEED,Q)
C      SEED = A*LO - R*HI
C      IF(SEED.LE.0) SEED = SEED + M
C      FK88RANDOM = SEED*MINV
C      END
c
c
c Initialization
c
c
      subroutine setpsb()
      implicit none
      include 'svbcblks.h'
      real * 8 pi,zup,zdo,aem,xmz,xmw,xmz2,xmw2,xalfaem
      parameter (pi=3.14159265358979312D0)
      parameter (zup=2.d0/3.d0,zdo=-1.d0/3.d0)
c Values from PDG 2003
      parameter (xmz=91.1876d0)
      parameter (xmw=80.425d0)
      real * 8 zel(1:6),zel2(1:6),ac(1:6),vc(1:6)
      real * 8 zqqcoup(1:6),ckm(1:6,1:6),ckm2(1:6,1:6),xkm(1:6,1:6)
      common/charges/zel,zel2
      common/czqqcoup/zqqcoup
      common/cckm2/ckm2
      real * 8 extrafact
      common/cextra/extrafact
      real * 8 pfcwon,pfcwmsb,pfczon,pfczmsb
      common/xsecpref/pfcwon,pfcwmsb,pfczon,pfczmsb
      integer iprdct
      common/ciprdct/iprdct
      integer ip4
      common/ci2part/ip4
      integer i,j
c
c Number of light flavours
      nl=5
c Fermi constant, from PDG2002
      gf=1.16639d-5
c alpha_em
      aem=1/137.0359895d0
c electron charge squared
      ze2=4*pi*aem
c sin and cos squared of theta_W; MSbar scheme, from PDG2003
      sthw2=0.23113d0
      cthw2=1-sthw2
c The matrix ZEL(I) returns the electric charge of flavour I. The
c numbering convention for the flavours should match the one of
c the structure function package. In our case, 
c 1=UP, 2=DOWN, 3=STRANGE, 4=CHARM, 5=BOTTOM, 6=TOP
c ZEL2(I) returns the charge squared.
c The variable extrafact is (sum_i N_C*ZEL(i)**2 + 3)/(12*pi**2), 3 being
c the sum of the lepton electric charges
      extrafact=3.d0
      zel(1)=zup
      zel(2)=zdo
      zel(3)=zdo
      zel(4)=zup
      zel(5)=zdo
      zel(6)=zup
      do i=1,6
        zel2(i)=zel(i)**2
        if(i.le.nl)extrafact=extrafact+3*zel2(i)
      enddo
      extrafact=extrafact/(12*pi**2)
c axial and vector couplings to the Z
      ac(1)=0.5d0
      ac(2)=-0.5d0
      ac(3)=-0.5d0
      ac(4)=0.5d0
      ac(5)=-0.5d0
      ac(6)=0.5d0
      vc(1)=0.5d0-4*sthw2/3.d0
      vc(2)=-0.5d0+2*sthw2/3.d0
      vc(3)=-0.5d0+2*sthw2/3.d0
      vc(4)=0.5d0-4*sthw2/3.d0
      vc(5)=-0.5d0+2*sthw2/3.d0
      vc(6)=0.5d0-4*sthw2/3.d0
c combinations that appear in Z production cross section
      do i=1,6
        zqqcoup(i)=ac(i)**2+vc(i)**2
      enddo
c ckm(i,j)=|CKM matrix elements|, with  i=1,4,6 --> up,charm,top
c                                       j=2,3,5 --> down,strange,bottom
c Values from PDG 2003
      do i=1,6
        do j=1,6
          ckm(i,j)=0.d0
        enddo
      enddo
c Values from PDG 2003; individual results from eqs.(11.6)-(11.13); not
c used since unitary is not taken into account
      xkm(1,2)=0.9734d0
      xkm(1,3)=0.2196d0
      xkm(1,5)=3.6d-3
      xkm(4,2)=0.224d0
      xkm(4,3)=0.996d0
      xkm(4,5)=41.2d-3
      xkm(6,2)=0.0035d0
      xkm(6,3)=0.04d0
      xkm(6,5)=0.999d0
c Centers of the ranges given in eq.(11.2), supposedly taking unitarity
c into accout; with the following entries, it holds better than 0.1%
      ckm(1,2)=0.9748d0
      ckm(1,3)=0.2225d0
      ckm(1,5)=0.0036d0
      ckm(4,2)=0.2225d0
      ckm(4,3)=0.9740d0
      ckm(4,5)=0.041d0
      ckm(6,2)=0.009d0
      ckm(6,3)=0.0405d0
      ckm(6,5)=0.9992d0
      do i=1,6
        do j=1,6
          ckm2(i,j)=ckm(i,j)**2
        enddo
      enddo
c Prefactors for W and Z cross sections in the on-shell and MSbar schemes
      xmw2=xmw**2
      pfcwon=sqrt(2.d0)*gf*xmw2
      pfcwmsb=pi*xalfaem(xmw2)/sthw2
      xmz2=xmz**2
      pfczon=sqrt(2.d0)*gf*xmz2
      pfczmsb=pi*xalfaem(xmz2)/(sthw2*cthw2)
c Final-state particle identities (except light parton)
      if(iprdct.eq.1396)then
        ip4=22
      elseif(iprdct.eq.1397)then
        ip4=23
      elseif(iprdct.eq.1497)then
        ip4=24
      elseif(iprdct.eq.1498)then
        ip4=-24
      else
        write(*,*)'Error in setpsb: unknown process',iprdct
        stop
      endif
      return
      end


      subroutine parsetpsb()
      implicit none
      integer jproc,i,itype,iprdct
      integer imapp(0:5)
      integer ivbhpro(4,2:3,6)
      integer idp1(4,2:3,6),idp2(4,2:3,6),idp3(4,2:3,6)
      common/cimapp/imapp
      common/civbhpro/ivbhpro
      common/cidpart/idp1,idp2,idp3
      common/ciprdct/iprdct
      integer nl
      common/nl/nl
c
c imapp(i) returns the PDG id number (1=d, 2=u, 3=s, 4=c, 5=b, 21=g)
c given our id number (1=u, 2=d, 3=s, 4=c, 5=b, 0=g)
      imapp(0)=21
      imapp(1)=2
      imapp(2)=1
      imapp(3)=3
      imapp(4)=4
      imapp(5)=5
c
c ivbhpro returns the process number associated to the entries; this is
c identical to i1hpro (see the routine store_evensb)
      do i=1,4
        do jproc=2,3
          do itype=1,6
            ivbhpro(i,jproc,itype)=0
          enddo
        enddo
      enddo
c 
      if(iprdct.eq.1396.or.iprdct.eq.1397)then
c gamma or Z production
        do itype=1,nl
          ivbhpro(1,2,itype)=401
          ivbhpro(3,2,itype)=403
          ivbhpro(1,3,itype)=402
          ivbhpro(2,3,itype)=404
          ivbhpro(3,3,itype)=405
          ivbhpro(4,3,itype)=406
        enddo
      elseif(iprdct.eq.1497)then
c W+ production
        do itype=1,6
          ivbhpro(1,2,itype)=401
          ivbhpro(3,2,itype)=403
        enddo
        do itype=1,6
          ivbhpro(1,3,itype)=402
          ivbhpro(2,3,itype)=404
          ivbhpro(3,3,itype)=405
          ivbhpro(4,3,itype)=406
        enddo
      elseif(iprdct.eq.1498)then
c W- production
        do itype=1,6
          ivbhpro(1,2,itype)=401
          ivbhpro(3,2,itype)=403
        enddo
        do itype=1,6
          ivbhpro(1,3,itype)=402
          ivbhpro(2,3,itype)=404
          ivbhpro(3,3,itype)=405
          ivbhpro(4,3,itype)=406
        enddo
      else
        write(*,*)'Unknown process in parsetpsb:',iprdct
        stop
      endif
c
c idpX returns the flavour of parton number X (1=coming from the left,
c 2=coming from the right, 3=outgoing) in the process associated to the
c entries. The labelling scheme of PDG has been used
      do i=1,4
        do jproc=2,3
          do itype=1,6
            idp1(i,jproc,itype)=0
            idp2(i,jproc,itype)=0
            idp3(i,jproc,itype)=0
          enddo
        enddo
      enddo
c
      if(iprdct.eq.1396.or.iprdct.eq.1397)then
c gamma or Z production
        do itype=1,nl
          idp1(1,2,itype)=imapp(itype)
          idp1(3,2,itype)=-imapp(itype)
c
          idp2(1,2,itype)=-imapp(itype)
          idp2(3,2,itype)=imapp(itype)
c
          idp3(1,2,itype)=21
          idp3(3,2,itype)=21
c
          idp1(1,3,itype)=imapp(itype)
          idp1(2,3,itype)=-imapp(itype)
          idp1(3,3,itype)=21
          idp1(4,3,itype)=21
c
          idp2(1,3,itype)=21
          idp2(2,3,itype)=21
          idp2(3,3,itype)=imapp(itype)
          idp2(4,3,itype)=-imapp(itype)
c
          idp3(1,3,itype)=imapp(itype)
          idp3(2,3,itype)=-imapp(itype)
          idp3(3,3,itype)=imapp(itype)
          idp3(4,3,itype)=-imapp(itype)
c
        enddo
      elseif(iprdct.eq.1497)then
c W+ production
        idp1(1,2,1)=imapp(1)
        idp1(1,2,2)=imapp(1)
        idp1(1,2,3)=imapp(1)
        idp1(1,2,4)=imapp(4)
        idp1(1,2,5)=imapp(4)
        idp1(1,2,6)=imapp(4)
c
        idp1(3,2,1)=-imapp(2)
        idp1(3,2,2)=-imapp(3)
        idp1(3,2,3)=-imapp(5)
        idp1(3,2,4)=-imapp(2)
        idp1(3,2,5)=-imapp(3)
        idp1(3,2,6)=-imapp(5)
c
        idp2(1,2,1)=-imapp(2)
        idp2(1,2,2)=-imapp(3)
        idp2(1,2,3)=-imapp(5)
        idp2(1,2,4)=-imapp(2)
        idp2(1,2,5)=-imapp(3)
        idp2(1,2,6)=-imapp(5)
c
        idp2(3,2,1)=imapp(1)
        idp2(3,2,2)=imapp(1)
        idp2(3,2,3)=imapp(1)
        idp2(3,2,4)=imapp(4)
        idp2(3,2,5)=imapp(4)
        idp2(3,2,6)=imapp(4)
c
        do itype=1,6
          idp3(1,2,itype)=21
          idp3(3,2,itype)=21
        enddo
c
        idp1(1,3,1)=imapp(1)
        idp1(1,3,2)=imapp(1)
        idp1(1,3,3)=imapp(1)
        idp1(1,3,4)=imapp(4)
        idp1(1,3,5)=imapp(4)
        idp1(1,3,6)=imapp(4)
c
        idp1(2,3,1)=-imapp(2)
        idp1(2,3,2)=-imapp(2)
        idp1(2,3,3)=-imapp(3)
        idp1(2,3,4)=-imapp(3)
        idp1(2,3,5)=-imapp(5)
        idp1(2,3,6)=-imapp(5)
c
        idp2(3,3,1)=imapp(1)
        idp2(3,3,2)=imapp(1)
        idp2(3,3,3)=imapp(1)
        idp2(3,3,4)=imapp(4)
        idp2(3,3,5)=imapp(4)
        idp2(3,3,6)=imapp(4)
c
        idp2(4,3,1)=-imapp(2)
        idp2(4,3,2)=-imapp(2)
        idp2(4,3,3)=-imapp(3)
        idp2(4,3,4)=-imapp(3)
        idp2(4,3,5)=-imapp(5)
        idp2(4,3,6)=-imapp(5)
c
        idp3(1,3,1)=imapp(2)
        idp3(1,3,2)=imapp(3)
        idp3(1,3,3)=imapp(5)
        idp3(1,3,4)=imapp(2)
        idp3(1,3,5)=imapp(3)
        idp3(1,3,6)=imapp(5)
c
        idp3(2,3,1)=-imapp(1)
        idp3(2,3,2)=-imapp(4)
        idp3(2,3,3)=-imapp(1)
        idp3(2,3,4)=-imapp(4)
        idp3(2,3,5)=-imapp(1)
        idp3(2,3,6)=-imapp(4)
c
        idp3(3,3,1)=imapp(2)
        idp3(3,3,2)=imapp(3)
        idp3(3,3,3)=imapp(5)
        idp3(3,3,4)=imapp(2)
        idp3(3,3,5)=imapp(3)
        idp3(3,3,6)=imapp(5)
c
        idp3(4,3,1)=-imapp(1)
        idp3(4,3,2)=-imapp(4)
        idp3(4,3,3)=-imapp(1)
        idp3(4,3,4)=-imapp(4)
        idp3(4,3,5)=-imapp(1)
        idp3(4,3,6)=-imapp(4)
c
        do itype=1,6
          idp2(1,3,itype)=21
          idp2(2,3,itype)=21
          idp1(3,3,itype)=21
          idp1(4,3,itype)=21
        enddo
      elseif(iprdct.eq.1498)then
c W- production
        idp1(1,2,1)=imapp(2)
        idp1(1,2,2)=imapp(2)
        idp1(1,2,3)=imapp(3)
        idp1(1,2,4)=imapp(3)
        idp1(1,2,5)=imapp(5)
        idp1(1,2,6)=imapp(5)
c
        idp1(3,2,1)=-imapp(1)
        idp1(3,2,2)=-imapp(4)
        idp1(3,2,3)=-imapp(1)
        idp1(3,2,4)=-imapp(4)
        idp1(3,2,5)=-imapp(1)
        idp1(3,2,6)=-imapp(4)
c
        idp2(1,2,1)=-imapp(1)
        idp2(1,2,2)=-imapp(4)
        idp2(1,2,3)=-imapp(1)
        idp2(1,2,4)=-imapp(4)
        idp2(1,2,5)=-imapp(1)
        idp2(1,2,6)=-imapp(4)
c
        idp2(3,2,1)=imapp(2)
        idp2(3,2,2)=imapp(2)
        idp2(3,2,3)=imapp(3)
        idp2(3,2,4)=imapp(3)
        idp2(3,2,5)=imapp(5)
        idp2(3,2,6)=imapp(5)
c
        do itype=1,6
          idp3(1,2,itype)=21
          idp3(3,2,itype)=21
        enddo
c
        idp1(1,3,1)=imapp(2)
        idp1(1,3,2)=imapp(2)
        idp1(1,3,3)=imapp(3)
        idp1(1,3,4)=imapp(3)
        idp1(1,3,5)=imapp(5)
        idp1(1,3,6)=imapp(5)
c
        idp1(2,3,1)=-imapp(1)
        idp1(2,3,2)=-imapp(1)
        idp1(2,3,3)=-imapp(1)
        idp1(2,3,4)=-imapp(4)
        idp1(2,3,5)=-imapp(4)
        idp1(2,3,6)=-imapp(4)
c
        idp2(3,3,1)=imapp(2)
        idp2(3,3,2)=imapp(2)
        idp2(3,3,3)=imapp(3)
        idp2(3,3,4)=imapp(3)
        idp2(3,3,5)=imapp(5)
        idp2(3,3,6)=imapp(5)
c
        idp2(4,3,1)=-imapp(1)
        idp2(4,3,2)=-imapp(1)
        idp2(4,3,3)=-imapp(1)
        idp2(4,3,4)=-imapp(4)
        idp2(4,3,5)=-imapp(4)
        idp2(4,3,6)=-imapp(4)
c
        idp3(1,3,1)=imapp(1)
        idp3(1,3,2)=imapp(4)
        idp3(1,3,3)=imapp(1)
        idp3(1,3,4)=imapp(4)
        idp3(1,3,5)=imapp(1)
        idp3(1,3,6)=imapp(4)
c
        idp3(2,3,1)=-imapp(2)
        idp3(2,3,2)=-imapp(3)
        idp3(2,3,3)=-imapp(5)
        idp3(2,3,4)=-imapp(2)
        idp3(2,3,5)=-imapp(3)
        idp3(2,3,6)=-imapp(5)
c
        idp3(3,3,1)=imapp(1)
        idp3(3,3,2)=imapp(4)
        idp3(3,3,3)=imapp(1)
        idp3(3,3,4)=imapp(4)
        idp3(3,3,5)=imapp(1)
        idp3(3,3,6)=imapp(4)
c
        idp3(4,3,1)=-imapp(2)
        idp3(4,3,2)=-imapp(3)
        idp3(4,3,3)=-imapp(5)
        idp3(4,3,4)=-imapp(2)
        idp3(4,3,5)=-imapp(3)
        idp3(4,3,6)=-imapp(5)
c
        do itype=1,6
          idp2(1,3,itype)=21
          idp2(2,3,itype)=21
          idp1(3,3,itype)=21
          idp1(4,3,itype)=21
        enddo
      else
        write(*,*)'Unknown process in parsetpsb:',iprdct
        stop
      endif
c
      call parcheckpsb()
      return
      end


      subroutine parcheckpsb()
      implicit none
      integer iallzero,i,jproc,itype,ihpro,i1,i2,i3
      parameter (iallzero=0)
      integer ivbhpro(4,2:3,6)
      common/civbhpro/ivbhpro
      integer idp1(4,2:3,6),idp2(4,2:3,6),idp3(4,2:3,6)
      common/cidpart/idp1,idp2,idp3
c
      call parcheckinsb()
      do jproc=2,3
        do i=1,4
          do itype=1,6
            ihpro=ivbhpro(i,jproc,itype)
            i1=idp1(i,jproc,itype)
            i2=idp2(i,jproc,itype)
            i3=idp3(i,jproc,itype)
            call parcheckfsb(ihpro,i1,i2,i3,iallzero)
          enddo
        enddo
      enddo
      return
      end


      subroutine parcheckfsb(ihpro,i1,i2,i3,iallzero)
      implicit none
      integer ihpro,i1,i2,i3,iallzero,isum
      real*8 tiny,chin,chout,chall
      parameter (tiny=1.d-8)
      logical ferror
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
      integer iprdct
      common/ciprdct/iprdct
c
      ferror=.false.
      isum=abs(i1)+abs(i2)+abs(i3)
      chin=chrg(i1)+chrg(i2)
      chout=chrg(i3)
      chall=chin-chout-chprdct
      if(iallzero.eq.0)then
c i1,i2,i3 must be either all nonzero, or all zero
        if( ( (i1.ne.0) .and. 
     #        (i2.eq.0.or.i3.eq.0) ) .or.
     #      ( (i2.ne.0) .and. 
     #        (i1.eq.0.or.i3.eq.0) ) .or.
     #      ( (i3.ne.0) .and. 
     #        (i1.eq.0.or.i2.eq.0) ) )ferror=.true.
      elseif(iallzero.eq.1)then
c all process parameters must be different from zero
        if(i1.eq.0.or.i2.eq.0.or.
     #     i3.eq.0.or.ihpro.eq.0)ferror=.true.
      else
        write(*,*)'parcheckfsb called improperly'
        stop
      endif
      if(isum.ne.0)then
c charge must be conserved
        if(abs(chall).gt.tiny)ferror=.true.
c 401 is qqbar
        if( ihpro.eq.401 .and.
     #      (i1.le.0 .or. i2.ge.0 .or.
     #       i3.ne.21 .or. 
     #      ( (iprdct.eq.1396.or.iprdct.eq.1397) .and.
     #        (i1+i2).ne.0 ) ) )ferror=.true.
c 402 is qg
        if( ihpro.eq.402 .and.
     #      (i1.le.0 .or. i2.ne.21 .or. 
     #       i3.le.0 .or. 
     #      ( (iprdct.eq.1396.or.iprdct.eq.1397) .and.
     #        i1.ne.i3 ) ) )ferror=.true.
c 403 is qbarq
        if( ihpro.eq.403 .and.
     #      (i1.ge.0 .or. i2.le.0 .or. 
     #       i3.ne.21 .or. 
     #      ( (iprdct.eq.1396.or.iprdct.eq.1397) .and.
     #        (i1+i2).ne.0 ) ) )ferror=.true.
c 404 is qbarg
        if( ihpro.eq.404 .and.
     #      (i1.ge.0 .or. i2.ne.21 .or. 
     #       i3.ge.0 .or. 
     #      ( (iprdct.eq.1396.or.iprdct.eq.1397) .and.
     #        i1.ne.i3 ) ) )ferror=.true.
c 405 is gq
        if( ihpro.eq.405 .and.
     #      (i1.ne.21 .or. i2.le.0 .or. 
     #       i3.le.0 .or. 
     #      ( (iprdct.eq.1396.or.iprdct.eq.1397) .and.
     #        i2.ne.i3 ) ) )ferror=.true.
c 406 is gqbar
        if( ihpro.eq.406 .and.
     #      (i1.ne.21 .or. i2.ge.0 .or. 
     #       i3.ge.0 .or. 
     #      ( (iprdct.eq.1396.or.iprdct.eq.1397) .and.
     #        i2.ne.i3 ) ) )ferror=.true.
c 407 is gg
        if( ihpro.eq.407 )ferror=.true.
      endif
      if(ferror)then
        write(*,*)'Error in parcheckfsb'
        write(*,*)'ihpro,i1,i2,i3:',ihpro,i1,i2,i3
        stop
      endif
      return
      end


      subroutine parcheckinsb()
      implicit none
      integer i
      real*8 chup,chdn
      parameter (chup=2.d0/3.d0)
      parameter (chdn=-1.d0/3.d0)
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
      integer iprdct
      common/ciprdct/iprdct
c
      do i=-5,21
        chrg(i)=1000.d0
      enddo
      chrg(1)=chdn
      chrg(2)=chup
      chrg(3)=chdn
      chrg(4)=chup
      chrg(5)=chdn
      chrg(21)=0.d0
      do i=1,5
        chrg(-i)=-chrg(i)
      enddo
      if(iprdct.eq.1396.or.iprdct.eq.1397)then
        chprdct=0.d0
      elseif(iprdct.eq.1497)then
        chprdct=1.d0
      elseif(iprdct.eq.1498)then
        chprdct=-1.d0
      else
        write(*,*)'Unknown process in parcheckinsb:',iprdct
        stop
      endif
      return
      end
c
c
c End initialization
c
c
c
c
c Begin of event file utilities
c
c
      subroutine whichosb(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
c Determines the type of event at random
      implicit none
      integer iseed,itot,mx_of_evta,mx_of_evtb,iunit,i0
      real*8 xpa,xpb,tiny,one,xsum,rnd,fk88random,prob
      parameter (tiny=1.d-4)
      logical flag
c
      if(itot.le.0)then
        write(6,*)'Fatal error #1 in whichosb'
        stop
      endif
      xpa=dfloat(mx_of_evta)/dfloat(itot)
      xpb=dfloat(mx_of_evtb)/dfloat(itot)
      one=xpa+xpb
      if(abs(one-1.d0).gt.tiny)then
        write(6,*)'Error #1 in whichosb: probability not normalized'
        stop
      endif
      i0=0
      flag=.true.
      xsum=0.d0
      rnd=fk88random(iseed)
      do while(flag)
        if(i0.gt.2)then
          write(6,*)'Fatal error #2 in whichosb'
          stop
        endif
        i0=i0+1
        prob=xpa
        if(i0.gt.1)prob=xpb
        xsum=xsum+prob
        if(rnd.lt.xsum)then
          flag=.false.
          itot=itot-1
          if(i0.le.1)then
            mx_of_evta=mx_of_evta-1
          else
            mx_of_evtb=mx_of_evtb-1
          endif
          iunit=20+i0
        endif
      enddo
      return
      end


      subroutine crosschesb(itot,mx_of_evta,mx_of_evtb)
c Checks whether whichosb did it right
      implicit none
      integer itot,mx_of_evta,mx_of_evtb
c
      if(itot.ne.0)then
        write(6,*)'Error: itot=',itot
        stop
      endif
      if(mx_of_evta.ne.0)then
        write(6,*)'Error: mx_of_evta=',mx_of_evta
        stop
      endif
      if(mx_of_evtb.ne.0)then
        write(6,*)'Error: mx_of_evtb=',mx_of_evtb
        stop
      endif
      return
      end


      subroutine retrieve_evensb(iunit,ii,dummy)
c Reads from disk the complete information on the events; see store_evensb
c for the conventions used
      implicit none
      integer iunit,ii,i,j
      real*8 dummy
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4
      common/ci2part/ip4
      integer iccode
      common/ciccode/iccode
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 xmom_lb(4,4)
      common/cxmomlb/xmom_lb
c
      read(iunit,901,end=997,err=998)i1hpro,iccode,np
      read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4
      read(iunit,903,end=997,err=998)xevsign
      read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,4)
      goto 999
 901  format(1x,i3,2(1x,i2))
 902  format(7(1x,i3))
 903  format(1x,d14.8)
 904  format(28(1x,d14.8))
 997  write(*,*)'unexpected end of file, iunit=',iunit
      stop
 998  write(*,*)'format error'
      write(77,*)'event #:',ii
      write(77,901)i1hpro,iccode,np
      write(77,902)ip1,ip2,ip3,ip4
      write(77,903)xevsign
      write(77,904)((xmom_lb(i,j),j=1,4),i=1,4)
      stop
 999  continue
      return
      end


      subroutine store_evensb(iunit,xpmone)
c Stores on disk the complete information on the events. Starting
c from version 3.1, each event has the following format:
c       IPR, IC, NP
c      (ID(I),I=1,NP)
c      ((P(J,I),J=1,4),I=1,NP)
c where IPR is the subprocess code (i1hpro), IC is the colour code
c (iccode, trivial here), NP is the number of partons entering the 
c reaction (thus, this includes the soft parton in the case of S events),
c ID(I) are the particle identities (ip1,...,ip4 here), and P(J,I) are 
c the particles four momenta in the lab frame (P(J,I)=xmom_lb(i,j) here).
c
c This routine is called with xpmone=1 when events are obtained from
c SPRING, and with xpmone=-1 after the events are read from the temporary
c files (via retrieve_evensb), to be stored in the final event file.
c When xpmone=1, one has xevsign=+1/-1, and the weight of the event is 
c xevsign*wgt[a,b]ev. When xpmone=-1, then xevsign is the weight of the event. 
c
c i1hpro has the following conventions:
c   i1hpro         process
c    401        q qbar -> g V
c    402        q g    -> q V
c    403        qbar q -> g V
c    404        qbar g -> qbar V
c    405        g q    -> q V
c    406        g qbar -> qbar V
c    407        g g    -> g V
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      implicit none
      integer iunit,i,j
      real*8 xpmone,xevwgt,xfact
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4
      common/ci2part/ip4
      integer iccode
      common/ciccode/iccode
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
c xmom_lb(i,j) is the j component of the four vector of the particle # i,
c given in the laboratory frame. j=4 is the energy. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is the vector boson.
c Momentum conservation is (1+2)-(3+4)=0
      real*8 xmom_lb(4,4)
      common/cxmomlb/xmom_lb
      integer iwgtnorm
      common/ciwgtnorm/iwgtnorm
      real*8 wgtaev,wgtbev
      common/cwgtev/wgtaev,wgtbev
c
      if(xpmone.eq.-1)then
        xevwgt=xevsign
        xfact=1.d0
      elseif(xpmone.eq.1)then
        np=4
        if(xmom_lb(3,4).eq.0.d0)then
          xevwgt=xevsign*wgtbev
        else
          xevwgt=xevsign*wgtaev
        endif
        xfact=1.d0
      else
        write(*,*)'Fatal error in store_evensb: xpmone=',xpmone
        stop
      endif
      write(iunit,901)i1hpro,iccode,np
      write(iunit,902)ip1,ip2,ip3,ip4
      write(iunit,903)xevwgt
      write(iunit,904)((xmom_lb(i,j),j=1,4),i=1,4)
 901  format(1x,i3,2(1x,i2))
 902  format(7(1x,i3))
 903  format(1x,d14.8)
 904  format(28(1x,d14.8))
      return
      end
c
c
c End of event file utilities
c
c
c
c
c Begin of MC subtraction terms
c
c
      subroutine xmcsusb(jproc,s,xii,y,ycnt,xmcxsec)
c xmcxsec is only output of this routine, and is defined using eq.(A.83)
c (for ycnt=1) or eq.(A.84) (for ycnt=-1) of FW, with phi_3 --> phi_2,
c and obvious modifications for the flavour. The definition is
c     (1-x)**2*(1-y**2)*dsigma_MC/dphi_2 = gs**2*e**2*zeq**2*xmcxsec
c This routine is derived from xmcsusb of the Higgs package.
c For consistency, the same notations are kept: their meaning is as follows
c xmin=(7+sqrt(17))/16==x_DZ, eq.(A.70) of FW
c xsoft=tilde{x_DZ}, eq.(A.87) of FW
c ydead_mod(x)=Y_DZ(x), eq.(A.71) of FW
c xlim1 is the part which multiplies G
c xlim2 is the part which multiplies 1-G
c The various singular limits are identical to those of the vector boson
c package, and have been checked independently
      implicit none
      real*8 s,xii,y,ycnt,xmcxsec(4)
      integer jproc
      real*8 x,tiny,vcf,vtf,xmin,xlim,xlim1,xlim2,al_gfun,be_gfun,
     # ccc_gfun,yd,ydead_mod,xfact,z,zherw_spl,xi,xiherw_spl,
     # ap,ap_kern,xjac_xizspl,zherw_smn,xiherw_smn,xjac_xizsmn,
     # tt,gfact,gfun,sbar,xsoft,born(4)
      integer icode,itwo,j
      parameter (tiny=1.d-6)
      parameter (vcf=4/3.d0)
      parameter (vtf=1/2.d0)
      parameter (xmin=0.69519410160110384d0)
      parameter (itwo=2)
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
c
      if(abs(ycnt).ne.1.d0) then
        write(6,*)'xmcsusb called improperly: ycnt=',ycnt
        stop
      endif
      x=1-xii
      xlim=0.d0
      xlim1=0.d0
      xlim2=0.d0
      sbar=x*s
      call f1sbrn(sbar,itwo,'mc',born)
      if(jproc.eq.2)then
        icode=4
        xsoft=1.d0-be_gfun+xmin*be_gfun
        yd=ydead_mod(x)
        if(ycnt.eq.1.d0)then
          if(x.gt.xsoft)then
            xlim2=ap_kern(x,icode)*(1+y)
          endif
          if(y.gt.yd)then
            if(1-x.lt.tiny)then
              xlim1=8*(1+y)/(3+y)-
     #              2*(1-x)*(1+y)*(7+6*y+3*y**2)/(3+y)**2
              xlim1=vcf*xlim1
            elseif(1-y.lt.tiny)then
              xlim1=2*(1+x**2)-
     #              (2-1/x-(7*x)/2.d0+5*x**2-3*x**3/2.d0)*(1-y)
              xlim1=vcf*xlim1
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_spl(x,y)
              xi=xiherw_spl(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim1=xjac_xizspl(x,y)*xfact*ap/xi
            endif
          endif
        elseif(ycnt.eq.-1.d0)then
          if(x.gt.xsoft)then
            xlim2=ap_kern(x,icode)*(1-y)
          endif
          if(y.lt.-yd)then
            if(1-x.lt.tiny)then
              xlim1=8*(1-y)/(3-y)-
     #              2*(1-x)*(1-y)*(7-6*y+3*y**2)/(3-y)**2
              xlim1=vcf*xlim1
            elseif(1+y.lt.tiny)then
              xlim1=2*(1+x**2)-
     #              (2-1/x-(7*x)/2.d0+5*x**2-3*x**3/2.d0)*(1+y)
              xlim1=vcf*xlim1
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_smn(x,y)
              xi=xiherw_smn(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim1=xjac_xizsmn(x,y)*xfact*ap/xi
            endif
          endif
        endif
        tt=(1.d0-x)/(1.d0-xsoft)
        gfact=gfun(tt)
        xlim=4*(xlim1*gfact+xlim2*(1.d0-gfact))/s
        do j=1,4
          xmcxsec(j)=xlim*born(j)
        enddo
      elseif(jproc.eq.3)then
        do j=1,4
          xmcxsec(j)=0.d0
        enddo
        icode=2
        yd=ydead_mod(x)
        if(ycnt.eq.1.d0)then
          if(y.gt.yd)then
            if(1-x.lt.tiny)then
              xlim=(1-x)*(1+y)
              xlim=vtf*xlim
            elseif(1-y.lt.tiny)then
              xlim=2-6*x+8*x**2-4*x**3-
     #             (4*x**5-19*x**4+31*x**3-23*x**2+8*x-1)*(1-y)/x
              xlim=vtf*xlim
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_spl(x,y)
              xi=xiherw_spl(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim=xjac_xizspl(x,y)*xfact*ap/xi
            endif
          endif
          xlim=4*xlim/s
          do j=3,4
            xmcxsec(j)=xlim*born(j)
          enddo
        elseif(ycnt.eq.-1.d0)then
          if(y.lt.-yd)then
            if(1-x.lt.tiny)then
              xlim=(1-x)*(1-y)
              xlim=vtf*xlim
            elseif(1+y.lt.tiny)then
              xlim=2-6*x+8*x**2-4*x**3-
     #             (4*x**5-19*x**4+31*x**3-23*x**2+8*x-1)*(1+y)/x
              xlim=vtf*xlim
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_smn(x,y)
              xi=xiherw_smn(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim=xjac_xizsmn(x,y)*xfact*ap/xi
            endif
          endif
          xlim=4*xlim/s
          do j=1,2
            xmcxsec(j)=xlim*born(j)
          enddo
        endif
      else
        write(*,*)'xmcsusb: unknown process',jproc
        stop
      endif
      return
      end


C      function ffunction5(xx,yy)
C      implicit real * 8 (a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      tmp=0.d0
C      yd=ydead(x)
C      if(abs(y).ge.yd)tmp=1.d0
C      ffunction5=tmp
C      return
C      end


C      function ydead(x)
C      implicit real*8(a-h,o-z)
C      parameter (xmin=0.69519410160110384d0)
Cc
C      tmp=0.d0
C      if(x.lt.xmin)tmp=1-x*(3.d0-sqrt(1+8*x))/(1-x)
C      ydead=tmp
C      return
C      end


C      function ydead_mod(x)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      if(1-x.lt.tiny)then
C        tmp=-1/3.d0-28.d0*(X-1)/27.d0
C      else
C        tmp=1-x*(3.d0-sqrt(1+8*x))/(1-x)
C      endif
C      ydead_mod=tmp
C      return
C      end


C      function gfun(xx)
C      implicit real*8(a-h,o-z)
C      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
Cc
C      x=xx
C      tmp=1.d0
C      if(x.lt.0.d0)then
C        write(6,*)'Fatal error in gfun'
C        stop
C      endif
C      if(x.le.1.d0.and.al_gfun.gt.0.d0)
C     #  tmp=x**(2*al_gfun)/(x**(2*al_gfun)+(1-x)**(2*al_gfun))
C      gfun=tmp
C      return
C      end
c
c
c End of MC subtraction terms
c
c
c
c
c Begin of utility routines for xi, z, and 2-->2 invsbiants. The functions
c for xi, z, and the jacobian have been checked numerically to coincide with
c those of the hvq package (except when closer than tiny to IR limits, since
c here more terms are kept -- which are numerically irrelevant). The present
c for is simpler and faster to computed, thanks to the unique choice of E0
c
c
C      function zherw_spl(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=1.d0+(Y+3)*(X-1)/4.d0
C      elseif(1-y.lt.tiny)then
C        tmp=X-(X**2-3*X+2)*(Y-1)/4.d0
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-t/xmv2
C        xb=u*t/xmv2
C        tmp=1/xa*( sqrt(1+2*xa-2*xb)-1 )
C      endif
C      zherw_spl=tmp
C      return
C      end


C      function zherw_smn(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=1-(Y-3)*(X-1)/4.d0
C      elseif(1+y.lt.tiny)then
C        tmp=X+(X**2-3*X+2)*(Y+1)/4.d0
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-u/xmv2
C        xb=u*t/xmv2
C        tmp=1/xa*( sqrt(1+2*xa-2*xb)-1 )
C      endif
C      zherw_smn=tmp
C      return
C      end


C      function xiherw_spl(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=-(2*Y-2)/(Y+3)-(2*Y**3+2*Y**2-2*Y-2)*(X-1)/(Y**2+6*Y+9)
C      elseif(1-y.lt.tiny)then
C        tmp=-X*(Y-1)/2
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-t/xmv2
C        z1=zherw_spl(x,y)
C        tmp=xa*z1**2/(1-z1)
C      endif
C      xiherw_spl=tmp
C      return
C      end


C      function xiherw_smn(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=-(2*Y+2)/(Y-3)+(2*Y**3-2*Y**2-2*Y+2)*(X-1)/(Y**2-6*Y+9)
C      elseif(1+y.lt.tiny)then
C        tmp=X*(Y+1)/2
C      else
C        xmv2=x
C        t=-0.5d0*(1-x)*(1-y)
C        u=-0.5d0*(1-x)*(1+y)
C        xa=-u/xmv2
C        z1=zherw_smn(x,y)
C        tmp=xa*z1**2/(1-z1)
C      endif
C      xiherw_smn=tmp
C      return
C      end


C      function xjac_xizspl(xx,yy)
C      implicit none
C      real*8 xjac_xizspl,x,y,xx,yy
C      real*8 z,xi,zherw_spl,xiherw_spl,tmp,tiny
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=2/(Y+3)+(2*Y**2+4*Y+2)*(X-1)/(Y**2+6*Y+9)
C      elseif(1-y.lt.tiny)then
C        tmp=X/2-(3*X**2-8*X+6)*(Y-1)/8
C      else
C        z=zherw_spl(x,y)
C        xi=xiherw_spl(x,y)
C        tmp=(X-1)**2*(Y-1)*(X*Y-Y+X+1)*Z**5/
C     #      (8*X**3*XI*(Z-1)**2*(XI*Z-Z-XI))
C      endif
C      xjac_xizspl=tmp
C      return
C      end


C      function xjac_xizsmn(xx,yy)
C      implicit none
C      real*8 xjac_xizsmn,x,y,xx,yy
C      real*8 z,xi,zherw_smn,xiherw_smn,tmp,tiny
C      parameter (tiny=1.d-4)
Cc
C      x=xx
C      y=yy
C      if(1-x.lt.tiny)then
C        tmp=2/(Y-3)-(2*Y**2-4*Y+2)*(X-1)/(Y**2-6*Y+9)
C      elseif(1+y.lt.tiny)then
C        tmp=-X/2-(3*X**2-8*X+6)*(Y+1)/8
C      else
C        z=zherw_smn(x,y)
C        xi=xiherw_smn(x,y)
C        tmp=-(X-1)**2*(Y+1)*(X*Y-Y-X-1)*Z**5/
C     #      (8*X**3*XI*(Z-1)**2*(XI*Z-Z-XI))
C      endif
C      xjac_xizsmn=-tmp
C      return
C      end
c
c
c End of utility routines for xi, z, and 2-->2 invsbiants
c
c
c
c
c Begin of utility routines for Bjorken x's
c
c
C      function x1soft(xx1,xx2,xx,yy)
C      implicit none
C      real*8 x1soft,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
C      parameter (tiny=1.d-5)
C      integer iprespl
C      common/ciprespl/iprespl
Cc
C      x1=xx1
C      x2=xx2
C      x=xx
C      y=yy
C      if(iprespl.eq.0)then
C        csi=sqrt( (2-(1-x)*(1+y))/(2-(1-x)*(1-y)) )
C        rx=sqrt(x)
C        tmp=x1*csi*rx
C      elseif(iprespl.eq.1)then
C        if(1-x.lt.tiny)then
C          tmp=x1*(1-(1-x)*(1+y)/2.d0)
C        elseif(1-y.lt.tiny)then
C          tmp=x*x1*(1+(1-x)*(1-y)*(x1+x2)/(2.d0*(x*x1+x2)))
C        elseif(1+y.lt.tiny)then
C          tmp=x1*(1-(1-x)*(1+y)*(x1+x2)/(2.d0*(x1+x*x2)))
C        else
C          xa=x*x1*x2
C          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
C          tmp=0.5d0*(sqrt(xb**2+4*xa)-xb)
C        endif
C      else
C        write(*,*)'Error in x1soft',iprespl
C        stop
C      endif
C      x1soft=tmp
C      return
C      end


C      function x2soft(xx1,xx2,xx,yy)
C      implicit none
C      real*8 x2soft,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
C      parameter (tiny=1.d-5)
C      integer iprespl
C      common/ciprespl/iprespl
Cc
C      x1=xx1
C      x2=xx2
C      x=xx
C      y=yy
C      if(iprespl.eq.0)then
C        csi=sqrt( (2-(1-x)*(1+y))/(2-(1-x)*(1-y)) )
C        rx=sqrt(x)
C        tmp=x2*rx/csi
C      elseif(iprespl.eq.1)then
C        if(1-x.lt.tiny)then
C          tmp=x2*(1-(1-x)*(1-y)/2.d0)
C        elseif(1-y.lt.tiny)then
C          tmp=x2*(1-(1-x)*(1-y)*(x1+x2)/(2.d0*(x*x1+x2)))
C        elseif(1+y.lt.tiny)then
C          tmp=x*x2*(1+(1-x)*(1+y)*(x1+x2)/(2.d0*(x1+x*x2)))
C        else
C          xa=x*x1*x2
C          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
C          tmp=0.5d0*(sqrt(xb**2+4*xa)+xb)
C        endif
C      else
C        write(*,*)'Error in x2soft',iprespl
C        stop
C      endif
C      x2soft=tmp
C      return
C      end


C      function x1x2jac(xx1,xx2,xx,yy)
C      implicit none
C      real*8 x1x2jac,xx1,xx2,xx,yy,tiny,x1,x2,x,y,tmp,xa,xb
C      parameter (tiny=1.d-5)
C      integer iprespl
C      common/ciprespl/iprespl
Cc
C      x1=xx1
C      x2=xx2
C      x=xx
C      y=yy
C      if(iprespl.eq.0)then
C        tmp=x
C      elseif(iprespl.eq.1)then
C        if(1-x.lt.tiny)then
C          tmp=x+(1-x)**2*(1-y**2)*x1*x2/(2.d0*(x1+x2)**2)
C        elseif(1-y.lt.tiny)then
C          tmp=x+(1-x)**2*x*(1-y)*x1*x2/(x*x1+x2)**2
C        elseif(1+y.lt.tiny)then
C          tmp=x+(1-x)**2*x*(1+y)*x1*x2/(x1+x*x2)**2
C        else
C          xa=x*x1*x2
C          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
C          tmp=x*((1-y+x*(1+y))*x1+(1+y+x*(1-y))*x2)/
C     #        (2.d0*sqrt(xb**2+4*xa))
C        endif
C      else
C        write(*,*)'Error in x1x2jac',iprespl
C        stop
C      endif
C      x1x2jac=abs(tmp)
C      return
C      end
c
c
c End of utility routines for Bjorken x's
c
c
c
c
c Running couplings
c
c
      function zgmu2_nsb()
c Sets the scales for NLO subtraction terms
      implicit none
      real * 8 zgmu2_nsb
      real * 8 pi
      parameter (pi=3.14159265358979312D0)
      include 'svbcblks.h'
      real * 8 pth,ptp
      common/perpen/pth,ptp
      integer inloscale
      common/cinloscale/inloscale
      real * 8 xmu2,as,alfas
c
      if(inloscale.eq.1)then
        xmu2 = pth**2 + xm2
      elseif(inloscale.eq.2)then
        xmu2 = xm2
      elseif(inloscale.eq.3)then
        xmu2 = max(xm2,pth**2)
      elseif(inloscale.eq.4)then
        xmu2 = ( max(sqrt(xm2)-pth,pth) )**2
      else
        write(*,*)'Unknown option in zgmu2_nsb',inloscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmuf2h1 = xmu2*xf2h1
      xmuf2h2 = xmu2*xf2h2
      xmur2  = xmu2*xren2
      as    = alfas(xmur2,xlam,nl)
      zgmu2_nsb = 4.d0*pi*as
      zg = sqrt(zgmu2_nsb)
      end


      function zgmu2_sb()
c Sets the scales for MC subtraction terms
      implicit none
      real * 8 zgmu2_sb
      real * 8 pi
      parameter (pi=3.14159265358979312D0)
      include 'svbcblks.h'
      real * 8 pth,ptp
      common/perpen/pth,ptp
      integer imcscale
      common/cimcscale/imcscale
      real * 8 xmu2,as,alfas
c
      if(imcscale.eq.1)then
        xmu2 = pth**2 + xm2
      elseif(imcscale.eq.2)then
        xmu2 = xm2
      elseif(imcscale.eq.3)then
        xmu2 = max(xm2,pth**2)
      elseif(imcscale.eq.4)then
        xmu2 = ( max(sqrt(xm2)-pth,pth) )**2
      elseif(imcscale.eq.5)then
        xmu2 = ( sqrt(xm2)-pth/2.d0 )**2
      else
        write(*,*)'Unknown option in zgmu2_sb',imcscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmumcf2h1 = xmu2*xf2h1mc
      xmumcf2h2 = xmu2*xf2h2mc
      xmumcr2  = xmu2*xren2mc
      as    = alfas(xmumcr2,xlam,nl)
      zgmu2_sb = 4.d0*pi*as
      zg = sqrt(zgmu2_sb)
      end


Cc-------------------------------------------------------------------------
C      function xalfaem(q2)
Cc Alpha_em(MSbar) at the scale q2 = q^2. 
Cc Uses alpha_Thomson below the electron mass, an interpolation between
Cc m_e and m_tau, and the evolution equation above m_b. This function is
Cc taken from the gamma*gamma* --> hadrons package
Cc-------------------------------------------------------------------------
C      implicit real*8 (a-z)
C      integer npoints,ideg
C      parameter (npoints=3,ideg=3)
C      real*4 ooa(npoints),xlogmu(npoints),divdif
Cc 1/alpha_em at m_e=0.000511,m_mu=0.1056,m_tau=1.777      
C      data ooa     / 137.036, 135.95, 133.513 /
Cc logs of sqrt(q2) at m_e=0.000511,m_mu=0.1056,m_tau=1.777      
C      data xlogmu  / -7.57914, -2.2481, 0.574927 /
C      data zm/91.2d0/,ooaz/127.934d0/,pi/3.1415927d0/,nc/3/
Cc
C      if(q2.lt.exp(2.*xlogmu(1))) then
C         xalfaem = 1.d0/ooa(1)	 
C      elseif(q2.lt.exp(2.*xlogmu(3))) then
C         xlogq = log(q2)/2.d0
C         xalfaem = 1.d0/divdif(ooa,xlogmu,npoints,sngl(xlogq),ideg)
C      elseif(q2.lt.5.**2) then
C         b = 3 + 2*nc*(1d0/3d0)**2 + 2*nc*(2d0/3d0)**2
C         xlq = log(q2) - 2.*xlogmu(3)
C         xalfaem = 1d0/ooa(3)/(1.d0 - 1.d0/3.d0/pi/ooa(3)*b*xlq)
C      else
C         b = 3 + 3*nc*(1d0/3d0)**2 + 2*nc*(2d0/3d0)**2
C         xlq = log(q2/zm**2)
C         xalfaem = 1d0/ooaz/(1.d0 - 1.d0/3.d0/pi/ooaz*b*xlq)
C      endif
C      return
C      end
