      SUBROUTINE HGMAIN
      implicit none
      include 'hgscblks.h'
      real * 8 value(20),xmass(-5:21)
      real * 8 
     #  xmh0,xpdflam4,xpdflam5,tmp,xren,xfh,xrenmc,xfhmc,ecm,xicut,roh,
     #  delta,deltas,deltac,dtot,avtot,ac1,ac2,xtotal,ytotal,av3a,
     #  av3nega,d3a,d3nega,ctime,pi,tmas,etacut,xmt,av3b,av3negb,d3b,
     #  d3negb,wgtaev,wgtbev,evfrac,evprcfrac,xares,yares,xbres,ybres,
     #  al_gfun,be_gfun,ccc_gfun,dummy,xmone,xicutss,gammax,
     #  xmhl,xmhu,xmhl2,xmhu2,bwxmdpl,bwxmdmn
      integer 
     #  ih1,ih2,ndns1,ndns2,iseld,nlf,ncl3,mx_of_evta,mx_of_evtb,
     #  loproc,maproc,iproc,iinput,iverbose,ichkmom,
     #  ibswrite,itmpih,itmpndns,idpdfset,ipdfih,ipdfgroup,ipdfndns,
     #  ifk88istrl,ifk88ih,ifk88ndns,maxevt,it1,it2,ifuntype,
     #  ndim,nwild,itd1,itd2,ibscall,iwgtnorm,iseed0,
     #  iseed,maxtrials,mode,lo,isubttype,iprespl,ibornex,
     #  ndiff,ntotal,i,ionshell,iwrong,iwrong1,nevts,ntrls,
     #  iunita,iunitb,iunit,ioutput,itot,ii,iwidth,inloscale,
     #  imcscale,ifk88seed,ip4,izero,ione
      character * 2 scheme
      character * 4 part1,part2
      character * 20 parm(20),gname
      character * 80 fname,fnamea,fnameb,fname1,fnamev
      character * 80 pref,prefn,prefev,prefnev
      character * 70 strin,strout,lhapdf
      logical evgen
      external shg5afw,shg5azw,shg5bfw,shg5bzw
      parameter (pi=3.14159265358979312D0)
      parameter (xmone=-1.d0)
      parameter (izero=0)
      parameter (ione=1)
c
c common /strfhg0/ is only in strfhg:
c ndns = pdf type
c ih1,ih2 = beam type (0=(p+n)/2, 1=p, -1=pbar, 2=n, -2=nbar)
      common/strfhg0/ih1,ih2,ndns1,ndns2
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
c ichkmom=0 --> enables checks on kinematics
      common/cichkmom/ichkmom
c ibornex=1 ==> Born exact in M_top
c ibornex=2 ==> Born in the M_top->infinity approximation
      common/cibornex/ibornex
c Higgs identity according to MC particle numbering scheme (=25); used 
c when writing the event file
      common/ci2part/ip4
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
c----------------------------------------------------------
c inloscale controls the reference scale in the NLO computation
      common/cinloscale/inloscale
c imcscale controls the reference scale in the MC subtraction terms
      common/cimcscale/imcscale
c----------------------------------------------------------
c Subprocesses: 'gg', 'qq', 'qg', corresponding to jproc=jproc0=1,2,3
c In the integration routines, loproc<=jproc<=maproc
      common/cwchproc/loproc,maproc
c Number of failures in flavour determination
      common/ciwrong/iwrong,iwrong1
c----------------------------------------------------------
c ifuntype=1 for shg5a, ifuntype=2 for shg5b
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

CCCC  ================================================================
CCCC  added by Fabian Stoeckli (fabian.stoeckli@cern.ch)
CCCC  in order to control bases output
      integer basesoutput
      character*100 stfilename
      common/fstbases/ basesoutput,stfilename
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


C
C------------------------------------------------------------------------
C                             START                                     -
C------------------------------------------------------------------------
c iinput=1 ==> all inputs are given by the user
c      iinput=0
CCC =================================================================
CCC   iinput = 2 mode added by fabian stoeckli
CCC   12.2.2007
CCC   in order to read in varaibles from CMSSW input-card
      iinput = 2
      mmdecide = 2
CCC =================================================================
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
      open(unit=11,file='higgslog',status=newver)
c
c Number of light flavours
      nl = 5
c Higgs identity according to MC particle numbering scheme
      ip4 = 25
c
CCC =================================================================
CCC   added by fabian stoeckli
CCC   read string in from file
      open(unit=33,file=stfilename,status='old')
CCC =================================================================

      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)
     #        'Enter prefix for name of BASES files'
         read (*,*) pref
      else
         read(33,*) pref
      endif
      write(11,*) ''''//pref(1:ifk88istrl(pref))//'''',
     # '  ! prefix for BASES files'
      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)
     #        'Enter prefix for name of event files'
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
c Fermi constant, from PDG2002
      gf=1.16639d-5
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
      if(iinput.lt.2) then
         write(*,*)' '
      write(*,*)'Enter Higgs mass, Higgs width, top mass (all GeV)'
         read(*,*)xmh0,gah,xmt
      else
         xmh0=mmxmh0
         gah = mmgah
         xmt=mmxmt
      endif
      write(11,'(3(1x,d10.4),1x,a)') xmh0,gah,xmt,'! M_H, Ga_H, M_top'
      xmh02 = xmh0**2
      xmt2 = xmt**2
      if(gah.ge.0.001d0)then
        iwidth=1
      else
        iwidth=0
      endif
      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)'Enter GammaX, M_H(min), M_H(max)'
       write(*,*)'  If GammaX>0, the Higgs mass is chosen in the range'
       write(*,*)'      M0-GammaX*width < M_H < M0+GammaX*width'
      write(*,*)'  If GammaX<0, the Higgs mass is chosen in the range'
      write(*,*)'             M_H(min) < M_H < M_H(max)'
         read(*,*)gammax,xmhl,xmhu
      else
         gammax=mmgammax
         xmhl=mmxmhl
         xmhu=mmxmhu
      endif
      write(11,'(3(1x,d10.4),1x,a)') gammax,xmhl,xmhu,
     #  '! GammaX, M_H(min), M_H(max)'
c Quark and gluon masses (must be consistent with Herwig)
      do i=-5,21
        xmass(i)=0.d0
      enddo
      if(iinput.lt.2) then
         write(*,*)' '
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
c Initialize parameters, such as labelling for parton processes
      call parsetphg()
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
      call getihpahg(part1,itmpih)
      ih1=itmpih
      call getihpahg(part2,itmpih)
      ih2=itmpih
c
      if(iinput.lt.2) then
         write(*,*)' '
      write(*,*)'Enter group name and id number for PDF set'
         read(*,*)gname,idpdfset
      else
         gname=mmgname
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
            write(*,*) 'Enter LambdaQCD_5, < 0 for default'
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
        write(*,*)'Enter 1 to set mu_0^2=M_H^2+pt_H^2'
        write(*,*)'      2 to set mu_0=M_H'
        write(*,*)'      3 to set mu_0=max(M_H,pt_H)'
        write(*,*)'      4 to set mu_0=max(M_H-pt,pt)'
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
        write(*,*)'Enter 1 to set mu_0^2=M_H^2+pt_H^2'
        write(*,*)'      2 to set mu_0=M_H'
        write(*,*)'      3 to set mu_0=max(M_H,pt_H)'
        write(*,*)'      4 to set mu_0=max(M_H-pt,pt)'
        write(*,*)' The default is 4'
        read(*,*) imcscale
        write(11,'(1(1x,i8),25x,a)') imcscale,
     #    '! 1->mu_0=mh+pt, 2->mu_0=mh'
      else
        imcscale=1
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
      write(*,*)'Enter 1 to compute Born exacly'
      write(*,*)
     #  '      2 to compute Born in the M_top->infinity limit'
         read(*,*)ibornex
      else
         ibornex=mmibornex
      endif
      write(11,'(1(1x,i8),25x,a)') ibornex,
     #  '! 1=Born exact, 2=M_top->inf'
c---------------------------------------------------------------
c Select subprocess
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*) 'Enter 1 for gg, 2 for qq, 3 for qg, 0 for all'
        read(*,*) iproc
        write(11,'(1x,i2,31x,a)') iproc,'! 1=gg, 2=qq, 3=qg, 0=all'
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
      loproc = 1
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
        xmh2=-1.d10
        if(gammax.gt.0)then
          if(gammax.lt.10.d0)gammax=10.d0
          xmhl2=(max(0.d0,xmh0-gammax*gah))**2
          xmhu2=(xmh0+gammax*gah)**2
        else
          xmhl2=xmhl**2
          xmhu2=xmhu**2
        endif
        xmhl2=max(100.d0,xmhl2)
        xmhu2=min(sh*0.95,xmhu2)
        bwxmdpl=xmhu2-xmh02
        bwxmdmn=xmh02-xmhl2
        if(bwxmdpl.lt.0.d0.or.bwxmdmn.lt.0.d0)then
          write(*,*)'Error in the computation of BW bounds'
          stop
        endif
        bwfmpl=atan(bwxmdpl/(xmh0*gah))
        bwfmmn=atan(bwxmdmn/(xmh0*gah))
        bwdelf=(bwfmpl+bwfmmn)/pi
      else
        ndim=3
        nwild=3
        xmh2=xmh02
        roh=xmh2/sh
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
          call run_bases(shg5afw,fnamea,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        else
          call run_bases(shg5azw,fnamea,ndim,nwild,ncl3,it1,it2,
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
          call run_bases(shg5bfw,fnameb,ndim,nwild,ncl3,it1,it2,
     #      ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #      ibswrite,ibscall)
        else
          call run_bases(shg5bzw,fnameb,ndim,nwild,ncl3,it1,it2,
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
        if(basesoutput.gt.0) then
           if(evgen)then
              write(*,*)'  '
              write(*,*)
     #  'The following number of events will be generated'
              write(*,*)'# events[a]:',mx_of_evta
              write(*,*)'# events[b]:',mx_of_evtb
           endif
           write(*,*)'  '
           write(*,*)
     #  'Estimated fractions of events with negative weights'
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
        call hgend(22)
        if(iwidth.eq.1)then
          call run_spring(shg5afw,fnamea,mx_of_evta,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        else
          call run_spring(shg5azw,fnamea,mx_of_evta,maxtrials,
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
        call hgend(22)
        if(iwidth.eq.1)then
          call run_spring(shg5bfw,fnameb,mx_of_evtb,maxtrials,
     #                    nevts,ntrls,ndim,nwild,iseed)
        else
          call run_spring(shg5bzw,fnameb,mx_of_evtb,maxtrials,
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
        write(ioutput,802)1600,'--> 1600+ID=Higgs'
        write(ioutput,813)xmh0,gah,xmt,'--> M_H, Ga_H, M_top'
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
          call whichohg(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
          call retrieve_evenhg(iunit,ii,dummy)
          call store_evengh(ioutput,xmone)
        enddo
        call crossckehg(itot,mx_of_evta,mx_of_evtb)
        close(iunita)
        close(iunitb)
        close(ioutput)
 111    continue
      endif
      if(basesoutput.gt.0) then
         write(*,*) 'Total for fully inclusive'
         write(*,200)ih1,ih2,ndns1,ndns2,nl,xlam
         write(*,201) 'tot'
         write(*,300)ecm,xmh0,gah,xfh,xren,avtot,dtot
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
 813  format(3(1x,d10.4),1x,a)
 804  format(6(1x,d10.4),1x,a)
 805  format(2(1x,a4),1x,a)
 806  format(1x,a8,1x,i6,1x,a)
 807  format(1x,d10.4,1x,a2,1x,a)
 811  format(1x,a3,1x,a)
      end


CCCC  ========================================================
CCCC  same as in mainll .....

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


      subroutine hgend(iunit)
      ios = 0    
      dowhile(ios.eq.0)
         read(unit=iunit,fmt='(1x)',iostat=ios)
      enddo                        
      end


      subroutine getihpahg(part,ih)
c Converts particle naming conventions, for Herwig to MLM
      implicit real * 8 (a-h,o-z)
      character * 4 part
c
      ih=-100
      if(part.eq.'P   ')then
        ih=1
      elseif(part.eq.'PBAR')then
        ih=-1
      elseif(part.eq.'N   ')then
        ih=2
      elseif(part.eq.'NBAR')then
        ih=-2
      elseif(part.eq.'GAMA')then
        ih=4
      elseif(part.eq.'E-  ')then
        ih=5
      else
        write(*,*)'Error in getihpahg'
        write(*,*)'No such particle in MLM:',part
        stop
      endif
      return
      end


      subroutine strfhg(x1,x2,sf)
c Return parton densities through the matrix
c  sf(idr,jproc,itype), with the following conventions:
c  idr=1 -> gg, qqbar, qg
c  idr=2 -> 0,  0    , qbarg
c  idr=3 -> 0,  qbarq, gq
c  idr=4 -> 0,  0    , gqbar
c  jproc=1,2,3 -> gg, qqbar, qg processes respectively
c  itype -> identifies the individual contribution to a given jproc
c
      implicit none
      real*4 fh1x1(-5:5),fh2x2(-5:5),smuf2h1,smuf2h2
      real*8 pi,x1,x2,sf(4,3,5)
      integer ih1,ih2,ndns1,ndns2,i,jproc,itype
      parameter(pi=3.14159265358979312D0)
      include 'hgscblks.h'
      common/strfhg0/ih1,ih2,ndns1,ndns2
      integer ipdfscale
      common/cipdfscale/ipdfscale
c ipdfscale=1 --> use NLO factorization scale
c ipdfscale=2 --> use MC factorization scale
c
      do i=1,4
        do jproc=1,3
          do itype=1,5
            sf(i,jproc,itype)=0.d0
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
        write(*,*)'Fatal error in strfhg: unknown ipdfscale',ipdfscale
        stop
      endif
c
      call mlmpdf(ndns1,ih1,smuf2h1,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smuf2h2,sngl(x2),fh2x2,5)
c
      sf(1,1,1) = dble(fh1x1(0)*fh2x2(0))
      sf(2,1,1) = 0.d0
      sf(3,1,1) = 0.d0
      sf(4,1,1) = 0.d0
c
      do itype=1,nl
        sf(1,2,itype) = dble(fh1x1( itype) * fh2x2(-itype))
        sf(2,2,itype) = 0.d0
        sf(3,2,itype) = dble(fh1x1(-itype) * fh2x2( itype))
        sf(4,2,itype) = 0.d0
      enddo
c
      do itype=1,nl
        sf(1,3,itype) = dble(fh1x1( itype) * fh2x2(     0))
        sf(2,3,itype) = dble(fh1x1(-itype) * fh2x2(     0))
        sf(3,3,itype) = dble(fh1x1(     0) * fh2x2( itype))
        sf(4,3,itype) = dble(fh1x1(     0) * fh2x2(-itype))
      enddo
c
      return
      end
c
c
c Begin of cross section routines. In the case of Higgs finite width,
c the cross section is obtained with the formula
c       sigma=\int dM^2 BW(M,M0,Ga) sigma_zw(M^2)
c sigma_zw(M^2) being the cross section relevant to zero width computed
c at a mass value of M, and 
c       BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c is the normalized (i.e., with integral equal to 1) Breit-Wigner function,
c which reduces to delta(M^2-M0^2) for Ga --> 0.
c The integration range in M^2 is controlled by gammax, xmhl, and xmhu
c
c
      function shg5azw(xx)
c H events
      implicit none
      real * 8 shg5azw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(3)
      include 'hgscblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,thg5a
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi
c
      xjac = 1
      roh = xmh2/sh
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
      shg5azw = thg5a(s,xii,yi,xjac)
      return
      end


      function shg5afw(xx)
c H events
      implicit none
      real * 8 shg5afw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(4)
      include 'hgscblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,xbwmahg2,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,thg5a
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi, xx(4) --> xmh2
c
      xjac = 1
c 
c The following change of variable includes the Breit-Wigner function
      xmh2=xbwmahg2(xx(4))
      xjac=bwdelf*xjac
      roh=xmh2/sh
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
      shg5afw = thg5a(s,xii,yi,xjac)
      return
      end


      function thg5a(xs,xxii,xyi,xjac)
      implicit none
      real * 8 thg5a,thg5as,thg5az,xs,xxii,xyi,s,xii,yi,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      s = xs
      xii = xxii
      yi = xyi
      if(isubttype.eq.0)then
        tmp=thg5as(s,xii,yi,xjac)
      elseif(isubttype.eq.1)then
        tmp=thg5az(s,xii,yi,xjac)
      else
        write(*,*)'Fatal error in thg5a:',isubttype
        stop
      endif
      thg5a=tmp
      return
      end


      function thg5as(xs,xxii,xyi,xjac)
c H events
      implicit none
      real * 8 thg5as,xs,xxii,xyi,xjac
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
      include 'hgscblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 sf(4,3,5),vv(4,3,5),vvs(4,3,5)
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
     # xnorm,zg2_nlo,zgmu2_nhg,zg2_mc,zgmu2_hg,
     # ffunval5,ffunction5,ytmp,zhwfct,zherw_spl,x1t,x1sohg,
     # x2t,x2sohg,x1x2j,x1x2jhg,zherw_smn,xsum,dummy,xint,xitmp
      integer jproc,loproc,maproc,jproc0,ipdfscale,i,itype,
     # iret,i2b,itoosoftkhg
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
      xphsp_ns = xjac * 2*pi/(64*pi2) * xmh2/(x**2*sh)
      xphsp_s = xjac * 2*pi/(64*pi2) * xmh2/(x*sh)
c
      do jproc=1,3
        do i=1,4
          do itype=1,5
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
        call invhg(xmh2,s,xii,yi,t,u)
        xnorm = gf * xphsp_ns/xii*( 1/(1-yi) + 1/(1+yi) )
        zg2_nlo = zgmu2_nhg()
        ipdfscale=1
        call strfhg(x1,x2,sf)
        do jproc=loproc,maproc
          jproc0=jproc
          call f2body(s,xii,yi,t,u,res)
          do i=1,4
            do itype=1,nl
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*res(i)*zg2_nlo**3*xnorm
            enddo
          enddo
        enddo
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
        ffunval5 = ffunction5(x,yi)
        if(ffunval5.ne.0.d0)then
          ipdfscale=2
          zg2_mc = zgmu2_hg()
          ytmp=1.d0
          zhwfct=zherw_spl(x,yi)
          x1t=x1sohg(x1,x2,x,yi)/zhwfct
          x2t=x2sohg(x1,x2,x,yi)
          if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jhg(x1,x2,x,yi)/zhwfct
            call strfhg(x1t,x2t,sf)
            do jproc=loproc,maproc
              jproc0=jproc
              call xmcsuhg(jproc,s,xii,yi,ytmp,xmcsec)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)-
     #              sf(i,jproc,itype)*xmcsec(i)*x1x2j*
     #              zg2_nlo**2*zg2_mc*xnorm
                enddo
              enddo
            enddo
          endif
          ytmp=-1.d0
          zhwfct=zherw_smn(x,yi)
          x1t=x1sohg(x1,x2,x,yi)
          x2t=x2sohg(x1,x2,x,yi)/zhwfct
          if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jhg(x1,x2,x,yi)/zhwfct
            call strfhg(x1t,x2t,sf)
            do jproc=loproc,maproc
              jproc0=jproc
              call xmcsuhg(jproc,s,xii,yi,ytmp,xmcsec)
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)-
     #              sf(i,jproc,itype)*xmcsec(i)*x1x2j*
     #              zg2_nlo**2*zg2_mc*xnorm
                enddo
              enddo
            enddo
          endif
        else
          call genvetosc()
        endif
      endif
c
      call checkhg(xsum,dummy,iret)
      if(iret.eq.1)then
        call invhg(xmh2,s,xii,yi,t,u)
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkhg()
        if(i2b.eq.1)then
          xitmp=0.d0
          ytmp=1.d0
          call invhg(xmh2,sx,xitmp,ytmp,t,u)
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      thg5as=abs(xint)
c
      return
      end


      function thg5az(xs,xxii,xyi,xjac)
      implicit real*8 (a-z)
      thg5az=0
      return
      end


      function shg5bzw(xx)
c S events
      implicit none
      real * 8 shg5bzw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(3)
      include 'hgscblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,tohgb
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi
c
      xjac = 1
      roh = xmh2/sh
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
      shg5bzw = tohgb(s,xii,yi,xjac)
      return
      end


      function shg5bfw(xx)
c S events
      implicit none
      real * 8 shg5bfw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(4)
      include 'hgscblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,ttt,th,xii,yi,omega,romxi,xbwmahg2,
     # ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,tohgb
c
c xx(1) --> ycm, xx(2) --> xii, xx(3) --> yi, xx(4) --> xmh2
c
      xjac = 1
c 
c The following change of variable includes the Breit-Wigner function
      xmh2=xbwmahg2(xx(4))
      xjac=bwdelf*xjac
      roh=xmh2/sh
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
      shg5bfw = tohgb(s,xii,yi,xjac)
      return
      end


      function tohgb(xs,xxii,xyi,xjac)
      implicit none
      real * 8 tohgb,tohgbs,tohgbz,xs,xxii,xyi,s,xii,yi,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      s = xs
      xii = xxii
      yi = xyi
      if(isubttype.eq.0)then
        tmp=tohgbs(s,xii,yi,xjac)
      elseif(isubttype.eq.1)then
        tmp=tohgbz(s,xii,yi,xjac)
      else
        write(*,*)'Fatal error in tohgb:',isubttype
        stop
      endif
      tohgb=tmp
      return
      end


      function tohgbs(xs,xxii,xyi,xjac)
c S events
      implicit none
      real * 8 tohgbs,xs,xxii,xyi,xjac
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
      include 'hgscblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 sf(4,3,5),vv(4,3,5),vvs(4,3,5)
      real * 8 res(4),resc(4),xmcsec(4)
      common/cvv/vv
      common/cvvs/vvs
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 xints(4,3),xborn(4,3),xcs(4,3),xsv(4,3)
      real * 8 s,xii,yi,x,sx,xphspb,xphsp_ns,xphsp_s,x1,x2,t,u,
     # xnorm,zg2_nlo,zgmu2_nhg,zg2_mc,zgmu2_hg,
     # ffunval5,ffunction5,ytmp,zhwfct,zherw_spl,x1t,x1sohg,
     # x2t,x2sohg,x1x2j,x1x2jhg,zherw_smn,xsum,dummy,xint,xitmp,
     # xintcp,xlmude,xnormc,xcplus,xintcm,xcminus,xintffs,xnormb,
     # xnormsv,xintcps,xintcms
      integer jproc,loproc,maproc,jproc0,ipdfscale,i,itype,
     # iret,ibornex
      common/cjproc/jproc0
      common/cwchproc/loproc,maproc
      common/cibornex/ibornex
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
      xphsp_ns = xjac * 2*pi/(64*pi2) * xmh2/(x**2*sh)
      xphsp_s = xjac * 2*pi/(64*pi2) * xmh2/(x*sh)
c
      do jproc=1,3
        do i=1,4
          do itype=1,5
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
        call invhg(xmh2,s,xii,yi,t,u)
        xnorm = gf * xphsp_ns/xii*( 1/(1-yi) + 1/(1+yi) )
        ipdfscale=2
        zg2_nlo = zgmu2_nhg()
        zg2_mc = zgmu2_hg()
        ytmp=1.d0
        zhwfct=zherw_spl(x,yi)
        x1t=x1sohg(x1,x2,x,yi)/zhwfct
        x2t=x2sohg(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jhg(x1,x2,x,yi)/zhwfct
          call strfhg(x1t,x2t,sf)
          do jproc=loproc,maproc
            jproc0=jproc
            call xmcsuhg(jproc,s,xii,yi,ytmp,xmcsec)
            do i=1,4
              do itype=1,nl
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*xmcsec(i)*x1x2j*
     #              zg2_nlo**2*zg2_mc*xnorm
              enddo
            enddo
          enddo
        endif
        ytmp=-1.d0
        zhwfct=zherw_smn(x,yi)
        x1t=x1sohg(x1,x2,x,yi)
        x2t=x2sohg(x1,x2,x,yi)/zhwfct
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jhg(x1,x2,x,yi)/zhwfct
          call strfhg(x1t,x2t,sf)
          do jproc=loproc,maproc
            jproc0=jproc
            call xmcsuhg(jproc,s,xii,yi,ytmp,xmcsec)
            do i=1,4
              do itype=1,nl
                vv(i,jproc,itype)=vv(i,jproc,itype)+
     #            sf(i,jproc,itype)*xmcsec(i)*x1x2j*
     #              zg2_nlo**2*zg2_mc*xnorm
              enddo
            enddo
          enddo
        endif
      else
        call genvetosc()
      endif
c
c Counter-event :
c
      ipdfscale=1
      if(yi.gt.1-delta) then
         ytmp = 1.d0
         x1t = x1sohg(x1,x2,x,yi)/x
         x2t = x2sohg(x1,x2,x,yi)
         if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jhg(x1,x2,x,yi)/x
            call invhg(xmh2,s,xii,ytmp,t,u) 
            zg2_nlo = zgmu2_nhg()
            call strfhg(x1t,x2t,sf)
            xnorm = x1x2j * zg2_nlo**3 * gf * xphsp_ns
            do jproc=loproc,maproc
              jproc0=jproc
              call f2body(s,xii,ytmp,t,u,res)
              xintcp = xnorm * 1.d0/xii*( - 1/(1-yi) )
c Adding the collinear contribution
              xlmude = log(s/xmuf2h1)+log(delta/2)
              xnormc = x1x2j * zg2_nlo**3 * gf * xphspb /
     #                 (8*pi2 * delta)
              xcplus = xnormc / xii
              call f2b_coll(s,xii,xii,ytmp,xlmude,resc)
              do i=1,4
                do itype=1,nl
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
         x1t = x1sohg(x1,x2,x,yi)
         x2t = x2sohg(x1,x2,x,yi)/x
         if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jhg(x1,x2,x,yi)/x
            call invhg(xmh2,s,xii,ytmp,t,u) 
            zg2_nlo = zgmu2_nhg()
            call strfhg(x1t,x2t,sf)
            xnorm = x1x2j * zg2_nlo**3 * gf * xphsp_ns
            do jproc=loproc,maproc
              jproc0=jproc
              call f2body(s,xii,ytmp,t,u,res)
              xintcm = xnorm * 1.d0/xii*( - 1/(1+yi) )
c Adding the collinear contribution
              xlmude = log(s/xmuf2h2)+log(delta/2)
              xnormc = x1x2j * zg2_nlo**3 * gf * xphspb /
     #                 (8*pi2 * delta)
              xcminus = xnormc / xii
              call f2b_coll(s,xii,xii,ytmp,xlmude,resc)
              do i=1,4
                do itype=1,nl
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
         x1t = x1sohg(x1,x2,x,yi)
         x2t = x2sohg(x1,x2,x,yi)
         if(x1t.lt.1.and.x2t.lt.1)then
            x1x2j = x1x2jhg(x1,x2,x,yi)
            call invhg(xmh2,sx,xitmp,yi,t,u)
            zg2_nlo = zgmu2_nhg()
            xnorm = x1x2j * zg2_nlo**3 * gf * xphsp_s
            do jproc=loproc,maproc
              jproc0=jproc
              do i=1,4
                xints(i,jproc) = 0.d0
                xborn(i,jproc) = 0.d0
                xcs(i,jproc) = 0.d0
                xsv(i,jproc) = 0.d0
              enddo
              call f2body(sx,xitmp,yi,t,u,res)
              xintffs = - xnorm * 1.d0/xii*
     #                  ( 1/(1-yi) + 1/(1+yi) )
              do i=1,4
                xints(i,jproc) = xintffs*res(i)
              enddo
c Adding the Born term
              xnormb = x1x2j * zg2_nlo**2 * gf * xphspb /(2*xicut)
              call f1born(sx,jproc0,ibornex,'bo',res)
              do i=1,4
                xborn(i,jproc) = xnormb*res(i)
              enddo
c Adding the soft-virtual contribution
              xnormsv = x1x2j * zg2_nlo**3 * gf * xphspb / 
     #                 (8*pi2 * 2*xicut)
              call f1sv(sx,jproc0,res)
              do i=1,4
                xsv(i,jproc) = xnormsv*res(i)
              enddo
            enddo
            if(yi.gt.1-delta) then
               ytmp = 1.d0
               call invhg(xmh2,sx,xitmp,ytmp,t,u)
               zg2_nlo = zgmu2_nhg()
               xnorm = x1x2j * zg2_nlo**3 * gf * xphsp_s
               do jproc=loproc,maproc
                 jproc0=jproc
                 call f2body(sx,xitmp,ytmp,t,u,res)
                 xintcps = - xnorm * 1.d0/xii*( - 1/(1-yi) )
                 do i=1,4
                   xints(i,jproc) = xints(i,jproc)+xintcps*res(i)
                 enddo
c Adding the collinear contribution
                 xlmude = log(sx/xmuf2h1)+log(delta/2)
                 xnormc = x1x2j * zg2_nlo**3 * gf * xphspb /
     #                    (8*pi2 * delta)
                 call f2b_coll(sx,xii,xitmp,ytmp,xlmude,res)
                 do i=1,4
                   xcs(i,jproc) = xcs(i,jproc) - xnormc / xii * res(i)
                 enddo
               enddo
            endif
            if(yi.lt.-1+delta) then
               ytmp = -1.d0
               call invhg(xmh2,sx,xitmp,ytmp,t,u)
               zg2_nlo = zgmu2_nhg()
               xnorm = x1x2j * zg2_nlo**3 * gf * xphsp_s
               do jproc=loproc,maproc
                 jproc0=jproc
                 call f2body(sx,xitmp,ytmp,t,u,res)
                 xintcms = - xnorm * 1.d0/xii*( - 1/(1+yi) )
                 do i=1,4
                   xints(i,jproc) = xints(i,jproc)+xintcms*res(i)
                 enddo
c Adding the collinear contribution
                 xlmude = log(sx/xmuf2h2)+log(delta/2)
                 xnormc = x1x2j * zg2_nlo**3 * gf * xphspb /
     #                    (8*pi2 * delta)
                 call f2b_coll(sx,xii,xitmp,ytmp,xlmude,res)
                 do i=1,4
                   xcs(i,jproc) = xcs(i,jproc) - xnormc / xii * res(i)
                 enddo
               enddo
            endif
            call strfhg(x1t,x2t,sf)
            do jproc=loproc,maproc
              jproc0=jproc
              do i=1,4
                do itype=1,nl
                  vv(i,jproc,itype)=vv(i,jproc,itype)+
     #              sf(i,jproc,itype)*( xints(i,jproc)+xsv(i,jproc)+
     #                                  xborn(i,jproc)+xcs(i,jproc) )
                enddo
              enddo
            enddo
         endif
      endif
c
      call checkhg(xsum,dummy,iret)
      if(iret.eq.1)then
        xitmp = 0.d0
        ytmp = 1.d0
        call invhg(xmh2,sx,xitmp,ytmp,t,u)
        x1t = x1sohg(x1,x2,x,yi)
        x2t = x2sohg(x1,x2,x,yi)
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
      tohgbs=abs(xint)
c
      return
      end


      function tohgbz(xs,xxii,xyi,xjac)
      implicit real*8 (a-z)
      tohgbz=0
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
c
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
c
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
      subroutine sprfhg()
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
      call xohg(iret)
      if(iret.eq.1)then
        if(i0.lt.1.or.i0.gt.4)then
          write(*,*)'Fatal error in sprfhg'
          stop
        endif
        ycm0=ycm
        call getx1x2(tau,ycm0)
        call getmhg(tau,ycm0)
        call store_evengh(iunit,xone)
      endif
      return
      end


C      subroutine getx1x2(tau,ycm)
C      implicit none
C      real*8 tau,ycm,x1,x2,stau,ey
C      common/cx1x2/x1,x2
c
C      stau=sqrt(tau)
C      ey=exp(ycm)
C      x1=stau*ey
C      x2=stau/ey
C      return
C      end


      subroutine getmhg(xtau,xycm)
      implicit none
      real*8 xtau,xycm
      include 'hgscblks.h'
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
      if(ichkmom.eq.0)call checkmhg(xmom_cm,sh,0.d0,3)
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
        call put_on_shehg(ycm,ycmnew)
        do i=1,4
          call boost(-ycmnew,
     #         xmom_prime(i,1),xmom_prime(i,2),
     #         xmom_prime(i,3),xmom_prime(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      endif
      if(ichkmom.eq.0)call checkmhg(xmom_lb,sh,-ycmnew,2)
      call momnewformhg()
      return
      end


      subroutine momnewformhg()
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
c
C      b1=a1
C      b2=a2
C      b3=a3*cosh(y)-a4*sinh(y)
C      b4=a4*cosh(y)-a3*sinh(y)
C      return
C      end


C      subroutine transrot(cth,sth,xpt1,xpt2)
C      implicit none
C      real*8 cth,sth,xpt1,xpt2,pt1,pt2
c
C      pt1=xpt1
C      pt2=xpt2
C      xpt1=pt1*cth+pt2*sth
C      xpt2=-pt1*sth+pt2*cth
C      return
C      end


      subroutine put_on_shehg(ycm,ycmnew)
      implicit none
      include 'hgscblks.h'
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
c given in the partonic CM frame. j=4 is the energy for MC@NLO versions
c up to 2.31, the mass for version 3.1 onwards. i=1,2 are the incoming
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
      xmss(4) = sqrt(xmh2)
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
      call getxmss(shat,ycm,
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
      if(ichkmom.eq.0)call checkmhg(xmom_prime,shat,0.d0,4)
      return
      end


C      subroutine getxmss(shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23)
Cc After putting the momenta on shell, the two incoming partons may
Cc travel in the same direction. This routine prevents this to happen,
Cc redefining Herwig masses if necessary
C      implicit none
C      real*8 shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23
C      real*8 tiny,fact,sqs,xm1s,xm2s,xkp2prime_norm2,xkp2prime_norm,
C     #  ytmp,e1,e2,p13p,p23p,s1p,s2p,xif,sol
C      integer iflag,idone,ileg
C      parameter (fact=0.98d0)
C      parameter (tiny=1.d-6)
Cc
C      sqs=sqrt(shat)
C      xm1s=xm1
C      xm2s=xm2
C      ytmp=-ycm
C      idone=0
C 100  continue
C      xkp2prime_norm2=( shat-2*(xm1**2+xm2**2)+
C     #                  (xm1**2-xm2**2)**2/shat )/4.d0
C      xkp2prime_norm=sqrt(xkp2prime_norm2)
C      if(sign(1.d0,p13cm).ne.1.d0.or.sign(1.d0,p23cm).ne.-1.d0)then
C        write(*,*)'Error # 0 in getxmss'
C        stop
C      endif
C      p13=xkp2prime_norm
C      p23=-xkp2prime_norm
C      e1=sqrt(p13**2+xm1**2)
C      e2=sqrt(p23**2+xm2**2)
C      p13p=p13*cosh(ytmp)-e1*sinh(ytmp)
C      p23p=p23*cosh(ytmp)-e2*sinh(ytmp)
C      s1p=sign(1.d0,p13p)
C      s2p=sign(1.d0,p23p)
C      iflag=0
C      if(s1p.eq.1.d0 .and. s2p.eq.-1.d0)then
C        iflag=1
C      elseif(s1p.eq.-1.d0 .and. s2p.eq.-1.d0)then
C        if(ytmp.lt.0.d0)then
C          write(*,*)'Wrong y sign, # 1'
C          stop
C        endif
C        ileg=1
C        xif=xm2**2/shat
C      elseif(s1p.eq.1.d0 .and. s2p.eq.1.d0)then
C        if(ytmp.gt.0.d0)then
C          write(*,*)'Wrong y sign, # 2'
C          stop
C        endif
C        ileg=2
C        xif=xm1**2/shat
C      else
C        write(*,*)'Error # 1 in getxmss'
C        stop
C      endif
C      if(iflag.eq.0)then
C        sol=xif+cosh(2*ytmp)-
C     #      sqrt(2.d0)*cosh(ytmp)*sqrt(cosh(2*ytmp)-1+2*xif)
C        if(sol.le.0.d0.or.idone.eq.1)then
Cc The procedure failed; pass the massless event to Herwig, and let Herwig
Cc deal with it
C          xm1=0.d0
C          xm2=0.d0
C          p13=sqs/2.d0
C          p23=-sqs/2.d0
C          return
C        endif
C        if(ileg.eq.1)then
C          xm1=fact*sqrt(sol*shat)
C          if(xm1.gt.xm1s)then
C            write(*,*)'Mass # 1 too large in getxmss'
C            stop
C          endif
C        elseif(ileg.eq.2)then
C          xm2=fact*sqrt(sol*shat)
C          if(xm2.gt.xm2s)then
C            write(*,*)'Mass # 2 too large in getxmss'
C            stop
C          endif
C        else
C          write(*,*)'Error # 2 in getxmss'
C          stop
C        endif
C        idone=1
C        goto 100
C      endif
C      return
C      end


C      subroutine fillvec(p1,p2,p3,p4,ytmp)
C      implicit none
C      real*8 p1,p2,p3,p4,ytmp(1:4)
Cc
C      ytmp(1)=p1
C      ytmp(2)=p2
C      ytmp(3)=p3
C      ytmp(4)=p4
C      return
C      endC

C
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
C
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
c
C      dotprod=p4*q4-p1*q1-p2*q2-p3*q3
C      return
C      end


C      function xnorm_3(p)
c Evaluates the norm of the spatial component of a four-momentum
c The result is positive by definition, regardless of the 4-metric
C      implicit none
C      real*8 xnorm_3,p(1:4),tmp
c
C      tmp=p(1)*p(1)+p(2)*p(2)+p(3)*p(3)
C      xnorm_3=sqrt(tmp)
C      return
C      end


C      subroutine vecsum(p,pfact,q,qfact,r)
c Weighted sum of the four-vectors p and q. The result is r
C      implicit none
C      integer i
C      real*8 p(1:4),q(1:4),r(1:4),pfact,qfact
Cc
C      do i=1,4
C        r(i)=pfact*p(i)+qfact*q(i)
C      enddo
C      return
C      end


      subroutine xohg(iret)
c This routine is called by sprfhg; it determines, on statistical
c basis, which partonic process has been generated.
c It also counts the number of unlike sign events (iwrong), and the number
c of these events (iwrong1) for which the relative difference between
c unlike signs exceeds 5%. If all the entries of vv are equal to zero,
c iret is set equal to 0 (by checkhg), and no operation is performed
      implicit none
      integer iret,iretvv,iretvvs,iproc,iproclo,iprocma,i,itype,
     #  iwh,iflag,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o
      real*8 wwx(4,3,5),xsum,xsumabs,xsumvvs,xsumabsvvs,xstsign,
     #  xg,wh,rmax,fk88random
      integer loproc,maproc
      common/cwchproc/loproc,maproc
      integer ifuntype
      common/cifuntype/ifuntype
      real*8 vv(4,3,5)
      common/cvv/vv
      real*8 vvs(4,3,5)
      common/cvvs/vvs
      integer iwrong,iwrong1
      common/ciwrong/iwrong,iwrong1
      integer i0,jproc0,itype0
      common/cidproc/i0,jproc0,itype0
      integer ivbhpro(4,3,5)
      common/civbhpro/ivbhpro
      integer idp1(4,3,5),idp2(4,3,5),idp3(4,3,5)
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
      call checkhg(xsum,xsumabs,iretvv)
      call checkhgs(xsumvvs,xsumabsvvs,iretvvs)
      if(iretvv.eq.0.and.iretvvs.eq.1)then
        write(6,*)'Fatal error in xohg:',iretvv,iretvvs
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
          write(*,*)'Fatal error in xohg: ifuntype=',ifuntype
          stop
        endif
        if(iretvvs.eq.1)then
          xsum=xsumvvs
          xsumabs=xsumabsvvs
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,nl
                wwx(i,iproc,itype)=vvs(i,iproc,itype)
              enddo
            enddo
          enddo
        else
          do iproc=iproclo,iprocma
            do i=1,4
              do itype=1,nl
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
            do itype=1,nl
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
          write(*,*)'Fatal error in xohg',i0,jproc0,itype0
          stop
        endif
        if(jproc0.eq.2.and.ifuntype.eq.2)then
          write(*,*)'Fatal error in xohg: qq in S events'
          stop
        endif
        ihpro=ivbhpro(i0,jproc0,itype0)
        i1=idp1(i0,jproc0,itype0)
        i2=idp2(i0,jproc0,itype0)
        i3=idp3(i0,jproc0,itype0)
        call parcrossihg(jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o)
        i1hpro=i1hproo
        ip1=ip1o
        ip2=ip2o
        ip3=ip3o
      endif
      return
      end


      subroutine parcrossihg(jproc0,ihpro,i1,i2,i3,
     #                       i1hproo,ip1o,ip2o,ip3o)
      implicit none
      integer jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o,iallzero
      parameter (iallzero=1)
      integer ifuntype
      common/cifuntype/ifuntype
c
      if( (ifuntype.eq.1) .or. (ifuntype.eq.2.and.jproc0.eq.1) )then
        i1hproo=ihpro
        ip1o=i1
        ip2o=i2
        ip3o=i3
      elseif(ifuntype.eq.2.and.jproc0.eq.3)then
        if(ihpro.eq.401.or.ihpro.eq.403.or.ihpro.eq.407)then
          write(*,*)'Error #1 in parcrossihg:',ihpro,i1,i2,i3
          stop
        endif
        i1hproo=407
        ip1o=21
        ip2o=21
        ip3o=21
      else
        write(*,*)'parcrossihg: do not know what to do'
        write(*,*)ifuntype,jproc0
        stop
      endif
      call parcheckfhg(i1hproo,ip1o,ip2o,ip3o,iallzero)
      return
      end


      subroutine checkhg(xsum,xsumabs,iret)
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
      real * 8 vv(4,3,5)
      common/cvv/vv
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=loproc,maproc
        do i=1,4
          do itype=1,nl
            if(vv(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vv(i,jproc,itype)
            xsumabs=xsumabs+abs(vv(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      subroutine checkhgs(xsum,xsumabs,iret)
c identical to checkhg, except for the fact that works on vvs instead of vv,
c and jproc is not fixed
      implicit none
      integer jproc,iret,i,itype
      integer nl
      common/nl/nl
      real * 8 vvs(4,3,5)
      common/cvvs/vvs
      real * 8 xsum,xsumabs
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do jproc=1,2
        do i=1,4
          do itype=1,nl
            if(vvs(i,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vvs(i,jproc,itype)
            xsumabs=xsumabs+abs(vvs(i,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      function itoosoftkhg()
c Returns 1 when a three-body kinematics can be safely approximated
c with a two-body kinematics. It is useful when three-body NLO configurations
c are obtained, which cannot be produced through showering
      implicit none
      integer itoosoftkhg,itmp
c
      itmp=0
      itoosoftkhg=itmp
      return
      end
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
      subroutine invhg(xmh2,s,xii,yi,t,u)
      implicit none
      real * 8 xmh2,s,xii,yi,t,u,sqs,syi,sqso2
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


      function xbwmahg2(t)
c Returns the Higgs mass squared, given 0<t<1, the nominal mass (xmh0),
c and the mass range (implicit in bwdelf and bwfmmn). This function
c is the inverse of F(M^2), where
c   F(M^2)=\int_{xmhl2}^{M^2} ds BW(sqrt(s),M0,Ga)
c   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c and therefore eats up the Breit-Wigner when changing integration 
c variable M^2 --> t
      implicit none
      real*8 xbwmahg2,t
      include 'hgscblks.h'
      real*8 pi,xmh0
      parameter (pi=3.1415926535897932d0)
c
      xmh0=sqrt(xmh02)
      xbwmahg2=xmh02+xmh0*gah*tan(pi*bwdelf*t-bwfmmn)
      return
      end


      subroutine checkmhg(xmom,smax,ybst,iflag)
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
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
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
      subroutine parsetphg()
      implicit none
      integer jproc,i,itype
      integer imapp(0:5)
      integer ivbhpro(4,3,5)
      integer idp1(4,3,5),idp2(4,3,5),idp3(4,3,5)
      common/cimapp/imapp
      common/civbhpro/ivbhpro
      common/cidpart/idp1,idp2,idp3
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
c identical to i1hpro (see the routine store_evengh)
      do i=1,4
        do jproc=1,3
          do itype=1,5
            ivbhpro(i,jproc,itype)=0
          enddo
        enddo
      enddo
c 
      ivbhpro(1,1,1)=407
      do itype=1,5
        ivbhpro(1,2,itype)=401
        ivbhpro(3,2,itype)=403
      enddo
      do itype=1,5
        ivbhpro(1,3,itype)=402
        ivbhpro(2,3,itype)=404
        ivbhpro(3,3,itype)=405
        ivbhpro(4,3,itype)=406
      enddo
c
c idpX returns the flavour of parton number X (1=coming from the left,
c 2=coming from the right, 3=outgoing) in the process associated to the
c entries. The labelling scheme of PDG has been used
      do i=1,4
        do jproc=1,3
          do itype=1,5
            idp1(i,jproc,itype)=0
            idp2(i,jproc,itype)=0
            idp3(i,jproc,itype)=0
          enddo
        enddo
      enddo
c
      idp1(1,1,1)=21
      idp2(1,1,1)=21
      idp3(1,1,1)=21
c
      do itype=1,5
        idp1(1,2,itype)=imapp(itype)
        idp1(3,2,itype)=-imapp(itype)
c
        idp1(1,3,itype)=imapp(itype)
        idp1(2,3,itype)=-imapp(itype)
        idp1(3,3,itype)=21
        idp1(4,3,itype)=21
c
        idp2(1,2,itype)=-imapp(itype)
        idp2(3,2,itype)=imapp(itype)
c
        idp2(1,3,itype)=21
        idp2(2,3,itype)=21
        idp2(3,3,itype)=imapp(itype)
        idp2(4,3,itype)=-imapp(itype)
c
        idp3(1,2,itype)=21
        idp3(3,2,itype)=21
c
        idp3(1,3,itype)=imapp(itype)
        idp3(2,3,itype)=-imapp(itype)
        idp3(3,3,itype)=imapp(itype)
        idp3(4,3,itype)=-imapp(itype)
      enddo
c
      call parcheckphg()
      return
      end


      subroutine parcheckphg()
      implicit none
      integer iallzero,i,jproc,itype,ihpro,i1,i2,i3
      parameter (iallzero=0)
      integer ivbhpro(4,3,5)
      common/civbhpro/ivbhpro
      integer idp1(4,3,5),idp2(4,3,5),idp3(4,3,5)
      common/cidpart/idp1,idp2,idp3
c
      call parcheckinhg()
      do i=1,4
        do jproc=1,3
          do itype=1,5
            ihpro=ivbhpro(i,jproc,itype)
            i1=idp1(i,jproc,itype)
            i2=idp2(i,jproc,itype)
            i3=idp3(i,jproc,itype)
            call parcheckfhg(ihpro,i1,i2,i3,iallzero)
          enddo
        enddo
      enddo
      return
      end


      subroutine parcheckfhg(ihpro,i1,i2,i3,iallzero)
      implicit none
      integer ihpro,i1,i2,i3,iallzero,isum,chin,chout,chall
      real*8 tiny
      parameter (tiny=1.d-8)
      logical ferror
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
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
        write(*,*)'parcheckfhg called improperly'
        stop
      endif
      if(isum.ne.0)then
c charge must be conserved
        if(abs(chall).gt.tiny)ferror=.true.
c 401 is qqbar
        if( ihpro.eq.401 .and.
     #      (i1.le.0 .or. i2.ge.0 .or.
     #       i3.ne.21 .or. (i1+i2).ne.0) )ferror=.true.
c 402 is qg
        if( ihpro.eq.402 .and.
     #      (i1.le.0 .or. i2.ne.21 .or. 
     #       i3.le.0 .or. i1.ne.i3) )ferror=.true.
c 403 is qqbar
        if( ihpro.eq.403 .and.
     #      (i1.ge.0 .or. i2.le.0 .or. 
     #       i3.ne.21 .or. (i1+i2).ne.0) )ferror=.true.
c 404 is qbarg
        if( ihpro.eq.404 .and.
     #      (i1.ge.0 .or. i2.ne.21 .or. 
     #       i3.ge.0 .or. i1.ne.i3) )ferror=.true.
c 405 is gq
        if( ihpro.eq.405 .and.
     #      (i1.ne.21 .or. i2.le.0 .or. 
     #       i3.le.0 .or. i2.ne.i3) )ferror=.true.
c 406 is gqbar
        if( ihpro.eq.406 .and.
     #      (i1.ne.21 .or. i2.ge.0 .or. 
     #       i3.ge.0 .or. i2.ne.i3) )ferror=.true.
c 407 is gg
        if( ihpro.eq.407 .and.
     #      (i1.ne.21 .or. i2.ne.21 .or. 
     #       i3.ne.21) )ferror=.true.
      endif
      if(ferror)then
        write(*,*)'Error in parcheckfhg'
        write(*,*)'ihpro,i1,i2,i3:',ihpro,i1,i2,i3
        stop
      endif
      return
      end


      subroutine parcheckinhg()
      implicit none
      integer i
      real*8 chup,chdn
      parameter (chup=2.d0/3.d0)
      parameter (chdn=-1.d0/3.d0)
      real*8 chrg(-5:21),chprdct
      common/ccharges/chrg,chprdct
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
      chprdct=0.d0
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
      subroutine whichohg(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
c Determines the type of event at random
      implicit none
      integer iseed,itot,mx_of_evta,mx_of_evtb,iunit,i0
      real*8 xpa,xpb,tiny,one,xsum,rnd,fk88random,prob
      parameter (tiny=1.d-4)
      logical flag
c
      if(itot.le.0)then
        write(6,*)'Fatal error #1 in whichohg'
        stop
      endif
      xpa=dfloat(mx_of_evta)/dfloat(itot)
      xpb=dfloat(mx_of_evtb)/dfloat(itot)
      one=xpa+xpb
      if(abs(one-1.d0).gt.tiny)then
        write(6,*)'Error #1 in whichohg: probability not normalized'
        stop
      endif
      i0=0
      flag=.true.
      xsum=0.d0
      rnd=fk88random(iseed)
      do while(flag)
        if(i0.gt.2)then
          write(6,*)'Fatal error #2 in whichohg'
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


      subroutine crossckehg(itot,mx_of_evta,mx_of_evtb)
c Checks whether whichohg did it right
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


      subroutine retrieve_evenhg(iunit,ii,dummy)
c Reads from disk the complete information on the events; see store_evengh
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
      real*8 emsca,emsca2
      common/cemsca/emsca,emsca2
      real*8 xmom_lb(4,4)
      common/cxmomlb/xmom_lb
c
      read(iunit,901,end=997,err=998)i1hpro,iccode,np
      read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4
      read(iunit,903,end=997,err=998)xevsign,emsca
      read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,4)
      goto 999
 901  format(1x,i3,2(1x,i2))
 902  format(7(1x,i3))
 903  format(2(1x,d14.8))
 904  format(28(1x,d14.8))
 997  write(*,*)'unexpected end of file, iunit=',iunit
      stop
 998  write(*,*)'format error'
      write(77,*)'event #:',ii
      write(77,901)i1hpro,iccode,np
      write(77,902)ip1,ip2,ip3,ip4
      write(77,903)xevsign,emsca
      write(77,904)((xmom_lb(i,j),j=1,4),i=1,4)
      stop
 999  continue
      return
      end


      subroutine store_evengh(iunit,xpmone)
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
c files (via retrieve_evenhg), to be stored in the final event file.
c When xpmone=1, one has xevsign=+1/-1, and the weight of the event is 
c xevsign*wgt[a,b]ev. When xpmone=-1, then xevsign is the weight of the event. 
c
c i1hpro has the following conventions:
c   i1hpro         process
c    401        q qbar -> g H
c    402        q g    -> q H
c    403        qbar q -> g H
c    404        qbar g -> qbar H
c    405        g q    -> q H
c    406        g qbar -> qbar H
c    407        g g    -> g H
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
      real*8 emsca,emsca2
      common/cemsca/emsca,emsca2
c xmom_lb(i,j) is the j component of the four vector of the particle # i,
c given in the laboratory frame. j=4 is the energy for MC@NLO versions
c up to 2.31, the mass for version 3.1 onwards. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is the Higgs.
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
        emsca=sqrt(emsca2)
      else
        write(*,*)'Fatal error in store_evengh: xpmone=',xpmone
        stop
      endif
      write(iunit,901)i1hpro,iccode,np
      write(iunit,902)ip1,ip2,ip3,ip4
      write(iunit,903)xevwgt,emsca
      write(iunit,904)((xmom_lb(i,j),j=1,4),i=1,4)
 901  format(1x,i3,2(1x,i2))
 902  format(7(1x,i3))
 903  format(2(1x,d14.8))
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
      subroutine xmcsuhg(jproc,s,xii,y,ycnt,xmcxsec)
c xmcxsec is only output of this routine, and is defined using eq.(A.83)
c (for ycnt=1) or eq.(A.84) (for ycnt=-1) of FW, with phi_3 --> phi_2,
c and obvious modifications for the flavour. The definition is
c     (1-x)**2*(1-y**2)*dsigma_MC/dphi_2 = gs**6*gf*xmcxsec
c This routine is derived from xmcsuhg of the vector boson pair package
c For consistency, the same notations are kept: their meaning is as follows
c xmin=(7+sqrt(17))/16==x_DZ, eq.(A.70) of FW
c xsoft=tilde{x_DZ}, eq.(A.87) of FW
c ydead_mod(x)=Y_DZ(x), eq.(A.71) of FW
c xlim1 is the part which multiplies G
c xlim2 is the part which multiplies 1-G
      implicit none
      real*8 s,xii,y,ycnt,xmcxsec(4)
      integer jproc
      include 'hgscblks.h'
      real*8 x,tiny,vca,vcf,xmin,xlim,xlim1,xlim2,al_gfun,be_gfun,
     # ccc_gfun,yd,ydead_mod,xfact,z,zherw_spl,xi,xiherw_spl,
     # ap,ap_kern,xjac_xizspl,zherw_smn,xiherw_smn,xjac_xizsmn,
     # tt,gfact,gfun,sbar,xsoft,ff1,ptmin,ptmax,emsca,emsca2,emscwgt,
     # emscafun,emscainv,pthw,ptresc,alpha,rrnd,fk88random,born(4)
      integer icode,ione,itwo,j,ifk88seed
      parameter (tiny=1.d-6)
      parameter (vca=3.d0)
      parameter (vcf=4/3.d0)
      parameter (xmin=0.69519410160110384d0)
      parameter (ff1=0.115d0)
      parameter (alpha=1.d0)
      parameter (ione=1)
      parameter (itwo=2)
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
      common/cemsca/emsca,emsca2
      common/cifk88seed/ifk88seed
c
      if(abs(ycnt).ne.1.d0) then
        write(6,*)'xmcsuhg called improperly: ycnt=',ycnt
        stop
      endif
      x=1-xii
      xlim=0.d0
      xlim1=0.d0
      xlim2=0.d0
      do j=1,4
        xmcxsec(j)=0.d0
      enddo
      pthw=xii/2.d0*sqrt(s*(1-y**2))
      ptmin=max(ff1*sqrt(xmh2),10.d0)
      ptmax=max(sqrt(xmh2),ptmin+20.d0)
      ptresc=(pthw-ptmin)/(ptmax-ptmin)
      if(ptresc.le.0.d0)then
        emscwgt=1.d0
      elseif(ptresc.le.1.d0)then
        emscwgt=1-emscafun(ptresc,alpha)
      else
        emsca2=ptmax**2
        return
      endif
      rrnd=fk88random(ifk88seed)
      rrnd=emscainv(rrnd,alpha)
      emsca2=( ptmin+rrnd*(ptmax-ptmin) )**2
      sbar=x*s
      call f1born(sbar,ione,itwo,'mc',born)
      if(jproc.eq.1)then
        icode=1
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
              xlim1=vca*xlim1
            elseif(1-y.lt.tiny)then
              xlim1=4*(1-x+x**2)**2/x+
     #              (20-5/x-35*x+34*x**2-19*x**3+4*x**4)*(1-y)
              xlim1=vca*xlim1
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
              xlim1=vca*xlim1
            elseif(1+y.lt.tiny)then
              xlim1=4*(1-x+x**2)**2/x+
     #              (20-5/x-35*x+34*x**2-19*x**3+4*x**4)*(1+y)
              xlim1=vca*xlim1
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
        xmcxsec(1)=xlim*born(1)*emscwgt
      elseif(jproc.eq.2)then
        continue
      elseif(jproc.eq.3)then
        icode=3
        yd=ydead_mod(x)
        if(ycnt.eq.1.d0)then
          if(y.gt.yd)then
            if(1-x.lt.tiny)then
              xlim=(1-x)*(1+y)
              xlim=vcf*xlim
            elseif(1-y.lt.tiny)then
              xlim=2*(1-x)*(2-2*x+x**2)/x-
     #             (10-28*x+31*x**2-16*x**3+3*x**4)/(2*x)*(1-y)
              xlim=vcf*xlim
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_spl(x,y)
              xi=xiherw_spl(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim=xjac_xizspl(x,y)*xfact*ap/xi
            endif
          endif
          xlim=4*xlim/s
          do j=1,2
            xmcxsec(j)=xlim*born(j)*emscwgt
          enddo
        elseif(ycnt.eq.-1.d0)then
          if(y.lt.-yd)then
            if(1-x.lt.tiny)then
              xlim=(1-x)*(1-y)
              xlim=vcf*xlim
            elseif(1+y.lt.tiny)then
              xlim=2*(1-x)*(2-2*x+x**2)/x-
     #             (10-28*x+31*x**2-16*x**3+3*x**4)/(2*x)*(1+y)
              xlim=vcf*xlim
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_smn(x,y)
              xi=xiherw_smn(x,y)
              ap=ap_kern(z,icode)/(1-z)
              xlim=xjac_xizsmn(x,y)*xfact*ap/xi
            endif
          endif
          xlim=4*xlim/s
          do j=3,4
            xmcxsec(j)=xlim*born(j)*emscwgt
          enddo
        endif
      else
        write(*,*)'xmcsuhg: unknown process',jproc
        stop
      endif
      return
      end


      subroutine genvetosc()
      implicit none
      include 'hgscblks.h'
      real*8 emsca,emsca2
      common/cemsca/emsca,emsca2
c
      emsca2=sh
      return
      end


CCC =====================================
CCC   removed by fst

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
c
C      tmp=0.d0
C      if(x.lt.xmin)tmp=1-x*(3.d0-sqrt(1+8*x))/(1-x)
C      ydead=tmp
C      return
C      end


C      function ydead_mod(x)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
c
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
c
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


      function emscafun(x,alpha)
      implicit none
      real*8 emscafun,x,alpha
c
      if(x.lt.0.d0.or.x.gt.1.d0)then
        write(6,*)'Fatal error in emscafun'
        stop
      endif
      emscafun=x**(2*alpha)/(x**(2*alpha)+(1-x)**(2*alpha))
      return
      end


      function emscainv(r,alpha)
c Inverse of emscafun; implemented only for alpha=1 for the moment
      implicit none
      real*8 emscainv,r,alpha
c
      if(r.lt.0.d0.or.r.gt.1.d0.or.alpha.ne.1.d0)then
        write(6,*)'Fatal error in emscafun'
        stop
      endif
      emscainv=(r-sqrt(r-r**2))/(2*r-1)
      return
      end
c
c
c End of MC subtraction terms
c
c
c
c
c Begin of utility routines for xi, z, and 2-->2 invhgiants. The functions
c for xi, z, and the jacobian have been checked numerically to coincide with
c those of the hvq package (except when closer than tiny to IR limits, since
c here more terms are kept -- which are numerically irrelevant). The present
c for is simpler and faster to computed, thanks to the unique choice of E0
c
c
C      function zherw_spl(xx,yy)
C      implicit real*8(a-h,o-z)
C      parameter (tiny=1.d-4)
c
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
c
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
c
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
c
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
c
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
c End of utility routines for xi, z, and 2-->2 invariants
c
c
c
c
c Begin of utility routines for Bjorken x's
c
c
      function x1sohg(xx1,xx2,xx,yy)
      implicit none
      real*8 x1sohg,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
      parameter (tiny=1.d-5)
      integer iprespl
      common/ciprespl/iprespl
c
      x1=xx1
      x2=xx2
      x=xx
      y=yy
      if(iprespl.eq.0)then
        csi=sqrt( (2-(1-x)*(1+y))/(2-(1-x)*(1-y)) )
        rx=sqrt(x)
        tmp=x1*csi*rx
      elseif(iprespl.eq.1)then
        if(1-x.lt.tiny)then
          tmp=x1*(1-(1-x)*(1+y)/2.d0)
        elseif(1-y.lt.tiny)then
          tmp=x*x1*(1+(1-x)*(1-y)*(x1+x2)/(2.d0*(x*x1+x2)))
        elseif(1+y.lt.tiny)then
          tmp=x1*(1-(1-x)*(1+y)*(x1+x2)/(2.d0*(x1+x*x2)))
        else
          xa=x*x1*x2
          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
          tmp=0.5d0*(sqrt(xb**2+4*xa)-xb)
        endif
      else
        write(*,*)'Error in x1sohg',iprespl
        stop
      endif
      x1sohg=tmp
      return
      end


      function x2sohg(xx1,xx2,xx,yy)
      implicit none
      real*8 x2sohg,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
      parameter (tiny=1.d-5)
      integer iprespl
      common/ciprespl/iprespl
c
      x1=xx1
      x2=xx2
      x=xx
      y=yy
      if(iprespl.eq.0)then
        csi=sqrt( (2-(1-x)*(1+y))/(2-(1-x)*(1-y)) )
        rx=sqrt(x)
        tmp=x2*rx/csi
      elseif(iprespl.eq.1)then
        if(1-x.lt.tiny)then
          tmp=x2*(1-(1-x)*(1-y)/2.d0)
        elseif(1-y.lt.tiny)then
          tmp=x2*(1-(1-x)*(1-y)*(x1+x2)/(2.d0*(x*x1+x2)))
        elseif(1+y.lt.tiny)then
          tmp=x*x2*(1+(1-x)*(1+y)*(x1+x2)/(2.d0*(x1+x*x2)))
        else
          xa=x*x1*x2
          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
          tmp=0.5d0*(sqrt(xb**2+4*xa)+xb)
        endif
      else
        write(*,*)'Error in x2sohg',iprespl
        stop
      endif
      x2sohg=tmp
      return
      end


      function x1x2jhg(xx1,xx2,xx,yy)
      implicit none
      real*8 x1x2jhg,xx1,xx2,xx,yy,tiny,x1,x2,x,y,tmp,xa,xb
      parameter (tiny=1.d-5)
      integer iprespl
      common/ciprespl/iprespl
c
      x1=xx1
      x2=xx2
      x=xx
      y=yy
      if(iprespl.eq.0)then
        tmp=x
      elseif(iprespl.eq.1)then
        if(1-x.lt.tiny)then
          tmp=x+(1-x)**2*(1-y**2)*x1*x2/(2.d0*(x1+x2)**2)
        elseif(1-y.lt.tiny)then
          tmp=x+(1-x)**2*x*(1-y)*x1*x2/(x*x1+x2)**2
        elseif(1+y.lt.tiny)then
          tmp=x+(1-x)**2*x*(1+y)*x1*x2/(x1+x*x2)**2
        else
          xa=x*x1*x2
          xb=0.5d0*((1-x)*y*(x1+x2)+(1+x)*(x2-x1))
          tmp=x*((1-y+x*(1+y))*x1+(1+y+x*(1-y))*x2)/
     #        (2.d0*sqrt(xb**2+4*xa))
        endif
      else
        write(*,*)'Error in x1x2jhg',iprespl
        stop
      endif
      x1x2jhg=abs(tmp)
      return
      end
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
      function zgmu2_nhg()
c Sets the scales for NLO subtraction terms
      implicit none
      real * 8 zgmu2_nhg
      real * 8 pi
      parameter (pi=3.14159265358979312D0)
      include 'hgscblks.h'
      real * 8 pth,ptp
      common/perpen/pth,ptp
      integer inloscale
      common/cinloscale/inloscale
      real * 8 xmu2,as,alfas
c
      if(inloscale.eq.1)then
        xmu2 = pth**2 + xmh2
      elseif(inloscale.eq.2)then
        xmu2 = xmh2
      elseif(inloscale.eq.3)then
        xmu2 = max(xmh2,pth**2)
      elseif(inloscale.eq.4)then
        xmu2 = ( max(sqrt(xmh2)-pth,pth) )**2
      else
        write(*,*)'Unknown option in zgmu2_nhg',inloscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmuf2h1 = xmu2*xf2h1
      xmuf2h2 = xmu2*xf2h2
      xmur2  = xmu2*xren2
      as    = alfas(xmur2,xlam,nl)
      zgmu2_nhg = 4.d0*pi*as
      zg = sqrt(zgmu2_nhg)
      end


      function zgmu2_hg()
c Sets the scales for MC subtraction terms
      implicit none
      real * 8 zgmu2_hg
      real * 8 pi,thrs
      parameter (pi=3.14159265358979312D0)
      include 'hgscblks.h'
      real * 8 pth,ptp
      common/perpen/pth,ptp
      integer imcscale
      common/cimcscale/imcscale
      real * 8 xmu2,as,alfas
c
      if(imcscale.eq.1)then
        xmu2 = pth**2 + xmh2
      elseif(imcscale.eq.2)then
        xmu2 = xmh2
      elseif(imcscale.eq.3)then
        xmu2 = max(xmh2,pth**2)
      elseif(imcscale.eq.4)then
        xmu2 = ( max(sqrt(xmh2)-pth,pth) )**2
      elseif(imcscale.eq.5)then
        xmu2 = ( sqrt(xmh2)-pth/2.d0 )**2
      elseif(imcscale.eq.6)then
        thrs = 0.8*sqrt(xmh2)
        xmu2 = pth**2 + xmh2
        if(pth.gt.thrs)xmu2 = xmu2*( 1+(pth-thrs)/thrs )
      elseif(imcscale.eq.7)then
        thrs = 0.8*sqrt(xmh2)
        xmu2 = pth**2 + xmh2
        if(pth.gt.thrs)xmu2 = xmu2*( 1+sqrt((pth-thrs)/thrs) )
      else
        write(*,*)'Unknown option in zgmu2_hg',imcscale
        stop
      endif
c      xmu2 = xmh2
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmumcf2h1 = xmu2*xf2h1mc
      xmumcf2h2 = xmu2*xf2h2mc
      xmumcr2  = xmu2*xren2mc
      as    = alfas(xmumcr2,xlam,nl)
      zgmu2_hg = 4.d0*pi*as
      zg = sqrt(zgmu2_hg)
      end
