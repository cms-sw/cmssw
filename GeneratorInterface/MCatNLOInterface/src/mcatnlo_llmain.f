      SUBROUTINE LLMAIN
      implicit none
      include 'llpcblks.h'
      real * 8 value(20),xmass(-5:21)
      real * 8 
     #  xm0,xpdflam4,xpdflam5,tmp,xren,xfh,xrenmc,xfhmc,ecm,xicut,
     #  delta,deltas,deltac,dtot,avtot,ac1,ac2,xtotal,ytotal,av3a,
     #  av3nega,d3a,d3nega,ctime,pi,tmas,etacut,av3b,av3negb,d3b,
     #  d3negb,wgtaev,wgtbev,evfrac,evprcfrac,xares,yares,xbres,ybres,
     #  al_gfun,be_gfun,ccc_gfun,dummy,xmone,xicutss,gammax,
     #  xmlow,xmupp,bwxmdpl,bwxmdmn,tsamp,wdtwon,wdtwmsb,wdtzon,
     #  wdtzmsb
      integer 
     #  ih1,ih2,ndns1,ndns2,iseld,nlf,ncl3,mx_of_evta,mx_of_evtb,
     #  loproc,maproc,iproc,iinput,iverbose,ichkmom,
     #  ibswrite,itmpih,itmpndns,idpdfset,ipdfih,ipdfgroup,ipdfndns,
     #  ifk88istrl,ifk88ih,ifk88ndns,maxevt,it1,it2,ifuntype,
     #  ndim,nwild,itd1,itd2,ibscall,iwgtnorm,iseed0,ifxdaem,
     #  iseed,maxtrials,mode,lo,isubttype,iprespl,iprdct,iprdct0,
     #  ndiff,ntotal,i,ionshell,iwrong,iwrong1,nevts,ntrls,ilepmass,
     #  iunita,iunitb,iunit,ioutput,itot,ii,inloscale,imcscale,
     #  iprdct0hw,ilep,ifk88seed,izero,ione
      character * 2 scheme
      character * 4 part1,part2
      character * 20 parm(20),gname
      character * 80 fname,fnamea,fnameb,fname1,fnamev
      character * 80 pref,prefn,prefev,prefnev
      character * 70 strin,strout,lhapdf
      logical evgen
      external sig5afw,sig5bfw
      parameter (pi=3.14159265358979312D0)
      parameter (xmone=-1.d0)
      parameter (izero=0)
      parameter (ione=1)
c
c common /strfun0/ is only in strfun:
c ndns = pdf type
c ih1,ih2 = beam type (0=(p+n)/2, 1=p, -1=pbar, 2=n, -2=nbar)
      common/strfun0/ih1,ih2,ndns1,ndns2
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
      common/ctsamp/tsamp
c etacut is the maximum allowed for [2*kt(gluon)/sqrt(shat)]^2
      common/cetacut/etacut
c Partial W and Z decay widths in the on-shell and MSbar schemes
      common/partdec/wdtwon,wdtwmsb,wdtzon,wdtzmsb
c----------------------------------------------------------
c inloscale controls the reference scale in the NLO computation
      common/cinloscale/inloscale
c imcscale controls the reference scale in the MC subtraction terms
      common/cimcscale/imcscale
c----------------------------------------------------------
c Process: iprdct0=135# for Z/gamma
c          iprdct0=136# for Z
c          iprdct0=137# for gamma
c          iprdct0=146# for W+ 
c          iprdct0=147# for W-
c with #=1,2,3,4,5,6 for e,mu,tau,nu_e,nu_mu,nu_tau; since # is relevant
c only to MC, in NLO code the variable iprdct=135,136,137,146,147 will be
c used instead (except in setpar). In Herwig, in the case of Z/gamma 
c production, the convention is #=1,2,3,4,5,6 for e,nu_e,mu,nu_mu,tau,nu_tau;
c we shall enter process code according to this convention, store it in
c the variable iprdct0hw, and use iprdct0 in this code
      common/ciprdct0/iprdct0
      common/ciprdct/iprdct
c Subprocesses: 'qq', 'qg', corresponding to jproc=jproc0=2,3
c In the integration routines, loproc<=jproc<=maproc
      common/cwchproc/loproc,maproc
c Number of failures in flavour determination
      common/ciwrong/iwrong,iwrong1
c----------------------------------------------------------
c ifuntype=1 for sig5a, ifuntype=2 for sig5b
      common/cifuntype/ifuntype
c Flag to put partons on shell, according to Herwig list of masses
      common/cionshell/ionshell
c Flag to put leptons on shell, according to PDF masses
      common/cilepmass/ilepmass
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
      iinput=2
      mmdecide = 1
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
      open(unit=11,file='llplog',status=newver)
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
      if(iinput.lt.2) then
         write(*,*)' '
         write(*,*)'Enter -(1)135# for Z/gamma'
         write(*,*)'      -(1)136# for Z'
         write(*,*)'      -(1)137# for gamma'
         write(*,*)'      -(1)146# for W+'
         write(*,*)'      -(1)147# for W-'
         read(*,*) iprdct0hw
      else
         iprdct0hw=mmiproc
      endif
      iprdct0hw=mod(-iprdct0hw,10000)
      iprdct=iprdct0hw/10
      if( .not.(iprdct0hw.ge.1351.and.iprdct0hw.le.1356) .and.
     #    .not.(iprdct0hw.ge.1361.and.iprdct0hw.le.1366) .and.
     #    .not.(iprdct0hw.ge.1371.and.iprdct0hw.le.1373) .and.
     #    .not.(iprdct0hw.ge.1461.and.iprdct0hw.le.1463) .and.
     #    .not.(iprdct0hw.ge.1471.and.iprdct0hw.le.1473) ) then
         write(*,*)'Process not implemented'
         stop
      endif
      write(11,'(1x,i6,27x,a)') -iprdct0hw,
     #  '! -135#/136#/137#/146#/147#=Zg/Z/g/W+/W-'
      iprdct0=iprdct0hw
      if(iprdct.lt.140)then
        ilep=iprdct0hw-10*iprdct
        if(ilep.eq.2)then
          iprdct0=10*iprdct+4
        elseif(ilep.eq.3)then
          iprdct0=10*iprdct+2
        elseif(ilep.eq.4)then
          iprdct0=10*iprdct+5
        elseif(ilep.eq.5)then
          iprdct0=10*iprdct+3
        endif
      endif


c Set constants
      call setpar()
c
      if(iprdct.ne.137)then
         if(iinput.lt.2) then         
            write(*,*)' '
            write(*,*)'Enter vector boson mass and width (GeV)'
           read(*,*)xm0,gah
        else
           xm0=mmxm0
           gah=mmgah
        endif
        write(11,'(2(1x,d10.4),12x,a)') xm0,gah,'! M_V, Ga_V'
        xm02 = xm0**2
        if(iinput.lt.2) then
           write(*,*)' '
        write(*,*)'Enter GammaX, M_V(min), M_V(max)'
        write(*,*)'  If GammaX>0, the pair mass is chosen in the range'
        write(*,*)'      M0-GammaX*width < M_V < M0+GammaX*width'
        write(*,*)'  and M_V(min), M_V(max) are ignored'
        write(*,*)'  If GammaX<0, the pair mass is chosen in the range'
        write(*,*)'            M_V(min) < M_V < M_V(max)'
           read(*,*)gammax,xmlow,xmupp
        else
           gammax=mmv1gammax
           xmlow=mmv1massinf
           xmupp=mmv1masssup
        endif
        write(11,'(3(1x,d10.4),1x,a)') gammax,xmlow,xmupp,
     #    '! GammaX, M_V(min), M_V(max)'
        if(gammax.eq.0)then
          write(*,*)'Enter a non-null value for GammaX'
          stop
        endif
      else
        if(iinput.lt.2) then
           write(*,*)' '
           write(*,*)'Enter M_ll(min), M_ll(max)'
           read(*,*)xmlow,xmupp
        else
           xmlow=mmv1massinf
           xmupp=mmv1masssup
        endif
        write(11,'(2(1x,d10.4),12x,a)') xmlow,xmupp,
     #    '! M_ll(min), M_ll(max)'
        xm0=(xmupp+xmlow)/2.d0
        xm02=xm0**2
        gah=xmupp-xmlow
      endif
c Sampling in t, the inverse of the BW function (see xbwmass2(t))
      tsamp=1.d0
      if(xmlow.lt.60.d0)tsamp=1.5d0-0.5*(max(xmlow,30.d0)-30.d0)/30.d0
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
        write(*,*)'Enter 1 to set mu_0^2=M_V^2+pt_V^2'
        write(*,*)'      2 to set mu_0=M_V'
        write(*,*)'      3 to set mu_0=max(M_V,pt_V)'
        write(*,*)'      4 to set mu_0=max(M_V-pt_V,pt_V)'
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
        write(*,*)'      4 to set mu_0=max(M_V-pt_V,pt_V)'
        write(*,*)'      5 to set mu_0=M_V-pt_V/2'
        write(*,*)' The default is 5'
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
         read(*,*)ifxdaem
      else
         ifxdaem=mmaemrun
      endif
      write(11,'(1(1x,i8),25x,a)') ifxdaem,
     #  '! 0=running, 1=fixed alpha_EM'
      if(ifxdaem.ne.0.and.ifxdaem.ne.1)then
        write(*,*)'No such option for alpha_em'
        stop
      endif
      if(ifxdaem.eq.0)ze2=0.d0
c---------------------------------------------------------------
c Initialize parameters, such as labelling for parton processes
      call parsetpar()
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
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to leave the leptons massless'
        write(*,*)'      2 to put leptons on mass shell'
        read(*,*) ilepmass
        write(11,'(1x,i1,32x,a)') 
     #      ilepmass,'! 0=massless, 2=massive leptons'
      else
        ilepmass=2
      endif
      if(ilepmass.ne.0.and.ilepmass.ne.2) then
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
c
      ndim=6
      nwild=5
      xmll2=-1.d10
      if(iprdct.eq.137)then
        xmlow2=xmlow**2
        xmupp2=xmupp**2
      else
        if(gammax.ge.0)then
          xmlow2=(max(0.d0,xm0-gammax*gah))**2
          xmupp2=(xm0+gammax*gah)**2
        else
          xmlow2=xmlow**2
          xmupp2=xmupp**2
        endif
      endif
      xmlow2=max(100.d0,xmlow2)
      xmupp2=min(sh*0.95,xmupp2)
      if(xmlow2.gt.xmupp2)then
        write(*,*)'Error in pair mass range'
        stop
      endif
      if(iprdct.ne.137)then
        bwxmdpl=xmupp2-xm02
        bwxmdmn=xm02-xmlow2
        bwfmpl=atan(bwxmdpl/(xm0*gah))
        bwfmmn=atan(bwxmdmn/(xm0*gah))
        bwdelf=(bwfmpl+bwfmmn)/pi
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
        call run_bases(sig5afw,fnamea,ndim,nwild,ncl3,it1,it2,
     #    ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #    ibswrite,ibscall)
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
        call run_bases(sig5bfw,fnameb,ndim,nwild,ncl3,it1,it2,
     #    ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #    ibswrite,ibscall)
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
      if(isubttype.eq.1.and.delta.ne.1.d0)then
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
     #             'The following number of events will be generated'
              write(*,*)'# events[a]:',mx_of_evta
              write(*,*)'# events[b]:',mx_of_evtb
           endif
           write(*,*)'  '
           write(*,*)
     #          'Estimated fractions of events with negative weights'
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
        call run_spring(sig5afw,fnamea,mx_of_evta,maxtrials,
     #                  nevts,ntrls,ndim,nwild,iseed)
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
        call run_spring(sig5bfw,fnameb,mx_of_evtb,maxtrials,
     #                  nevts,ntrls,ndim,nwild,iseed)
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
        write(ioutput,802)iprdct0hw,
     #    '--> 135#/136#/137#/146#/147#=Zg/Z/g/W+/W-'
        if(iprdct.ne.137)then
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
          call whichone(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
          call retrieve_events(iunit,ii,dummy)
          call store_events(ioutput,xmone)
        enddo
        call crosscheck(itot,mx_of_evta,mx_of_evtb)
        close(iunita)
        close(iunitb)
        close(ioutput)
 111    continue
      endif
      if(basesoutput.gt.0) then
         if(iprdct.eq.135.or.iprdct.eq.136)then
            write(*,*) '   '
            write(*,*)'Decay width in MSbar and on-shell schemes'
            write(*,*)'MSbar is used in the present computation'
            write(*,*)'MSbar:    Gamma =',wdtzmsb
            write(*,*)'On-shell: Gamma =',wdtzon
            write(*,*)'MSbar:    Gamma/Gamma_tot =',wdtzmsb/gah
            write(*,*)'On-shell: Gamma/Gamma_tot =',wdtzon/gah
         elseif(iprdct.eq.146.or.iprdct.eq.147) then
            write(*,*) '   '
            write(*,*)'Decay width in MSbar and on-shell schemes'
            write(*,*)'MSbar is used in the present computation'
            write(*,*)'MSbar:    Gamma =',wdtwmsb
            write(*,*)'On-shell: Gamma =',wdtwon
            write(*,*)'MSbar:    Gamma/Gamma_tot =',wdtwmsb/gah
            write(*,*)'On-shell: Gamma/Gamma_tot =',wdtwon/gah
         endif
         write(*,*) '   '
         write(*,*) 'Total for fully inclusive'
         write(*,200)ih1,ih2,ndns1,ndns2,nl,xlam
         if(iprdct.ne.137)then
            write(*,201) 'tot'
            write(*,300)ecm,xm0,gah,xfh,xren,avtot,dtot
         else
            write(*,206) 'tot'
            write(*,300)ecm,xmlow,xmupp,xfh,xren,avtot,dtot
         endif
      endif
 200  format(' had1=',i2,'  had2=',i2,'  strf1=',i6,'  strf2=',i6,
     #     '  nl=',i2,'  lambda5=',d10.4)
 201  format(' ecm or ebeam  mass      width     xf   xr   ',a,
     # '        err    ')
 206  format(' ecm or ebeam  Mmin      Mmax      xf   xr   ',a,
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


      subroutine getset(str,ndns,ih)
      implicit real * 8 (a-h,o-z)
      character * (*) str
 2    write(*,*) str
      write(*,*)
     # '   (< 0 for a display of the features of the various sets'
      read(*,*) ndns
      if(ndns.lt.0) then
        call prntsf
        go to 2
      endif
      end


      subroutine toend(iunit)
      ios = 0    
      dowhile(ios.eq.0)
         read(unit=iunit,fmt='(1x)',iostat=ios)
      enddo                        
      end


      subroutine getihpart(part,ih)
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
        write(*,*)'Error in getihpart'
        write(*,*)'No such particle in MLM:',part
        stop
      endif
      return
      end


      subroutine strfun(x1,x2,sf)
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
      real * 8 pi,x1,x2,sf(4,2:3,6)
      integer ih1,ih2,ndns1,ndns2,ii,jproc,itype,iprdct
      parameter(pi=3.14159265358979312D0)
      include 'llpcblks.h'
      common/strfun0/ih1,ih2,ndns1,ndns2
      common/ciprdct/iprdct
      real*8 zel(1:6),zel2(1:6)
      real * 8 ckm2(1:6,1:6)
      common/charges/zel,zel2
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
        write(*,*)'Fatal error in strfun: unknown ipdfscale',ipdfscale
        stop
      endif
c
      call mlmpdf(ndns1,ih1,smuf2h1,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smuf2h2,sngl(x2),fh2x2,5)
c
c jproc=2
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
        do ii=1,nl
          sf(1,2,ii)=dble(fh1x1( ii) * fh2x2(-ii))
          sf(3,2,ii)=dble(fh1x1(-ii) * fh2x2( ii))
        enddo
      elseif(iprdct.eq.146)then
        sf(1,2,1)=ckm2(1,2)*dble(fh1x1( 1) * fh2x2(-2)) 
        sf(1,2,2)=ckm2(1,3)*dble(fh1x1( 1) * fh2x2(-3)) 
        sf(1,2,3)=ckm2(1,5)*dble(fh1x1( 1) * fh2x2(-5)) 
        sf(1,2,4)=ckm2(4,2)*dble(fh1x1( 4) * fh2x2(-2)) 
        sf(1,2,5)=ckm2(4,3)*dble(fh1x1( 4) * fh2x2(-3)) 
        sf(1,2,6)=ckm2(4,5)*dble(fh1x1( 4) * fh2x2(-5)) 
        sf(3,2,1)=ckm2(1,2)*dble(fh1x1(-2) * fh2x2( 1)) 
        sf(3,2,2)=ckm2(1,3)*dble(fh1x1(-3) * fh2x2( 1)) 
        sf(3,2,3)=ckm2(1,5)*dble(fh1x1(-5) * fh2x2( 1)) 
        sf(3,2,4)=ckm2(4,2)*dble(fh1x1(-2) * fh2x2( 4)) 
        sf(3,2,5)=ckm2(4,3)*dble(fh1x1(-3) * fh2x2( 4)) 
        sf(3,2,6)=ckm2(4,5)*dble(fh1x1(-5) * fh2x2( 4)) 
      elseif(iprdct.eq.147)then
        sf(1,2,1)=ckm2(1,2)*dble(fh1x1( 2) * fh2x2(-1)) 
        sf(1,2,2)=ckm2(4,2)*dble(fh1x1( 2) * fh2x2(-4)) 
        sf(1,2,3)=ckm2(1,3)*dble(fh1x1( 3) * fh2x2(-1)) 
        sf(1,2,4)=ckm2(4,3)*dble(fh1x1( 3) * fh2x2(-4)) 
        sf(1,2,5)=ckm2(1,5)*dble(fh1x1( 5) * fh2x2(-1)) 
        sf(1,2,6)=ckm2(4,5)*dble(fh1x1( 5) * fh2x2(-4)) 
        sf(3,2,1)=ckm2(1,2)*dble(fh1x1(-1) * fh2x2( 2)) 
        sf(3,2,2)=ckm2(4,2)*dble(fh1x1(-4) * fh2x2( 2)) 
        sf(3,2,3)=ckm2(1,3)*dble(fh1x1(-1) * fh2x2( 3)) 
        sf(3,2,4)=ckm2(4,3)*dble(fh1x1(-4) * fh2x2( 3)) 
        sf(3,2,5)=ckm2(1,5)*dble(fh1x1(-1) * fh2x2( 5)) 
        sf(3,2,6)=ckm2(4,5)*dble(fh1x1(-4) * fh2x2( 5)) 
      else
        write(*,*)'Error in strfun: unknown IPROC',iprdct
        stop
      endif
c jproc=3
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
        do ii=1,nl
          sf(1,3,ii)=dble(fh1x1( ii) * fh2x2(  0))
          sf(2,3,ii)=dble(fh1x1(-ii) * fh2x2(  0))
          sf(3,3,ii)=dble(fh1x1(  0) * fh2x2( ii))
          sf(4,3,ii)=dble(fh1x1(  0) * fh2x2(-ii))
        enddo
      elseif(iprdct.eq.146)then
        sf(1,3,1)=ckm2(1,2)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,2)=ckm2(1,3)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,3)=ckm2(1,5)*dble(fh1x1( 1) * fh2x2( 0)) 
        sf(1,3,4)=ckm2(4,2)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(1,3,5)=ckm2(4,3)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(1,3,6)=ckm2(4,5)*dble(fh1x1( 4) * fh2x2( 0)) 
        sf(2,3,1)=ckm2(1,2)*dble(fh1x1(-2) * fh2x2( 0)) 
        sf(2,3,2)=ckm2(4,2)*dble(fh1x1(-2) * fh2x2( 0)) 
        sf(2,3,3)=ckm2(1,3)*dble(fh1x1(-3) * fh2x2( 0)) 
        sf(2,3,4)=ckm2(4,3)*dble(fh1x1(-3) * fh2x2( 0)) 
        sf(2,3,5)=ckm2(1,5)*dble(fh1x1(-5) * fh2x2( 0)) 
        sf(2,3,6)=ckm2(4,5)*dble(fh1x1(-5) * fh2x2( 0)) 
        sf(3,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,2)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,3)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2( 1)) 
        sf(3,3,4)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(3,3,5)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(3,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2( 4)) 
        sf(4,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2(-2)) 
        sf(4,3,2)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2(-2)) 
        sf(4,3,3)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2(-3)) 
        sf(4,3,4)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2(-3)) 
        sf(4,3,5)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2(-5)) 
        sf(4,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2(-5)) 
      elseif(iprdct.eq.147)then
        sf(1,3,1)=ckm2(1,2)*dble(fh1x1( 2) * fh2x2( 0)) 
        sf(1,3,2)=ckm2(4,2)*dble(fh1x1( 2) * fh2x2( 0)) 
        sf(1,3,3)=ckm2(1,3)*dble(fh1x1( 3) * fh2x2( 0)) 
        sf(1,3,4)=ckm2(4,3)*dble(fh1x1( 3) * fh2x2( 0)) 
        sf(1,3,5)=ckm2(1,5)*dble(fh1x1( 5) * fh2x2( 0)) 
        sf(1,3,6)=ckm2(4,5)*dble(fh1x1( 5) * fh2x2( 0)) 
        sf(2,3,1)=ckm2(1,2)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,2)=ckm2(1,3)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,3)=ckm2(1,5)*dble(fh1x1(-1) * fh2x2( 0)) 
        sf(2,3,4)=ckm2(4,2)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(2,3,5)=ckm2(4,3)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(2,3,6)=ckm2(4,5)*dble(fh1x1(-4) * fh2x2( 0)) 
        sf(3,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2( 2)) 
        sf(3,3,2)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2( 2)) 
        sf(3,3,3)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2( 3)) 
        sf(3,3,4)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2( 3)) 
        sf(3,3,5)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2( 5)) 
        sf(3,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2( 5)) 
        sf(4,3,1)=ckm2(1,2)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,2)=ckm2(1,3)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,3)=ckm2(1,5)*dble(fh1x1( 0) * fh2x2(-1)) 
        sf(4,3,4)=ckm2(4,2)*dble(fh1x1( 0) * fh2x2(-4)) 
        sf(4,3,5)=ckm2(4,3)*dble(fh1x1( 0) * fh2x2(-4)) 
        sf(4,3,6)=ckm2(4,5)*dble(fh1x1( 0) * fh2x2(-4)) 
      else
        write(*,*)'Error in strfun: unknown IPROC',iprdct
        stop
      endif
c
      return
      end
c
c
c NLO cross section
c
c
      function sig5afw(xx)
c H events
      implicit none
      real * 8 sig5afw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'llpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tsamp
      common/ctsamp/tsamp
      integer iprespl
      common/ciprespl/iprespl
      integer iprdct
      common/ciprdct/iprdct
      integer ifxdaem
      common/cifxdaem/ifxdaem
      real * 8 xjac,zzz,x,ttt,th,y,xbwmass2,bwfunc,csi,rx,ymax,ymin,s,
     #  roh,xxa1,xxa2,xxc,xxymax,xxymin,ro,cth1,th2,cth2,tot5a,xalfaem
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> cth2
c
      xjac = 1
      roh = xmlow2/sh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-roh)
      xjac = xjac * (1-roh)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(4)
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
      y    = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits. Since there are cuts on the invariant mass of the pair, tau
c is obtained by defining M^2 first
c
      if(iprdct.eq.137)then
        ttt=xx(1)**2
        xjac=xjac*2*xx(1)
        xmll2=xmlow2+(xmupp2-xmlow2)*ttt
        xjac=xjac*(xmupp2-xmlow2)
      else
        ttt=xx(1)
        if(tsamp.ne.1)then
          ttt=xx(1)**tsamp
          xjac=xjac*tsamp*xx(1)**(tsamp-1)
        endif
        xmll2=xbwmass2(ttt)
        xjac=xjac*bwdelf/bwfunc(xmll2,xm02,gah)
      endif
      if(xmll2.lt.xmlow2.or.xmll2.gt.xmupp2)then
        write(*,*)'Fatal error in sig5afw',xmll2
        stop
      endif
      s=xmll2/x
      tau=s/sh
      xjac=xjac/(x*sh)
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmll2)
      xicut=xicutss
      if(xicut.gt.(1-roh))xicut=1-roh
      if(iprespl.eq.0)then
        csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
        rx = sqrt(x)
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
      ro = xmlow2/s
      call zzchvar(xx(5),cth1,xjac,ro)
c
      th2 = xx(6) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5afw = tot5a(s,x,y,cth1,cth2,xjac)
      return
      end



      function tot5a(s,x,yi,cth1,cth2,xjac)
      implicit none
      real * 8 tot5a,tot5as,tot5az,s,x,yi,cth1,cth2,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5as(s,x,yi,cth1,cth2,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5az(s,x,yi,cth1,cth2,xjac)
      else
        write(*,*)'Fatal error in tot5a:',isubttype
        stop
      endif
      tot5a=tmp
      return
      end


      function tot5as(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5as,xs,xx,xyi,xcth1,xcth2,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'llpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      common/cxvacoef/xveccoef,xaxlcoef
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 vv(4,2:3,6),vvs(4,2:3,6)
      common/cvv/vv
      common/cvvs/vvs
      real * 8 sf(4,2:3,6)
      real * 8 vecre(1:4,1:3),axlre(1:4,1:3)
      real * 8 vecmc(1:4,1:3),axlmc(1:4,1:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,xphsp_ns,xphsp_s,xphspb,x1,x2,
     #  tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,zg2_nlo,zgmu2_nlo,zg2_mc,
     #  zgmu2_mc,xnorm,ffunval5,ffunction5,ytmp,zhwfct,zherw_spl,
     #  x1t,x2t,x1soft,x2soft,x1x2j,x1x2jac,zherw_smn,xsum,dummy,
     #  xint,xtmp
      integer loproc,maproc,jproc,j,itype,i,ipdfscale,i2b,
     #  itoosoftkin,iret
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      sx = x*s
      xii = 1-x
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac*s/(2*1024*pi**4)
      xphsp_s = xjac*sx/(2*1024*pi**4)
      xphspb = xjac/(32*pi2)
c
      do jproc=2,3
        do j=1,4
          do itype=1,6
            vv(j,jproc,itype)=0.d0
            vvs(j,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(x1.lt.1.and.x2.lt.1)then
        call invar(zero,s,x,yi,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        zg2_nlo = zgmu2_nlo()
        ipdfscale=1
        call strfun(x1,x2,sf)
        xnorm = ze2**2 * zg2_nlo * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        do jproc=loproc,maproc
          call freal(s,x,yi,tk,uk,q1q,q2q,jproc,vecre,axlre)
          do j=1,4
            do itype=1,6
              do i=1,3
                vv(j,jproc,itype)=vv(j,jproc,itype)+
     #            sf(j,jproc,itype)*xnorm*
     #            ( vecre(j,i)*xveccoef(itype,i) +
     #              axlre(j,i)*xaxlcoef(itype,i) )
              enddo
            enddo
          enddo
        enddo
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone
        ffunval5 = ffunction5(x,yi)
        if(ffunval5.ne.0.d0)then
          zg2_mc = zgmu2_mc()
          ipdfscale=2
          xnorm = ze2**2 * zg2_mc * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
          ytmp=1.d0
          zhwfct=zherw_spl(x,yi)
          x1t=x1soft(x1,x2,x,yi)/zhwfct
          x2t=x2soft(x1,x2,x,yi)
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfun(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            do jproc=loproc,maproc
              call xmcsubt(jproc,s,cth1,x,yi,ytmp,vecmc,axlmc)
              do j=1,4
                do itype=1,6
                  do i=1,3
                    vv(j,jproc,itype)=vv(j,jproc,itype)-
     #                sf(j,jproc,itype)*xnorm*x1x2j*
     #                ( vecmc(j,i)*xveccoef(itype,i) +
     #                  axlmc(j,i)*xaxlcoef(itype,i) )
                  enddo
                enddo
              enddo
            enddo
          endif
          ytmp=-1.d0
          zhwfct=zherw_smn(x,yi)
          x1t=x1soft(x1,x2,x,yi)
          x2t=x2soft(x1,x2,x,yi)/zhwfct
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfun(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
            do jproc=loproc,maproc
              call xmcsubt(jproc,s,cth1,x,yi,ytmp,vecmc,axlmc)
              do j=1,4
                do itype=1,6
                  do i=1,3
                    vv(j,jproc,itype)=vv(j,jproc,itype)-
     #                sf(j,jproc,itype)*xnorm*x1x2j*
     #                ( vecmc(j,i)*xveccoef(itype,i) +
     #                  axlmc(j,i)*xaxlcoef(itype,i) )
                  enddo
                enddo
              enddo
            enddo
          endif
        endif
      endif
c
      call checkvv(xsum,dummy,iret)
      if(iret.eq.1)then
        call invar(zero,s,x,yi,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
c Cross section in pb (momenta are in GeV)
        xint=1.d6*hc2*xsum
        xevsign=1.d0
        if(xint.lt.0.d0)xevsign=-1.d0
        i2b=itoosoftkin()
        if(i2b.eq.1)then
          xtmp=1.d0
          ytmp=1.d0
          call invar(zero,sx,xtmp,ytmp,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        endif
      else
        xint=0.d0
        xevsign=1.d0
      endif
c
      bsfsgn=xevsign
      tot5as=abs(xint)
c
      return
      end


      function tot5az(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit real*8 (a-z)
      tot5az=0
      return
      end


      function sig5bfw(xx)
c S events
      implicit none
      real * 8 sig5bfw,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      include 'llpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta
      common/parsub/xicut,delta
      real * 8 xicutss
      common/xisave/xicutss
      real * 8 tsamp
      common/ctsamp/tsamp
      integer iprespl
      common/ciprespl/iprespl
      integer iprdct
      common/ciprdct/iprdct
      integer ifxdaem
      common/cifxdaem/ifxdaem
      real * 8 xjac,zzz,x,ttt,th,y,xbwmass2,bwfunc,csi,rx,ymax,ymin,s,
     #  roh,xxa1,xxa2,xxc,xxymax,xxymin,ro,cth1,th2,cth2,tot5b,xalfaem
c
c xx(1) --> tau, xx(2)-->ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> cth2
c
      xjac = 1
      roh = xmlow2/sh
c
c To improve convergence in the soft regions
      zzz = tiny+(1-tiny)*xx(3)**2
      xjac = xjac * xx(3) * 2
      x = 1 - zzz*(1-roh)
      xjac = xjac * (1-roh)
c
c To improve convergence in the collinear regions
c
      zzz = 1-2*xx(4)
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
      y    = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits. Since there are cuts on the invariant mass of the pair, tau
c is obtained by defining M^2 first
c
      if(iprdct.eq.137)then
        ttt=xx(1)**2
        xjac=xjac*2*xx(1)
        xmll2=xmlow2+(xmupp2-xmlow2)*ttt
        xjac=xjac*(xmupp2-xmlow2)
      else
        ttt=xx(1)
        if(tsamp.ne.1)then
          ttt=xx(1)**tsamp
          xjac=xjac*tsamp*xx(1)**(tsamp-1)
        endif
        xmll2=xbwmass2(ttt)
        xjac=xjac*bwdelf/bwfunc(xmll2,xm02,gah)
      endif
      if(xmll2.lt.xmlow2.or.xmll2.gt.xmupp2)then
        write(*,*)'Fatal error in sig5bfw',xmll2
        stop
      endif
      s=xmll2/x
      tau=s/sh
      xjac=xjac/(x*sh)
      if(ifxdaem.eq.0)ze2=4*pi*xalfaem(xmll2)
      xicut=xicutss
      if(xicut.gt.(1-roh))xicut=1-roh
      if(iprespl.eq.0)then
        csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
        rx = sqrt(x)
        ymax= -log(tau)/2 + log(1/(csi*rx))
        ymin=  log(tau)/2 - log(csi/rx)
      else
        xxa1 = (1+x-y*(1-x))/2.d0
        xxa2 = (1+x+y*(1-x))/2.d0
        xxc = (1-x*tau)/sqrt(tau)
        xxymax = (xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        xxymin = (-xxc+sqrt(xxc**2+4*xxa1*xxa2))/(2*xxa1)
        ymax = max(log(xxymax),-log(tau)/2.d0)
        ymin = min(log(xxymin),log(tau)/2.d0)
      endif
      ycm = ymin + xx(2)*(ymax-ymin)
      xjac= xjac * (ymax-ymin)
c
      ro = xmlow2/s
      call zzchvar(xx(5),cth1,xjac,ro)
c
      th2 = xx(6) * 2 * pi
      xjac = xjac * 2* pi
      cth2 = cos(th2)
c
      sig5bfw = tot5b(s,x,y,cth1,cth2,xjac)
      return
      end



      function tot5b(s,x,yi,cth1,cth2,xjac)
      implicit none
      real * 8 tot5b,tot5bs,tot5bz,s,x,yi,cth1,cth2,xjac,tmp
      integer isubttype
      common/cisubttype/isubttype
c
      if(isubttype.eq.0)then
        tmp=tot5bs(s,x,yi,cth1,cth2,xjac)
      elseif(isubttype.eq.1)then
        tmp=tot5bz(s,x,yi,cth1,cth2,xjac)
      else
        write(*,*)'Fatal error in tot5b:',isubttype
        stop
      endif
      tot5b=tmp
      return
      end


      function tot5bs(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit none
      real * 8 tot5bs,xs,xx,xyi,xcth1,xcth2,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2 = pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      character * 2 str
      parameter (str='p1')
      include 'llpcblks.h'
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xicut,delta,deltas,deltac
      common/parsub/xicut,delta
      common/pmerge/deltas,deltac
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      common/cxvacoef/xveccoef,xaxlcoef
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 bsewgt
      common/cbswgt/bsewgt
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 vv(4,2:3,6),vvs(4,2:3,6)
      common/cvv/vv
      common/cvvs/vvs
      real * 8 sf(4,2:3,6)
      real * 8 xintsvc(1:4,2:3,3),xbornvc(1:4,2:3,3)
      real * 8 xcsvc(1:4,2:3,3),xsvvc(1:4,2:3,3)
      real * 8 xintsax(1:4,2:3,3),xbornax(1:4,2:3,3)
      real * 8 xcsax(1:4,2:3,3),xsvax(1:4,2:3,3)
      real * 8 vecre(1:4,1:3),axlre(1:4,1:3)
      real * 8 veccl(1:4,1:3),axlcl(1:4,1:3)
      real * 8 vecbrn(1:4,1:3),axlbrn(1:4,1:3)
      real * 8 vec2sv(1:4,1:3),axl2sv(1:4,1:3)
      real * 8 vecmc(1:4,1:3),axlmc(1:4,1:3)
      real * 8 s,x,yi,cth1,cth2,sx,xii,xphsp_ns,xphsp_s,xphspb,x1,x2,
     #  ffunval5,ffunction5,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h,xnorm,
     #  zg2_mc,zgmu2_mc,ytmp,zhwfct,zherw_spl,x1t,x2t,x1soft,x2soft,
     #  x1x2j,x1x2jac,zherw_smn,zg2_nlo,zgmu2_nlo,xlmude,xnormc,
     #  xtmp,xnormb,xnormsv,xsum,dummy,xint
      integer loproc,maproc,jproc,j,itype,i,ipdfscale,iret
      common/cwchproc/loproc,maproc
      common/cipdfscale/ipdfscale
c
      s = xs
      x = xx
      yi = xyi
      cth1 = xcth1
      cth2 = xcth2
      sx = x*s
      xii = 1-x
c The normalization of born and soft phase spaces already accounts
c for event projection
      xphsp_ns = xjac*s/(2*1024*pi**4)
      xphsp_s = xjac*sx/(2*1024*pi**4)
      xphspb = xjac/(32*pi2)
c
      do jproc=2,3
        do j=1,4
          do itype=1,6
            vv(j,jproc,itype)=0.d0
            vvs(j,jproc,itype)=0.d0
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
        call invar(zero,s,x,yi,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
        zg2_mc = zgmu2_mc()
        ipdfscale=2
        xnorm = ze2**2 * zg2_mc * xphsp_ns
        xnorm = xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
        ytmp=1.d0
        zhwfct=zherw_spl(x,yi)
        x1t=x1soft(x1,x2,x,yi)/zhwfct
        x2t=x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          call strfun(x1t,x2t,sf)
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          do jproc=loproc,maproc
            call xmcsubt(jproc,s,cth1,x,yi,ytmp,vecmc,axlmc)
            do j=1,4
              do itype=1,6
                do i=1,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*xnorm*x1x2j*
     #              ( vecmc(j,i)*xveccoef(itype,i) +
     #                axlmc(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
        ytmp=-1.d0
        zhwfct=zherw_smn(x,yi)
        x1t=x1soft(x1,x2,x,yi)
        x2t=x2soft(x1,x2,x,yi)/zhwfct
        if(x1t.lt.1.and.x2t.lt.1)then
          call strfun(x1t,x2t,sf)
          x1x2j = x1x2jac(x1,x2,x,yi)/zhwfct
          do jproc=loproc,maproc
            call xmcsubt(jproc,s,cth1,x,yi,ytmp,vecmc,axlmc)
            do j=1,4
              do itype=1,6
                do i=1,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*xnorm*x1x2j*
     #              ( vecmc(j,i)*xveccoef(itype,i) +
     #                axlmc(j,i)*xaxlcoef(itype,i) )
                enddo
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
          call invar(zero,s,x,ytmp,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          zg2_nlo = zgmu2_nlo()
          call strfun(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1-yi) ) 
          xlmude = log(s/xmuf2h1)+log(delta/2)
          xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /(8*pi2 * delta)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call freal(s,x,ytmp,tk,uk,q1q,q2q,jproc,vecre,axlre)
            call f2pr(s,q2q,x,x,ytmp,xlmude,jproc,veccl,axlcl)
            do j=1,4
              do itype=1,6
                do i=1,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*
     #              ( xnorm*vecre(j,i)*xveccoef(itype,i) +
     #                xnorm*axlre(j,i)*xaxlcoef(itype,i) +
     #                xnormc*veccl(j,i)*xveccoef(itype,i) +
     #                xnormc*axlcl(j,i)*xaxlcoef(itype,i) )
                enddo
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
          call invar(zero,s,x,ytmp,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          zg2_nlo = zgmu2_nlo()
          call strfun(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_ns
          xnorm = xnorm * 1.d0/xii*( - 1/(1+yi) ) 
          xlmude = log(s/xmuf2h2)+log(delta/2)
          xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /(8*pi2 * delta)
          xnormc = xnormc/xii 
          do jproc=loproc,maproc
            call freal(s,x,ytmp,tk,uk,q1q,q2q,jproc,vecre,axlre)
            call f2pr(s,q1q,x,x,ytmp,xlmude,jproc,veccl,axlcl)
            do j=1,4
              do itype=1,6
                do i=1,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)+
     #              sf(j,jproc,itype)*
     #              ( xnorm*vecre(j,i)*xveccoef(itype,i) +
     #                xnorm*axlre(j,i)*xaxlcoef(itype,i) +
     #                xnormc*veccl(j,i)*xveccoef(itype,i) +
     #                xnormc*axlcl(j,i)*xaxlcoef(itype,i) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
c     Soft part of the counter-event:
c
      if(xii.lt.xicut) then
        xtmp = 1.d0
        x1t = x1soft(x1,x2,x,yi)
        x2t = x2soft(x1,x2,x,yi)
        if(x1t.lt.1.and.x2t.lt.1)then
          x1x2j = x1x2jac(x1,x2,x,yi)
          call invar(zero,sx,xtmp,yi,cth1,cth2,str,
     #               tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
          zg2_nlo = zgmu2_nlo()
          call strfun(x1t,x2t,sf)
          xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_s
          xnorm = - xnorm * 1.d0/xii*( 1/(1-yi) + 1/(1+yi) )
          xnormb = x1x2j * ze2**2 * xphspb /(2*xicut)
          xnormsv = x1x2j * ze2**2 * zg2_nlo * xphspb / 
     #             (8*pi2 * 2*xicut)
          do jproc=loproc,maproc
            call freal(sx,xtmp,yi,tk,uk,q1q,q2q,jproc,vecre,axlre)
            call fborn(sx,q1q,jproc,vecbrn,axlbrn)
            call f2sv(sx,q1q,jproc,vec2sv,axl2sv)
            do j=1,4
              do i=1,3
                xintsvc(j,jproc,i)=xnorm*vecre(j,i)
                xintsax(j,jproc,i)=xnorm*axlre(j,i)
                xbornvc(j,jproc,i)=xnormb*vecbrn(j,i)
                xbornax(j,jproc,i)=xnormb*axlbrn(j,i)
                xsvvc(j,jproc,i)=xnormsv*vec2sv(j,i)
                xsvax(j,jproc,i)=xnormsv*axl2sv(j,i)
                xcsvc(j,jproc,i)=0.d0
                xcsax(j,jproc,i)=0.d0
              enddo
            enddo
          enddo
c
          if(yi.gt.1-delta) then
             ytmp = 1.d0
             call invar(zero,sx,xtmp,ytmp,cth1,cth2,str,
     #                  tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
             xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_s
             xnorm = - xnorm * 1.d0/xii*( - 1/(1-yi) ) 
             xlmude = log(sx/xmuf2h1)+log(delta/2)
             xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /
     #                (8*pi2 * delta)
             xnormc = -xnormc/xii
             do jproc=loproc,maproc
               call freal(sx,xtmp,ytmp,tk,uk,q1q,q2q,jproc,
     #                    vecre,axlre)
               call f2pr(sx,q2q,x,xtmp,ytmp,xlmude,jproc,veccl,axlcl)
               do j=1,4
                 do i=1,3
                   xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                                xnorm*vecre(j,i)
                   xintsax(j,jproc,i)=xintsax(j,jproc,i)+
     #                                xnorm*axlre(j,i)
                   xcsvc(j,jproc,i)=xnormc*veccl(j,i)
                   xcsax(j,jproc,i)=xnormc*axlcl(j,i)
                 enddo
               enddo
             enddo
          endif
c
          if(yi.lt.-1+delta) then
             ytmp = -1.d0
             call invar(zero,sx,xtmp,ytmp,cth1,cth2,str,
     #                  tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
             xnorm = x1x2j * ze2**2 * zg2_nlo * xphsp_s
             xnorm = - xnorm * 1.d0/xii*( - 1/(1+yi) ) 
             xlmude = log(sx/xmuf2h2)+log(delta/2)
             xnormc = x1x2j * ze2**2 * zg2_nlo * xphspb /
     #                (8*pi2 * delta)
             xnormc = -xnormc/xii
             do jproc=loproc,maproc
               call freal(sx,xtmp,ytmp,tk,uk,q1q,q2q,jproc,
     #                    vecre,axlre)
               call f2pr(sx,q1q,x,xtmp,ytmp,xlmude,jproc,veccl,axlcl)
               do j=1,4
                 do i=1,3
                   xintsvc(j,jproc,i)=xintsvc(j,jproc,i)+
     #                                xnorm*vecre(j,i)
                   xintsax(j,jproc,i)=xintsax(j,jproc,i)+
     #                                xnorm*axlre(j,i)
                   xcsvc(j,jproc,i)=xcsvc(j,jproc,i)+
     #                              xnormc*veccl(j,i)
                   xcsax(j,jproc,i)=xcsax(j,jproc,i)+
     #                              xnormc*axlcl(j,i)
                 enddo
               enddo
             enddo
          endif
c
          do jproc=loproc,maproc
            do j=1,4
              do itype=1,6
                do i=1,3
                  vv(j,jproc,itype)=vv(j,jproc,itype)
     #              +sf(j,jproc,itype)*(
     #  xveccoef(itype,i)*( xintsvc(j,jproc,i)+xbornvc(j,jproc,i)+
     #                      xsvvc(j,jproc,i)+xcsvc(j,jproc,i) )+
     #  xaxlcoef(itype,i)*( xintsax(j,jproc,i)+xbornax(j,jproc,i)+
     #                      xsvax(j,jproc,i)+xcsax(j,jproc,i) ) )
                enddo
              enddo
            enddo
          enddo
        endif
      endif
c
      call checkvv(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp = 1.d0
        ytmp = 1.d0
        call invar(zero,sx,xtmp,ytmp,cth1,cth2,str,
     #             tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
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
      tot5bs=abs(xint)
c
      return
      end


      function tot5bz(xs,xx,xyi,xcth1,xcth2,xjac)
      implicit real*8 (a-z)
      tot5bz=0
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
      function svn(ro)
      implicit none
      real*8 svn,ro,tmp,be4,ybar,etacut
      common/cetacut/etacut
c
      tmp=0.d0
      if(ro.lt.1.d0-sqrt(etacut))then
        be4=(1-ro)**2
        ybar=sqrt(1-etacut/be4)
        tmp=-(1-ro)*ybar+sqrt(etacut)*asin(ybar)
      endif
      svn=tmp
      return
      end


      function f1fun(ro)
      implicit real * 8 (a-z)
      common/cetacut/etacut
c
      tmp=0.d0
      if(ro.lt.1.d0-sqrt(etacut))then
        be4=(1-ro)**2
        ybar=sqrt(1-etacut/be4)
        tmp=log((1+ybar)/(1-ybar))*( log(etacut/be4)
     #        -log(1-ybar**2)/2.d0-log(2.d0) )
     #     +ddilog((1+ybar)/2.d0)-ddilog((1-ybar)/2.d0) 
        tmp=tmp/4.d0
      endif
      f1fun=tmp
      return
      end


      function bdelta(x)
      implicit none
      real*8 bdelta,x,tmp,etacut
      common/cetacut/etacut
c
      tmp=0.d0
      if(x.lt.1.d0-dsqrt(etacut))tmp=sqrt(1-etacut/(1-x)**2)
      bdelta=tmp
      return
      end
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
      subroutine sprfin()
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
      call xout(iret)
      if(iret.eq.1)then
        if(i0.lt.1.or.i0.gt.4)then
          write(*,*)'Fatal error in sprfin'
          stop
        endif
        ycm0=ycm
        call getx1x2(tau,ycm0)
        call getmom(tau,ycm0)
        call store_events(iunit,xone)
      endif
      return
      end


      subroutine getx1x2(tau,ycm)
      implicit none
      real*8 tau,ycm,x1,x2,stau,ey
      common/cx1x2/x1,x2
c
      stau=sqrt(tau)
      ey=exp(ycm)
      x1=stau*ey
      x2=stau/ey
      return
      end


      subroutine getmom(xtau,xycm)
      implicit none
      real*8 xtau,xycm
      include 'llpcblks.h'
      real*8 pi
      parameter (pi=3.14159265358979312D0)
      integer i
      real*8 tau,ycm,theta,cth,sth,fk88random,sqsh,ycmnew
      real*8 x1,x2
      common/cx1x2/x1,x2
      real*8 xmom_cm(5,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_lb(5,4)
      common/cxmomlb/xmom_lb
      real*8 xmom_prime(5,4)
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
      do i=3,5
        call transrot(cth,sth,xmom_cm(i,1),xmom_cm(i,2))
      enddo
      if(ichkmom.eq.0)call checkmom(xmom_cm,sh,0.d0,3)
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
        do i=3,5
          call boost(-ycm,
     #         xmom_cm(i,1),xmom_cm(i,2),
     #         xmom_cm(i,3),xmom_cm(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      else
c put the partons on Herwig mass shell
        call put_on_shell(ycm,ycmnew)
        do i=1,5
          call boost(-ycmnew,
     #         xmom_prime(i,1),xmom_prime(i,2),
     #         xmom_prime(i,3),xmom_prime(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      endif
      if(ichkmom.eq.0)call checkmom(xmom_lb,sh,-ycmnew,2)
      call momnewformat()
      return
      end


      subroutine momnewformat()
c Replaces the energy with the mass in the fourth component of xmom_lb,
c to comply with the new format of the event file. Must be called as the
c last step before storing events on the temporary event files.
c If the energy is zero, the fourth component is left unchanged,
c since the LH interface uses it to distinguish between S and H events.
      implicit none
      real*8 xmom_lb(5,4)
      common/cxmomlb/xmom_lb
      real*8 xmss(1:5)
      common/procmass/xmss
      integer i
c
      do i=1,5
        if(xmom_lb(i,4).ne.0.d0)xmom_lb(i,4)=xmss(i)
      enddo
      return
      end


      subroutine boost(y,a1,a2,a3,a4,b1,b2,b3,b4)
      implicit none
      real*8 y,a1,a2,a3,a4,b1,b2,b3,b4
c
      b1=a1
      b2=a2
      b3=a3*cosh(y)-a4*sinh(y)
      b4=a4*cosh(y)-a3*sinh(y)
      return
      end


      subroutine transrot(cth,sth,xpt1,xpt2)
      implicit none
      real*8 cth,sth,xpt1,xpt2,pt1,pt2
c
      pt1=xpt1
      pt2=xpt2
      xpt1=pt1*cth+pt2*sth
      xpt2=-pt1*sth+pt2*cth
      return
      end


      subroutine put_on_shell(ycm,ycmnew)
      implicit none
      include 'llpcblks.h'
      integer i2b,i,j
      real*8 xmss(1:5),xtmp(1:4),xk1tmp(1:4),ytmp1(1:4),ytmp2(1:4),
     #  xavg3(1:3)
      real*8 ycm,ycmnew,pi,one,delta_thrs,shat,xkp2prime_norm2,
     #  xkp2prime_norm,xkprime_0,xsign,xnorm_3,delta,gamma,xmprime,
     #  xk1prime_norm,fakemass,xk1tmp_norm,xkprime_norm,xavgnorm
      parameter (pi=3.14159265358979312D0)
      parameter (one=1.d0)
      parameter (delta_thrs=0.5d-3)
      common/procmass/xmss
      real*8 xmass(-5:21)
      common/parmass/xmass
c Lepton masses
      real * 8 xlep1mass,xlep2mass
      common/clepmass/xlep1mass,xlep2mass
c x1 and x2 are the Bjorken variables; x1 is relevant to the parton
c coming from the left
      real*8 x1,x2
      common/cx1x2/x1,x2
c xmom_cm(i,j) is the j component of the four vector of the particle # i,
c given in the partonic CM frame. j=4 is the energy. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is the outgoing lepton with positive
c charge (for Z/gamma or W+ production) or a neutrino (for W- production),
c 5 is the other outgoing lepton. Momentum conservation is (1+2)-(3+4+5)=0.
      real*8 xmom_cm(5,4)
      common/cxmomcm/xmom_cm
c new momenta (put on shell) are stored here
      real*8 xmom_prime(5,4)
      common/cxmomprime/xmom_prime
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
c here, ionshell=1 or ionshell=2
      integer ionshell
      common/cionshell/ionshell
      integer ilepmass
      common/cilepmass/ilepmass
      integer ichkmom
      common/cichkmom/ichkmom
c 
      xmss(1) = xmass(ip1)
      xmss(2) = xmass(ip2)
      xmss(3) = xmass(ip3)
      if(ilepmass.eq.0)then
        xmss(4) = 0.d0
        xmss(5) = 0.d0
      elseif(ilepmass.eq.2)then
        xmss(4) = xlep1mass
        xmss(5) = xlep2mass
      else
        write(*,*)'Error in put_on_shell: unknown ilepmass',ilepmass
        stop
      endif
c i2b=0 --> 3-body kinematics; i2b=1 --> 2-body kinematics
      i2b = 0
      if(xmom_cm(3,4).lt.1.d-14)i2b=1
      if(ionshell.eq.1)then
c don't change the 3-momenta of partons 1,2 and 3, if possible
        do i=1,3
          do j=1,3
            xmom_prime(i,j)=xmom_cm(i,j)
          enddo
        enddo
        shat=(xmom_cm(1,4)+xmom_cm(2,4))**2
      elseif(ionshell.eq.2)then
c don't change the 3-momentum of parton 3, and shat, if possible
        do j=1,3
          xmom_prime(3,j)=xmom_cm(3,j)
        enddo
        do i=1,2
          do j=1,2
            xmom_prime(i,j)=xmom_cm(i,j)
          enddo
        enddo
        shat=(xmom_cm(1,4)+xmom_cm(2,4))**2
        call getxmss(shat,ycm,
     #               xmom_cm(1,3),xmss(1),
     #               xmom_cm(2,3),xmss(2),
     #               xmom_prime(1,3),xmom_prime(2,3))
      else
        write(*,*)'Fatal error in put_on_shell: unknown ionshell'
        stop
      endif
      xkprime_0=0.d0
      do i=1,3
        xsign=1.d0
        if(i.eq.3)xsign=-1.d0
        if(i.eq.3.and.i2b.eq.1)then
          xmom_prime(i,4)=0.d0
        else
          call getenergy(xmom_prime(i,1),xmom_prime(i,2),
     #                   xmom_prime(i,3),xmss(i),xmom_prime(i,4))
        endif
        xkprime_0=xkprime_0+xsign*xmom_prime(i,4)
      enddo
c compute the modulus of the 3-momentum of the pair, which is equal
c to that of parton 3 in the CM frame. The energy doesn't play any role
      call fillvec(xmom_cm(3,1),xmom_cm(3,2),
     #             xmom_cm(3,3),xmom_cm(3,4),xtmp)
      xkprime_norm=xnorm_3(xtmp)
c delta is the would-be invariant mass of the pair, minus the sum
c of the masses of the leptons
      delta=sqrt(xkprime_0**2-xkprime_norm**2)-xmss(4)-xmss(5)
      if(delta.lt.delta_thrs)then
c parton 3-momenta cannot be kept fixed: the total available energy
c is not sufficient; modify 3-momenta of the incoming partons
        gamma=sqrt( (xmss(4)+xmss(5)+delta_thrs)**2+xkprime_norm**2 )+
     #        xmom_prime(3,4)
        if(gamma.lt.(xmss(1)+xmss(2)))then
          write(6,*)'Fatal error #0 in put_on_shell'
          write(6,*)gamma,xmom_prime(3,4)
          stop
        endif
        xkp2prime_norm2=( gamma**2-2*(xmss(1)**2+xmss(2)**2)+
     #                    (xmss(1)**2-xmss(2)**2)**2/gamma**2 )/4.d0
        xkp2prime_norm=sqrt(xkp2prime_norm2)
        xmom_prime(1,3)=sign(1.d0,xmom_cm(1,3))*xkp2prime_norm
        xmom_prime(1,4)=sqrt(xkp2prime_norm2+xmss(1)**2)
        xmom_prime(2,3)=sign(1.d0,xmom_cm(2,3))*xkp2prime_norm
        xmom_prime(2,4)=sqrt(xkp2prime_norm2+xmss(2)**2)
        xkprime_0=xmom_prime(1,4)+xmom_prime(2,4)-xmom_prime(3,4)
        shat=(xmom_prime(1,4)+xmom_prime(2,4))**2 -
     #       (xmom_prime(1,3)+xmom_prime(2,3))**2
      endif
c now the parton 3-momenta have been defined in such a way
c that the momenta of the leptons can be transformed.
c xtmp is the 4-momentum of the pair. CM frame stays the same,
c so does the boost
      ycmnew=ycm
      xtmp(1)=-xtmp(1)
      xtmp(2)=-xtmp(2)
      xtmp(3)=-xtmp(3)
      xtmp(4)=xkprime_0
      xmprime=sqrt(xkprime_0**2-xkprime_norm**2)
      xk1prime_norm=xmprime**2-2*(xmss(4)**2+xmss(5)**2)+
     #              (xmss(4)**2-xmss(5)**2)**2/xmprime**2
      xk1prime_norm=sqrt(xk1prime_norm)/2.d0
      do j=1,3
        xavg3(j)=0.d0
      enddo
      do i=4,5
        xsign=1.d0
        if(i.eq.5)xsign=-1.d0
        call fillvec(xmom_cm(i,1),xmom_cm(i,2),
     #               xmom_cm(i,3),xmom_cm(i,4),ytmp1)
        call xhwulof(xtmp,xmprime,
     #               ytmp1,xmss(i),
     #               xk1tmp,fakemass)
        if(abs(fakemass-xmss(i)).gt.1.d-4)then
          write(6,*)'Fatal error #1 in put_on_shell'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        xk1tmp_norm=xnorm_3(xk1tmp)
c xavg is the direction along which the l1 and l2 momenta are placed
c in the new ll rest frame. It is arbitrarily defined by averaging 
c (hence the 1/2 in the definition) the directions of the original 
c l1 and l2 momenta. It may not have modulus 1, so normalize it
        do j=1,3
          xavg3(j)=xavg3(j)+xsign*xk1tmp(j)/(2*xk1tmp_norm)
        enddo
      enddo
      xavgnorm=sqrt(xavg3(1)**2+xavg3(2)**2+xavg3(3)**2)
      do j=1,3
        xavg3(j)=xavg3(j)/xavgnorm
      enddo
      do i=4,5
        xsign=1.d0
        if(i.eq.5)xsign=-1.d0
        do j=1,3
          xk1tmp(j)=xsign*xk1prime_norm*xavg3(j)
        enddo
        xk1tmp(4)=sqrt(xk1prime_norm**2+xmss(i)**2)
        call xhwulob(xtmp,xmprime,
     #               xk1tmp,xmss(i),
     #               ytmp2,fakemass)
        if(abs(fakemass-xmss(i)).gt.1.d-4)then
          write(6,*)'Fatal error #2 in put_on_shell'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        call getvec(ytmp2,xmom_prime(i,1),xmom_prime(i,2),
     #                    xmom_prime(i,3),xmom_prime(i,4))
      enddo
      if(ichkmom.eq.0)call checkmom(xmom_prime,shat,0.d0,4)
      return
      end


      subroutine getxmss(shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23)
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
        write(*,*)'Error # 0 in getxmss'
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
        write(*,*)'Error # 1 in getxmss'
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
            write(*,*)'Mass # 1 too large in getxmss'
            stop
          endif
        elseif(ileg.eq.2)then
          xm2=fact*sqrt(sol*shat)
          if(xm2.gt.xm2s)then
            write(*,*)'Mass # 2 too large in getxmss'
            stop
          endif
        else
          write(*,*)'Error # 2 in getxmss'
          stop
        endif
        idone=1
        goto 100
      endif
      return
      end


      subroutine fillvec(p1,p2,p3,p4,ytmp)
      implicit none
      real*8 p1,p2,p3,p4,ytmp(1:4)
c
      ytmp(1)=p1
      ytmp(2)=p2
      ytmp(3)=p3
      ytmp(4)=p4
      return
      end


      subroutine getvec(ytmp,p1,p2,p3,p4)
      implicit none
      real*8 ytmp(1:4),p1,p2,p3,p4
c
      p1=ytmp(1)
      p2=ytmp(2)
      p3=ytmp(3)
      p4=ytmp(4)
      return
      end

c-----------------------------------------------------------------------
      subroutine xhwulob(ps,ps5,pi,pi5,pf,pf5)
c     transforms pi (given in rest frame of ps) into pf (in lab)
c     n.b. p(1,2,3,4,5) = (px,py,pz,e,m)
c-----------------------------------------------------------------------
      real*8 pf4,fn,ps(4),ps5,pi(4),pi5,pf(4),pf5
      if (ps(4).eq.ps5) then
        pf(1)= pi(1)
        pf(2)= pi(2)
        pf(3)= pi(3)
        pf(4)= pi(4)
      else
        pf4  = (pi(1)*ps(1)+pi(2)*ps(2)
     &         +pi(3)*ps(3)+pi(4)*ps(4))/ps5
        fn   = (pf4+pi(4)) / (ps(4)+ps5)
        pf(1)= pi(1) + fn*ps(1)
        pf(2)= pi(2) + fn*ps(2)
        pf(3)= pi(3) + fn*ps(3)
        pf(4)= pf4
      end if
      pf5= pi5
      end

c-----------------------------------------------------------------------
      subroutine xhwulof(ps,ps5,pi,pi5,pf,pf5)
c     transforms pi (given in lab) into pf (in rest frame of ps)
c     n.b. p(1,2,3,4,5) = (px,py,pz,e,m)
c-----------------------------------------------------------------------
      real*8 pf4,fn,ps(4),ps5,pi(4),pi5,pf(4),pf5
      if (ps(4).eq.ps5) then
        pf(1)= pi(1)
        pf(2)= pi(2)
        pf(3)= pi(3)
        pf(4)= pi(4)
      else
        pf4  = (pi(4)*ps(4)-pi(3)*ps(3)
     &         -pi(2)*ps(2)-pi(1)*ps(1))/ps5
        fn   = (pf4+pi(4)) / (ps(4)+ps5)
        pf(1)= pi(1) - fn*ps(1)
        pf(2)= pi(2) - fn*ps(2)
        pf(3)= pi(3) - fn*ps(3)
        pf(4)= pf4
      end if
      pf5= pi5
      end


      subroutine getenergy(p1,p2,p3,xm,en)
      implicit none
      real*8 p1,p2,p3,xm,en
c
      en=sqrt(p1**2+p2**2+p3**2+xm**2)
      return
      end


      function dotprod(p1,p2,p3,p4,q1,q2,q3,q4)
      implicit none
      real*8 dotprod,p1,p2,p3,p4,q1,q2,q3,q4
c
      dotprod=p4*q4-p1*q1-p2*q2-p3*q3
      return
      end


      function xnorm_3(p)
c Evaluates the norm of the spatial component of a four-momentum
c The result is positive by definition, regardless of the 4-metric
      implicit none
      real*8 xnorm_3,p(1:4),tmp
c
      tmp=p(1)*p(1)+p(2)*p(2)+p(3)*p(3)
      xnorm_3=sqrt(tmp)
      return
      end


      subroutine vecsum(p,pfact,q,qfact,r)
c Weighted sum of the four-vectors p and q. The result is r
      implicit none
      integer i
      real*8 p(1:4),q(1:4),r(1:4),pfact,qfact
c
      do i=1,4
        r(i)=pfact*p(i)+qfact*q(i)
      enddo
      return
      end


      subroutine xout(iret)
c This routine is called by sprfin; it determines, on statistical
c basis, which partonic process has been generated.
c It also counts the number of unlike sign events (iwrong), and the number
c of these events (iwrong1) for which the relative difference between
c unlike signs exceeds 5%. If all the entries of vv are equal to zero,
c iret is set equal to 0 (by checkvv), and no operation is performed
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
      call checkvv(xsum,xsumabs,iretvv)
      call checkvvs(xsumvvs,xsumabsvvs,iretvvs)
      if(iretvv.eq.0.and.iretvvs.eq.1)then
        write(6,*)'Fatal error in xout:',iretvv,iretvvs
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
          write(*,*)'Fatal error in xout: ifuntype=',ifuntype
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
          write(*,*)'Fatal error in xout',i0,jproc0,itype0
          stop
        endif
        ihpro=ivbhpro(i0,jproc0,itype0)
        i1=idp1(i0,jproc0,itype0)
        i2=idp2(i0,jproc0,itype0)
        i3=idp3(i0,jproc0,itype0)
        call parcrossing(jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o)
        i1hpro=i1hproo
        ip1=ip1o
        ip2=ip2o
        ip3=ip3o
      endif
      return
      end


      subroutine parcrossing(jproc0,ihpro,i1,i2,i3,
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
          write(*,*)'Error #1 in parcrossing:',ihpro,i1,i2,i3
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
        write(*,*)'parcrossing: do not know what to do'
        write(*,*)ifuntype,jproc0
        stop
      endif
      call parcheckfin(i1hproo,ip1o,ip2o,ip3o,iallzero)
      return
      end


      subroutine checkvv(xsum,xsumabs,iret)
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


      subroutine checkvvs(xsum,xsumabs,iret)
c identical to checkvv, except for the fact that works on vvs instead of vv,
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


      function itoosoftkin()
c Returns 1 when a three-body kinematics can be safely approximated
c with a two-body kinematics. It is useful when three-body NLO configurations
c are obtained, which cannot be produced through showering
      implicit none
      integer itoosoftkin,itmp
c
      itmp=0
      itoosoftkin=itmp
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
      subroutine invar
     #(xm2,s,x,y,cth1,cth2,str,tk,uk,q1q,q2q,q1c,q2c,w1,w2,w1h,w2h)
c This routine is taken from the QQ code; the original code is that
c of MNR. Only str='p1' is a legal choice, which coincides with the 
c formulae of the paper. Call with xm2=0
      implicit none
      real * 8 xm2,s,x,y,cth1,cth2,tk,uk,q1q,q2q,q1c,q2c,
     # w1,w2,w1h,w2h
      character * 2 str
      real * 8 ptvl1,ptvl2,ptvg,yl1,yl2,yg
      common/perpen/ptvl1(2),ptvl2(2),ptvg(2)
      common/ycmvar/yl1,yl2,yg
      real * 8 s2,drs2,p10,p20,k0,k10,k20,bx,sth1,cpsi,
     # spsi,cpsi2,spsi2,cpsi1,spsi1,xktsq,xkt1sq,xkt2sq,
     # xkt,xkt1,xkt2,tmp,sqs,tiny
      parameter (tiny=1.d-14)
      real*8 xmom_cm(5,4)
      common/cxmomcm/xmom_cm
      integer ichkmom
      common/cichkmom/ichkmom
c
      tk  = -s/2*(1-x)*(1-y)
      uk  = -s/2*(1-x)*(1+y)
      s2  = tk+uk+s
      drs2 = 2*sqrt(s2)
      p10 = (s+tk)/drs2
      p20 = (s+uk)/drs2
      k0  = -(tk+uk)/drs2
      k10 = drs2/4
      k20 = drs2/4
      bx = sqrt(1-4*xm2/s2)
      sth1 = sqrt(1-cth1**2)
      if(str.eq.'p1') then
         cpsi2 = 1
         spsi2 = 0
         cpsi = 1-8*x/((1+y+x*(1-y))*(1-y+x*(1+y)))
         spsi = 4*(1-x)*sqrt(x*(1-y**2))/
     #          ((1+y+x*(1-y))*(1-y+x*(1+y)))
         cpsi1 = (1+y-x*(1-y))/(1+y+x*(1-y))
         spsi1 = sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
      else
         write(6,*) 'error in invar: str=',str
         stop
      endif
      q1q = - 2*p10*k10*(1-bx*(cth2*sth1*spsi2+cth1*cpsi2))
      q2q = - 2*p20*k20*(1+bx*(cth2*sth1*spsi +cth1*cpsi ))
      q1c = - s - tk - q1q
      q2c = - s - uk - q2q
      w1  = - q1q + q2q - tk
      w2  = - q2q + q1q - uk
      w1h = 1-bx*(cth2*sth1*spsi1+cth1*cpsi1)
      w2h = 1+bx*(cth2*sth1*spsi1+cth1*cpsi1)
c
      if(abs(q1q).lt.tiny) then
        yl1  = 1.d8
      elseif(abs(q2c).lt.tiny) then
        yl1  = -1.d8
      else
        yl1 = .5d0*log( q2c/q1q )
      endif
      if(abs(q1c).lt.tiny) then
        yl2  = 1.d8
      elseif(abs(q2q).lt.tiny) then
        yl2  = -1.d8
      else
        yl2 = .5d0*log( q2q/q1c )
      endif
      if(abs(tk).lt.tiny) then
        yg  = 1.d8
      elseif(abs(uk).lt.tiny) then
        yg  = -1.d8
      else
        yg  = .5d0*log( uk/tk )
      endif
c-----------------------------------------------------------------
c xktsq, xkt1sq e xkt2sq are the square of transverse momenta of g, l1, 
c and l2 respectively. The axis orientation is such that l1 is always
c along the x direction
c
      xktsq  = uk*tk/s
      if(xktsq.eq.0) then
         ptvl1(1) = sqrt(x*s)/2.d0*bx*sth1
         ptvl1(2) = 0.d0
         ptvl2(1) = -ptvl1(1)
         ptvl2(2) = 0.d0
         ptvg(1) = 0.d0
         ptvg(2) = 0.d0
      else
         xkt1sq = q2c*q1q/s - xm2
         xkt2sq = q2q*q1c/s - xm2
         xkt = sqrt(xktsq)
         xkt1 = sqrt(xkt1sq)
         xkt2 = sqrt(xkt2sq)
         ptvl1(1) = xkt1
         ptvl1(2) = 0.d0
         ptvl2(1) = (xktsq-xkt1sq-xkt2sq)/(2.d0*xkt1)
         tmp = xkt2sq-ptvl2(1)**2
         if(tmp.gt.0.d0)then
            ptvl2(2) = sqrt(tmp)
         else
            ptvl2(2) = 0.d0
         endif
         ptvg(1) = (xkt2sq-xkt1sq-xktsq)/(2.d0*xkt1)
         tmp = xktsq-ptvg(1)**2
         if(tmp.gt.0.d0)then
            ptvg(2) = -sqrt(tmp)
         else
            ptvg(2) = 0.d0
         endif
      endif
      if(ichkmom.eq.0)call checkptcon(ptvl1,ptvl2,ptvg)
c
      sqs=sqrt(s)
      xmom_cm(1,1)=0.d0
      xmom_cm(1,2)=0.d0
      xmom_cm(1,3)=sqs/2.d0
      xmom_cm(1,4)=sqs/2.d0
      xmom_cm(2,1)=0.d0
      xmom_cm(2,2)=0.d0
      xmom_cm(2,3)=-sqs/2.d0
      xmom_cm(2,4)=sqs/2.d0
      call fillmom(ptvl1,yl1,xm2,xmom_cm,4)
      call fillmom(ptvl2,yl2,xm2,xmom_cm,5)
      if(tk.eq.0.d0.and.uk.eq.0.d0)then
        xmom_cm(3,1)=0.d0
        xmom_cm(3,2)=0.d0
        xmom_cm(3,3)=0.d0
        xmom_cm(3,4)=0.d0
      elseif(tk.eq.0)then
        xmom_cm(3,1)=0.d0
        xmom_cm(3,2)=0.d0
        xmom_cm(3,3)=-uk/(2*sqs)
        xmom_cm(3,4)=xmom_cm(3,3)
      elseif(uk.eq.0)then
        xmom_cm(3,1)=0.d0
        xmom_cm(3,2)=0.d0
        xmom_cm(3,3)=tk/(2*sqs)
        xmom_cm(3,4)=-xmom_cm(3,3)
      else
        call fillmom(ptvg,yg,0.d0,xmom_cm,3)
      endif
      if(ichkmom.eq.0)call checkmom(xmom_cm,s,0.d0,1)
      return
      end


      subroutine fillmom(pt,y,xm2,xmom,ipart)
      implicit none
      real * 8 pt(2),y,xm2,xmom(5,4)
      integer ipart
      real * 8 xmt
c
      xmt=sqrt(pt(1)**2+pt(2)**2+xm2)
      xmom(ipart,1)=pt(1)
      xmom(ipart,2)=pt(2)
      xmom(ipart,3)=xmt*sinh(y)
      xmom(ipart,4)=xmt*cosh(y)
      return
      end


      subroutine checkmom(xmom,smax,ybst,iflag)
      implicit none
      real * 8 xmom(5,4)
      real * 8 smax,ybst,xpmax
      real*8 x1,x2
      common/cx1x2/x1,x2
      real * 8 tiny,vtiny,xsum(4),xsuma(4),xsign,xrat(4)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-4)
      integer iflag,i,j,jj,jflag,jeflag
c
      jflag=0
      jeflag=0
      xpmax=sqrt(smax)/2.d0*(1+vtiny)
      do i=1,4
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=1,5
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
          do j=1,5
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
        do j=1,5
          write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
        enddo
        stop
      endif
      return
      end


      subroutine checkptcon(ptvl1,ptvl2,ptvg)
      implicit none
      real*8 ptvl1(2),ptvl2(2),ptvg(2),tiny,pt1,pt2,ptmax
      parameter (tiny=1.d-5)
      integer jj
c
      ptmax=max(abs(ptvl1(1)),abs(ptvl2(1)),abs(ptvg(1)),
     #          abs(ptvl1(2)),abs(ptvl2(2)),abs(ptvg(2)))
      pt1=ptvl1(1)+ptvl2(1)+ptvg(1)
      pt2=ptvl1(2)+ptvl2(2)+ptvg(2)
      if(pt1.gt.ptmax*tiny.or.pt2.gt.ptmax*tiny)then
        write(*,*)'Transverse momentum is not conserved'
        write(*,'(4(d14.8,1x))') (ptvl1(jj),jj=1,2)
        write(*,'(4(d14.8,1x))') (ptvl2(jj),jj=1,2)
        write(*,'(4(d14.8,1x))') (ptvg(jj),jj=1,2)
        stop
      endif
      return
      end


      function bwfunc(s,xm02,gah)
c Returns the Breit Wigner function, normalized in such a way that
c its integral in the range (-inf,inf) is one
      implicit none
      real*8 bwfunc,s,xm02,gah
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      bwfunc=xm0*gah/(pi*((s-xm02)**2+xm02*gah**2))
      return
      end


      function xbwmass2(t)
c Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
c and the mass range (implicit in bwdelf and bwfmmn). This function
c is the inverse of F(M^2), where
c   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
c   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c and therefore eats up the Breit-Wigner when changing integration 
c variable M^2 --> t
      implicit none
      real*8 xbwmass2,t
      include 'llpcblks.h'
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      xbwmass2=xm02+xm0*gah*tan(pi*bwdelf*t-bwfmmn)
      return
      end


      subroutine zzchvar(parth1,cth1,xjac,ro)
c
c Given 0<parth1<1 returns -1<cth1<1
c and multiplies xjac times the d cth1 / d parth1 jacobian
c
      implicit none
      real * 8 parth1,cth1,xjac,ro,bb,xlgbb,yy,expyy
      bb = 1-ro**2/16
      xlgbb = log((1+bb)/(1-bb))
      yy = ( parth1 * 2 - 1 ) * xlgbb
      xjac = xjac * 2 * xlgbb
      expyy = exp(-yy)
      cth1 = (1-expyy)/(1+expyy)/bb
      xjac = xjac * 2 * expyy/(1+expyy)**2 / bb
      return
      end
c
c
c End of phase-space routines
c
c
      FUNCTION FK88RANDOM(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      DOUBLE PRECISION MINV,FK88RANDOM
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      FK88RANDOM = SEED*MINV
      END
c
c
c Initialization
c
c
      subroutine setpar()
      implicit none
      include 'llpcblks.h'
      real * 8 pi,zup,zdo,aem,xmz,xmw,xmz2,xmw2,xme,xmmu,xmtau,
     #  cthw2,sthw2,deccf,xalfaem
      parameter (pi=3.14159265358979312D0)
      parameter (zup=2.d0/3.d0,zdo=-1.d0/3.d0)
c Values from PDG 2003
      parameter (xmz=91.1876d0)
      parameter (xmw=80.425d0)
      parameter (xme=0.510998902d-3)
      parameter (xmmu=105.6583568d-3)
      parameter (xmtau=1776.99d-3)
      real * 8 zel(1:6),zel2(1:6),ac(1:6),vc(1:6)
      real * 8 ckm(1:6,1:6),ckm2(1:6,1:6),xkm(1:6,1:6)
      real * 8 alowal(1:6),blowal(1:6),auppal(1:6),buppal(1:6)
      real * 8 xalalow,xalblow,xalaupp,xalbupp
      real * 8 xnualow,xnublow,xnuaupp,xnubupp
      real * 8 weinsc(1:3),xlepacf(1:3),xlepbcf(1:3)
      real * 8 xqrkacf(1:6,1:3),xqrkbcf(1:6,1:3)
      real * 8 xveccoef(1:6,1:3),xaxlcoef(1:6,1:3)
      real * 8 xlep1mass,xlep2mass
      real * 8 wdtwon,wdtwmsb,wdtzon,wdtzmsb
      common/charges/zel,zel2
      common/cckm2/ckm2
      common/cxvacoef/xveccoef,xaxlcoef
      common/clepmass/xlep1mass,xlep2mass
      common/partdec/wdtwon,wdtwmsb,wdtzon,wdtzmsb
      integer iprdct0
      common/ciprdct0/iprdct0
      integer iprdct
      common/ciprdct/iprdct
      integer ip4,ip5
      common/ci2part/ip4,ip5
c Neutrino from Z decay
      integer ineuz(1:6)
      data ineuz/0,0,0,12,14,16/
      integer i,j,ilep
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
      zel(1)=zup
      zel(2)=zdo
      zel(3)=zdo
      zel(4)=zup
      zel(5)=zdo
      zel(6)=zup
      do i=1,6
        zel2(i)=zel(i)**2
      enddo
c axial and vector couplings to the Z (Ellis Stirling Webber)
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
c a and b coefficients, Aurenche and Lindfors NPB185(81)274, eq.(A.9)
      do i=1,6
        alowal(i)=vc(i)/2.d0
        blowal(i)=-ac(i)/2.d0
      enddo
c A and B coefficients, Aurenche and Lindfors NPB185(81)274, eq.(A.10)
      do i=1,6
        auppal(i)=alowal(i)**2+blowal(i)**2
        buppal(i)=2*alowal(i)*blowal(i)
      enddo
c a, b, A, and B coefficients, for charged leptons
      xalalow=-0.25d0+sthw2
      xalblow=0.25d0
      xalaupp=xalalow**2+xalblow**2
      xalbupp=2*xalalow*xalblow
c a, b, A, and B coefficients, for neutrinos
      xnualow=0.25d0
      xnublow=-0.25d0
      xnuaupp=xnualow**2+xnublow**2
      xnubupp=2*xnualow*xnublow
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
c Theta_W dependence of the three terms in eqs.(7) and (8) of NPB185(81)274
      if(iprdct.eq.135)then
        weinsc(1)=1/4.d0
        weinsc(2)=-2/(4*sthw2*cthw2)
        weinsc(3)=1/(4*sthw2**2*cthw2**2)
      elseif(iprdct.eq.136)then
        weinsc(1)=0.d0
        weinsc(2)=0.d0
        weinsc(3)=1/(4*sthw2**2*cthw2**2)
      elseif(iprdct.eq.137)then
        weinsc(1)=1/4.d0
        weinsc(2)=0.d0
        weinsc(3)=0.d0
      elseif(iprdct.eq.146.or.iprdct.eq.147)then
        weinsc(1)=0.d0
        weinsc(2)=0.d0
        weinsc(3)=1/(16*sthw2**2)
      else
        write(*,*)'Error in setpar: unknown IPROC',iprdct
        stop
      endif
c Coefficients appearing in eqs.(7) and (8) of NPB185(81)274, which depend
c on lepton identities
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
        if((iprdct0-10*iprdct).le.3)then
          xlepacf(1)=1.d0
          xlepbcf(1)=0.d0
          xlepacf(2)=xalalow
          xlepbcf(2)=xalblow
          xlepacf(3)=xalaupp
          xlepbcf(3)=xalbupp
        else
          xlepacf(1)=0.d0
          xlepbcf(1)=0.d0
          xlepacf(2)=0.d0
          xlepbcf(2)=0.d0
          xlepacf(3)=xnuaupp
          xlepbcf(3)=xnubupp
        endif
      elseif(iprdct.eq.146.or.iprdct.eq.147)then
        xlepacf(1)=0.d0
        xlepbcf(1)=0.d0
        xlepacf(2)=0.d0
        xlepbcf(2)=0.d0
        xlepacf(3)=0.5d0
        xlepbcf(3)=-0.5d0
      endif
c Coefficients appearing in eqs.(7) and (8) of NPB185(81)274, which depend
c on quark identities. The first index runs over types of incoming parton 
c pairs, according to the conventions given in the subroutine strfun; the 
c second index runs over the values 1,2,3, corresponding to the first, third, 
c and second term in eqs.(7) respectively (consistently with the conventions
c of the subroutine xmewrap
      do i=1,6
        do j=1,3
          xqrkacf(i,j)=0.d0
          xqrkbcf(i,j)=0.d0
        enddo
      enddo
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
        do i=1,6
          xqrkacf(i,1)=zel2(i)
        enddo
        do i=1,6
          xqrkacf(i,2)=zel(i)*alowal(i)
          xqrkbcf(i,2)=zel(i)*blowal(i)
        enddo
        do i=1,6
          xqrkacf(i,3)=auppal(i)
          xqrkbcf(i,3)=buppal(i)
        enddo
      elseif(iprdct.eq.146.or.iprdct.eq.147)then
        do i=1,6
          xqrkacf(i,3)=0.5d0
          xqrkbcf(i,3)=-0.5d0
        enddo
      endif
c Coefficients multiplying the vector (xveccoef()) and axial (xaxlcoef())
c terms in eqs.(7) and (8) of NPB185(81)274. When multiplied by the
c corresponding output of the subroutine xmewrap, the complete partonic
c cross section is obtained, up to e^4 and g_S. The index conventions 
c follow those of strfun
      do i=1,6
        do j=1,3
          xveccoef(i,j)=xqrkacf(i,j)*xlepacf(j)*weinsc(j)
          xaxlcoef(i,j)=xqrkbcf(i,j)*xlepbcf(j)*weinsc(j)
        enddo
      enddo
c Lepton masses
      ilep=iprdct0-10*iprdct
      if(ilep.lt.1.or.ilep.gt.6)then
        write(*,*)'Error in setpar: unknown lepton type',ilep
        stop
      endif
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
        if(ilep.eq.1)then
          xlep1mass=xme
          xlep2mass=xme
          ip4=-11
          ip5=11
        elseif(ilep.eq.2)then
          xlep1mass=xmmu
          xlep2mass=xmmu
          ip4=-13
          ip5=13
        elseif(ilep.eq.3)then
          xlep1mass=xmtau
          xlep2mass=xmtau
          ip4=-15
          ip5=15
        else
          if(iprdct.eq.137)then
            write(*,*)'Error in setpar: inconsistent entries'
            stop
          endif
          xlep1mass=0.d0
          xlep2mass=0.d0
          ip4=-ineuz(ilep)
          ip5=ineuz(ilep)
        endif
      elseif(iprdct.eq.146)then
        if(ilep.eq.1)then
          xlep1mass=xme
          xlep2mass=0.d0
          ip4=-11
          ip5=12
        elseif(ilep.eq.2)then
          xlep1mass=xmmu
          xlep2mass=0.d0
          ip4=-13
          ip5=14
        elseif(ilep.eq.3)then
          xlep1mass=xmtau
          xlep2mass=0.d0
          ip4=-15
          ip5=16
        else
          write(*,*)'Error in setpar: inconsistent entries'
          stop
        endif
      elseif(iprdct.eq.147)then
        if(ilep.eq.1)then
          xlep1mass=0.d0
          xlep2mass=xme
          ip4=-12
          ip5=11
        elseif(ilep.eq.2)then
          xlep1mass=0.d0
          xlep2mass=xmmu
          ip4=-14
          ip5=13
        elseif(ilep.eq.3)then
          xlep1mass=0.d0
          xlep2mass=xmtau
          ip4=-16
          ip5=15
        else
          write(*,*)'Error in setpar: inconsistent entries'
          stop
        endif
      endif
c Partial W and Z decay widths in the on-shell and MSbar schemes
      if(iprdct.lt.140)then
        if(ilep.le.3)then
          deccf=4*xalaupp
        else
          deccf=4*xnuaupp
        endif
        xmz2=xmz**2
        wdtzon=deccf*gf*xmz2*xmz/(6*sqrt(2.d0)*pi)
        wdtzmsb=deccf*xmz*xalfaem(xmz2)/(12*sthw2*cthw2)
      else
        xmw2=xmw**2
        wdtwon=gf*xmw2*xmw/(6*sqrt(2.d0)*pi)
        wdtwmsb=xmw*xalfaem(xmw2)/(12*sthw2)
      endif
      return
      end


      subroutine parsetpar()
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
c identical to i1hpro (see the routine store_events)
      do i=1,4
        do jproc=2,3
          do itype=1,6
            ivbhpro(i,jproc,itype)=0
          enddo
        enddo
      enddo
c 
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
c Z/gamma, Z or gamma production
        do itype=1,nl
          ivbhpro(1,2,itype)=401
          ivbhpro(3,2,itype)=403
          ivbhpro(1,3,itype)=402
          ivbhpro(2,3,itype)=404
          ivbhpro(3,3,itype)=405
          ivbhpro(4,3,itype)=406
        enddo
      elseif(iprdct.eq.146)then
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
      elseif(iprdct.eq.147)then
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
        write(*,*)'Unknown process in parsetpar:',iprdct
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
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
c Z/gamma, Z or gamma production
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
      elseif(iprdct.eq.146)then
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
      elseif(iprdct.eq.147)then
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
        write(*,*)'Unknown process in parsetpar:',iprdct
        stop
      endif
c
      call parcheckpar()
      return
      end


      subroutine parcheckpar()
      implicit none
      integer iallzero,i,jproc,itype,ihpro,i1,i2,i3
      parameter (iallzero=0)
      integer ivbhpro(4,2:3,6)
      common/civbhpro/ivbhpro
      integer idp1(4,2:3,6),idp2(4,2:3,6),idp3(4,2:3,6)
      common/cidpart/idp1,idp2,idp3
c
      call parcheckinit()
      do jproc=2,3
        do i=1,4
          do itype=1,6
            ihpro=ivbhpro(i,jproc,itype)
            i1=idp1(i,jproc,itype)
            i2=idp2(i,jproc,itype)
            i3=idp3(i,jproc,itype)
            call parcheckfin(ihpro,i1,i2,i3,iallzero)
          enddo
        enddo
      enddo
      return
      end


      subroutine parcheckfin(ihpro,i1,i2,i3,iallzero)
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
        write(*,*)'parcheckfin called improperly'
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
        write(*,*)'Error in parcheckfin'
        write(*,*)'ihpro,i1,i2,i3:',ihpro,i1,i2,i3
        stop
      endif
      return
      end


      subroutine parcheckinit()
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
      if(iprdct.eq.135.or.iprdct.eq.136.or.iprdct.eq.137)then
        chprdct=0.d0
      elseif(iprdct.eq.146)then
        chprdct=1.d0
      elseif(iprdct.eq.147)then
        chprdct=-1.d0
      else
        write(*,*)'Unknown process in parcheckinit:',iprdct
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
      subroutine whichone(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
c Determines the type of event at random
      implicit none
      integer iseed,itot,mx_of_evta,mx_of_evtb,iunit,i0
      real*8 xpa,xpb,tiny,one,xsum,rnd,fk88random,prob
      parameter (tiny=1.d-4)
      logical flag
c
      if(itot.le.0)then
        write(6,*)'Fatal error #1 in whichone'
        stop
      endif
      xpa=dfloat(mx_of_evta)/dfloat(itot)
      xpb=dfloat(mx_of_evtb)/dfloat(itot)
      one=xpa+xpb
      if(abs(one-1.d0).gt.tiny)then
        write(6,*)'Error #1 in whichone: probability not normalized'
        stop
      endif
      i0=0
      flag=.true.
      xsum=0.d0
      rnd=fk88random(iseed)
      do while(flag)
        if(i0.gt.2)then
          write(6,*)'Fatal error #2 in whichone'
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


      subroutine crosscheck(itot,mx_of_evta,mx_of_evtb)
c Checks whether whichone did it right
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


      subroutine retrieve_events(iunit,ii,dummy)
c Reads from disk the complete information on the events; see store_events
c for the conventions used
      implicit none
      integer iunit,ii,i,j
      real*8 dummy
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5
      common/ci2part/ip4,ip5
      integer iccode
      common/ciccode/iccode
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 xmom_lb(5,4)
      common/cxmomlb/xmom_lb
c
      read(iunit,901,end=997,err=998)i1hpro,iccode,np
      read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4,ip5
      read(iunit,903,end=997,err=998)xevsign
      read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,5)
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
      write(77,902)ip1,ip2,ip3,ip4,ip5
      write(77,903)xevsign
      write(77,904)((xmom_lb(i,j),j=1,4),i=1,5)
      stop
 999  continue
      return
      end


      subroutine store_events(iunit,xpmone)
c Stores on disk the complete information on the events. Starting
c from version 3.1, each event has the following format:
c       IPR, IC, NP
c      (ID(I),I=1,NP)
c      ((P(J,I),J=1,4),I=1,NP)
c where IPR is the subprocess code (i1hpro), IC is the colour code
c (iccode, trivial here), NP is the number of partons entering the 
c reaction (thus, this includes the soft parton in the case of S events),
c ID(I) are the particle identities (ip1,...,ip5 here), and P(J,I) are 
c the particles four momenta in the lab frame (P(J,I)=xmom_lb(i,j) here).
c
c This routine is called with xpmone=1 when events are obtained from
c SPRING, and with xpmone=-1 after the events are read from the temporary
c files (via retrieve_events), to be stored in the final event file.
c When xpmone=1, one has xevsign=+1/-1, and the weight of the event is 
c xevsign*wgt[a,b]ev. When xpmone=-1, then xevsign is the weight of the event. 
c
c i1hpro has the following conventions:
c   i1hpro         process
c    401        q qbar -> g X
c    402        q g    -> q X
c    403        qbar q -> g X
c    404        qbar g -> qbar X
c    405        g q    -> q X
c    406        g qbar -> qbar X
c    407        g g    -> g X
c X being the llbar system here (thus, 407 is unused).
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      implicit none
      integer iunit,i,j
      real*8 xpmone,xevwgt,xfact
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5
      common/ci2part/ip4,ip5
      integer iccode
      common/ciccode/iccode
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
c xmom_lb(i,j) is the j component of the four vector of the particle # i,
c given in the laboratory frame. j=4 is the energy for MC@NLO versions
c up to 2.31, the mass for version 3.1 onwards. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 and 5 are the outgoing
c antifermion and fermion respectively (leptons from the V decay).
c Momentum conservation is (1+2)-(3+4+5)=0
      real*8 xmom_lb(5,4)
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
        np=5
        if(xmom_lb(3,4).eq.0.d0)then
          xevwgt=xevsign*wgtbev
        else
          xevwgt=xevsign*wgtaev
        endif
        xfact=1.d0
      else
        write(*,*)'Fatal error in store_events: xpmone=',xpmone
        stop
      endif
      write(iunit,901)i1hpro,iccode,np
      write(iunit,902)ip1,ip2,ip3,ip4,ip5
      write(iunit,903)xevwgt
      write(iunit,904)((xmom_lb(i,j),j=1,4),i=1,5)
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
      subroutine xmcsubt(jproc,s,cth1,x,y,ycnt,vecout,axlout)
      implicit none
      real*8 s,cth1,x,y,ycnt,vecout(1:4,1:3),axlout(1:4,1:3)
      integer jproc
      real*8 xii,tiny,vcf,vtf,xmin,xlim,xlim1,xlim2,al_gfun,be_gfun,
     # ccc_gfun,yd,ydead_mod,xfact,z,zherw_spl,xi,xiherw_spl,
     # ap,ap_kern,xjac_xizspl,zherw_smn,xiherw_smn,xjac_xizsmn,
     # tt,gfact,gfun,sbar,tbar,xsoft,vecin(1:4,1:3),axlin(1:4,1:3)
      integer icode,itwo,i,j
      parameter (tiny=1.d-6)
      parameter (vcf=4/3.d0)
      parameter (vtf=1/2.d0)
      parameter (xmin=0.69519410160110384d0)
      parameter (itwo=2)
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
c
      if(abs(ycnt).ne.1.d0) then
        write(6,*)'xmcsubt called improperly: ycnt=',ycnt
        stop
      endif
      xii=1-x
      xlim=0.d0
      xlim1=0.d0
      xlim2=0.d0
      sbar=x*s
      tbar=-sbar/2.d0*(1-cth1)
      call fborn(sbar,tbar,itwo,vecin,axlin)
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
        do i=1,4
          do j=1,3
            vecout(i,j)=xlim*vecin(i,j)
            axlout(i,j)=xlim*axlin(i,j)
          enddo
        enddo
      elseif(jproc.eq.3)then
        do i=1,4
          do j=1,3
            vecout(i,j)=0.d0
            axlout(i,j)=0.d0
          enddo
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
          do i=3,4
            do j=1,3
              vecout(i,j)=xlim*vecin(i,j)
              axlout(i,j)=xlim*axlin(i,j)
            enddo
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
          do i=1,2
            do j=1,3
              vecout(i,j)=xlim*vecin(i,j)
              axlout(i,j)=xlim*axlin(i,j)
            enddo
          enddo
        endif
      else
        write(*,*)'xmcsubt: unknown process',jproc
        stop
      endif
      return
      end


      function ffunction5(xx,yy)
      implicit real * 8 (a-h,o-z)
      parameter (tiny=1.d-4)
c
      x=xx
      y=yy
      tmp=0.d0
      yd=ydead(x)
      if(abs(y).ge.yd)tmp=1.d0
      ffunction5=tmp
      return
      end


      function ydead(x)
      implicit real*8(a-h,o-z)
      parameter (xmin=0.69519410160110384d0)
c
      tmp=0.d0
      if(x.lt.xmin)tmp=1-x*(3.d0-sqrt(1+8*x))/(1-x)
      ydead=tmp
      return
      end


      function ydead_mod(x)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-4)
c
      if(1-x.lt.tiny)then
        tmp=-1/3.d0-28.d0*(X-1)/27.d0
      else
        tmp=1-x*(3.d0-sqrt(1+8*x))/(1-x)
      endif
      ydead_mod=tmp
      return
      end


      function gfun(xx)
      implicit real*8(a-h,o-z)
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
c
      x=xx
      tmp=1.d0
      if(x.lt.0.d0)then
        write(6,*)'Fatal error in gfun'
        stop
      endif
      if(x.le.1.d0.and.al_gfun.gt.0.d0)
     #  tmp=x**(2*al_gfun)/(x**(2*al_gfun)+(1-x)**(2*al_gfun))
      gfun=tmp
      return
      end
c
c
c End of MC subtraction terms
c
c
c
c
c Begin of utility routines for xi, z, and 2-->2 invariants. The functions
c for xi, z, and the jacobian have been checked numerically to coincide with
c those of the hvq package (except when closer than tiny to IR limits, since
c here more terms are kept -- which are numerically irrelevant). The present
c for is simpler and faster to computed, thanks to the unique choice of E0
c
c
      function zherw_spl(xx,yy)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-4)
c
      x=xx
      y=yy
      if(1-x.lt.tiny)then
        tmp=1.d0+(Y+3)*(X-1)/4.d0
      elseif(1-y.lt.tiny)then
        tmp=X-(X**2-3*X+2)*(Y-1)/4.d0
      else
        xmv2=x
        t=-0.5d0*(1-x)*(1-y)
        u=-0.5d0*(1-x)*(1+y)
        xa=-t/xmv2
        xb=u*t/xmv2
        tmp=1/xa*( sqrt(1+2*xa-2*xb)-1 )
      endif
      zherw_spl=tmp
      return
      end


      function zherw_smn(xx,yy)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-4)
c
      x=xx
      y=yy
      if(1-x.lt.tiny)then
        tmp=1-(Y-3)*(X-1)/4.d0
      elseif(1+y.lt.tiny)then
        tmp=X+(X**2-3*X+2)*(Y+1)/4.d0
      else
        xmv2=x
        t=-0.5d0*(1-x)*(1-y)
        u=-0.5d0*(1-x)*(1+y)
        xa=-u/xmv2
        xb=u*t/xmv2
        tmp=1/xa*( sqrt(1+2*xa-2*xb)-1 )
      endif
      zherw_smn=tmp
      return
      end


      function xiherw_spl(xx,yy)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-4)
c
      x=xx
      y=yy
      if(1-x.lt.tiny)then
        tmp=-(2*Y-2)/(Y+3)-(2*Y**3+2*Y**2-2*Y-2)*(X-1)/(Y**2+6*Y+9)
      elseif(1-y.lt.tiny)then
        tmp=-X*(Y-1)/2
      else
        xmv2=x
        t=-0.5d0*(1-x)*(1-y)
        u=-0.5d0*(1-x)*(1+y)
        xa=-t/xmv2
        z1=zherw_spl(x,y)
        tmp=xa*z1**2/(1-z1)
      endif
      xiherw_spl=tmp
      return
      end


      function xiherw_smn(xx,yy)
      implicit real*8(a-h,o-z)
      parameter (tiny=1.d-4)
c
      x=xx
      y=yy
      if(1-x.lt.tiny)then
        tmp=-(2*Y+2)/(Y-3)+(2*Y**3-2*Y**2-2*Y+2)*(X-1)/(Y**2-6*Y+9)
      elseif(1+y.lt.tiny)then
        tmp=X*(Y+1)/2
      else
        xmv2=x
        t=-0.5d0*(1-x)*(1-y)
        u=-0.5d0*(1-x)*(1+y)
        xa=-u/xmv2
        z1=zherw_smn(x,y)
        tmp=xa*z1**2/(1-z1)
      endif
      xiherw_smn=tmp
      return
      end


      function xjac_xizspl(xx,yy)
      implicit none
      real*8 xjac_xizspl,x,y,xx,yy
      real*8 z,xi,zherw_spl,xiherw_spl,tmp,tiny
      parameter (tiny=1.d-4)
c
      x=xx
      y=yy
      if(1-x.lt.tiny)then
        tmp=2/(Y+3)+(2*Y**2+4*Y+2)*(X-1)/(Y**2+6*Y+9)
      elseif(1-y.lt.tiny)then
        tmp=X/2-(3*X**2-8*X+6)*(Y-1)/8
      else
        z=zherw_spl(x,y)
        xi=xiherw_spl(x,y)
        tmp=(X-1)**2*(Y-1)*(X*Y-Y+X+1)*Z**5/
     #      (8*X**3*XI*(Z-1)**2*(XI*Z-Z-XI))
      endif
      xjac_xizspl=tmp
      return
      end


      function xjac_xizsmn(xx,yy)
      implicit none
      real*8 xjac_xizsmn,x,y,xx,yy
      real*8 z,xi,zherw_smn,xiherw_smn,tmp,tiny
      parameter (tiny=1.d-4)
c
      x=xx
      y=yy
      if(1-x.lt.tiny)then
        tmp=2/(Y-3)-(2*Y**2-4*Y+2)*(X-1)/(Y**2-6*Y+9)
      elseif(1+y.lt.tiny)then
        tmp=-X/2-(3*X**2-8*X+6)*(Y+1)/8
      else
        z=zherw_smn(x,y)
        xi=xiherw_smn(x,y)
        tmp=-(X-1)**2*(Y+1)*(X*Y-Y-X-1)*Z**5/
     #      (8*X**3*XI*(Z-1)**2*(XI*Z-Z-XI))
      endif
      xjac_xizsmn=-tmp
      return
      end
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
      function x1soft(xx1,xx2,xx,yy)
      implicit none
      real*8 x1soft,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
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
        write(*,*)'Error in x1soft',iprespl
        stop
      endif
      x1soft=tmp
      return
      end


      function x2soft(xx1,xx2,xx,yy)
      implicit none
      real*8 x2soft,xx1,xx2,xx,yy,tiny,x1,x2,x,y,csi,rx,tmp,xa,xb
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
        write(*,*)'Error in x2soft',iprespl
        stop
      endif
      x2soft=tmp
      return
      end


      function x1x2jac(xx1,xx2,xx,yy)
      implicit none
      real*8 x1x2jac,xx1,xx2,xx,yy,tiny,x1,x2,x,y,tmp,xa,xb
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
        write(*,*)'Error in x1x2jac',iprespl
        stop
      endif
      x1x2jac=abs(tmp)
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
      function zgmu2_nlo()
c Sets the scales for NLO subtraction terms
      implicit none
      real * 8 zgmu2_nlo
      real * 8 pi,ptvl1,ptvl2,ptvg,pth,pth2
      common/perpen/ptvl1(2),ptvl2(2),ptvg(2)
      parameter (pi=3.14159265358979312D0)
      include 'llpcblks.h'
      real * 8 xmu2,as,alfas
      integer inloscale
      common/cinloscale/inloscale
c
      pth2 = ptvg(1)**2 + ptvg(2)**2
      if(inloscale.eq.1)then
        xmu2 = pth2 + xmll2
      elseif(inloscale.eq.2)then
        xmu2 = xmll2
      elseif(inloscale.eq.3)then
        xmu2 = max(xmll2,pth2)
      elseif(inloscale.eq.4)then
        pth = sqrt(pth2)
        xmu2 = ( max(sqrt(xmll2)-pth,pth) )**2
      else
        write(*,*)'Unknown option in zgmu2_nlo',inloscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmuf2h1 = xmu2*xf2h1
      xmuf2h2 = xmu2*xf2h2
      xmur2  = xmu2*xren2
      as    = alfas(xmur2,xlam,nl)
      zgmu2_nlo = 4.d0*pi*as
      zg = sqrt(zgmu2_nlo)
      end


      function zgmu2_mc()
c Sets the scales for MC subtraction terms
      implicit none
      real * 8 zgmu2_mc
      real * 8 pi,ptvl1,ptvl2,ptvg,pth,pth2
      common/perpen/ptvl1(2),ptvl2(2),ptvg(2)
      parameter (pi=3.14159265358979312D0)
      include 'llpcblks.h'
      real * 8 xmu2,as,alfas
      integer imcscale
      common/cimcscale/imcscale
c
      pth2 = ptvg(1)**2 + ptvg(2)**2
      if(imcscale.eq.1)then
        xmu2 = pth2 + xmll2
      elseif(imcscale.eq.2)then
        xmu2 = xmll2
      elseif(imcscale.eq.3)then
        xmu2 = max(xmll2,pth2)
      elseif(imcscale.eq.4)then
        pth = sqrt(pth2)
        xmu2 = ( max(sqrt(xmll2)-pth,pth) )**2
      elseif(imcscale.eq.5)then
        pth = sqrt(pth2)
        xmu2 = ( sqrt(xmll2)-pth/2.d0 )**2
      else
        write(*,*)'Unknown option in zgmu2_mc',imcscale
        stop
      endif
c set the factorization scales for hadron 1 and 2, and the
c renormalization scale
      xmumcf2h1 = xmu2*xf2h1mc
      xmumcf2h2 = xmu2*xf2h2mc
      xmumcr2  = xmu2*xren2mc
      as    = alfas(xmumcr2,xlam,nl)
      zgmu2_mc = 4.d0*pi*as
      zg = sqrt(zgmu2_mc)
      end


c-------------------------------------------------------------------------
      function xalfaem(q2)
c Alpha_em(MSbar) at the scale q2 = q^2. 
c Uses alpha_Thomson below the electron mass, an interpolation between
c m_e and m_tau, and the evolution equation above m_b. This function is
c taken from the gamma*gamma* --> hadrons package
c-------------------------------------------------------------------------
      implicit real*8 (a-z)
      integer npoints,ideg
      parameter (npoints=3,ideg=3)
      real*4 ooa(npoints),xlogmu(npoints),divdif
c 1/alpha_em at m_e=0.000511,m_mu=0.1056,m_tau=1.777      
      data ooa     / 137.036, 135.95, 133.513 /
c logs of sqrt(q2) at m_e=0.000511,m_mu=0.1056,m_tau=1.777      
      data xlogmu  / -7.57914, -2.2481, 0.574927 /
      data zm/91.2d0/,ooaz/127.934d0/,pi/3.1415927d0/,nc/3/
c
      if(q2.lt.exp(2.*xlogmu(1))) then
         xalfaem = 1.d0/ooa(1)	 
      elseif(q2.lt.exp(2.*xlogmu(3))) then
         xlogq = log(q2)/2.d0
         xalfaem = 1.d0/divdif(ooa,xlogmu,npoints,sngl(xlogq),ideg)
      elseif(q2.lt.5.**2) then
         b = 3 + 2*nc*(1d0/3d0)**2 + 2*nc*(2d0/3d0)**2
         xlq = log(q2) - 2.*xlogmu(3)
         xalfaem = 1d0/ooa(3)/(1.d0 - 1.d0/3.d0/pi/ooa(3)*b*xlq)
      else
         b = 3 + 3*nc*(1d0/3d0)**2 + 2*nc*(2d0/3d0)**2
         xlq = log(q2/zm**2)
         xalfaem = 1d0/ooaz/(1.d0 - 1.d0/3.d0/pi/ooaz*b*xlq)
      endif
      return
      end
