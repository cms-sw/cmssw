      SUBROUTINE VBMAIN
c Integrates vector boson pair cross sections, and produces the event
c file which serves as input to Herwig. Quantities relevant to H and S
c events are indentified with letters a and b respectively
      implicit none
      real * 8 value(20),xmass(-5:21),xkm(3,3),xkm2(3,3),xmomshifts(4),
     # ecmlst(100),sclstf(100),sclstr(100),sclmcstf(100),sclmcstr(100),
     # xares(3),yares(3),xbres(3),ybres(3)
      integer mx_of_evta(3),mx_of_evtb(3)
      real * 8 pi,xmone,sh,xmufct2,xmuren2,as,xnc,xmcfct2,xmcren2,xlam,
     # xmw,xmz,xmw2,xmz2,zmw,zmz,zmw2,zmz2,sw,ze2,gup,gdown,ez,gw,
     # sclfct,sclren,sclmcfct,sclmcren,ga1,bw1fmpl,bw1fmmn,bw1delf,
     # xm1low2,xm1upp2,ga2,bw2fmpl,bw2fmmn,bw2delf,xm2low2,xm2upp2,
     # tmas,xpdflam4,xpdflam5,etacut,wgtaev,wgtbev,betfac,delta,
     # deltas,deltac,al_gfun,be_gfun,ccc_gfun,ecm,xscr,xscf,xmcscr,
     # xmcscf,gaw,gaz,gammax1,xm1low,xm1upp,gammax2,xm2low,xm2upp,
     # tmp,ac1,ac2,bw1mdpl,bw1mdmn,bw2mdpl,bw2mdmn,avtot,dtot,xtotal,
     # ytotal,av3a,d3a,av3nega,d3nega,ctime,av3b,d3b,av3negb,d3negb,
     # xaresall,xbresall,yaresall,ybresall,evfrac,evprcfrac,dummy,
     # al_spcfun,be_spcfun,ga1mn,ga1pl,ga2mn,ga2pl,brrv1on,brrv2on,
     # brrv1msb,brrv2msb
      integer nl,ih1,ih2,ndns1,ndns2,ifk88seed,ifk88ih,ifk88ndns,
     # ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,iwgtnorm,iprespl,ichkmom,
     # nsamp,idec,iwidth,il1hw,il2hw,ia1ora2,jproc0,iwrong,iwrong1,
     # ifuntype,ionshell,ilepmass,iinput,iverbose,ibswrite,ifk88istrl,
     # jecm,itmp,itmpvv,i,itmpih,idpdfset,itmpndns,maxevt,iseed0,iseed,
     # maxtrials,iproc,it1,it2,iseld,ncl3,loproc,maproc,jloop,ndim,
     # nwild,jproc,ibscall,ntotal,ntotala,ntotalb,ndiff,nevts,ntrls,
     # iunita,iunitb,ioutput,itot,ii,iunit,neventsuw,nqeventsuw,ifailuw,
     # ncntuws,nqcntuws,nmaxuw,nqmaxuw,ifxdaem,itd1,itd2,izero,ione
      character * 2 scheme,prc,prdct,xproc(3)
      character * 4 part1,part2
      character * 20 parm(20),gname
      character * 80 fname,fnamea,fnameb,fname1,fnamev
      character * 80 pref,prefn,prefev,prefnev
      character * 70 strin,strout,lhapdf
      character * 7 newver
      logical evgen
      external sig5a,sig5b
      parameter (pi=3.14159265358979312D0)
      parameter (xmone=-1.d0)
      parameter (izero=0)
      parameter (ione=1)
c----------------------------------------------------------
c Common blocks for the fixed variables
c
c  xlam = lambda_QCD (5 flavours)
c  xmufct2 = factorization scale squared (NLO) 
c  xmuren2 = renormalization scale squared (NLO)
c  xmcfct2 = factorization scale squared (MC) 
c  xmcren2 = renormalization scale squared (MC)
c  as = alpha_strong(xmuren2)
c  sh = (p_had1 + p_had2)^2
      common/newver/newver
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/xmumc/xmcfct2,xmcren2
      common/fk88lambda/xlam
c xmw and xmz are the masses of W and Z bosons respectively
      common/mass/xmw,xmz,xmw2,xmz2
c gaw and gaz are the widths of W and Z bosons respectively
      common/width/gaw,gaz
c Branching ratios for the two vector bosons produced
      common/brratios/brrv1on,brrv2on,brrv1msb,brrv2msb
c zmw and zmz are the masses of vector bosons #1 and #2 respectively;
c the nature of vector bosons depends on prdct. More precisely
c (#1,#2) = (W+,Z), (W-,Z), (Z,Z), (W+,W-) for 
c  prdct  =  'w+'    'w-'    'z '   'ww'
      common/zmass/zmw,zmz,zmw2,zmz2
      common/weakcoup/xkm,xkm2,sw,ze2,gup,gdown,ez,gw
      common/scalef/sclfct,sclren
      common/scalemcf/sclmcfct,sclmcren
c Common blocks used for decays with non-zero mass ranges
      common/bw1cmm/ga1,ga1mn,ga1pl,bw1fmpl,bw1fmmn,bw1delf,
     #              xm1low2,xm1upp2
      common/bw2cmm/ga2,ga2mn,ga2pl,bw2fmpl,bw2fmmn,bw2delf,
     #              xm2low2,xm2upp2
c Number of light flavours
      common/nl/nl
c ih1,ih2 = beam type 
      common/pdf/ih1,ih2,ndns1,ndns2
c----------------------------------------------------------
c Common blocks for general MC@NLO routines
c common block for internal rnd number generation, independent of bases
      common/cifk88seed/ifk88seed
c common block fk88ipdfs is filled by our interface to MLMPDF
      common/fk88ipdfs/ifk88ih,ifk88ndns
c common block w50511 and w50512 are filled by PDFLIB 
      common/w50511/ipdfih,ipdfgroup,ipdfndns,mode,nlf,lo,tmas
      common/w50512/xpdflam4,xpdflam5
c scheme = 'DI' for deep inelastic,  'MS' for msbar scheme
      common/scheme/scheme
c etacut is the maximum allowed for [2*kt(gluon)/sqrt(shat)]^2
      common/cetacut/etacut
c quark and gluon masses, used by Herwig. PDF labeling convention
      common/parmass/xmass
c al_gfun, be_gfun, ccc_gfun are the parameters entering gfun
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
c al_spcfun, be_spcfun are the parameters entering spcdamp
      common/cspcpar/al_spcfun,be_spcfun
c iwgtnorm=0 for weight=+1/-1, iwgtnorm=1 otherwise
      common/ciwgtnorm/iwgtnorm
c wgtaev and wgtbev are the norms of weights for H and S events respectively
      common/cwgtev/wgtaev,wgtbev
c iprespl=0 ==> preserves rapidity
c iprespl=1 ==> preserves longitudinal momentum
      common/ciprespl/iprespl
c ifxdaem=0 ==> uses running alpha_EM(M^2)
c ifxdaem=1 ==> uses alpha_EM=1/137.0359895
c Not implemented at the moment, to be used in future versions
      common/cifxdaem/ifxdaem
c ichkmom=0 --> enables checks on kinematics
      common/cichkmom/ichkmom
c----------------------------------------------------------
c Variables that control the integrations
c
      common/betfac/betfac,delta
      common/samp/nsamp
c cuts alla Owens-Ohnemus
      common/pmerge/deltas,deltac
c Decay of the vector boson: idec=0    -->   V's decay
c                            idec=1    -->   V's don't decay
      common/cidec/idec
c Mass ranges: iwidth=0    -->   V on shell
c              iwidth=1    -->   V off shell
      common/ciwidth/iwidth
c Type of V decays; il1hw and il2hw are entered following HERWIG conventions:
c     IL=1,..,6   for Z   ==>  e,nu_e,mu,nu_mu,tau,nu_tau
c     IL=1,2,3    for W   ==>  e,mu,tau
c The NLO conventions (used internally in this code, see setpvb) are
c     IL=1,..,6   ==>  e,mu,tau,nu_e,nu_mu,nu_tau    in all cases
      common/cilhw/il1hw,il2hw
c----------------------------------------------------------
c The following refers to the computation of MC subtraction terms
c ia1ora2=1 -> full invvbiants, ia1ora2=2 -> simplified invariants
      common/cia1ora2/ia1ora2
c----------------------------------------------------------
c Flag to select subprocess: prc = 'qq', 'qg', 'ag',
c and products: prdct = 'w+','w-','z ','ww'
c xproc(1)='qq', xproc(2)='qg', xproc(3)='ag'
      common/cxproc/xproc
      common/process/prc,prdct
      common/cjproc/jproc0
c Number of failures in flavour determination
      common/ciwrong/iwrong,iwrong1
c Common blocks for statistics relevant to secondary unweighting
      common/c1iunwgt/neventsuw,nqeventsuw,ifailuw
      common/c2iunwgt/ncntuws,nqcntuws,nmaxuw,nqmaxuw
c Average shifts in momenta, due to quark and lepton masses
      common/cshifts/xmomshifts
c----------------------------------------------------------
c ifuntype=1 for sig5a, ifuntype=2 for sig5b
      common/cifuntype/ifuntype
c Flag to put partons on shell, according to Herwig list of masses
      common/cionshell/ionshell
c Flag to put leptons on shell, according to PDF masses
      common/cilepmass/ilepmass
c

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
      iinput = 2
      mmdecide = 4
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
      open(unit=11,file='vblog',status=newver)
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
      write(11,*) ''''//prefev(1:ifk88istrl(prefev))//'''',
     # '  ! prefix for event files'
c
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
      endif
      jecm = 0
      ecm = 1.d-8
c Allow only one set of entries in this version
      dowhile(jecm.lt.1.and.ecm.gt.0)
         if(iinput.lt.2) then
            read(*,*) ecm, xscr, xscf, xmcscr, xmcscf
         else
            ecm=mmecm
            xscr=mmxren
            xscf=mmxfh
            xmcscr=mmxrenmc
            xmcscf=mmxfhmc
         endif
         write(11,'(5(1x,d10.4),1x,a)') ecm,xscr,xscf,xmcscr,xmcscf,
     #    '! energy, fren, ffact, frenmc, ffactmc'
         jecm=jecm+1
         ecmlst(jecm)=ecm*1.d-3
         sclstf(jecm)=xscf
         sclstr(jecm)=xscr
c Will use xmcscf and xmcscr in future versions
         sclmcstf(jecm)=xscf
         sclmcstr(jecm)=xscr
      enddo
      if(jecm.eq.100.and.ecm.gt.0.and.xscf.gt.0.and.xscr.gt.0.and.
     #    xmcscf.gt.0.and.xmcscr.gt.0) then
         write(*,*) 'no more than 100 values'
         stop
      endif
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter -(1)2850 for W^+W^- production'
      write(*,*)'      -(1)2860 for ZZ production'  
      write(*,*)'      -(1)2870 for W^+Z production'
      write(*,*)'      -(1)2880 for W^-Z production'
         read(*,*) itmp
      else
         itmp=mmiproc
      endif
      itmpvv=itmp
      if(itmp.eq.-2870.or.itmp.eq.-12870) then
         prdct = 'w+'
      elseif(itmp.eq.-2880.or.itmp.eq.-12880) then
         prdct = 'w-'
      elseif(itmp.eq.-2860.or.itmp.eq.-12860) then
         prdct = 'z '
      elseif(itmp.eq.-2850.or.itmp.eq.-12850) then
         prdct = 'ww'
      else
         write(*,*) 'Error: wrong IPROC'
         stop
      endif
      write(11,'(1x,i6,27x,a)') itmp,'! -2850/60/70/80=WW/ZZ/ZW+/ZW-'
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter IL=1..6, following HERWIG conventions'
      write(*,*)'      IL=7 for undecayed vector bosons'
      write(*,*)'for vector bosons 1 and 2'
         read(*,*) il1hw,il2hw
      else
         il1hw=mmil1code
         il2hw=mmil2code
      endif
      write(11,'(1x,i2,1x,i2,28x,a)') il1hw,il2hw,
     #  '! 1..6 -> V dec, 7 -> V undec'
      if( (il1hw.eq.7.and.il2hw.ne.7) .or.
     #    (il1hw.ne.7.and.il2hw.eq.7) )then
        write(*,*) 'Vector bosons must both decay or being stable'
        stop
      elseif(il1hw.eq.7.and.il2hw.eq.7)then
        idec=1
      elseif( (il1hw.ge.1.and.il1hw.le.6) .and.
     #        (il2hw.ge.1.and.il2hw.le.6) )then
        idec=0
      else
        write(*,*) 'Unknown options:',il1hw,il2hw
        stop
      endif
c Remove the following lines when spin correlations for all processes
c will be implemented
      if(idec.eq.0.and.mod(-itmpvv,10000).gt.2850)then
        write(*,*) 'Spin correlations not implemented for this process'
        stop
      endif
c
c initialize physical parameters
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter W mass and width (GeV)'
         read(*,*)xmw,gaw
      else
         xmw=mmxwm
         gaw=mmxww
      endif
      write(11,'(2(1x,d10.4),12x,a)') xmw,gaw,'! M_W, Gamma_W'
      xmw=xmw*1.d-3
      gaw=gaw*1.d-3
      xmw2=xmw**2
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter Z mass and width (GeV)'
         read(*,*)xmz,gaz
      else
         xmz=mmxzm
         gaz=mmxzw
      endif
      write(11,'(2(1x,d10.4),12x,a)') xmz,gaz,'! M_Z, Gamma_Z'
      xmz=xmz*1.d-3
      gaz=gaz*1.d-3
      xmz2=xmz**2
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter GammaX, M_V1(min), M_V1(max) for boson #1'
      write(*,*)'  If GammaX>0, the boson mass is chosen in the range'
      write(*,*)'      M0-GammaX*width < M_V1 < M0+GammaX*width'
      write(*,*)'  and M_V1(min), M_V1(max) are ignored'
      write(*,*)'  If GammaX<0, the boson mass is chosen in the range'
      write(*,*)'            M_V1(min) < M_V1 < M_V1(max)'
      write(*,*)
     #  '  If GammaX=0, the boson mass is set equal to the pole mass'
         read(*,*)gammax1,xm1low,xm1upp
      else
         gammax1=mmv1gammax
         xm1low=mmv1massinf
         xm1upp=mmv1masssup
      endif
      write(11,'(3(1x,d10.4),1x,a)') gammax1,xm1low,xm1upp,
     #     '! GammaX, M_V1(min), M_V1(max)'
      if(gammax1.lt.0.and.xm1low.ge.xm1upp)then
         write(*,*)'Enter a non-zero range'
         stop
      endif
c
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter GammaX, M_V2(min), M_V2(max) for boson #2'
      write(*,*)'  If GammaX>0, the boson mass is chosen in the range'
      write(*,*)'      M0-GammaX*width < M_V2 < M0+GammaX*width'
      write(*,*)'  and M_V2(min), M_V2(max) are ignored'
      write(*,*)'  If GammaX<0, the boson mass is chosen in the range'
      write(*,*)'            M_V2(min) < M_V2 < M_V2(max)'
      write(*,*)
     #  '  If GammaX=0, the boson mass is set equal to the pole mass'
         read(*,*)gammax2,xm2low,xm2upp
      else
         gammax2=mmv1gammax
         xm2low=mmv1massinf
         xm2upp=mmv1masssup       
      endif
      write(11,'(3(1x,d10.4),1x,a)') gammax2,xm2low,xm2upp,
     #  '! GammaX, M_V2(min), M_V2(max)'
      if(gammax2.lt.0.and.xm2low.ge.xm2upp)then
        write(*,*)'Enter a non-zero range'
        stop
      endif
c quark and gluon masses (must be consistent with Herwig)
      do i=-5,21
        xmass(i)=0.d0
      enddo
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter d, u, s, c, b, glu (Herwig) masses (GeV)'
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
      do i=-5,21
        xmass(i)=1.d-3*xmass(i)
      enddo
c initialize the labelling for parton processes
      call parsetpvb()
c set masses of the bosons produced. Postpone the exchange up and down 
c couplings for 'w-' after the call of setpvb()
      if(prdct.eq.'w+') then
         zmw=xmw
         zmz=xmz
         zmw2=xmw2
         zmz2=xmz2
         ga1=gaw
         ga2=gaz
      elseif(prdct.eq.'w-') then
         zmw=xmw
         zmz=xmz
         zmw2=xmw2
         zmz2=xmz2
         ga1=gaw
         ga2=gaz
      elseif(prdct.eq.'z ') then
         zmw=xmz
         zmz=xmz
         zmw2=xmz2
         zmz2=xmz2
         ga1=gaz
         ga2=gaz
      elseif(prdct.eq.'ww') then
         zmw=xmw
         zmz=xmw
         zmw2=xmw2
         zmz2=xmw2
         ga1=gaw
         ga2=gaw
      else
         write(*,*) 'Non implemented final state ', prdct
         stop
      endif
c
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
     #  ''''//part2//'''','! colliding hadron types'
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
         write(*,*) 'Enter Lambda_QCD_5 (GeV), < 0 for default'
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
        write(*,*)'Enter alpha and beta for the function SPC_damp'
        write(*,*)' Defaults are: alpha=1, beta=0.5'
        write(*,*)' Allowed ranges: alpha>=1, 0<beta<=1'
        read(*,*) al_spcfun,be_spcfun
        write(11,'(2(1x,d10.4),12x,a)') al_spcfun,be_spcfun,
     #    '! alpha, beta (spin corr)'
      else
        al_spcfun=1.d0
        be_spcfun=0.5d0
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
        write(*,*)
     #    'For the computation of the MEs in the MC subtraction terms'
        write(*,*)'Enter 1 to use full 2->3 invariants'
        write(*,*)'      2 to use simplified invariants'
        write(*,*)' The default is 1'
        read(*,*) ia1ora2
        write(11,'(1x,i2,31x,a)') ia1ora2,
     #    '! 1 for full, 2 for simplified invariants'
      else
        ia1ora2=1
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
     #  '! 0 => wgt=+1/-1, 1 => wgt=+w/-w'
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
c Don't modify the following parameters
      betfac = 1.d0
      delta = 1.d0
      deltas = 0
      deltac = 0
c-----------------------------------------------------------------
      if(iinput.lt.2) then
      write(*,*)' '
      write(*,*)'Enter zi ( [ 2*kt(gluon)/sqrt(shat) ]^2 < zi )'
      write(*,*)' Default is: zi=0.2'
         read(*,*) etacut
      else
         etacut=0.2d0
      endif
      write(11,'(1x,d10.4,23x,a)') etacut,'! zi'
c 
c We should actually choose iprespl=0 for Herwig 6.5 and iprespl=1 for
c Herwig 6.4 or lower, but in any case versions different from 6.5 have
c Thomas precession not taken into account (a NNLO effect anyhow).
c We always recommend the most recent version
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*)'Enter 0 to preserve rapidity'
        write(*,*)'      1 to preserve longitudinal momentum'
        read(*,*)iprespl
        write(11,'(1(1x,i8),25x,a)') iprespl,'! 0=y, 1=k_3 preserved'
      else
        iprespl=0
      endif
c
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
c---------------------------------------------------------------
c Select subprocess
c
      if(iinput.eq.1)then
        write(*,*)' '
        write(*,*) 'Enter 1 for qq, 2 for qg, 3 for ag, 0 for all'
        write(*,*) 'to select the subprocess'
        read(*,*) iproc
        write(11,'(1x,i2,31x,a)') iproc,'! 1=qq, 2=qg, 3=ag, 0=all'
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
      loproc = 1
      maproc = 3
      if(prdct.eq.'z ') then
c- ag is computed together with qg in this case.
        maproc = 2
      else
        maproc = 3
      endif
      if(iproc.ne.0) then
        loproc=iproc
        maproc=iproc
      endif
c When V's decay and have non-zero widths, compute Breit-Wigner 
c related quantities
      if(idec.eq.0)then
        if( (gammax1.ne.0.d0.and.ga1.eq.0.d0) .or.
     #      (gammax2.ne.0.d0.and.ga2.eq.0.d0) )then
          write(*,*)'Non-zero mass range require non-zero widths'
          stop
        endif
        if(gammax1.eq.0.and.gammax2.eq.0)then
          iwidth=0
          xm1low2=-1.d0
          xm1upp2=-1.d0
          xm2low2=-1.d0
          xm2upp2=-1.d0
        elseif(gammax1.ne.0.and.gammax2.ne.0)then
          iwidth=1
          if(gammax1.ge.0)then
            xm1low2=(max(0.d0,zmw-gammax1*ga1))**2
            xm1upp2=(zmw+gammax1*ga1)**2
          else
            xm1low2=xm1low**2
            xm1upp2=xm1upp**2
          endif
          xm1low2=max(100.d-6,xm1low2)
          if(xm1low2.gt.xm1upp2)then
            write(*,*)'Error in pair mass range #1'
            write(*,*)xm1low2,xm1upp2
            stop
          endif
c Parameters for the skewed Breit Wigner function
          ga1mn=ga1
          ga1pl=1.15d0*ga1
          bw1mdpl=xm1upp2-zmw2
          bw1mdmn=zmw2-xm1low2
          bw1fmpl=ga1pl/ga1*atan(bw1mdpl/(zmw*ga1pl))
          bw1fmmn=ga1mn/ga1*atan(bw1mdmn/(zmw*ga1mn))
          bw1delf=(bw1fmpl+bw1fmmn)/pi
c
          if(gammax2.ge.0)then
            xm2low2=(max(0.d0,zmz-gammax2*ga2))**2
            xm2upp2=(zmz+gammax2*ga2)**2
          else
            xm2low2=xm2low**2
            xm2upp2=xm2upp**2
          endif
          xm2low2=max(100.d-6,xm2low2)
          if(xm2low2.gt.xm2upp2)then
            write(*,*)'Error in pair mass range #2'
            write(*,*)xm2low2,xm2upp2
            stop
          endif
c Parameters for the skewed Breit Wigner function
          ga2mn=ga2
          ga2pl=1.15d0*ga2
          bw2mdpl=xm2upp2-zmz2
          bw2mdmn=zmz2-xm2low2
          bw2fmpl=ga2pl/ga2*atan(bw2mdpl/(zmz*ga2pl))
          bw2fmmn=ga2mn/ga2*atan(bw2mdmn/(zmz*ga2mn))
          bw2delf=(bw2fmpl+bw2fmmn)/pi
        else
          write(*,*)'Both mass ranges must be non-zero'
          stop
        endif
      endif
c
      call setpvb()
c W^-Z is related to W^+Z by exchange of the couplings
      if(prdct.eq.'w-') then
        tmp = gdown
        gdown = gup
        gup = tmp
        ez = - ez
      endif
c
      do jloop=1,jecm
c main loop (over energies and scale factors); jecm>1, append
c loop number at prefix
         prefn = pref
         if(jecm.gt.1) call fk88strnum(prefn,jloop)
         prefnev = prefev
         if(jecm.gt.1) call fk88strnum(prefnev,jloop)
         sh = ecmlst(jloop)**2
         sclfct = sclstf(jloop)
         sclren = sclstr(jloop)
         sclmcfct = sclmcstf(jloop)
         sclmcren = sclmcstr(jloop)
c tau generated according to a flat distribution in (1/tau)**nsamp  
         nsamp = 1
c
         avtot = 0.d0
         dtot  = 0.d0
         ndim=6
         nwild=5
c double differential
         if(iseld.eq.1)then
           xtotal=0.d0
           ytotal=0.d0
           do jproc=1,3
             xares(jproc)=0.d0
             yares(jproc)=0.d0
             xbres(jproc)=0.d0
             ybres(jproc)=0.d0
             mx_of_evta(jproc)=0
             mx_of_evtb(jproc)=0
           enddo
           do jproc=loproc,maproc
             jproc0=jproc
             prc=xproc(jproc)
             if(jproc.eq.3.and.prdct.ne.'ww')call swap(jproc)
             call fk88strcat(prefn,prc,fname)
c
             call fk88strcat(fname,'_a',fnamea)
             call run_bases(sig5a,fnamea,ndim,nwild,ncl3,it1,it2,
     #         ac1,ac2,av3a,d3a,av3nega,d3nega,ctime,itd1,itd2,iseed0,
     #         ibswrite,ibscall)
             if(basesoutput.gt.0) then
                write(*,*)'   '
                write(*,*)'|integral[a]|(',prc,'):',av3a,' +- ',d3a
              write(*,*)' integral[a] (',prc,'):',av3nega,' +- ',d3nega
             endif
             xares(jproc)=av3a
             yares(jproc)=av3nega
             xtotal=xtotal+xares(jproc)
             ytotal=ytotal+yares(jproc)
             dtot=dtot+d3nega**2
c     
             call fk88strcat(fname,'_b',fnameb)
             call run_bases(sig5b,fnameb,ndim,nwild,ncl3,it1,it2,
     #           ac1,ac2,av3b,d3b,av3negb,d3negb,ctime,itd1,itd2,iseed0,
     #            ibswrite,ibscall)
             if(basesoutput.gt.0) then
                write(*,*)'   '
             write(*,*)'|integral[b]|(',prc,'):',av3b,' +- ',d3b
             write(*,*)' integral[b] (',prc,'):',av3negb,' +- ',d3negb
             endif
             xbres(jproc)=av3b
             ybres(jproc)=av3negb
             xtotal=xtotal+xbres(jproc)
             ytotal=ytotal+ybres(jproc)
             dtot=dtot+d3negb**2
c
             if(jproc.eq.3.and.prdct.ne.'ww')call swap(jproc)
          enddo
          avtot=ytotal
          dtot=sqrt(dtot)
          call fk88strcat(prefn,'.integrals',fname)
          open(unit=21,file=fname,
     #          form='formatted',status='unknown')
          write(21,240)(xares(jproc),jproc=1,3)
          write(21,240)(xbres(jproc),jproc=1,3)
          write(21,240)(yares(jproc),jproc=1,3)
          write(21,240)(ybres(jproc),jproc=1,3)
          close(21)
 240      format(3(1x,d14.8))
       endif
c Sanity check
       if(betfac.ne.1.d0.or.delta.ne.1.d0)then
          write(*,*)'Fatal error: betfac, delta=',betfac,delta
          stop
       endif
       if(iseld.eq.0)then
c Read integrals from disk only if the integration step has been skipped
          call fk88strcat(prefn,'.integrals',fname)
          open(unit=21,file=fname,
     #         form='formatted',status='old')
          read(21,240)(xares(jproc),jproc=1,3)
          read(21,240)(xbres(jproc),jproc=1,3)
          read(21,240)(yares(jproc),jproc=1,3)
          read(21,240)(ybres(jproc),jproc=1,3)
          close(21)
       endif
c Generates events when evgen=.true.; if evgen=.false., maxevt=100000 in
c order to estimate the number of negative weights
       if(maxevt.ne.0)then
          ntotal=0
          ntotala=0
          ntotalb=0
          xtotal=0.d0
          ytotal=0.d0
          xaresall=0.d0
          xbresall=0.d0
          yaresall=0.d0
          ybresall=0.d0
          do jproc=loproc,maproc
             xtotal=xtotal+xares(jproc)+xbres(jproc)
             ytotal=ytotal+yares(jproc)+ybres(jproc)
             xaresall=xaresall+xares(jproc)
             xbresall=xbresall+xbres(jproc)
             yaresall=yaresall+yares(jproc)
             ybresall=ybresall+ybres(jproc)
          enddo
          avtot=ytotal
c     For future upgrades, define the weights of H and S events; this is 
c necessary when the relative number of H and S events is not generated
c according to total rates
          if(iwgtnorm.eq.0)then
             wgtaev=1.d0
             wgtbev=1.d0
          else
               wgtaev=xtotal/dfloat(maxevt)
               wgtbev=xtotal/dfloat(maxevt)
            endif
            do jproc=loproc,maproc
               mx_of_evta(jproc)=int(maxevt*xares(jproc)/xtotal)
               mx_of_evtb(jproc)=int(maxevt*xbres(jproc)/xtotal)
               ntotal=ntotal+mx_of_evta(jproc)+mx_of_evtb(jproc)
               ntotala=ntotala+mx_of_evta(jproc)
               ntotalb=ntotalb+mx_of_evtb(jproc)
            enddo
            ndiff=maxevt-ntotal
            if(ndiff.gt.0)then
               mx_of_evta(loproc)=mx_of_evta(loproc)+ndiff
               ntotala=ntotala+ndiff
            endif
            if(ndiff.lt.0)then
               write(6,*)'Fatal error:',maxevt,ntotal
               stop
            endif
            if(evgen)then
               if(basesoutput.gt.0) then           
                  write(*,*)'  '
                  write(*,*)
     #  'The following number of events will be generated'
               endif
               do jproc=loproc,maproc
                  if(basesoutput.gt.0) then           
                     write(*,*)
     #  'process: ',xproc(jproc),', # events[a]:',mx_of_evta(jproc)
                  endif
                  prc=xproc(jproc)
                  call fk88strcat(prefnev,prc,fname)
                  call fk88strcat(fname,'_a.events',fname1)
                  open(unit=22,file=fname1,
     #                 form='formatted',status='unknown')
                  write(22,250)mx_of_evta(jproc)
                  close(22)
                  if(basesoutput.gt.0) then           
                     write(*,*)
     #  'process: ',xproc(jproc),', # events[b]:',mx_of_evtb(jproc)
                  endif
               prc=xproc(jproc)
               call fk88strcat(prefnev,prc,fname)
               call fk88strcat(fname,'_b.events',fname1)
               open(unit=22,file=fname1,
     #              form='formatted',status='unknown')
               write(22,250)mx_of_evtb(jproc)
               close(22)
             enddo
          endif
          if(basesoutput.gt.0) then           
             write(*,*)'  '
             write(*,*)
     #  'Estimated fractions of events with negative weights'
          endif
          evfrac=0.d0
           evprcfrac=(xaresall-yaresall)/
     #               (xaresall+yaresall)
           evprcfrac=evprcfrac/(1+evprcfrac)
           evfrac=evfrac+evprcfrac*ntotala
           if(basesoutput.gt.0) then
              write(*,*)'Events[a]: w<0/all:',evprcfrac
           endif
           evprcfrac=(xbresall-ybresall)/
     #               (xbresall+ybresall)
           evprcfrac=evprcfrac/(1+evprcfrac)
           if(basesoutput.gt.0) then           
              write(*,*)'Events[b]: w<0/all:',evprcfrac
           endif
           evfrac=evfrac+evprcfrac*ntotalb
           evfrac=evfrac/dfloat(maxevt)
           if(basesoutput.gt.0) then           
              write(*,*)'Events[all]: w<0/all:',evfrac
           endif
c
           if(.not.evgen)goto 111
           do jproc=loproc,maproc
             jproc0=jproc
             prc=xproc(jproc)
             call fk88strcat(prefn,prc,fname)
             call fk88strcat(prefnev,prc,fnamev)
c
             iwrong=0
             iwrong1=0
             neventsuw=0
             nqeventsuw=0
             ifailuw=0
             ncntuws=0
             nqcntuws=0
             nmaxuw=0
             nqmaxuw=0
             do i=1,4
               xmomshifts(i)=0.d0
             enddo
             ifuntype=1
             call fk88strcat(fname,'_a',fnamea)
             call fk88strcat(fnamev,'_a.events',fname1)
             open(unit=22,file=fname1,
     #            form='formatted',status='old')
             call toend(22)
             if(jproc.eq.3.and.prdct.ne.'ww')call swap(jproc)
             call run_spring(sig5a,fnamea,mx_of_evta(jproc),maxtrials,
     #                       nevts,ntrls,ndim,nwild,iseed)
             close(22)
             if(jproc.eq.3.and.prdct.ne.'ww')call swap(jproc)
             if(iverbose.eq.1)then
               write(*,*)'   '
               write(*,*)'Process[a]: ',prc
               write(*,*)'Trials:',ntrls
               write(*,*)'Events generated:',nevts
               write(*,*)'Unlike sign events(1):',iwrong
               write(*,*)'Unlike sign events(2):',iwrong1
               write(*,*)'Unlike sign(1)/all events:',
     #                   iwrong/dfloat(nevts)
               write(*,*)'Unlike sign(2)/all events:',
     #                   iwrong1/dfloat(nevts)
               if(idec.eq.0)then
                 if(neventsuw.ne.mx_of_evta(jproc))then
                   write(*,*)'Error in spin correlations [a]'
                   stop
                 endif
                 write(*,*)'   '
                 write(*,*)'Secondary unweighting for spin correlations'
                 write(*,*)'Failures',ifailuw
                 write(*,*)'Average trials',ncntuws/dfloat(neventsuw)
                 write(*,*)'Maximum trials',nmaxuw
                 write(*,*)'Efficiency',neventsuw/dfloat(ncntuws)
                 if(iwidth.eq.1)then
                   write(6,*)'Maximum trials [Q]',nqmaxuw
                   write(6,*)'Efficiency [Q]',
     #                       nqeventsuw/dfloat(nqcntuws)
                 endif
               endif
               write(*,*)'   '
               write(*,*)'Average momentum shifts due to masses'
               do i=1,4
                 if(idec.eq.0)then
                   write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(9*nevts)
                 else
                    write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(5*nevts)
                 endif
              enddo
           endif
c     
           iwrong=0
             iwrong1=0
             neventsuw=0
             nqeventsuw=0
             ifailuw=0
             ncntuws=0
             nqcntuws=0
             nmaxuw=0
             nqmaxuw=0
             do i=1,4
               xmomshifts(i)=0.d0
             enddo
             ifuntype=2
             call fk88strcat(fname,'_b',fnameb)
             call fk88strcat(fnamev,'_b.events',fname1)
             open(unit=22,file=fname1,
     #            form='formatted',status='old')
             call toend(22)
             if(jproc.eq.3.and.prdct.ne.'ww')call swap(jproc)
             call run_spring(sig5b,fnameb,mx_of_evtb(jproc),maxtrials,
     #                       nevts,ntrls,ndim,nwild,iseed)
             close(22)
             if(jproc.eq.3.and.prdct.ne.'ww')call swap(jproc)
             if(iverbose.eq.1)then
               write(*,*)'   '
               write(*,*)'Process[b]: ',prc
               write(*,*)'Trials:',ntrls
               write(*,*)'Events generated:',nevts
               write(*,*)'Unlike sign events(1):',iwrong
               write(*,*)'Unlike sign events(2):',iwrong1
               write(*,*)'Unlike sign(1)/all events:',
     #                   iwrong/dfloat(nevts)
               write(*,*)'Unlike sign(2)/all events:',
     #                   iwrong1/dfloat(nevts)
               if(idec.eq.0)then
                 if(neventsuw.ne.mx_of_evtb(jproc))then
                   write(*,*)'Error in spin correlations [b]'
                   stop
                 endif
                 write(*,*)'   '
                 write(*,*)'Secondary unweighting for spin correlations'
                 write(*,*)'Failures',ifailuw
                 write(*,*)'Average trials',ncntuws/dfloat(neventsuw)
                 write(*,*)'Maximum trials',nmaxuw
                 write(*,*)'Efficiency',neventsuw/dfloat(ncntuws)
                 if(iwidth.eq.1)then
                   write(6,*)'Maximum trials [Q]',nqmaxuw
                   write(6,*)'Efficiency [Q]',
     #                       nqeventsuw/dfloat(nqcntuws)
                 endif
               endif
               write(*,*)'   '
               write(*,*)'Average momentum shifts due to masses'
               do i=1,4
                 if(idec.eq.0)then
                   write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(8*nevts)
                 else
                   write(*,*)'  ',i,': ',xmomshifts(i)/dfloat(4*nevts)
                 endif
               enddo
             endif
           enddo
c write a single event file
           do jproc=loproc,maproc
             prc=xproc(jproc)
             call fk88strcat(prefnev,prc,fname)
             iunita=20+jproc
             call fk88strcat(fname,'_a.events',fname1)
             open(unit=iunita,file=fname1,form='formatted',status='old')
             read(iunita,250)mx_of_evta(jproc)
             iunitb=23+jproc
             call fk88strcat(fname,'_b.events',fname1)
             open(unit=iunitb,file=fname1,form='formatted',status='old')
             read(iunitb,250)mx_of_evtb(jproc)
           enddo
           call fk88strcat(prefnev,'.events',fname1)
           ioutput=30
           open(unit=ioutput,file=fname1,form='formatted',
     #          status='unknown')
c Write all the quantities which identify the run
           write(ioutput,801)
     #       ecmlst(jloop)*1.d3,sclstr(jloop),sclstf(jloop),
     #       sclmcstr(jloop),sclmcstf(jloop),
     #       '--> CM energy, muR/mu0[NLO], muF/mu0[NLO], '//
     #       'muR/mu0[MC], muF/mu0[MC]'
           write(ioutput,802)abs(itmpvv),
     #       '--> 2850/60/70/80=WW/ZZ/ZW+/ZW-'
           write(ioutput,803)xmw*1.d3,xmz*1.d3,'--> M_W, M_Z'
           write(ioutput,810)il1hw,il2hw,'--> IL1, IL2 (1,..,7)'
           write(ioutput,813)zmw*1.d3,ga1*1.d3,gammax1,
     #       '--> M_V1, Ga_V1, GammaX(V1)'
           write(ioutput,813)zmz*1.d3,ga2*1.d3,gammax2,
     #       '--> M_V2, Ga_V2, GammaX(V2)'
           write(ioutput,804)xmass(1)*1.d3,xmass(2)*1.d3,
     #                       xmass(3)*1.d3,xmass(4)*1.d3,
     #                       xmass(5)*1.d3,xmass(21)*1.d3,
     #                       '--> quark and gluon masses'
           write(ioutput,805)part1,part2,'--> colliding particles'
           write(ioutput,806)gname(1:8),idpdfset,
     #       '--> PDF group and id number'
           write(ioutput,807)xlam,scheme,'--> Lambda_5, scheme'
           write(ioutput,811)'P,M','--> Format of v3.1 and higher'
           write(ioutput,250)maxevt
           itot=maxevt
           do ii=1,maxevt
             call whichovb(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
             call retrieve_evenvb(iunit,ii,dummy)
             call store_evenvb(ioutput,xmone)
           enddo
           call crosschevb(itot,mx_of_evta,mx_of_evtb)
           do jproc=loproc,maproc
             iunita=20+jproc
             iunitb=23+jproc
             close(iunita)
             close(iunitb)
             close(ioutput)
           enddo
 111       continue
         endif
         if(basesoutput.gt.0) then
         if(idec.eq.0)then
           write(*,*) '   '
           write(*,*)'Branching ratios in MSbar and on-shell schemes'
           write(*,*)'MSbar is used in the present computation'
           write(*,*)'V1, MSbar:    Gamma/Gamma_tot =',brrv1msb
           write(*,*)'V1, On-shell: Gamma/Gamma_tot =',brrv1on
           write(*,*)'V2, MSbar:    Gamma/Gamma_tot =',brrv2msb
           write(*,*)'V2, On-shell: Gamma/Gamma_tot =',brrv2on
         endif
         write(*,*) '   '
         write(*,*) 'Total for fully inclusive'
         write(*,200)ih1,ih2,ndns1,ndns2,nl,xlam
         write(*,201) 'tot'
         write(*,300)ecmlst(jloop)*1.d3,sclstf(jloop),sclstr(jloop),
     #               avtot,dtot
         endif
c end of the main loop
      enddo
 200  format(' had1=',i2,'  had2=',i2,'  strf1=',i6,'  strf2=',i6,
     #  '  nl=',i2,'  lambda5=',d10.4)
 201  format(' ecm           xf   xr   ',a,
     # '        err    ')
 300  format((1x,1pd9.3),4x,2(1x,0pf4.2),2(1x,0pd10.4))
 250  format(1x,i8)
 801  format(5(1x,d10.4),1x,a)
 802  format(1x,i6,1x,a)
 803  format(2(1x,d10.4),1x,a)
 804  format(6(1x,d10.4),1x,a)
 805  format(2(1x,a4),1x,a)
 806  format(1x,a8,1x,i6,1x,a)
 807  format(1x,d10.4,1x,a2,1x,a)
 810  format(2(1x,i2),1x,a)
 811  format(1x,a3,1x,a)
 813  format(3(1x,d10.4),1x,a)
      end


      subroutine getsvb(str,ndns,ih)
      implicit real * 8 (a-h,o-z)
      character * (*) str, scheme * 2
 2    write(*,*) str
      write(*,*)
     # '   (< 0 for a display of the features of the various sets'
      read(*,*) ndns
      if(ndns.lt.0) then
        call prntsf
        go to 2
      endif
      call pdfpar(ndns,ih,xlam,scheme,iret)
      if(iret.ne.0) goto 2
      end


C      subroutine toend(iunit)
C      ios = 0    
C      dowhile(ios.eq.0)
C         read(unit=iunit,fmt='(1x)',iostat=ios)
C      enddo                        
C      end


      function sig5a(xx)
c Integrand function for H events
      implicit none
      real * 8 sig5a,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,x,ttt,th,y,csi,rx,rohx,taumax,ximax0,
     #  ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,rox,cth1,
     #  th2,tvb5a
c
c xx(1) --> tau, xx(2) --> ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> th2
c
      xjac = 1
      roh = (zmw+zmz)**2/sh
c
c To improve convergence in the soft regions
c
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
      y    = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohx = roh/x
      taumax = 1/x
      ximax0 = rohx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(1)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
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
      s = sh * tau
c
c Change variables from xx(5) to cth1, xjac--> xjac * d cth1/d xx(5)
c
      rox = 2*(zmw2+zmz2)/(s*x)-(zmw2-zmz2)**2/(s*x)**2
      call wzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * pi
      xjac = xjac * pi
c
      sig5a = tvb5a(s,x,y,cth1,th2,xjac)
      return
      end


      function tvb5a(s,xx,xy,xcth1,xth2,xjac)
      implicit none
      real * 8 tvb5a,s,xx,xy,xcth1,xth2,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2=pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      integer ione
      parameter (ione=1)
      character * 2 str
      parameter (str='p1')
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/weakcoup/xkm,xkm2,sw,ze2,gup,gdown,ez,gw
      real * 8 etacut
      common/cetacut/etacut
      real * 8 betfac,delta
      common/betfac/betfac,delta
      real * 8 deltas,deltac
      common/pmerge/deltas,deltac
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 ps,px,py,pcth1,pcth2
      common/cpsave/ps,px,py,pcth1,pcth2
      real * 8 vv(2,2,3,10),vvs(2,2,3,10)
      common/cvv/vv
      common/cvvs/vvs
      integer jproc
      common/cjproc/jproc
      integer ipdfscale
      common/cipdfscale/ipdfscale
      integer idec
      common/cidec/idec
      character * 2 prc,prdct
      common/process/prc,prdct
      real * 8 sf(2,2,3,10),www(2,3)
      real * 8 x,y,cth1,th2,cth2,sx,ro,b,rox,bx,rolimx,blimx,btildex,
     #  rotildx,xktrel,csi,rx,x1,x2,xnorm,xnormc,xnormsv,xnormborn,
     #  xlgomx,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,zg2,zgmu2_nvb,
     #  xfact,ww3up,ww3do,ffunval5,ffunction5,zgmu2_vb,ytmp,zhwfct,
     #  zherw_spl,x1t,x1soft,x2t,x2soft,x1x2j,x1x2jac,zherw_smn,xsum,
     #  dummy,xint,xtmp
      integer i,j,itype,iret,i2b,itoosoftkin
c
      x      = xx
      y      = xy
      cth1   = xcth1
      th2    = xth2
      cth2   = cos(th2)
c
      sx      = s*x
      ro      = 2*(zmw2+zmz2)/s-(zmw2-zmz2)**2/s**2
      b       = sqrt(1-ro)
      rox     = 2*(zmw2+zmz2)/(sx)-(zmw2-zmz2)**2/(sx)**2
      bx      = sqrt(1-rox)
      rolimx  = (zmw+zmz)**2/sx
      blimx   = sqrt(1-rolimx)
      btildex = blimx*betfac
      rotildx = 1-btildex**2
      xktrel = (1-x)**2*(1-y**2)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-'.or.prdct.eq.'z ')then
        xnorm = ze2**2 * xjac / s/(8*pi)**2 / (16*pi2)
        xnormc = ze2**2/(16*pi2) * xjac / (16*pi)
      elseif(prdct.eq.'ww')then
        xnorm = ze2**2 * xjac/( (4*pi*s) * (4*pi)**2 * (16*pi) )
        xnormc = ze2**2 * xjac/(8*pi2*16*pi)
      else
        write(*,*) 'tvb5a: non implemented final state', prdct
        stop
      endif
      xnormsv = ze2**2 * xjac/(16*pi2)/(16*pi)
      xnormborn = ze2**2 * xjac/(16*pi)
c
      xlgomx = log(1-x)
      do i=1,2
        do j=1,2
          do itype=1,10
            vv(i,j,jproc,itype)=0.d0
            vvs(i,j,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
      if(x1.lt.1.and.x2.lt.1)then
c
c Event
c
        call invvb(zmw2,zmz2,s,x,y,cth1,cth2,zero,zero,zero,zero,
     #        str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
        zg2 = zgmu2_nvb()
        ipdfscale=1
        call strfvb(x1,x2,sf)
        xfact = xnorm*zg2*bx/(1-x)*( 1/(1-y) + 1/(1+y) )
        call vbfpp(s,x,y,q1q,q2q,ww3up,ww3do)
        www(1,jproc) = xfact * ww3up
        www(2,jproc) = xfact * ww3do
        do i=1,2
          do j=1,2
            do itype=1,10
              vv(i,j,jproc,itype)=sf(i,j,jproc,itype)*www(i,jproc)
            enddo
          enddo
        enddo
c
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone. Notice
c that these contribute only if x1<1, x2<1 as for the event
c 
        ffunval5 = ffunction5(x,y)
        if(ffunval5.ne.0.d0)then
          zg2 = zgmu2_vb()
          ipdfscale=2
          xfact = xnorm*zg2*bx/(1-x)*( 1/(1-y) + 1/(1+y) )
          if(prc.eq.'qq') then
            ytmp = 1.d0
            zhwfct=zherw_spl(x,y)
            x1t = x1soft(x1,x2,x,y)/zhwfct
            x2t = x2soft(x1,x2,x,y)
            if(x1t.lt.1.and.x2t.lt.1)then
              call strfvb(x1t,x2t,sf)
              x1x2j = x1x2jac(x1,x2,x,y)/zhwfct
              call xmcsuvb(s,x,y,cth1,cth2,x1,x2,ytmp,ww3up,ww3do)
              www(1,jproc) = - xfact * x1x2j * ww3up 
              www(2,jproc) = - xfact * x1x2j * ww3do 
              do i=1,2
                do j=1,2
                  do itype=1,10
                    vv(i,j,jproc,itype)=vv(i,j,jproc,itype)+
     #                             sf(i,j,jproc,itype)*www(i,jproc)
                  enddo
                enddo
              enddo
            endif
          endif
c
          ytmp = -1.d0
          zhwfct=zherw_smn(x,y)
          x1t = x1soft(x1,x2,x,y)
          x2t = x2soft(x1,x2,x,y)/zhwfct
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfvb(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,y)/zhwfct
            call xmcsuvb(s,x,y,cth1,cth2,x1,x2,ytmp,ww3up,ww3do)
            www(1,jproc) = - xfact * x1x2j * ww3up 
            www(2,jproc) = - xfact * x1x2j * ww3do 
            do i=1,2
              do j=1,2
                do itype=1,10
                  vv(i,j,jproc,itype)=vv(i,j,jproc,itype)+
     #                           sf(i,j,jproc,itype)*www(i,jproc)
                enddo
              enddo
            enddo
          endif
        endif
      endif
c
      call checkvb(xsum,dummy,iret)
      if(iret.eq.1)then
        call invvb(zmw2,zmz2,s,x,y,cth1,cth2,zero,zero,zero,zero,
     #        str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
        if(idec.eq.0)then
          ps=s
          px=x
          py=y
          pcth1=cth1
          pcth2=cth2
        endif
c Cross section in pb (momenta are in TeV in the computation)
        xint = hc2*xsum
        xevsign = 1.d0
        if(xint.lt.0.d0)xevsign = -1.d0
        i2b = itoosoftkin()
        if(i2b.eq.1)then
          xtmp = 1.d0
          ytmp = 1.d0
          call invvb(zmw2,zmz2,sx,xtmp,ytmp,cth1,cth2,zero,zero,zero,
     #      zero,str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
          if(idec.eq.0)then
            ps=sx
            px=xtmp
            py=ytmp
            pcth1=cth1
            pcth2=cth2
          endif
        endif
      else
        xint = 0.d0
        xevsign = 1.d0
      endif
c
      bsfsgn = xevsign
      tvb5a = abs(xint)
c
      return
      end


      function sig5b(xx)
c Integrand function for S events
      implicit none
      real * 8 sig5b,xx
      real * 8 pi,tiny
      parameter (pi=3.14159265358979312D0)
      parameter (tiny=1.d-5)
      dimension xx(6)
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      integer nsamp
      common/samp/nsamp
      integer iprespl
      common/ciprespl/iprespl
      real * 8 xjac,roh,zzz,x,ttt,th,y,csi,rx,rohx,taumax,ximax0,
     #  ximin0,tmp,ymax,ymin,xxa1,xxa2,xxc,xxymax,xxymin,s,rox,cth1,
     #  th2,tvb5b
c
c xx(1) --> tau, xx(2) --> ycm, xx(3) --> x, xx(4) --> y, xx(5) --> cth1,
c xx(6) --> th2
c
      xjac = 1
      roh = (zmw+zmz)**2/sh
c
c To improve convergence in the soft regions
c
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
      y    = cos(th)
      xjac = xjac * sin(th)
c
c Generation of tau and ycm values and computation of the integration
c limits:
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
      rohx = roh/x
      taumax = 1/x
      ximax0 = rohx**(-nsamp)
      ximin0 = taumax**(-nsamp)
      tmp  = ximin0 + xx(1)*(ximax0-ximin0)
      tau = tmp**(-1/dfloat(nsamp))
      xjac= xjac/nsamp*tau**(nsamp+1)*(ximax0-ximin0)
      if(iprespl.eq.0)then
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
      s = sh * tau
c
c Change variables from xx(5) to cth1, xjac--> xjac * d cth1/d xx(5)
c
      rox = 2*(zmw2+zmz2)/(s*x)-(zmw2-zmz2)**2/(s*x)**2
      call wzchvar(xx(5),cth1,xjac,rox)
c
      th2 = xx(6) * pi
      xjac = xjac * pi
c
      sig5b = tvb5b(s,x,y,cth1,th2,xjac)
      return
      end


      function tvb5b(s,xx,xy,xcth1,xth2,xjac)
      implicit none
      real * 8 tvb5b,s,xx,xy,xcth1,xth2,xjac
      real * 8 pi,pi2,zero,hc2
      parameter (pi=3.14159265358979312D0)
      parameter (pi2=pi*pi)
      parameter (zero=0.d0)
c GeV to microbarn conversion factor: sigma (mub) = hc2 * sigma (GeV^-2)
c TeV to picobarn conversion factor: sigma (pb) = hc2 * sigma (TeV^-2)
c sigma (pb) = 10^6 * sigma (mub)
      parameter (hc2=3.8937966d2)
      integer ione
      parameter (ione=1)
      character * 2 str
      parameter (str='p1')
      real * 8 bsfsgn
      common/cbssgn/bsfsgn
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/weakcoup/xkm,xkm2,sw,ze2,gup,gdown,ez,gw
      real * 8 etacut
      common/cetacut/etacut
      real * 8 betfac,delta
      common/betfac/betfac,delta
      real * 8 deltas,deltac
      common/pmerge/deltas,deltac
      real * 8 ycm,tau
      common/x1x2/ycm,tau
      real * 8 xevsign
      common/cxevsign/xevsign
      real * 8 ps,px,py,pcth1,pcth2
      common/cpsave/ps,px,py,pcth1,pcth2
      real * 8 vv(2,2,3,10),vvs(2,2,3,10)
      common/cvv/vv
      common/cvvs/vvs
      integer jproc
      common/cjproc/jproc
      integer ipdfscale
      common/cipdfscale/ipdfscale
      integer idec
      common/cidec/idec
      character * 2 prc,prdct
      common/process/prc,prdct
      real * 8 sf(2,2,3,10),www(2,3)
      real * 8 x,y,cth1,th2,cth2,sx,ro,b,rox,bx,rolimx,blimx,btildex,
     #  rotildx,xktrel,csi,rx,x1,x2,xnorm,xnormc,xnormsv,xnormborn,
     #  xlgomx,ffunval5,ffunction5,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,
     #  v3c,v13,zg2,zgmu2_vb,xfact,ytmp,zhwfct,zherw_spl,x1t,x1soft,
     #  x2t,x2soft,x1x2j,x1x2jac,ww3up,ww3do,zherw_smn,zgmu2_nvb,
     #  xlmude,bdelta,wwpup,wwpdo,wwlup,wwldo,xtmp,brnup,brndo,yfac,
     #  svn,ww2up,ww2do,f1fact,f1fun,f1up,f1do,xsum,dummy,xint
      integer i,j,itype,iret
c
      x      = xx
      y      = xy
      cth1   = xcth1
      th2    = xth2
      cth2   = cos(th2)
c
      sx      = s*x
      ro      = 2*(zmw2+zmz2)/s-(zmw2-zmz2)**2/s**2
      b       = sqrt(1-ro)
      rox     = 2*(zmw2+zmz2)/(sx)-(zmw2-zmz2)**2/(sx)**2
      bx      = sqrt(1-rox)
      rolimx  = (zmw+zmz)**2/sx
      blimx   = sqrt(1-rolimx)
      btildex = blimx*betfac
      rotildx = 1-btildex**2
      xktrel = (1-x)**2*(1-y**2)
c
      csi = sqrt((1-(1-x)*(1+y)/2)/(1-(1-x)*(1-y)/2))
      rx = sqrt(x)
c
      x1 = sqrt(tau) * exp(ycm)
      x2 = tau/x1
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-'.or.prdct.eq.'z ')then
        xnorm = ze2**2 * xjac / s/(8*pi)**2 / (16*pi2)
        xnormc = ze2**2/(16*pi2) * xjac / (16*pi)
      elseif(prdct.eq.'ww')then
        xnorm = ze2**2 * xjac/( (4*pi*s) * (4*pi)**2 * (16*pi) )
        xnormc = ze2**2 * xjac/(8*pi2*16*pi)
      else
        write(*,*) 'tvb5b: non implemented final state', prdct
        stop
      endif
      xnormsv = ze2**2 * xjac/(16*pi2)/(16*pi)
      xnormborn = ze2**2 * xjac/(16*pi)
c
      xlgomx = log(1-x)
      do i=1,2
        do j=1,2
          do itype=1,10
            vv(i,j,jproc,itype)=0.d0
            vvs(i,j,jproc,itype)=0.d0
          enddo
        enddo
      enddo
c
c MC subtraction terms; ffunval5=0,1 --> dead zone, live zone. Notice
c that these contribute only if x1<1, x2<1 as for the event
c 
      ffunval5 = ffunction5(x,y)
      if(x1.lt.1.and.x2.lt.1.and.ffunval5.ne.0.d0)then
        call invvb(zmw2,zmz2,s,x,y,cth1,cth2,zero,zero,zero,zero,
     #        str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
        zg2 = zgmu2_vb()
        ipdfscale=2
        xfact = xnorm*zg2*bx/(1-x)*( 1/(1-y) + 1/(1+y) )
        if(prc.eq.'qq') then
          ytmp = 1.d0
          zhwfct=zherw_spl(x,y)
          x1t = x1soft(x1,x2,x,y)/zhwfct
          x2t = x2soft(x1,x2,x,y)
          if(x1t.lt.1.and.x2t.lt.1)then
            call strfvb(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,y)/zhwfct
            call xmcsuvb(s,x,y,cth1,cth2,x1,x2,ytmp,ww3up,ww3do)
            www(1,jproc) = xfact * x1x2j * ww3up 
            www(2,jproc) = xfact * x1x2j * ww3do 
            do i=1,2
              do j=1,2
                do itype=1,10
                  vv(i,j,jproc,itype)=vv(i,j,jproc,itype)+
     #                           sf(i,j,jproc,itype)*www(i,jproc)
                enddo
              enddo
            enddo
          endif
        endif
c
        ytmp = -1.d0
        zhwfct=zherw_smn(x,y)
        x1t = x1soft(x1,x2,x,y)
        x2t = x2soft(x1,x2,x,y)/zhwfct
        if(x1t.lt.1.and.x2t.lt.1)then
          call strfvb(x1t,x2t,sf)
          x1x2j = x1x2jac(x1,x2,x,y)/zhwfct
          call xmcsuvb(s,x,y,cth1,cth2,x1,x2,ytmp,ww3up,ww3do)
          www(1,jproc) = xfact * x1x2j * ww3up 
          www(2,jproc) = xfact * x1x2j * ww3do 
          do i=1,2
            do j=1,2
              do itype=1,10
                vv(i,j,jproc,itype)=vv(i,j,jproc,itype)+
     #                         sf(i,j,jproc,itype)*www(i,jproc)
              enddo
            enddo
          enddo
        endif
      endif
c
c All counterevents must have ktrel<zeta
c
      if(xktrel.lt.etacut)then
        ipdfscale=1
c
c Counter-event (x,y)=(x,1)
c
        if(prc.eq.'qq'.and.y.gt.1-delta) then
          ytmp = 1.d0
          www(1,jproc) = 0.d0
          www(2,jproc) = 0.d0
          x1t = x1soft(x1,x2,x,y)/x
          x2t = x2soft(x1,x2,x,y)
          if(x1t.lt.1.and.x2t.lt.1)then
            call invvb(zmw2,zmz2,s,x,ytmp,cth1,cth2,zero,zero,zero,
     #        zero,str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
            zg2 = zgmu2_nvb()
            call strfvb(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,y)/x
            xfact = xnorm * x1x2j * zg2 * bx/(1-x)*( - 1/(1-y) )
            call vbfpp(s,x,ytmp,q1q,q2q,ww3up,ww3do)
            www(1,jproc) = www(1,jproc) + xfact * ww3up 
            www(2,jproc) = www(2,jproc) + xfact * ww3do 
c Adding the collinear contribution
            xlmude = log(s/xmufct2)+log(delta/2)+
     #               log( (1-bdelta(x))/delta )
            xfact = xnormc * x1x2j * zg2 * bx / (pi*(1-bdelta(x)))
            call vbppcolp(ytmp,s,q2q,x,xlmude,wwpup,wwpdo)
            call vbppcoll(ytmp,s,q2q,x,wwlup,wwldo)
            www(1,jproc) = www(1,jproc) + xfact *
     #          ( 1/(1-x)*wwpup + xlgomx/(1-x)*wwlup )
            www(2,jproc) = www(2,jproc) + xfact *
     #          ( 1/(1-x)*wwpdo + xlgomx/(1-x)*wwldo )
c end collinear contribution
            do i=1,2
              do j=1,2
                do itype=1,10
                  vv(i,j,jproc,itype)=vv(i,j,jproc,itype)+
     #                           sf(i,j,jproc,itype)*www(i,jproc)
                enddo
              enddo
            enddo
          endif
        endif
c
c Counter-event (x,y)=(x,-1)
c
        if(y.lt.-1+delta) then
          ytmp = -1.d0
          www(1,jproc) = 0.d0
          www(2,jproc) = 0.d0
          x1t = x1soft(x1,x2,x,y)
          x2t = x2soft(x1,x2,x,y)/x
          if(x1t.lt.1.and.x2t.lt.1)then
            call invvb(zmw2,zmz2,s,x,ytmp,cth1,cth2,zero,zero,zero,
     #        zero,str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
            zg2 = zgmu2_nvb()
            call strfvb(x1t,x2t,sf)
            x1x2j = x1x2jac(x1,x2,x,y)/x
            xfact = xnorm * x1x2j * zg2 * bx/(1-x)*( - 1/(1+y) )
            call vbfpp(s,x,ytmp,q1q,q2q,ww3up,ww3do)
            www(1,jproc) = www(1,jproc) + xfact * ww3up 
            www(2,jproc) = www(2,jproc) + xfact * ww3do 
c Adding the collinear contribution
            xlmude = log(s/xmufct2)+log(delta/2)+
     #               log( (1-bdelta(x))/delta )
            xfact = xnormc * x1x2j * zg2 * bx / (pi*(1-bdelta(x)))
            call vbppcolp(ytmp,s,q1q,x,xlmude,wwpup,wwpdo)
            call vbppcoll(ytmp,s,q1q,x,wwlup,wwldo)
            www(1,jproc) = www(1,jproc) + xfact *
     #        ( 1/(1-x)*wwpup + xlgomx/(1-x)*wwlup )
            www(2,jproc) = www(2,jproc) + xfact *
     #        ( 1/(1-x)*wwpdo + xlgomx/(1-x)*wwldo )
c end collinear contribution
            do i=1,2
              do j=1,2
                do itype=1,10
                  vv(i,j,jproc,itype)=vv(i,j,jproc,itype)+
     #                           sf(i,j,jproc,itype)*www(i,jproc)
                enddo
              enddo
            enddo
          endif
        endif
c
c The invariants become independent upon y in the limit x --> 1,
c and therefore there's no need of inserting counterterms of
c the type (1,1) or (1,-1)
c
c
c Counter-event (x,y)=(1,y)
c
        xtmp = 1.d0
        x1t = x1soft(x1,x2,x,y)
        x2t = x2soft(x1,x2,x,y)
        if( prc.eq.'qq' .and. x.gt.rotildx )then
          www(1,jproc) = 0.d0
          www(2,jproc) = 0.d0
          if(x1t.lt.1.and.x2t.lt.1)then
            yfac = 0
            if( y .gt. -1+delta ) yfac = yfac - 1/(1+y)
            if( y .lt.  1-delta ) yfac = yfac - 1/(1-y)
            call invvb(zmw2,zmz2,sx,xtmp,y,cth1,cth2,zero,zero,zero,
     #        zero,str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
            zg2 = zgmu2_nvb()
            call strfvb(x1t,x2t,sf)
            if(xktrel.lt.etacut)then
              x1x2j = x1x2jac(x1,x2,x,y)
              xfact = xnorm * zg2 * bx * (x1x2j/x) /(1-x) * yfac
              call vbfpp(sx,xtmp,y,q1q,q2q,ww3up,ww3do)
              www(1,jproc) = xfact * ww3up 
              www(2,jproc) = xfact * ww3do 
c Adding the Born term
              xfact = xnormborn * bx * x1x2j / 
     #                (2*pi*(1-rotildx+svn(rotildx)))
              call vbborn(sx,q1q,zmw2,zmz2,brnup,brndo)
              www(1,jproc) = www(1,jproc) + xfact * brnup
              www(2,jproc) = www(2,jproc) + xfact * brndo
c Adding the soft-virtual contribution
              xfact = xnormsv * zg2 * bx * x1x2j / 
     #                (2*pi*(1-rotildx+svn(rotildx)))
              call vb2(sx,q1q,zmw2,zmz2,ww2up,ww2do)
              f1fact = 16*f1fun(rotildx)*4.d0/3.d0 
              f1up = f1fact*brnup
              f1do = f1fact*brndo
              www(1,jproc) = www(1,jproc) + 
     #                       xfact * (ww2up + f1up)
              www(2,jproc) = www(2,jproc) + 
     #                       xfact * (ww2do + f1do)
c Adding the collinear contributions
              if(y.gt.1-delta) then
                ytmp = 1.d0
                xlmude = log(sx/xmufct2)+log(delta/2)+
     #                   log( (1-bdelta(x))/delta )
                xfact = - xnormc * zg2 * bx * x1x2j / (pi*(1-bdelta(x)))
                call vbppcolp(ytmp,sx,q2q,xtmp,xlmude,wwpup,wwpdo)
                call vbppcoll(ytmp,sx,q2q,xtmp,wwlup,wwldo)
                www(1,jproc) = www(1,jproc) + xfact *
     #               ( 1/(1-x)*wwpup + xlgomx/(1-x)*wwlup )
                www(2,jproc) = www(2,jproc) + xfact *
     #               ( 1/(1-x)*wwpdo + xlgomx/(1-x)*wwldo )
              endif
              if(y.lt.-1+delta) then
                ytmp = -1.d0
                xlmude = log(sx/xmufct2)+log(delta/2)+
     #                   log( (1-bdelta(x))/delta )
                xfact = - xnormc * zg2 * bx * x1x2j / (pi*(1-bdelta(x)))
                call vbppcolp(ytmp,sx,q1q,xtmp,xlmude,wwpup,wwpdo)
                call vbppcoll(ytmp,sx,q1q,xtmp,wwlup,wwldo)
                www(1,jproc) = www(1,jproc) + xfact *
     #               ( 1/(1-x)*wwpup + xlgomx/(1-x)*wwlup )
                www(2,jproc) = www(2,jproc) + xfact *
     #               ( 1/(1-x)*wwpdo + xlgomx/(1-x)*wwldo )
              endif
            endif
            do i=1,2
              do j=1,2
                do itype=1,10
                  vv(i,j,jproc,itype)=vv(i,j,jproc,itype)+
     #                           sf(i,j,jproc,itype)*www(i,jproc)
                enddo
              enddo
            enddo
          endif
        endif
      endif
c
      call checkvb(xsum,dummy,iret)
      if(iret.eq.1)then
        xtmp = 1.d0
        ytmp = 1.d0
        call invvb(zmw2,zmz2,sx,xtmp,ytmp,cth1,cth2,zero,zero,zero,
     #       zero,str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
        x1t = x1soft(x1,x2,x,y)
        x2t = x2soft(x1,x2,x,y)
        ycm = 0.5d0*log(x1t/x2t)
        tau = x * tau
        if(idec.eq.0)then
          ps=sx
          px=xtmp
          py=ytmp
          pcth1=cth1
          pcth2=cth2
        endif
c Cross section in pb (momenta are in TeV in the computation)
        xint = hc2*xsum
        xevsign = 1.d0
        if(xint.lt.0.d0)xevsign = -1.d0
      else
        xint = 0.d0
        xevsign = 1.d0
      endif
c
      bsfsgn = xevsign
      tvb5b = abs(xint)
c
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
C      implicit real * 8 (a-z)
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
C      implicit real * 8 (a-z)
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
      subroutine sprfvb()
c This routine is called by run_spring; the entry is dummy, all the 
c parameters must be passed through common blocks
      implicit none
      integer iunit
      parameter (iunit=22)
      real*8 xone
      parameter (xone=1.d0)
      real*8 ycm,tau
      common/x1x2/ycm,tau
      integer i0,j0,jproc0,itype0
      common/cidproc/i0,j0,jproc0,itype0
      integer idec
      common/cidec/idec
      integer iret
      real*8 ycm0
c
      call xovb(iret)
      if(iret.eq.1)then
        if(j0.eq.1)then
          call labmom(ycm)
          ycm0=ycm
        elseif(j0.eq.2)then
          call reflex(ycm)
          ycm0=-ycm
        else
          write(*,*)'Fatal error in sprfvb'
          stop
        endif
        call getx1x2(tau,ycm0)
        if(idec.eq.0)call getspincorr(ycm0,j0)
        call getmvb(tau,ycm0,j0)
        call store_evenvb(iunit,xone)
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


      subroutine getmvb(tau,ycm,j0)
      implicit none
      real * 8 tau,ycm,pi
      integer j0
      parameter (pi=3.14159265358979312D0)
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      real * 8 x1,x2
      common/cx1x2/x1,x2
      real * 8 xmom_cm(9,4)
      common/cxmomcm/xmom_cm
      real * 8 xmom_lb(9,4)
      common/cxmomlb/xmom_lb
      real * 8 xmom_prime(9,4)
      common/cxmomprime/xmom_prime
      real * 8 pw(2),pz(2),pp(2)
      common/plbvar/pw,pz,pp
      integer ionshell
      common/cionshell/ionshell
      integer ichkmom
      common/cichkmom/ichkmom
      integer idec
      common/cidec/idec
      integer ifk88seed
      common/cifk88seed/ifk88seed
      real * 8 xsign,theta,fk88random,cth,sth,sqsh,ycmnew
      integer i,j,imax
c
      if(j0.eq.1)then
        xsign=1.d0
      elseif(j0.eq.2)then
        xsign=-1.d0
      else
        write(*,*)'Fatal error in getmvb'
        stop
      endif
      imax=5
      if(idec.eq.0)imax=9
      do i=3,imax
        do j=1,3
          xmom_cm(i,j)=xsign*xmom_cm(i,j)
        enddo
      enddo
c perform a random rotation in the transverse plane
      theta=2*pi*fk88random(ifk88seed)
      cth=cos(theta)
      sth=sin(theta)
      call transrot(cth,sth,pw(1),pw(2))
      call transrot(cth,sth,pz(1),pz(2))
      call transrot(cth,sth,pp(1),pp(2))
      do i=3,imax
        call transrot(cth,sth,xmom_cm(i,1),xmom_cm(i,2))
      enddo
      if(ichkmom.eq.0)call checkmvb(xmom_cm,sh,0.d0,3,2)
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
        do i=3,imax
          call boost(-ycm,
     #         xmom_cm(i,1),xmom_cm(i,2),
     #         xmom_cm(i,3),xmom_cm(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
        if(ichkmom.eq.0)call checkmvb2(xmom_lb,ycm)
      else
c put the partons on Herwig mass shell
        call put_on_shevb(ycm,ycmnew)
        do i=1,imax
          call boost(-ycmnew,
     #         xmom_prime(i,1),xmom_prime(i,2),
     #         xmom_prime(i,3),xmom_prime(i,4),
     #         xmom_lb(i,1),xmom_lb(i,2),xmom_lb(i,3),xmom_lb(i,4))
        enddo
      endif
c      if(ichkmom.eq.0)call checkmvb(xmom_lb,sh,-ycm,2,2)
      call momnewformvb()
      return
      end


      subroutine momnewformvb()
c Replaces the energy with the mass in the fourth component of xmom_lb,
c to comply with the new format of the event file. Must be called as the
c last step before storing events on the temporary event files.
c If the energy is zero, the fourth component is left unchanged,
c since the LH interface uses it to distinguish between S and H events.
      implicit none
      real*8 xmom_lb(9,4)
      common/cxmomlb/xmom_lb
      real*8 xmss(1:9)
      common/procmass/xmss
      integer i
c
      do i=1,9
        if(xmom_lb(i,4).ne.0.d0)xmom_lb(i,4)=xmss(i)
      enddo
      return
      end


C      subroutine boost(y,a1,a2,a3,a4,b1,b2,b3,b4)
C      implicit none
C      real * 8 y,a1,a2,a3,a4,b1,b2,b3,b4
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


      subroutine put_on_shevb(ycm,ycmnew)
      implicit none
      integer i2b,i,j,iv,il
      real*8 xmss(1:9),xtmp(1:4),xk1tmp(1:4),ytmp1(1:4),ytmp2(1:4),
     #  xavg3(1:3)
      real*8 ycm,ycmnew,pi,one,delta_thrs,shat,xkp2prime_norm2,
     #  xkp2prime_norm,xkprime_0,xsign,xnorm_3,delta,gamma,xmprime,
     #  xk1prime_norm,fakemass,xk1tmp_norm,xkprime_norm,zero,
     #  xlepnorm,tmplmass,xavgnorm
      parameter (pi=3.14159265358979312D0)
      parameter (zero=0.d0)
      parameter (one=1.d0)
      parameter (delta_thrs=0.5d-3)
      real * 8 q12,q22
      common/cvirt/q12,q22
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      common/procmass/xmss
      real*8 xmass(-5:21)
      common/parmass/xmass
c Lepton masses
      real * 8 xlep1mass(2),xlep2mass(2)
      common/clepmass/xlep1mass,xlep2mass
c x1 and x2 are the Bjorken variables; x1 is relevant to the parton
c coming from the left
      real*8 x1,x2
      common/cx1x2/x1,x2
c xmom_cm(i,j) is the j component of the four vector of the particle # i,
c given in the partonic CM frame. j=4 is the energy. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is V1, 5 is V2, 6 and 7 are the leptons
c originating from the decay of V1, 8 and 9 those originating from the 
c decay of V2. See xmadevww() for the fermion/antifermion assignments
c in the case of WW production. Momentum conservation is 
c (1+2)-(3+4+5)=0 or (1+2)-(3+6+7+8+9)=0 
      real*8 xmom_cm(9,4)
      common/cxmomcm/xmom_cm
c new momenta (put on shell) are stored here
      real*8 xmom_prime(9,4)
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
      integer idec
      common/cidec/idec
c 
      xmss(1) = xmass(ip1)
      xmss(2) = xmass(ip2)
      xmss(3) = xmass(ip3)
      if(idec.eq.0)then
        xmss(4) = sqrt(q12)
        xmss(5) = sqrt(q22)
        if(ilepmass.eq.0)then
          xmss(6) = 0.d0
          xmss(7) = 0.d0
          xmss(8) = 0.d0
          xmss(9) = 0.d0
        elseif(ilepmass.eq.2)then
          xmss(6) = xlep1mass(1)
          xmss(7) = xlep2mass(1)
          xmss(8) = xlep1mass(2)
          xmss(9) = xlep2mass(2)
        else
          write(*,*)'Error in put_on_shevb: unknown ilepmass',ilepmass
          stop
        endif
      elseif(idec.eq.1)then
        xmss(4) = zmw
        xmss(5) = zmz
        xmss(6) = -1.d10
        xmss(7) = -1.d10
        xmss(8) = -1.d10
        xmss(9) = -1.d10
      else
        write(6,*) 'Error in put_on_shevb: idec=',idec
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
        call getxmvb(shat,ycm,
     #               xmom_cm(1,3),xmss(1),
     #               xmom_cm(2,3),xmss(2),
     #               xmom_prime(1,3),xmom_prime(2,3))
      else
        write(*,*)'Error in put_on_shevb: unknown ionshell'
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
c of the masses of the V and H
      delta=sqrt(xkprime_0**2-xkprime_norm**2)-xmss(4)-xmss(5)
      if(delta.lt.delta_thrs)then
c parton 3-momenta cannot be kept fixed: the total available energy
c is not sufficient; modify 3-momenta of the incoming partons
        gamma=sqrt( (xmss(4)+xmss(5)+delta_thrs)**2+xkprime_norm**2 )+
     #        xmom_prime(3,4)
        if(gamma.lt.(xmss(1)+xmss(2)))then
          write(6,*)'Error #0 in put_on_shevb'
          write(6,*)gamma,xmom_prime(3,4)
          stop
        endif
        xkp2prime_norm2=( gamma**2-2*(xmss(1)**2+xmss(2)**2)+
     #                    (xmss(1)**2-xmss(2)**2)**2/gamma**2 )/4.d0
        xkp2prime_norm=sqrt(xkp2prime_norm2)
        xmom_prime(1,3)=sign(1.d0,xmom_cm(1,3))*xkp2prime_norm
        xmom_prime(1,4)=gamma/2.d0*(1+(xmss(1)**2-xmss(2)**2)/gamma**2)
        xmom_prime(2,3)=sign(1.d0,xmom_cm(2,3))*xkp2prime_norm
        xmom_prime(2,4)=gamma/2.d0*(1-(xmss(1)**2-xmss(2)**2)/gamma**2)
        xkprime_0=xmom_prime(1,4)+xmom_prime(2,4)-xmom_prime(3,4)
        shat=(xmom_prime(1,4)+xmom_prime(2,4))**2 -
     #       (xmom_prime(1,3)+xmom_prime(2,3))**2
      endif
c now the parton 3-momenta have been defined in such a way
c that the momenta of V1 and V2 can be transformed.
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
          write(6,*)'Error #1 in put_on_shevb'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        xk1tmp_norm=xnorm_3(xk1tmp)
c xavg is the direction along which the V1 and V2 momenta are placed
c in the new VV rest frame. It is arbitrarily defined by averaging 
c (hence the 1/2 in the definition) the directions of the original 
c V1 and V2 momenta. It may not have modulus 1, so normalize it
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
        xk1tmp(4)=xmprime/2.d0*
     #            (1+xsign*(xmss(4)**2-xmss(5)**2)/xmprime**2)
        call xhwulob(xtmp,xmprime,
     #               xk1tmp,xmss(i),
     #               ytmp2,fakemass)
        if(abs(fakemass-xmss(i)).gt.1.d-4)then
          write(6,*)'Error #2 in put_on_shevb'
          write(6,*)i,xmss(i),fakemass
          stop
        endif
        call getvec(ytmp2,xmom_prime(i,1),xmom_prime(i,2),
     #                    xmom_prime(i,3),xmom_prime(i,4))
      enddo
      if(idec.eq.0)then
c repeat what done for the VV, boosting the lepton momenta in
c the new V's rest frames
        do iv=4,5
          il=iv-3
          call fillvec(xmom_prime(iv,1),xmom_prime(iv,2),
     #                 xmom_prime(iv,3),xmom_prime(iv,4),xtmp)
          xlepnorm=xmss(iv)**2-2*(xlep1mass(il)**2+xlep2mass(il)**2)+
     #             (xlep1mass(il)**2-xlep2mass(il)**2)**2/xmss(iv)**2
          xlepnorm=sqrt(xlepnorm)/2.d0
          do j=1,3
            xavg3(j)=0.d0
          enddo
          do i=6+2*(il-1),7+2*(il-1)
            if(i.eq.6.or.i.eq.8)then
              xsign=1.d0
              tmplmass=xlep1mass(il)
            else
              xsign=-1.d0
              tmplmass=xlep2mass(il)
            endif
            call fillvec(xmom_cm(i,1),xmom_cm(i,2),
     #                   xmom_cm(i,3),xmom_cm(i,4),ytmp1)
            call xhwulof(xtmp,xmss(iv),
     #                   ytmp1,tmplmass,
     #                   xk1tmp,fakemass)
            xk1tmp_norm=xnorm_3(xk1tmp)
            do j=1,3
              xavg3(j)=xavg3(j)+xsign*xk1tmp(j)/(2*xk1tmp_norm)
            enddo
          enddo
          xavgnorm=sqrt(xavg3(1)**2+xavg3(2)**2+xavg3(3)**2)
          do j=1,3
            xavg3(j)=xavg3(j)/xavgnorm
          enddo
          do i=6+2*(il-1),7+2*(il-1)
            if(i.eq.6.or.i.eq.8)then
              xsign=1.d0
              tmplmass=xlep1mass(il)
            else
              xsign=-1.d0
              tmplmass=xlep2mass(il)
            endif
            do j=1,3
              xk1tmp(j)=xsign*xlepnorm*xavg3(j)
            enddo
            xk1tmp(4)=xmss(iv)/2.d0*
     #        (1+xsign*(xlep1mass(il)**2-xlep2mass(il)**2)/xmss(iv)**2)
            call xhwulob(xtmp,xmss(iv),
     #                   xk1tmp,tmplmass,
     #                   ytmp2,fakemass)
            call getvec(ytmp2,xmom_prime(i,1),xmom_prime(i,2),
     #                        xmom_prime(i,3),xmom_prime(i,4))
          enddo
        enddo
      else
        do i=6,9
          do j=1,4
            xmom_prime(i,j)=0.d0
          enddo
        enddo
      endif
      if(ichkmom.eq.0)then
        call checkmvb(xmom_prime,shat,0.d0,4,2)
        if(idec.eq.0)then
          call checkdvb(xmom_prime,4,6,7)
          call checkdvb(xmom_prime,5,8,9)
          call checkmvb(xmom_prime,shat,0.d0,4,1)
        endif
        if(xmass(1).eq.0.and.xmass(2).eq.0.and.xmass(3).eq.0.and.
     #     xmass(4).eq.0.and.xmass(5).eq.0.and.xmass(21).eq.0.and.
     #     xlep1mass(1).eq.0.and.xlep2mass(1).eq.0.and.
     #     xlep1mass(2).eq.0.and.xlep2mass(2).eq.0)then
          call checkonvb(1)
        else
          call checkonvb(2)
        endif
      endif
      return
      end


      subroutine getxmvb(shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23)
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
        write(*,*)'Error # 0 in getxmvb'
        stop
      endif
      p13=xkp2prime_norm
      p23=-xkp2prime_norm
      e1=sqrt(shat)/2.d0*(1+(xm1**2-xm2**2)/shat)
      e2=sqrt(shat)/2.d0*(1-(xm1**2-xm2**2)/shat)
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
        write(*,*)'Error # 1 in getxmvb'
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
            write(*,*)'Mass # 1 too large in getxmvb'
            stop
          endif
        elseif(ileg.eq.2)then
          xm2=fact*sqrt(sol*shat)
          if(xm2.gt.xm2s)then
            write(*,*)'Mass # 2 too large in getxmvb'
            stop
          endif
        else
          write(*,*)'Error # 2 in getxmvb'
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
C      implicit real*8(a-h,o-z)
Cc
C      en=sqrt(p1**2+p2**2+p3**2+xm**2)
C      return
C      end


C      function xnorm_3(p)
Cc Evaluates the norm of the spatial component of a four-momentum
Cc The result is positive by definition, regardless of the 4-metric
C      implicit real*8(a-h,o-z)
C      dimension p(1:4)
Cc
C      tmp=p(1)*p(1)+p(2)*p(2)+p(3)*p(3)
C      xnorm_3=sqrt(tmp)
C      return
C      end


C      subroutine vecsum(p,pfact,q,qfact,r)
Cc Weighted sum of the four-vectors p and q. The result is r
C      implicit real*8(a-h,o-z)
C      dimension p(1:4),q(1:4),r(1:4)
Cc
C      do i=1,4
C        r(i)=pfact*p(i)+qfact*q(i)
C      enddo
C      return
C      end


      subroutine checkonvb(itype)
c Checks that put_on_shevb is harmless if masses are zero (itype=1),
c or computes (itype=2) the average of the shifts due to the masses
      real*8 tiny
      parameter (tiny=1.d-4)
      integer itype
      real*8 xmom_cm(9,4)
      common/cxmomcm/xmom_cm
      real*8 xmom_prime(9,4)
      common/cxmomprime/xmom_prime
      real*8 xmomshifts(4)
      common/cshifts/xmomshifts
      integer i,j,imax,iflag
      integer idec
      common/cidec/idec
c
      if(itype.ne.1.and.itype.ne.2)then
        write(*,*)'Unknown option in checkonvb'
        stop
      endif
      iflag=1
      imax=5
      if(idec.eq.0)imax=9
      do i=1,imax
        do j=1,4
          if(abs(xmom_cm(i,j)).lt.1.d0)then
            xtmp=xmom_cm(i,j)-xmom_prime(i,j)
          else
            xtmp=(xmom_cm(i,j)-xmom_prime(i,j))/xmom_cm(i,j)
          endif
          if(abs(xtmp).gt.tiny)iflag=0
          xmomshifts(j)=xmomshifts(j)+abs(xtmp)
        enddo
      enddo
      if(iflag.eq.0.and.itype.eq.1)then
        write(*,*)'Error in checkonvb'
        write(*,*)'  '
        write(*,*)'xmom_cm:'
        do i=1,imax
          write(*,'(4(d14.8,1x))') (xmom_cm(i,j),j=1,4)
        enddo
        write(*,*)'  '
        write(*,*)'xmom_prime:'
        do i=1,imax
          write(*,'(4(d14.8,1x))') (xmom_prime(i,j),j=1,4)
        enddo
        stop
      endif
      return
      end


      subroutine xovb(iret)
c This routine is called by sprfvb; it determines, on statistical
c basis, which partonic process has been generated (jproc is kept fixed).
c It also counts the number of unlike sign events (iwrong), and the number
c of these events (iwrong1) for which the relative difference between
c unlike signs exceeds 5%. If all the entries of vv are equal to zero,
c iret is set equal to 0 (by checkvb), and no operation is performed
      implicit real * 8(a-h,o-z)
      common/cjproc/jproc
      common/cifuntype/ifuntype
      common/cvv/vv(2,2,3,10)
      common/cvvs/vvs(2,2,3,10)
      common/ciwrong/iwrong,iwrong1
      common/cidproc/i0,j0,jproc0,itype0
      common/civbhpro/ivbhpro(2,2,3,10)
      common/cidpart/idp1(2,2,3,10),idp2(2,2,3,10),idp3(2,2,3,10)
      common/ci1hpro/i1hpro
      common/ci1part/ip1,ip2,ip3
      integer ifk88seed
      common/cifk88seed/ifk88seed
      dimension wwx(2,2,3,10)
c
      i0=0
      j0=0
      itype0=0
      iret=0
      call checkvb(xsum,xsumabs,iretvv)
      call checkvbs(xsumvvs,xsumabsvvs,iretvvs)
      if(iretvv.eq.0.and.iretvvs.eq.1)then
        write(6,*)'Fatal error in xovb:',iretvv,iretvvs
        stop
      endif
      if(iretvv.eq.1)then
        iret=iretvv
        if(iretvvs.eq.1)then
          xsum=xsumvvs
          xsumabs=xsumabsvvs
          do i=1,2
            do j=1,2
              do itype=1,10
                wwx(i,j,jproc,itype)=vvs(i,j,jproc,itype)
              enddo
            enddo
          enddo
        else
          do i=1,2
            do j=1,2
              do itype=1,10
                wwx(i,j,jproc,itype)=vv(i,j,jproc,itype)
              enddo
            enddo
          enddo
        endif
        if(ifuntype.eq.1)then
          jproc0=jproc
        elseif(ifuntype.eq.2)then
          if(iretvvs.eq.1)then
            jproc0=1
          else
            jproc0=jproc
          endif
        else
          write(*,*)'Fatal error in xovb: ifuntype=',ifuntype
          stop
        endif
        xstsign=sign(1.d0,xsum)
        xg=fk88random(ifk88seed)
        wh=0.d0
        iwh=0
        iflag=0
        rmax=0.d0
        do i=1,2
          do j=1,2
            do itype=1,10
              if(iwh.eq.0)then
                wh=wh+abs(wwx(i,j,jproc,itype))/xsumabs
                if(wh.gt.xg)then
                  i0=i
                  j0=j
                  itype0=itype
                  iwh=1
                endif
              endif
              if(wwx(i,j,jproc,itype).ne.0.d0)then
                if(xstsign.ne.sign(1.d0,wwx(i,j,jproc,itype)))then
                  if(iflag.eq.0)then
                    iwrong=iwrong+1
                    iflag=1
                  endif
                  rmax=max(rmax,abs(wwx(i,j,jproc,itype)))
                endif
              endif
            enddo
          enddo
        enddo
        if(iflag.eq.1)then
          if(xsum.ne.0.d0)rmax=rmax/xsum
          if(rmax.gt.0.05d0.or.xsum.eq.0.d0)iwrong1=iwrong1+1
        endif
        if(i0.eq.0.or.j0.eq.0.or.jproc0.eq.0.or.itype0.eq.0)then
          write(*,*)'Fatal error in xovb'
          stop
        endif
        ihpro=ivbhpro(i0,j0,jproc0,itype0)
        i1=idp1(i0,j0,jproc0,itype0)
        i2=idp2(i0,j0,jproc0,itype0)
        i3=idp3(i0,j0,jproc0,itype0)
        call parcrossivb(jproc0,ihpro,i1,i2,i3,i1hproo,ip1o,ip2o,ip3o)
        i1hpro=i1hproo
        ip1=ip1o
        ip2=ip2o
        ip3=ip3o
      endif
      return
      end


      subroutine parcrossivb(jproc0,ihpro,i1,i2,i3,
     #                       i1hproo,ip1o,ip2o,ip3o)
      implicit real * 8(a-h,o-z)
      parameter (iallzero=1)
      dimension ihprotrans(402:406)
      common/cifuntype/ifuntype
      data ihprotrans/401,0,403,403,401/
c
      if( (ifuntype.eq.1) .or. (ifuntype.eq.2.and.jproc0.eq.1) )then
        i1hproo=ihpro
        ip1o=i1
        ip2o=i2
        ip3o=i3
      elseif(ifuntype.eq.2.and.jproc0.ne.1)then
        if(ihpro.eq.401.or.ihpro.eq.403)then
          write(*,*)'error in parcrossivb:',ihpro,i1,i2,i3
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
        else
          write(*,*)'error in parcrossivb:',ihpro,i1,i2,i3
          stop
        endif
      else
        write(*,*)'parcrossivb:do not know what to do'
        write(*,*)ifuntype,jproc0
        stop
      endif
      call parcheckfvb(i1hproo,ip1o,ip2o,ip3o,iallzero)
      return
      end


      subroutine checkvb(xsum,xsumabs,iret)
c iret=0 -> all vv entries are equal to zero
c iret=1 -> there is at least one entry which is not zero
c xsum is the sum of all the entries of vv
c xsumabs is the sum of the absolute value of all the entries of vv
      implicit real * 8(a-h,o-z)
      common/cjproc/jproc
      common/cvv/vv(2,2,3,10)
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do i=1,2
        do j=1,2
          do itype=1,10
            if(vv(i,j,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vv(i,j,jproc,itype)
            xsumabs=xsumabs+abs(vv(i,j,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      subroutine checkvbs(xsum,xsumabs,iret)
c identical to checkvb, but works on vvs instead of vv
      implicit real * 8(a-h,o-z)
      common/cjproc/jproc
      common/cvvs/vvs(2,2,3,10)
c
      xsum=0.d0
      xsumabs=0.d0
      iret=0
      do i=1,2
        do j=1,2
          do itype=1,10
            if(vvs(i,j,jproc,itype).ne.0.d0)iret=1
            xsum=xsum+vvs(i,j,jproc,itype)
            xsumabs=xsumabs+abs(vvs(i,j,jproc,itype))
          enddo
        enddo
      enddo
      return
      end


      subroutine getspincorr(ycm,j0)
c Determines the lepton momenta, by performing an unweighting using
c the exact real and Born lepton matrix elements
      implicit none
      integer j0
      real * 8 ycm,pi,tolerance,vwl,awl,pdftiny
      parameter (pi=3.14159265358979312D0)
      parameter (tolerance=1.d-2)
      parameter (vwl=1.d0)
      parameter (awl=1.d0)
      parameter (pdftiny=1.d-8)
      character * 2 str
      parameter (str='p1')
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      real * 8 xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/weakcoup/xkm,xkm2,sw,ze2,gup,gdown,ez,gw
      real * 8 xmom_cm(9,4)
      common/cxmomcm/xmom_cm
      real * 8 ps,px,py,pcth1,pcth2
      common/cpsave/ps,px,py,pcth1,pcth2
      real * 8 q12,q22
      common/cvirt/q12,q22
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 ga1,ga1mn,ga1pl,bw1fmpl,bw1fmmn,bw1delf,xm1low2,xm1upp2
      common/bw1cmm/ga1,ga1mn,ga1pl,bw1fmpl,bw1fmmn,bw1delf,
     #              xm1low2,xm1upp2
      real * 8 ga2,ga2mn,ga2pl,bw2fmpl,bw2fmmn,bw2delf,xm2low2,xm2upp2
      common/bw2cmm/ga2,ga2mn,ga2pl,bw2fmpl,bw2fmmn,bw2delf,
     #              xm2low2,xm2upp2
      real * 8 al_gfun,be_gfun,ccc_gfun
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
      real * 8 x1,x2
      common/cx1x2/x1,x2
      integer jproc0
      common/cjproc/jproc0
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ifuntype
      common/cifuntype/ifuntype
      integer iwidth
      common/ciwidth/iwidth
      integer ichkmom
      common/cichkmom/ichkmom
      integer neventsuw,nqeventsuw,ifailuw
      common/c1iunwgt/neventsuw,nqeventsuw,ifailuw
      integer ncntuws,nqcntuws,nmaxuw,nqmaxuw
      common/c2iunwgt/ncntuws,nqcntuws,nmaxuw,nqmaxuw
      integer idec
      common/cidec/idec
      integer ifk88seed
      common/cifk88seed/ifk88seed
      character * 2 prc,prdct
      common/process/prc,prdct
      real * 8 rho,xtmp,prob,spcdamp,rrnd,fk88random,e,f,g,h,phil,
     # cthl,phir,cthr,o,p,xbwmass3_mod,psp,xp,xpsave,
     # wfact,fw,phspfact1,phspfact2,xbound,tk,uk,q1q,q2q,v1a,v1b,v1c,
     # v3a,v3b,v3c,v13,unxdec,xww,xdec,xmadevww,rat,staup,ycmp,x1p,x2p,
     # xlstaup,bwfunc_mod,xlumund,getlumspc,xlumdec
      integer iborn,iproj,jjprc,idr,iflav,ispcflav,icntuw,iqcntuw,
     # kp1,kp2,kp3,ifunsave,idum
c
      if(ichkmom.eq.0.and.iwidth.eq.0)call spccheck(1)
      rho=(zmw+zmz)**2/ps
      if(ifuntype.eq.2)then
        if(px.ne.1.d0.or.xmom_cm(3,4).ne.0.d0)then
          write(*,*)'Error #1 in getspincorr'
          stop
        else
          iborn=0
          iproj=0
          xtmp=px
        endif
      endif
      if(ifuntype.eq.1)then
        prob=spcdamp(px,py)
        rrnd=fk88random(ifk88seed)
        if(rrnd.ge.prob)then
c Close to the soft/collinear limits: use Born kinematics for the unweighting
          iborn=0
          iproj=1
          xtmp=1.d0
        else
c Away from the soft/collinear limit: use real kinematics for the unweighting
          iborn=1
          iproj=0
          xtmp=px
        endif
      endif
c When iproj=0, the Born and real kinematics are used to perform unweighting
c for S and H events respectively, and the parton identities are those passed
c by xovb (ip1, ip2, ip3). When iproj=1, the real kinematics is close to the 
c soft/collinear limits, and the Born is used to unweight; thus, it is 
c necessary to get the parton identities at the Born level
      if(iproj.eq.0)then
        kp1=ip1
        kp2=ip2
        kp3=ip3
      elseif(iproj.eq.1)then
        if(ifuntype.ne.1)then
          write(*,*)'Error #2 in getspincorr'
          stop
        else
c Get parton identities at the Born level; parcrossivb previously called
c by xovb was simply the identity, since ifuntype=1. It is not necessary 
c to change prc, since the function that returns the Born cross section
c doesn't have a check on whether prc='qq'
          ifunsave=ifuntype
          ifuntype=2
          call parcrossivb(jproc0,i1hpro,ip1,ip2,ip3,
     #                     idum,kp1,kp2,kp3)
          ifuntype=ifunsave
        endif
      else
        write(*,*)'Error #3 in getspincorr'
        stop
      endif
c Determines the nature of the process using kp1 and kp2, and set the
c value of idr, according to the conventions given in xww and xmadevww
      if(kp1.lt.21.and.kp2.lt.21)then
        jjprc=2
        idr=1
        if(kp1.lt.0)idr=3
        iflav=ispcflav(kp1)
      elseif(kp1.eq.21.or.kp2.eq.21)then
        jjprc=3
        if(kp1.lt.21)then
          iflav=ispcflav(kp1)
          idr=1
          if(kp1.lt.0)idr=2
        else
          iflav=ispcflav(kp2)
          idr=3
          if(kp2.lt.0)idr=4
        endif
      else
        write(*,*)'Error #4 in getspincorr',ip1,ip2,ip3,kp1,kp2,kp3
        stop
      endif
c When j0=2, getmvb will change signs to the 3-vectors of final state 
c particles for these processes (reflection of events). On the other hand,
c the functions xww and xmadevww cannot know about this reflection.
c Thus, to generate the correct distribution for the parton process with
c initial state ab, one unweights the process with initial state ba,
c which will be turned into ab by getmvb. A few sanity checks are 
c included in the following; they are specific to WW production
      if(j0.eq.2)then
        if(idr.le.2)then
          idr=idr+2
          if( (ifuntype.eq.1.and.iproj.eq.0) .or.
     #        (ifuntype.eq.2.and.jproc0.ne.3) )then
            write(*,*)'Inconsistent parameters in getspincorr',
     #        ifuntype,jproc0,idr,kp1,kp2,kp3
            stop
          endif
        elseif(idr.le.4)then
          idr=idr-2
          if( (ifuntype.eq.1.and.iproj.eq.1.and.jproc0.eq.3) .or.
     #        (ifuntype.eq.2.and.jproc0.eq.3) )then
            write(*,*)'Inconsistent parameters in getspincorr',
     #        ifuntype,jproc0,idr,kp1,kp2,kp3
            stop
          endif
        else
          write(*,*)'Unknown idr value in getspincorr',idr
          stop
        endif
      elseif(j0.ne.1)then
        write(*,*)'Unknown j0 value in getspincorr',j0
        stop
      endif
c
      neventsuw=neventsuw+1
      icntuw=0
 100  icntuw=icntuw+1
      e=fk88random(ifk88seed)
      f=fk88random(ifk88seed)
      g=fk88random(ifk88seed)
      h=fk88random(ifk88seed)
      phil=2*pi*e
      cthl=-1.d0+2*f
      phir=2*pi*g
      cthr=-1.d0+2*h
 300  continue
      if(iwidth.eq.1)then
        iqcntuw=0
 200    iqcntuw=iqcntuw+1
        o=fk88random(ifk88seed)
        p=fk88random(ifk88seed)
c Since the upper bound is q-dependent, distribute q according to the
c form of the bound, which is a skewed Breit Wigner
        q12=xbwmass3_mod(o,zmw2,ga1,ga1mn,ga1pl,bw1delf,bw1fmmn)
        q22=xbwmass3_mod(p,zmz2,ga2,ga2mn,ga2pl,bw2delf,bw2fmmn)
        nqcntuws=nqcntuws+iqcntuw
        if(iqcntuw.gt.nqmaxuw)nqmaxuw=iqcntuw
        nqeventsuw=nqeventsuw+1
        xp=xtmp
        xpsave=px
        psp=min( ((sqrt(q12)+sqrt(q22))/(zmw+zmz))**2*ps , sh )
        staup=sqrt(psp/sh)
        xlstaup=log(staup)
        ycmp=ycm
        if(ycmp.le.xlstaup)ycmp=0.999*xlstaup
        if(ycmp.ge.-xlstaup)ycmp=-0.999*xlstaup
        x1p=staup*exp(ycmp)
        x2p=staup/exp(ycmp)
      else
        q12=zmw2
        q22=zmz2
        xp=xtmp
        xpsave=px
        psp=ps
        x1p=-1.d8
        x2p=-1.d8
      endif
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-'.or.prdct.eq.'z ')then
        write(*,*)'Spin correlations not yet implemented'
        stop
      elseif(prdct.eq.'ww')then
        fw=gw
        wfact=2*zmw2*fw**2*(vwl+awl)**2
      else
        write(*,*)'getspincorr: no such process'
        stop
      endif
      if(iwidth.eq.0)then
        phspfact1=1.d0/(zmw2*ga1**2)
        phspfact2=1.d0/(zmz2*ga2**2)
      else
        phspfact1=pi/(zmw*ga1)*bwfunc_mod(q12,zmw2,ga1,ga1mn,ga1pl)
        phspfact2=pi/(zmz*ga2)*bwfunc_mod(q22,zmz2,ga2,ga2mn,ga2pl)
      endif
      xbound=wfact**2*phspfact1*phspfact2
      call invvb(zmw2,zmz2,ps,xtmp,py,pcth1,pcth2,phil,cthl,phir,cthr,
     #           str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,idec)
      unxdec=xww(iflav,iborn,jjprc,idr,
     #           zmw2,ps,xtmp,py,tk,uk,q1q,q2q)
      if(iwidth.eq.0)then
        xlumund=1.d0
      else
        xlumund=getlumspc(x1,x2,kp1,kp2,j0)
      endif
      call invvb(q12,q22,psp,xp,py,pcth1,pcth2,phil,cthl,phir,cthr,
     #           str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,idec)
      xdec=xmadevww(iflav,iborn,jjprc,idr,psp,tk,uk,xmom_cm)*
     #     psp/ps
      if(iwidth.eq.0)then
        xlumdec=1.d0
      else
        xlumdec=getlumspc(x1p,x2p,kp1,kp2,j0)
      endif
      if(xlumdec.gt.pdftiny.and.xlumund.gt.pdftiny)then
        rat=xdec*xlumdec/((1+tolerance)*unxdec*xlumund*xbound)
      else
        rat=xdec/((1+tolerance)*unxdec*xbound)
      endif
      if(rat.gt.1.d0)then
c The bound may fail if too a large mass range is selected; throw the
c q1,q2 values away, keeping the original kinematics, and restart
        ifailuw=ifailuw+1
        goto 300
      endif
      rrnd=fk88random(ifk88seed)
      if(rat.lt.rrnd)goto 100
      ncntuws=ncntuws+icntuw
      if(icntuw.gt.nmaxuw)nmaxuw=icntuw
c The event is accepted; regenerate kinematics if Born was used for 
c unweighting H events (to get xmom_cm corresponding to a real emission
c configuration), perform a sanity check otherwise
      if(iproj.eq.0)then
        if(xp.ne.xpsave)then
          write(*,*)'Error #5 in getspincorr'
          stop
        endif
      elseif(iproj.eq.1)then
        if(xp.ne.1.d0)then
          write(*,*)'Error #6 in getspincorr'
          stop
        endif
        call invvb(q12,q22,psp,xpsave,py,pcth1,pcth2,phil,cthl,phir,
     #        cthr,str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,idec)
      else
        write(*,*)'Error #7 in getspincorr'
        stop
      endif
      if(ichkmom.eq.0.and.iwidth.eq.0)call spccheck(2)
      return
      end


      function ispcflav(iid)
c Returns 1 (for up)   if id=2,4,6 (PDG codes for up quarks),
c         2 (for down) if id=1,3,5 (PDG codes for down quarks);
c to be used as a utility for getspincorr()
      implicit none
      integer ispcflav,iid,id
c
      id=abs(iid)
      if(id.eq.1.or.id.eq.3.or.id.eq.5)then
        ispcflav=2
      elseif(id.eq.2.or.id.eq.4.or.id.eq.6)then
        ispcflav=1
      else
        write(*,*)'Invalid parton id in ispcflav'
        stop
      endif
      return
      end


      function spcdamp(x,y)
c This function is defined in such a way that
c    spcdamp=0  if  tt=0
c  0<spcdamp<1  if  0<tt<1
c    spcdamp=1  if  tt>1
c and tt is a measure in the (x,y) plane, such that tt=0 in the soft
c and collinear limits (x=1, or y=1, or y=-1), growing monotonically
c away from these limits. In terms of invariants, tt=4*tk*uk/((1-xlim)*s)**2,
c which can easily be generalized for any kind of emissions. 
c Since when spcdamp=1 the real matrix elements are used in the 
c unweighting, xlim has been defined in such a way that, if be_spcfun=1,
c spcdamp is equal to 1 in a region similar to the dead zone. This is
c by no means necessary, and the dependence upon xlim in tt can be
c eliminated altogether. Call this function with al_spcfun>=1, 0<be_spcfun<=1
      implicit none
      real * 8 spcdamp,x,y,xmin,tt,xlim
      parameter (xmin=0.69519410160110384d0)
      real * 8 al_spcfun,be_spcfun
      common/cspcpar/al_spcfun,be_spcfun
c
      xlim=1.d0-be_spcfun+xmin*be_spcfun
      tt=(1-x)**2*(1-y**2)/(1-xlim)**2
      if(tt.lt.0.d0)then
        write(*,*)'Error in spcdamp',tt
        stop
      endif
      if(tt.gt.1.d0)tt=1.d0
      spcdamp=tt**(2*al_spcfun)/
     #       (tt**(2*al_spcfun)+(1-tt)**(2*al_spcfun))
      return
      end


      function getlumspc(x1,x2,kp1,kp2,j0)
      implicit none
      real * 8 getlumspc,x1,x2
      integer kp1,kp2,j0
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      integer ih1,ih2,ndns1,ndns2
      common/pdf/ih1,ih2,ndns1,ndns2
      character * 2 prc,prdct
      common/process/prc,prdct
      real * 8 zg2,zgmu2_nvb,tmp
      real * 4 smufct2
      real * 4 fh1x1(-5:5),fh2x2(-5:5)
      integer ip1,ip2
c Maps HERWIG conventions for parton identities to MLM
      integer imappdf(-6:6)
      data imappdf/-6,-5,-4,-3,-1,-2,0,2,1,3,4,5,6/
c
      if( x1.lt.0.d0.or.x1.gt.1.d0 .or.
     #    x2.lt.0.d0.or.x2.gt.1.d0 )then
        write(*,*)'Error in getlumspc',x1,x2
        stop
      endif
      zg2 = zgmu2_nvb()
      smufct2=sngl(xmufct2*1.d6)
      call mlmpdf(ndns1,ih1,smufct2,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smufct2,sngl(x2),fh2x2,5)
c See getspincorr for a comment concerning the treatment of reflected events
      if(j0.eq.1)then
        ip1=kp1
        ip2=kp2
      elseif(j0.eq.2)then
        ip1=kp2
        ip2=kp1
      else
        write(*,*)'getlumspc: wrong j0 value',j0
        stop
      endif
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-'.or.prdct.eq.'z ')then
        write(*,*)'Spin correlations not yet implemented'
        stop
      elseif(prdct.eq.'ww')then
        tmp=dble(fh1x1(imappdf(ip1))*fh2x2(imappdf(ip2)))
      else
        write(*,*)'getlumspc: no such process'
        stop
      endif
      getlumspc=tmp
      return
      end


      subroutine spccheck(iflag)
      implicit none
      real * 8 tiny,xmom_save(5,4)
      parameter (tiny=1.d-4)
      real * 8 xmom_cm(9,4)
      common/cxmomcm/xmom_cm
      integer iflag,i,j,itmp
      save xmom_save
c
      if(iflag.eq.1)then
        itmp=0
        do i=1,5
          do j=1,4
            xmom_save(i,j)=xmom_cm(i,j)
          enddo
        enddo
      elseif(iflag.eq.2)then
        itmp=0
        do i=1,5
          do j=1,4
            if(abs(xmom_save(i,j)-xmom_cm(i,j)).gt.tiny)itmp=1
          enddo
        enddo
      else
        write(*,*)'Wrong call to spccheck'
        stop
      endif
      if(itmp.eq.1)then
        write(*,*)'The check in spccheck failed'
        write(*,*)'Original momenta:'
        do i=1,5
          write(*,900)(xmom_save(i,j),j=1,4)
        enddo
        write(*,*)'  '
        write(*,*)'New momenta:'
        do i=1,5
          write(*,900)(xmom_cm(i,j),j=1,4)
        enddo
        stop
      endif
 900  format(4(1x,d14.8))
      return
      end


      function xww(iflav,iborn,jproc,idr,xmw2,s,x,y,tk,uk,q1q,q2q)
c Wrapper for the undecayed matrix elements of the original code.
c For Born matrix elements, q1q is t (consistently with the
c routine invar). The conventions used in this routine are different
c from those used in the rest of the code
c   iflav = 1(up), 2(down)
c   iborn = 0(born), 1(real)
c   jproc = 2(qq), 3(qg)
c   idr   = 1(qqbar,qg), 2(qbarg), 3(qbarq,gq), 4(gqbar)
      implicit none
      real*8 xww,xmw2,s,x,y,tk,uk,q1q,q2q
      integer iflav,iborn,jproc,idr
      real*8 s0,x0,y0,tk0,uk0,q1q0,q2q0,t0,tmpup,tmpdo
c
      if(iborn.eq.0.and.jproc.eq.3)then
        write(*,*)'Error #1 in xww: iborn,jproc=',iborn,jproc
        stop
      endif
      if(iborn.eq.1.and.jproc.eq.2.and.(idr.eq.2.or.idr.eq.4))then
        write(*,*)'Error #2 in xww: iborn,jproc,idr=',iborn,jproc,idr
        stop
      endif
      if(idr.le.2)then
        s0=s
        if(iborn.eq.0)then
          t0=q1q
        else
          x0=x
          y0=y
          tk0=tk
          uk0=uk
          q1q0=q1q
          q2q0=q2q
        endif
      else
c Reflected events: get qbarq, gq, and gqbar from qqbar, qg, and qbarg
c respectively, by exchanging p1 <--> p2
        s0=s
        if(iborn.eq.0)then
          t0=2*xmw2-s-q1q
        else
          x0=x
          y0=-y
          tk0=uk
          uk0=tk
          q1q0=2*xmw2-s-uk-q2q
          q2q0=2*xmw2-s-tk-q1q
        endif
      endif
      if(iborn.eq.0)then
        call wwborn(s0,t0,tmpup,tmpdo)
      else
        call wwfpp(s0,x0,y0,q1q0,q2q0,tmpup,tmpdo)
      endif
      if(iflav.eq.1)then
        xww=tmpup
      else
        xww=tmpdo
      endif
      return
      end


      function bqcorr(xm2,xm02)
c The virtuality of either W's as obtained from the leptonic matrix elements
c in WW production is bounded from above by the Breit-Wigner times bqcorr;
c this bound appears to work decently for M0-30*Ga<M<M0+30*Ga. This function
c is not used in the present version of getspincorr(), since it doesn't take
c into account the effect of the PDFs
      implicit none
      real*8 bqcorr,xm2,xm02,fdim,a,ap,am,sb,xm,xm0,del,aeff
c Set fdim=1.d0 if units are GeV, fdim=1.d3 if unites are TeV
      parameter (fdim=1.d3)
      parameter (a=0.03465d0*fdim)
      parameter (ap=a)
      parameter (am=a*0.8)
      parameter (sb=0.01d0*fdim)
c
      xm=sqrt(xm2)
      xm0=sqrt(xm02)
      del=xm-xm0
      if(del.ge.0.d0)then
        aeff=ap
      else
        aeff=am
      endif
      bqcorr=exp(aeff*del+sb**2*del**2/2.d0)
      return
      end
c
c
c End of event-generation routines
c
c
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


      subroutine strfvb(x1,x2,sf)
c Return parton densities through the matrix
c  sf(ifl,idr,jproc,itype), with the following conventions:
c  ifl=1 -> up quarks for WW, any flavours for WZ and ZZ
c  ifl=2 -> down quarks for WW, unused for WZ and ZZ
c  idr=1 -> direct events
c  idr=2 -> reflected events
c  jproc=1,2,3 -> qqbar, qg, ag processes respectively; 3 is unused for ZZ
c  itype -> identifies the individual contribution to a given jproc
      implicit real * 8 (a-h,o-z)
      character * 2 prc,prdct
      real * 4 smufct2
      real * 4 fh1x1(-5:5),fh2x2(-5:5)
      real * 4 fh1x2(-5:5),fh2x1(-5:5)
      real * 4 u11,  d11,  s11,  c11,  b11, g11
      real * 4 ub11, db11, sb11, cb11, bb11
      real * 4 u21,  d21,  s21,  c21,  b21, g21
      real * 4 ub21, db21, sb21, cb21, bb21
      real * 4 u12,  d12,  s12,  c12,  b12, g12
      real * 4 ub12, db12, sb12, cb12, bb12
      real * 4 u22,  d22,  s22,  c22,  b22, g22
      real * 4 ub22, db22, sb22, cb22, bb22
      parameter (pi=3.14159265358979312D0)
      dimension sf(2,2,3,10)
      common/process/prc,prdct
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/xmumc/xmcfct2,xmcren2
      common/zzvar/gzzup,gzzdn
      common/zmass/zmw,zmz,zmw2,zmz2
      common/pdf/ih1,ih2,ndns1,ndns2
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/nl/nl
      common/cipdfscale/ipdfscale
c ipdfscale=1 --> use NLO factorization scale
c ipdfscale=2 --> use MC factorization scale
c     
      equivalence (fh1x1(1),u11),(fh1x1(2),d11),(fh1x1(3),s11),
     #            (fh1x1(4),c11),(fh1x1(5),b11)
      equivalence (fh1x1(-1),ub11),(fh1x1(-2),db11),(fh1x1(-3),sb11),
     #            (fh1x1(-4),cb11),(fh1x1(-5),bb11)
      equivalence (fh1x1(0),g11)
c
      equivalence (fh1x2(1),u12),(fh1x2(2),d12),(fh1x2(3),s12),
     #            (fh1x2(4),c12),(fh1x2(5),b12)
      equivalence (fh1x2(-1),ub12),(fh1x2(-2),db12),(fh1x2(-3),sb12),
     #            (fh1x2(-4),cb12),(fh1x2(-5),bb12)
      equivalence (fh1x2(0),g12)
c
      equivalence (fh2x2(1),u22),(fh2x2(2),d22),(fh2x2(3),s22),
     #            (fh2x2(4),c22),(fh2x2(5),b22)
      equivalence (fh2x2(-1),ub22),(fh2x2(-2),db22),(fh2x2(-3),sb22),
     #            (fh2x2(-4),cb22),(fh2x2(-5),bb22)
      equivalence (fh2x2(0),g22)
c
      equivalence (fh2x1(1),u21),(fh2x1(2),d21),(fh2x1(3),s21),
     #            (fh2x1(4),c21),(fh2x1(5),b21)
      equivalence (fh2x1(-1),ub21),(fh2x1(-2),db21),(fh2x1(-3),sb21),
     #            (fh2x1(-4),cb21),(fh2x1(-5),bb21)
      equivalence (fh2x1(0),g21)
c
      do i=1,2
        do j=1,2
          do jproc=1,3
            do itype=1,10
              sf(i,j,jproc,itype)=0.d0
            enddo
          enddo
        enddo
      enddo
c
      if(ipdfscale.eq.1)then
        smufct2=sngl(xmufct2*1.d6)
      elseif(ipdfscale.eq.2)then
        smufct2=sngl(xmcfct2*1.d6)
      else
        write(*,*)'Fatal error in strfvb: unknown ipdfscale',ipdfscale
        stop
      endif
c
      call mlmpdf(ndns1,ih1,smufct2,sngl(x1),fh1x1,5)
      call mlmpdf(ndns2,ih2,smufct2,sngl(x2),fh2x2,5)
      call mlmpdf(ndns1,ih1,smufct2,sngl(x2),fh1x2,5)
      call mlmpdf(ndns2,ih2,smufct2,sngl(x1),fh2x1,5)
c
      if(prdct.eq.'w+') then
        sf(1,1,1,1)=dble(u11*db22*xkm2(1,1))
        sf(1,1,1,2)=dble(u11*sb22*xkm2(1,2))
        sf(1,1,1,3)=dble(u11*bb22*xkm2(1,3))
        sf(1,1,1,4)=dble(c11*db22*xkm2(2,1))
        sf(1,1,1,5)=dble(c11*sb22*xkm2(2,2))
        sf(1,1,1,6)=dble(c11*bb22*xkm2(2,3))
c
        sf(1,2,1,1)=dble(db12*u21*xkm2(1,1))
        sf(1,2,1,2)=dble(sb12*u21*xkm2(1,2))
        sf(1,2,1,3)=dble(bb12*u21*xkm2(1,3))
        sf(1,2,1,4)=dble(db12*c21*xkm2(2,1))
        sf(1,2,1,5)=dble(sb12*c21*xkm2(2,2))
        sf(1,2,1,6)=dble(bb12*c21*xkm2(2,3))
c
        sf(1,1,2,1)=dble(u11*g22*xkm2(1,1))
        sf(1,1,2,2)=dble(u11*g22*xkm2(1,2))
        sf(1,1,2,3)=dble(u11*g22*xkm2(1,3))
        sf(1,1,2,4)=dble(c11*g22*xkm2(2,1))
        sf(1,1,2,5)=dble(c11*g22*xkm2(2,2))
        sf(1,1,2,6)=dble(c11*g22*xkm2(2,3))
c
        sf(1,2,2,1)=dble(g12*u21*xkm2(1,1))
        sf(1,2,2,2)=dble(g12*u21*xkm2(1,2))
        sf(1,2,2,3)=dble(g12*u21*xkm2(1,3))
        sf(1,2,2,4)=dble(g12*c21*xkm2(2,1))
        sf(1,2,2,5)=dble(g12*c21*xkm2(2,2))
        sf(1,2,2,6)=dble(g12*c21*xkm2(2,3))
c
        sf(1,1,3,1)=dble(db11*g22*xkm2(1,1))
        sf(1,1,3,2)=dble(db11*g22*xkm2(2,1))
        sf(1,1,3,3)=dble(sb11*g22*xkm2(1,2))
        sf(1,1,3,4)=dble(sb11*g22*xkm2(2,2))
        sf(1,1,3,5)=dble(bb11*g22*xkm2(1,3))
        sf(1,1,3,6)=dble(bb11*g22*xkm2(2,3))
c
        sf(1,2,3,1)=dble(g12*db21*xkm2(1,1))
        sf(1,2,3,2)=dble(g12*db21*xkm2(2,1))
        sf(1,2,3,3)=dble(g12*sb21*xkm2(1,2))
        sf(1,2,3,4)=dble(g12*sb21*xkm2(2,2))
        sf(1,2,3,5)=dble(g12*bb21*xkm2(1,3))
        sf(1,2,3,6)=dble(g12*bb21*xkm2(2,3))
c
        do i=1,2
          do j=1,2
            do jproc=1,3
              do itype=1,10
                sf(i,j,jproc,itype)=sf(i,j,jproc,itype)*gw**2
              enddo
            enddo
          enddo
        enddo
      elseif(prdct.eq.'w-') then
        sf(1,1,1,1)=dble(d11*ub22*xkm2(1,1))
        sf(1,1,1,2)=dble(s11*ub22*xkm2(1,2))
        sf(1,1,1,3)=dble(b11*ub22*xkm2(1,3))
        sf(1,1,1,4)=dble(d11*cb22*xkm2(2,1))
        sf(1,1,1,5)=dble(s11*cb22*xkm2(2,2))
        sf(1,1,1,6)=dble(b11*cb22*xkm2(2,3))
c
        sf(1,2,1,1)=dble(ub12*d21*xkm2(1,1))
        sf(1,2,1,2)=dble(ub12*s21*xkm2(1,2))
        sf(1,2,1,3)=dble(ub12*b21*xkm2(1,3))
        sf(1,2,1,4)=dble(cb12*d21*xkm2(2,1))
        sf(1,2,1,5)=dble(cb12*s21*xkm2(2,2))
        sf(1,2,1,6)=dble(cb12*b21*xkm2(2,3))
c
        sf(1,1,2,1)=dble(d11*g22*xkm2(1,1))
        sf(1,1,2,2)=dble(d11*g22*xkm2(2,1))
        sf(1,1,2,3)=dble(s11*g22*xkm2(1,2))
        sf(1,1,2,4)=dble(s11*g22*xkm2(2,2))
        sf(1,1,2,5)=dble(b11*g22*xkm2(1,3))
        sf(1,1,2,6)=dble(b11*g22*xkm2(2,3))
c                                         
        sf(1,2,2,1)=dble(g12*d21*xkm2(1,1))
        sf(1,2,2,2)=dble(g12*d21*xkm2(2,1))
        sf(1,2,2,3)=dble(g12*s21*xkm2(1,2))
        sf(1,2,2,4)=dble(g12*s21*xkm2(2,2))
        sf(1,2,2,5)=dble(g12*b21*xkm2(1,3))
        sf(1,2,2,6)=dble(g12*b21*xkm2(2,3))
c
        sf(1,1,3,1)=dble(ub11*g22*xkm2(1,1))
        sf(1,1,3,2)=dble(ub11*g22*xkm2(1,2))
        sf(1,1,3,3)=dble(ub11*g22*xkm2(1,3))
        sf(1,1,3,4)=dble(cb11*g22*xkm2(2,1))
        sf(1,1,3,5)=dble(cb11*g22*xkm2(2,2))
        sf(1,1,3,6)=dble(cb11*g22*xkm2(2,3))
c
        sf(1,2,3,1)=dble(g12*ub21*xkm2(1,1))
        sf(1,2,3,2)=dble(g12*ub21*xkm2(1,2))
        sf(1,2,3,3)=dble(g12*ub21*xkm2(1,3))
        sf(1,2,3,4)=dble(g12*cb21*xkm2(2,1))
        sf(1,2,3,5)=dble(g12*cb21*xkm2(2,2))
        sf(1,2,3,6)=dble(g12*cb21*xkm2(2,3))
c
        do i=1,2
          do j=1,2
            do jproc=1,3
              do itype=1,10
                sf(i,j,jproc,itype)=sf(i,j,jproc,itype)*gw**2
              enddo
            enddo
          enddo
        enddo
      elseif(prdct.eq.'z ') then
        sf(1,1,1,1)=dble(u11*ub22*gzzup)
        sf(1,1,1,2)=dble(d11*db22*gzzdn)
        sf(1,1,1,3)=dble(s11*sb22*gzzdn)
        sf(1,1,1,4)=dble(c11*cb22*gzzup)
        sf(1,1,1,5)=dble(b11*bb22*gzzdn)
c
        sf(1,2,1,1)=dble(ub12*u21*gzzup)
        sf(1,2,1,2)=dble(db12*d21*gzzdn)
        sf(1,2,1,3)=dble(sb12*s21*gzzdn)
        sf(1,2,1,4)=dble(cb12*c21*gzzup)
        sf(1,2,1,5)=dble(bb12*b21*gzzdn)
c
        sf(1,1,2,1)=dble(u11*g22*gzzup)
        sf(1,1,2,2)=dble(d11*g22*gzzdn)
        sf(1,1,2,3)=dble(s11*g22*gzzdn)
        sf(1,1,2,4)=dble(c11*g22*gzzup)
        sf(1,1,2,5)=dble(b11*g22*gzzdn)
        sf(1,1,2,6)=dble(ub11*g22*gzzup)
        sf(1,1,2,7)=dble(db11*g22*gzzdn)
        sf(1,1,2,8)=dble(sb11*g22*gzzdn)
        sf(1,1,2,9)=dble(cb11*g22*gzzup)
        sf(1,1,2,10)=dble(bb11*g22*gzzdn)
c
        sf(1,2,2,1)=dble(g12*u21*gzzup)
        sf(1,2,2,2)=dble(g12*d21*gzzdn)
        sf(1,2,2,3)=dble(g12*s21*gzzdn)
        sf(1,2,2,4)=dble(g12*c21*gzzup)
        sf(1,2,2,5)=dble(g12*b21*gzzdn)
        sf(1,2,2,6)=dble(g12*ub21*gzzup)
        sf(1,2,2,7)=dble(g12*db21*gzzdn)
        sf(1,2,2,8)=dble(g12*sb21*gzzdn)
        sf(1,2,2,9)=dble(g12*cb21*gzzup)
        sf(1,2,2,10)=dble(g12*bb21*gzzdn)
      elseif(prdct.eq.'ww') then
        sf(1,1,1,1)=dble(u11*ub22)
        sf(1,1,1,2)=dble(c11*cb22)
c
        sf(1,2,1,1)=dble(ub12*u21)
        sf(1,2,1,2)=dble(cb12*c21)
c
        sf(2,1,1,1)=dble(d11*db22)
        sf(2,1,1,2)=dble(s11*sb22)
        sf(2,1,1,3)=dble(b11*bb22)
c
        sf(2,2,1,1)=dble(db12*d21)
        sf(2,2,1,2)=dble(sb12*s21)
        sf(2,2,1,3)=dble(bb12*b21)
c
        sf(1,1,2,1)=dble(u11*g22)
        sf(1,1,2,2)=dble(c11*g22)
c
        sf(1,2,2,1)=dble(g12*u21)
        sf(1,2,2,2)=dble(g12*c21)
c
        sf(2,1,2,1)=dble(d11*g22)
        sf(2,1,2,2)=dble(s11*g22)
        sf(2,1,2,3)=dble(b11*g22)
c
        sf(2,2,2,1)=dble(g12*d21)
        sf(2,2,2,2)=dble(g12*s21)
        sf(2,2,2,3)=dble(g12*b21)
c
        sf(1,1,3,1)=dble(ub11*g22)
        sf(1,1,3,2)=dble(cb11*g22)
c
        sf(1,2,3,1)=dble(g12*ub21)
        sf(1,2,3,2)=dble(g12*cb21)
c
        sf(2,1,3,1)=dble(db11*g22)
        sf(2,1,3,2)=dble(sb11*g22)
        sf(2,1,3,3)=dble(bb11*g22)
c
        sf(2,2,3,1)=dble(g12*db21)
        sf(2,2,3,2)=dble(g12*sb21)
        sf(2,2,3,3)=dble(g12*bb21)
      else
        write(6,*) 
     #   'Fatal error in strfvb: non existent final state',prdct
        stop
      endif
      return
      end


      subroutine swap(jproc)
      implicit real * 8(a-h,o-z)
      character * 2 prc,prdct
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/process/prc,prdct
c
      if(prdct.eq.'ww'.or.prdct.eq.'z ')then
        write(*,*)'swap has been called improperly'
        stop
      endif
      if(jproc.eq.3)then
        tmp = gup
        gup = gdown
        gdown = tmp
        ez = -ez
      endif
      return
      end


      subroutine labmom(y)
c boost CM momenta to the lab system
      implicit real * 8 (a-h,o-z)
      common/ycmvar/yw0,yz0,yp0,yl1,yl2,yr1,yr2
      common/perpen/pw0(2),pz0(2),pp0(2),ptvl1(2),ptvl2(2),
     # ptvr1(2),ptvr2(2)
      common/ylbvar/yw,yz,yp
      common/plbvar/pw(2),pz(2),pp(2)
c
      yw = yw0 + y
      yz = yz0 + y
      yp = yp0 + y
      do j=1,2
         pw(j) = pw0(j)
         pz(j) = pz0(j)
         pp(j) = pp0(j)
      enddo
      return
      entry reflex(y)
      yw = - yw0 - y
      yz = - yz0 - y
      yp = - yp0 - y
      do j=1,2
         pw(j) = - pw0(j)
         pz(j) = - pz0(j)
         pp(j) = - pp0(j)
      enddo
      return
      end
c
c
c Begin of phase-space routines
c
c
      subroutine invvb(xm12,xm22,s,x,y,cth1,cth2,phil,cthl,phir,cthr,
     #            str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,idec)
c This routine has been obtained by modifying the analogous routine
c in the VH code, which in turn was obtained from that of older version
c of the VV code; refer to the VH for a comment on the previous changes.
c Here, V and H are replaced by V1 and V2, the two vector bosons.
c The hard process is
c                          +--> r(r1)+rbar(r2)
c                          |
c   a(p1)+b(p2) --> V1(k1)+V2(k2)+c(k)
c                   |
c                   +--> l(l1)+lbar(l2)
c where a, b, and c are light partons, V1 and V2 are vector bosons with 
c k1^2=xm12, and k2^2=xm22, and l, lbar, r, rbar are the leptons emerging
c from the V decays: at this stage, the difference between the lepton and
c the antilepton is purely conventional, and this convention can be freely
c modified without any modification to this routine. The process can be
c described by the same invariants as in FNR [eqs.(2.6) and (2.7)], 
c augmented by 
c
c   v1a = (p1-l1)^2
c   v1b = (p2-l1)^2
c   v1c = (k2+l1)^2 - k2^2
c   v3a = (p1-r1)^2
c   v3b = (p2-r1)^2
c   v3c = (k1+r1)^2 - k1^2
c   v13 = (l1+r1)^2
c
c and the llbar, rrbar invariant masses, which coincide with xm12 and xm22
c respectively. In terms of the invariants, the dot products are 
c (see newinv2.m)
c
c    p1.p2 = s/2
c    p1.k  = -tk/2
c    p2.k  = -uk/2
c    p1.k1 = -(q1q-xm12)/2
c    p2.k2 = -(q2q-xm22)/2
c    k1.k2 = (s2-xm12-xm22)/2
c    p2.k1 = -(q2c-xm12)/2
c    p1.k2 = -(q1c-xm22)/2
c    k.k1  = (w1-xm12)/2
c    k.k2  = (w2-xm22)/2
c    p1.l1 = -v1a/2
c    p2.l1 = -v1b/2
c    k2.l1 = v1c/2
c    k.l1  = (-xm12-v1a-v1b-v1c)/2
c    k.l2  = (xm12-q1q+q2q-tk+v1a+v1b+v1c)/2
c    k2.l2 = (-xm12-xm22+s+tk+uk-v1c)/2
c    p1.l2 = (xm12-q1q+v1a)/2
c    p2.l2 = (-xm22+q2q+s+uk+v1b)/2
c    p1.r1 = -v3a/2
c    p2.r1 = -v3b/2
c    k1.r1 = v3c/2
c    l1.r1 = v13/2
c    l2.r1 = (v3c-v13)/2
c    k.r1  = (-xm22-v3a-v3b-v3c)/2 
c    p1.r2 = (-xm12+q1q+s+tk+v3a)/2 
c    p2.r2 = (xm22-q2q+v3b)/2 
c    k.r2  = (xm22+q1q-q2q-uk+v3a+v3b+v3c)/2 
c    l1.r2 = (v1c-v13)/2
c    l2.r2 = (-xm12-xm22+s+tk+uk+v13-v1c-v3c)/2
c    k1.r2 = (-xm12-xm22+s+tk+uk-v3c)/2
c
c The four momenta are given in the V1V2 rest frame as follows
c     p1 = p10*(1,0,spsi2,cpsi2)
c     p2 = p20*(1,0,spsi ,cpsi )
c     k  = p20*(1,0,spsi1,cpsi1).
c     k1 = (k10, bx*sth2*sth1, bx*cth2*sth1, bx*cth1)
c     k2 = (k20,-bx*sth2*sth1,-bx*cth2*sth1,-bx*cth1).
c If str='p1', we define p1 = p10 (1,0,0,1) (psi2 =0), with psi and psi1
c determined using momentum conservation; according to the work done for 
c Drell Yan, the other options for str have been disabled.
c The momenta of l, lbar are defined in the rest frames of V1 as
c     l1 = sqrt(xm12)/2*(1, spl*sthl, cpl*sthl, cthl)
c     l2 = sqrt(xm12)/2*(1,-spl*sthl,-cpl*sthl,-cthl)
c whereas the momenta of r, rbar are defined in the rest frames of V2 as
c     r1 = sqrt(xm22)/2*(1, spr*sthr, cpr*sthr, cthr)
c     r2 = sqrt(xm22)/2*(1,-spr*sthr,-cpr*sthr,-cthr)
c These and boosted numerically to the V1V2 rest frame in order to compute 
c the invariants. Here, spl=sin(phil), cpl=cos(phil), sthl^2=1-cthl^2,
c where phil and cthl are the two phase-space (angular) variables which
c fully describe the V1 decay in its rest frame. The same applies to 
c spr=sin(phir), cpr=cos(phir), sthr^2=1-cthr^2, with phir and cthr the
c angles relevant to the decay of V2.
c
c The 4-momenta of the particles involved in the hard reaction are stored
c in xmom_cm(ipart,icomp), with icomp=1,2,3,4 corresponding to px,py,pz,E,
c and ipart=1,..,9 with the following identifications:
c ipart=1 -> a; ipart=2 -> b; ipart=3 -> c; ipart=4 -> V1; ipart=5 -> V2;
c ipart=6 -> l; ipart=7 -> lbar; ipart=8 -> r; ipart=9 -> rbar.
c
c Notice that  bx = sqrt(s2)/2 * beta_x[FNR paper]
c
c Call with:   idec=0    -->   V's decay
c              idec=1    -->   V's don't decay (v1a,..,v13 left blank)
c
      implicit none
      real * 8 xm12,xm22,s,x,y,cth1,cth2,phil,cthl,phir,cthr,
     # tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13
      integer idec
      character * 2 str
      real * 8 ptv1,ptv2,ptvg,ptvl1,ptvl2,ptvr1,ptvr2,
     # y1,y2,yg,yl1,yl2,yr1,yr2
      common/perpen/ptv1(2),ptv2(2),ptvg(2),ptvl1(2),ptvl2(2),
     # ptvr1(2),ptvr2(2)
      common/ycmvar/y1,y2,yg,yl1,yl2,yr1,yr2
      real*8 xmom_cm(9,4)
      common/cxmomcm/xmom_cm
      real * 8 s2,drs2,p10,p20,k0,k10,k20,bx,sth1,cpsi,
     # spsi,cpsi2,spsi2,cpsi1,spsi1,xktsq,xkt1sq,xkt2sq,
     # xkt,xkt1,xkt2,tmp,sqs,tiny,zero,sth2,q1c,q2c,w1,w2,xm1,
     # k11,k12,k13,cpl,spl,sthl,l1bar0,l1bar1,l1bar2,l1bar3,
     # l10,l11,l12,l13,tp1l2,tp2l2,xktl1sq,xktl2sq,xktl1,xktl2,
     # e1lab,pl1lab,e2lab,pl2lab,el1lab,pll1lab,el2lab,pll2lab,
     # cosrho,sinrho,cpr,spr,sthr,xm2,k21,k22,k23,r1bar0,r1bar1,
     # r1bar2,r1bar3,r10,r11,r12,r13,tp1r2,tp2r2,xktr1sq,xktr1,
     # xktr2sq,xktr2,er1lab,plr1lab,er2lab,plr2lab
      parameter (tiny=1.d-14)
      parameter (zero=0.d0)
      integer i,j
      integer ichkmom
      common/cichkmom/ichkmom
c
      tk=-s*(1-x)*(1-y)/2.d0
      uk=-s*(1-x)*(1+y)/2.d0
      s2 = tk+uk+s
      drs2 = 2*sqrt(s2)
      p10 = (s+tk)/drs2
      p20 = (s+uk)/drs2
      k0  = -(tk+uk)/drs2
      k10 = (s2+xm12-xm22)/drs2
      k20 = (s2+xm22-xm12)/drs2
      bx=sqrt(s2**2+xm22**2+xm12**2-2*(s2*xm22+s2*xm12+xm22*xm12))/drs2
      sth1 = sqrt(1-cth1**2)
      sth2 = sqrt(1-cth2**2)
      if(str.eq.'p1') then
         cpsi2 = 1
         spsi2 = 0
         cpsi = 1-8*x/((1+y+x*(1-y))*(1-y+x*(1+y)))
         spsi = 4*(1-x)*sqrt(x*(1-y**2))/
     #          ((1+y+x*(1-y))*(1-y+x*(1+y)))
         cpsi1 = (1+y-x*(1-y))/(1+y+x*(1-y))
         spsi1 = sqrt(4*x*(1-y**2))/(1+y+x*(1-y))
      else
         write(6,*) 'Error in invvb: str=',str
         stop
      endif
      q1q = xm12 - 2*p10*(k10-bx*(cth2*sth1*spsi2+cth1*cpsi2))
      q2q = xm22 - 2*p20*(k20+bx*(cth2*sth1*spsi +cth1*cpsi ))
      q1c = xm12 + xm22 - s - tk - q1q
      q2c = xm12 + xm22 - s - uk - q2q
      w1  = xm12 - q1q + q2q - tk
      w2  = xm22 - q2q + q1q - uk
c
      if(abs(q1q-xm12).lt.tiny) then
        y1  = 1.d8
      elseif(abs(q2c-xm12).lt.tiny) then
        y1  = -1.d8
      else
        y1 = .5d0*log( (xm12-q2c)/(xm12-q1q) )
      endif
      if(abs(q1c-xm22).lt.tiny) then
        y2  = 1.d8
      elseif(abs(q2q-xm22).lt.tiny) then
        y2  = -1.d8
      else
        y2 = .5d0*log( (xm22-q2q)/(xm22-q1c) )
      endif
      if(abs(tk).lt.tiny) then
        yg  = 1.d8
      elseif(abs(uk).lt.tiny) then
        yg  = -1.d8
      else
        yg  = .5d0*log( uk/tk )
      endif
c-----------------------------------------------------------------
c xktsq, xkt1sq e xkt2sq are the square of transverse momenta of g, V1, 
c and V2 respectively. The axis orientation is such that V1 is always
c along the x direction. The component of p_T(V2) along the y direction
c is always positive or zero
c
      xktsq = uk*tk/s
      if(xktsq.eq.0) then
         ptv1(1) = bx*sth1
         ptv1(2) = 0.d0
         ptv2(1) = -ptv1(1)
         ptv2(2) = 0.d0
         ptvg(1) = 0.d0
         ptvg(2) = 0.d0
         xkt1 = ptv1(1)
         xkt2 = xkt1
      else
         xkt1sq = (xm12-q2c)*(xm12-q1q)/s - xm12
         xkt2sq = (xm22-q2q)*(xm22-q1c)/s - xm22
         xkt = sqrt(xktsq)
         xkt1 = sqrt(xkt1sq)
         xkt2 = sqrt(xkt2sq)
         ptv1(1) = xkt1
         ptv1(2) = 0.d0
         ptv2(1) = (xktsq-xkt1sq-xkt2sq)/(2.d0*xkt1)
         tmp = xkt2sq-ptv2(1)**2
         if(tmp.gt.0.d0)then
            ptv2(2) = sqrt(tmp)
         else
            ptv2(2) = 0.d0
         endif
         ptvg(1) = (xkt2sq-xkt1sq-xktsq)/(2.d0*xkt1)
         tmp = xktsq-ptvg(1)**2
         if(tmp.gt.0.d0)then
            ptvg(2) = -sqrt(tmp)
         else
            ptvg(2) = 0.d0
         endif
      endif
      if(ichkmom.eq.0)call checkptcon(ptv1,ptv2,ptvg)
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
        xmom_cm(3,1)=ptvg(1)
        xmom_cm(3,2)=ptvg(2)
        xmom_cm(3,3)=sqs/2.d0*(1-x)*y
        xmom_cm(3,4)=sqs/2.d0*(1-x)
      endif
      e1lab=(2*xm12-q1q-q2c)/(2*sqs)
      pl1lab=(q1q-q2c)/(2*sqs)
      xmom_cm(4,1)=ptv1(1)
      xmom_cm(4,2)=ptv1(2)
      xmom_cm(4,3)=pl1lab
      xmom_cm(4,4)=e1lab
      e2lab=(2*xm22-q1c-q2q)/(2*sqs)
      pl2lab=(q1c-q2q)/(2*sqs)
      xmom_cm(5,1)=ptv2(1)
      xmom_cm(5,2)=ptv2(2)
      xmom_cm(5,3)=pl2lab
      xmom_cm(5,4)=e2lab
c
      if(idec.eq.0)then
        cpl=cos(phil)
        spl=sin(phil)
        sthl=sqrt(1-cthl**2)
        xm1=sqrt(xm12)
        cpr=cos(phir)
        spr=sin(phir)
        sthr=sqrt(1-cthr**2)
        xm2=sqrt(xm22)
c Momentum of k1 in the rest frame of k1+k2
        k11=bx*sth2*sth1
        k12=bx*cth2*sth1
        k13=bx*cth1
c Momentum of l1 in the rest frame of k1
        l1bar0=xm1/2.d0
        l1bar1=l1bar0*spl*sthl
        l1bar2=l1bar0*cpl*sthl
        l1bar3=l1bar0*cthl
c (l10,..,l13) is the momentum of l1 in the rest frame of k1+k2
        call hvhlb4(k11,k12,k13,k10,xm1,
     #              l1bar1,l1bar2,l1bar3,l1bar0,
     #              l11,l12,l13,l10)
c Momentum of k2 in the rest frame of k1+k2
        k21=-bx*sth2*sth1
        k22=-bx*cth2*sth1
        k23=-bx*cth1
c Momentum of r1 in the rest frame of k2
        r1bar0=xm2/2.d0
        r1bar1=r1bar0*spr*sthr
        r1bar2=r1bar0*cpr*sthr
        r1bar3=r1bar0*cthr
c (r10,..,r13) is the momentum of r1 in the rest frame of k1+k2
        call hvhlb4(k21,k22,k23,k20,xm2,
     #              r1bar1,r1bar2,r1bar3,r1bar0,
     #              r11,r12,r13,r10)
c
        v1a=-2*p10*(l10-l12*spsi2-l13*cpsi2)
        v1b=-2*p20*(l10-l12*spsi -l13*cpsi )
        v1c=2*(k20*l10+bx*(sth2*sth1*l11+cth2*sth1*l12+cth1*l13))
        v3a=-2*p10*(r10-r12*spsi2-r13*cpsi2)
        v3b=-2*p20*(r10-r12*spsi -r13*cpsi )
        v3c=2*(k10*r10-bx*(sth2*sth1*r11+cth2*sth1*r12+cth1*r13))
        v13=2*(l10*r10-l11*r11-l12*r12-l13*r13)
        tp1l2=xm12-q1q+v1a
        tp2l2=-xm22+q2q+s+uk+v1b
        xktl1sq=v1a*v1b/s
        xktl1=sqrt(xktl1sq)
        xktl2sq=tp1l2*tp2l2/s
        xktl2=sqrt(xktl2sq)
        yl1=0.5d0*log(v1b/v1a)
        yl2=0.5d0*log(tp2l2/tp1l2)
        el1lab=-(v1a+v1b)/(2.d0*sqs)
        pll1lab=(v1a-v1b)/(2.d0*sqs)
        tp1r2=-xm12+q1q+s+tk+v3a
        tp2r2=xm22-q2q+v3b
        xktr1sq=v3a*v3b/s
        xktr1=sqrt(xktr1sq)
        xktr2sq=tp1r2*tp2r2/s
        xktr2=sqrt(xktr2sq)
        yr1=0.5d0*log(v3b/v3a)
        yr2=0.5d0*log(tp2r2/tp1r2)
        er1lab=-(v3a+v3b)/(2.d0*sqs)
        plr1lab=(v3a-v3b)/(2.d0*sqs)
c Exploit the fact that V1 is always along x in the transverse plane
        if(xkt1.ne.0.d0)then
          ptvl1(1)=(e1lab*el1lab-pl1lab*pll1lab-xm12/2.d0)/xkt1
          tmp=xktl1sq-ptvl1(1)**2
          if(tmp.gt.0.d0)then
            ptvl1(2)=sqrt(tmp)
            if(x.lt.1.d0.and.abs(y).lt.1.d0)then
c tmp here is pt(V2)_y * pt(l1)_y, and we have set pt(V2)_y > 0
              tmp=e2lab*el1lab-pl2lab*pll1lab-ptv2(1)*ptvl1(1)-v1c/2.d0
              if(tmp.lt.0.d0)ptvl1(2)=-ptvl1(2)
            else
c The rest frame of k1+k2 coincides with the partonic cm frame in the
c soft and (up to a longitudinal boost) collinear limits. We can thus
c get ptv1(1) by rotating k1 as given before, and apply the same rotation
c to the lepton. Notice that in our parametrization the components 1 and 2
c are proportional to sin and cos (usually, it is the opposite)
              cosrho=k12/ptv1(1)
              sinrho=k11/ptv1(1)
              if( sign(1.d0,ptvl1(2)) .ne.
     #          sign(1.d0,-l12*sinrho+l11*cosrho) )ptvl1(2)=-ptvl1(2)
            endif
          else
            ptvl1(2)=0.d0
          endif
c Repeat for r1 what done for l1; the role of V1 and V2 is unchanged, since
c they only serve to define the axes in the transverse plane
          ptvr1(1)=(e1lab*er1lab-pl1lab*plr1lab-v3c/2.d0)/xkt1
          tmp=xktr1sq-ptvr1(1)**2
          if(tmp.gt.0.d0)then
            ptvr1(2)=sqrt(tmp)
            if(x.lt.1.d0.and.abs(y).lt.1.d0)then
              tmp=e2lab*er1lab-pl2lab*plr1lab-ptv2(1)*ptvr1(1)-
     #            xm22/2.d0
              if(tmp.lt.0.d0)ptvr1(2)=-ptvr1(2)
            else
              cosrho=k12/ptv1(1)
              sinrho=k11/ptv1(1)
              if( sign(1.d0,ptvr1(2)) .ne.
     #          sign(1.d0,-r12*sinrho+r11*cosrho) )ptvr1(2)=-ptvr1(2)
            endif
          else
            ptvr1(2)=0.d0
          endif
        else
c Strictly speaking, this is not always correct; however, events in which
c V1 and V2 have zero transverse momentum are unlikely to matter
          ptvl1(1)=0.d0
          ptvl1(2)=0.d0
          ptvr1(1)=0.d0
          ptvr1(2)=0.d0
        endif
        xmom_cm(6,1)=ptvl1(1)
        xmom_cm(6,2)=ptvl1(2)
        xmom_cm(6,3)=pll1lab
        xmom_cm(6,4)=el1lab
        el2lab=(tp1l2+tp2l2)/(2.d0*sqs)
        pll2lab=(tp2l2-tp1l2)/(2.d0*sqs)
        ptvl2(1)=ptv1(1)-ptvl1(1)
        ptvl2(2)=ptv1(2)-ptvl1(2)
        xmom_cm(7,1)=ptvl2(1)
        xmom_cm(7,2)=ptvl2(2)
        xmom_cm(7,3)=pll2lab
        xmom_cm(7,4)=el2lab
        xmom_cm(8,1)=ptvr1(1)
        xmom_cm(8,2)=ptvr1(2)
        xmom_cm(8,3)=plr1lab
        xmom_cm(8,4)=er1lab
        er2lab=(tp1r2+tp2r2)/(2.d0*sqs)
        plr2lab=(tp2r2-tp1r2)/(2.d0*sqs)
        ptvr2(1)=ptv2(1)-ptvr1(1)
        ptvr2(2)=ptv2(2)-ptvr1(2)
        xmom_cm(9,1)=ptvr2(1)
        xmom_cm(9,2)=ptvr2(2)
        xmom_cm(9,3)=plr2lab
        xmom_cm(9,4)=er2lab
      elseif(idec.eq.1)then
        v1a=0.d0
        v1b=0.d0
        v1c=0.d0
        v3a=0.d0
        v3b=0.d0
        v3c=0.d0
        v13=0.d0
        do i=6,9
          do j=1,4
            xmom_cm(i,j)=0.d0
          enddo
        enddo
      else
        write(6,*) 'Error in invvb: idec=',idec
        stop
      endif
      if(ichkmom.eq.0)then
        call checkmvb(xmom_cm,s,0.d0,1,2)
        if(idec.eq.0)then
          call checkmvb(xmom_cm,s,0.d0,1,1)
          call checkdvb(xmom_cm,4,6,7)
          call checkdvb(xmom_cm,5,8,9)
        endif
      endif
      return
      end


      subroutine checkmvb(xmom,smax,ybst,iflag,itype)
      implicit none
      real * 8 xmom(9,4)
      real * 8 smax,ybst,xpmax
      real*8 x1,x2
      common/cx1x2/x1,x2
      real * 8 tiny,vtiny,xsum(4),xsuma(4),xsign,xrat(4)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-4)
      integer iflag,itype,i,j,jj,jflag,jeflag,jmax
c
      if(itype.eq.1)then
        jmax=9
      elseif(itype.eq.2)then
        jmax=5
      else
        write(6,*)'Wrong option in checkmvb'
        stop
      endif
      jflag=0
      jeflag=0
      xpmax=sqrt(smax)/2.d0*(1+vtiny)
      do i=1,4
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=1,jmax
          if((itype.eq.1.and.j.ne.4.and.j.ne.5).or.itype.eq.2)then
            if(i.ne.4.and.xmom(j,i).gt.xpmax)jeflag=1
            xsign=1.d0
            if(j.eq.1.or.j.eq.2)xsign=-1.d0
            xsum(i)=xsum(i)+xmom(j,i)*xsign
            xsuma(i)=xsuma(i)+abs(xmom(j,i))
          endif
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
          do j=1,7
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
        do j=1,7
          write(*,'(4(d14.8,1x))') (xmom(j,jj),jj=1,4)
        enddo
        stop
      endif
      return
      end


C      subroutine checkptcon(ptv1,ptv2,ptvg)
C      implicit none
C      real*8 ptv1(2),ptv2(2),ptvg(2),tiny,pt1,pt2,ptmax
C      parameter (tiny=1.d-5)
C      integer jj
Cc
C      ptmax=max(abs(ptv1(1)),abs(ptv2(1)),abs(ptvg(1)),
C     #          abs(ptv1(2)),abs(ptv2(2)),abs(ptvg(2)))
C      pt1=ptv1(1)+ptv2(1)+ptvg(1)
C      pt2=ptv1(2)+ptv2(2)+ptvg(2)
C      if(pt1.gt.ptmax*tiny.or.pt2.gt.ptmax*tiny)then
C        write(*,*)'Transverse momentum is not conserved'
C        write(*,'(4(d14.8,1x))') (ptv1(jj),jj=1,2)
C        write(*,'(4(d14.8,1x))') (ptv2(jj),jj=1,2)
C        write(*,'(4(d14.8,1x))') (ptvg(jj),jj=1,2)
C        stop
C      endif
C      return
C      end


      subroutine checkdvb(xmom,idec,iprod1,iprod2)
      implicit none
      real * 8 xmom(9,4)
      real * 8 tiny,xsum(4),xsuma(4),xrat(4)
      parameter (tiny=5.d-3)
      integer idec,iprod1,iprod2,jflag,i,jj
c
      jflag=0
      do i=1,4
        xsum(i)=xmom(idec,i)-xmom(iprod1,i)-xmom(iprod2,i)
        xsuma(i)=abs(xmom(idec,i))+
     #           abs(xmom(iprod1,i))+abs(xmom(iprod2,i))
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved'
          write(*,'(4(d14.8,1x))') (xmom(idec,jj),jj=1,4)
          write(*,'(4(d14.8,1x))') (xmom(iprod1,jj),jj=1,4)
          write(*,'(4(d14.8,1x))') (xmom(iprod2,jj),jj=1,4)
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=1,4)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=1,4)
        stop
      endif
      return
      end


Cc This is obtained by modifying the routine HWULB4 of HERWIG
CC-----------------------------------------------------------------------
C      subroutine hwulb4(ps1,ps2,ps3,ps4,ps5,
C     #                  pi1,pi2,pi3,pi4,
C     #                  pf1,pf2,pf3,pf4)
CC-----------------------------------------------------------------------
CC     TRANSFORMS PI (GIVEN IN REST FRAME OF PS) INTO PF (IN LAB)
CC     N.B. P(1,2,3,4) = (PX,PY,PZ,E); PS(5)=M
CC-----------------------------------------------------------------------
C      real*8 ps1,ps2,ps3,ps4,ps5,pi1,pi2,pi3,pi4,pf1,pf2,pf3,pf4,
C     # tmp,fn
C      if (ps4.eq.ps5) then
C        pf1= pi1
C        pf2= pi2
C        pf3= pi3
C        pf4= pi4
C      else
C        tmp  = (pi1*ps1+pi2*ps2
C     &         +pi3*ps3+pi4*ps4)/ps5
C        fn   = (tmp+pi4) / (ps4+ps5)
C        pf1= pi1 + fn*ps1
C        pf2= pi2 + fn*ps2
C        pf3= pi3 + fn*ps3
C        pf4= tmp
C      end if
C      end


C      subroutine fillmom(pt,y,xm2,xmom,ipart)
C      implicit none
C      real * 8 pt(2),y,xm2,xmom(9,4)
C      integer ipart
C      real * 8 xmt
Cc
C      xmt=sqrt(pt(1)**2+pt(2)**2+xm2)
C      xmom(ipart,1)=pt(1)
C      xmom(ipart,2)=pt(2)
C      xmom(ipart,3)=xmt*sinh(y)
C      xmom(ipart,4)=xmt*cosh(y)
C      return
C      end


      subroutine checkmvb2(xmom,ycm)
c Performs a check only on the vector boson kinematics
      implicit none
      real * 8 xmom(9,4),ycm
      real * 8 tiny
      parameter (tiny=5.d-3)
      integer iflag,i,j,jj
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 yw,yz,yp
      common/ylbvar/yw,yz,yp
      real * 8 yw0,yz0,yp0,yl1,yl2,yr1,yr2
      common/ycmvar/yw0,yz0,yp0,yl1,yl2,yr1,yr2
      real * 8 pw(2),pz(2),pp(2)
      common/plbvar/pw,pz,pp
      real * 8 xmom_cm(9,4)
      common/cxmomcm/xmom_cm
      real * 8 xtmp(9,4),xdiff(4),xrat(4),xmax
c
      do i=1,2
        call boost(-ycm,
     #         xmom_cm(i,1),xmom_cm(i,2),xmom_cm(i,3),xmom_cm(i,4),
     #         xtmp(i,1),xtmp(i,2),xtmp(i,3),xtmp(i,4))
      enddo
      if(pp(1).eq.0.d0.and.pp(2).eq.0)then
        iflag=1
      else
        iflag=0
        call fillmom(pp,yp,0.d0,xtmp,3)
      endif
      call fillmom(pw,yw,zmw2,xtmp,4)
      call fillmom(pz,yz,zmz2,xtmp,5)
      do i=1,5
        do j=1,4
          if(i.ne.3.or.(i.eq.3.and.iflag.eq.0))then
            call getmax(xtmp(i,1),xtmp(i,2),xtmp(i,3),xtmp(i,4),xmax)
            xdiff(j)=xtmp(i,j)-xmom(i,j)
            if(xmax.eq.0.d0)then
              xrat(j)=abs(xdiff(j))
            else
              xrat(j)=abs(xdiff(j))/xmax
            endif
            if(xrat(j).gt.tiny)then
              write(*,*)'Wrong momentum reconstruction'
              write(*,*)'i,j=',i,j
              write(*,*)ycm
              write(*,*)yw,yz,yp
              write(*,*)yw0,yz0,yp0
              write(*,'(8(d10.4,1x))') (xmom_cm(i,jj),jj=1,4)
              write(*,'(8(d10.4,1x))') (xmom(i,jj),jj=1,4)
              write(*,'(8(d10.4,1x))') (xtmp(i,jj),jj=1,4)
              write(*,'(8(d10.4,1x))') (xdiff(jj),jj=1,4)
              write(*,'(8(d10.4,1x))') (xrat(jj),jj=1,4)
              stop
            endif
          endif
        enddo
      enddo
      return
      end


      subroutine getmax(a1,a2,a3,a4,xmax)
      implicit none
      real * 8 a1,a2,a3,a4,xmax
c
      xmax=0.d0
      if(abs(a1).gt.xmax)xmax=abs(a1)
      if(abs(a2).gt.xmax)xmax=abs(a2)
      if(abs(a3).gt.xmax)xmax=abs(a3)
      if(abs(a4).gt.xmax)xmax=abs(a4)
      return
      end


C      function bwfunc(s,xm02,gah)
Cc Returns the Breit Wigner function, normalized in such a way that
Cc its integral in the range (-inf,inf) is one
C      implicit none
C      real*8 bwfunc,s,xm02,gah
C      real*8 pi,xm0
C      parameter (pi=3.1415926535897932d0)
Cc
C      xm0=sqrt(xm02)
C      bwfunc=xm0*gah/(pi*((s-xm02)**2+xm02*gah**2))
C      return
C      end


C      function xbwmass3(t,xm02,ga,bwdelf,bwfmmn)
Cc Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
Cc and the mass range (implicit in bwdelf and bwfmmn). This function
Cc is the inverse of F(M^2), where
Cc   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
Cc   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
Cc and therefore eats up the Breit-Wigner when changing integration 
Cc variable M^2 --> t
C      implicit none
C      real*8 xbwmass3,t,xm02,ga,bwdelf,bwfmmn
C      real*8 pi,xm0
C      parameter (pi=3.1415926535897932d0)
Cc
C      xm0=sqrt(xm02)
C      xbwmass3=xm02+xm0*ga*tan(pi*bwdelf*t-bwfmmn)
C      return
C      end


      function bwfunc_mod(s,xm02,ga0,gamn,gapl)
c Returns a skewed Breit Wigner function, defined as follows
c   skBW(M)=BW(M,M0,G0,G_mn)  if  M<M0
c   skBW(M)=BW(M,M0,G0,G_pl)  if  M>M0
c where 
c   BW(M,M0,G0,G)=(M0*G^2)/(pi*G0) 1/((M^2-M0^2)^2+M0^2 G^2)
c whose normalization is therefore such that
c   BW(M0,M0,G0,G)=1/(pi*M0*G0)
      implicit none
      real*8 bwfunc_mod,s,xm02,ga0,gamn,gapl
      real*8 pi,xm0,tmp
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      if(s.le.xm02)then
        tmp=xm0*gamn**2/(ga0*pi*((s-xm02)**2+xm02*gamn**2))
      else
        tmp=xm0*gapl**2/(ga0*pi*((s-xm02)**2+xm02*gapl**2))
      endif
      bwfunc_mod=tmp
      return
      end


      function xbwmass3_mod(t,xm02,ga0,gamn,gapl,bwdelf,bwfmmn)
c Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
c the widths of the skewed Breit Wigner function, and the mass range 
c (implicit in bwdelf and bwfmmn). This function is the inverse of 
c F(M^2), where
c   F(M^2)=\int_{xmlow2}^{M^2} ds skBW(sqrt(s))
c   skBW(M)=BW(M,M0,G0,G_mn)  if  M<M0
c   skBW(M)=BW(M,M0,G0,G_pl)  if  M>M0
c with
c   BW(M,M0,G0,G)=(M0*G^2)/(pi*G0) 1/((M^2-M0^2)^2+M0^2 G^2)
c whose normalization is therefore such that
c   BW(M0,M0,G0,G)=1/(pi*M0*G0)
c and therefore eats up the skewed Breit-Wigner when changing integration 
c variable M^2 --> t
      implicit none
      real*8 xbwmass3_mod,t,xm02,ga0,gamn,gapl,bwdelf,bwfmmn
      real*8 pi,xm0,tmp
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      if(t.le.(bwfmmn/pi))then
        tmp=xm02+xm0*gamn*tan( ga0/gamn*(pi*bwdelf*t-bwfmmn) )
      else
        tmp=xm02+xm0*gapl*tan( ga0/gapl*(pi*bwdelf*t-bwfmmn) )
      endif
      xbwmass3_mod=tmp
      return
      end


      subroutine chvar(z,x,xjac,del)
      implicit real * 8 (a-h,o-z)
      data odel/0.d0/
      if(odel.ne.del) then
         odel = del
         xld = log(del)
         x0  = 1 + del + 1/xld
         if(x0.le.0.or.x0.ge.1) then
            write(*,*) 'inappropriate delta'
            stop
         endif
         y0  = log(1-x0+del)
         yh = y0 - x0*xld
         yl = xld
         ydif = yh-yl
      endif
      y = yl + z*ydif
      xjac = xjac * ydif
      if(y.gt.y0) then
          x = - (yh - y)/xld
          xjac = xjac / (-xld)
      else
          t = exp(y)
          xjac = xjac * t
          x = 1 - t + del
      endif
      return
      end


      subroutine wzchvar(parth1,cth1,xjac,ro)
c--------------------------------------------------
c Data la variabile 0<parth1<1 ottiene -1<cth1<1
c e moltiplica xjac per il d cth1 / d parth1
c
      implicit real * 8 (a-z)
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
c Begin of MC-related functions (dead zone)
c
c
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


C      function itoosoftkin()
c Returns 1 when a three-body kinematics can be safely approximated
c with a two-body kinematics. It is useful when three-body NLO configurations
c are obtained, which cannot be produced through showering
C      implicit real*8(a-h,o-z)
c
C      itmp=0
C      itoosoftkin=itmp
C      return
C      end
c
c
c End of MC-related functions (dead zone)
c
c
c
c
c Begin of MC subtraction terms
c
c
      subroutine xmcsuvb(s,x,y,cth1,cth2,x1,x2,ycnt,ww3up,ww3do)
c Returns the MC subtraction terms, times 4*tk*uk. In other MC@NLO codes
c the damping factor is 4*tk*uk/s**2; here, the 1/s**2 is not present,
c since the real emission matrix elements are also multiplied by 4*tk*uk.
      implicit none
      real * 8 s,x,y,cth1,cth2,x1,x2,ycnt,ww3up,ww3do
      real * 8 tiny,vtf,vcf,xmin,xlim,xlim1,xlim2,sbar,tbar,ubar,
     # brnup,brndo,xsoft,yd,ydead_mod,xfact,z,zherw_spl,xi,xiherw_spl,
     # ap,xjac_xizspl,zherw_smn,xiherw_smn,xjac_xizsmn,tt,gfact,gfun
      parameter (tiny=1.d-8)
      parameter (vtf=1/2.d0)
      parameter (vcf=4/3.d0)
      parameter (xmin=0.69519410160110384d0)
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 al_gfun,be_gfun,ccc_gfun
      common/cgfunpar/al_gfun,be_gfun,ccc_gfun
      character * 2 prc,prdct
      common/process/prc,prdct
      integer ia1ora2
      common/cia1ora2/ia1ora2
c
      if(prdct.ne.'w+'.and.prdct.ne.'w-'.and.
     #   prdct.ne.'z '.and.prdct.ne.'ww')then
        write(6,*)'xmcsuvb: process not implemented',prdct
        stop
      endif
      if(abs(ycnt).ne.1.d0) then
        write(6,*)'xmcsuvb called improperly: ycnt=',ycnt
        stop
      endif
      xlim=0.d0
      xlim1=0.d0
      xlim2=0.d0
c Compute the Born cross section; for WZ production, the t<-->u is not
c necessary, owing to the call to swap() in the main code.
      call getmcinvvb(zmw2,zmz2,s,x,y,cth1,cth2,x1,x2,
     #                sbar,tbar,ia1ora2)
      ubar=zmw2+zmz2-sbar-tbar
      if(prdct.eq.'ww'.or.prdct.eq.'z ')then
        if(prc.eq.'qq'.or.prc.eq.'qg')then
          call vbborn(sbar,tbar,zmw2,zmz2,brnup,brndo)
        elseif(prc.eq.'ag')then
          call vbborn(sbar,ubar,zmw2,zmz2,brnup,brndo)
        endif
      else
        call vbborn(sbar,tbar,zmw2,zmz2,brnup,brndo)
      endif
c Compute the process-independent factors
      if(prc.eq.'qq')then
        xsoft=1.d0-be_gfun+xmin*be_gfun
        yd=ydead_mod(x)
        if(ycnt.eq.1.d0)then
          if(x.gt.xsoft)then
            xlim2=(1+x**2)*(1+y)
          endif
          if(y.gt.yd)then
            if(1-x.lt.tiny)then
              xlim1=8*(1+y)/(3+y)-
     #              2*(1-x)*(1+y)*(7+6*y+3*y**2)/(3+y)**2
            elseif(1-y.lt.tiny)then
              xlim1=2*(1+x**2)-
     #              (2-1/x-(7*x)/2.d0+5*x**2-3*x**3/2.d0)*(1-y)
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_spl(x,y)
              xi=xiherw_spl(x,y)
              ap=(1+z**2)/(1-z)
              xlim1=xjac_xizspl(x,y)*xfact*ap/xi
            endif
          endif
        elseif(ycnt.eq.-1.d0)then
          if(x.gt.xsoft)then
            xlim2=(1+x**2)*(1-y)
          endif
          if(y.lt.-yd)then
            if(1-x.lt.tiny)then
              xlim1=8*(1-y)/(3-y)-
     #              2*(1-x)*(1-y)*(7-6*y+3*y**2)/(3-y)**2
            elseif(1+y.lt.tiny)then
              xlim1=2*(1+x**2)-
     #              (2-1/x-(7*x)/2.d0+5*x**2-3*x**3/2.d0)*(1+y)
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_smn(x,y)
              xi=xiherw_smn(x,y)
              ap=(1+z**2)/(1-z)
              xlim1=xjac_xizsmn(x,y)*xfact*ap/xi
            endif
          endif
        endif
        tt=(1.d0-x)/(1.d0-xsoft)
        gfact=gfun(tt)
        xlim=4*s*vcf*(xlim1*gfact+xlim2*(1.d0-gfact))
      elseif(prc.eq.'qg'.or.prc.eq.'ag')then
        yd=ydead_mod(x)
        if(ycnt.eq.1.d0)then
          xlim=0.d0
        elseif(ycnt.eq.-1.d0)then
          if(y.lt.-yd)then
            if(1-x.lt.tiny)then
              xlim=(1-x)*(1-y)
            elseif(1+y.lt.tiny)then
              xlim=2-6*x+8*x**2-4*x**3-
     #             (4*x**5-19*x**4+31*x**3-23*x**2+8*x-1)*(1+y)/x
            else
              xfact=(1-x)*(1-y**2)
              z=zherw_smn(x,y)
              xi=xiherw_smn(x,y)
              ap=z**2+(1-z)**2
              xlim=xjac_xizsmn(x,y)*xfact*ap/xi
            endif
          endif
        endif
        xlim=4*s*vtf*xlim
      else
        write(*,*)'xmcsuvb: unknown process',prc
        stop
      endif
      ww3up = xlim*brnup
      ww3do = xlim*brndo
      return
      end


      subroutine getmcinvvb(xm12,xm22,s,x,y,cth1,cth2,x1,x2,
     #                      sbar,tbar,iinv)
c Returns the invariants of the reduced hard process, consistently with
c the boosts performed by Herwig; x1 and x2 are the Bjorken x's relevant 
c to the 2-->3 hard process. Refer to the comment at the beginning of
c invar for the definition of the kinematics. Call with
c    iinv=1  -->  exact (2->3) to (2->2) relations
c    iinv=2  -->  collinear approximation
c This function is obtained from the analogous one of the VH code,
c by eliminating the computations of the invariants relevant to leptons.
c WARNING: the invariants returned by invvb are defined with the
c conventions of FNR, NPB383(92)3. Thus, for consistency, we define
c           tbar = (p1-k1)^2 == -2p1bar.k1bar+k1bar^2
c whereas the analytical results are relevant to the dot product only
      implicit none
      real*8 xm12,xm22,s,x,y,cth1,cth2,x1,x2,sbar,tbar
      integer iinv
      real*8 tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,q1c,q2c,
     # xmn,xpl,galonred,betalon,dm12,zero,one
      parameter (zero=0.d0)
      parameter (one=1.d0)
      integer ione
      parameter (ione=1)
      character * 2 str
      parameter (str='p1')
c
      if(iinv.eq.1)then
        call invvb(xm12,xm22,s,x,y,cth1,cth2,zero,zero,zero,zero,
     #        str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
        q1c=xm12+xm22-s-tk-q1q
        q2c=xm12+xm22-s-uk-q2q
        sbar=s*x
        xmn=((s+uk)*x1/s-(s+tk)*x2/s)/2.d0
        xpl=((s+uk)*x1/s+(s+tk)*x2/s)/2.d0
        galonred=sqrt(xpl**2-x1*x2*tk*uk/s**2)
        betalon=-xmn/galonred
        dm12=xm12-xm22
        tbar=-sbar/2.d0*( 1-(x2*(q1q-q1c-dm12)+x1*(q2q-q2c+dm12))/
     #                      (2*s*galonred) )
     #       -dm12/2.d0*(1-betalon) + xm12
      elseif(iinv.eq.2)then
c Use here the y=1 kinematics to compute the reduced invariants. The y=-1
c kinematics could also be used, with tbar=q1q; in fact, in both cases we 
c get tbar=-s*x/2.d0*(1-betax*cth1)+(xm12+xm22)/2
        call invvb(xm12,xm22,s,x,one,cth1,cth2,zero,zero,zero,zero,
     #        str,tk,uk,q1q,q2q,v1a,v1b,v1c,v3a,v3b,v3c,v13,ione)
        sbar=s*x
        tbar=q2q
      else
        write(*,*)'Error in getmcinvvb: iinv=',iinv
        stop
      endif 
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
c End of utility routines for xi, z, and 2-->2 invariants
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
c Initialization
c
c
      subroutine setpvb()
      implicit none
      real * 8 pi,one,xme,xmmu,xmtau,gaw_pdg,gaz_pdg
      parameter (pi=3.14159265358979312D0)
      parameter (one=1.d0)
c Values from PDG 2003; this code uses TeV for computations
      parameter (xme=0.510998902d-6)
      parameter (xmmu=105.6583568d-6)
      parameter (xmtau=1776.99d-6)
      parameter (gaw_pdg=2.124d-3)
      parameter (gaz_pdg=2.495d-3)
      real * 8 sh,xmufct2,xmuren2,as,xnc
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      real * 8 gzzup,gzzdn
      common/zzvar/gzzup,gzzdn
      real * 8 xmw,xmz,xmw2,xmz2
      common/mass/xmw,xmz,xmw2,xmz2
      real * 8 zmw,zmz,zmw2,zmz2
      common/zmass/zmw,zmz,zmw2,zmz2
      real * 8 gaw,gaz
      common/width/gaw,gaz
      real * 8 xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/weakcoup/xkm,xkm2,sw,ze2,gup,gdown,ez,gw
      real * 8 xlep1mass(2),xlep2mass(2)
      common/clepmass/xlep1mass,xlep2mass
      real * 8 brrv1on,brrv2on,brrv1msb,brrv2msb
      common/brratios/brrv1on,brrv2on,brrv1msb,brrv2msb
      character * 2 prc,prdct
      common/process/prc,prdct
      integer nl
      common/nl/nl
      integer idec
      common/cidec/idec
      integer iwidth
      common/ciwidth/iwidth
c Type of V decays, with HERWIG conventions; see the beginning of this file
      integer il1hw,il2hw
      common/cilhw/il1hw,il2hw
c Identities of the vector bosons or leptons in the final state 
c (PDG conventions)
      integer ip4,ip5,ip6,ip7
      common/ci2part/ip4,ip5,ip6,ip7
c Maps HERWIG conventions for decays into NLO conventions
      integer imapz(1:7)
      data imapz/1,4,2,5,3,6,7/
c PDG codes for charged leptons and neutrinos for a given IL (NLO) code;
c the particle code (not the antiparticle) is entered here
c Charged lepton from W decay
      integer ichlw(1:3)
      data ichlw/11,13,15/
c Neutrino from W decay
      integer ineuw(1:3)
      data ineuw/12,14,16/
c Charged lepton from Z decay
      integer ichlz(1:6)
      data ichlz/11,13,15,0,0,0/
c Neutrino from Z decay
      integer ineuz(1:6)
      data ineuz/0,0,0,12,14,16/
      real * 8 tmpmss(3),gf,alfaem2,cos2w,sin2w,tmp,gvup,gvdo,gaup,gado,
     # vcel,acel,aelupp,belupp,vcnu,acnu,anuupp,bnuupp,wdtzon,wdtzmsb,
     # xalfaem,wdtwon,wdtwmsb,xmwme,xmzme,gawme,gazme,dec2cf,dec1cf
      integer i,j
c
c Check that units in this code are TeV
      if(abs(xmw-80.d-3).gt.10.d-3 .or. 
     #   abs(xmz-91.d-3).gt.10.d-3 )then
        write(*,*)'Error in setpvb: W and Z masses wrong'
        stop
      endif
      xnc = 3.d0
c Fermi constant, from PDG2002, in TeV^-2
      gf=1.16639d-5*1.d6
c Electron charge squared (average of the values at the V1 and V2 masses)
      alfaem2 = xalfaem(zmw2*1.d6)*xalfaem(zmz2*1.d6)
      ze2 = 4*pi*sqrt(alfaem2)
c sin and cos squared of theta_W; MSbar scheme, from PDG2003
      sin2w=0.23113d0
      cos2w=1-sin2w
      sw = dsqrt(sin2w)
c The quantities gup, gdown, ez, gw, gzzup, and gzzdown have beed defined
c by setting the positron charge equal to one (4*pi*alfaem-->1) starting
c from MC@NLO v3.1; accordingly, the cross sections are multiplied by
c e^4 in the main integration routines.
c Z couplings according to eq.(3.4) in NPB383(92)3 (WZ)
      tmp = sqrt(1.d0/(4*cos2w*sin2w))
      gvup = ( .5d0 - 4.d0/3.d0 * sin2w )*tmp
      gvdo = (-.5d0 + 2.d0/3.d0 * sin2w )*tmp
      gaup = ( .5d0 )*tmp
      gado = (-.5d0 )*tmp
      gup    = gvup+gaup
      gdown  = gvdo+gado
c WWZ vertex: (ez = gup - gdown in the SM) -- see eq.(3.8) in NPB383(92)3 (WZ)
      ez = sqrt(cos2w/sin2w)
c This is g_weak/(2*sqrt(2)) -- introduced because of the factor F_ij
c in eq.(3.5) in NPB383(92)3 (WZ)
      gw     = sqrt(1.d0/(8*sin2w))
c Combinations of weak couplings entering the ZZ cross section --
c see eq.(2.19) in NPB357(91)409 (ZZ)
      gzzup    = (gvup**4+gaup**4+6*gaup**2*gvup**2)/xnc
      gzzdn    = (gvdo**4+gado**4+6*gado**2*gvdo**2)/xnc
c CKM factors. Values from PDG 2003.
c Centers of the ranges given in eq.(11.2), supposedly taking unitarity
c into accout; with the following entries, it holds better than 0.1%
c     1 --> u,d
c     2 --> c,s
c     3 --> t,b
c
      xkm(1,1) = 0.9748d0
      xkm(1,2) = 0.2225d0
      xkm(1,3) = 0.0036d0
c
      xkm(2,1) = 0.2225d0
      xkm(2,2) = 0.9740d0
      xkm(2,3) = 0.041d0
c
      xkm(3,1) = 0.009d0
      xkm(3,2) = 0.0405d0
      xkm(3,3) = 0.9992d0
c
      do i=1,3
        do j=1,3
          xkm2(i,j)=xkm(i,j)**2
        enddo
      enddo
c Number of light flavours
      nl = 5
c
      if(idec.eq.0)then
c A and B coefficients, for charged leptons-Z couplings: ESW conventions
        vcel=-0.5d0+2*sin2w
        acel=-0.5d0
        aelupp=vcel**2+acel**2
        belupp=-2*vcel*acel
c A and B coefficients, for neutrinos-Z couplings: ESW conventions
        vcnu=0.5d0
        acnu=0.5d0
        anuupp=vcnu**2+acnu**2
        bnuupp=-2*vcnu*acnu
c Partial decay widths, up to lepton-dependent coefficients, to be
c inserted below. These will be used in the computations of BR's,
c where the total widths will be taken from input (if zero is entered,
c the code crashes here as a safety measure against unprotected
c operations in MadEvent routines)
        wdtzon=gf*xmz2*xmz/(6*sqrt(2.d0)*pi)
        wdtzmsb=xmz*xalfaem(xmz2*1.d6)/(12*sin2w*cos2w)
        wdtwon=gf*xmw2*xmw/(6*sqrt(2.d0)*pi)
        wdtwmsb=xmw*xalfaem(xmw2*1.d6)/(12*sin2w)
c Lepton masses: xlep#mass(i) is the mass of lepton # in the decay of
c vector boson i(=1,2). The identity of the lepton # is determined by
c the conventions of MadEvent -- see the routines xmadevXX
        tmpmss(1)=xme
        tmpmss(2)=xmmu
        tmpmss(3)=xmtau
        if(prdct.eq.'w+'.or.prdct.eq.'w-')then
c MadEvent inputs, given below to setmepar
          xmwme=xmw
          xmzme=xmz
          gawme=gaw
          gazme=gaz
c Change the following after implementing spin correlations
          if(prdct.eq.'w+')then
            ip4=0
            ip5=0
          else
            ip4=0
            ip5=0
          endif
          ip6=0
          ip7=0
c
          if(il1hw.le.3)then
            xlep1mass(1)=tmpmss(il1hw)
            xlep2mass(1)=0.d0
          else
            write(*,*)'Error in setpvb: inconsistent entries'
            stop
          endif
          if(imapz(il2hw).le.3)then
            dec2cf=aelupp
            xlep1mass(2)=tmpmss(imapz(il2hw))
            xlep2mass(2)=tmpmss(imapz(il2hw))
          else
            dec2cf=anuupp
            xlep1mass(2)=0.d0
            xlep2mass(2)=0.d0
          endif
          brrv1on=wdtwon/gaw
          brrv2on=dec2cf*wdtzon/gaz
          brrv1msb=wdtwmsb/gaw
          brrv2msb=dec2cf*wdtzmsb/gaz
        elseif(prdct.eq.'z ')then
c MadEvent inputs, given below to setmepar
          xmwme=xmw
          xmzme=xmz
          gawme=gaw
          gazme=gaz
c Change the following after implementing spin correlations
          ip4=0
          ip5=0
          ip6=0
          ip7=0
c
          if(imapz(il1hw).le.3)then
            dec1cf=aelupp
            xlep1mass(1)=tmpmss(imapz(il1hw))
            xlep2mass(1)=tmpmss(imapz(il1hw))
          else
            dec1cf=anuupp
            xlep1mass(1)=0.d0
            xlep2mass(1)=0.d0
          endif
          if(imapz(il2hw).le.3)then
            dec2cf=aelupp
            xlep1mass(2)=tmpmss(imapz(il2hw))
            xlep2mass(2)=tmpmss(imapz(il2hw))
          else
            dec2cf=anuupp
            xlep1mass(2)=0.d0
            xlep2mass(2)=0.d0
          endif
          brrv1on=dec1cf*wdtzon/gaz
          brrv2on=dec2cf*wdtzon/gaz
          brrv1msb=dec1cf*wdtzmsb/gaz
          brrv2msb=dec2cf*wdtzmsb/gaz
        elseif(prdct.eq.'ww')then
c MadEvent inputs, given below to setmepar
          xmwme=xmw
          xmzme=xmz
          gawme=gaw
          if(iwidth.eq.1)then
            gazme=gaz
          else
            gazme=0.d0
          endif
c
          ip4=-ichlw(il1hw)
          ip5=ineuw(il1hw)
          ip6=ichlw(il2hw)
          ip7=-ineuw(il2hw)
c
          if(il1hw.le.3)then
            xlep1mass(1)=tmpmss(il1hw)
            xlep2mass(1)=0.d0
          else
            write(*,*)'Error in setpvb: inconsistent entries'
            stop
          endif
          if(il2hw.le.3)then
            xlep1mass(2)=tmpmss(il2hw)
            xlep2mass(2)=0.d0
          else
            write(*,*)'Error in setpvb: inconsistent entries'
            stop
          endif
          brrv1on=wdtwon/gaw
          brrv2on=wdtwon/gaw
          brrv1msb=wdtwmsb/gaw
          brrv2msb=wdtwmsb/gaw
        else
          write(*,*) 'Error in setpvb: unknown process', prdct
          stop
        endif
      elseif(idec.eq.1)then
        if(prdct.eq.'w+')then
          ip4=24
          ip5=23
          ip6=0
          ip7=0
        elseif(prdct.eq.'w-')then
          ip4=-24
          ip5=23
          ip6=0
          ip7=0
        elseif(prdct.eq.'z ')then
          ip4=23
          ip5=23
          ip6=0
          ip7=0
        elseif(prdct.eq.'ww')then
          ip4=24
          ip5=-24
          ip6=0
          ip7=0
        else
          write(*,*) 'Error in setpvb: unknown process', prdct
          stop
        endif
      else
        write(*,*)'Error in setpvb: unknown idec',idec
        stop
      endif
c Fills MadEvent common blocks. Set positron charge and QCD coupling g 
c equal to one; the matrix elements factorize e^4 (*g^2 in the case of
c real emissions)
      call setmepar(xmwme,gawme,xmzme,gazme,sin2w,one,one)
c
      return
      end


      subroutine parsetpvb()
      implicit none
      integer i,j,jproc,itype
      character * 2 prc,prdct,xproc(3)
      common/process/prc,prdct
      common/cxproc/xproc
c ivbhpro returns the process number associated to the entries; this number
c is analogous to that relevant to photon production in herwig (41-->401,...)
      integer ivbhpro(2,2,3,10)
      common/civbhpro/ivbhpro
c idpX returns the flavour of parton number X (1=coming from the left,
c 2=coming from the right, 3=outgoing) in the process associated to the
c entries. The labelling scheme of PDG has been used
      integer idp1(2,2,3,10),idp2(2,2,3,10),idp3(2,2,3,10)
      common/cidpart/idp1,idp2,idp3
c Colour codes -- always trivial here
      integer iccode
      common/ciccode/iccode
c
      iccode=0
c
      xproc(1)='qq'
      xproc(2)='qg'
      xproc(3)='ag'
c
      do i=1,2
        do j=1,2
          do jproc=1,3
            do itype=1,10
              ivbhpro(i,j,jproc,itype)=0
              idp1(i,j,jproc,itype)=0
              idp2(i,j,jproc,itype)=0
              idp3(i,j,jproc,itype)=0
            enddo
          enddo
        enddo
      enddo
      if(prdct.eq.'w+')then
        do j=1,2
          do itype=1,10
            if(j.eq.1)then
              ivbhpro(1,j,1,itype)=401
              ivbhpro(1,j,2,itype)=402
              ivbhpro(1,j,3,itype)=404
            else
              ivbhpro(1,j,1,itype)=403
              ivbhpro(1,j,2,itype)=405
              ivbhpro(1,j,3,itype)=406
            endif
          enddo
        enddo
c
        idp1(1,1,1,1)=2
        idp1(1,1,1,2)=2
        idp1(1,1,1,3)=2
        idp1(1,1,1,4)=4
        idp1(1,1,1,5)=4
        idp1(1,1,1,6)=4
c
        idp1(1,2,1,1)=-1
        idp1(1,2,1,2)=-3
        idp1(1,2,1,3)=-5
        idp1(1,2,1,4)=-1
        idp1(1,2,1,5)=-3
        idp1(1,2,1,6)=-5
c
        idp1(1,1,2,1)=2
        idp1(1,1,2,2)=2
        idp1(1,1,2,3)=2
        idp1(1,1,2,4)=4
        idp1(1,1,2,5)=4
        idp1(1,1,2,6)=4
c
        idp1(1,2,2,1)=21
        idp1(1,2,2,2)=21
        idp1(1,2,2,3)=21
        idp1(1,2,2,4)=21
        idp1(1,2,2,5)=21
        idp1(1,2,2,6)=21
c
        idp1(1,1,3,1)=-1
        idp1(1,1,3,2)=-1
        idp1(1,1,3,3)=-3
        idp1(1,1,3,4)=-3
        idp1(1,1,3,5)=-5
        idp1(1,1,3,6)=-5
c
        idp1(1,2,3,1)=21
        idp1(1,2,3,2)=21
        idp1(1,2,3,3)=21
        idp1(1,2,3,4)=21
        idp1(1,2,3,5)=21
        idp1(1,2,3,6)=21
c
        idp2(1,1,1,1)=-1
        idp2(1,1,1,2)=-3
        idp2(1,1,1,3)=-5
        idp2(1,1,1,4)=-1
        idp2(1,1,1,5)=-3
        idp2(1,1,1,6)=-5
c
        idp2(1,2,1,1)=2
        idp2(1,2,1,2)=2
        idp2(1,2,1,3)=2
        idp2(1,2,1,4)=4
        idp2(1,2,1,5)=4
        idp2(1,2,1,6)=4
c
        idp2(1,1,2,1)=21
        idp2(1,1,2,2)=21
        idp2(1,1,2,3)=21
        idp2(1,1,2,4)=21
        idp2(1,1,2,5)=21
        idp2(1,1,2,6)=21
c
        idp2(1,2,2,1)=2
        idp2(1,2,2,2)=2
        idp2(1,2,2,3)=2
        idp2(1,2,2,4)=4
        idp2(1,2,2,5)=4
        idp2(1,2,2,6)=4
c
        idp2(1,1,3,1)=21
        idp2(1,1,3,2)=21
        idp2(1,1,3,3)=21
        idp2(1,1,3,4)=21
        idp2(1,1,3,5)=21
        idp2(1,1,3,6)=21
c
        idp2(1,2,3,1)=-1
        idp2(1,2,3,2)=-1
        idp2(1,2,3,3)=-3
        idp2(1,2,3,4)=-3
        idp2(1,2,3,5)=-5
        idp2(1,2,3,6)=-5
c
        idp3(1,1,1,1)=21
        idp3(1,1,1,2)=21
        idp3(1,1,1,3)=21
        idp3(1,1,1,4)=21
        idp3(1,1,1,5)=21
        idp3(1,1,1,6)=21
c
        idp3(1,2,1,1)=21
        idp3(1,2,1,2)=21
        idp3(1,2,1,3)=21
        idp3(1,2,1,4)=21
        idp3(1,2,1,5)=21
        idp3(1,2,1,6)=21
c
        idp3(1,1,2,1)=1
        idp3(1,1,2,2)=3
        idp3(1,1,2,3)=5
        idp3(1,1,2,4)=1
        idp3(1,1,2,5)=3
        idp3(1,1,2,6)=5
c
        idp3(1,2,2,1)=1
        idp3(1,2,2,2)=3
        idp3(1,2,2,3)=5
        idp3(1,2,2,4)=1
        idp3(1,2,2,5)=3
        idp3(1,2,2,6)=5
c
        idp3(1,1,3,1)=-2
        idp3(1,1,3,2)=-4
        idp3(1,1,3,3)=-2
        idp3(1,1,3,4)=-4
        idp3(1,1,3,5)=-2
        idp3(1,1,3,6)=-4
c
        idp3(1,2,3,1)=-2
        idp3(1,2,3,2)=-4
        idp3(1,2,3,3)=-2
        idp3(1,2,3,4)=-4
        idp3(1,2,3,5)=-2
        idp3(1,2,3,6)=-4
      elseif(prdct.eq.'w-')then
        do j=1,2
          do itype=1,10
            if(j.eq.1)then
              ivbhpro(1,j,1,itype)=401
              ivbhpro(1,j,2,itype)=402
              ivbhpro(1,j,3,itype)=404
            else
              ivbhpro(1,j,1,itype)=403
              ivbhpro(1,j,2,itype)=405
              ivbhpro(1,j,3,itype)=406
            endif
          enddo
        enddo
c
        idp1(1,1,1,1)=1
        idp1(1,1,1,2)=3
        idp1(1,1,1,3)=5
        idp1(1,1,1,4)=1
        idp1(1,1,1,5)=3
        idp1(1,1,1,6)=5
c
        idp1(1,2,1,1)=-2
        idp1(1,2,1,2)=-2
        idp1(1,2,1,3)=-2
        idp1(1,2,1,4)=-4
        idp1(1,2,1,5)=-4
        idp1(1,2,1,6)=-4
c
        idp1(1,1,2,1)=1
        idp1(1,1,2,2)=1
        idp1(1,1,2,3)=3
        idp1(1,1,2,4)=3
        idp1(1,1,2,5)=5
        idp1(1,1,2,6)=5
c
        idp1(1,2,2,1)=21
        idp1(1,2,2,2)=21
        idp1(1,2,2,3)=21
        idp1(1,2,2,4)=21
        idp1(1,2,2,5)=21
        idp1(1,2,2,6)=21
c
        idp1(1,1,3,1)=-2
        idp1(1,1,3,2)=-2
        idp1(1,1,3,3)=-2
        idp1(1,1,3,4)=-4
        idp1(1,1,3,5)=-4
        idp1(1,1,3,6)=-4
c
        idp1(1,2,3,1)=21
        idp1(1,2,3,2)=21
        idp1(1,2,3,3)=21
        idp1(1,2,3,4)=21
        idp1(1,2,3,5)=21
        idp1(1,2,3,6)=21
c
        idp2(1,1,1,1)=-2
        idp2(1,1,1,2)=-2
        idp2(1,1,1,3)=-2
        idp2(1,1,1,4)=-4
        idp2(1,1,1,5)=-4
        idp2(1,1,1,6)=-4
c
        idp2(1,2,1,1)=1
        idp2(1,2,1,2)=3
        idp2(1,2,1,3)=5
        idp2(1,2,1,4)=1
        idp2(1,2,1,5)=3
        idp2(1,2,1,6)=5
c
        idp2(1,1,2,1)=21
        idp2(1,1,2,2)=21
        idp2(1,1,2,3)=21
        idp2(1,1,2,4)=21
        idp2(1,1,2,5)=21
        idp2(1,1,2,6)=21
c
        idp2(1,2,2,1)=1
        idp2(1,2,2,2)=1
        idp2(1,2,2,3)=3
        idp2(1,2,2,4)=3
        idp2(1,2,2,5)=5
        idp2(1,2,2,6)=5
c
        idp2(1,1,3,1)=21
        idp2(1,1,3,2)=21
        idp2(1,1,3,3)=21
        idp2(1,1,3,4)=21
        idp2(1,1,3,5)=21
        idp2(1,1,3,6)=21
c
        idp2(1,2,3,1)=-2
        idp2(1,2,3,2)=-2
        idp2(1,2,3,3)=-2
        idp2(1,2,3,4)=-4
        idp2(1,2,3,5)=-4
        idp2(1,2,3,6)=-4
c
        idp3(1,1,1,1)=21
        idp3(1,1,1,2)=21
        idp3(1,1,1,3)=21
        idp3(1,1,1,4)=21
        idp3(1,1,1,5)=21
        idp3(1,1,1,6)=21
c
        idp3(1,2,1,1)=21
        idp3(1,2,1,2)=21
        idp3(1,2,1,3)=21
        idp3(1,2,1,4)=21
        idp3(1,2,1,5)=21
        idp3(1,2,1,6)=21
c
        idp3(1,1,2,1)=2
        idp3(1,1,2,2)=4
        idp3(1,1,2,3)=2
        idp3(1,1,2,4)=4
        idp3(1,1,2,5)=2
        idp3(1,1,2,6)=4
c
        idp3(1,2,2,1)=2
        idp3(1,2,2,2)=4
        idp3(1,2,2,3)=2
        idp3(1,2,2,4)=4
        idp3(1,2,2,5)=2
        idp3(1,2,2,6)=4
c
        idp3(1,1,3,1)=-1
        idp3(1,1,3,2)=-3
        idp3(1,1,3,3)=-5
        idp3(1,1,3,4)=-1
        idp3(1,1,3,5)=-3
        idp3(1,1,3,6)=-5
c
        idp3(1,2,3,1)=-1
        idp3(1,2,3,2)=-3
        idp3(1,2,3,3)=-5
        idp3(1,2,3,4)=-1
        idp3(1,2,3,5)=-3
        idp3(1,2,3,6)=-5
      elseif(prdct.eq.'z ')then
        do j=1,2
          do itype=1,10
            if(j.eq.1)then
              ivbhpro(1,j,1,itype)=401
              if(itype.le.5)then
                ivbhpro(1,j,2,itype)=402
              else
                ivbhpro(1,j,2,itype)=404
              endif
            else
              ivbhpro(1,j,1,itype)=403
              if(itype.le.5)then
                ivbhpro(1,j,2,itype)=405
              else
                ivbhpro(1,j,2,itype)=406
              endif
            endif
          enddo
        enddo
c
        idp1(1,1,1,1)=2
        idp1(1,1,1,2)=1
        idp1(1,1,1,3)=3
        idp1(1,1,1,4)=4
        idp1(1,1,1,5)=5
c
        idp1(1,2,1,1)=-2
        idp1(1,2,1,2)=-1
        idp1(1,2,1,3)=-3
        idp1(1,2,1,4)=-4
        idp1(1,2,1,5)=-5
c
        idp1(1,1,2,1)=2
        idp1(1,1,2,2)=1
        idp1(1,1,2,3)=3
        idp1(1,1,2,4)=4
        idp1(1,1,2,5)=5
        idp1(1,1,2,6)=-2
        idp1(1,1,2,7)=-1
        idp1(1,1,2,8)=-3
        idp1(1,1,2,9)=-4
        idp1(1,1,2,10)=-5
c
        idp1(1,2,2,1)=21
        idp1(1,2,2,2)=21
        idp1(1,2,2,3)=21
        idp1(1,2,2,4)=21
        idp1(1,2,2,5)=21
        idp1(1,2,2,6)=21
        idp1(1,2,2,7)=21
        idp1(1,2,2,8)=21
        idp1(1,2,2,9)=21
        idp1(1,2,2,10)=21
c
        idp2(1,1,1,1)=-2
        idp2(1,1,1,2)=-1
        idp2(1,1,1,3)=-3
        idp2(1,1,1,4)=-4
        idp2(1,1,1,5)=-5
c
        idp2(1,2,1,1)=2
        idp2(1,2,1,2)=1
        idp2(1,2,1,3)=3
        idp2(1,2,1,4)=4
        idp2(1,2,1,5)=5
c
        idp2(1,1,2,1)=21
        idp2(1,1,2,2)=21
        idp2(1,1,2,3)=21
        idp2(1,1,2,4)=21
        idp2(1,1,2,5)=21
        idp2(1,1,2,6)=21
        idp2(1,1,2,7)=21
        idp2(1,1,2,8)=21
        idp2(1,1,2,9)=21
        idp2(1,1,2,10)=21
c
        idp2(1,2,2,1)=2 
        idp2(1,2,2,2)=1 
        idp2(1,2,2,3)=3 
        idp2(1,2,2,4)=4 
        idp2(1,2,2,5)=5 
        idp2(1,2,2,6)=-2
        idp2(1,2,2,7)=-1
        idp2(1,2,2,8)=-3
        idp2(1,2,2,9)=-4
        idp2(1,2,2,10)=-5
c
        do j=1,2
          do jproc=1,2
            do itype=1,10
              if(idp1(1,j,jproc,itype).eq.21)then
                idp3(1,j,jproc,itype)=idp2(1,j,jproc,itype)
              elseif(idp2(1,j,jproc,itype).eq.21)then
                idp3(1,j,jproc,itype)=idp1(1,j,jproc,itype)
              elseif(idp1(1,j,jproc,itype).ne.0 .and.
     #               idp2(1,j,jproc,itype).ne.0)then
                idp3(1,j,jproc,itype)=21
              else
                idp3(1,j,jproc,itype)=0
              endif
            enddo
          enddo
        enddo
      elseif(prdct.eq.'ww')then
        do i=1,2
          do j=1,2
            do itype=1,10
              if(j.eq.1)then
                ivbhpro(i,j,1,itype)=401
                ivbhpro(i,j,2,itype)=402
                ivbhpro(i,j,3,itype)=404
              else
                ivbhpro(i,j,1,itype)=403
                ivbhpro(i,j,2,itype)=405
                ivbhpro(i,j,3,itype)=406
              endif
            enddo
          enddo
        enddo
c
        idp1(1,1,1,1)=2
        idp1(1,1,1,2)=4
        idp1(1,2,1,1)=-2
        idp1(1,2,1,2)=-4
        idp1(2,1,1,1)=1
        idp1(2,1,1,2)=3
        idp1(2,1,1,3)=5
        idp1(2,2,1,1)=-1
        idp1(2,2,1,2)=-3
        idp1(2,2,1,3)=-5
c
        idp1(1,1,2,1)=2
        idp1(1,1,2,2)=4
        idp1(1,2,2,1)=21
        idp1(1,2,2,2)=21
        idp1(2,1,2,1)=1
        idp1(2,1,2,2)=3
        idp1(2,1,2,3)=5
        idp1(2,2,2,1)=21
        idp1(2,2,2,2)=21
        idp1(2,2,2,3)=21
c
        idp1(1,1,3,1)=-2
        idp1(1,1,3,2)=-4
        idp1(1,2,3,1)=21
        idp1(1,2,3,2)=21
        idp1(2,1,3,1)=-1
        idp1(2,1,3,2)=-3
        idp1(2,1,3,3)=-5
        idp1(2,2,3,1)=21
        idp1(2,2,3,2)=21
        idp1(2,2,3,3)=21
c
        idp2(1,1,1,1)=-2
        idp2(1,1,1,2)=-4
        idp2(1,2,1,1)=2
        idp2(1,2,1,2)=4
        idp2(2,1,1,1)=-1
        idp2(2,1,1,2)=-3
        idp2(2,1,1,3)=-5
        idp2(2,2,1,1)=1
        idp2(2,2,1,2)=3
        idp2(2,2,1,3)=5
c
        idp2(1,1,2,1)=21
        idp2(1,1,2,2)=21
        idp2(1,2,2,1)=2
        idp2(1,2,2,2)=4
        idp2(2,1,2,1)=21
        idp2(2,1,2,2)=21
        idp2(2,1,2,3)=21
        idp2(2,2,2,1)=1
        idp2(2,2,2,2)=3
        idp2(2,2,2,3)=5
c
        idp2(1,1,3,1)=21
        idp2(1,1,3,2)=21
        idp2(1,2,3,1)=-2
        idp2(1,2,3,2)=-4
        idp2(2,1,3,1)=21
        idp2(2,1,3,2)=21
        idp2(2,1,3,3)=21
        idp2(2,2,3,1)=-1
        idp2(2,2,3,2)=-3
        idp2(2,2,3,3)=-5
c
        do i=1,2
          do j=1,2
            do jproc=1,3
              do itype=1,10
                if(idp1(i,j,jproc,itype).eq.21)then
                  idp3(i,j,jproc,itype)=idp2(i,j,jproc,itype)
                elseif(idp2(i,j,jproc,itype).eq.21)then
                  idp3(i,j,jproc,itype)=idp1(i,j,jproc,itype)
                elseif(idp1(i,j,jproc,itype).ne.0 .and.
     #                 idp2(i,j,jproc,itype).ne.0)then
                  idp3(i,j,jproc,itype)=21
                else
                  idp3(i,j,jproc,itype)=0
                endif
              enddo
            enddo
          enddo
        enddo
      else
        write(*,*)'Fatal error in parsetpvb: unknown type',prdct
        stop
      endif
      call parcheckpvb()
      return
      end


      subroutine parcheckpvb()
      implicit real * 8(a-h,o-z)
      parameter (iallzero=0)
      common/civbhpro/ivbhpro(2,2,3,10)
      common/cidpart/idp1(2,2,3,10),idp2(2,2,3,10),idp3(2,2,3,10)
c
      call parcheckinvb()
      do i=1,2
        do j=1,2
          do jproc=1,3
            do itype=1,10
              ihpro=ivbhpro(i,j,jproc,itype)
              i1=idp1(i,j,jproc,itype)
              i2=idp2(i,j,jproc,itype)
              i3=idp3(i,j,jproc,itype)
              call parcheckfvb(ihpro,i1,i2,i3,iallzero)
            enddo
          enddo
        enddo
      enddo
      return
      end


      subroutine parcheckfvb(ihpro,i1,i2,i3,iallzero)
      implicit real * 8(a-h,o-z)
      parameter (tiny=1.d-8)
      logical ferror
      common/ccharges/chrg(-5:21),chprdct
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
        write(*,*)'parcheckfvb called improperly'
        stop
      endif
      if(isum.ne.0)then
c charge must be conserved
        if(abs(chall).gt.tiny)ferror=.true.
c 401 is qqbar
        if( ihpro.eq.401 .and.
     #      (i1.le.0 .or. i2.ge.0 .or.
     #       i3.ne.21) )ferror=.true.
c 402 is qg
        if( ihpro.eq.402 .and.
     #      (i1.le.0 .or. i2.ne.21 .or. 
     #       i3.le.0) )ferror=.true.
c 403 is qqbar
        if( ihpro.eq.403 .and.
     #      (i1.ge.0 .or. i2.le.0 .or. 
     #       i3.ne.21) )ferror=.true.
c 404 is qbarg
        if( ihpro.eq.404 .and.
     #      (i1.ge.0 .or. i2.ne.21 .or. 
     #       i3.ge.0) )ferror=.true.
c 405 is gq
        if( ihpro.eq.405 .and.
     #      (i1.ne.21 .or. i2.le.0 .or. 
     #       i3.le.0) )ferror=.true.
c 406 is gqbar
        if( ihpro.eq.406 .and.
     #      (i1.ne.21 .or. i2.ge.0 .or. 
     #       i3.ge.0) )ferror=.true.
      endif
      if(ferror)then
        write(*,*)'Error in parcheckfvb'
        write(*,*)'ihpro,i1,i2,i3:',ihpro,i1,i2,i3
        stop
      endif
      return
      end


      subroutine parcheckinvb()
      implicit real * 8(a-h,o-z)
      parameter (chup=2.d0/3.d0)
      parameter (chdn=-1.d0/3.d0)
      character * 2 prc,prdct
      common/process/prc,prdct
      common/ccharges/chrg(-5:21),chprdct
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
      if(prdct.eq.'w+') then
        chprdct=1.d0
      elseif(prdct.eq.'w-') then
        chprdct=-1.d0
      elseif(prdct.eq.'z ') then
        chprdct=0.d0
      elseif(prdct.eq.'ww') then
        chprdct=0.d0
      else
        write(*,*)'parcheckinvb: non implemented final state ', prdct
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
      subroutine whichovb(iseed,itot,mx_of_evta,mx_of_evtb,iunit)
      implicit real * 8 (a-h,o-z)
c Determines the type of event at random
      parameter (tiny=1.d-4)
      logical flag
      integer mx_of_evta(3),mx_of_evtb(3)
      dimension xpa(3),xpb(3)
c
      if(itot.le.0)then
        write(6,*)'fatal error #1 in whichovb'
        stop
      endif
      do jproc=1,3
        xpa(jproc)=dfloat(mx_of_evta(jproc))/dfloat(itot)
        xpb(jproc)=dfloat(mx_of_evtb(jproc))/dfloat(itot)
      enddo
      one=0.d0
      do jproc=1,3
        one=one+xpa(jproc)+xpb(jproc)
      enddo
      if(abs(one-1.d0).gt.tiny)then
        write(6,*)'probability not normalized'
        stop
      endif
      i0=0
      flag=.true.
      xsum=0.d0
      rnd=fk88random(iseed)
      do while(flag)
        if(i0.gt.6)then
          write(6,*)'fatal error #2 in whichovb'
          stop
        endif
        i0=i0+1
        prob=xpa(i0)
        if(i0.gt.3)prob=xpb(i0-3)
        xsum=xsum+prob
        if(rnd.lt.xsum)then
          flag=.false.
          itot=itot-1
          if(i0.le.3)then
            mx_of_evta(i0)=mx_of_evta(i0)-1
          else
            mx_of_evtb(i0-3)=mx_of_evtb(i0-3)-1
          endif
          iunit=20+i0
        endif
      enddo
      return
      end


      subroutine crosschevb(itot,mx_of_evta,mx_of_evtb)
      implicit real * 8 (a-h,o-z)
c Checks whether whichovb did it right
      integer mx_of_evta(3),mx_of_evtb(3)
c
      if(itot.ne.0)then
        write(6,*)'Error: itot=',itot
        stop
      endif
      do i=1,3
        if(mx_of_evta(i).ne.0)then
          write(6,*)'Error: mx_of_evta(',i,')=',mx_of_evta(i)
          stop
        endif
        if(mx_of_evtb(i).ne.0)then
          write(6,*)'Error: mx_of_evtb(',i,')=',mx_of_evtb(i)
          stop
        endif
      enddo
      return
      end


      subroutine retrieve_evenvb(iunit,ii,dummy)
c Reads from disk the complete information on the events; see store_evenvb
c for the conventions used
      implicit none
      integer iunit,ii,i,j
      real*8 dummy
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5,ip6,ip7
      common/ci2part/ip4,ip5,ip6,ip7
      integer iccode
      common/ciccode/iccode
      integer idec
      common/cidec/idec
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
      real*8 xmom_lb(9,4)
      common/cxmomlb/xmom_lb
c
      read(iunit,901,end=997,err=998)i1hpro,iccode,np
      if(idec.eq.0)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4,ip5,ip6,ip7
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,3),
     #    ((xmom_lb(i,j),j=1,4),i=6,9)
      elseif(idec.eq.1)then
        read(iunit,902,end=997,err=998)ip1,ip2,ip3,ip4,ip5
        read(iunit,903,end=997,err=998)xevsign
        read(iunit,904,end=997,err=998)((xmom_lb(i,j),j=1,4),i=1,5)
      endif
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
      write(77,902)ip1,ip2,ip3,ip4,ip5,ip6,ip7
      write(77,903)xevsign
      write(77,904)((xmom_lb(i,j),j=1,4),i=1,9)
      stop
 999  continue
      return
      end


      subroutine store_evenvb(iunit,xpmone)
c Stores on disk the complete information on the events. Starting
c from version 3.1, each event has the following format:
c       IPR, IC, NP
c      (ID(I),I=1,NP)
c      ((P(J,I),J=1,4),I=1,NP)
c where IPR is the subprocess code (i1hpro), IC is the colour code
c (iccode, trivial here), NP is the number of partons entering the 
c reaction (thus, this includes the soft parton in the case of S events),
c ID(I) are the particle identities (ip1,...,ip7 here), and P(J,I) are 
c the particles four momenta in the lab frame (P(J,I)=xmom_lb(i,j) here).
c
c This routine is called with xpmone=1 when events are obtained from
c SPRING, and with xpmone=-1 after the events are read from the temporary
c files (via retrieve_evenvb), to be stored in the final event file.
c When xpmone=1, one has xevsign=+1/-1, and the weight of the event is 
c xevsign*wgt[a,b]ev*BR, where BR=1 in the case of undecayed vector
c bosons, and the product of the relevant branching ratios otherwise.
c When xpmone=-1, then xevsign is the weight of the event. Furthermore, 
c the momenta are expressed in TeV when xpmone=1, and in GeV when xpmone=-1
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
c X being either the VV or the llbarrrbar system here (thus, 407 is unused).
c ipX is the parton code relevant to parton # X. PDG conventions are
c used: 1=d, 2=u, 3=s, 4=c, 5=b, 21=g
      implicit none
      integer iunit,i,j
      real*8 xpmone,xevwgt,xfact,brfact
      integer i1hpro
      common/ci1hpro/i1hpro
      integer ip1,ip2,ip3
      common/ci1part/ip1,ip2,ip3
      integer ip4,ip5,ip6,ip7
      common/ci2part/ip4,ip5,ip6,ip7
      integer iccode
      common/ciccode/iccode
      integer idec
      common/cidec/idec
      integer np
      common/cnp/np
      real*8 xevsign
      common/cxevsign/xevsign
c xmom_lb(i,j) is the j component of the four vector of the particle # i,
c given in the laboratory frame. j=4 is the energy for MC@NLO versions
c up to 2.31, the mass for version 3.1 onwards. i=1,2 are the incoming
c partons, 3 is the outgoing parton, 4 is V1, 5 is V2, 6 and 7 are the
c leptons from the decay of V1, 8 and 9 are the leptons from the decay of V2.
c See xmadevww() for the fermion/antifermion assignments in the case of 
c WW production.
c Momentum conservation is (1+2)-(3+4+5)=0 or (1+2)-(3+6+7+8+9)=0 
      real*8 xmom_lb(9,4)
      common/cxmomlb/xmom_lb
      integer iwgtnorm
      common/ciwgtnorm/iwgtnorm
      real*8 wgtaev,wgtbev
      common/cwgtev/wgtaev,wgtbev
c Branching ratios, to be inserted in the case of decayed vector bosons
      real * 8 brrv1on,brrv2on,brrv1msb,brrv2msb
      common/brratios/brrv1on,brrv2on,brrv1msb,brrv2msb
c
      if(xpmone.eq.-1)then
c Events are already stored in temporary files, and are passed to this
c routines through common blocks filled by retrieve_evenvb
        xevwgt=xevsign
        xfact=1.d3
      elseif(xpmone.eq.1)then
c Events are obtained from SPRING, and are written to temporary files
c for the first time
        if(idec.eq.0)then
          np=7
          brfact=brrv1msb*brrv2msb
        elseif(idec.eq.1)then
          np=5
          brfact=1.d0
        else
          write(6,*) 'Error in store_evenvb: idec=',idec
          stop
        endif
        if(xmom_lb(3,4).eq.0.d0)then
          xevwgt=xevsign*wgtbev*brfact
        else
          xevwgt=xevsign*wgtaev*brfact
        endif
        xfact=1.d0
      else
        write(*,*)'Fatal error in store_evenvb: xpmone=',xpmone
        stop
      endif
      write(iunit,901)i1hpro,iccode,np
      if(idec.eq.0)then
        write(iunit,902)ip1,ip2,ip3,ip4,ip5,ip6,ip7
        write(iunit,903)xevwgt
        write(iunit,904)((xfact*xmom_lb(i,j),j=1,4),i=1,3),
     #                  ((xfact*xmom_lb(i,j),j=1,4),i=6,9)
      elseif(idec.eq.1)then
        write(iunit,902)ip1,ip2,ip3,ip4,ip5
        write(iunit,903)xevwgt
        write(iunit,904)((xfact*xmom_lb(i,j),j=1,4),i=1,5)
      endif
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
c Running couplings
c
c
      function zgmu2_nvb()
c Sets the desired factorization scale and returns the strong coupling squared
c To be called is association to pure NLO terms
      implicit real * 8 (a-h,o-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/xmumc/xmcfct2,xmcren2
      common/zmass/zmw,zmz,zmw2,zmz2
      common/ycmvar/yw,yz,yp,yl1,yl2,yr1,yr2
      common/perpen/pw(2),pz(2),pp(2),ptvl1(2),ptvl2(2),
     # ptvr1(2),ptvr2(2)
      common/scalef/sclfct,sclren
      common/fk88lambda/xlam
      common/nl/nl
      parameter (pi=3.14159265358979312D0)
c  momenta
      ptw = dsqrt(pw(1)**2+pw(2)**2)
      ptz = dsqrt(pz(1)**2+pz(2)**2)
      tmw = dsqrt(ptw**2+zmw2)
      tmz = dsqrt(ptz**2+zmz2)
      plw = tmw * sinh(yw)
      plz = tmz * sinh(yz)
      enw = tmw * cosh(yw)
      enz = tmz * cosh(yz)
c
      xmu2 = (tmw**2+tmz**2)/2
      xmufct2  = sclfct**2 * xmu2
      xmuren2  = sclren**2 * xmu2
      as    = alfas(xmuren2*1.d6,xlam,nl)
      zgmu2_nvb = 4.d0*pi*as
      end


      function zgmu2_vb()
c Sets the desired factorization scale and returns the strong coupling squared
c To be called is association to MC subtraction terms
      implicit real * 8 (a-h,o-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/xmumc/xmcfct2,xmcren2
      common/zmass/zmw,zmz,zmw2,zmz2
      common/ycmvar/yw,yz,yp,yl1,yl2,yr1,yr2
      common/perpen/pw(2),pz(2),pp(2),ptvl1(2),ptvl2(2),
     # ptvr1(2),ptvr2(2)
      common/scalemcf/sclmcfct,sclmcren
      common/fk88lambda/xlam
      common/nl/nl
      parameter (pi=3.14159265358979312D0)
c  momenta
      ptw = dsqrt(pw(1)**2+pw(2)**2)
      ptz = dsqrt(pz(1)**2+pz(2)**2)
      tmw = dsqrt(ptw**2+zmw2)
      tmz = dsqrt(ptz**2+zmz2)
      plw = tmw * sinh(yw)
      plz = tmz * sinh(yz)
      enw = tmw * cosh(yw)
      enz = tmz * cosh(yz)
c
      xmu2 = (tmw**2+tmz**2)/2
      xmcfct2  = sclmcfct**2 * xmu2
      xmcren2  = sclmcren**2 * xmu2
      as    = alfas(xmcren2*1.d6,xlam,nl)
      zgmu2_vb = 4.d0*pi*as
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
c
c
c End of running couplings
c
c
