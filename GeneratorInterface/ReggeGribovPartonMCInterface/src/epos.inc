c---------------------------------------------------------------------------
c                         dimensions
c---------------------------------------------------------------------------

      integer   mmry,mxptl,nmxhep,myptl,nzeta,nflav,mxstr,mystr,mxtau
     *         ,mxtrig,mxpri,mxbins,matau,mxnucl,mxhisarg,idxD0,idxD1
     *         ,idxD,nclha,nclegy,mamxx,mxjerr,mxvol,mxeps,mxidco,mxcoox
     *         ,mxcooy
      parameter (mmry=1)   !memory saving factor

      parameter (mxptl=200000/mmry) !max nr of particles in epos ptl list
      parameter (nmxhep=99900)       !max nr of particles in hep ptl list
      parameter (myptl=1000)        !max nr of droplets in epos ptl list
      parameter (nzeta=60)          !max nr of zeta bins for droplets
      parameter (nflav=6)           !max nr of flavors
      parameter (mxstr=20000/mmry)  !max nr of strings in epos string list 
      parameter (mystr=20000/mmry)    
      parameter (mxtau=4,mxvol=10,mxeps=16)
      parameter (mxtrig=99,mxidco=99)
      parameter (mxpri=200) 
      parameter (mxbins=10000)
      parameter (matau=10,mxcoox=40,mxcooy=10)
      parameter (mxnucl=20)
      parameter (mxhisarg=100)
      parameter (idxD0=0,idxD1=2,idxD=1,nclha=4,nclegy=100) 
      parameter (mamxx=250)
      parameter (mxjerr=10)
           
c---------------------------------------------------------------------------
c                   epos event common block
c---------------------------------------------------------------------------

      real        phievt,bimevt,pmxevt,egyevt
     *,xbjevt,qsqevt,zppevt,zptevt
      integer     nevt,kolevt,koievt,kohevt,npjevt
     *,ntgevt,npnevt,nppevt,ntnevt,ntpevt,jpnevt,jppevt,jtnevt,jtpevt
     *,nglevt,minfra,maxfra
      common/cevt/phievt,nevt,bimevt,kolevt,koievt,pmxevt,egyevt,npjevt
     *,ntgevt,npnevt,nppevt,ntnevt,ntpevt,jpnevt,jppevt,jtnevt,jtpevt
     *,xbjevt,qsqevt,nglevt,zppevt,zptevt,minfra,maxfra,kohevt
      real         rglevt,sglevt,eglevt,fglevt,typevt
      integer      ng1evt,ng2evt,ikoevt
      common/c2evt/ng1evt,ng2evt,rglevt,sglevt,eglevt,fglevt,ikoevt
     *,typevt            !qgsjet03.f and qgsjetII.f and phojet.f
           
c     nevt .......... error code. 1=valid event, 0=invalid event
c     bimevt ........ absolute value of impact parameter
c     phievt ........ angle of impact parameter
c     kolevt ........ number of collisions
c     koievt ........ number of inelastic collisions
c     kohevt ........ number of hard collisions
c     pmxevt ........ reference momentum
c     egyevt ........ pp cm energy (hadron) or string energy (lepton)
c     npjevt ........ number of primary projectile participants
c     ntgevt ........ number of primary target participants
c     npnevt ........ number of primary projectile neutron spectators
c     nppevt ........ number of primary projectile proton spectators
c     ntnevt ........ number of primary target neutron spectators
c     ntpevt ........ number of primary target proton spectators
c     jpnevt ........ number of absolute projectile neutron spectators
c     jppevt ........ number of absolute projectile proton spectators
c     jtnevt ........ number of absolute target neutron spectators
c     jtpevt ........ number of absolute target proton spectators
c     xbjevt ........ bjorken x for dis     
c     qsqevt ........ q**2 for dis  
c     sigtot ........ total cross section
c     nglevt ........ number of collisions acc to  Glauber 
c     zppevt ........ average Z-parton-proj 
c     zptevt ........ average Z-parton-targ
c     ng1evt ........ number of Glauber participants with at least one IAs 
c     ng2evt ........ number of Glauber participants with at least two IAs 
c     ikoevt ........ number of elementary parton-parton scatterings
c     typevt ........ type of event (1=Non Diff, 2=Double Diff, 3=Central Diff, 4=AB->XB, -4=AB->AX)

c---------------------------------------------------------------------------
c                   epos particle list common block
c---------------------------------------------------------------------------
      
      real        pptl,tivptl,xorptl
      integer     nptl,iorptl,idptl,istptl,ifrptl,jorptl,ibptl,ityptl
      common/cptl/nptl,pptl(5,mxptl),iorptl(mxptl),idptl(mxptl)
     *,istptl(mxptl),tivptl(2,mxptl),ifrptl(2,mxptl),jorptl(mxptl)
     *,xorptl(4,mxptl),ibptl(4,mxptl),ityptl(mxptl)
      real         ekievt
      integer      itsptl
      common/c1ptl/ekievt,itsptl(mxptl)
     
c     nptl .......... current particle index (=number of ptls stored)
c     idptl(i) ...... particle id
c     pptl(1,i) ..... x-component of particle momentum 
c     pptl(2,i) ..... y-component of particle momentum 
c     pptl(3,i) ..... z-component of particle momentum 
c     pptl(4,i) ..... particle energy 
c     pptl(5,i) ..... particle mass 
c     iorptl(i) ..... particle number of father (if .le. 0 : no father) 
c     jorptl(i) ..... particle number of mother (if .le. 0 : no mother)
c     istptl(i) ..... status: 40 and 41 : Remnant
c                             30 and 31 : Pomeron
c                             20 and 21 : Parton
c                             10 and 11 : Droplet
c                             00 and 01 : Particle
c                            last digit = 0 : last generation
c                            last digit = 1 : not last generation
c     xorptl(1,i) ... x-component of formation point
c     xorptl(2,i) ... y-component of formation point
c     xorptl(3,i) ... z-component of formation point
c     xorptl(4,i) ... formation time
c     tivptl(1,i) ... formation time (always in the pp-cms!)
c     tivptl(2,i) ... destruction time (always in the pp-cms!)
c     ityptl(i)  .... type of particles origin:
c                         10-19: target
c                         20-29: soft Pom
c                         30-39: hard Pom 
c                         40-49: projectile 
c                         50: string, droplet
c     itsptl(i) ..... string type of particles origin (if string)  

      real         radptl
      integer      iaaptl
      common/c2ptl/iaaptl(mxptl),radptl(mxptl)
      real         desptl,dezptl
      common/c3ptl/desptl(mxptl),dezptl(mxptl)
      integer      nptlbd
      common/c4ptl/nptlbd
      real         rinptl,vrad
      integer      inbxxx
      common/c6ptl/rinptl(mxptl),vrad,inbxxx
      real         qsqptl,zpaptl
      common/c8ptl/qsqptl(mxptl),zpaptl(2,mxptl)


c---------------------------------------------------------------------------
c                   hep standard event commonblock.
c---------------------------------------------------------------------------

      double precision phep,vhep
      integer       nevhep,nhep,isthep,idhep,jmohep,jdahep

      common/hepcom/nevhep,nhep,isthep(nmxhep),idhep(nmxhep),
     &jmohep(2,nmxhep),jdahep(2,nmxhep),phep(5,nmxhep),vhep(4,nmxhep)
     
c---------------------------------------------------------------------------
c
c         nevhep      -   event number
c         nhep        -   number of entries in the event record
c
c         isthep(i)   -   status code
c         idhep(i)    -   particle id (particle data group standard)
c
c         jmohep(1,i) -   position of mother particle in list
c         jmohep(2,i) -   position of second mother particle in list
c         jdahep(1,i) -   position of first daughter in list
c         jdahep(2,i) -   position of first daughter in list
c
c         phep(1,i)   -   p_x momentum in gev/c
c         phep(2,i)   -   p_y momentum in gev/c
c         phep(3,i)   -   p_z momentum in gev/c
c         phep(4,i)   -   energy in gev
c         phep(5,i)   -   mass in gev/c**2
c
c         vhep(1,i)   -   x position of production vertex in mm 
c         vhep(2,i)   -   y position of production vertex in mm 
c         vhep(3,i)   -   z position of production vertex in mm 
c         vhep(4,i)   -   time of production  in mm/c 
c
c          (note:  1 mm = 10^-12 fm = 5.07 10^-12 1/gev)

c------------------------------------------------------------------------
c  Parameters set in sr aaset and variables to communicate between moduls
c------------------------------------------------------------------------

      integer      ifop,ifmt,ifch,ifcx,ifhi,ifdt,ifcp,ifdr
      common/files/ifop,ifmt,ifch,ifcx,ifhi,ifdt,ifcp,ifdr
      integer       ifin
      common/files2/ifin
      character*500  fnch,fnhi,fndt,fnii,fnid,fnie,fnrj,fnmt
     * ,fngrv,fncp,fnnx,fncs,fndr,fnhpf
      common/fname/  fnch, fnhi, fndt, fnii, fnid, fnie, fnrj, fnmt
     * ,fngrv,fncp,fnnx,fncs,fndr,fnhpf
      integer       nfnch,nfnhi,nfndt,nfnii,nfnid,nfnie,nfnrj,nfnmt
     *,nfngrv,nfncp,nfnnx,nfncs,nfndr,nfnhpf
      common/nfname/nfnch,nfnhi,nfndt,nfnii,nfnid,nfnie,nfnrj,nfnmt
     *,nfngrv,nfncp,nfnnx,nfncs,nfndr,nfnhpf
      character*500  fnin
      common/fname2/ fnin
      integer        nfnin
      common/nfname2/nfnin
      real         delvol,deleps,dlzeta,etafac,facnuc,taurea,epscri
      common/resc2/delvol,deleps,dlzeta,etafac,facnuc,taurea,epscri(3)
      real         epsdfm
      common/resc4/epsdfm
      character*3  hydt
      common/hydr1/hydt
      real         pud,pmqu,pmqd,pmqs,pmqc,pmqq
      integer      ndecay,maxres
      common/frag1/ndecay,maxres,pud,pmqu,pmqd,pmqs,pmqc,pmqq
      real         pdiqua,delrex,ptfraqq,ptfra,ptfrasr
      integer      ioptf
      common/frag2/pdiqua,delrex,ptfraqq,ptfra,ptfrasr,ioptf
      real         aouni,pbreak,fkappa,strcut,diqcut,fkappag,pbreakg
     *,zetacut
      integer      itflav
      common/frag3/aouni,pbreak,fkappa,itflav,strcut,diqcut
     *,fkappag,pbreakg,zetacut
      real         difud,difus,difuc,pudd,puds,pudc,difuuu,difuud
     *,difuus,difuuc,difudd,difuds,difudc,difuss,difusc,difucc
      integer      nrflav
      common/frag4/difud,difus,difuc,pudd,puds,pudc,difuuu,difuud
     *,difuus,difuuc,difudd,difuds,difudc,difuss,difusc,difucc,nrflav
      real         qmass
      integer      isospin
      common/frag5/qmass(0:6),isospin(0:6)
      real         pnll,ptq,exmass,cutmss,wproj,wtarg
      common/hadr1/pnll,ptq,exmass,cutmss,wproj,wtarg
      real          rstrau,rstrad,rstras,rstrac,rstrasi
      common/hadr10/rstrau(4),rstrad(4),rstras(4),rstrac(4),rstrasi
      real          wgtqqq,wgtval,wgtsea,wgtdiq
      common/wgtqrk/wgtqqq(4),wgtval,wgtsea,wgtdiq
      double precision timeini,timefin
      common/time1/timeini,timefin
      integer       iotst1,iotst2,iotst3,iotst4
      common/ciotst/iotst1,iotst2,iotst3,iotst4
      real         taumin,deltau,factau,numtau,amsiac,amprif
      common/resc1/taumin,deltau,factau,numtau,amsiac,amprif
      real         dscale,cepara,delamf,deuamf
      integer      iceopt
      common/resc3/dscale,cepara,iceopt,delamf,deuamf
      integer      ispherio,icotabm,icotabr,icocore
      common/sprio/ispherio,icotabm,icotabr,icocore
      integer      iurqmd
      common/urqm1/iurqmd
      real         cutico,dssico
      common/incon/cutico,dssico
      real         gaumx
      integer      istore,istmax,irescl,ntrymx,nclean,iopdg,ioidch
      common/othe1/istore,istmax,gaumx,irescl,ntrymx,nclean,iopdg,ioidch
      integer      ifrade,iframe,idecay,jdecay,iremn
      common/othe2/ifrade,iframe,idecay,jdecay,iremn
      integer       jframe,kframe
      common/othe3/jframe,kframe
      integer      iselect
      common/othe4/iselect
      real         taumx,sigj
      integer      jpsi,jpsifi,nsttau,ijphis,ijtauan
      common/jpsif/jpsi,jpsifi,taumx,nsttau,sigj,ijphis,ijtauan
      real         themas
      integer      iopenu
      common/strlt/iopenu,themas
      integer      iappl,model
      common/appli/iappl,model
      integer      nevent,nfull,nfreeze,ninicon
      common/events/nevent,nfull,nfreeze,ninicon
      real         egymin,egymax,elab,ecms,ekin
      common/enrgy/egymin,egymax,elab,ecms,ekin
      integer      iprmpt,ish,ishsub,irandm,irewch,iecho,modsho,idensi
      common/prnt1/iprmpt,ish,ishsub,irandm,irewch,iecho,modsho,idensi
      real         engy,elepti,elepto,angmue
      integer      icinpu
      common/lept1/engy,elepti,elepto,angmue,icinpu
      real         core,fctrmx
      integer       laproj,maproj,latarg,matarg
      common/nucl1/laproj,maproj,latarg,matarg,core,fctrmx
      real         bmaxim,bminim,phimax,phimin
      common/nucl2/bmaxim,bminim,phimax,phimin
      real         ymximi,wtimet,wtimei,wtimea
      integer      imihis,iclhis,iwtime
      common/wana1/ymximi,imihis,iclhis,iwtime,wtimet,wtimei,wtimea
      real         wtmini,wtstep
      integer      isphis,ispall,iwcent,iana,nbdky
      common/wana2/isphis,ispall,wtmini,wtstep,iwcent,iana,nbdky
      real         asuhax,asuhay
      common/drop4/asuhax(7),asuhay(7)
      real         grigam,grirsq,gridel,grislo,gricel,sigppi,sigppd
      common/gribo/grigam,grirsq,gridel,grislo,gricel,sigppi,sigppd
      real         bag4rt,dezzer,amuseg,taunll,yradmx,facts,factb,factq
      common/drop3/bag4rt,dezzer,amuseg,taunll,yradmx,facts,factb,factq
      real         rcoll,ylongmx,facecc,yradpp,yradmi,yrmaxi,fradflii
      integer      nsegsu,nsegce
      common/drop2/rcoll,ylongmx,nsegsu,nsegce,facecc,yradpp,yradmi
     *             ,yrmaxi,fradflii
      real         ptclu,yradpi,yradpx,fploss,fvisco,fplmin
      integer      ioclude,iocluin,ioquen,iohole,kigrid
      common/drop7/ptclu,yradpi,yradpx,fploss,fvisco,fplmin,ioclude
     *            ,iocluin,ioquen,iohole,kigrid
      real         fsgrid
      common/drop8/fsgrid
      integer      iospec,iocova,iopair,iozero,ioflac,iomom
      common/metr1/iospec,iocova,iopair,iozero,ioflac,iomom
      integer      nadd,iograc,iocite,ioceau,iociau
      common/metr2/nadd,iograc,iocite,ioceau,iociau
      integer      iomodl,idproj,idtarg
      real         wexcit
      common/hadr2/iomodl,idproj,idtarg,wexcit
      real          rexdifi,rexndii
      integer       idprojin,idtargin,irdmpr,isoproj,isotarg
      common/hadr25/idprojin,idtargin,rexdifi(4),rexndii(4),irdmpr,
     *              isoproj,isotarg
      integer      iostat,ioinco,ionlat,ioobsv,iosngl,iorejz,iompar
      common/metr3/iostat,ioinco,ionlat,ioobsv,iosngl,iorejz,iompar
      integer      ioinfl,ioinct,iowidn
      real         epsgc
      common/metr4/ioinfl,ioinct,iowidn,epsgc
      real         prob
      integer      nstmax,icbac,icfor
      common/lept2/nstmax,prob(99),icbac(99,2),icfor(99,2)
      integer      iolept,igampr,idisco
      common/lept3/iolept,igampr,idisco
      real        engmin,engmax
      integer     noebin,nrebin,iologe,iologl
      common/ebin/noebin,engmin,engmax,nrebin,iologe,iologl
      real         pi,pii,hquer,prom,piom,ainfin 
      common/cnsta/pi,pii,hquer,prom,piom,ainfin 
      integer      iversn,iverso 
      common/versn/iversn,iverso 
      integer      imsg,jerr,ntevt,nrevt,naevt,nrstr,nrptl
      common/accum/imsg,jerr(mxjerr),ntevt,nrevt,naevt,nrstr,nrptl
      integer       nglacc
      common/accum2/nglacc
      integer      nptlu,nrclu
      common/cptlu/nptlu /cnrclu/nrclu
      real         tecm,volu
      common/drop6/tecm,volu
      integer      iterma,iternc,iterpr,iterpl,iozinc,iozevt
      common/metr5/iterma,iternc,iterpr,iterpl,iozinc,iozevt
      real         epsr
      integer      keepr 
      common/metr6/epsr,keepr 
      integer      keu,ked,kes,kec,keb,ket
      common/drop5/keu,ked,kes,kec,keb,ket
      double precision seedi,seedj,seedj2,seedc
      integer      iseqini,iseqsim
      common/cseed/seedi,seedj,seedj2,seedc,iseqini,iseqsim  
      real          clust
      common/cjintc/clust(mxtau,mxvol,mxeps)
      real          volsum,vo2sum
      integer       nclsum
      common/cjintd/volsum(mxtau),vo2sum(mxtau),nclsum(mxtau)
      integer       iutotc,iutote
      common/ciutot/iutotc,iutote
      integer      nopen,nopenr
      common/copen/nopen,nopenr
      integer      kchopen,khiopen,kdtopen,kcpopen,klgopen,knxopen
      common/kopen/kchopen,khiopen,kdtopen,kcpopen,klgopen,knxopen
      character*6  xvaria,yvaria
      real         xminim,xmaxim,hisfac
      integer      normal,nrbins
      common/vana1/xvaria,yvaria,normal,xminim,xmaxim,nrbins,hisfac
      integer      iologb,iocnxb
      common/vana3/iologb,iocnxb
      integer      mxnody,nrnody,nody
      parameter(mxnody=200)
      common/nodcy/nrnody,nody(mxnody)
      real         ctaumin
      common/ctdcy/ctaumin
      character*20 subpri
      integer      nrpri,ishpri 
      common/prnt2/nrpri,subpri(mxpri),ishpri(mxpri) 
      integer      ishevt,ixtau,iwseed,jwseed,ixgeometry
      common/prnt3/ishevt,ixtau,iwseed,jwseed,ixgeometry
      integer      mxcnt,ionoerr
      real         ar,ary,ardy
      parameter (mxcnt=20)
      common/vana4/ar(mxbins,5),ary(mxbins,mxcnt),ardy(mxbins,mxcnt)
     *,ionoerr
      real         xpar1,xpar2,xpar3,xpar4,xpar5,xpar6,xpar7,xpar8
     *,xpar99
      common/xpars/xpar1,xpar2,xpar3,xpar4,xpar5,xpar6,xpar7,xpar8
     *,xpar99
      integer      khisto
      common/khist/khisto
      integer      nctcor,ncttim
      common/ctcor/nctcor/ccttim/ncttim
      integer      kdensi
      real         tauv
      common/densi/kdensi(matau,nzeta,mxcoox,mxcooy),tauv(matau)
      integer       iorsce,iorsdf,iorshh,ionudi
      common/cjinti/iorsce,iorsdf,iorshh,ionudi
      real         amimfs,amimel
      common/camim/amimfs,amimel
      real          scr,scs,hacore
      common/craddf/scr,scs,hacore
      integer      iokoll
      common/ckoll/iokoll
      integer      ncnt,inicnt,nemsi
      common/cncnt/ncnt  /cicnt/inicnt /cnemsi/nemsi
      real        gfactor,gwidth
      integer     iemspl,iemsct
      common/ems1/iemspl,iemsct,gfactor,gwidth
      real           amproj,amtarg,ypjtl,yhaha,pnullx
      common/chadron/amproj,amtarg,ypjtl,yhaha,pnullx
      real         xshift,etacut
      common/vana5/xshift,etacut
      double precision rnucl
      real         bnucl,xbtot
      integer      ixbDens
      common/nucl6/infragm
      integer      infragm
      common/nucl5/rnucl(mxnucl,2),bnucl(mxnucl,4),xbtot(4),ixbDens
      real         drnucl,rnuclo
      integer      nrnucl
      common/nucl4/nrnucl(2),drnucl(2),rnuclo(mxnucl,2)
      real       xsig,xpom
      common/sig/xsig(7),xpom(7)
      integer      ktnbod
      common/metr7/ktnbod
      integer      iregge,isopom,ishpom,iscreen,nprmax,inueff,irmdrop
      common/hadr3/iregge,isopom,ishpom,iscreen,nprmax,inueff,irmdrop
      real         sigtot,sigcut,sigela,sloela,sigsd,sigine,sigdif
     *,sigineaa,sigtotaa,sigelaaa,sigcutaa,sigdd
      common/hadr5/sigtot,sigcut,sigela,sloela,sigsd,sigine,sigdif
     *,sigineaa,sigtotaa,sigelaaa,sigcutaa,sigdd
      integer      intpol,isigma,iomega,isetcs
      common/hadr6/intpol,isigma,iomega,isetcs
      real         alppom,slopom,gamhad,r2had,chad,wdiff
     *,gamtil,facdif,facmc,r2hads,gamhads,slopoms
      integer      isplit
      common/hadr4/alppom,slopom,gamhad(4),r2had(4),chad(4),wdiff(4)
     &      ,gamtil,facdif,facmc,r2hads(4),gamhads(4),slopoms,isplit
      real          gamhadsi
      common/hadr42/gamhadsi(4)
      real         alpreg,sloreg,gamreg,r2reg,ptdiff,ptsend,xminremn
     &,xmindiff,ptsecu
      common/hadr7/alpreg,sloreg,gamreg,r2reg,ptdiff,ptsend,xminremn
     &,xmindiff,ptsecu
      real         alpqua,alppar,alpsea,alpval,alpdiq,alplea,alpdif
      common/hadr8/alpqua,alppar,alpsea,alpval,alpdiq,alplea(4),alpdif
      real          alpndi,alpdi,ptsendi,zdrinc,zmsinc,ptsems
      integer       irzptn
      common/hadr14/alpndi,alpdi,ptsendi,zdrinc,zmsinc,ptsems,irzptn
      real          zbcut,zopinc,zipinc,zoeinc,xmxrem
      common/hadr15/zbcut,zopinc,zipinc,zoeinc,xmxrem
      real          fkainc,fkamax,zodinc,zbrmax,zdfinc,xzcut,ptvpom
      common/hadr16/fkainc,fkamax,zodinc,zbrmax,zdfinc,xzcut,ptvpom
      real          edmaxi,epmaxi
      common/hadr17/edmaxi,epmaxi
      real          ammsqq,ammsqd,ammsdd,cumpom,rexndi,rexdif
     *             ,reminv,rexpdif,rexres,zrminc,rexndf
      common/hadr9/ammsqq,ammsqd,ammsdd,cumpom,rexndi(4),rexdif(4)
     *             ,reminv,rexpdif(4),rexres(4),zrminc,rexndf
      integer      iclpro,icltar,iclegy
      common/had10/iclpro,icltar,iclegy
      integer      iclpro1,iclpro2,icltar1,icltar2,iclegy1,iclegy2
      common/had11/iclpro1,iclpro2,icltar1,icltar2,iclegy1,iclegy2
      real         egylow,egyfac
      common/had12/egylow,egyfac
      real         amdrmax,amdrmin,alpdro
      common/had13/amdrmax,amdrmin,alpdro(3)
      real         alpcoso,alpcose,betcoso,betcose
      common/had14/alpcoso,alpcose,betcoso,betcose
      real         accept,reject
      common/emsx1/accept,reject
      integer      iemspr,iemspm,iemspx,iemsrx,iemspu,iemsi2,iemspbx
      common/xems1/iemspr,iemspm,iemspx,iemsrx,iemspu,iemsi2,iemspbx
      integer      iemsse,iemsi1,iemsb,iemsbg,ioems,iemsdr
      common/xems2/iemsse,iemsi1,iemsb,iemsbg,ioems,iemsdr
      integer        ispacetime
      common/xspatim/ispacetime
      real           difnuc,radnuc
      common /psar10/difnuc(mamxx),radnuc(mamxx)
      integer        mxbarray,nbarray
      real           barray
      parameter (mxbarray=100)
      common/cbarray/barray(mxbarray),nbarray
      real          airznxs,airanxs,airwnxs
     *             ,airavznxs,airavanxs
      common/nxsair/airznxs(3),airanxs(3),airwnxs(3)
     *             ,airavznxs,airavanxs
      real            qgsincs
      common/mod2incs/qgsincs
      real            gheincs
      common/mod3incs/gheincs
      real            pytincs
      common/mod4incs/pytincs
      real            hijincs
      common/mod5incs/hijincs
      real            sibincs
      common/mod6incs/sibincs
      real            qgsIIincs
      common/mod7incs/qgsIIincs
      real            phoincs
      common/mod8incs/phoincs
      real            fluincs
      common/mod9incs/fluincs
      real            urqincs
      common/mod10incs/urqincs
      real           antot,ansh,ansf,pp4max,pp4ini,andropl,anstrg0
     *,anshf,ansff,antotf
     *,anstrg1,anreso0,anreso1,anghadr,antotre
      common/testpom/antot,ansh,ansf,pp4max,pp4ini,andropl,anstrg0
     *,anshf,ansff,antotf
     *,anstrg1,anreso0,anreso1,anghadr,antotre
      real         anintdiff,anintsdif,anintine
     *,sigineex,sigdifex,sigsdex
      common/cdiff/anintdiff,anintsdif,anintine
     *,sigineex,sigdifex,sigsdex
      real           epszero,alpff,betff
      common/cepszer/epszero,alpff(nclha),betff(2)
      real        tgss,wgss
      common/cgss/tgss(7,7),wgss(7,7)
      
      real alpDs,alpDps,alpDpps,betDs,betDps,betDpps,gamDs,delDs
     *    ,alpD,alpDp,alpDpp,betD,betDp,betDpp,gamD,delD
      common/Dparams/alpDs(  idxD0:idxD, nclegy, nclha,nclha),
     *              alpDps( idxD0:idxD, nclegy, nclha,nclha),
     *              alpDpps(idxD0:idxD, nclegy, nclha,nclha),
     *              betDs(  idxD0:idxD, nclegy, nclha,nclha),
     *              betDps( idxD0:idxD, nclegy, nclha,nclha),
     *              betDpps(idxD0:idxD, nclegy, nclha,nclha),
     *              gamDs(  idxD0:idxD, nclegy, nclha,nclha),
     *              delDs(  idxD0:idxD, nclegy, nclha,nclha)
      real          bmxdif,bkmxndif
      integer       idxDmin
      common/Dparam/alpD(  idxD0:idxD1, nclha, nclha),
     *              alpDp( idxD0:idxD1, nclha, nclha),
     *              alpDpp(idxD0:idxD1, nclha, nclha),
     *              betD(  idxD0:idxD1, nclha, nclha),
     *              betDp( idxD0:idxD1, nclha, nclha),
     *              betDpp(idxD0:idxD1, nclha, nclha),
     *              gamD(  idxD0:idxD1, nclha, nclha),
     *              delD(  idxD0:idxD1, nclha, nclha),
     *     idxDmin,bmxdif(nclha, nclha),bkmxndif
      double precision alpUni,betUni,betpUni,fpUni,ftUni,betfom
     *                 ,epspUni,epstUni,zzpUni,zztUni
      real           alpfom,alpfomi,gamfom
      common/DparUni/alpUni(  idxD0:idxD1,2),
     *               betUni(  idxD0:idxD1,2),
     *               betpUni(idxD0:idxD1,2),fpUni,ftUni,
     *               epspUni(idxD0:idxD1),zzpUni,
     *               epstUni(idxD0:idxD1),zztUni,
     *               betfom,alpfom,alpfomi,gamfom
      integer      idlead,ilprtg
      common/crvar/idlead,ilprtg
      integer        iLHC
      common/LHCtune/iLHC
