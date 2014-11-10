c*******************************************

      SUBROUTINE myini
      
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision nbcol,npart,npart0
      external pydata
      double precision npar0,nbco0 
      common /hyjets/ nhj,nhp,khj(150000,5),phj(150000,5),vhj(150000,5) 
      common /hyfpar/ bgen,nbcol,npart,npart0,npyt,nhyd
      common /hyflow/ ytfl,ylfl,Tf,fpart 
      common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet        
      common /pydat1/ mstu(200),paru(200),mstj(200),parj(200)
      common /pysubs/ msel,mselpd,msub(500),kfin(2,-40:40),ckin(200) 
      common /pypars/ mstp(200),parp(200),msti(200),pari(200)
      common /pyqpar/ T0,tau0,nf,ienglu,ianglu 
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hypyin/ ene,rzta,rnta,bfix,ifb,nh
      
      COMMON/PYDATR/MRPY(6),RRPY(100)
      common/SERVICE/iseed_fromC,iPythDecay,charm

      
      common /hypart/ppart(20,150000),bmin,bmax,njp
      save /hyjets/,/hyflow/,/hyjpar/,/hyfpar/,/pyqpar/,
     >     /pysubs/,/pypars/,/pydat1/

* ----------- INITIALIZATION OF RANDOM GENERATOR
      MRPY(1)=iseed_fromC 
c      write(*,*)'--initialization of high-pt part-- :  '
c      write(*,*)'evnt generator phase  ',iseed_fromC
 
* initialize HYINIT with the input parameters   
c      write(*,*)' AW= ', AW,' energy= ', ene, ' ptmin= ',ptmin 
c      write(*,*)' bminh= ',bminh,' bmaxh= ', bmaxh 
c      write(*,*)' ifb= ', ifb, ' bfix= ', bfix 
c      write(*,*)' ishad= ',ishad, ' nhsel=  ',nhsel 
c      write(*,*)' ienglu= ',ienglu, ' ianglu= ',ianglu 
c      write(*,*)' T0= ',T0,' tau0= ',tau0,' nf= ', nf
                                      
* set input PYTHIA parameters from SERVICE common:
c      PARP(2)=5.d0              ! minimum c.m.s. energy of pp collision  
c      MSTJ(22)=2                ! particle decays if lifetime < parj(71)
c      PARJ(71)=10.              ! ctau=10 mm 
      CKIN(3)=ptmin               ! minimum pt in initial hard scattering, GeV 
                       
      call hyinit     ! ml      (ene,AW,ifb,bmin,bmax,bfix1,nh) 

      end
****************end myini***************************************************************



********************************* HYINIT ***************************
      SUBROUTINE HYINIT
      
*ml--      (energy,A,ifb1,bmin,bmax,bfix1,nh1) 
*     PYTHIA inizialization, calculation of total and hard cross sections and   
*     # of participants and binary sub-collisions at "reference point" (Pb,b=0),
*     tabulation of nuclear thickness function and nuclear overlap function, 
*     test of input parametes 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision numpar,npar0,nbco0 
      external numpar,rhoaa,hfunc3
      common /pyint7/ sigt(0:6,0:6,0:5)
      common /pypars/ mstp(200),parp(200),msti(200),pari(200)
      common /pysubs/ msel,mselpd,msub(500),kfin(2,-40:40),ckin(200) 
      common /pyjets/ n,npad,k(4000,5),p(4000,5),v(4000,5)
      common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet 
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hypyin/ ene,rzta,rnta,bfix,ifb,nh
      common /hyflow/ ytfl,ylfl,Tf,fpart
      common /hygeom/ BC
      common /hythic/ BAB(110),TAB(110),TAAB(110)
      common /hynup1/ bp,x
      common/SERVICE/iseed_fromC,iPythDecay,charm

*

      save /pyint7/,/pypars/,/pysubs/,/hyjpar/,/hyipar/,/hypyin/,
     >     /hyflow/,/hygeom/,/hythic/,/pyjets/    
            
* start HYDJET initialization 
      init=1
      ipr=1
      
* set beam paramters
c-ml      ene=energy                               ! c.m.s. energy per nucleon
c-ml      AW=A                                     ! atomic weight


      RA=1.15d0*AW**0.333333d0                 ! nuclear radius
      rzta=1.d0/(1.98d0+0.015d0*AW**0.666667d0) ! fraction of protons in nucleus
      rnta=1.d0-rzta                           ! fraction of neutrons in nucleus
c-ml      ifb=ifb1                                 ! centrality flag
c-ml      bfix=bfix1                               ! fixed impact parameter
c-ml      nh=nh1                                   ! mean soft mult. in central PbPb
c-ml      ptmin=ckin(3)                            ! minimum pt of hard scattering 

* Pythia inizialization 
      call pyinit('cms','p','p',ene)

c      if(nhsel.ne.0) then
 
       mstp(111)=0 

* no printout of Pythia initialization information hereinafter 
       mstp(122)=0  

* initialize HYINIT with the input parameters   
       write(*,*)'in hyinit AW= ', AW,' energy= ', ene, ' ptmin= ',ptmin 
       write(*,*)'bminh= ',bminh,' bmaxh= ', bmaxh 
       write(*,*)'ifb= ', ifb, ' bfix= ', bfix 

* Pythia pre-run to calculate charm production in pp events
      charm=0.d0
      ckin(3)=0.d0 
      ckin(4)=ptmin
      call pyinit('cms','p','p',ene)
      do i=1,10000
        call pyevnt
	do ip=9,n  
	 if(abs(k(ip,2)).eq.4) charm=charm+1.d0          
        end do
       end do 
       ckin(3)=ptmin
       ckin(4)=-1.d0
       charm=0.0001d0*charm*pari(1)/(sigt(0,0,0)-sigt(0,0,1)) 
c       write(6,*) 'Charm',charm

* Pythia inizialization for pp collisions 
       call pyinit('cms','p','p',ene)
* Pythia test pp event run 
       do i=1,1000
        call pyevnt 
       end do
* hard scattering pp cross section 
       sjpp=pari(1)
       
* Pythia inizialization for pn collisions 
      call pyinit('cms','p','n',ene)
* Pythia test pn event run 
       do i=1,1000
        call pyevnt 
       end do
* hard scattering pn cross section 
       sjpn=pari(1)       
       
* Pythia inizialization for nn collisions 
      call pyinit('cms','n','n',ene)
* Pythia test nn event run 
       do i=1,1000
        call pyevnt 
       end do
* hard scattering nn cross section 
       sjnn=pari(1)       
       
       sigjet=rzta*rzta*sjpp+rnta*rnta*sjnn+2.d0*rzta*rnta*sjpn
       
c      end if 
      
* total inelastic cross section 
      if(sigin.lt.10.d0.or.sigin.gt.200.d0) 
     >   sigin=sigt(0,0,0)-sigt(0,0,1)  

* # of nucelons-participants and NN sub-collisions at "reference point" (Pb,b=0)  
      Apb=207.d0      
      Rpb=1.15d0*Apb**0.333333d0  
      EPS=0.005d0
      Z2=4.d0*Rpb
      Z1=-1.d0*Z2
      H=0.01d0*(Z2-Z1)
      do ib=1,110    
       BC=3.d0*Rpb*(ib-1)/109.d0
       CALL SIMPA(Z1,Z2,H,EPS,1.d-8,rhoaa,Z,RES,AIH,AIABS) 
       BAB(ib)=BC
       TAB(ib)=Apb*RES
      end do     
      Z1=0.d0
      Z2=6.28318d0 
      H=0.01d0*(Z2-Z1)     
      bp=0.d0
      CALL SIMPA(Z1,Z2,H,EPS,1.d-8,HFUNC3,X,TAAPB0,AIH,AIABS)    
      
      npar0=numpar(0.d0)                            ! Npart(Pb,b=0)  
      nbco0=0.1d0*sigin*TAAPB0                      ! Nsub(Pb,b=0) 
       
      init=0 

* creation of arrays for tabulation of beam/target nuclear thickness function
      Z2=4.d0*RA
      Z1=-1.d0*Z2
      H=0.01d0*(Z2-Z1)
      do ib=1,110    
       BC=3.d0*RA*(ib-1)/109.d0
       CALL SIMPA(Z1,Z2,H,EPS,1.d-8,rhoaa,Z,RES,AIH,AIABS)     
       BAB(ib)=BC
       TAB(ib)=AW*RES
      end do 

* creation of arrays for tabulation of nuclear overlap function
      Z1=0.d0
      Z2=6.28318d0 
      H=0.01d0*(Z2-Z1)    
      do ib=1,110 
       bp=BAB(ib)
       CALL SIMPA(Z1,Z2,H,EPS,1.d-8,HFUNC3,X,RES,AIH,AIABS)
       TAAB(ib)=RES 
      end do   

       bmin=bminh 
       bmax=bmaxh 
     
c       write(*,*)'in HYINIT bmin', bmin,'bmax', bmax
     
     
* test of centrality selection 
      if(ifb.eq.0) then 
       if(bfix.lt.0.d0) then    
        write(6,*) 'Impact parameter less than zero!'  
        bfix=0.d0 
       end if  
       if (bfix.gt.3.d0) then 
        write(6,*) 'Impact parameter larger than three nuclear radius!'  
        bfix=3.d0        
       end if 
      else        
       if(bmin.lt.0.d0) then    
        write(6,*) 'Impact parameter less than zero!'  
        bmin=0.d0 
       end if 
       if(bmax.gt.3.d0) then    
        write(6,*) 'Impact parameter larger than three nuclear radius!'
        bmax=3.d0  
       end if             
      end if 
      
* test of flow parameter selection  
      if (Tf.lt.0.08d0.or.Tf.gt.0.2d0) Tf=0.1d0       ! freeze-out temperature
      if (ylfl.lt.0.01d0.or.ylfl.gt.7.d0) ylfl=4.d0   ! longitudinal flow rapidity
      if (ytfl.lt.0.01d0.or.ytfl.gt.3.d0) ytfl=1.5d0  ! transverse flow rapidity
      if (fpart.le.0.d0.or.fpart.gt.1.d0) fpart=1.d0  ! fraction of soft multiplicity
                                                      ! proport. to # of participants
* test of 'nhsel' selection      
      if(nhsel.ne.1.and.nhsel.ne.2.and.nhsel.ne.3.and.nhsel.ne.4) 
     > nhsel=0 
  
      return 
      end 
****************************** END HYINIT ***************************
      SUBROUTINE MYDELTA
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      common/SERVICEEV/psiv3,delta,KC,ipdg
      COMMON /PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      integer KC, PYCOMP
      real delta, psiv3
       KC=PYCOMP(ipdg)
       delta=PMAS(int(KC),3)
c       write(*,*)" ipdg ", ipdg, " KC ",KC," delta ",delta
       return
       end


********************************* HYEVNT **************************** 
      SUBROUTINE HYEVNT
*     generation of single HYDJET event (soft+hard parts) at given parameters 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision numpar,npar0,nbco0,npart,nbcol,npart0 
      external hsin,gauss,hftaa,numpar,hyhard,hipsear,pyr,pymass,PYCOMP 
      common /hyjets/ nhj,nhp,khj(150000,5),phj(150000,5),vhj(150000,5)
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hypyin/ ene,rzta,rnta,bfix,ifb,nh
      common /hyfpar/ bgen,nbcol,npart,npart0,npyt,nhyd
      common /hyflow/ ytfl,ylfl,Tf,fpart
      common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet  
c-ml
      common /hypart/ppart(20,150000),bmin,bmax,njp
      COMMON /PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)

      save /hyjets/,/hyipar/,/hyfpar/,/hyflow/,/hyjpar/,/hypyin/


* reset lujets and hyjets arrays before event generation 
c      write(*,*)'in hyevnt 0'


      nhj=0 
      do ncl=1,150000
       do j=1,5
        phj(ncl,j)=0.d0 
        vhj(ncl,j)=0.d0  
        khj(ncl,j)=0 
       enddo
      end do 
* 
      pi=3.14159d0
      
* generate impact parameter of A-A collision 
      if(ifb.eq.0) then 
       b1=bfix*RA
       bgen=bfix 
      else          
       call hipsear(fmax1,xmin1) 
       fmax=fmax1 
       xmin=xmin1 
 3     bb1=xmin*pyr(0)+bminh*RA  
       ff1=fmax*pyr(0) 
       fb=hsin(bb1) 
       if(ff1.gt.fb) goto 3    
       b1=bb1       
       bgen=bb1/RA 
      end if 
      
c       write(*,*)'in hyevnt RA b1 bgen sigin',RA,bminh,bgen,sigin
                    
* calculate # of nucelons-participants and binary NN sub-collisions 
      npart=numpar(b1)                          ! Npart(b) 
      npart0=numpar(0.d0)                          ! Npart(b) 

c      write(*,*)'in hyevnt npart',npart   
      nbcol=0.1d0*sigin*hftaa(b1)               ! Nsub(b)        
c      write(*,*)'in hyevnt nbcol',nbcol   
* generate hard parton-parton scatterings (Q>ptmin) 'njet' times   
      njet=0 
      if(nhsel.ne.0) then 
       pjet=sigjet/sigin   
c       write(*,*)'in hyevnt pjet',pjet  
       do i=1,int(nbcol) 
        if(pyr(0).lt.pjet) njet=njet+1 
       end do  
c       write(*,*)'before hyhard' 
       call hyhard 
      end if 

      npyt=nhj 
      
c      write(*,*)'in hyevnt pjet njet=', pjet,njet
c-ml
* fill array 'ppart' 
     
      do ih=1,nhj
        if(ih.le.150000)then 
        ppart(1,ih)= khj(ih,1) ! status code
        ppart(2,ih)= khj(ih,2) ! pdg         
        ppart(3,ih)=phj(ih,1) ! px
        ppart(4,ih)=phj(ih,2) ! py
        ppart(5,ih)=phj(ih,3) ! pz
        ppart(6,ih)=phj(ih,4) ! E
        ppart(7,ih)= vhj(ih,1) !x 
        ppart(8,ih)= vhj(ih,2) !y     
        ppart(9,ih)= vhj(ih,3) !z     
        ppart(10,ih)= vhj(ih,4) !t
* mother information
        ppart(11,ih)= khj(ih,3) ! line number of parent particle        
        ppart(12,ih)= khj(ih,4) ! line number of first daughter     
        ppart(13,ih)= khj(ih,5) ! line number of last daughter

c        write(*,*)'progs: pdg ', ppart(2,ih), 'mother line',ppart(11,ih)
c        write(*,*)'progs: motherInt ', khj(ih,3)
c        write(*,*)'progs: motherDouble ', ppart(11,ih)
c        write(*,*)'progs: d1Int ', khj(ih,4)
c        write(*,*)'progs: d1Double ', ppart(12,ih)
c        write(*,*)'progs: d2Int ', khj(ih,5)
c        write(*,*)'progs: d2Double ', ppart(13,ih)
        endif
      end do
 
       njp=nhj ! fill number of jet particles for InitialStateBjorken
c--
     
      return
      end
****************************** END HYEVNT ************************** 
********************************* HYHARD ***************************
      SUBROUTINE HYHARD 
*     generate 'njet' number of hard parton-parton scatterings with PYTHIA 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision npar0,nbco0,npart,nbcol,npart0
      INTEGER PYK,PYCOMP
      CHARACTER beam*2,targ*2
      external pydata 
      external pyp,pyr,pyk,pyquen,shad1,gauss
      common /pyjets/ n,npad,k(4000,5),p(4000,5),v(4000,5)
      common /hyjets/ nhj,nhp,khj(150000,5),phj(150000,5),vhj(150000,5)
      COMMON /PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON /PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON /PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      common /pypars/ mstp(200),parp(200),msti(200),pari(200)
      common /parimp/ b1, psib1, r0, rb1, rb2, noquen 
      common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hyfpar/ bgen,nbcol,npart,npart0,npyt,nhyd 
      common /hypyin/ ene,rzta,rnta,bfix,ifb,nh
      save /pyjets/,/pypars/,/pydat1/,/pydat2/,/pydat3/,/pysubs/,
     +     /hyjets/,/parimp/,/hyjpar/,/hyipar/,/hyfpar/,/hypyin/

* generate 'njet' PYTHIA events and fill arrays for partons and hadrons 
      nshad=0
      noquen=0 
      if(nhsel.eq.1.or.nhsel.eq.3) noquen=1
      if(njet.ge.1) then 
       mdcy(pycomp(111),1)=0                     ! no pi0 decay 
       mdcy(pycomp(310),1)=0                     ! no K_S0 decay 
     
       ifbp=0                                    ! fix impact parameter 
       bfixp=RA*bgen 
       Ap=AW                                     ! atomic weight            
 
       do ihard=1,njet       
        mstp(111)=0

* generate type of NN sub-collision (pp, pn or nn) 
        rand1=pyr(0)
        if(rand1.lt.rzta) then 
         beam='p'
        else 
         beam='n'
        end if 
        rand2=pyr(0)
        if(rand2.lt.rzta) then 
         targ='p'
        else 
         targ='n'
        end if 
        call pyinit('cms',beam,targ,ene)
c        mstj(41)=0                           ! vacuum showering off 
        call pyevnt                           ! generate hard scattering

* PYQUEN: quenched jets if noquen=0 or non-quenched jets if noquen=1 and ishad=1
        if(ishad.eq.1.or.nhsel.eq.2.or.nhsel.eq.4) 
     >   call pyquen(Ap,ifbp,bfixp) 
     
c-ml    coordinate info if we need of it 
        Q=pari(21) 
        x=r0*cos(psib1)  !fm
        y=r0*sin(psib1)  !fm2
        tau=1./Q ! 1/GeV
        tau=tau*0.197 !fm/c
        rm=0.0d0
        sig=1.0d0
	etaLj=gauss(rm,sig)!fm
	z=tau*sinh(etaLj)
	t=tau*cosh(etaLj) 
c        write(*,*)'x y z t', x,y,z,t,etaLj

* treatment of "nuclear shadowing" (for Pb, Au, Pd or Ca beams only) 
        if(ishad.eq.1) then
         kfh1=abs(k(3,2)) 
         kfh2=abs(k(4,2))
         xh1=pari(33) 
         xh2=pari(34)
         Q2=pari(22)
         shad=shad1(kfh1,xh1,Q2,rb1)*shad1(kfh2,xh2,Q2,rb2)
         if(pyr(0).gt.shad) then 
          nshad=nshad+1 
          goto 53
         end if 
        end if 
           
        call pyexec                         ! hadronization done 
c-ml
        if(iPyhist.ne.0) call pyedit(2)                      ! remove partons & leave hadrons 

* fill array of final particles
        nu=nhj+n 
        if(nu.gt.150000-np) then 
         write(6,*) 'Warning, multiplicity too large! Cut hard part.'   
         goto 52
        end if 
        nhj=nu  
        do i=nhj-n+1,nhj
           ip=i+n-nhj  
                                   
           do j=1,5
              phj(i,j)=p(ip,j)
           end do


c-ml         
           vhj(i,1)=x 
           vhj(i,2)=y 
           vhj(i,3)=z 
           vhj(i,4)=t 
         
           do j=1,5
              khj(i,j)=k(ip,j)
           end do
           do j=3,5
              kk=khj(i,j) 
              if(kk.gt.0) then 
                 khj(i,j)=kk+nhj-n 
              end if
           end do
        end do               
      
 53     continue
      end do 
 52   njet=ihard-1 
      end if  
      njet=njet-nshad
        
      return 
      end 
****************************** END HYHARD **************************      
********************************* HIPSEAR ***************************
      SUBROUTINE HIPSEAR (fmax,xmin) 
* find maximum and 'sufficient minimum' of differential inelasic AA cross 
* section as a function of impact paramater (xm, fm are outputs)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision npar0,nbco0 
      external hsin  
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      save /hyipar/ 
      xmin=(bmaxh-bminh)*RA 
      
c      write(*,*)'bmaxh bminh',bmaxh,bminh
      
      fmax=0.d0
      do j=1,1000
      x=bminh*RA+xmin*(j-1)/999.d0
      f=hsin(x) 
       if(f.gt.fmax) then
        fmax=f
       endif
      end do   
      return
      end
****************************** END HIPSEAR **************************

* differential inelastic AA cross section  
      double precision function hsin(x) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external hftaa 
      common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet   
      save /hyjpar/ 
      br=x 
      hsin=br*(1.d0-dexp(-0.1d0*hftaa(br)*sigin)) 
      return 
      end 

* number of nucleons-participants at impact parameter b 
      double precision function numpar(c) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external HFUNC1 
      common /hynup1/ bp,x    
      EPS=0.005d0  
      A=0.d0 
      B=6.28318d0 
      H=0.01d0*(B-A)    
      bp=c    
      CALL SIMPA(A,B,H,EPS,1.d-8,HFUNC1,X,RES,AIH,AIABS)
      numpar=2.d0*RES 
      return 
      end   
*
      double precision function HFUNC1(x) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision npar0,nbco0
      external HFUNC2 
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hynup1/ bp,xx
      save /hyipar/  
      if(init.eq.1) then 
       Rl=Rpb
      else 
       Rl=RA
      end if  
      xx=x 
      EPS=0.005d0
      A=0.d0 
      B=3.d0*Rl
      H=0.01d0*(B-A)    
      CALL SIMPB(A,B,H,EPS,1.d-8,HFUNC2,Y,RES,AIH,AIABS)
      HFUNC1=RES 
      return 
      end   
*      
      double precision function HFUNC2(y) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision npar0,nbco0
      external hythik 
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet 
      common /hynup1/ bp,x 
      save /hyipar/,/hyjpar/ 
      r1=dsqrt(abs(y*y+bp*bp/4.d0+y*bp*dcos(x))) 
      r2=dsqrt(abs(y*y+bp*bp/4.d0-y*bp*dcos(x)))
      s=1.d0-dexp(-0.1d0*sigin*hythik(r2))
      HFUNC2=y*hythik(r1)*s 
      return 
      end   

* nuclear overlap function at impact parameter b  
      double precision function hftaa(c)  
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      common /hythic/ BAB(110),TAB(110),TAAB(110)
      save /hythic/ 
      call parinv(c,BAB,TAAB,110,RES) 
      hftaa=RES 
      return 
      end   
*
      double precision function HFUNC3(x)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z) 
      double precision npar0,nbco0
      external HFUNC4 
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hynup1/ bp,xx
      save /hyipar/  
      if(init.eq.1) then 
       Rl=Rpb
      else 
       Rl=RA
      end if  
      xx=x 
      EPS=0.005d0 
      A=0.d0 
      B=3.d0*Rl
      H=0.01d0*(B-A)    
      CALL SIMPB(A,B,H,EPS,1.d-8,HFUNC4,Y,RES,AIH,AIABS)
      HFUNC3=RES 
      return 
      end   
*      
      double precision function HFUNC4(y) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision npar0,nbco0
      external hythik 
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet 
      common /hynup1/ bp,x 
      save /hyipar/,/hyjpar/ 
      r1=dsqrt(abs(y*y+bp*bp/4.d0+y*bp*dcos(x))) 
      r2=dsqrt(abs(y*y+bp*bp/4.d0-y*bp*dcos(x)))
      HFUNC4=y*hythik(r1)*hythik(r2) 
      return 
      end   

* nuclear thickness function 
       double precision function hythik(r)   
       IMPLICIT DOUBLE PRECISION(A-H, O-Z)
       common /hythic/ BAB(110),TAB(110),TAAB(110)
       save /hythic/ 
       call parinv(r,BAB,TAB,110,RES) 
       hythik=RES 
       return
       end

* Wood-Saxon nucleon distrubution  
       double precision function rhoaa(z)   
       IMPLICIT DOUBLE PRECISION(A-H, O-Z)
       double precision npar0,nbco0 
       common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
       common /hygeom/ BC 
       save /hyipar/,/hygeom/ 
       if(init.eq.1) then 
        Rl=Rpb
       else 
        Rl=RA
       end if  
       pi=3.14159d0
       df=0.54d0
       r=sqrt(bc*bc+z*z)
       rho0=3.d0/(4.d0*pi*Rl**3)/(1.d0+(pi*df/Rl)**2)
       rhoaa=rho0/(1.d0+dexp((r-Rl)/df))
       return
       end
 
* function to calculate nuclear shadowing factor (for Pb, Au, Pd or Ca beams)
      double precision function shad1(kfh,xbjh,Q2h,r)  
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      double precision npar0,nbco0,nbcol,npart,npart0
      external ggshad
      common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr
      common /hyfpar/ bgen,nbcol,npart,npart0,npyt,nhyd 
      common /hyshad/ bbmin,bbmax,inuc 
      save /hyipar/,/hyfpar/,/hyshad/ 
      dimension res(2)
      kf=kfh 
      xbj=xbjh 
      Q2=Q2h 
      bb=r 
      inuc=0  
      if(AW.gt.205.d0.and.AW.lt.209.d0) inuc=4         ! Pb-206, 207 or 208  
      if(AW.gt.196.d0.and.AW.lt.198.d0) inuc=3         ! Au-197
      if(AW.gt.109.d0.and.AW.lt.111.d0) inuc=2         ! Pd-110 
      if(AW.gt.39.d0.and.AW.lt.41.d0) inuc=1           ! Ca-40 
      if(inuc.eq.0.and.ipr.eq.1) then      
       write(6,*) 
     > 'Warning! Shadowing is not foreseen for atomic weigth  A  ='
     > ,AW 
       write(6,*)'******************************************************
     >************************'
       ipr=0 
      end if   
      if(inuc.ne.0) then 
       xbj=max(5.d-5,xbj) 
       xbj=min(0.95d0,xbj)  
       Q2=max(4.d0,Q2)
       Q2=min(520.d0,Q2)
       bb=max(0.d0,bb)
       call ggshad(inuc,xbj,Q2,bb,res,ta)
       if(kf.eq.21) then 
        shad1=res(1) 
       elseif(kf.eq.1.or.kf.eq.2.or.kf.eq.3) then 
        shad1=res(2)
       else 
        shad1=1.d0 
       end if 
       else 
        shad1=1.d0 
      end if
      return 
      end   

******************************************************************************
* The part of the code which follows below, includes nuclear shadowing model *
* and has been written by Konrad Tywoniuk (Oslo University, Norway)          *
******************************************************************************
c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$
c$$$
c$$$
c$$$  Shadowing from Glauber-Gribov theory for
c$$$  gluons and light quarks (u,d,s and their antiquarks).
c$$$  All Pomeron tree diagrams have been summed (Schwimmer model).
c$$$  More details about the model in 
c$$$  K. Tywoniuk, I.C. Arsene, L. Bravina, A. Kaidalov and E. 
c$$$  Zabrodin, Phys. Lett. B 657 (2007) 170
c$$$  
c$$$  We use FIT B from the H1 parameterization published in 
c$$$  A. Aktas et al. (H1 Collaboration), Eur. Phys. J C 48 (2006) 715
c$$$  A. Aktas et al. (H1 Collaboration), Eur. Phys. J C 48 (2006) 749
c$$$  
c$$$  Main routine is GGSHAD which provides the user with the 
c$$$  ratio of nuclear and nucleon parton distribution function 
c$$$  normalized by the atomic number for a given
c$$$  INUCL:     nucleus (1=Ca,2=Pd,3=Au,4=Pb)
c$$$  X:         Bjorken x
c$$$  Q2:        q**2, scale squared
c$$$  B:         impact parameter
c$$$  
c$$$  Then      RES(1):    gluon shadowing
c$$$            RES(2):    sea quark shadowing
c$$$            TA:        nuclear profile function
c$$$
c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$$$c$
      SUBROUTINE GGSHAD(INUCL,X,Q2,B,RES,TAF)
      IMPLICIT NONE
      DOUBLE PRECISION TA(31,4),IMPAR(31),ANUCL(4)
      DOUBLE PRECISION XB(36),Q2V(13)
      DOUBLE PRECISION XMAX,XMAXX,Q2MIN,Q2MAX,BMAX
      DOUBLE PRECISION G(36,13,4),LQ(36,13,4)
      DOUBLE PRECISION TATMP(31),SHAD(2)
      DOUBLE PRECISION C(100),D(100),E(100)
      DOUBLE PRECISION X,Q2,B
      DOUBLE PRECISION RES(2)
      DOUBLE PRECISION TAF
      DOUBLE PRECISION SEVAL
      INTEGER INUCL,TMAX
      INTEGER I,IK

      PARAMETER(TMAX=31)
      PARAMETER(XMAX=0.1)
      PARAMETER(XMAXX=0.95)
      PARAMETER(Q2MIN=4.)
      PARAMETER(Q2MAX=520.)
      PARAMETER(BMAX=30.)

      COMMON/GG07/XB,Q2V,G,LQ

      DATA ANUCL/40.,110.,197.,206./
      DATA IMPAR
     >     /0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.,
     >     8.5,9.,9.5,10.,10.5,11.,11.5,12.,12.5,13.,13.5,14.,14.5,15./
      DATA TA
     >     /0.0291662,0.0288636,0.0279354,0.0263044,0.0238341,0.0203565,
     >     0.0158333,0.0107343,0.00617758,0.00307859,0.00139738,
     >     0.000603839,0.000254843,0.000106329,4.40965e-05,1.82206e-05,
     >     7.50949e-06,3.08881e-06,1.26837e-06,5.20082e-07,2.12978e-07,
     >     8.71154e-08,3.55956e-08,1.45304e-08,5.92622e-09,2.41504e-09,
     >     9.83429e-10,4.00184e-10,1.62741e-10,6.61407e-11,2.68657e-11,
     >     0.0153487,0.0152775,0.0150614,0.0146926,0.0141561,0.0134268,
     >     0.0124651,0.0112115,0.00959258,0.00756907,0.00527514,
     >     0.00313482,0.00159852,0.000732479,0.000316628,0.000133116,
     >     5.52508e-05,2.27904e-05,9.3692e-06,3.84349e-06,1.57423e-06,
     >     6.43962e-07,2.63132e-07,1.07414e-07,4.38089e-08,1.78529e-08,
     >     7.2699e-09,2.95832e-09,1.20304e-09,4.8894e-10,1.98603e-10,
     >     0.0105299,0.010498,0.0104017,0.010239,0.010006,0.00969677,
     >     0.00930205,0.00880781,0.00819264,0.00742455,0.00646058,
     >     0.00526252,0.00385835,0.00243988,0.00131252,0.000621079,
     >     0.000272322,0.000114993,4.77312e-05,1.96569e-05,8.06377e-06,
     >     3.30066e-06,1.34904e-06,5.50751e-07,2.24633e-07,9.15434e-08,
     >     3.72777e-08,1.51694e-08,6.16885e-09,2.50714e-09,1.01838e-09,
     >     0.0102179,0.0101879,0.0100975,0.00994467,0.00972606,
     >     0.00943628,0.0090671,0.00860605,0.00803416,0.00732289,
     >     0.00643266,0.00532296,0.00400025,0.00261264,0.00144993,
     >     0.00070098,0.000310867,0.000131949,5.48887e-05,2.26247e-05,
     >     9.28457e-06,3.80091e-06,1.55358e-06,6.34273e-07,2.58701e-07,
     >     1.05427e-07,4.29316e-08,1.74701e-08,7.10447e-09,2.88739e-09,
     >     1.17283e-09/
      DATA Q2V
     >     /4.000,   6.000,   9.000,  13.500,  20.250,  30.375,  45.562,
     >     68.344, 102.516, 153.773, 230.660, 345.990, 518.985/
      DATA XB
     >     /0.99999998E-05,  0.13000000E-04,  0.16899999E-04,  
     >     0.21970000E-04,  0.28561000E-04,  0.37129299E-04,  
     >     0.48268099E-04,  0.62748499E-04,  0.81573096E-04,  
     >     0.10604500E-03,  0.13785800E-03,  0.17921600E-03,  
     >     0.23298099E-03,  0.30287501E-03,  0.39373801E-03,  
     >     0.51185902E-03,  0.66541700E-03,  0.86504198E-03,  
     >     0.11245500E-02,  0.14619200E-02,  0.19005000E-02,  
     >     0.24706500E-02,  0.32118401E-02,  0.41753901E-02,  
     >     0.54280101E-02,  0.70564100E-02,  0.91733299E-02,  
     >     0.11925300E-01,  0.15502900E-01,  0.20153800E-01,  
     >     0.26200000E-01,  0.34059901E-01,  0.44277899E-01,  
     >     0.57561301E-01,  0.74829698E-01,  0.97278602E-01/
      DATA G
     >     /2.542140,2.457280,2.372200,2.283920,2.195320,2.105540,
     >     2.014040,1.921480,1.827600,1.733930,1.639990,1.546170,
     >     1.453220,1.361040,1.268750,1.178500,1.088730,1.000740,
     >     0.914606,0.830842,0.749315,0.671419,0.596883,0.526173,
     >     0.460067,0.398489,0.341895,0.289998,0.242628,0.199405,
     >     0.159625,0.122521,0.087252,0.053699,0.022804,0.000414,
     >     1.825600,1.768440,1.712550,1.654660,1.595830,1.533980,
     >     1.472860,1.409910,1.346060,1.282030,1.217520,1.153050,
     >     1.089030,1.025930,0.962709,0.900094,0.837787,0.776733,
     >     0.716072,0.656570,0.598295,0.541607,0.486661,0.433846,
     >     0.383421,0.335585,0.290685,0.248542,0.209305,0.172614,
     >     0.138166,0.105552,0.074246,0.044592,0.017935,0.000291,
     >     1.445600,1.405920,1.364150,1.322050,1.276900,1.232020,
     >     1.184670,1.136620,1.086950,1.037370,0.987225,0.937488,
     >     0.888244,0.839061,0.790370,0.741798,0.693426,0.645511,
     >     0.598298,0.551516,0.505500,0.460400,0.416326,0.373488,
     >     0.332281,0.292614,0.254859,0.218944,0.184976,0.152759,
     >     0.122077,0.092747,0.064477,0.037873,0.014527,0.000216,
     >     1.209700,1.179300,1.146560,1.112700,1.077790,1.041400,
     >     1.003150,0.964598,0.923825,0.882423,0.841304,0.800381,
     >     0.759486,0.718947,0.678587,0.638579,0.598671,0.558978,
     >     0.519524,0.480617,0.442116,0.404231,0.367057,0.330599,
     >     0.295291,0.261100,0.228220,0.196605,0.166367,0.137331,
     >     0.109496,0.082677,0.056835,0.032696,0.012013,0.000165,
     >     1.049390,1.024810,0.998839,0.971705,0.942633,0.912223,
     >     0.880504,0.847236,0.812716,0.777477,0.741670,0.706578,
     >     0.671021,0.636183,0.601384,0.566923,0.532196,0.497956,
     >     0.463884,0.430069,0.396612,0.363536,0.330915,0.298900,
     >     0.267627,0.237155,0.207697,0.179169,0.151566,0.124961,
     >     0.099266,0.074436,0.050544,0.028462,0.010013,0.000127,
     >     0.943086,0.922388,0.900700,0.877793,0.853134,0.826808,
     >     0.798937,0.769284,0.738709,0.706829,0.674818,0.643041,
     >     0.611250,0.580168,0.548982,0.517908,0.486720,0.456021,
     >     0.425312,0.394899,0.364616,0.334682,0.305173,0.276042, 
     >     0.247465,0.219584,0.192474,0.166033,0.140391,0.115481,
     >     0.091347,0.068001,0.045636,0.025180,0.008510,0.000101,
     >     0.858923,0.841883,0.823827,0.803952,0.782124,0.759447,
     >     0.734659,0.708141,0.680281,0.651690,0.622372,0.593451,
     >     0.564610,0.536045,0.507521,0.479248,0.450845,0.422731,
     >     0.394687,0.366774,0.339095,0.311580,0.284350,0.257527,
     >     0.231101,0.205195,0.179929,0.155178,0.131077,0.107555,
     >     0.084712,0.062629,0.041557,0.022497,0.007326,0.000082,
     >     0.790086,0.776082,0.760408,0.743282,0.724441,0.703812,
     >     0.681673,0.657893,0.632354,0.605918,0.578976,0.552364,
     >     0.525820,0.499665,0.473306,0.447125,0.421027,0.394926,
     >     0.369017,0.343250,0.317596,0.292061,0.266805,0.241770,
     >     0.217121,0.192830,0.169123,0.145809,0.122984,0.100685,
     >     0.078967,0.057983,0.038065,0.020236,0.006364,0.000067, 
     >     0.732957,0.720881,0.707633,0.692613,0.676200,0.657914,
     >     0.637626,0.615846,0.592373,0.567813,0.542877,0.518036,
     >     0.493386,0.468899,0.444596,0.420291,0.395908,0.371641,
     >     0.347423,0.323358,0.299396,0.275511,0.251830,0.228313,
     >     0.205146,0.182274,0.159813,0.137693,0.115983,0.094703,
     >     0.073974,0.053979,0.035068,0.018327,0.005580,0.000056,
     >     0.684126,0.674389,0.662432,0.649473,0.634783,0.618458,
     >     0.600173,0.580108,0.558464,0.535421,0.512051,0.488760,
     >     0.465717,0.442837,0.419997,0.397309,0.374393,0.351598,
     >     0.328943,0.306305,0.283685,0.261210,0.238875,0.216696,
     >     0.194737,0.173035,0.151692,0.130607,0.109864,0.089483,
     >     0.069628,0.050489,0.032492,0.016714,0.004940,0.000047,
     >     0.642684,0.634012,0.624311,0.612687,0.599751,0.584885,
     >     0.568165,0.549716,0.529352,0.507582,0.485673,0.463844,
     >     0.442031,0.420480,0.398892,0.377446,0.355834,0.334383,
     >     0.312903,0.291520,0.270128,0.248790,0.227632,0.206518,
     >     0.185624,0.164939,0.144527,0.124348,0.104415,0.084843,
     >     0.065754,0.047403,0.030217,0.015305,0.004396,0.000040,
     >     0.606784,0.599501,0.591216,0.581053,0.569262,0.555644,
     >     0.540298,0.522979,0.503888,0.483449,0.462725,0.441941,
     >     0.421308,0.400965,0.380515,0.360166,0.339667,0.319306,
     >     0.298896,0.278531,0.258224,0.237932,0.217694,0.197574,
     >     0.177567,0.157731,0.138168,0.118766,0.099587,0.080708,
     >     0.062310,0.044656,0.028210,0.014081,0.003935,0.000035,
     >     0.575250,0.569055,0.561834,0.552756,0.542358,0.530173,
     >     0.515880,0.499697,0.481592,0.462197,0.442485,0.422845,
     >     0.403137,0.383704,0.364332,0.344943,0.325467,0.306020,
     >     0.286496,0.267106,0.247644,0.228251,0.208881,0.189557,
     >     0.170377,0.151316,0.132475,0.113748,0.095226,0.076992,
     >     0.059214,0.042195,0.026424,0.013004,0.003541,0.000030,
     >     2.510860,2.424890,2.338800,2.251430,2.161030,2.070080,
     >     1.976500,1.884380,1.789520,1.695430,1.600900,1.507110,
     >     1.413980,1.321430,1.229430,1.138950,1.049380,0.961914,
     >     0.875863,0.792403,0.711728,0.634220,0.560088,0.490193,
     >     0.424574,0.363647,0.307681,0.256400,0.209654,0.167105,
     >     0.128273,0.092911,0.060892,0.033178,0.011804,0.000172,
     >     1.797890,1.740890,1.683730,1.625330,1.564220,1.502940,
     >     1.439440,1.375560,1.311540,1.247060,1.182260,1.117820,
     >     1.053940,0.990598,0.927193,0.864982,0.802868,0.741596,
     >     0.681232,0.621931,0.563990,0.507817,0.452982,0.400714,
     >     0.350809,0.303514,0.259139,0.217505,0.178849,0.142987,
     >     0.109689,0.079022,0.051121,0.027212,0.009208,0.000121,
     >     1.421860,1.380980,1.338550,1.294150,1.249120,1.201850,
     >     1.153940,1.105280,1.054910,1.004680,0.954677,0.905086,
     >     0.855141,0.806467,0.757283,0.708817,0.660443,0.613233,
     >     0.565935,0.519307,0.473770,0.428741,0.385002,0.342676,
     >     0.301797,0.262591,0.225442,0.189974,0.156683,0.125290,
     >     0.095921,0.068630,0.043899,0.022874,0.007411,0.000090,
     >     1.187690,1.155530,1.122280,1.087030,1.051020,1.013270,
     >     0.973852,0.933953,0.892905,0.851713,0.810455,0.769192,
     >     0.728379,0.687798,0.647423,0.607517,0.567475,0.528048,
     >     0.488850,0.450119,0.411897,0.374353,0.337466,0.301396,
     >     0.266522,0.232713,0.200387,0.169339,0.139725,0.111661,
     >     0.085284,0.060612,0.038324,0.019572,0.006096,0.000069,
     >     1.028240,1.002610,0.975714,0.947100,0.916605,0.885287,
     >     0.852214,0.817988,0.782800,0.747390,0.711939,0.676325,
     >     0.641102,0.606200,0.571466,0.536994,0.502546,0.468438,
     >     0.434547,0.400992,0.367663,0.334910,0.302578,0.270890,
     >     0.240102,0.210041,0.181153,0.153175,0.126406,0.100826,
     >     0.076605,0.054074,0.033774,0.016896,0.005055,0.000053,
     >     0.923239,0.901357,0.878307,0.853928,0.827994,0.800448,
     >     0.770833,0.740864,0.709208,0.677436,0.645505,0.613815,
     >     0.582211,0.551121,0.519863,0.488847,0.457919,0.427291,
     >     0.396728,0.366593,0.336576,0.306991,0.277689,0.248974,
     >     0.220914,0.193485,0.166892,0.141077,0.116297,0.092513,
     >     0.069960,0.049024,0.030239,0.014840,0.004277,0.000042,
     >     0.840109,0.821572,0.802221,0.780737,0.758007,0.733570,
     >     0.707391,0.680060,0.651741,0.622598,0.593525,0.564790,
     >     0.535923,0.507607,0.479180,0.450770,0.422703,0.394808,
     >     0.366922,0.339201,0.311718,0.284555,0.257727,0.231240,
     >     0.205291,0.179821,0.155134,0.131145,0.107877,0.085551,
     >     0.064423,0.044815,0.027334,0.013173,0.003668,0.000034,
     >     0.771898,0.756403,0.739454,0.720917,0.700462,0.678667,
     >     0.655360,0.630295,0.604073,0.577412,0.550894,0.524237,
     >     0.497613,0.471552,0.445325,0.419311,0.393354,0.367582,
     >     0.341816,0.316292,0.290960,0.265746,0.240859,0.216190,
     >     0.191974,0.168255,0.145100,0.122534,0.100715,0.079595,
     >     0.059666,0.041209,0.024869,0.011778,0.003176,0.000028,
     >     0.715432,0.701697,0.687335,0.671043,0.652774,0.633196,
     >     0.611618,0.588703,0.564555,0.539941,0.515046,0.490273,
     >     0.465666,0.441417,0.417189,0.393083,0.368842,0.344918,
     >     0.320853,0.296990,0.273346,0.249853,0.226554,0.203403,
     >     0.180727,0.158303,0.136484,0.115160,0.094448,0.074509,
     >     0.055551,0.038122,0.022769,0.010610,0.002775,0.000023,
     >     0.666960,0.655627,0.643169,0.628497,0.612317,0.594375,
     >     0.574670,0.553372,0.531024,0.507805,0.484742,0.461630,
     >     0.438595,0.415762,0.393171,0.370479,0.347803,0.325380,
     >     0.302957,0.280541,0.258199,0.236074,0.214141,0.192375,
     >     0.170880,0.149710,0.129011,0.108739,0.089084,0.069999,
     >     0.052020,0.035461,0.020980,0.009629,0.002450,0.000020,
     >     0.626089,0.616427,0.605000,0.592191,0.577470,0.561221,
     >     0.543098,0.523316,0.502350,0.480560,0.458749,0.436948,
     >     0.415413,0.393933,0.372648,0.351193,0.329853,0.308661,
     >     0.287323,0.266214,0.245090,0.224191,0.203371,0.182736,
     >     0.162324,0.142202,0.122478,0.103078,0.084237,0.066064,
     >     0.048902,0.033114,0.019409,0.008778,0.002175,0.000017,
     >     0.590664,0.582220,0.572479,0.560768,0.547507,0.532490,
     >     0.515657,0.497052,0.477369,0.456893,0.436199,0.415686,
     >     0.395192,0.374819,0.354542,0.334339,0.314120,0.293919,
     >     0.273897,0.253708,0.233747,0.213772,0.193977,0.174220,
     >     0.154737,0.135524,0.116625,0.098102,0.080014,0.062548,
     >     0.046083,0.031041,0.018032,0.008042,0.001942,0.000014,
     >     0.559453,0.552473,0.543647,0.533124,0.521070,0.507229,
     >     0.491452,0.474143,0.455506,0.435945,0.416256,0.396977,
     >     0.377326,0.358052,0.338792,0.319598,0.300262,0.281122,
     >     0.261883,0.242706,0.223579,0.204566,0.185577,0.166710,
     >     0.148088,0.129570,0.111434,0.093593,0.076187,0.059415,
     >     0.043600,0.029214,0.016816,0.007398,0.001744,0.000012,
     >     2.494430,2.407750,2.320680,2.230570,2.139410,2.045820,
     >     1.954300,1.859160,1.764370,1.669990,1.574490,1.481120,
     >     1.386830,1.294720,1.202200,1.111670,1.021960,0.934479,
     >     0.848525,0.765125,0.684539,0.607178,0.533502,0.463759,
     >     0.398700,0.338188,0.282593,0.231728,0.185473,0.143600,
     >     0.105860,0.072262,0.043334,0.020543,0.005838,0.000062,
     >     1.781830,1.724140,1.666660,1.605560,1.544230,1.481280,
     >     1.418440,1.353130,1.288210,1.223120,1.158330,1.093950,
     >     1.029490,0.965787,0.902413,0.840091,0.777784,0.716620,
     >     0.656180,0.597156,0.539338,0.483075,0.428748,0.376710,
     >     0.327154,0.280125,0.236105,0.194961,0.156820,0.121629,
     >     0.089495,0.060685,0.035892,0.016616,0.004508,0.000044,
     >     1.407480,1.365110,1.320730,1.276090,1.228930,1.181910,
     >     1.132930,1.082570,1.032770,0.982143,0.932084,0.882066,
     >     0.832465,0.783250,0.734272,0.685821,0.637505,0.589736,
     >     0.542652,0.496248,0.450545,0.405906,0.362451,0.320187,
     >     0.279650,0.240757,0.203901,0.169026,0.136216,0.105650,
     >     0.077490,0.052176,0.030469,0.013808,0.003599,0.000032,
     >     1.173640,1.140320,1.105880,1.069540,1.032320,0.993897,
     >     0.953913,0.913021,0.871617,0.829958,0.788517,0.747238,
     >     0.706258,0.666094,0.625534,0.585422,0.545512,0.506014,
     >     0.466972,0.428374,0.390208,0.352773,0.316057,0.280282,
     >     0.245648,0.212195,0.180160,0.149604,0.120645,0.093445,
     >     0.068267,0.045631,0.026330,0.011698,0.002940,0.000025,
     >     1.014900,0.987942,0.959684,0.930270,0.899044,0.866678,
     >     0.832642,0.797620,0.762091,0.726232,0.690987,0.655077,
     >     0.620091,0.585191,0.550543,0.515986,0.481432,0.447342,
     >     0.413584,0.380034,0.346933,0.314295,0.282265,0.250834,
     >     0.220232,0.190535,0.162003,0.134530,0.108400,0.083726,
     >     0.060878,0.040604,0.022978,0.010002,0.002422,0.000019,
     >     0.909560,0.887017,0.862965,0.837572,0.810211,0.781875, 
     >     0.751746,0.720666,0.688703,0.656776,0.624847,0.593122,
     >     0.561398,0.530391,0.499138,0.468280,0.437229,0.406831,
     >     0.376391,0.346350,0.316458,0.286984,0.257979,0.229524,
     >     0.201720,0.174606,0.148473,0.123248,0.099111,0.076308,
     >     0.055184,0.036277,0.020402,0.008711,0.002038,0.000015,
     >     0.827323,0.808240,0.787194,0.764785,0.741102,0.715710,
     >     0.688724,0.660472,0.631563,0.602307,0.573344,0.544275, 
     >     0.515643,0.487203,0.458901,0.430660,0.402505,0.374606,
     >     0.346855,0.319408,0.292040,0.265141,0.238553,0.212326,
     >     0.186666,0.161641,0.137398,0.113940,0.091458,0.070180,
     >     0.050481,0.032923,0.018295,0.007672,0.001739,0.000012,
     >     0.759389,0.742705,0.724897,0.705170,0.684041,0.661178,
     >     0.636646,0.610897,0.584460,0.557556,0.530715,0.504308,
     >     0.477779,0.451618,0.425597,0.399640,0.373702,0.348041,
     >     0.322356,0.297020,0.271779,0.246872,0.222152,0.197859,
     >     0.173964,0.150642,0.127976,0.105994,0.084903,0.064934,
     >     0.046463,0.030071,0.016527,0.006814,0.001499,0.000010,
     >     0.703019,0.688814,0.673059,0.655722,0.636535,0.615667,
     >     0.593245,0.569605,0.545086,0.520194,0.495411,0.470863,
     >     0.446201,0.421921,0.397793,0.373652,0.349528,0.325620,
     >     0.301845,0.278188,0.254649,0.231349,0.208318,0.185538,
     >     0.163153,0.141239,0.119901,0.099172,0.079281,0.060422,
     >     0.043027,0.027647,0.015032,0.006099,0.001305,0.000008,
     >     0.655549,0.643053,0.628966,0.613794,0.596440,0.577313,
     >     0.556691,0.534653,0.511805,0.488542,0.465448,0.442353,
     >     0.419418,0.396807,0.374153,0.351526,0.328993,0.306561,
     >     0.284259,0.262079,0.240000,0.218086,0.196361,0.174927,
     >     0.153790,0.133102,0.112916,0.093252,0.074414,0.056530,
     >     0.040069,0.025572,0.013769,0.005504,0.001148,0.000007,
     >     0.614886,0.603957,0.591709,0.577655,0.561834,0.544400,
     >     0.525193,0.504891,0.483435,0.461525,0.439729,0.418166,
     >     0.396478,0.375064,0.353725,0.332565,0.311308,0.290188,
     >     0.269125,0.248171,0.227323,0.206600,0.186070,0.165736,
     >     0.145686,0.125993,0.106807,0.088112,0.070133,0.053112,
     >     0.037464,0.023751,0.012665,0.004991,0.001015,0.000006,
     >     0.579569,0.570010,0.559085,0.546465,0.532125,0.516026,
     >     0.498089,0.478853,0.458681,0.438173,0.417404,0.396958, 
     >     0.376499,0.356301,0.336115,0.316014,0.295930,0.275886,
     >     0.255871,0.236010,0.216223,0.196524,0.176991,0.157626,
     >     0.138515,0.119742,0.101404,0.083545,0.066339,0.050078,
     >     0.035167,0.022152,0.011703,0.004550,0.000904,0.000005,
     >     0.548753,0.540580,0.530520,0.519276,0.506004,0.491061,
     >     0.474436,0.456132,0.436996,0.417488,0.397755,0.378421,
     >     0.359085,0.339851,0.320700,0.301518,0.282357,0.263296,
     >     0.244273,0.225311,0.206421,0.187624,0.168959,0.150424,
     >     0.132145,0.114172,0.096557,0.079446,0.062967,0.047378,
     >     0.033126,0.020736,0.010987,0.004166,0.000809,0.000004,
     >     2.486670,2.401390,2.313150,2.223100,2.132210,2.040620,
     >     1.946240,1.852960,1.758660,1.663790,1.569300,1.475400,
     >     1.381630,1.288980,1.197380,1.106900,1.017420,0.930068,
     >     0.844430,0.760955,0.680549,0.603582,0.530075,0.460570,
     >     0.395641,0.335338,0.279919,0.229212,0.183101,0.141385,
     >     0.103826,0.070754,0.041898,0.019580,0.005430,0.000056,
     >     1.776150,1.719900,1.661320,1.600440,1.539110,1.477010,
     >     1.412790,1.348550,1.283220,1.218900,1.153830,1.088770,
     >     1.025080,0.961812,0.898582,0.836002,0.773940,0.712963,
     >     0.652650,0.593610,0.536099,0.480001,0.425916,0.373940,
     >     0.324454,0.277574,0.233716,0.192654,0.154624,0.119622,
     >     0.087676,0.059112,0.034659,0.015816,0.004189,0.000039,
     >     1.403340,1.361000,1.316930,1.271610,1.225380,1.177160,
     >     1.128240,1.078580,1.028640,0.977947,0.928063,0.878521,
     >     0.828550,0.779706,0.730569,0.682199,0.633948,0.586522,
     >     0.539401,0.493171,0.447639,0.403145,0.359673,0.317729,
     >     0.277222,0.238454,0.201716,0.166913,0.134251,0.103840,
     >     0.075864,0.050772,0.029388,0.013129,0.003341,0.000029,
     >     1.169950,1.136380,1.102100,1.065820,1.028190,0.989956,
     >     0.950189,0.908974,0.867893,0.826368,0.784940,0.743815,
     >     0.702872,0.662550,0.622187,0.582284,0.542431,0.503183,
     >     0.464112,0.425583,0.387521,0.350185,0.313576,0.277943,
     >     0.243398,0.210036,0.178125,0.147636,0.118821,0.091774,
     >     0.066794,0.044372,0.025371,0.011112,0.002727,0.000022,
     >     1.011370,0.984724,0.956435,0.926814,0.895443,0.862964,
     >     0.829274,0.794479,0.758875,0.722999,0.687335,0.652160,
     >     0.616882,0.582106,0.547373,0.512881,0.478596,0.444627,
     >     0.410813,0.377446,0.344502,0.311951,0.279953,0.248594,
     >     0.218124,0.188529,0.160091,0.132732,0.106696,0.082174,
     >     0.059499,0.039204,0.022122,0.009492,0.002245,0.000017,
     >     0.906689,0.883943,0.860162,0.834079,0.807067,0.778589,
     >     0.748446,0.717460,0.685716,0.653595,0.621726,0.590095,
     >     0.558631,0.527321,0.496368,0.465511,0.434680,0.404249,
     >     0.373821,0.343908,0.314123,0.284707,0.255819,0.227414,
     >     0.199673,0.172714,0.146637,0.121510,0.097519,0.074849,
     >     0.053902,0.035215,0.019623,0.008259,0.001888,0.000014,
     >     0.824084,0.805081,0.784198,0.761980,0.737877,0.712401,
     >     0.685185,0.657419,0.628573,0.599438,0.570456,0.541554,
     >     0.512854,0.484478,0.456174,0.428019,0.399930,0.372215,
     >     0.344523,0.317092,0.289826,0.262998,0.236396,0.210320,
     >     0.184715,0.159801,0.135667,0.112306,0.089938,0.068791,
     >     0.049274,0.031936,0.017585,0.007270,0.001610,0.000011, 
     >     0.756740,0.739909,0.722048,0.702184,0.681164,0.658293,
     >     0.633526,0.608140,0.581469,0.554572,0.527904,0.501517,
     >     0.475145,0.448970,0.422970,0.397121,0.371248,0.345564,
     >     0.320059,0.294788,0.269623,0.244755,0.220148,0.195923,
     >     0.172134,0.148871,0.126312,0.104417,0.083450,0.063624,
     >     0.045337,0.029156,0.015873,0.006450,0.001387,0.000009,
     >     0.700680,0.686067,0.670562,0.652740,0.633579,0.612848,
     >     0.590476,0.566706,0.542207,0.517556,0.492696,0.468105,
     >     0.443562,0.419350,0.395237,0.371204,0.347185,0.323339,
     >     0.299543,0.276011,0.252614,0.229355,0.206336,0.183677,
     >     0.161386,0.139527,0.118283,0.097686,0.077903,0.059183,
     >     0.041956,0.026789,0.014429,0.005771,0.001207,0.000008,
     >     0.652991,0.640812,0.626563,0.610980,0.593505,0.574586,
     >     0.553939,0.531965,0.509196,0.485938,0.462796,0.439793,
     >     0.416876,0.394269,0.371716,0.349126,0.326720,0.304355,
     >     0.282067,0.259966,0.237971,0.216140,0.194490,0.173138,
     >     0.152089,0.131483,0.111379,0.091839,0.073081,0.055347,
     >     0.039055,0.024764,0.013208,0.005204,0.001061,0.000006,
     >     0.612493,0.601527,0.589260,0.574987,0.559293,0.541793,
     >     0.522714,0.502071,0.480580,0.459021,0.437202,0.415483,
     >     0.394052,0.372703,0.351511,0.330271,0.309080,0.288012,
     >     0.267062,0.246156,0.225353,0.204703,0.184235,0.163954,
     >     0.144013,0.124421,0.105317,0.086709,0.068861,0.051973,
     >     0.036498,0.022991,0.012143,0.004716,0.000938,0.000005,
     >     0.577224,0.567718,0.556768,0.543992,0.529614,0.513264,
     >     0.495593,0.476332,0.456118,0.435643,0.414958,0.394475,
     >     0.374159,0.353948,0.333917,0.313860,0.293761,0.273842,
     >     0.253865,0.234066,0.214285,0.194711,0.175202,0.155932,
     >     0.136903,0.118198,0.099962,0.082192,0.065107,0.048988,
     >     0.034247,0.021430,0.011215,0.004297,0.000835,0.000005,
     >     0.546580,0.538342,0.528410,0.516932,0.503593,0.488594,
     >     0.471844,0.453700,0.434588,0.415081,0.395496,0.376054,
     >     0.356702,0.337597,0.318488,0.299383,0.280225,0.261271,
     >     0.242274,0.223404,0.204576,0.185839,0.167243,0.148787,
     >     0.130586,0.112676,0.095184,0.078140,0.061781,0.046329,
     >     0.032244,0.020050,0.010400,0.003932,0.000747,0.000004/
      DATA LQ
     >     /0.514104,0.515832,0.517091,0.517536,0.517832,0.517221,
     >     0.516574,0.515224,0.513287,0.510858,0.508000,0.504917,
     >     0.501246,0.497403,0.492983,0.488152,0.482358,0.476122,
     >     0.468649,0.460339,0.450645,0.439812,0.427166,0.412836,
     >     0.396439,0.377753,0.356388,0.331566,0.302870,0.269780,
     >     0.231439,0.187668,0.138736,0.086850,0.036211,0.000604,
     >     0.579741,0.577476,0.575013,0.571386,0.567464,0.562828,
     >     0.557414,0.551410,0.544332,0.536891,0.528953,0.521049,
     >     0.512612,0.503967,0.494945,0.485619,0.475531,0.464958,
     >     0.453985,0.442176,0.429817,0.416392,0.402026,0.386267,
     >     0.369251,0.350413,0.329631,0.306106,0.279316,0.248539,
     >     0.213030,0.172492,0.127054,0.078865,0.032177,0.000513,
     >     0.617205,0.612889,0.607448,0.601280,0.593987,0.585725,
     >     0.577239,0.567645,0.557436,0.546468,0.535124,0.523690,
     >     0.511913,0.500192,0.488271,0.476132,0.463482,0.450653,
     >     0.437460,0.423920,0.409767,0.395178,0.379804,0.363620,
     >     0.346492,0.327958,0.307882,0.285606,0.260283,0.231441,
     >     0.198228,0.160224,0.117620,0.072457,0.029003,0.000445,
     >     0.635918,0.629589,0.622092,0.614001,0.604751,0.594717,
     >     0.583683,0.572041,0.559188,0.546115,0.532523,0.518899,
     >     0.505201,0.491782,0.477675,0.463752,0.449738,0.435623,
     >     0.421062,0.406484,0.391587,0.376279,0.360631,0.344349,
     >     0.327299,0.309280,0.289970,0.268596,0.244665,0.217365,
     >     0.185991,0.150104,0.109826,0.067166,0.026423,0.000392,
     >     0.642211,0.634584,0.625818,0.616705,0.605951,0.594718,
     >     0.582351,0.569037,0.554986,0.540276,0.525303,0.510521,
     >     0.495596,0.480637,0.465827,0.451058,0.435872,0.421066,
     >     0.405880,0.390796,0.375397,0.359903,0.344185,0.327987,
     >     0.311213,0.293614,0.274949,0.254454,0.231575,0.205516,
     >     0.175652,0.141477,0.103138,0.062618,0.024219,0.000348,
     >     0.639180,0.631092,0.621893,0.612137,0.600935,0.589006,
     >     0.576122,0.561754,0.547048,0.531226,0.515512,0.499878,
     >     0.484089,0.468662,0.453258,0.437735,0.422292,0.407026,
     >     0.391544,0.376258,0.360735,0.345208,0.329470,0.313579,
     >     0.297158,0.280066,0.261996,0.242200,0.220305,0.195335,
     >     0.166765,0.134023,0.097367,0.058717,0.022356,0.000312,
     >     0.631156,0.622788,0.613580,0.603527,0.592277,0.579779,
     >     0.566484,0.551813,0.536423,0.520508,0.504251,0.488021,
     >     0.472093,0.456329,0.440551,0.424908,0.409169,0.393745,
     >     0.378229,0.362803,0.347473,0.331992,0.316442,0.300771,
     >     0.284762,0.268174,0.250682,0.231645,0.210500,0.186498,
     >     0.158976,0.127569,0.092372,0.055348,0.020770,0.000282,
     >     0.619698,0.611519,0.602336,0.591943,0.580814,0.568514,
     >     0.554982,0.540509,0.524970,0.508623,0.492412,0.476046,
     >     0.459935,0.444036,0.428166,0.412462,0.396815,0.381305,
     >     0.365892,0.350546,0.335395,0.320121,0.304877,0.289437,
     >     0.273841,0.257707,0.240774,0.222305,0.201894,0.178726,
     >     0.152209,0.121898,0.087984,0.052399,0.019398,0.000257,
     >     0.606343,0.598499,0.589365,0.579821,0.568434,0.556327,
     >     0.542872,0.528454,0.512900,0.496651,0.480390,0.464149,
     >     0.447972,0.432310,0.416261,0.400748,0.385116,0.369727,
     >     0.354542,0.339434,0.324280,0.309328,0.294433,0.279362,
     >     0.264086,0.248363,0.231853,0.214025,0.194261,0.171833,
     >     0.146197,0.116876,0.084113,0.049805,0.018206,0.000236,
     >     0.591896,0.584256,0.575797,0.566248,0.555494,0.543797,
     >     0.530664,0.516396,0.501105,0.484951,0.468541,0.452368,
     >     0.436358,0.420764,0.405075,0.389617,0.374238,0.359020,
     >     0.343972,0.329043,0.314268,0.299607,0.284884,0.270198,
     >     0.255337,0.239949,0.223985,0.206644,0.187466,0.165701,
     >     0.140843,0.112412,0.080664,0.047512,0.017165,0.000218,
     >     0.576697,0.569585,0.561998,0.552972,0.542790,0.531174,
     >     0.518335,0.504287,0.489160,0.473285,0.457139,0.441230,
     >     0.425520,0.409903,0.394534,0.379177,0.364141,0.349046,
     >     0.334214,0.319602,0.305086,0.290690,0.276241,0.261890,
     >     0.247381,0.232411,0.216817,0.199969,0.181340,0.160182,
     >     0.135997,0.108375,0.077551,0.045446,0.016236,0.000202,
     >     0.561779,0.555491,0.548244,0.539719,0.529917,0.518731,
     >     0.506334,0.492791,0.477798,0.462049,0.446208,0.430534,
     >     0.414938,0.399619,0.384500,0.369572,0.354611,0.339912,
     >     0.325296,0.310898,0.296649,0.282465,0.268439,0.254375,
     >     0.240158,0.225578,0.210376,0.193910,0.175777,0.155185,
     >     0.131618,0.104727,0.074745,0.043586,0.015406,0.000188,
     >     0.547406,0.541522,0.534772,0.526637,0.517320,0.506665,
     >     0.494929,0.481541,0.466958,0.451581,0.435977,0.420382,
     >     0.405163,0.390039,0.375242,0.360562,0.345699,0.331352,
     >     0.316934,0.302875,0.288897,0.275075,0.261211,0.247494,
     >     0.233546,0.219335,0.204477,0.188465,0.170694,0.150613,
     >     0.127623,0.101395,0.072188,0.041898,0.014659,0.000176,
     >     0.510949,0.512461,0.513125,0.513775,0.513627,0.512953,
     >     0.511676,0.510153,0.508026,0.505586,0.502488,0.499514,
     >     0.495918,0.491777,0.487221,0.482318,0.476322,0.469865,
     >     0.462254,0.453539,0.443579,0.431957,0.418405,0.402714,
     >     0.384513,0.363211,0.338441,0.309525,0.276023,0.237418,
     >     0.194095,0.146991,0.098692,0.053921,0.018675,0.000251,
     >     0.573840,0.571539,0.568660,0.564636,0.560125,0.555429,
     >     0.549250,0.542654,0.535827,0.528339,0.520207,0.512007,
     >     0.503505,0.494854,0.485721,0.476289,0.466253,0.455649,
     >     0.444534,0.432471,0.419685,0.405709,0.390526,0.373656,
     >     0.355038,0.334208,0.310578,0.283443,0.252376,0.217005,
     >     0.177144,0.133976,0.089700,0.048645,0.016531,0.000213,
     >     0.610274,0.604837,0.598760,0.591967,0.584049,0.575825,
     >     0.566634,0.556589,0.545957,0.534847,0.523533,0.511998,
     >     0.500328,0.488702,0.476646,0.464532,0.451851,0.438900,
     >     0.425698,0.412022,0.397498,0.382650,0.366763,0.349574,
     >     0.330976,0.310680,0.288132,0.262614,0.233680,0.200673,
     >     0.163728,0.123628,0.082475,0.044455,0.014854,0.000185,
     >     0.627192,0.620322,0.611968,0.603362,0.593426,0.582683,
     >     0.571079,0.558798,0.545824,0.532347,0.519069,0.505318,
     >     0.491597,0.478201,0.464023,0.450314,0.436183,0.422097,
     >     0.407576,0.392963,0.377856,0.362363,0.346096,0.329071,
     >     0.310942,0.291304,0.269699,0.245618,0.218210,0.187279,
     >     0.152695,0.115069,0.076563,0.041003,0.013496,0.000163,
     >     0.631625,0.623836,0.614536,0.604467,0.593362,0.581189,
     >     0.568126,0.554473,0.540005,0.525342,0.510320,0.495556,
     >     0.480543,0.465856,0.450901,0.436047,0.421171,0.406166,
     >     0.391154,0.375993,0.360616,0.344868,0.328695,0.311826,
     >     0.294125,0.275057,0.254485,0.231491,0.205428,0.176164,
     >     0.143362,0.107930,0.071534,0.038049,0.012338,0.000145,
     >     0.628674,0.619289,0.609887,0.598892,0.587603,0.574461,
     >     0.560634,0.545957,0.530648,0.515019,0.499229,0.483499,
     >     0.467857,0.452587,0.437187,0.421817,0.406489,0.391076,
     >     0.375987,0.360550,0.345003,0.329399,0.313389,0.296812,
     >     0.279488,0.261191,0.241276,0.219209,0.194576,0.166551,
     >     0.135508,0.101696,0.067217,0.035540,0.011363,0.000130,
     >     0.619655,0.610842,0.600741,0.589506,0.577444,0.564098,
     >     0.550163,0.534861,0.519180,0.503427,0.486955,0.471011,
     >     0.455040,0.439305,0.423732,0.408132,0.392629,0.377169,
     >     0.361831,0.346498,0.331073,0.315724,0.299859,0.283638,
     >     0.266810,0.249063,0.229985,0.208716,0.185019,0.158309,
     >     0.128580,0.096355,0.063475,0.033376,0.010535,0.000117,
     >     0.607474,0.598863,0.589070,0.578071,0.565959,0.552353,
     >     0.538238,0.523141,0.507028,0.491211,0.474478,0.458296,
     >     0.442260,0.426619,0.410731,0.395250,0.379629,0.364356,
     >     0.348971,0.333841,0.318608,0.303265,0.287813,0.272025,
     >     0.255702,0.238406,0.219910,0.199549,0.176755,0.151092,
     >     0.122529,0.091666,0.060228,0.031496,0.009822,0.000107,
     >     0.593760,0.585285,0.576038,0.564716,0.553007,0.539720,
     >     0.525715,0.510544,0.494716,0.478059,0.461888,0.445882,
     >     0.429808,0.413922,0.398545,0.382937,0.367557,0.352300,
     >     0.337238,0.322197,0.307168,0.292182,0.277095,0.261674,
     >     0.245761,0.228939,0.211113,0.191442,0.169413,0.144749,
     >     0.117225,0.087581,0.057361,0.029844,0.009202,0.000098,
     >     0.579324,0.570883,0.561883,0.551297,0.539642,0.526840,
     >     0.512914,0.497903,0.481998,0.465838,0.449788,0.433693,
     >     0.417703,0.402286,0.386792,0.371437,0.356215,0.341283,
     >     0.326313,0.311637,0.296947,0.282140,0.267321,0.252270,
     >     0.236761,0.220580,0.203236,0.184230,0.162954,0.139137,
     >     0.112557,0.083990,0.054824,0.028391,0.008663,0.000090,
     >     0.564315,0.556562,0.547782,0.537905,0.526572,0.514011,
     >     0.500033,0.485378,0.470013,0.453915,0.438022,0.422153,
     >     0.406530,0.391112,0.375843,0.360792,0.345939,0.331089,
     >     0.316383,0.301913,0.287442,0.273075,0.258564,0.243883,
     >     0.228827,0.213013,0.196181,0.177712,0.157167,0.134036,
     >     0.108376,0.080670,0.052555,0.027085,0.008183,0.000084,
     >     0.549474,0.542540,0.534001,0.524222,0.513538,0.501429,
     >     0.487998,0.473665,0.458530,0.442760,0.426794,0.411187,
     >     0.395839,0.380807,0.365652,0.350917,0.336154,0.321565,
     >     0.307328,0.293072,0.278886,0.264801,0.250668,0.236339,
     >     0.221517,0.206147,0.189835,0.171902,0.151860,0.129450,
     >     0.104545,0.077707,0.050505,0.025913,0.007753,0.000078,
     >     0.535066,0.528539,0.520546,0.511350,0.500893,0.489256,
     >     0.476407,0.462092,0.447135,0.431648,0.416302,0.400932,
     >     0.385885,0.370871,0.356128,0.341562,0.327354,0.312933,
     >     0.298890,0.284864,0.270948,0.257177,0.243326,0.229321,
     >     0.214967,0.199986,0.183994,0.166574,0.147078,0.125313,
     >     0.101089,0.075065,0.048640,0.024855,0.007368,0.000073,
     >     0.509409,0.510722,0.511674,0.511738,0.511286,0.510766,
     >     0.509198,0.507564,0.505152,0.502704,0.499650,0.496452,
     >     0.492756,0.488747,0.483986,0.478817,0.472835,0.466164,
     >     0.458412,0.449362,0.438892,0.426796,0.412430,0.395391,
     >     0.375497,0.352110,0.324651,0.292275,0.254919,0.212461,
     >     0.165886,0.117521,0.071431,0.033526,0.009193,0.000091,
     >     0.570888,0.568118,0.564921,0.561074,0.555970,0.550478,
     >     0.544438,0.537826,0.530394,0.523005,0.514685,0.506342,
     >     0.497778,0.489102,0.479852,0.470348,0.460202,0.449558,
     >     0.438121,0.426046,0.412891,0.398444,0.382493,0.364765,
     >     0.344689,0.321925,0.295886,0.265915,0.231610,0.192817,
     >     0.150406,0.106352,0.064426,0.030028,0.008099,0.000077,
     >     0.605452,0.599834,0.593296,0.586627,0.578039,0.569200,
     >     0.559754,0.549624,0.538672,0.527535,0.516054,0.504396,
     >     0.492657,0.481104,0.468974,0.456682,0.443949,0.431100,
     >     0.417683,0.403750,0.389161,0.373852,0.357151,0.339250,
     >     0.319633,0.297684,0.273111,0.245042,0.213167,0.177272,
     >     0.138115,0.097498,0.058870,0.027271,0.007249,0.000067,
     >     0.621534,0.614157,0.605835,0.596272,0.585946,0.574335,
     >     0.562822,0.550233,0.536841,0.523598,0.509920,0.496293,
     >     0.482476,0.468781,0.455150,0.441275,0.427053,0.412946,
     >     0.398426,0.383702,0.368436,0.352438,0.335827,0.317985,
     >     0.298878,0.277758,0.254411,0.228040,0.198146,0.164659,
     >     0.128112,0.090261,0.054331,0.025017,0.006563,0.000059,
     >     0.625646,0.616575,0.607120,0.596395,0.584780,0.572447,
     >     0.558731,0.544391,0.529948,0.515041,0.499832,0.485344,
     >     0.470145,0.455555,0.440783,0.425851,0.410922,0.396061,
     >     0.381038,0.365825,0.350234,0.334195,0.317648,0.300202,
     >     0.281570,0.261329,0.238985,0.213898,0.185675,0.154079,
     >     0.119726,0.084159,0.050488,0.023108,0.005981,0.000052,
     >     0.621816,0.611958,0.601503,0.590515,0.577713,0.564667,
     >     0.550311,0.535181,0.519529,0.503933,0.488235,0.472641,
     >     0.456837,0.441572,0.426210,0.410935,0.395680,0.380387,
     >     0.365098,0.349710,0.334114,0.318243,0.301798,0.284744,
     >     0.266642,0.247149,0.225778,0.201848,0.175065,0.145114,
     >     0.112575,0.078976,0.047163,0.021485,0.005492,0.000047,
     >     0.612456,0.602615,0.592112,0.580629,0.567547,0.554000,
     >     0.539266,0.523505,0.507494,0.491317,0.475170,0.459335,
     >     0.443356,0.427588,0.412089,0.396613,0.381018,0.365836,
     >     0.350418,0.335168,0.319712,0.303995,0.287942,0.271269,
     >     0.253737,0.234846,0.214374,0.191514,0.165911,0.137420,
     >     0.106436,0.074522,0.044401,0.020096,0.005079,0.000042,
     >     0.599906,0.590658,0.579871,0.568198,0.555552,0.541752,
     >     0.526573,0.511088,0.494690,0.478321,0.462067,0.446064,
     >     0.430078,0.414174,0.398700,0.383180,0.367731,0.352392,
     >     0.337125,0.322012,0.306729,0.291406,0.275582,0.259369,
     >     0.242384,0.224158,0.204420,0.182500,0.157984,0.130669,
     >     0.101087,0.070603,0.041960,0.018893,0.004724,0.000039,
     >     0.586385,0.576832,0.566175,0.554843,0.542371,0.528570,
     >     0.513481,0.497995,0.481584,0.465454,0.449166,0.432955,
     >     0.417165,0.401569,0.386012,0.370517,0.355274,0.340119,
     >     0.324992,0.310107,0.295066,0.280058,0.264679,0.248876,
     >     0.232336,0.214712,0.195681,0.174542,0.150960,0.124776,
     >     0.096383,0.067222,0.039821,0.017842,0.004416,0.000035,
     >     0.571562,0.562230,0.553483,0.541106,0.528820,0.515154,
     >     0.500724,0.484959,0.468895,0.452796,0.436478,0.420645,
     >     0.404877,0.389321,0.373971,0.358762,0.343687,0.328774,
     >     0.313968,0.299216,0.284619,0.269828,0.254780,0.239398,
     >     0.223345,0.206321,0.187886,0.167537,0.144752,0.119523,
     >     0.092229,0.064212,0.037929,0.016920,0.004149,0.000033,
     >     0.556430,0.547716,0.538118,0.527507,0.515330,0.502167,
     >     0.487523,0.472378,0.456637,0.440513,0.424536,0.408900,
     >     0.393308,0.378059,0.362874,0.347899,0.333001,0.318366,
     >     0.303866,0.289409,0.275001,0.260570,0.245889,0.230899,
     >     0.215327,0.198784,0.180938,0.161191,0.139240,0.114867,
     >     0.088515,0.061527,0.036242,0.016094,0.003912,0.000030,
     >     0.541088,0.533114,0.524472,0.513843,0.502197,0.489243,
     >     0.475392,0.460194,0.444773,0.428968,0.413337,0.397800,
     >     0.382488,0.367387,0.352568,0.337799,0.323189,0.308771,
     >     0.294545,0.280383,0.266247,0.252107,0.237863,0.223231,
     >     0.208050,0.191998,0.174706,0.155516,0.134237,0.110630,
     >     0.085174,0.059101,0.034727,0.015356,0.003700,0.000028,
     >     0.526746,0.519306,0.510794,0.500680,0.489449,0.477026,
     >     0.463177,0.448622,0.433350,0.417867,0.402498,0.387171,
     >     0.372235,0.357466,0.342866,0.328463,0.314082,0.299986,
     >     0.285991,0.272106,0.258305,0.244422,0.230511,0.216246,
     >     0.201466,0.185822,0.169011,0.150402,0.129717,0.106819,
     >     0.082146,0.056897,0.033348,0.014690,0.003511,0.000026,
     >     0.508294,0.509537,0.510232,0.510364,0.510028,0.509066,
     >     0.507574,0.506096,0.503764,0.501269,0.498261,0.494940,
     >     0.491357,0.487139,0.482585,0.477518,0.471487,0.464685,
     >     0.456954,0.448138,0.437547,0.425359,0.410953,0.393893,
     >     0.373877,0.350282,0.322653,0.290145,0.252556,0.209889,
     >     0.163221,0.114885,0.069163,0.031966,0.008546,0.000081,
     >     0.569358,0.566803,0.562983,0.559001,0.554723,0.548955,
     >     0.542742,0.536150,0.528626,0.520981,0.513074,0.504898,
     >     0.496364,0.487653,0.478428,0.468815,0.458593,0.447996,
     >     0.436563,0.424615,0.411428,0.396895,0.381014,0.363131,
     >     0.342953,0.320173,0.294051,0.263852,0.229272,0.190367,
     >     0.147875,0.103902,0.062322,0.028615,0.007526,0.000069,
     >     0.603670,0.598438,0.591834,0.584727,0.576222,0.567680,
     >     0.558196,0.547591,0.536787,0.525654,0.514148,0.502844,
     >     0.491102,0.479283,0.467390,0.455046,0.442422,0.429358,
     >     0.416062,0.402212,0.387705,0.372222,0.355668,0.337678,
     >     0.318005,0.295922,0.271145,0.242997,0.210936,0.174968,
     >     0.135724,0.095201,0.057278,0.025971,0.006733,0.000060,
     >     0.619938,0.612660,0.603237,0.594399,0.584220,0.572945,
     >     0.560906,0.547974,0.535000,0.521640,0.508074,0.494503,
     >     0.480788,0.467052,0.453309,0.439430,0.425326,0.411154,
     >     0.396834,0.381957,0.366731,0.350903,0.334155,0.316435,
     >     0.297197,0.275976,0.252539,0.225982,0.196020,0.162415,
     >     0.125838,0.088092,0.052512,0.023816,0.006094,0.000053,
     >     0.623752,0.615162,0.604853,0.594199,0.582617,0.570319,
     >     0.556672,0.542512,0.527993,0.513108,0.498185,0.483234,
     >     0.468378,0.453629,0.438899,0.424139,0.409158,0.394331,
     >     0.379329,0.364110,0.348592,0.332662,0.316045,0.298521,
     >     0.279860,0.259548,0.237136,0.211942,0.183600,0.151947,
     >     0.117534,0.082104,0.048767,0.021982,0.005551,0.000047,
     >     0.619727,0.610281,0.600069,0.588100,0.575830,0.562394,
     >     0.548038,0.533038,0.517667,0.501730,0.486231,0.470564,
     >     0.454906,0.439606,0.424254,0.409150,0.393803,0.378587,
     >     0.363406,0.347957,0.332413,0.316594,0.300201,0.283135,
     >     0.264987,0.245411,0.223977,0.199987,0.173072,0.143020,
     >     0.110486,0.077018,0.045586,0.020431,0.005096,0.000042,
     >     0.610495,0.600588,0.589924,0.578181,0.565270,0.551795,
     >     0.537196,0.521624,0.505437,0.489150,0.472983,0.457321,
     >     0.441422,0.425796,0.410213,0.394710,0.379352,0.363971,
     >     0.348802,0.333422,0.317927,0.302300,0.286286,0.269619,
     >     0.252021,0.233168,0.212605,0.189631,0.163957,0.135368,
     >     0.104416,0.072632,0.042854,0.019101,0.004711,0.000038,
     >     0.597856,0.588522,0.577800,0.566289,0.553520,0.539436,
     >     0.524435,0.508812,0.492453,0.476231,0.459950,0.443865,
     >     0.427942,0.412263,0.396763,0.381179,0.365851,0.350734,
     >     0.335459,0.320164,0.305048,0.289712,0.274016,0.257736,
     >     0.240735,0.222511,0.202688,0.180671,0.156046,0.128701,
     >     0.099140,0.068812,0.040482,0.017948,0.004381,0.000035,
     >     0.584084,0.574752,0.564446,0.552705,0.540225,0.526223,
     >     0.511434,0.495863,0.479694,0.463424,0.447226,0.431058,
     >     0.415167,0.399560,0.384026,0.368648,0.353514,0.338375,
     >     0.323406,0.308441,0.293400,0.278400,0.263020,0.247257,
     >     0.230713,0.213051,0.193943,0.172756,0.149112,0.122832,
     >     0.094502,0.065475,0.038404,0.016946,0.004094,0.000032,
     >     0.569517,0.560385,0.550236,0.539113,0.526421,0.513222,
     >     0.498378,0.482790,0.466696,0.450744,0.434441,0.418603,
     >     0.402901,0.387328,0.372129,0.356950,0.341855,0.326972,
     >     0.312231,0.297539,0.282901,0.268148,0.253147,0.237801,
     >     0.221735,0.204645,0.186195,0.165769,0.142932,0.117672,
     >     0.090399,0.062530,0.036573,0.016065,0.003846,0.000029,
     >     0.554361,0.545724,0.536138,0.525127,0.513348,0.500011,
     >     0.485538,0.470383,0.454424,0.438520,0.422677,0.406888,
     >     0.391373,0.376154,0.360890,0.345988,0.331296,0.316666,
     >     0.302083,0.287727,0.273311,0.258904,0.244293,0.229265,
     >     0.213676,0.197142,0.179286,0.159481,0.137449,0.113024,
     >     0.086731,0.059884,0.034929,0.015279,0.003625,0.000027,
     >     0.539566,0.531375,0.522021,0.511690,0.499982,0.487119,
     >     0.474212,0.458150,0.442576,0.426822,0.411177,0.395600,
     >     0.380520,0.365481,0.350710,0.336033,0.321423,0.307074,
     >     0.292885,0.278746,0.264660,0.250555,0.236263,0.221649,
     >     0.206449,0.190350,0.173011,0.154255,0.132811,0.108865,
     >     0.083444,0.057506,0.033461,0.014573,0.003428,0.000025,
     >     0.524865,0.517319,0.508721,0.498692,0.487348,0.474698,
     >     0.461245,0.446504,0.431299,0.416072,0.400507,0.385278,
     >     0.370252,0.355605,0.341042,0.326661,0.312354,0.298301,
     >     0.284289,0.270473,0.256711,0.242903,0.228985,0.214684,
     >     0.199885,0.184233,0.167367,0.148740,0.128003,0.105093,
     >     0.080442,0.055358,0.032128,0.013936,0.003252,0.000024/
            
C     IF (X.GE.XMAXX.OR.Q2.LT.Q2MIN.OR.Q2.GT.Q2MAX) THEN
C        WRITE(*,*) 'X or Q2 out of range!!'
C        WRITE(*,*) 'Valid range is:'
C        WRITE(*,*) '1E-05 < X  < 0.95 and'
C        WRITE(*,*) '4     < Q2 < 520 GeV**2'
C        RETURN
C     END IF
      
      IF (X.GE.XMAX.OR.B.GT.BMAX) THEN
         RES(1) = 1.0
         RES(2) = 1.0
         RETURN
      END IF
      IF(Q2.LT.Q2MIN) THEN
         Q2 = Q2MIN
      ELSE IF(Q2.GT.Q2MAX) THEN
         Q2 = Q2MAX
      ENDIF
      
      CALL GGINTER(INUCL,X,Q2,SHAD)

      DO I=1,31
         TATMP(I) = TA(I,INUCL)
      ENDDO
      CALL SPLINE(TMAX,IMPAR,TATMP,C,D,E)
      TAF = SEVAL(TMAX,B,IMPAR,TATMP,C,D,E)
      
      DO IK=1,2
         RES(IK) = 1.0/(1.0 + (ANUCL(INUCL)-1.0)*SHAD(IK)*TAF)
      ENDDO
      
      RETURN
      END

      SUBROUTINE GGINTER(INUCL,X,Q2,SHAD)
      IMPLICIT DOUBLE PRECISION(A-H,L-Z)
      IMPLICIT INTEGER(I-K)

      DIMENSION SHAD(2)
      DIMENSION G(36,13,4),LQ(36,13,4)
      DIMENSION XB(36),Q2V(13)
      DIMENSION GQTMP(13),LQQTMP(13)
      DIMENSION GXTMP(36),LQXTMP(36)
      DIMENSION C(100),D(100),E(100)

      PARAMETER(INMAX=13,IMMAX=36)

      COMMON/GG07/ XB,Q2V,G,LQ

      DO K=1,36
         DO J=1,13
            GQTMP(J)  = G(K,J,INUCL)
            LQQTMP(J) = LQ(K,J,INUCL)
         ENDDO
         CALL SPLINE(INMAX,Q2V,GQTMP,C,D,E)
         GXTMP(K)  = SEVAL(INMAX,Q2,Q2V,GQTMP,C,D,E)
         CALL SPLINE(INMAX,Q2V,LQQTMP,C,D,E)
         LQXTMP(K)  = SEVAL(INMAX,Q2,Q2V,LQQTMP,C,D,E)
      ENDDO
      
      CALL SPLINE(IMMAX,XB,GXTMP,C,D,E)
      SHAD(1) = SEVAL(IMMAX,X,XB,GXTMP,C,D,E)
      CALL SPLINE(IMMAX,XB,LQXTMP,C,D,E)
      SHAD(2) = SEVAL(IMMAX,X,XB,LQXTMP,C,D,E)

      RETURN
      END

C ---------------------------------------------------------------------
      SUBROUTINE SPLINE(N,X,Y,B,C,D)
C ---------------------------------------------------------------------
c***************************************************************************
C     CALCULATE THE COEFFICIENTS B,C,D IN A CUBIC SPLINE INTERPOLATION.
C     INTERPOLATION SUBROUTINES ARE TAKEN FROM
C     G.E. FORSYTHE, M.A. MALCOLM AND C.B. MOLER,
C     COMPUTER METHODS FOR MATHEMATICAL COMPUTATIONS (PRENTICE-HALL, 1977).
      IMPLICIT double precision (A-H,O-Z)
      DIMENSION X(N),Y(N),B(N),C(N),D(N)
      NM1=N-1
      IF(N.LT.2) RETURN
      IF(N.LT.3) GO TO 250
      D(1)=X(2)-X(1)
      C(2)=(Y(2)-Y(1))/D(1)
      DO 210 I=2,NM1
        D(I)=X(I+1)-X(I)
        B(I)=2.0D0*(D(I-1)+D(I))
        C(I+1)=(Y(I+1)-Y(I))/D(I)
        C(I)=C(I+1)-C(I)
 210  CONTINUE
      B(1)=-D(1)
      B(N)=-D(N-1)
      C(1)=0.0D0
      C(N)=0.0D0
      IF(N.EQ.3) GO TO 215
      C(1)=C(3)/(X(4)-X(2))-C(2)/(X(3)-X(1))
      C(N)=C(N-1)/(X(N)-X(N-2))-C(N-2)/(X(N-1)-X(N-3))
      C(1)=C(1)*D(1)**2.0D0/(X(4)-X(1))
      C(N)=-C(N)*D(N-1)**2.0D0/(X(N)-X(N-3))
 215  CONTINUE
      DO 220 I=2,N
        T=D(I-1)/B(I-1)
        B(I)=B(I)-T*D(I-1)
        C(I)=C(I)-T*C(I-1)
 220  CONTINUE
      C(N)=C(N)/B(N)
      DO 230 IB=1,NM1
        I=N-IB
        C(I)=(C(I)-D(I)*C(I+1))/B(I)
 230  CONTINUE
      B(N)=(Y(N)-Y(NM1))/D(NM1)+D(NM1)*(C(NM1)+2.0D0*C(N))
      DO 240 I=1,NM1
        B(I)=(Y(I+1)-Y(I))/D(I)-D(I)*(C(I+1)+2.0D0*C(I))
        D(I)=(C(I+1)-C(I))/D(I)
        C(I)=3.0D0*C(I)
 240  CONTINUE
      C(N)=3.0D0*C(N)
      D(N)=D(N-1)
      RETURN
 250  CONTINUE
      B(1)=(Y(2)-Y(1))/(X(2)-X(1))
      C(1)=0.0D0
      D(1)=0.0D0
      B(2)=B(1)
      C(2)=0.0D0
      D(2)=0.0D0
      RETURN
      END
c
c***************************************************************************
C ---------------------------------------------------------------------
      double precision FUNCTION SEVAL(N,XX,X,Y,B,C,D)
C ---------------------------------------------------------------------
c***************************************************************************
C CALCULATE THE DISTRIBUTION AT XX BY CUBIC SPLINE INTERPOLATION.
      implicit double precision(A-H,O-Z)
      DIMENSION X(N),Y(N),B(N),C(N),D(N)
      DATA I/1/
      IF(I.GE.N) I=1
      IF(XX.LT.X(I)) GO TO 310
      IF(XX.LE.X(I+1)) GO TO 330
 310  CONTINUE
      I=1
      J=N+1
 320  CONTINUE
      K=(I+J)/2
      IF(XX.LT.X(K)) J=K
      IF(XX.GE.X(K)) I=K
      IF(J.GT.I+1) GO TO 320
 330  CONTINUE
      DX=XX-X(I)
      SEVAL=Y(I)+DX*(B(I)+DX*(C(I)+DX*D(I)))
      RETURN
      END
c
c***************************************************************************




