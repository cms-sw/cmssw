
*----------------------------------------------------------------------
*
*  Filename               : PYQUEN.F
*
*  Author                 : Igor Lokhtin  (Igor.Lokhtin@cern.ch)
*  Version                : PYQUEN1_5.f, v.1.5.1 
*  Last revision v.5.1    : 19-DEC-2007 
*  Last revision v.5.1.1  : 06-MAY-2010
*
*======================================================================
*
*  Description : Event generator for simulation of parton rescattering 
*                and energy loss in expanding quark-gluon plasma created  
*                in ultrarelativistic heavy ion AA collisons   
*               (implemented as modification of standard Pythia jet event) 
*
*  Reference: I.P. Lokhtin, A.M. Snigirev, Eur. Phys. J. C 46 (2006) 211   
*                   
*======================================================================

      SUBROUTINE PYQUEN(A,ifb,bfix,bmin,bmax)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c      INTEGER PYK,PYCHGE,PYCOMP
      external pydata  
      external pyp,pyr,pyk,pyjoin,pyshow
      external funbip,prhoaa,pfunc1
      common /pyjets/ n,npad,k(4000,5),p(4000,5),v(4000,5)
      common /pydat1/ mstu(200),paru(200),mstj(200),parj(200)       
      common /pysubs/ msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)      
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
      common /plglur/ glur(1000,4),kglu(1000,6),nrg,nrgm 
      common /plquar/ pqua(1000,5),kqua(1000,5),nrq 
      common /parimp/ b1,psib1,rb1,rb2,noquen 
      common /pyqpar/ T0u,tau0u,nfu,ienglu,ianglu 
      common /plfpar/ bgen
      common /pygeom/ BC
      common /pythic/ PBAB(110),PTAB(110),PTAAB(110)
      common /pynup1/ bp,x  
      save/pyjets/,/pydat1/,/pysubs/,/plglur/,/plquar/,/pygeom/,
     >    /pythic/,/plpar1/,/parimp/,/pyqpar/,/plfpar/
      dimension ijoik(2),ijoin(1000),ijoin0(1000),nis(500),nss(500),
     >          nas(500),nus(500)
                  
* set initial event paramters  
      AW=A                                 ! atomic weight 
      RA=1.15d0*AW**0.333333d0             ! nucleus radius in fm
      RA3=3.d0*RA 
      mvisc=0                              ! flag of QGP viscosity (off here) 
      TC=0.2d0                             ! crutical temperature 
      
      if(nfu.ne.0.and.nfu.ne.1.and.nfu.ne.2.and.nfu.ne.3) nfu=0
      nf=nfu                               ! number of active flavours in QGP
      if(tau0u.lt.0.01d0.or.tau0u.gt.10.d0) tau0u=0.1d0  
      tau0=tau0u                           ! proper time of QGP formation
      if(T0u.lt.0.2d0.or.T0u.gt.2.d0) T0u=1.d0  
      T0=T0u*(AW/207.d0)**0.166667d0       ! initial QGP temperatute at b=0
      if(ienglu.ne.0.and.ienglu.ne.1.and.ienglu.ne.2) ienglu=0 ! e-loss type
      if(ianglu.ne.0.and.ianglu.ne.1.and.ianglu.ne.2) ianglu=0 ! angular spec.  
*    
      pi=3.14159d0 

* avoid stopping run if pythia does not conserve energy due to collisional loss 
      mstu(21)=1 

* creation of arrays for tabulation of beam/target nuclear thickness function
      Z2=4.d0*RA
      Z1=-1.d0*Z2
      H=0.01d0*(Z2-Z1)
      do ib=1,110    
       BC=RA3*(ib-1)/109.d0
       CALL SIMPA(Z1,Z2,H,0.005d0,1.d-8,prhoaa,Z,RES,AIH,AIABS)     
       PBAB(ib)=BC
       PTAB(ib)=AW*RES
      end do   
      
* calculation of beam/target nuclear overlap function at b=0
* if ifb=1: creation of arrays for tabulation of nuclear overlap function
      npb=1
      if (ifb.eq.1) npb=110  
      Z1=0.d0 
      Z2=6.28318d0 
      H=0.01d0*(Z2-Z1)    
      do ib=1,npb 
       bp=PBAB(ib)
       CALL SIMPA(Z1,Z2,H,0.05d0,1.d-8,PFUNC1,X,RES,AIH,AIABS)
       PTAAB(ib)=RES 
      end do   

* generate impact parameter of A-A collision with jet production  
      if(ifb.eq.0) then 
       if(bfix.lt.0.d0) then    
        write(6,*) 'Impact parameter less than zero!'  
        bfix=0.d0  
       end if  
       if (bfix.gt.RA3) then 
        write(6,*) 'Impact parameter larger than three nuclear radius!'  
        bfix=RA3
       end if 
       b1=bfix  
      else      
 112     call bipsear(fmax1,xmin1) 
       fmax=fmax1 
       xmin=xmin1    
 11    bb1=xmin*pyr(0) 
       ff1=fmax*pyr(0) 
       fb=funbip(bb1) 
       if(ff1.gt.fb) goto 11    
       b1=bb1  
       if(bb1.gt.bmax) goto 112
       if(bb1.lt.bmin) goto 112
      end if  
      bgen=b1 
      
* generate single event with partonic energy loss 
      nrg=0 
      ehard=ckin(3) 
      call plinit(ehard)  
      call plevnt(ehard)

* reset all in-vacuum radiated guark 4-momenta and codes to zero 
      do i=1,1000  
       do j=1,5
        pqua(i,j)=0.d0
        kqua(i,j)=0  
       end do          
      end do   
      nrq=0 

* generate final state shower in vacuum if it was excluded before 
      nrgm=nrg                        ! fix number of in-medium emitted gluons  
      ip1=0
      ip2=0
      ip01=0
      ip02=0
      ip001=0
      ip002=0  
      if(mstj(41).ne.0) goto 5
      mstj(41)=1  
      nn=n 
      kfh1=0 
      kfh2=0 

      do i=9,nn 
       if(k(i,3).eq.7) then  
        ip1=i                    ! first hard parton (line ip1) 
        kfh1=k(i,1)              ! status code of first hard parton 
        qmax1=pyp(i,10)          ! transverse momentum of first hard parton
       end if
       if(k(i,3).eq.8) then 
        ip2=i                    ! second hard parton (line ip2)  
        kfh2=k(i,1)              ! status code of second hard parton 
        qmax2=pyp(i,10)          ! transverse momentum of second hard parton 
       end if
      end do
      
      n1=n  
      call pyshow(ip1,0,qmax1)    ! vacuum showering for first hard parton  
      if(n.eq.n1) ip1=0     
      n2=n 
      call pyshow(ip2,0,qmax2)    ! vacuum showering for second hard parton 
      if(n.eq.n2) ip2=0   
      mstj(41)=0 
      if(n.eq.nn) goto 5  
      
* find two leading partons after showering  
      do i=nn+1,n 
       if(k(i,3).eq.ip1) ip001=i   ! first daughter of first hard parton 
       if(k(i,3).eq.ip2) ip002=i   ! first daughter of second hard parton 
      end do
      ptle1=0.d0
      ptle2=0.d0    
      do i=nn+1,n
       if (k(i,1).eq.14) goto 3
       if(i.ge.ip002.and.ip002.gt.0) then 
        ptl02=pyp(i,10) 
        if(ptl02.gt.ptle2.and.k(i,2).eq.k(ip2,2)) then 
         ip02=i                   ! leading parton in second shower (line ip02)
         ptle2=ptl02              ! pt of the leading parton 
        end if 
       elseif(ip001.gt.0) then  
        ptl01=pyp(i,10) 
        if(ptl01.gt.ptle1.and.k(i,2).eq.k(ip1,2)) then 
         ip01=i                   ! leading parton in first shower (line ip01)
         ptle1=ptl01              ! pt of the leading parton 
        end if 
       end if
 3     continue 
      end do

* replace two hard partons by two leading partons in original event record 
      if(ip1.gt.0) then 
       do j=1,5 
        v(ip1,j)=v(ip01,j)  
        p(ip1,j)=p(ip01,j) 
       end do 
       k(ip1,1)=kfh1
* fix first/last daughter for moving entry 
        do jgl=1,n
         if(k(jgl,4).eq.ip01) k(jgl,4)=ip1
         if(k(jgl,5).eq.ip01) k(jgl,5)=ip1  
        end do 
*
      end if 
      if(ip2.gt.0) then   
       do j=1,5  
        v(ip2,j)=v(ip02,j)  
        p(ip2,j)=p(ip02,j) 
       end do 
       k(ip2,1)=kfh2  
* fix first/last daughter for moving entry  
        do jgl=1,n
         if(k(jgl,4).eq.ip02) k(jgl,4)=ip2
         if(k(jgl,5).eq.ip02) k(jgl,5)=ip2  
        end do 
*
      end if 
 
* add final showering gluons to the list of in-medium emitted gluons,
* fill the list of emitted quarks by final showering quark pairs,  
* and remove showering gluons and quarks from the event record 
      do i=nn+1,n 
       if(k(i,1).eq.14.or.i.eq.ip01.or.i.eq.ip02) goto 12       
       if(k(i,2).ne.21) then           ! filling 'plquar' arrays for quarks 
       nrq=nrq+1 
        do j=1,5 
         kqua(nrq,j)=k(i,j)
         pqua(nrq,j)=p(i,j)
        end do 
        kqua(nrq,1)=2 
        goto 12        
       end if   
       if(i.ge.ip002.and.ip002.gt.0) then 
        ish=ip2
       else  
        ish=ip1
       end if 
       nrg=nrg+1
       nur=nrg 
 7     ishm=kglu(nur-1,6)
       if(ish.ge.ishm.or.nur.le.2) goto 6   ! adding gluons in 'plglur' arrays 
       do j=1,6
        kglu(nur,j)=kglu(nur-1,j)
       end do 
       do j=1,4 
        glur(nur,j)=glur(nur-1,j)
       end do 
       nur=nur-1 
       goto 7                                                    
 6     kglu(nur,1)=2                              ! status code 
       kglu(nur,2)=k(i,2)                         ! particle identificator      
       kglu(nur,3)=k(ish,3)                       ! parent line number  
       kglu(nur,4)=0                              ! special colour info
       kglu(nur,5)=0                              ! special colour info  
       kglu(nur,6)=ish                            ! associated parton number 
       glur(nur,1)=p(i,4)                         ! energy  
       glur(nur,2)=pyp(i,10)                      ! pt  
       glur(nur,3)=pyp(i,15)                      ! phi
       glur(nur,4)=pyp(i,19)                      ! eta   
 12    continue        
* remove partons from event list
       do j=1,5                             
        v(i,j)=0.d0 
        k(i,j)=0 
        p(i,j)=0.d0  
       end do        
      end do 
      n=nn        
       
 5    continue   
          
* stop generate event if there are no additional gluons 
      if(nrg.lt.1) goto 1 

* define number of stirngs (ns) and number of entries in strings before 
* in-medium radiation (nis(ns))  
      ns=0 
      nes=0 
      i0=0  
      i1=0  
      do i=1,500  
       nis(i)=0 
       nas(i)=0 
       nss(i)=0 
       nus(i)=0 
      end do                      
      do i=9,n 
       ks=k(i,1) 
       ksp=k(i-1,1) 
       if(ks.eq.2) then 
        nis(ns+1)=nis(ns+1)+1   
       elseif(ks.eq.1.and.nis(ns+1).gt.0) then 
        nis(ns+1)=nis(ns+1)+1
        nes=nes+nis(ns+1)     ! nes - total number of entries  
        nss(ns+1)=nes 
        ns=ns+1 
       elseif(ks.ne.2.and.ksp.ne.2.and.ns.gt.0) then 
        i1=i1+1               ! last i1 lines not included in strings 
       end if 
      end do 
      i0=n-nes-i1             ! first i0 lines not included in strings 
      do i=1,ns 
       nss(i)=nss(i)+i0 
      end do  
      
* move fragmented particles in bottom of event list  
      i=i0+1      
 2    ks=k(i,1)
      ksp=k(i-1,1) 
      if(ks.ne.2.and.ksp.ne.2) then 
       n=n+1 
       do j=1,5 
        v(n,j)=v(i,j) 
        k(n,j)=k(i,j) 
        p(n,j)=p(i,j) 
       end do 
* fix first/last daughter for moving entry 
       do jgl=1,n
        if(k(jgl,4).eq.i) k(jgl,4)=n
        if(k(jgl,5).eq.i) k(jgl,5)=n 
       end do
*
       do in=i+1,n 
        do j=1,5 
         v(in-1,j)=v(in,j) 
         k(in-1,j)=k(in,j) 
         p(in-1,j)=p(in,j)
        end do 
* fix first/last daughter for moving entry 
        do jgl=1,n
         if(k(jgl,4).eq.in) k(jgl,4)=in-1
         if(k(jgl,5).eq.in) k(jgl,5)=in-1 
        end do
*
       end do 
       do ip=1,nrg 
        ku=kglu(ip,6) 
        if(ku.gt.i) kglu(ip,6)=ku-1 
       end do 
       n=n-1
      else  
       i=i+1  
      end if 
      if(i.le.n-i1) goto 2  

* define number of additional entries in strings, nas(ns)                     
      do i=1,nrg 
       kas=kglu(i,6) 
       if(kas.le.nss(1)) then 
        nas(1)=nas(1)+1 
       else 
        do j=2,ns 
         if(kas.le.nss(j).and.kas.gt.nss(j-1)) 
     >   nas(j)=nas(j)+1 
        end do
       end if 	        
      end do 
      do j=1,ns   
       do i=1,j   
        nus(j)=nus(j)+nas(i) 
       end do 
      end do 
	    
* add emitted gluons in event list  
      nu=n 
      n=n+nrg 
      do i=nu,nu-i1,-1 
       is=i+nrg 
       do j=1,5 
        v(is,j)=v(i,j) 
        k(is,j)=k(i,j) 
        p(is,j)=p(i,j) 
       end do 
* fix first/last daughter for moving entries 
       do jgl=1,n
        if(k(jgl,4).eq.i) k(jgl,4)=is
        if(k(jgl,5).eq.i) k(jgl,5)=is 
       end do
*
      end do 

      do ia=ns-1,1,-1  
       do i=nss(ia+1)-1,nss(ia),-1 
        is=i+nus(ia) 
        do j=1,5 
         v(is,j)=v(i,j) 
         k(is,j)=k(i,j) 
         p(is,j)=p(i,j) 
        end do
* fix first/last daughter for moving entries 
        do jgl=1,n
         if(k(jgl,4).eq.i) k(jgl,4)=is
         if(k(jgl,5).eq.i) k(jgl,5)=is 
        end do
*       
       end do 
      end do 

      do i=1,nrg 
       if(i.le.nus(1)) then 
       ia=nss(1)-1+i 
       else  
        do in=2,ns 
         if(i.le.nus(in).and.i.gt.nus(in-1)) 
     >   ia=nss(in)-1+i 
        end do 
       end if 
       eg=glur(i,1)
       ptg=glur(i,2)
       phig=glur(i,3)    
       etag=glur(i,4)   
       do j=1,5 
        v(ia,j)=0.d0 
        k(ia,j)=kglu(i,j) 
       end do 
       p(ia,1)=ptg*dcos(phig)
       p(ia,2)=ptg*dsin(phig) 
       p(ia,3)=dsqrt(abs(eg*eg-ptg*ptg))
       if(etag.lt.0.d0) p(ia,3)=-1.d0*p(ia,3)  
       p(ia,4)=dsqrt(ptg*ptg+p(ia,3)**2)      
       p(ia,5)=0.d0   
      end do  
      
* rearrange partons to form strings in event list 
      do ij=1,1000 
       ijoin(ij)=0 
       ijoin0(ij)=0 
      end do 
      do i=1,ns 
       njoin=nis(i)+nas(i) 
       if(i.eq.1) then 
        do j=1,njoin 
         ijoin(j)=i0+j
        end do 
       else 
        do j=1,njoin 
         ijoin(j)=nss(i-1)+nus(i-1)+j 
        end do  
       end if 
       
* re-oder additional gluons by z-coordinate along the string
       if(nas(i).gt.0) then
        ja=njoin-nas(i)
        jo1=ijoin(1)
        jon=ijoin(njoin)
        etasum=0.d0
        detaj=pyp(jo1,19)-pyp(jon,19)
        do j=ja,njoin-1
         jnum=0
         etaj=pyp(jo1+j-1,19)
	 etasum=etasum+etaj
         do jj=ja,njoin-1
          etajj=pyp(jo1+jj-1,19)
          if(detaj.lt.0) then
           if(etajj.lt.etaj.and.j.ne.jj) jnum=jnum+1
          else
           if(etajj.gt.etaj.and.j.ne.jj) jnum=jnum+1
          end if
          if(etajj.eq.etaj.and.j.lt.jj) jnum=jnum+1
	 end do
         ijoin(ja+jnum)=jo1+j-1
        end do
        detas1=abs(pyp(jo1,19)-etasum)
        detasn=abs(pyp(jon,19)-etasum)
	if(detasn.gt.detas1) then
	 do j=1,njoin 
	  ijoin0(j)=ijoin(j)
	 end do 
         do j=2,nas(i)+1
          ijoin(j)=ijoin0(ja+j-2)
         end do 	 	
         do j=nas(i)+2,njoin-1  	
          ijoin(j)=ijoin0(j-nas(i))
         end do
        end if 
       end if

* form strings
       call pyjoin(njoin,ijoin)

      end do  

* add in-vacuum emitted quark pairs 
      if(nrq.lt.2) goto 1                    
      do i=1,nrq,2  
       n=n+2 
       do j=1,5  
        v(n-1,j)=0.d0 
        k(n-1,j)=kqua(i,j) 
        p(n-1,j)=pqua(i,j) 
       end do
       in=i+1 
 4     ktest=k(n-1,2)+kqua(in,2)
       if(ktest.eq.0.or.in.eq.nrq) goto 8 
       in=in+1 
       goto 4 
 8     do j=1,5  
        v(n,j)=0.d0 
        k(n,j)=kqua(in,j) 
        p(n,j)=pqua(in,j) 
       end do
       if(in.gt.i+1) then 
        do j=1,5   
         kqua(in,j)=kqua(i+1,j) 
         pqua(in,j)=pqua(i+1,j) 
        end do
       end if  
      end do
 
      do ij=1,2 
       ijoik(ij)=0 
      end do 
      do i=1,nrq-1,2 
       k(n+1-i,1)=1 
       ijoik(1)=n-i 
       ijoik(2)=n+1-i 
       call pyjoin(2,ijoik)
      end do  

 1    continue 
           
      return 
      end 

********************************* PLINIT ***************************
      SUBROUTINE PLINIT(ET) 
* set time-dependence of plasma parameters   
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external plvisc  
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf  
      common /plpar2/ pln0,taupl,tauh,sigpl,sigh,sigplh,sigqqh,rg,rgn 
      common /plevol/ taup(5000),temp(5000),denp(5000),enep(5000)  
      save /plevol/,/plpar1/,/plpar2/  
*
      et=et
      pi=3.14159d0 
      pi2=pi*pi  

* set number degrees of freedom in QGP                
      hgd=3.d0
      rg=(16.d0+10.5d0*nf)/hgd   
      rgn=(16.d0+9.d0*nf)/hgd 
      
* set 'fiction' sigma for parton rescattering in QGP  
      sigqq=4.2d0 
      sigpl=2.25d0*2.25d0*sigqq*(16.d0+4.d0*nf)/(16.d0+9.d0*nf) 

* set intial plasma temperature, density and energy density in perfect 
* (if mvisc=0) or viscous (mvisc=1,2) QGP with PLVISC subroitine  
      hst=0.15d0   
      if(T0.gt.1.5d0.or.mvisc.eq.2) hst=0.25d0
      if(T0.gt.1.5d0.and.mvisc.ne.0) hst=0.9d0  
      T01=T0*5.06d0                 
      TC1=TC*5.06d0
      pln0=(16.d0+9.d0*nf)*1.2d0*(T01**3)/pi2
      ened0=pi2*(16.d0+10.5d0*nf)*(T01**4)/30.d0  
      hh=hst*tau0  
      tau=tau0                          ! proper time
      T=T01                             ! temperature
      den=pln0                          ! number density 
      ened=ened0                        ! energy density 

* create array of parameters to configurate QGP time evolution 
      DO I=1,5000
       taup(i)=tau                      ! proper time 
       temp(i)=T/5.06d0                 ! temperature  
       denp(i)=den                      ! number density 
       enep(i)=ened/5.06d0              ! energy density
       ened1=0.5d0*hh*(1.3333d0*plvisc(T)/(tau*tau)-1.3333d0 
     >       *ened/tau)+ened
       T1=(30.d0*ened1/((16.d0+10.5d0*nf)*pi2))**0.25d0 
       tau1=tau+0.5d0*hh 
       ened=hh*(1.3333d0*plvisc(T1)/(tau1*tau1)-1.3333d0
     >      *ened1/tau1)+ened 
       TPR=T 
       T=(30.d0*ened/((16.d0+10.5d0*nf)*pi2))**0.25d0 
       den=(16.d0+9.d0*nf)*1.2d0*(T**3)/pi2 
       tau=tau+hh 
       if(TPR.gt.TC1.and.T.le.TC1) taupl=tau-0.5d0*hh  ! QGP lifetime taupl 
      END DO 
      tauh=taupl*rg                                    ! mixed phase lifetime        

      return 
      end 
******************************** END PLINIT ************************** 

******************************* PLEVNT ******************************
      SUBROUTINE PLEVNT(ET)    
* generate hard parton production vertex and passing with rescattering,
* collisional and radiative energy loss of each parton through plasma        
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c      INTEGER PYK,PYCHGE,PYCOMP
      external plthik, pln, plt, pls, gauss, gluang 
      external pyp,pyr,pyk 
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
      common /plpar2/ pln0,taupl,tauh,sigpl,sigh,sigplh,sigqqh,rg,rgn 
      common /thikpa/ fmax,xmin 
      common /pyjets/ n,npad,k(4000,5),p(4000,5),v(4000,5)
      common /pyqpar/ T0u,tau0u,nfu,ienglu,ianglu 
      common /plglur/ glur(1000,4),kglu(1000,6),nrg,nrgm  
      common /factor/ cfac, kf 
      common /pleave/ taul, temlev   
      common /parimp/ b1, psib1, rb1, rb2, noquen 
      common /plen/ epartc, um 
      common /plos/ elr,rsk 
      common /numje1/ nuj1, nuj2  
      save/pyjets/,/plglur/,/plpar1/,/plpar2/,/thikpa/,/factor/,
     <    /pleave/,/parimp/,/plen/,/plos/,/numje1/
*
      pi=3.14159d0              
      et=et
* find minimum of nuclear thikness function with subroutine plsear      
      psib1=pi*(2.d0*pyr(0)-1.d0) 
      call plsear (fmax1,xmin1)
      fmax=fmax1 
      xmin=xmin1  

* generate vertex of jet production  
      iv=0 
 1    rr1=xmin*pyr(0) 
      ff1=fmax*pyr(0) 
      f=plthik(rr1)
      iv=iv+1  
      if(ff1.gt.f.and.iv.le.100000) goto 1    
      r0=rr1 
      rb1=dsqrt(abs(r0*r0+b1*b1/4.d0+r0*b1*dcos(psib1))) 
      rb2=dsqrt(abs(r0*r0+b1*b1/4.d0-r0*b1*dcos(psib1))) 
      rb1=max(rb1,1.d-4) 
      rb2=max(rb2,1.d-4) 
* no quenching if noquen=1 or jet production vertex is out of effective dense zone 
      if(noquen.eq.1.or.rb1.gt.RA.or.rb2.gt.RA) goto 7

* find maximum of angular spectrum of radiated gluons with subroutine gluang 
      temin=0.5d0*pi 
      temax=0.5d0*(1.d0+dsqrt(5.d0))*0.0863d0   
      ftemax=gluang(temax) 

* reset all radiated gluon 4-momenta and codes to zero -------------------
      do i=1,1000  
       do j=1,4
        glur(i,j)=0.d0
        kglu(i,j)=0  
       end do 
       kglu(i,5)=0        
       kglu(i,6)=0 
      end do   
      nrg=0 

* generate changing 4-momentum of partons due to rescattering and energy loss 
* (for partons with |eta|<3.5 and pt>3 GeV/c)
      nuj1=9                            ! minimum line number of rescattered parton 
      nuj2=n                            ! maximum line number of rescattered parton   
      do 2 ip=nuj1,nuj2                 ! cycle on travelling partons 
       irasf=0 
       iraz=0 
       ks=k(ip,1)                       ! parton status code 
       kf=k(ip,2)                       ! parton identificator 
       ka=abs(kf) 
       ko=k(ip,3)                       ! origin (parent line number) 
       epart=abs(pyp(ip,10))            ! parton transverse momentum
       etar=pyp(ip,19)                  ! parton pseudorapidity  
       if(ko.gt.6.and.epart.ge.3.d0.and.abs(etar).
     >  le.7.d0) then 
       if(ka.eq.21.or.ka.eq.1.or.ka.eq.2.or.ka.eq.3.
     >  or.ka.eq.4.or.ka.eq.5.or.ka.eq.6.or.ka.eq.7.  
     >  or.ka.eq.8) then    
       if(ks.eq.2.or.ks.eq.1) then  
        phir=pyp(ip,15)                 ! parton azimuthal angle  
        tetr=pyp(ip,13)                 ! parton polar angle         
        yrr=pyp(ip,17)                  ! parton rapidity 
        stetr=dsin(tetr)                ! parton sin(theta) 
        if(abs(stetr).le.1.d-05) then 
         if(stetr.ge.0.d0) then 
          stetr=1.d-05
         else 
          stetr=-1.d-05
         end if 
        end if 
        phir1=-1.d0*phir 
        tetr1=-1.d0*tetr 

* set colour factor 
        if(kf.eq.21) then 
         cfac=1.d0                      ! for gluon 
        else 
         cfac=0.44444444d0              ! for quark 
        end if    

* boost from laboratory system to system of hard parton  
        ipar=ip 
        bet0=(r0*dcos(psib1)+0.5d0*b1)/rb1 
        if(bet0.le.-1.d0) bet0=-0.99999d0
        if(bet0.ge.1.d0) bet0=0.99999d0   
        bet=dacos(bet0)
        if(psib1.lt.0.d0) bet=-1.d0*bet 
        phip=phir-bet 
        if(phip.gt.pi) phip=phip-2.d0*pi 
        if(phip.lt.-1.d0*pi) phip=phip+2.d0*pi   
        call pyrobo(ip,ip,0.d0,phir1,0.d0,0.d0,0.d0)  
        call pyrobo(ip,ip,tetr1,0.d0,0.d0,0.d0,0.d0)  
 
* calculate proper time of parton leaving the effective dense zone
        aphin=(r0*r0-b1*b1/4.d0)/(rb1*rb2) 
        if(aphin.le.-1.d0) aphin=-0.99999d0
        if(aphin.ge.1.d0) aphin=0.99999d0   
        phin=dacos(aphin) 
        if(psib1.le.0.d0) phin=-1.d0*phin 
        phid=phip-phin    
        if(phid.gt.pi) phid=phid-2.d0*pi 
        if(phid.lt.-1.d0*pi) phid=phid+2.d0*pi 
        taul1=abs(dsqrt(abs(RA*RA-(rb1*dsin(phip))**2))-rb1*dcos(phip)) 
        taul2=abs(dsqrt(abs(RA*RA-(rb2*dsin(phid))**2))-rb2*dcos(phid))    
        taul=min(taul1,taul2)             ! escape time taul 
        temlev=plt(taul,rb1,rb2,yrr)      ! QGP temperature at taul 
        if(taul.le.tau0) goto 100        ! escape from QGP if taul<tau0  

* start parton rescattering in QGP with proper time iterations  
        tau=tau0 
        xj=r0*dcos(psib1)
        yj=r0*dsin(psib1)
        rj1=rb1
        rj2=rb2
 3      tfs=plt(tau,rj1,rj2,yrr) 
        xi=-10.d0*dlog(max(pyr(0),1.d-10))/(sigpl*pln(tau,rj1,rj2,yrr))
        vel=abs(p(ip,3))/dsqrt(p(ip,3)**2+p(ip,5)**2) ! parton velocity 
        if(vel.lt.0.3d0) goto 4      
        tau=tau+xi*vel    
        xj=xj+xi*vel*dcos(phir)
        yj=yj+xi*vel*dsin(phir)
        rj1=sqrt(abs(yj**2+(xj+0.5d0*b1)**2))
        rj2=sqrt(abs(yj**2+(xj-0.5d0*b1)**2))
        if(tfs.le.TC) goto 100     ! escape if temperature drops below TC

* transform parton 4-momentum due to next scattering with subroutine pljetr
        epartc=p(ip,4)                         ! parton energy 
        um=p(ip,5)                             ! parton mass 
        sigtr=pls(tfs)*cfac*((p(ip,4)/pyp(ip,8))**2)   
        prob=sigpl/(sigtr/stetr+sigpl) 
        ran=pyr(0) 
        irasf=irasf+1 
        if(irasf.gt.100000) goto 100 
        if(ran.lt.prob) goto 3  
        pltp=plt(tau,rj1,rj2,yrr) 
        if(pltp.le.TC) goto 100     ! escape if temperature drops below TC
        pltp3=3.d0*pltp 
        pass=50.6d0/(pln(tau,rj1,rj2,yrr)*sigtr)    
        elr=0.d0 
        rsk=0.d0 
        call pljetr(tau,pass,pltp,ipar,epart) 
        irasf=0 

* set 4-momentum (in lab system) of next radiated gluon for parton number >8  
* and fill arrays of radiated gluons in common block plglur  
        if(nrg.le.1000) then 
         if(abs(elr).gt.0.1d0.and.ip.gt.8) then   
* generate the angle of emitted gluon 
          if(ianglu.eq.0) then 
 6         te1=temin*pyr(0) 
           fte1=ftemax*pyr(0) 
           fte2=gluang(te1)
           if(fte1.gt.fte2) goto 6  
           tgl=te1                              
          elseif (ianglu.eq.1) then              
           tgl=((0.5d0*pi*epartc)**pyr(0))/epartc
          else 
           tgl=0.d0 
          end if  	                                          
          pgl=pi*(2.d0*pyr(0)-1.d0) 
* in comoving system 
          pxgl=abs(elr)*stetr*(dcos(phir)*dcos(tgl)-
     >      dsin(phir)*dsin(tgl)*dsin(pgl)) 
          pygl=abs(elr)*stetr*(dsin(phir)*dcos(tgl)+
     >      dcos(phir)*dsin(tgl)*dsin(pgl))  
          pzgl=-1.d0*abs(elr)*stetr*dsin(tgl)*dcos(pgl) 
          ptgl=dsqrt(abs(pxgl*pxgl+pygl*pygl))
          psgl=dsqrt(abs(ptgl*ptgl+pzgl*pzgl)) 
* recalculate in lab system 
          dyg=0.5d0*dlog(max(1.d-9,(psgl+pzgl)/(psgl-pzgl)))
          pzgl=ptgl*dsinh(yrr+dyg) 
          psgl=dsqrt(abs(ptgl*ptgl+pzgl*pzgl))
*
          dpgl=pygl/pxgl        
          glur1=abs(elr)                                       ! energy 
          glur3=datan(dpgl)                                    ! phi
          if(pxgl.lt.0.d0) then 
           if(pygl.ge.0.d0) then 
            glur3=glur3+pi 
           else 
            glur3=glur3-pi  
           end if 
          end if   
          glur4=0.5d0*dlog(max(1.d-9,(psgl+pzgl)/(psgl-pzgl))) ! eta  
          glur2=glur1/dcosh(glur4)                             ! pt 

* put in event list radiated gluons with pt > 0.2 GeV only 
          if(glur2.ge.0.2d0) then 
           nrg=nrg+1 
* set gluon 4-momentum 
           glur(nrg,1)=glur1                     ! energy
           glur(nrg,2)=glur2                     ! pt
           glur(nrg,3)=glur3                     ! phi 
           glur(nrg,4)=glur4                     ! eta
* set gluon codes 
           kglu(nrg,1)=2                         ! status code 
           kglu(nrg,2)=21                        ! particle identificator 
           kglu(nrg,3)=k(ipar,3)                 ! parent line number  
           kglu(nrg,4)=0                         ! special colour info
           kglu(nrg,5)=0                         ! special colour info  
           kglu(nrg,6)=ipar                      ! associated parton number 
          end if 
         end if  
        else 
         write(6,*) 'Warning! Number of emitted gluons is too large!' 
        end if 

* set parton "thermalization" if pt<T
        if(abs(p(ip,3)).gt.pltp3) goto 3   
 4      continue  
        if(p(ip,3).ge.0.d0) then 
         sigp=1.d0 
        else 
         sigp=-1.d0 
        end if     
 5      iraz=iraz+1  
        if(iraz.gt.100000) goto 100  
        ep0=-0.15d0*(dlog(max(1.d-10,pyr(0)))+dlog(max(1.d-10,pyr(0)))+
     >  dlog(max(1.d-10,pyr(0)))) 
        if(ep0.le.p(ip,5).or.ep0.ge.100.d0) goto 5   
        pp0=dsqrt(abs(ep0**2-p(ip,5)**2)) 
        probt=pp0/ep0 
        if(pyr(0).gt.probt) goto 5  
        ctp0=2.d0*pyr(0)-1.d0 
        stp0=dsqrt(abs(1.d0-ctp0**2)) 
        php0=pi*(2.d0*pyr(0)-1.d0)  
        p(ip,1)=pp0*stp0*dcos(php0)       
        p(ip,2)=pp0*stp0*dsin(php0)         
        p(ip,3)=sigp*pp0*ctp0
        p(ip,4)=dsqrt(p(ip,1)**2+p(ip,2)**2+p(ip,3)**2+p(ip,5)**2) 

* boost to laboratory system 
 100    call pyrobo(ip,ip,tetr,phir,0.d0,0.d0,0.d0)
       end if 
       end if 
       end if 
 2    continue 
 7    continue
 
      return 
      end     
******************************* END PLEVNT ************************* 

******************************* PLJETR *****************************
      SUBROUTINE PLJETR(tau,y,x,ip,epart)       
* transform parton 4-momentum due to scattering in plasma at time = tau 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c      INTEGER PYK,PYCHGE,PYCOMP
      external plfun1, pls 
      external pyp,pyr   
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
      common /plpar2/ pln0,taupl,tauh,sigpl,sigh,sigplh,sigqqh,rg,rgn        
      common /pyjets/ n,npad,k(4000,5),p(4000,5),v(4000,5)
      common /pyqpar/ T0u,tau0u,nfu,ienglu,ianglu 
      common /pljdat/ ej, z, ygl, alfs, um, epa 
      common /pleave/ taul, temlev        
      common /radcal/ aa, bb 
      common /factor/ cfac, kf 
      common /plos/ elr,rsk 
      save /pyjets/,/plpar1/,/plpar2/,/pyqpar/,/pljdat/,/pleave/,/plos/,
     <     /factor/,/radcal/
*
      pi=3.14159d0 
      spi=dsqrt(pi)
      tauu=x                            ! redenote temerature tauu=x 
      i=ip                              ! redenote parton number i=ip       
      iter=0 
      iraz=0 
      epart=epart
      tau=tau

* boost to system of comoving plasma constituent  
      phir=pyp(i,15)                    ! parton phi  
      tetr=pyp(i,13)                    ! parton theta   
      phir1=-1.d0*phir 
      tetr1=-1.d0*tetr 
      call pyrobo(i,i,0.d0,phir1,0.d0,0.d0,0.d0)  
      call pyrobo(i,i,tetr1,0.d0,0.d0,0.d0,0.d0)  
      pp=pyp(i,8)                       ! parton total momentum   
      ppl=abs(p(i,3))                   ! parton pz 
      um=p(i,5)                         ! parton mass 
      epa=p(i,4)                        ! parton energy 
      ppt=pyp(i,10)                     ! parton pt 
      pphi=pyp(i,15)                    ! parton phi       

      if(ppl.lt.3.d0) goto 222          ! no energy loss if pz<3 GeV/c 

* generation hard parton-plasma scattering with momentum transfer rsk 
 221   ep0=-1.*tauu*(dlog(max(1.d-10,pyr(0)))+dlog(max(1.d-10,
     >   pyr(0)))+dlog(max(1.d-10,pyr(0))))     ! energy of 'thermal' parton 
       iter=iter+1 
       if(ep0.lt.1.d-10.and.iter.le.100000) goto 221   
       scm=2.*ep0*epa+um*um+ep0*ep0 
       qm2=(scm-((um+ep0)**2))*(scm-((um-ep0)**2))/scm  
       bub=4.d0*tauu/TC   
       alf=6.d0*pi/((33.d0-2.d0*nf)*dlog(max(bub,1.d-10)))
       z=abs(pi*4.d0*tauu*tauu*alf*(1.+nf/6.d0))  
       bubs=dsqrt(z)/TC 
       alfs=6.d0*pi/((33.d0-2.d0*nf)*dlog(max(bubs,1.d-10))) 
       phmin2=z 
       phmax2=max(phmin2,qm2)  
       fqmax2=1.d0/(dlog(max(phmin2/(TC*TC),1.d-10)))**2           
 12    rn1=pyr(0)
       tp=1.d0/(rn1/phmax2+(1.d0-rn1)/phmin2)
       ftp=1.d0/(dlog(max(tp/(TC*TC),1.d-10)))**2 
       fprob=ftp/fqmax2 
       rn2=pyr(0) 
       if(fprob.lt.rn2) goto 12             
       rsk=dsqrt(abs(tp))
       if(rsk.gt.ppl) rsk=ppl          

* calculate radiative energy loss per given scattering with subroutine plfun1 
       ygl=y*cfac                      ! mean gluon free path in GeV^{-1}
       elp=ygl*z                       ! mimimum radiated energy in LPM regime
       ej=ppl                           
       bb=ej                           ! maximum radiated energy 
       bbi=max(dsqrt(z),1.000001d0*elp)  
       aa=min(bb,bbi)                  ! minimum radiated energy 
       hh=0.00001d0*(bb-aa)    
       REPS=0.01d0 
       AEPS=1.d-8 
       CALL SIMPA(aa,bb,hh,REPS,AEPS,plfun1,om,resun,AIH,AIABS)    
*                                      ! integral over omega for radiative loss
       call radsear(ermax1,eomin1) 
       ermax=ermax1 
       eomin=eomin1 
 11    resu=eomin*pyr(0)+aa 
       fres=ermax*pyr(0) 
       fres1=plfun1(resu) 
       iraz=iraz+1 
       if(fres.gt.fres1.and.iraz.lt.100000) goto 11   
       elr=resu*resun                   ! energy of radiated gluon 

* to chancel radiative energy loss (optional case) 
       if(ienglu.eq.2) elr=0.d0
* to chancel collisional energy loss (optional case) 
       if(ienglu.eq.1) rsk=0.d0 

* determine the direction of parton moving 
       if(p(i,3).ge.0.d0) then 
        sigp=1.d0 
       else 
        sigp=-1.d0
       end if     

* calculate new 4-momentum of hard parton 
       phirs=2.d0*pi*pyr(0)
       epan=epa-rsk*rsk/(2.d0*ep0)-abs(elr)  
       if(epan.lt.0.1d0) then 
        epan=epan+abs(elr) 
        elr=0.d0
        if(epan.lt.0.1d0) then
         rsk=0.d0 
         epan=epa
        end if  
       end if  
       pptn=dsqrt(abs(rsk*rsk+(rsk**4)*(1.d0-epa*epa/(ppl*ppl))/
     >      (4.d0*ep0*ep0)-(rsk**4)*epa/(2.d0*ep0*ppl*ppl)-(rsk**4)/
     >      (4.d0*ppl*ppl))) 
       ppln=dsqrt(abs(epan*epan-pptn*pptn-p(i,5)**2))   
       p(i,1)=pptn*dcos(phirs)                                 ! px 
       p(i,2)=pptn*dsin(phirs)                                 ! py
       p(i,3)=sigp*ppln                                        ! pz 
       p(i,4)=dsqrt(p(i,1)**2+p(i,2)**2+p(i,3)**2+p(i,5)**2)   ! E 
* boost to system of hard parton 
 222   call pyrobo(i,i,tetr,phir,0.d0,0.d0,0.d0)

      return
      end
******************************* END PLJETR **************************

******************************** PLSEAR ***************************
       SUBROUTINE PLSEAR (fmax,xmin) 
* find maximum and 'sufficient minimum' of jet production vertex distribution
* xm, fm are outputs. 
       IMPLICIT DOUBLE PRECISION(A-H, O-Z) 
       external plthik
       common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
       save /plpar1/
       xmin=3.d0*RA 
       fmax=0.d0
       do 10 j=1,1000
        x=xmin*(j-1)/999.d0
        f=plthik(x) 
        if(f.gt.fmax) then
         fmax=f
        end if
  10   continue
       return
       end
****************************** END PLSEAR **************************

******************************** RADSEAR ***************************
       SUBROUTINE RADSEAR (fmax,xmin)
* find maximum and 'sufficient minimum' of radiative energy loss distribution 
* xm, fm are outputs. 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external plfun1  
      common /radcal/ aa, bb
      save /radcal/ 
      xmin=bb-aa     
      fmax=0.d0
      do j=1,1000
       x=aa+xmin*(j-1)/999.d0
       f=plfun1(x)   
       if(f.gt.fmax) then
        fmax=f
       end if
      end do   
      return
      end
****************************** END RADSEAR **************************

********************************* BIPSEAR ***************************
      SUBROUTINE BIPSEAR (fmax,xmin) 
* find maximum and 'sufficient minimum' of jet production cross section  
* as a function of impact paramater (xm, fm are outputs)       
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external funbip 
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
      save /plpar1/
      xmin=3.d0*RA 
      fmax=0.d0 
      do j=1,1000
       x=xmin*(j-1)/999.d0 
       f=funbip(x) 
       if(f.gt.fmax) then
        fmax=f
       end if
      end do  
      return
      end
****************************** END RADSEAR **************************

**************************** SIMPA **********************************
      SUBROUTINE SIMPA (A1,B1,H1,REPS1,AEPS1,FUNCT,X,                   
     1                     AI,AIH,AIABS)                                
* calculate intergal of function FUNCT(X) on the interval from A1 to B1 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      DIMENSION F(7), P(5)                                             
      H=dSIGN ( H1, B1-A1 )                                             
      S=dSIGN (1.d0, H )                                                   
      A=A1                                                              
      B=B1                                                              
      AI=0.0d0                                                            
      AIH=0.0d0                                                           
      AIABS=0.0d0                                                        
      P(2)=4.d0                                                           
      P(4)=4.d0                                                           
      P(3)=2.d0                                                           
      P(5)=1.d0                                                           
      IF(B-A)1,2,1                                                      
 1    REPS=ABS(REPS1)                                                   
      AEPS=ABS(AEPS1)                                                  
      DO 3 K=1,7                                                        
 3    F(K)=10.d16                                                       
      X=A                                                              
      C=0.d0                                                              
      F(1)=FUNCT(X)/3.d0                                                  
 4    X0=X                                                              
      IF( (X0+4.d0*H-B)*S)5,5,6                                           
 6    H=(B-X0)/4.d0                                                       
      IF ( H ) 7,2,7                                                   
 7    DO 8 K=2,7                                                      
 8    F(K)=10.d16                                                       
      C=1.d0                                                           
 5    DI2=F (1)                                                       
      DI3=ABS( F(1) )                                                   
      DO 9 K=2,5                                                       
      X=X+H                                                           
      IF((X-B)*S)23,24,24                                              
 24   X=B                                                              
 23   IF(F(K)-10.d16)10,11,10                                          
 11   F(K)=FUNCT(X)/3.d0                                               
 10   DI2=DI2+P(K)*F(K)                                                 
 9    DI3=DI3+P(K)*ABS(F(K))                                            
      DI1=(F(1)+4.*F(3)+F(5))*2.d0*H                                      
      DI2=DI2*H                                                         
      DI3=DI3*H                                                        
      IF (REPS) 12,13,12                                               
 13   IF (AEPS) 12,14,12                                                
 12   EPS=ABS((AIABS+DI3)*REPS)                                         
      IF(EPS-AEPS)15,16,16                                              
 15   EPS=AEPS                                                          
 16   DELTA=ABS(DI2-DI1)                                               
      IF(DELTA-EPS)20,21,21                                             
 20   IF(DELTA-EPS/8.d0)17,14,14                                          
 17   H=2.d0*H                                                            
      F(1)=F(5)                                                         
      F(2)=F(6)                                                         
      F(3)=F(7)                                                         
      DO 19 K=4,7                                                       
 19   F(K)=10.d16                                                      
      GO TO 18                                                         
 14   F(1)=F(5)                                                         
      F(3)=F(6)                                                         
      F(5)=F(7)                                                         
      F(2)=10.d16                                                       
      F(4)=10.d16                                                      
      F(6)=10.d16                                                      
      F(7)=10.d16                                                      
 18   DI1=DI2+(DI2-DI1)/15.d0                                            
      AI=AI+DI1                                                         
      AIH=AIH+DI2                                                      
      AIABS=AIABS+DI3                                                   
      GO TO 22                                                          
 21   H=H/2.d0                                                            
      F(7)=F(5)                                                        
      F(6)=F(4)                                                        
      F(5)=F(3)                                                        
      F(3)=F(2)                                                         
      F(2)=10.d16                                                      
      F(4)=10.d16                                                      
      X=X0                                                            
      C=0.                                                             
      GO TO 5                                                          
 22   IF(C)2,4,2                                                      
 2    RETURN                                                        
      END                                                              
************************* END SIMPA *******************************

**************************** SIMPB **********************************
      SUBROUTINE SIMPB (A1,B1,H1,REPS1,AEPS1,FUNCT,X,                   
     1                     AI,AIH,AIABS)                                
* calculate intergal of function FUNCT(X) on the interval from A1 to B1 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      DIMENSION F(7), P(5)                                             
      H=dSIGN ( H1, B1-A1 )                                             
      S=dSIGN (1.d0, H )                                                   
      A=A1                                                              
      B=B1                                                              
      AI=0.0d0                                                            
      AIH=0.0d0                                                           
      AIABS=0.0d0                                                        
      P(2)=4.d0                                                           
      P(4)=4.d0                                                           
      P(3)=2.d0                                                           
      P(5)=1.d0                                                           
      IF(B-A)1,2,1                                                      
 1    REPS=ABS(REPS1)                                                   
      AEPS=ABS(AEPS1)                                                  
      DO 3 K=1,7                                                        
 3    F(K)=10.d16                                                       
      X=A                                                              
      C=0.d0                                                              
      F(1)=FUNCT(X)/3.d0                                                  
 4    X0=X                                                              
      IF( (X0+4.d0*H-B)*S)5,5,6                                           
 6    H=(B-X0)/4.d0                                                       
      IF ( H ) 7,2,7                                                   
 7    DO 8 K=2,7                                                      
 8    F(K)=10.d16                                                       
      C=1.d0                                                           
 5    DI2=F (1)                                                       
      DI3=ABS( F(1) )                                                   
      DO 9 K=2,5                                                       
      X=X+H                                                           
      IF((X-B)*S)23,24,24                                              
 24   X=B                                                              
 23   IF(F(K)-10.d16)10,11,10                                          
 11   F(K)=FUNCT(X)/3.d0                                               
 10   DI2=DI2+P(K)*F(K)                                                 
 9    DI3=DI3+P(K)*ABS(F(K))                                            
      DI1=(F(1)+4.*F(3)+F(5))*2.d0*H                                      
      DI2=DI2*H                                                         
      DI3=DI3*H                                                        
      IF (REPS) 12,13,12                                               
 13   IF (AEPS) 12,14,12                                                
 12   EPS=ABS((AIABS+DI3)*REPS)                                         
      IF(EPS-AEPS)15,16,16                                              
 15   EPS=AEPS                                                          
 16   DELTA=ABS(DI2-DI1)                                               
      IF(DELTA-EPS)20,21,21                                             
 20   IF(DELTA-EPS/8.d0)17,14,14                                          
 17   H=2.d0*H                                                            
      F(1)=F(5)                                                         
      F(2)=F(6)                                                         
      F(3)=F(7)                                                         
      DO 19 K=4,7                                                       
 19   F(K)=10.d16                                                      
      GO TO 18                                                         
 14   F(1)=F(5)                                                         
      F(3)=F(6)                                                         
      F(5)=F(7)                                                         
      F(2)=10.d16                                                       
      F(4)=10.d16                                                      
      F(6)=10.d16                                                      
      F(7)=10.d16                                                      
 18   DI1=DI2+(DI2-DI1)/15.d0                                            
      AI=AI+DI1                                                         
      AIH=AIH+DI2                                                      
      AIABS=AIABS+DI3                                                   
      GO TO 22                                                          
 21   H=H/2.d0                                                            
      F(7)=F(5)                                                        
      F(6)=F(4)                                                        
      F(5)=F(3)                                                        
      F(3)=F(2)                                                         
      F(2)=10.d16                                                      
      F(4)=10.d16                                                      
      X=X0                                                            
      C=0.                                                             
      GO TO 5                                                          
 22   IF(C)2,4,2                                                      
 2    RETURN                                                        
      END                                                              
************************* END SIMPB *******************************

************************* PARINV **********************************
      SUBROUTINE PARINV(X,A,F,N,R)                                      
* gives interpolation of function F(X) with  arrays A(N) and F(N) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      DIMENSION A(N),F(N)                                              
      IF(X.LT.A(1))GO TO 11                                            
      IF(X.GT.A(N))GO TO 4                                              
      K1=1                                                              
      K2=N                                                              
 2    K3=K2-K1                                                          
      IF(K3.LE.1)GO TO 6                                               
      K3=K1+K3/2                                                        
      IF(A(K3)-X) 7,8,9                                                 
 7    K1=K3                                                             
      GOTO2                                                            
 9    K2=K3                                                            
      GOTO2                                                             
 8    P=F(K3)                                                          
      RETURN                                                          
 3    B1=A(K1)                                                          
      B2=A(K1+1)                                                      
      B3=A(K1+2)                                                        
      B4=F(K1)                                                        
      B5=F(K1+1)                                                        
      B6=F(K1+2)                                                       
      R=B4*((X-B2)*(X-B3))/((B1-B2)*(B1-B3))+B5*((X-B1)*(X-B3))/       
     1 ((B2-B1)*(B2-B3))+B6*((X-B1)*(X-B2))/((B3-B1)*(B3-B2))           
      RETURN                                                          
 6    IF(K2.NE.N)GO TO 3                                               
      K1=N-2                                                            
      GOTO3                                                            
 4    C=ABS(X-A(N))                                                     
      IF(C.LT.0.1d-7) GO TO 5                                           
      K1=N-2                                                           
 13   CONTINUE                                                          
C13   PRINT 41,X                                                        
C41   FORMAT(25H X IS OUT OF THE INTERVAL,3H X=,F15.9)                  
      GO TO 3                                                           
 5    R=F(N)                                                           
      RETURN                                                            
 11   C=ABS(X-A(1))                                                     
      IF(C.LT.0.1d-7) GO TO 12                                         
      K1=1                                                             
      GOTO 13                                                           
 12   R=F(1)                                                            
      RETURN                                                            
      END                                                              
C************************** END PARINV *************************************

* quark-quark scattering differential cross section 
       double precision FUNCTION PLSIGH(Z)
       IMPLICIT DOUBLE PRECISION(A-H, O-Z)
       common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
       save /plpar1/
       pi=3.14159d0
       beta=(33.d0-2.d0*nf)/(12.d0*pi)  
       alfs=1.d0/(beta*dlog(max(1.d-10,z/(TC*TC)))) 
       PLSIGH=8.d0*pi*alfs*alfs/(9.d0*z*z) 
       return
       end 

* differential radiated gluon spectrum in BDMS model 
      double precision FUNCTION PLFUN1(or) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
      common /pljdat/ ej, z, ygl, alfs, um, epa 
      common /pleave/ taul, temlev    
      common /factor/ cfac, kf 
      save /plpar1/,/pljdat/,/pleave/,/factor/
      pi=3.14159d0
      x=min((1.d0-ygl*z/or),or/ej)  
      if(x.le.0.d0) x=0.d0 
      if(x.ge.1.d0) x=0.9999d0      
      if(kf.eq.21) then 
       if(x.ge.0.5d0) x=1.d0-x 
       spinf=0.5d0*(1.+(1.d0-x)**4+x**4)/(1.d0-x)            
      else 
       spinf=1.d0-x+0.5d0*x*x 
      end if   
      ak=ygl*z/(or*(1.d0-x)) 
      al=taul*5.06d0 
      uu=0.5d0*al*dsqrt(abs(0.5d0*(1.d0-x+cfac*x*x)*ak*
     >   dlog(max(16.d0/ak,1.d-10))))/ygl  
* if  quark production outside the QGP then 
* arg=(((dsin(uu)*cosh(uu))**2)+((dcos(uu)*sinh(uu))**2))/(2.d0*uu*uu);   
* here quark production inside the QGP  
      arg=((dcos(uu)*cosh(uu))**2)+((dsin(uu)*sinh(uu))**2)   
      gl1=(ygl/(cfac*z))**0.3333333d0
      gl2=(um/epa)**1.333333d0  
      dc=1.d0/((1.d0+((gl1*gl2*or)**1.5d0))**2)       ! massive parton    
c      dc=1.d0                                         !massless parton 
      plfun1=dc*3.d0*alfs*ygl*dlog(max(arg,1.d-20))*spinf/(pi*al*or)   
      return 
      end  

* angular distribution of emitted gluons       
      double precision function gluang(x) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      s=0.0863d0 
      gluang=x*dexp(-1.d0*(x-s)*(x-s)/(2.d0*s*s)) 
      return
      end    

* temperature-dependence of parton-plasma integral cross section 
       double precision FUNCTION PLS(X)
       IMPLICIT DOUBLE PRECISION(A-H, O-Z)
       external plsigh 
       common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
       common /plpar2/ pln0,taupl,tauh,sigpl,sigh,sigplh,sigqqh,rg,rgn 
       common /plen/ epartc, um  
       save /plpar1/,/plpar2/,/plen/ 
       t=X 
       pi=3.14159d0
       bub=4.d0*t/TC   
       alf=6.d0*pi/((33.d0-2.d0*nf)*dlog(max(bub,1.d-10)))
       ZZ0=4.d0*t*t*pi*alf*(1.d0+nf/6.d0)
       scm=4.d0*t*epartc+um*um+4.d0*t*t  
       ZZ1=max((scm-((um+2.d0*t)**2))*(scm-((um-2.d0*t)**2))/scm,ZZ0)      
       HH1=0.01d0*ZZ1  
       REPS=0.01d0 
       AEPS=1.d-8
       CALL SIMPA(ZZ0,ZZ1,HH1,REPS,AEPS,plsigh,ZZ,RESS,AIH,AIABS) 
       PLS=0.39d0*2.25d0*2.25d0*RESS*(16.d0+4.d0*nf)/(16.d0+9.d0*nf) 
       return
       end

* temperature-dependence of QGP viscosity (if mvisc=1,2)  
      double precision FUNCTION PLVISC(X) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
      save /plpar1/
      pi=3.14159d0
      T=X 
      TC1=5.06d0*TC 
      if(X.le.TC1) T=TC1  
      if(mvisc.eq.0) then 
       c=0.d0
      elseif(mvisc.eq.1) then 
       a=3.4d0*(1.d0+0.12d0*(2.d0*nf+1.d0))
       b=15.d0*(1.d0+0.06d0*nf)
       c=4.d0*pi*pi*(10.5d0*nf/a+16.d0/b)/675.d0         
      else 
       c=(1.7d0*nf+1.d0)*0.342d0/(1.d0+nf/6.d0)
      end if 
      bub=4.d0*T/TC1   
      alf=6.d0*pi/((33.d0-2.d0*nf)*dlog(max(bub,1.d-10)))
      alf1=1.d0/alf 
      PLVISC=c*(T**3)/(alf*alf*dlog(max(1.d-10,alf1)))  
      return
      end 

* space-time dependence of QGP number density 
       double precision FUNCTION PLN(X,r1,r2,y)  
       IMPLICIT DOUBLE PRECISION(A-H, O-Z)
       external pythik
       common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
       common /plpar2/ pln0,taupl,tauh,sigpl,sigh,sigplh,sigqqh,rg,rgn 
       common /plevol/ taup(5000),temp(5000),denp(5000),enep(5000)  
       common /pythic/ PBAB(110),PTAB(110),PTAAB(110)
       save /plpar1/,/plpar2/,/plevol/,/pythic/
       pi=3.14159d0
       t=X       
       if(t.lt.taupl) then
        call parinv(t,taup,denp,5000,res)    
       else
        res=1.2d0*(16.d0+9.d0*nf)*((5.06d0*TC)**3)/(pi*pi)
       end if 
       res=res*(pythik(r1)*pythik(r2)*pi*RA*RA/PTAAB(1))**0.75d0 
       res=res*dexp(-1.d0*y*y/24.5d0)
       PLN=max(1.d-8,res)
       return 
       end

* space-time dependence of QGP temperature 
       double precision FUNCTION PLT(X,r1,r2,y)  
       IMPLICIT DOUBLE PRECISION(A-H, O-Z)
       common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf
       common /plpar2/ pln0,taupl,tauh,sigpl,sigh,sigplh,sigqqh,rg,rgn 
       common /plevol/ taup(5000),temp(5000),denp(5000),enep(5000) 
       common /pythic/ PBAB(110),PTAB(110),PTAAB(110)
       save /plpar1/,/plpar2/,/plevol/,/pythic/
       pi=3.14159d0
       t=X       
       if(t.lt.taupl) then
        call parinv(t,taup,temp,5000,res)    
       else
        res=TC
       end if 
        res=res*(pythik(r1)*pythik(r2)*pi*RA*RA/PTAAB(1))**0.25d0 
	res=res*(dexp(-1.d0*y*y/24.5d0))**0.333333d0
        PLT=max(1.d-8,res)  
       return 
       end

* impact parameter dependence of jet production cross section
      double precision function funbip(x) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external ftaa 
      common /pyint7/ sigt(0:6,0:6,0:5)
      save /pyint7/ 
      br=x 
      sigin=sigt(0,0,0)-sigt(0,0,1)
      taa=ftaa(br)
      funbip=taa*br*(1.d0-dexp(-0.1d0*taa*sigin)) 
      return 
      end 

* distribution over jet production vertex position  
      double precision FUNCTION plthik(X)  
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external pythik
      common /parimp/ b1, psib1, rb1, rb2, noquen 
      save /parimp/
      bu=X
      r12=dsqrt(abs(bu*bu+b1*b1/4.d0+bu*b1*dcos(psib1))) 
      r22=dsqrt(abs(bu*bu+b1*b1/4.d0-bu*b1*dcos(psib1)))  
      PLTHIK=bu*pythik(r12)*pythik(r22) 
      return
      end

* nuclear overlap function at impact parameter b  
      double precision function ftaa(r)  
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      common /pythic/ PBAB(110),PTAB(110),PTAAB(110)
      save /pythic/ 
      call parinv(r,PBAB,PTAAB,110,RES) 
      ftaa=RES 
      return 
      end   
*
      double precision function PFUNC1(x) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external PFUNC2 
      common /pynup1/ bp,xx 
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf       
      save /plpar1/
      xx=x 
      EPS=0.05d0 
      A=0.d0 
      B=3.d0*RA
      H=0.01d0*(B-A)    
      CALL SIMPB(A,B,H,EPS,1.d-8,PFUNC2,Y,RES,AIH,AIABS)
      PFUNC1=RES 
      return 
      end   
*      
      double precision function PFUNC2(y) 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      external pythik 
      common /pynup1/ bp,x 
      r1=sqrt(abs(y*y+bp*bp/4.+y*bp*cos(x))) 
      r2=sqrt(abs(y*y+bp*bp/4.-y*bp*cos(x)))
      PFUNC2=y*pythik(r1)*pythik(r2) 
      return 
      end  

* nuclear thickness function 
      double precision function pythik(r)   
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      common /pythic/ PBAB(110),PTAB(110),PTAAB(110)
      save /pythic/ 
      call parinv(r,PBAB,PTAB,110,RES) 
      pythik=RES 
      return
      end

* Wood-Saxon nucleon distrubution  
      double precision function prhoaa(z)  
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      common /plpar1/ tau0,T0,TC,sigqq,AW,RA,mvisc,nf       
      common /pygeom/ BC 
      save /plpar1/,/pygeom/
      pi=3.14159d0
      df=0.54d0
      r=sqrt(bc*bc+z*z)
      rho0=3.d0/(4.d0*pi*RA**3)/(1.d0+(pi*df/RA)**2)
      prhoaa=rho0/(1.d0+exp((r-RA)/df))
      return
      end

* function to generate gauss distribution
      double precision function gauss(x0,sig)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
 41   u1=pyr(0) 
      u2=pyr(0)  
      v1=2.d0*u1-1.d0
      v2=2.d0*u2-1.d0 
      s=v1**2+v2**2
      if(s.gt.1) go to 41
      gauss=v1*dsqrt(-2.d0*dlog(s)/s)*sig+x0
      return
      end    
**************************************************************************   
