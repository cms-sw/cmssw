      subroutine f2bosb(xs,xxii,xyi,xt,xu,res)
c Returns the real matrix elements times xii**2*(1-yi**2)=4*t*u/s**2.
c The normalization is such that
c   sigma_2body=gs**2*e**2*zeq**2*(f2bosb/(xii**2*(1-yi**2)))*dphi_2
c for gamma production; the factor zeq**2 is replaced by
c   |Vqq|**2         for W production
c   Aq**2+Vq**2      for Z production
      implicit none
      real * 8 xs,xxii,xyi,xt,xu,res(4)
      include 'svbcblks.h'
      real * 8 s,xii,yi,t,u,tmp(4)
      integer jproc0,j
      common/cjproc/jproc0
c
      s=xs
      xii=xxii
      yi=xyi
      t=xt
      u=xu
      if(jproc0.eq.2)then
        call f2bosb_qq(s,xii,yi,t,u,tmp)
      elseif(jproc0.eq.3)then
        call f2bosb_qg(s,xii,yi,t,u,tmp)
      else
        write(*,*)'Unknown process in f2bosb',jproc0
        stop
      endif
      do j=1,4
        res(j)=tmp(j)
      enddo
      return
      end

        
      subroutine f2bosb_qq(xs,xxii,xyi,xt,xu,res)
c qq --> Vg, real matrix element
      implicit none
      real * 8 xs,xxii,xyi,xt,xu,res(4)
      include 'svbcblks.h'
      real * 8 s,xii,yi,t,u,tiny,pi,vcf,xnc,tofks,xmat,xnorm,
     #  x_ap,s_red,ap_kern,coupling,tmp(4)
      integer iprdct,icode,itwo,j
      common/ciprdct/iprdct
      parameter (tiny=1.d-6)
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (xnc=3.d0)
      parameter (itwo=2)
c
      s=xs
      xii=xxii
      yi=xyi
      t=xt
      u=xu
      tofks=2*pi/s
      if(xii.lt.tiny)then
        s_red=s*(1-xii)
        call f1sbrn(s_red,itwo,'sf',tmp)
        do j=1,4
          res(j)=16*vcf/s_red*tmp(j)
        enddo
      elseif(yi.gt.1-tiny)then
        x_ap=1-xii
        s_red=s*x_ap
        icode=4
        call f1sbrn(s_red,itwo,'c+',tmp)
        do j=1,4
          res(j)=4*(1+yi)/s*ap_kern(x_ap,abs(icode))*tmp(j)
        enddo
      elseif(yi.lt.-1+tiny)then
        x_ap=1-xii
        s_red=s*x_ap
        icode=4
        call f1sbrn(s_red,itwo,'c-',tmp)
        do j=1,4
          res(j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*tmp(j)
        enddo
      else
c From eq.(77) of NPB157(79)461
        if(iprdct.eq.1396)then
          coupling=1.d0
        elseif(iprdct.eq.1397)then
          coupling=xm2/(4.d0*sthw2*cthw2*xm02)
        elseif(iprdct.eq.1497.or.iprdct.eq.1498)then
          coupling=xm2/(4.d0*sthw2*xm02)
        else
          write(*,*)'Error in f2bosb_qq: unknown IPROC',iprdct
          stop
        endif
        xnorm=vcf/(2*pi*xnc)
        xmat=4*(2*s*xm2+u**2+t**2)/s**2
        do j=1,4
          res(j)=coupling*xnorm*xmat*tofks
        enddo
      endif
      return
      end


      subroutine f2bosb_qg(xs,xxii,xyi,xt,xu,res)
c qg --> Vq, real matrix element
      implicit none
      real * 8 xs,xxii,xyi,xt,xu,res(4)
      include 'svbcblks.h'
      real * 8 s,xii,yi,t,u,tiny,pi,vtf,xnc,tofks,xmatd,xmatr,xnorm,
     #  x_ap,s_red,ap_kern,coupling,tmp(4)
      integer iprdct,icode,itwo,j
      common/ciprdct/iprdct
      parameter (tiny=1.d-6)
      parameter (pi=3.14159265358979312D0)
      parameter (vtf=1.d0/2.d0)
      parameter (xnc=3.d0)
      parameter (itwo=2)
c
      s=xs
      xii=xxii
      yi=xyi
      t=xt
      u=xu
      tofks=2*pi/s
      do j=1,4
        res(j)=0.d0
      enddo
      if(xii.lt.tiny)then
        continue
      elseif(yi.gt.1-tiny)then
        x_ap=1-xii
        s_red=s*x_ap
        icode=2
        call f1sbrn(s_red,itwo,'c+',tmp)
        do j=3,4
          res(j)=4*(1+yi)/s*ap_kern(x_ap,abs(icode))*tmp(j)
        enddo
      elseif(yi.lt.-1+tiny)then
        x_ap=1-xii
        s_red=s*x_ap
        icode=2
        call f1sbrn(s_red,itwo,'c-',tmp)
        do j=1,2
          res(j)=4*(1-yi)/s*ap_kern(x_ap,abs(icode))*tmp(j)
        enddo
      else
c From eq.(85) of NPB157(79)461
        if(iprdct.eq.1396)then
          coupling=1.d0
        elseif(iprdct.eq.1397)then
          coupling=xm2/(4.d0*sthw2*cthw2*xm02)
        elseif(iprdct.eq.1497.or.iprdct.eq.1498)then
          coupling=xm2/(4.d0*sthw2*xm02)
        else
          write(*,*)'Error in f2bosb_qg: unknown IPROC',iprdct
          stop
        endif
        xnorm=vtf/(2*pi*xnc)
        xmatd=-4*t*(2*t*xm2+u**2+s**2)/s**3
        xmatr=-4*u*(2*u*xm2+t**2+s**2)/s**3
        do j=1,2
          res(j)=coupling*xnorm*xmatd*tofks
        enddo
        do j=3,4
          res(j)=coupling*xnorm*xmatr*tofks
        enddo
      endif
      return
      end


      subroutine f1sbrn(xs,jproc,c2,res)
c Born matrix element, times flux factor times normalizations and averages.
c   sigma_born=e**2*zeq**2*f1sbrn*dphi_1       for gamma production
c where e is the positron charge, zeq is the charge of the quark in 
c units of e. The factor zeq**2 is replaced by
c   |Vqq|**2         for W production
c   Aq**2+Vq**2      for Z production
      implicit none
      character * 2 c2
      real * 8 xs,res(4)
      integer jproc
      include 'svbcblks.h'
      real * 8 s,pi,xnc,tiny,coupling,xmat
      integer iprdct,j
      common/ciprdct/iprdct
      parameter (pi=3.14159265358979312D0)
      parameter (xnc=3.d0)
      parameter (tiny=1.d-8)
c
      s=xs
      if(gah.eq.0.d0.and.abs(s-xm2).gt.tiny)then
        write(*,*)'Fatal error in f1sbrn',s,xm2,c2
        stop
      endif
      if(iprdct.eq.1396)then
        coupling=1.d0
      elseif(iprdct.eq.1397)then
        coupling=xm2/(4.d0*sthw2*cthw2*xm02)
      elseif(iprdct.eq.1497.or.iprdct.eq.1498)then
        coupling=xm2/(4.d0*sthw2*xm02)
      else
        write(*,*)'Error in f1sbrn: unknown IPROC',iprdct
        stop
      endif
      if(jproc.eq.2)then
        xmat=coupling/(2*xnc)
      else
        xmat=0.d0
      endif
      do j=1,4
        res(j)=xmat
      enddo
      return
      end


      subroutine fosb(xs,jproc,res)
c Returns sig_2pv of FKS. It is derived from the subroutine xmatel_2pv_contr
c of that package. The Ellis-Sexton scale is set equal to the factorization
c scale, thus there's no contribution from Q. The normalization is
c   sigma_fosb=gs**2*e**2*zeq**2/(8*pi**2)*fosb*dphi_1
c for gamma production; the factor zeq**2 is replaced by
c   |Vqq|**2         for W production
c   Aq**2+Vq**2      for Z production
      implicit none
      real * 8 xs,res(4)
      include 'svbcblks.h'
      real * 8 tiny,pi,s,eikcon,fincon,vcf,xicut,delta,xmat,tmp(4)
      integer jproc,itwo,j
      common/parsub/xicut,delta
      parameter (tiny=1.d-8)
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (itwo=2)
c
      s=xs
      if(gah.eq.0.d0.and.abs(s-xm2).gt.tiny)then
        write(*,*)'Fatal error in fosb',s,xm2
        stop
      endif
      if(abs(xmuf2h1-xmuf2h2).gt.tiny .or.
     #   abs(xmuf2h1-xmur2).gt.tiny)then
        write(*,*)'No such scale choice'
        stop
      endif
      if(jproc.eq.2)then
        eikcon=2*vcf*( 0.5d0*log(xicut**2*s/xmuf2h1)**2-
     #                 pi**2/6.d0 )
        fincon=-vcf*(log(xmuf2h1/xm2)**2+3*log(xmuf2h1/xm2)-pi**2+8)
        xmat=eikcon+fincon
      else
        xmat=0.d0
      endif
      if(xmat.ne.0.d0)call f1sbrn(s,itwo,'sv',tmp)
      do j=1,4
        res(j)=xmat*tmp(j)
      enddo
      return
      end


      subroutine f2b_cosb(xs,xxii,xxiic,xyic,xxlmude,res)
c Returns sig_2pr of FKS. It is derived from the subroutine xmatel_coll
c of that package, which returns xmtel and xmtel_sc, the latter being
c the contribution of the delta term and of the regular part of the
c change of scheme, as defined in NPB357(91)409. These contributions
c are not associated to plus prescriptions: therefore, they are multiplied
c here by xii, since a factor 1/xii appears in the main code. The possible
c numerical inaccuracies motivated the definition of the subroutine in
c the jet package -- here we just ignore the problem. The normalization is
c   sigma_2pr=gs**2*e**2*zeq**2/(8*pi**2)*f2b_cosb/xii*dphi_1
c for gamma production; the factor zeq**2 is replaced by
c   |Vqq|**2         for W production
c   Aq**2+Vq**2      for Z production
      implicit none
      real * 8 xs,xxii,xxiic,xyic,xxlmude,res(4)
      include 'svbcblks.h'
      real * 8 s,xii,xiic,yic,xlmude,x_ap,s_red,one,xicut,delta,
     # xdfct1,xdfct2,xdfct3p,xdfct3l,xdfct5,xrfct1,xrfct2,xrfct3p,
     # xrfct3l,xrfct5,ap_kern,apprime_kern,xkplus,xklog,xkreg,
     # xkdelta,xfct4(4)
      common/parsub/xicut,delta
      parameter (one=1.d0)
      character * 2 scheme
      integer jproc0,icoded,icoder,j
      common/cjproc/jproc0
      integer itwo
      parameter (itwo=2)
c
      s=xs
      xii=xxii
      xiic=xxiic
      yic=xyic
      xlmude=xxlmude
c
      x_ap=1-xiic
      s_red=s*x_ap
      if(yic.eq.1.d0)then
        scheme=schhad1
        if(jproc0.eq.2)then
          icoded=4
          icoder=4
        elseif(jproc0.eq.3)then
          icoded=0
          icoder=2
        else
          write(*,*)'Unknown process in f2b_cosb',jproc0
          stop
        endif
      elseif(yic.eq.-1.d0)then
        scheme=schhad2
        if(jproc0.eq.2)then
          icoded=4
          icoder=4
        elseif(jproc0.eq.3)then
          icoded=2
          icoder=0
        else
          write(*,*)'Unknown process in f2b_cosb',jproc0
          stop
        endif
      else
        write(6,*)'Error in f2b_cosb',yic
        stop
      endif
      if(icoded.ne.0.or.icoder.ne.0)then
        call f1sbrn(s_red,itwo,'pr', xfct4)
      else
        do j=1,4
          res(j)=0.d0
        enddo
      endif
      if(icoded.ne.0)then
        xdfct1=ap_kern(x_ap,abs(icoded))
        xdfct2=apprime_kern(x_ap,abs(icoded))
        xdfct3p=0.d0
        xdfct3l=0.d0
        xdfct5=0.d0
c
        if(scheme.eq.'DI')then
        xdfct3p=xkplus(x_ap,abs(icoded))
          xdfct3l=xklog(x_ap,abs(icoded))
          if(xiic.ne.0.d0)then
            xdfct5=xkreg(x_ap,abs(icoded))
          else
            xdfct5=xkdelta(abs(icoded))
     #            +xkplus(one,abs(icoded))*log(xicut)
     #            +xklog(one,abs(icoded))*log(xicut)**2/2.d0
c This part contributes to sig2pr(soft), which is integrated in xi
c over the range (0,xicut). This implies the presence of a jacobian
c equal to xicut in the soft term, which has to be removed by hand
c in this case
            xdfct5=xdfct5/xicut
          endif
        elseif(scheme.ne.'MS')then
          write(6,*)'Error in f2b_cosb, y=',yic
          write(6,*)'Factorization scheme ',scheme,' not known'
        endif
c
        do j=1,2
          res(j)=( xdfct1*(xlmude+2*log(xii))-xdfct2
     #            -xdfct3p-xdfct3l*log(xii) )*xfct4(j)
     #           -xii*xdfct5*xfct4(j)
        enddo
      else
        do j=1,2
          res(j)=0.d0
        enddo
      endif
      if(icoder.ne.0)then
        xrfct1=ap_kern(x_ap,abs(icoder))
        xrfct2=apprime_kern(x_ap,abs(icoder))
        xrfct3p=0.d0
        xrfct3l=0.d0
        xrfct5=0.d0
c
        if(scheme.eq.'DI')then
        xrfct3p=xkplus(x_ap,abs(icoder))
          xrfct3l=xklog(x_ap,abs(icoder))
          if(xiic.ne.0.d0)then
            xrfct5=xkreg(x_ap,abs(icoder))
          else
            xrfct5=xkdelta(abs(icoder))
     #            +xkplus(one,abs(icoder))*log(xicut)
     #            +xklog(one,abs(icoder))*log(xicut)**2/2.d0
c This part contributes to sig2pr(soft), which is integrated in xi
c over the range (0,xicut). This implies the presence of a jacobian
c equal to xicut in the soft term, which has to be removed by hand
c in this case
            xrfct5=xrfct5/xicut
          endif
        elseif(scheme.ne.'MS')then
          write(6,*)'Error in f2b_cosb, y=',yic
          write(6,*)'Factorization scheme ',scheme,' not known'
        endif
c
        do j=3,4
          res(j)=( xrfct1*(xlmude+2*log(xii))-xrfct2
     #            -xrfct3p-xrfct3l*log(xii) )*xfct4(j)
     #           -xii*xrfct5*xfct4(j)
        enddo
      else
        do j=3,4
          res(j)=0.d0
        enddo
      endif
      return
      end

Cc
Cc From the jet package, Altarelli-Parisi kernels and change of scheme
Cc
C      function ap_kern(x,index)
Cc This function returns the quantity (1-x)*P_{ab}(x), where
Cc P_{ab} are the Altarelli-Parisi kernels, and the splitting partons
Cc {ab} are defined with the following conventions
Cc
Cc         index          ab
Cc
Cc           1            gg
Cc           2            qg
Cc           3            gq
Cc           4            qq
Cc
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
Cc
C      if(index.eq.1)then
C        ap_kern=2*vca*(x+(1-x)**2/x+x*(1-x)**2)
C      elseif(index.eq.2)then
C        ap_kern=vtf*(1-x)*(x**2+(1-x)**2)
C      elseif(index.eq.3)then
C        ap_kern=vcf*(1-x)*(1+(1-x)**2)/x
C      elseif(index.eq.4)then
C        ap_kern=vcf*(1+x**2)
C      else
C        write(6,*)'Error in ap_kern: wrong index value'
C        stop
C      endif
C      return
C      end


C      function apprime_kern(x,index)
Cc This function returns the quantity (1-x)*P_{ab}^{prime}(x), where
Cc P_{ab}^{prime} is the ep-dependent part of the Altarelli-Parisi kernels, 
Cc and the codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
Cc
C      if(index.eq.1)then
C        apprime_kern=0.d0
C      elseif(index.eq.2)then
C        apprime_kern=-2*vtf*x*(1-x)**2
C      elseif(index.eq.3)then
C        apprime_kern=-vcf*(1-x)*x
C      elseif(index.eq.4)then
C        apprime_kern=-vcf*(1-x)**2
C      else
C        write(6,*)'Error in apprime_kern: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xkdelta(index)
Cc This function returns the quantity K^{(d)}_{ab}, relevant for
Cc the MS --> DIS change in the factorization scheme. 
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (pi=3.14159265358979312D0)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xkdelta=0.d0
C      elseif(index.eq.2)then
C        xkdelta=0.d0
C      elseif(index.eq.3)then
C        xkdelta=vcf*(9.d0/2.d0+pi**2/3.d0)
C      elseif(index.eq.4)then
C        xkdelta=-vcf*(9.d0/2.d0+pi**2/3.d0)
C      else
C        write(6,*)'Error in xkdelta: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xkplus(x,index)
Cc This function returns the quantity K^{(+)}_{ab}(x), relevant for
Cc the MS --> DIS change in the factorization scheme. Notice that
Cc there's NO multiplicative (1-x) factor like in the previous functions.
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xkplus=0.d0
C      elseif(index.eq.2)then
C        xkplus=0.d0
C      elseif(index.eq.3)then
C        xkplus=-vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
C      elseif(index.eq.4)then
C        xkplus=vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
C      else
C        write(6,*)'Error in xkplus: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xklog(x,index)
Cc This function returns the quantity K^{(l)}_{ab}(x), relevant for
Cc the MS --> DIS change in the factorization scheme. Notice that
Cc there's NO multiplicative (1-x) factor like in the previous functions.
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xklog=0.d0
C      elseif(index.eq.2)then
C        xklog=0.d0
C      elseif(index.eq.3)then
C        xklog=-vcf*(1+x**2)
C      elseif(index.eq.4)then
C        xklog=vcf*(1+x**2)
C      else
C        write(6,*)'Error in xklog: wrong index value'
C        stop
C      endif
C      return
C      end


C      function xkreg(x,index)
Cc This function returns the quantity K^{(reg)}_{ab}(x), relevant for
Cc the MS --> DIS change in the factorization scheme. Notice that
Cc there's NO multiplicative (1-x) factor like in the previous functions.
Cc The codes for the splitting partons {ab} are defined above
C      implicit real * 8 (a-h,o-z)
C      parameter (vcf=4.d0/3.d0)
C      parameter (vtf=1.d0/2.d0)
C      parameter (vca=3.d0)
C      parameter (xnc=3.d0)
C      common/nl/nl
Cc
C      if(index.eq.1)then
C        xkreg=-2*nl*vtf*( (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1 )
C      elseif(index.eq.2)then
C        xkreg=vtf*( (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1 )
C      elseif(index.eq.3)then
C        xkreg=0.d0
C      elseif(index.eq.4)then
C        xkreg=0.d0
C      else
C        write(6,*)'Error in xkreg: wrong index value'
C        stop
C      endif
C      return
C      end


