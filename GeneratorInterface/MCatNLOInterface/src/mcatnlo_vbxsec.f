c
c
c Begin of general routines
c
c
      subroutine vbborn(s,t,zmw2,zmz2,brnup,brndo)
      implicit real * 8 (a-h,o-z)
      character * 2 prc,prdct
      common/process/prc,prdct
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-') then
        brnup = wzborn(t,s,zmw2,zmz2)
        brndo=0.d0
      elseif(prdct.eq.'z ') then
        u = 2*zmz2-s-t
        brnup = ( zzborn(s,t,zmz2)+zzborn(s,u,zmz2) )/2.d0
        brndo=0.d0
      elseif(prdct.eq.'ww') then
        call wwborn(s,t,xbrnup,xbrndo)
        brnup=xbrnup
        brndo=xbrndo
      else
        write(*,*) 'non implemented final state', prdct
        stop
      endif
      end


      subroutine vb2(s,t,zmw2,zmz2,ww2up,ww2do)
      implicit real * 8 (a-h,o-z)
      character * 2 prc,prdct
      common/process/prc,prdct
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
c
      if(prc.ne.'qq') then
        write(6,*)'vb2: non existent process ',prc
        stop
      endif
      if(prdct.eq.'w+'.or.prdct.eq.'w-') then
        ww2up = 4.d0/3.d0*wz2b(t,s,zmw2,zmz2)
        ww2do=0.d0
      elseif(prdct.eq.'z ') then
        u = 2*zmz2-s-t
        xmu2 = xmufct2
        ww2up = 4.d0/3.d0*( zz2b(t,s,zmz2,xmu2)+
     #                      zz2b(u,s,zmz2,xmu2) )/2.d0
        ww2do=0.d0
      elseif(prdct.eq.'ww') then
        call ww2b(s,t,xww2up,xww2do)
        ww2up=xww2up
        ww2do=xww2do
      else
        write(*,*) 'non implemented final state', prdct
        stop
      endif
      end


      subroutine vbppcolp(y,s,q1q,x,xlmude,wwpup,wwpdo)
      implicit real * 8 (a-z)
      character * 2 prc,prdct
      common/zmass/zmw,zmz,zmw2,zmz2
      common/process/prc,prdct
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-')then
        ymw2=zmw2
        ymz2=zmz2
        if(prc.eq.'qq') then
          wwpup = 4/3.d0*wzcolp(x,q1q,s,ymw2,ymz2,xlmude)
          wwpdo = 0.d0
        elseif(prc.eq.'qg'.or.prc.eq.'ag') then
          wwpup = .5d0*(1-x)*wzcolg(x,q1q,s,ymw2,ymz2,xlmude)
          wwpdo = 0.d0
        else
          write(6,*) 'error in vbppcolp: prc=',prc
          stop
        endif
      elseif(prdct.eq.'z ')then
        ymw2=zmw2
        ymz2=zmz2
        q1c = 2*zmz2-x*s-q1q
        if(prc.eq.'qq') then
          wwpup = 4/3.d0*( zzcolp(x,q1q,s,ymz2,xlmude)+
     #                     zzcolp(x,q1c,s,ymz2,xlmude) )/2.d0
          wwpdo = 0.d0
        elseif(prc.eq.'qg'.or.prc.eq.'ag') then
          wwpup = .5d0*(1-x)*( zzcolg(x,q1q,s,ymz2,xlmude)+
     #                         zzcolg(x,q1c,s,ymz2,xlmude) )/2.d0
          wwpdo = 0.d0
        else
          write(6,*) 'error in vbppcolp: prc=',prc
          stop
        endif
      elseif(prdct.eq.'ww')then
        call wwppcolp(y,s,q1q,x,xlmude,xwwpup,xwwpdo)
        wwpup = xwwpup
        wwpdo = xwwpdo
      else
        write(*,*) 'non implemented final state', prdct
        stop
      endif
      return
      end


      subroutine vbppcoll(y,s,q1q,x,wwlup,wwldo)
      implicit real * 8 (a-z)
      character * 2 prc,prdct
      common/zmass/zmw,zmz,zmw2,zmz2
      common/process/prc,prdct
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-')then
        ymw2=zmw2
        ymz2=zmz2
        if(prc.eq.'qq') then
          wwlup = 4/3.d0*wzcoll(x,q1q,s,ymw2,ymz2)
          wwldo = 0.d0
        elseif(prc.eq.'qg'.or.prc.eq.'ag') then
          wwlup = 0.d0
          wwldo = 0.d0
        else
          write(6,*) 'error in vbppcoll: prc=',prc
          stop
        endif
      elseif(prdct.eq.'z ')then
        ymw2=zmw2
        ymz2=zmz2
        q1c = 2*zmz2-x*s-q1q
        if(prc.eq.'qq') then
          wwlup = 4/3.d0*( zzcoll(x,q1q,s,ymz2)+
     #                     zzcoll(x,q1c,s,ymz2) )/2.d0
          wwldo = 0.d0
        elseif(prc.eq.'qg'.or.prc.eq.'ag') then
          wwlup = 0.d0
          wwldo = 0.d0
        else
          write(6,*) 'error in vbppcoll: prc=',prc
          stop
        endif
      elseif(prdct.eq.'ww')then
        call wwppcoll(y,s,q1q,x,xwwlup,xwwldo)
        wwlup = xwwlup
        wwldo = xwwldo
      else
        write(*,*) 'non implemented final state', prdct
        stop
      endif
      return
      end


      subroutine vbfpp(s,x,y,q1q,q2q,ww3up,ww3do)
      implicit real * 8 (a-z)
      character * 2 prc,prdct
      common/zmass/zmw,zmz,zmw2,zmz2
      common/process/prc,prdct
c
      if(prdct.eq.'w+'.or.prdct.eq.'w-') then
        ymw2=zmw2
        ymz2=zmz2
        tt = -s/2*(1-x)*(1-y)
        uu = -s/2*(1-x)*(1+y)
        if(prc.eq.'qq') then
          ww3up = 4/3.d0*wz3b(x,y,tt,uu,q1q,q2q,s,ymw2,ymz2)
          ww3do = 0.d0
        elseif(prc.eq.'qg'.or.prc.eq.'ag') then
c includiamo un fattore -2*tt/s (wz3g = -2*uu*s*M^2)
          ww3up = .5d0*(1-x)*(1-y)*wz3g(x,y,tt,uu,q1q,q2q,s,ymw2,ymz2)
          ww3do = 0.d0
        else
          write(6,*) 'non existent subprocess ',prc
          stop
        endif
      elseif(prdct.eq.'z ') then
        ymw2=zmw2
        ymz2=zmz2
        tt = -s/2*(1-x)*(1-y)
        uu = -s/2*(1-x)*(1+y)
        q1c = -tt-s-q1q+2*zmz2
        q2c = -uu-s-q2q+2*zmz2
        if(prc.eq.'qq') then
          ww3up = 4/3.d0*( zzff(x,y,tt,uu,q1q,q2q,s,ymz2)+
     #                     zzff(x,y,tt,uu,q1c,q2c,s,ymz2) )/2.d0
          ww3do = 0.d0
        elseif(prc.eq.'qg') then
c includiamo un fattore -2*tt/s (wz3g = -2*uu*s*M^2)
          ww3up = .5d0*(1-x)*(1-y)*( 
     #                zzffg(x,y,tt,uu,q1q,q2q,s,ymz2)+
     #                zzffg(x,y,tt,uu,q1c,q2c,s,ymz2) )/2.d0
          ww3do = 0.d0
        elseif(prc.eq.'ag') then
          write(6,*) 'non existent subprocess in ZZ production ',prc
          stop
        else
          write(6,*) 'non existent subprocess ',prc
          stop
        endif
      elseif(prdct.eq.'ww')then
        call wwfpp(s,x,y,q1q,q2q,xww3up,xww3do)
        ww3up = xww3up
        ww3do = xww3do
      else
        write(6,*) 'non existent final state',prdct
        stop
      endif
      return
      end
c
c
c End of general routines
c
c
c
c
c Begin of wwcross.for
c
c
c
c Born term
c
      function xbrntt(s,t,mw2)
      implicit real * 8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      u = -t-s+2*mw2
      xbrntt = 2*(8*(mw2**2/t**2+1/4.d0)*(t*u/mw2**2-1)+8*s/mw2)
      xbrntt = xbrntt/(8*s*xnc)
      return 
      end 


      function xbrnts(s,t,mw2)
      implicit real * 8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      u = -t-s+2*mw2
      xbrnts = 16*s*((-mw2**2/(s*t)-mw2/(2.d0*s)+1/4.d0)*(t*u/mw2**2-1
     1   )+2*mw2/t+s/mw2-2)
      xbrnts = xbrnts/(8*s*xnc)
      return 
      end 


      function xbrnss(s,t,mw2)
      implicit real * 8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      u = -t-s+2*mw2
      xbrnss = 8*s**2*((-mw2/s+3*mw2**2/s**2+1/4.d0)*(t*u/mw2**2-1)+s/m
     1   w2-4)
      xbrnss = xbrnss/(8*s*xnc)
      return 
      end 


      subroutine wwborn(s,t,brnup,brndo)
      implicit real*8(a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
c
      u = 2*xmw2-s-t
      ctt=1/(16*sw**4)
      ctsup=  ( 2/3.d0+(1/2.d0-2*sw**2/3)*s/(sw**2*(s-xmz2)) )/
     #        (4*s*sw**2)
      ctsdown=( -1/3.d0+(-1/2.d0+sw**2/3)*s/(sw**2*(s-xmz2)) )/
     #        (4*s*sw**2)
      tvup=  s*(1/2.d0-4*sw**2/3)/(2*sw**2*(s-xmz2))
      taup=  s/(4*sw**2*(s-xmz2))
      tvdown=s*(-1/2.d0+2*sw**2/3)/(2*sw**2*(s-xmz2))
      tadown=-taup
      cssup=  ( (2/3.d0+tvup)**2+taup**2 )/s**2
      cssdown=( (-1/3.d0+tvdown)**2+tadown**2 )/s**2
c
      brnup = ctt*xbrntt(s,t,xmw2)-
     #        ctsup*xbrnts(s,t,xmw2)+
     #        cssup*xbrnss(s,t,xmw2)
      brndo = ctt*xbrntt(s,u,xmw2)+
     #        ctsdown*xbrnts(s,u,xmw2)+
     #        cssdown*xbrnss(s,u,xmw2)
c
      return
      end
c
c
c Collinear term
c
c
      subroutine wwppcolp(y,s,q1q,x,xlmude,wwpup,wwpdo)
      implicit real*8(a-h,o-z)
      character * 2 prc,prdct
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/process/prc,prdct
      dimension bbp(6)
c
      if(prc.eq.'qq')then
         call wwqqcolp(s,q1q,x,xlmude,bbp)
      elseif(y.eq.-1.and.prc.eq.'qg')then
         call wwqgcolp2(s,q1q,x,xlmude,bbp)
      elseif(y.eq.-1.and.prc.eq.'ag')then
         call wwagcolp2(s,q1q,x,xlmude,bbp)
      elseif(y.eq.1.and.prc.eq.'qg'.or.y.eq.1.and.prc.eq.'ag')then
         do i=1,6
            bbp(i)=0
         enddo
      else
         write(6,*)'wwppcolp: non esistent process'
         stop
      endif
c
      xs = x*s
      ctt=1/(16*sw**4)
      ctsup=  ( 2/3.d0+(1/2.d0-2*sw**2/3)*xs/(sw**2*(xs-xmz2)) )/
     #        (4*xs*sw**2)
      ctsdown=( -1/3.d0+(-1/2.d0+sw**2/3)*xs/(sw**2*(xs-xmz2)) )/
     #        (4*xs*sw**2)
      tvup=  xs*(1/2.d0-4*sw**2/3)/(2*sw**2*(xs-xmz2))
      taup=  xs/(4*sw**2*(xs-xmz2))
      tvdown=xs*(-1/2.d0+2*sw**2/3)/(2*sw**2*(xs-xmz2))
      tadown=-taup
      cssup=  ( (2/3.d0+tvup)**2+taup**2 )/xs**2
      cssdown=( (-1/3.d0+tvdown)**2+tadown**2 )/xs**2
c
      wwpup = ctt*bbp(1)-
     #        ctsup*bbp(3)+
     #        cssup*bbp(5)
      wwpdo = ctt*bbp(2)+
     #        ctsdown*bbp(4)+
     #        cssdown*bbp(6)
c
      return
      end


      subroutine wwppcoll(y,s,q1q,x,wwlup,wwldo)
      implicit real*8(a-h,o-z)
      character * 2 prc,prdct
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/process/prc,prdct
      dimension bbl(6)
c
      if(prc.eq.'qq')then
         call wwqqcoll(s,q1q,x,bbl)
      elseif(y.eq.-1.and.prc.eq.'qg')then
         call wwqgcoll2(s,q1q,x,bbl)
      elseif(y.eq.-1.and.prc.eq.'ag')then
         call wwagcoll2(s,q1q,x,bbl)
      elseif(y.eq.1.and.prc.eq.'qg'.or.y.eq.1.and.prc.eq.'ag')then
         do i=1,6
            bbl(i)=0
         enddo
      else
         write(6,*)'wwppcoll: non esistent process'
         stop
      endif
c
      xs = x*s
      ctt=1/(16*sw**4)
      ctsup=  ( 2/3.d0+(1/2.d0-2*sw**2/3)*xs/(sw**2*(xs-xmz2)) )/
     #        (4*xs*sw**2)
      ctsdown=( -1/3.d0+(-1/2.d0+sw**2/3)*xs/(sw**2*(xs-xmz2)) )/
     #        (4*xs*sw**2)
      tvup=  xs*(1/2.d0-4*sw**2/3)/(2*sw**2*(xs-xmz2))
      taup=  xs/(4*sw**2*(xs-xmz2))
      tvdown=xs*(-1/2.d0+2*sw**2/3)/(2*sw**2*(xs-xmz2))
      tadown=-taup
      cssup=  ( (2/3.d0+tvup)**2+taup**2 )/xs**2
      cssdown=( (-1/3.d0+tvdown)**2+tadown**2 )/xs**2
c
      wwlup = ctt*bbl(1)-
     #        ctsup*bbl(3)+
     #        cssup*bbl(5)
      wwldo = ctt*bbl(2)+
     #        ctsdown*bbl(4)+
     #        cssdown*bbl(6)
c
      return
      end


      subroutine wwqqcolp(s,qq,x,xlmude,bbp)
      implicit real*8(a-h,o-z)
      character * 2 scheme
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/scheme/scheme
      common/nl/nl
      dimension bbp(6)
c
      vcf=4/3.d0
      xs=x*s
      xt=qq
      xu=2*xmw2-xs-xt
      xfact=vcf*( (1-x)**2 + (1+x**2)*xlmude )
      if(scheme.eq.'DI') then
         xfact=xfact-xkpqq(x,nl)
      elseif(scheme.ne.'MS') then
         write(6,*)'wwqqcolp: scheme not known'
         stop
      endif
      bbp(1)=xfact*xbrntt(xs,xt,xmw2)
      bbp(2)=xfact*xbrntt(xs,xu,xmw2)
      bbp(3)=xfact*xbrnts(xs,xt,xmw2)
      bbp(4)=xfact*xbrnts(xs,xu,xmw2)
      bbp(5)=xfact*xbrnss(xs,xt,xmw2)
      bbp(6)=xfact*xbrnss(xs,xu,xmw2)
      return
      end


      subroutine wwqqcoll(s,qq,x,bbl)
      implicit real*8(a-h,o-z)
      character * 2 scheme
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/scheme/scheme
      common/nl/nl
      dimension bbl(6)
c
      vcf=4/3.d0
      xs=x*s
      xt=qq
      xu=2*xmw2-xs-xt
      xfact=2*vcf*(1+x**2)
      if(scheme.eq.'DI') then
         xfact=xfact-xklqq(x,nl)
      elseif(scheme.ne.'MS') then
         write(6,*)'wwqqcoll: scheme not known'
         stop
      endif
      bbl(1)=xfact*xbrntt(xs,xt,xmw2)
      bbl(2)=xfact*xbrntt(xs,xu,xmw2)
      bbl(3)=xfact*xbrnts(xs,xt,xmw2)
      bbl(4)=xfact*xbrnts(xs,xu,xmw2)
      bbl(5)=xfact*xbrnss(xs,xt,xmw2)
      bbl(6)=xfact*xbrnss(xs,xu,xmw2)
      return
      end


      subroutine wwqgcolp2(s,q1q,x,xlmude,bbp)
      implicit real*8(a-h,o-z)
      character * 2 scheme
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/scheme/scheme
      common/nl/nl
      dimension bbp(6)
c
      vtf=1/2.d0
      xs=x*s
      xt=q1q
      xu=2*xmw2-xs-xt
      xfact=vtf*(1-x)*( 2*x*(1-x) + (x**2+(1-x)**2)*xlmude )
      if(scheme.eq.'DI') then
         xfact=xfact-xkpqg(x,nl)
      elseif(scheme.ne.'MS') then
         write(6,*)'wwqgcolp2: scheme not known'
         stop
      endif
      bbp(1)=xfact*xbrntt(xs,xt,xmw2)
      bbp(2)=xfact*xbrntt(xs,xu,xmw2)
      bbp(3)=xfact*xbrnts(xs,xt,xmw2)
      bbp(4)=xfact*xbrnts(xs,xu,xmw2)
      bbp(5)=xfact*xbrnss(xs,xt,xmw2)
      bbp(6)=xfact*xbrnss(xs,xu,xmw2)
      return
      end


      subroutine wwqgcoll2(s,q1q,x,bbl)
      implicit real*8(a-h,o-z)
      character * 2 scheme
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/scheme/scheme
      common/nl/nl
      dimension bbl(6)
c
      vtf=1/2.d0
      xs=x*s
      xt=q1q
      xu=2*xmw2-xs-xt
      xfact=2*vtf*(1-x)*(x**2+(1-x)**2)
      if(scheme.eq.'DI') then
         xfact=xfact-xklqg(x,nl)
      elseif(scheme.ne.'MS') then
         write(6,*)'wwqgcoll2: scheme not known'
         stop
      endif
      bbl(1)=xfact*xbrntt(xs,xt,xmw2)
      bbl(2)=xfact*xbrntt(xs,xu,xmw2)
      bbl(3)=xfact*xbrnts(xs,xt,xmw2)
      bbl(4)=xfact*xbrnts(xs,xu,xmw2)
      bbl(5)=xfact*xbrnss(xs,xt,xmw2)
      bbl(6)=xfact*xbrnss(xs,xu,xmw2)
      return
      end


      subroutine wwagcolp2(s,q1q,x,xlmude,bbp)
      implicit real*8(a-h,o-z)
      character * 2 scheme
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/scheme/scheme
      common/nl/nl
      dimension bbp(6)
c
      vtf=1/2.d0
      xs=x*s
      xt=q1q
      xu=2*xmw2-xs-xt
      xfact=vtf*(1-x)*( 2*x*(1-x) + (x**2+(1-x)**2)*xlmude )
      if(scheme.eq.'DI') then
         xfact=xfact-xkpqg(x,nl)
      elseif(scheme.ne.'MS') then
         write(6,*)'wwagcolp2: scheme not known'
         stop
      endif
      bbp(2)=xfact*xbrntt(xs,xt,xmw2)
      bbp(1)=xfact*xbrntt(xs,xu,xmw2)
      bbp(4)=xfact*xbrnts(xs,xt,xmw2)
      bbp(3)=xfact*xbrnts(xs,xu,xmw2)
      bbp(6)=xfact*xbrnss(xs,xt,xmw2)
      bbp(5)=xfact*xbrnss(xs,xu,xmw2)
      return
      end


      subroutine wwagcoll2(s,q1q,x,bbl)
      implicit real*8(a-h,o-z)
      character * 2 scheme
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/scheme/scheme
      common/nl/nl
      dimension bbl(6)
c
      vtf=1/2.d0
      xs=x*s
      xt=q1q
      xu=2*xmw2-xs-xt
      xfact=2*vtf*(1-x)*(x**2+(1-x)**2)
      if(scheme.eq.'DI') then
         xfact=xfact-xklqg(x,nl)
      elseif(scheme.ne.'MS') then
         write(6,*)'wwagcoll2: scheme not known'
         stop
      endif
      bbl(2)=xfact*xbrntt(xs,xt,xmw2)
      bbl(1)=xfact*xbrntt(xs,xu,xmw2)
      bbl(4)=xfact*xbrnts(xs,xt,xmw2)
      bbl(3)=xfact*xbrnts(xs,xu,xmw2)
      bbl(6)=xfact*xbrnss(xs,xt,xmw2)
      bbl(5)=xfact*xbrnss(xs,xu,xmw2)
      return
      end
c
c
c Two body term
c
c
      subroutine ww2b(s,t,ww2up,ww2do)
      implicit real*8(a-h,o-z)
      character * 2 scheme
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/betfac/betfac,delta
      common/scheme/scheme
      common/nl/nl
      dimension bbb(6)
c
      ro=4*xmw2/s
      b = dsqrt(1-ro)
      xlb = log(b*betfac)
      u = -t-s+2*xmw2
      bbb(1)=xtt(s,t,xmw2)
      bbb(2)=xtt(s,u,xmw2)
      bbb(3)=xts(s,t,xmw2)
      bbb(4)=xts(s,u,xmw2)
      bbb(5)=xss(s,t,xmw2)
      bbb(6)=xss(s,u,xmw2)
      if(scheme.eq.'DI')then
         one = 1.d0
         xk = xkdqq(nl) + 2*xkpqq(one,nl)*xlb + 2*xklqq(one,nl)*xlb**2
         bbb(1)=bbb(1)-4*xk*xbrntt(s,t,xmw2)
         bbb(2)=bbb(2)-4*xk*xbrntt(s,u,xmw2)
         bbb(3)=bbb(3)-4*xk*xbrnts(s,t,xmw2)
         bbb(4)=bbb(4)-4*xk*xbrnts(s,u,xmw2)
         bbb(5)=bbb(5)-4*xk*xbrnss(s,t,xmw2)
         bbb(6)=bbb(6)-4*xk*xbrnss(s,u,xmw2)
      elseif(scheme.ne.'MS')then
         write(6,*)'ww2b: error in scheme'
         stop
      endif
c
      ctt=1/(16*sw**4)
      ctsup=  ( 2/3.d0+(1/2.d0-2*sw**2/3)*s/(sw**2*(s-xmz2)) )/
     #        (4*s*sw**2)
      ctsdown=( -1/3.d0+(-1/2.d0+sw**2/3)*s/(sw**2*(s-xmz2)) )/
     #        (4*s*sw**2)
      tvup=  s*(1/2.d0-4*sw**2/3)/(2*sw**2*(s-xmz2))
      taup=  s/(4*sw**2*(s-xmz2))
      tvdown=s*(-1/2.d0+2*sw**2/3)/(2*sw**2*(s-xmz2))
      tadown=-taup
      cssup=  ( (2/3.d0+tvup)**2+taup**2 )/s**2
      cssdown=( (-1/3.d0+tvdown)**2+tadown**2 )/s**2
c
      ww2up = ctt*bbb(1)-
     #        ctsup*bbb(3)+
     #        cssup*bbb(5)
      ww2do = ctt*bbb(2)+
     #        ctsdown*bbb(4)+
     #        cssdown*bbb(6)
c
      return
      end   

      
      function xtt(s,t,mw2)
      implicit real * 8 (a-z)
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/betfac/betfac,delta
c
      pi2=pi*pi
      vcf = 4/3.d0
      mu2 = xmufct2
      beta = (1-4*mw2/s)**(1/2.d0)
      xlbeta = log(betfac*beta)
      xlp = (beta+1)/2.d0
      xlm = (1-beta)/2.d0
      xls = log(s/mw2)
      xlt = log(-t/mw2)
      xlsmu = log(s/mu2)
      xi4 = (-4*ddilog(t/mw2)+2*log(-t/mw2)**2+4*log(-mw2/t)*log((mw2-t)
     1   /mw2))/(s*t)
      xi3t = (-log(-t/s)**2/2.d0+log(mw2/s)**2/2.d0-pi2/2.d0)/(mw2-t)
      xi3l = (log(xlm/xlp)**2+4*ddilog(-xlm/xlp)+pi2/3.d0)/(beta*s)
      sfc = (16*xlbeta+6)*xlsmu+32*xlbeta**2-4*pi2/3.d0
      xttbrn = 2*sfc*(8*(mw2**2/t**2+1/4.d0)*((-t-s+2*mw2)*t/mw2**2-1)+
     1   8*s/mw2)
      xttreg = (-16*(t**2-3*s*t-3*mw2**2)/t**2-64*s/(t-mw2))*xlt+(-(16*t
     1   **2-64*s*t-13*s**2)/((s-4*mw2)*t)+4*(6*t**2+8*s*t-19*mw2*t+12*m
     2   w2**2)/t**2+3*s*(4*t+s)**2/((s-4*mw2)**2*t))*xls+16*s*(t**2+s*t
     3   -2*mw2*t+2*mw2**2)*xi4/t
      xttreg = xttreg-32*(t-mw2)*(t**2+s*t-2*mw2*t+2*mw2**2)*xi3t/t**2+(
     1   (16*t**2+16*s*t+32*mw2*t-49*s**2+72*mw2*s-48*mw2**2)/(4.d0*t)+
     2   s*(8*t**2-14*s*t-3*s**2)/((s-4*mw2)*t)-3*s**2*(4*t+s)**2/(4.d0
     3   *(s-4*mw2)**2*t))*xi3l+2*(80*t**2+73*s*t-140*mw2*t+72*mw2**2)/t
     4   **2+pi2*(-16*(3*t+2*s-4*mw2)/(3.d0*t)+32*(t+2*s)/( 3.d0*mw2)-1
     5   6*t*(t+s)/(3.d0*mw2**2))-2*(4*t+s)**2/((s-4*mw2)*t)-64*(t+2*s)
     6   /mw2+32*t*(t+s)/mw2**2
      xtt = 2*(xttreg+xttbrn/2)
c      xtt = 2*xttbrn
      xtt = xtt*vcf/(8*s*xnc)
      return 
      end 
      

      function xts(s,t,mw2)
      implicit real * 8 (a-z)
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/betfac/betfac,delta
c
      pi2=pi*pi
      vcf = 4/3.d0
      mu2 = xmufct2
      beta = (1-4*mw2/s)**(1/2.d0)
      xlbeta = log(betfac*beta)
      xlp = (beta+1)/2.d0
      xlm = (1-beta)/2.d0
      xls = log(s/mw2)
      xlt = log(-t/mw2)
      xlsmu = log(s/mu2)
      xi4 = (-4*ddilog(t/mw2)+2*log(-t/mw2)**2+4*log(-mw2/t)*log((mw2-t)
     1   /mw2))/(s*t)
      xi3t = (-log(-t/s)**2/2.d0+log(mw2/s)**2/2.d0-pi2/2.d0)/(mw2-t)
      xi3l = (log(xlm/xlp)**2+4*ddilog(-xlm/xlp)+pi2/3.d0)/(beta*s)
      sfc = (16*xlbeta+6)*xlsmu+32*xlbeta**2-4*pi2/3.d0
      xtsbrn = 16*s*sfc*((-mw2**2/(s*t)-mw2/( 2.d0*s)+1/4.d0)*((-t-s+2*
     1   mw2)*t/mw2**2-1)+2*mw2/t+s/mw2-2)
      xtsreg = (8*(t**2-5*s*t+2*mw2*t-6*mw2*s-3*mw2**2)/t+32*s*(2*t+s)/(
     1   t-mw2)-16*s**2*t/(t-mw2)**2)*xlt+(8*s*(4*t+s)/(s-4*mw2)-8*(3*t*
     2   *2-2*s*t-6*mw2*t+6*mw2*s+3*mw2**2)/t)*xls+(-8*s*t**2+16*mw2*s*(
     3   t-2*s)+8*s**2*t-16*mw2**2*s)*xi4
      xtsreg = xtsreg+16*(t-mw2)*(t**2-s*t-2*mw2*t+4*mw2*s+2*mw2**2)*xi3
     1   t/t+(-2*s**2*(4*t+s)/(s-4*mw2)-8*mw2*(2*t+7*s)+16*s*t-6*s**2+16
     2   *mw2**2)*xi3l+pi2*(-16*(t**2+4*s*t+4*mw2*t-4*mw2*s-2*mw2**2)/( 
     3   3.d0*t)+32*(t**2+2*s*t+2*s**2)/(3.d0*mw2)-16*s*t*(t+s)/(3.d0*
     4   mw2**2))-64*(t**2+2*s*t+2*s**2)/mw2-8*(t**2-21*s*t-26*mw2*t+34*
     5   mw2*s+17*mw2**2)/t+32*s*t*(t+s)/mw2**2+16*s**2/(t-mw2)
      xts = 2*(xtsreg+xtsbrn/2)
c      xts = 2*xtsbrn
      xts = xts*vcf/(8*s*xnc)
      return 
      end 
      

      function xss(s,t,mw2)
      implicit real * 8 (a-z)
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/betfac/betfac,delta
c
      pi2=pi*pi
      vcf = 4/3.d0
      mu2 = xmufct2
      beta = (1-4*mw2/s)**(1/2.d0)
      xlbeta = log(betfac*beta)
      xlsmu = log(s/mu2)
      sfc = (16*xlbeta+6)*xlsmu+32*xlbeta**2-4*pi2/3.d0
      xssbrn = 8*s**2*sfc*((-mw2/s+3*mw2**2/s**2+1/4.d0)*((-t-s+2*mw2)*
     1   t/mw2**2-1)+s/mw2-4)
      xssreg = pi2*(-8*(12*t**2+20*s*t-24*mw2*t+17*s**2-4*mw2*s+12*mw2**
     1   2)/3.d0+16*s*(2*t**2+3*s*t+2*s**2)/(3.d0*mw2)-8*s**2*t*(t+s)/
     2   ( 3.d0*mw2**2))-32*s*(2*t**2+3*s*t+2*s**2)/mw2+192*t**2-64*mw2*
     3   (6*t+s)+16*s**2*t*(t+s)/mw2**2+320*s*t+272*s**2+192*mw2**2
      xss = 2*(xssreg+xssbrn/2)
c      xss = 2*xssbrn
      xss = xss*vcf/(8*s*xnc)
      return 
      end 
c      
c
c Three body term
c
c
      subroutine wwfpp(s,x,y,q1q,q2q,ww3up,ww3do)
      implicit real*8(a-h,o-z)
      character * 2 prc,prdct
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/process/prc,prdct
      dimension bbb(6)
c
      if(prc.eq.'qq') then
         call ww3qq(s,x,y,q1q,q2q,bbb)
      elseif(prc.eq.'qg') then
         call ww3qg(s,x,y,q1q,q2q,bbb)
      elseif(prc.eq.'ag') then
         call ww3ag(s,x,y,q1q,q2q,bbb)
      else
         write(6,*)'wwfpp: non esistent process'
         stop
      endif
c
      xs = x*s
      ctt=1/(16*sw**4)
      ctsup=  ( 2/3.d0+(1/2.d0-2*sw**2/3)*xs/(sw**2*(xs-xmz2)) )/
     #        (4*xs*sw**2)
      ctsdown=( -1/3.d0+(-1/2.d0+sw**2/3)*xs/(sw**2*(xs-xmz2)) )/
     #        (4*xs*sw**2)
      tvup=  xs*(1/2.d0-4*sw**2/3)/(2*sw**2*(xs-xmz2))
      taup=  xs/(4*sw**2*(xs-xmz2))
      tvdown=xs*(-1/2.d0+2*sw**2/3)/(2*sw**2*(xs-xmz2))
      tadown=-taup
      cssup=  ( (2/3.d0+tvup)**2+taup**2 )/xs**2
      cssdown=( (-1/3.d0+tvdown)**2+tadown**2 )/xs**2
c
      ww3up = ctt*bbb(1)-
     #        ctsup*bbb(3)+
     #        cssup*bbb(5)
      ww3do = ctt*bbb(2)+
     #        ctsdown*bbb(4)+
     #        cssdown*bbb(6)
c
      return
      end


      subroutine ww3qq(s,x,y,q1q,q2q,bbb)
      implicit real*8(a-h,o-z)
      character * 2 prc,prdct
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/mass/xmw,xmz,xmw2,xmz2
      common/betfac/betfac,delta
      common/process/prc,prdct
      common/nl/nl
      dimension bbb(6)
c
      tiny = .1d-5
      vtf=1/2.d0
      vcf=4/3.d0  
      tt=-s*(1-x)*(1-y)/2.d0
      uu=-s*(1-x)*(1+y)/2.d0
      if(1-x.le.tiny)then
         t = q1q
         u = 2*xmw2-s-uu-q2q
         xlim = 16*s*vcf
         bbb(1)=xlim*xbrntt(s,t,xmw2)
         bbb(2)=xlim*xbrntt(s,u,xmw2)
         bbb(3)=xlim*xbrnts(s,t,xmw2)
         bbb(4)=xlim*xbrnts(s,u,xmw2)
         bbb(5)=xlim*xbrnss(s,t,xmw2)
         bbb(6)=xlim*xbrnss(s,u,xmw2)
      elseif(1+y.le.tiny)then
         pqq = vcf*(1+x**2)/(1-x)
         xlim = -8*tt*pqq
         t = q1q
         sx = s*x
         u = 2*xmw2 - sx - t
         bbb(1)=xlim*xbrntt(sx,t,xmw2)
         bbb(2)=xlim*xbrntt(sx,u,xmw2)
         bbb(3)=xlim*xbrnts(sx,t,xmw2)
         bbb(4)=xlim*xbrnts(sx,u,xmw2)
         bbb(5)=xlim*xbrnss(sx,t,xmw2)
         bbb(6)=xlim*xbrnss(sx,u,xmw2)
      elseif(1-y.le.tiny)then
         pqq = vcf*(1+x**2)/(1-x)
         xlim = -8*uu*pqq
         t = q2q
         sx = s*x
         u = 2*xmw2 - sx - t
         bbb(1)=xlim*xbrntt(sx,t,xmw2)
         bbb(2)=xlim*xbrntt(sx,u,xmw2)
         bbb(3)=xlim*xbrnts(sx,t,xmw2)
         bbb(4)=xlim*xbrnts(sx,u,xmw2)
         bbb(5)=xlim*xbrnss(sx,t,xmw2)
         bbb(6)=xlim*xbrnss(sx,u,xmw2)
      else
         q1c = 2*xmw2-s-tt-q1q
         q2c = 2*xmw2-s-uu-q2q
         bbb(1)=vcf*xkrr(tt,uu,q1q,q2q,s,xmw2)
         bbb(2)=vcf*xkrr(tt,uu,q1c,q2c,s,xmw2)
         bbb(3)=vcf*xkrg(tt,uu,q1q,q2q,s,xmw2)
         bbb(4)=vcf*xkrg(tt,uu,q1c,q2c,s,xmw2)
         bbb(5)=vcf*xkgg(tt,uu,q1q,q2q,s,xmw2)
         bbb(6)=vcf*xkgg(tt,uu,q1c,q2c,s,xmw2)
      endif
      return
      end


      subroutine ww3qg(s,x,y,q1q,q2q,bbb)
      implicit real*8(a-h,o-z)
      character * 2 prc,prdct
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/mass/xmw,xmz,xmw2,xmz2
      common/betfac/betfac,delta
      common/process/prc,prdct
      common/nl/nl
      dimension bbb(6)
c
      tiny = .1d-5
      tt=-s*(1-x)*(1-y)/2.d0
      uu=-s*(1-x)*(1+y)/2.d0
      vtf=1/2.d0
      vcf=4/3.d0
      if(x.eq.1.or.y.eq.1)then
         do i=1,6
            bbb(i)=0
         enddo
      elseif(1+y.le.tiny)then
         pqg = vtf*(x**2+(1-x)**2)
         xlim = -8*tt*pqg
         t = q1q
         sx = s*x
         u = 2*xmw2 - sx - t
         bbb(1)=xlim*xbrntt(sx,t,xmw2)
         bbb(2)=xlim*xbrntt(sx,u,xmw2)
         bbb(3)=xlim*xbrnts(sx,t,xmw2)
         bbb(4)=xlim*xbrnts(sx,u,xmw2)
         bbb(5)=xlim*xbrnss(sx,t,xmw2)
         bbb(6)=xlim*xbrnss(sx,u,xmw2)
      else
         q1c = 2*xmw2-s-tt-q1q
         q2c = 2*xmw2-s-uu-q2q
         xflux = - vtf*tt**2/s**2
         zs   = tt
         ztt  = s
         zuu  = uu
         zq1q = q1q
         zq2q = xmw2+q1q-q2q-uu
         zq1c = q1c
         zq2c = xmw2-q1q+q2q-tt
         bbb(1)=xflux*xkrr(ztt,zuu,zq1q,zq2q,zs,xmw2)
         bbb(2)=xflux*xkrr(ztt,zuu,zq1c,zq2c,zs,xmw2)
         bbb(3)=xflux*xkrg(ztt,zuu,zq1q,zq2q,zs,xmw2)
         bbb(4)=xflux*xkrg(ztt,zuu,zq1c,zq2c,zs,xmw2)
         bbb(5)=xflux*xkgg(ztt,zuu,zq1q,zq2q,zs,xmw2)
         bbb(6)=xflux*xkgg(ztt,zuu,zq1c,zq2c,zs,xmw2)
      endif
      return
      end


      subroutine ww3ag(s,x,y,q1q,q2q,bbb)
      implicit real*8(a-h,o-z)
      character * 2 prc,prdct
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/mass/xmw,xmz,xmw2,xmz2
      common/betfac/betfac,delta
      common/process/prc,prdct
      common/nl/nl
      dimension bbb(6)
c
      tiny = .1d-5
      tt=-s*(1-x)*(1-y)/2.d0
      uu=-s*(1-x)*(1+y)/2.d0
      vtf=1/2.d0
      vcf=4/3.d0
      if(x.eq.1.or.y.eq.1)then
         do i=1,6
            bbb(i)=0
         enddo
      elseif(1+y.le.tiny)then
         pqg = vtf*(x**2+(1-x)**2)
         xlim = -8*tt*pqg
         t = q1q
         sx = s*x
         u = 2*xmw2 - sx - t
         bbb(2)=xlim*xbrntt(sx,t,xmw2)
         bbb(1)=xlim*xbrntt(sx,u,xmw2)
         bbb(4)=xlim*xbrnts(sx,t,xmw2)
         bbb(3)=xlim*xbrnts(sx,u,xmw2)
         bbb(6)=xlim*xbrnss(sx,t,xmw2)
         bbb(5)=xlim*xbrnss(sx,u,xmw2)
      else
         q1c = 2*xmw2-s-tt-q1q
         q2c = 2*xmw2-s-uu-q2q
         xflux = - vtf*tt**2/s**2
         zs   = tt
         ztt  = s
         zuu  = uu
         zq1q = q1q
         zq2q = xmw2+q1q-q2q-uu
         zq1c = q1c
         zq2c = xmw2-q1q+q2q-tt
         bbb(2)=xflux*xkrr(ztt,zuu,zq1q,zq2q,zs,xmw2)
         bbb(1)=xflux*xkrr(ztt,zuu,zq1c,zq2c,zs,xmw2)
         bbb(4)=xflux*xkrg(ztt,zuu,zq1q,zq2q,zs,xmw2)
         bbb(3)=xflux*xkrg(ztt,zuu,zq1c,zq2c,zs,xmw2)
         bbb(6)=xflux*xkgg(ztt,zuu,zq1q,zq2q,zs,xmw2)
         bbb(5)=xflux*xkgg(ztt,zuu,zq1c,zq2c,zs,xmw2)
      endif
      return
      end


      function xkrr(tt,uu,q1q,q2q,s,mw2)
      implicit real * 8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
c
      xkrr = 128*mw2*(q2q*tt*uu**2+q1q*tt**2*uu+6*q1q*q2q*tt*uu+q1q*q2q*
     1   s*uu+q1q*q2q**2*uu+q1q**3*uu+q1q*q2q*s*tt+q2q**3*tt+q1q**2*q2q*
     2   tt-2*q1q*q2q**2*s-2*q1q**2*q2q*s)/(q1q**2*q2q**2)-64*(2*tt*uu**
     3   2-2*q1q*uu**2+2*tt**2*uu+2*q2q*tt*uu+2*q1q*tt*uu-2*q1q*s*uu-q1q
     4   *q2q*uu-2*q2q*tt**2-2*q2q*s*tt-q1q*q2q*tt-2*q2q*s**2-2*q1q*s**2
     5   -2*q2q**2*s-q1q*q2q*s-2*q1q**2*s)/(q1q*q2q)
      xkrr = xkrr-32*(4*uu**2+2*tt*uu+9*s*uu+2*q2q*uu+2*q1q*uu+4*tt**2+9
     1   *s*tt+2*q2q*tt+2*q1q*tt+8*s**2+2*q2q*s+2*q1q*s)/mw2-256*mw2**2*
     2   (q2q*tt*uu+q1q*tt*uu+q1q*q2q*uu+q1q**2*uu+q2q**2*tt+q1q*q2q*tt-
     3   q1q*q2q*s)/(q1q**2*q2q**2)+32*(uu+tt+s)*(q2q*uu+q1q*tt+q2q*s+q1
     4   q*s+q2q**2+q1q**2)/mw2**2+256*mw2**3*(q1q*uu+q2q*tt)/(q1q**2*q2
     5   q**2)
      xkrr = - xkrr/(8*s*xnc)
      return 
      end 
      

      function xkrg(tt,uu,q1q,q2q,s,mw2)
      implicit real * 8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
c
      xkrg = 4*uu**2+2*tt*uu+9*s*uu+4*q2q*uu+2*q1q*uu+4*tt**2+9*s*tt+2*q
     1   2q*tt+4*q1q*tt+8*s**2+4*q2q*s+4*q1q*s+2*q2q**2+2*q1q**2
      xkrg = -32*(uu+tt+s)*xkrg/mw2
      xkrg = xkrg-64*(2*tt**2*uu**2+s*tt*uu**2+q2q*tt*uu**2+q1q*s*uu**2-
     1   2*q1q*q2q*uu**2+s*tt**2*uu+q1q*tt**2*uu-4*q1q*q2q*tt*uu-q2q*s**
     2   2*uu+q1q*s**2*uu-q2q**2*s*uu-5*q1q*q2q*s*uu-q1q**2*s*uu-q1q*q2q
     3   **2*uu-2*q1q**2*q2q*uu+q1q**3*uu+q2q*s*tt**2-2*q1q*q2q*tt**2+q2
     4   q*s**2*tt-q1q*s**2*tt-q2q**2*s*tt-5*q1q*q2q*s*tt-q1q**2*s*tt+q2
     5   q**3*tt-2*q1q*q2q**2*tt-q1q**2*q2q*tt-q2q**2*s**2-3*q1q*q2q*s**
     6   2-q1q**2*s**2+q2q**3*s-q1q*q2q**2*s-q1q**2*q2q*s+q1q**3*s)/(q1q
     7   *q2q)
      xkrg = xkrg+64*mw2*(4*tt*uu**2+s*uu**2+q2q*uu**2-2*q1q*uu**2+4*tt*
     1   *2*uu+6*s*tt*uu+q2q*tt*uu+q1q*tt*uu-q2q*s*uu-3*q1q*s*uu-q2q**2*
     2   uu+q1q**2*uu+s*tt**2-2*q2q*tt**2+q1q*tt**2-3*q2q*s*tt-q1q*s*tt+
     3   q2q**2*tt-q1q**2*tt-4*q2q*s**2-4*q1q*s**2+2*q2q**2*s+2*q1q*q2q*
     4   s+2*q1q**2*s)/(q1q*q2q)-128*mw2**2*(uu**2+tt*uu+2*s*uu-q2q*uu+t
     5   t**2+2*s*tt-q1q*tt+q2q*s+q1q*s)/(q1q*q2q)
      xkrg = xkrg+32*(uu+tt+s)**2*(q2q*uu+q1q*tt+q2q*s+q1q*s+q2q**2+q1q*
     1   *2)/mw2**2-128*mw2**3*(uu+tt)/(q1q*q2q)
      xkrg = - xkrg/(8*s*xnc)
      return 
      end 
      

      function xkgg(tt,uu,q1q,q2q,s,mw2)
      implicit real * 8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
c
      tmp1 = 9*uu**2+6*tt*uu+20*s*uu+10*q2q*uu+4*q1q*uu+9*tt**2+20*s*tt+
     1   4*q2q*tt+10*q1q*tt+17*s**2+10*q2q*s+10*q1q*s+6*q2q**2+6*q1q**2
      tmp1 = 32*tmp1*(uu+tt+s)
      tmp0 = 4*uu**2+2*tt*uu+9*s*uu+6*q2q*uu+2*q1q*uu+4*tt**2+9*s*tt+2*q
     1   2q*tt+6*q1q*tt+8*s**2+6*q2q*s+6*q1q*s+4*q2q**2+4*q1q**2
      tmp0 = -16*tmp0*(uu+tt+s)**2/mw2
      tmp0 = tmp1+tmp0
      xkgg = q2q*uu+q1q*tt+q2q*s+q1q*s+q2q**2+q1q**2
      xkgg = 16*(uu+tt+s)**3*xkgg/mw2**2
      xkgg = xkgg+tmp0
      xkgg = xkgg-64*mw2*(2*uu**2+10*tt*uu+7*s*uu+6*q2q*uu+6*q1q*uu+2*tt
     1   **2+7*s*tt+6*q2q*tt+6*q1q*tt+2*s**2+6*q2q*s+6*q1q*s)+384*mw2**2
     2   *(uu+tt+s)
      xkgg = - xkgg/(8*s*xnc)
      return 
      end 

c
c
c Funzioni di cambio di schema MS ----> DI
c
c
      function xkpqq(x,nl)
      implicit double precision (a-z)
      parameter (fot=4/3.d0)
      integer nl
      xkpqq = fot*(-1.5d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
      return
      end

      function xkdqq(nl)
      implicit double precision (a-z)
      parameter (fot=4/3.d0)
      integer nl
      data pi/3.141 592 653 589 793/
      xkdqq = -fot*(4.5d0 + pi**2/3)
      return
      end

      function xklqq(x,nl)
      implicit double precision (a-z)
      parameter (fot=4/3.d0)
      integer nl
      xklqq = fot*(1+x**2)
      return
      end

      function xkpqg(x,nl)
      implicit double precision (a-z)
      integer nl
      xkpqg = (1-x)*(-(x**2+(1-x)**2)*log(x)+8*x*(1-x)-1)/2
      return
      end

      function xkdqg(nl)
      implicit double precision (a-z)
      integer nl
      xkdqg = 0
      return
      end

      function xklqg(x,nl)
      implicit double precision (a-z)
      integer nl
      xklqg = (1-x)*(x**2+(1-x)**2)/2
      return
      end

      function xkdgg(nl)
      implicit double precision (a-z)
      integer nl
      xkdgg = - 2 * nl * xkdqg(nl)
      return
      end

      function xkpgg(x,nl)
      implicit double precision (a-z)
      integer nl
      xkpgg = - 2 * nl * xkpqg(x,nl)
      return
      end

      function xklgg(x,nl)
      implicit double precision (a-z)
      integer nl
      xklgg = - 2 * nl * xklqg(x,nl)
      return
      end

      function xkdgq(nl)
      implicit double precision (a-z)
      integer nl
      xkdgq = - xkdqq(nl)
      return
      end

      function xkpgq(x,nl)
      implicit double precision (a-z)
      integer nl
      xkpgq = - xkpqq(x,nl)
      return
      end

      function xklgq(x,nl)
      implicit double precision (a-z)
      integer nl
      xklgq = - xklqq(x,nl)
      return
      end
c
c
c Totale
c
c
      function xaint(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/mass/xmw,xmz,xmw2,xmz2
c
      s=4*xmw2/ro
      b=sqrt(1-ro)
      xaint=s**3*b**3/xmw2**2*
     #      ( 1/24.d0+5*xmw2/(6*s)+xmw2**2/(2*s**2) )
      xaint=xaint*s**2
      return
      end


      function xiint(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/mass/xmw,xmz,xmw2,xmz2
c
      s=4*xmw2/ro
      b=sqrt(1-ro)
      xiint=s**3*b/xmw2**2*
     #      ( 1/24.d0+3*xmw2/(4*s)-7*xmw2**2/(6*s**2)-xmw2**3/s**3 )
     #     -4*xmw2*(1.d0+xmw2/(2*s))*log((1+b)/(1-b))
      xiint=2*xiint*s
      return
      end


      function xeint(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/mass/xmw,xmz,xmw2,xmz2
c
      s=4*xmw2/ro
      b=sqrt(1-ro)
      xeint=s**3*b/xmw2**2*
     #     ( 1/24.d0+5*xmw2/(6*s)-2*xmw2**2/s**2 )
     #    +2*s*(1.d0-2*xmw2/s)*log((1+b)/(1-b))
      xeint=2*xeint
      return
      end


      function qqt0up(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
c
      s=4*xmw2/ro
      xnorm = 1/(8*s**2*pi*xnc)
      ctt=1/(16*sw**4)
      ctsup=  ( 2/3.d0+(1/2.d0-2*sw**2/3)*s/(sw**2*(s-xmz2)) )/
     #        (4*s*sw**2)
      tvup=  s*(1/2.d0-4*sw**2/3)/(2*sw**2*(s-xmz2))
      taup=  s/(4*sw**2*(s-xmz2))
      cssup=  ( (2/3.d0+tvup)**2+taup**2 )/s**2
      bornww = ctt*xeint(ro)-
     #         ctsup*xiint(ro)+
     #         cssup*xaint(ro)
      qqt0up = xnorm*bornww
      return
      end

      function qqt0do(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
c
      s=4*xmw2/ro
      xnorm = 1/(8*s**2*pi*xnc)
      ctt=1/(16*sw**4)
      ctsdown=( -1/3.d0+(-1/2.d0+sw**2/3)*s/(sw**2*(s-xmz2)) )/
     #        (4*s*sw**2)
      tvdown=s*(-1/2.d0+2*sw**2/3)/(2*sw**2*(s-xmz2))
      tadown=-s/(4*sw**2*(s-xmz2))
      cssdown=( (-1/3.d0+tvdown)**2+tadown**2 )/s**2
      bornww = ctt*xeint(ro)+
     #         ctsdown*xiint(ro)+
     #         cssdown*xaint(ro)
      qqt0do = xnorm*bornww
      return
      end


      function qqtup(ro)
      implicit real * 8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(9),f(9)
c data for two body contribution
c      data a/0.109461229D+01,0.201336566D+00,0.660359910D+00,
c     #       0.492864040D+00,-.675314369D-01,0.117296025D+00,
c     #       0.242052903D-01,0.403187989D+00,0.182958926D+00/
c
c data for two body + collinear contribution
c      data a/0.813993467D+00,0.558209013D+00,0.177587600D+01,
c     #       0.247003493D+01,0.502329200D+00,0.167561766D+00,
c     #       0.450787628D-01,0.407287917D+00,0.833196275D+00/
c
      data a/0.122180151D+01,0.166642696D+00,0.105739085D+01,
     #       0.111875308D+01,0.373203431D-01,-.215446708D-01,
     #       0.525151742D-01,0.409310242D+00,0.212281680D+00/
c
      b = sqrt(1-ro)
      born = xmw2*qqt0up(ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = born
      f(2) = born*ro
      f(3) = born*xlgb
      f(4) = -born*ro*xlgb
      f(5) = -born*ro*xlgx
      f(6) = -born*xlgx
      f(7) = born*xlgx**2
      f(8) = born*ro*xlgb**2
      f(9) = born*ro**2*xlgb

      sum = 0
      do i=1,9
        sum = sum + a(i)*f(i)
      enddo

      qqtup = sum
      return
      end


      function qqtdo(ro)
      implicit real * 8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(9),f(9)
c data for two body contribution
c      data a/0.109496154D+01,0.201469535D+00,0.667035160D+00,
c     #       0.501974049D+00,-.637432433D-01,0.117594412D+00,
c     #       0.242026690D-01,0.403226335D+00,0.185665081D+00/
c
c data for two body + collinear contribution
c      data a/0.809970412D+00,0.570512699D+00,0.180126267D+01,
c     #       0.247031770D+01,0.516764605D+00,0.168402352D+00,
c     #       0.450253286D-01,0.408033126D+00,0.813080862D+00/
c
      data a/0.120622602D+01,0.189602534D+00,0.113614667D+01,
     #       0.121859262D+01,0.759329949D-01,-.156271553D-01,
     #       0.520447209D-01,0.409937288D+00,0.237673334D+00/
c
      b = sqrt(1-ro)
      born = xmw2*qqt0do(ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = born
      f(2) = born*ro
      f(3) = born*xlgb
      f(4) = -born*ro*xlgb
      f(5) = -born*ro*xlgx
      f(6) = -born*xlgx
      f(7) = born*xlgx**2
      f(8) = born*ro*xlgb**2
      f(9) = born*ro**2*xlgb

      sum = 0
      do i=1,9
        sum = sum + a(i)*f(i)
      enddo

      qqtdo = sum
      return
      end


      function qq1bup(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/-.184743682D+01,0.180570890D+00,-.319652465D+01,
     #       -.587526010D+01,-.451067079D+00,0.738678311D+00,
     #       -.313189024D-02,-.437657631D-01,-.269263650D+01,
     #       0.421718470D-01,-.669043077D+00/
c 
      vcf=4/3.d0
      b=sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2
      f(11) = ro**2*xlgx
      sum = 0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      ffqqup = xmw2*sum*qqt0up(ro)
      qq1bup = -2*ffqqup/(8*pi**2)
     #         -2*vcf*(3/2.d0+4*log(b))*xmw2*qqt0up(ro)/(8*pi**2)
      return
      end


      function qq1bdo(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/-.186221984D+01,0.174137193D+00,-.300029266D+01,
     #       -.580430330D+01,-.429284162D+00,0.739861049D+00,
     #       -.318791684D-02,-.236252143D+00,-.282722341D+01,
     #       0.233627097D+00,-.966892821D+00/
c
      vcf=4/3.d0
      b=sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2
      f(11) = ro**2*xlgx
      sum = 0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      ffqqdo = xmw2*sum*qqt0do(ro)
      qq1bdo = -2*ffqqdo/(8*pi**2)
     #         -2*vcf*(3/2.d0+4*log(b))*xmw2*qqt0do(ro)/(8*pi**2)
      return
      end


      function cqqup(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
c
      cqqup = fcqqup_d(ro)+fcqqup_p(ro)+fcqqup_l(ro)
      cqqup = -2*cqqup/(8*pi**2)
      return
      end


      function fcqqup_d(ro)
      implicit real*8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      common/nl/nl
c
      b = sqrt(1-ro)
      xfact = xkdqq(nl)+2*xkpqq(1.d0,nl)*log(b)+
     #        2*xklqq(1.d0,nl)*log(b)**2
      xfact = xfact*xmw2
      fcqqup_d = xfact*qqt0up(ro)
      return
      end


      function fcqqup_p(ro)
      implicit real*8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.599533062D+01,-.469504818D+01,-.253700952D+01,
     #       -.294860885D+01,-.934929471D+00,0.121869212D+01,
     #       0.213339892D+00,0.936523370D+00,-.380582533D+00,
     #       -.933179743D+00,0.327829782D+01/
c 
      b=sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2 
      f(11) = ro**2*xlgx
      sum = 0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      fcqqup_p = xmw2*sum*qqt0up(ro)
      return
      end


      function fcqqup_l(ro)
      implicit real*8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.639350579D+00,0.102127404D+01,0.463161789D+01,
     #       0.101954192D+02,0.158459627D+01,-.181949117D+00,
     #       0.760452574D-02,0.171968905D-01,0.240640115D+01,
     #       -.951093526D-04,-.204293566D+01/
c 
      b=sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**4
      f(11) = ro**4*xlgx**2
      sum = 0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      fcqqup_l = xmw2*sum*qqt0up(ro)
      return
      end


      function cqqdo(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
c
      cqqdo = fcqqdo_d(ro)+fcqqdo_p(ro)+fcqqdo_l(ro)
      cqqdo = -2*cqqdo/(8*pi**2)
      return
      end


      function fcqqdo_d(ro)
      implicit real*8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      common/nl/nl
c
      b = sqrt(1-ro)
      xfact = xkdqq(nl)+2*xkpqq(1.d0,nl)*log(b)+
     #        2*xklqq(1.d0,nl)*log(b)**2
      xfact = xfact*xmw2
      fcqqdo_d = xfact*qqt0do(ro)
      return
      end


      function fcqqdo_p(ro)
      implicit real*8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.598774200D+01,-.467162401D+01,-.360405601D+01,
     #       -.448752719D+01,-.105742465D+01,0.121179707D+01,
     #       0.213426310D+00,0.117006555D+01,-.845776510D+00,
     #       -.116600777D+01,0.376563047D+01/
c 
      b=sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2 
      f(11) = ro**2*xlgx
      sum = 0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      fcqqdo_p = xmw2*sum*qqt0do(ro)
      return
      end


      function fcqqdo_l(ro)
      implicit real*8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.637236345D+00,0.986890897D+00,0.458606605D+01,
     #       0.100798003D+02,0.163756651D+01,-.181103038D+00,
     #       0.754127293D-02,0.114008070D-01,0.230877484D+01,
     #       -.236697307D-04,-.262210328D+01/
c 
      b=sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**4
      f(11) = ro**4*xlgx**2
      sum = 0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      fcqqdo_l = xmw2*sum*qqt0do(ro)
      return
      end


      function qgtup(ro)
      implicit real * 8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(14),f(14)
      data a/-.336542055D-03,-.734850131D-02,0.181537971D-01,
     #       0.719115080D-01,0.304132146D-01,0.470747078D-02,
     #       0.120155817D-03,-.274727164D-01,-.234503129D-02,
     #       -.174993518D+00,-.192793055D-01,-.962841575D-02,
     #       -.150068597D-02,0.246412146D+00/
c
      b = sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      born = b**2*ro
      f(1) = born*ro**3
      f(2) = born*ro**3*b**2
      f(3) = born*ro**3*b**4
      f(4) = born*b**2
      f(5) = born*b**2*xlgx
      f(6) = born*b**2*xlgx**2
      f(7) = born*b**2*xlgx**3
      f(8) = born*ro*b**2*xlgx**2
      f(9) = born*ro*b**2*xlgx**3
      f(10) = born*ro*b**2*xlgb
      f(11) = born*b*xlgb
      f(12) = born*b*xlgb**2
      f(13) = born*b*xlgb**3
      f(14) = born*b**2*xlgb

      sum = 0
      do i=1,14
        sum = sum + a(i)*f(i)
      enddo
      qgtup = sum
      return
      end


      function qgtdo(ro)
      implicit real * 8 (a-h,o-z)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(14),f(14)
      data a/-.341254475D-03,-.739275944D-02,0.182912755D-01,
     #       0.726720551D-01,0.307564844D-01,0.476402065D-02,
     #       0.123364831D-03,-.277576914D-01,-.238352425D-02,
     #       -.176697711D+00,-.195347289D-01,-.975258666D-02,
     #       -.152053881D-02,0.248957790D+00/
c
      b = sqrt(1-ro)
      xlgx = log(ro)
      xlgb = log(b)
      born = b**2*ro
      f(1) = born*ro**3
      f(2) = born*ro**3*b**2
      f(3) = born*ro**3*b**4
      f(4) = born*b**2
      f(5) = born*b**2*xlgx
      f(6) = born*b**2*xlgx**2
      f(7) = born*b**2*xlgx**3
      f(8) = born*ro*b**2*xlgx**2
      f(9) = born*ro*b**2*xlgx**3
      f(10) = born*ro*b**2*xlgb
      f(11) = born*b*xlgb
      f(12) = born*b*xlgb**2
      f(13) = born*b*xlgb**3
      f(14) = born*b**2*xlgb

      sum = 0
      do i=1,14
        sum = sum + a(i)*f(i)
      enddo
      qgtdo = sum
      return
      end


      function qg1bup(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(12),f(12)
      data a/0.124524966D-04,-.378694242D-02,0.943455727D-03,
     #       0.104521255D-01,0.366980971D-02,0.345392665D-02,
     #       -.208603184D-04,0.263954543D-02,-.436262307D-03,
     #       -.984645260D-02,-.450923484D-02,-.570717307D-03/
c 
      b = sqrt(1-ro)
      born = b**2*ro
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = ro**3
      f(2) = ro**3*b**2
      f(3) = ro**3*b**4
      f(4) = b**2
      f(5) = b**2*xlgx
      f(6) = b**2*xlgx**2
      f(7) = b**2*xlgx**3
      f(8) = ro*b**2*xlgx**2
      f(9) = ro*b**2*xlgx**3
      f(10) = ro*b**2*xlgb
      f(11) = b*xlgb
      f(12) = b*xlgb**2
      sum = 0
      do i=1,12
        sum = sum + a(i)*f(i)
      enddo
      qg1bup = -born*sum/(8*pi**2)
      return
      end


      function qg1bdo(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(12),f(12)
      data a/0.124386724D-04,-.375904232D-02,0.944846648D-03,
     #       0.103646947D-01,0.373218252D-02,0.345251663D-02,
     #       -.208739841D-04,0.270313750D-02,-.457809830D-03,
     #       -.987308498D-02,-.450787976D-02,-.570387337D-03/
c 
      b = sqrt(1-ro)
      born = b**2*ro
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = ro**3
      f(2) = ro**3*b**2
      f(3) = ro**3*b**4
      f(4) = b**2
      f(5) = b**2*xlgx
      f(6) = b**2*xlgx**2
      f(7) = b**2*xlgx**3
      f(8) = ro*b**2*xlgx**2
      f(9) = ro*b**2*xlgx**3
      f(10) = ro*b**2*xlgb
      f(11) = b*xlgb
      f(12) = b*xlgb**2
      sum = 0
      do i=1,12
        sum = sum + a(i)*f(i)
      enddo
      qg1bdo = -born*sum/(8*pi**2)
      return
      end


      function cqgup(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(12),f(12)
      data a/-.303231929D-04,0.316192148D-02,-.998633518D-03,
     #       -.195992549D-02,-.789161883D-02,-.230494415D-02,
     #       -.125175703D-02,-.233571149D-03,0.784443681D-03,
     #       0.284191498D-01,0.310662303D-01,0.129544440D-02/
c 
      b = sqrt(1-ro)
      born = b**2*ro
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = ro**3
      f(2) = ro**3*b**2
      f(3) = ro**3*b**4
      f(4) = b**2
      f(5) = b**2*xlgx
      f(6) = b**2*xlgx**2
      f(7) = b**2*xlgx**3
      f(8) = ro*b**2*xlgx**2
      f(9) = ro*b**2*xlgx**3
      f(10) = ro*b**2*xlgb
      f(11) = b*xlgb
      f(12) = b*xlgb**2
      sum = 0
      do i=1,12
        sum = sum + a(i)*f(i)
      enddo
      cqgup = -born*sum/(8*pi**2)
      return
      end


      function cqgdo(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(12),f(12)
      data a/-.302382081D-04,0.318186782D-02,-.936841512D-03,
     #       -.159792202D-02,-.786655275D-02,-.234673585D-02,
     #       -.125188972D-02,-.729517209D-03,0.780796381D-03,
     #       0.285736315D-01,0.310586938D-01,0.129356356D-02/
c 
      b = sqrt(1-ro)
      born = b**2*ro
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = ro**3
      f(2) = ro**3*b**2
      f(3) = ro**3*b**4
      f(4) = b**2
      f(5) = b**2*xlgx
      f(6) = b**2*xlgx**2
      f(7) = b**2*xlgx**3
      f(8) = ro*b**2*xlgx**2
      f(9) = ro*b**2*xlgx**3
      f(10) = ro*b**2*xlgb
      f(11) = b*xlgb
      f(12) = b*xlgb**2
      sum = 0
      do i=1,12
        sum = sum + a(i)*f(i)
      enddo
      cqgdo = -born*sum/(8*pi**2)
      return
      end
c
c
c Nuovi (20/4/93) fit per il totale. L'equivalenza fra i due sistemi
c di fit e' dimostrata. Il vecchio fit e' piu' preciso e leggibile,
c ma ha l'inconveniente di contenere non fattorizzata la dipendenza
c da alpha strong.
c
c
      function qqup_ms(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.252096950D+00,-.569187214D-01,-.816296947D+00,
     #       -.996627695D+00,-.186816009D+00,-.813325694D-01,
     #       0.372300809D-01,0.262638821D+00,-.329258824D+00,
     #       -.260568297D+00,0.355960451D+00/
c 
      b = sqrt(1-ro)
      born = xmw2*qqt0up(ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2
      f(11) = ro**2*xlgx
      sum=0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      qqup_ms = born*sum
      return
      end


      function qqup_di(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.704879251D-01,0.525942613D-01,-.591862482D+00,
     #       -.795334461D+00,-.139291231D+00,-.103800349D+00,
     #       0.313645646D-01,0.195520089D+00,-.271206930D+00,
     #       -.193620754D+00,0.165955388D+00/
c 
      b = sqrt(1-ro)
      born = xmw2*qqt0up(ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2
      f(11) = ro**2*xlgx
      sum=0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      qqup_di = born*sum
      return
      end


      function qqdo_ms(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.243630165D+00,-.558810010D-01,-.800400704D+00,
     #       -.990363550D+00,-.166288615D+00,-.779214764D-01,
     #       0.369548595D-01,0.229338241D+00,-.342588946D+00,
     #       -.227737817D+00,0.352355512D+00/
c 
      b = sqrt(1-ro)
      born = xmw2*qqt0do(ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2
      f(11) = ro**2*xlgx
      sum=0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      qqdo_ms = born*sum
      return
      end


      function qqdo_di(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(11),f(11)
      data a/0.650058662D-01,0.465039982D-01,-.545800116D+00,
     #       -.751561197D+00,-.119499085D+00,-.101130801D+00,
     #       0.311607861D-01,0.140882268D+00,-.279264745D+00,
     #       -.139717557D+00,0.134780642D+00/
c 
      b = sqrt(1-ro)
      born = xmw2*qqt0do(ro)
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = 1
      f(2) = ro
      f(3) = xlgb
      f(4) = -ro*xlgb
      f(5) = -ro*xlgx
      f(6) = -xlgx
      f(7) = xlgx**2
      f(8) = ro*xlgb**2
      f(9) = ro**2*xlgb
      f(10) = xlgb**2
      f(11) = ro**2*xlgx
      sum=0
      do i=1,11
        sum = sum + a(i)*f(i)
      enddo
      qqdo_di = born*sum
      return
      end


      function qgup_ms(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(14),f(14)
      data a/-.231664426D-03,-.505846787D-02,0.124964796D-01,
     #       0.495015145D-01,0.209354539D-01,0.324046728D-02,
     #       0.827113399D-04,-.189113146D-01,-.161424151D-02,
     #       -.120459784D+00,-.132712373D-01,-.662788313D-02,
     #       -.103302258D-02,0.169622011D+00/
c
      b = sqrt(1-ro)
      born = b**2*ro
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = ro**3
      f(2) = ro**3*b**2
      f(3) = ro**3*b**4
      f(4) = b**2
      f(5) = b**2*xlgx
      f(6) = b**2*xlgx**2
      f(7) = b**2*xlgx**3
      f(8) = ro*b**2*xlgx**2
      f(9) = ro*b**2*xlgx**3
      f(10) = ro*b**2*xlgb
      f(11) = b*xlgb
      f(12) = b*xlgb**2
      f(13) = b*xlgb**3
      f(14) = b**2*xlgb
      sum=0
      do i=1,14
        sum = sum + a(i)*f(i)
      enddo
      qgup_ms = born*sum
      return
      end


      function qgdo_ms(ro)
      implicit real * 8 (a-h,o-z)
      parameter (pi=3.14159265358979312D0)
      common/mass/xmw,xmz,xmw2,xmz2
      dimension a(14),f(14)
      data a/-.234908013D-03,-.508891979D-02,0.125910836D-01,
     #       0.500249538D-01,0.211717125D-01,0.327938938D-02,
     #       0.849200910D-04,-.191074427D-01,-.164073745D-02,
     #       -.121632643D+00,-.134470412D-01,-.671334845D-02,
     #       -.104668717D-02,0.171374013D+00/
c
      b = sqrt(1-ro)
      born = b**2*ro
      xlgx = log(ro)
      xlgb = log(b)
      f(1) = ro**3
      f(2) = ro**3*b**2
      f(3) = ro**3*b**4
      f(4) = b**2
      f(5) = b**2*xlgx
      f(6) = b**2*xlgx**2
      f(7) = b**2*xlgx**3
      f(8) = ro*b**2*xlgx**2
      f(9) = ro*b**2*xlgx**3
      f(10) = ro*b**2*xlgb
      f(11) = b*xlgb
      f(12) = b*xlgb**2
      f(13) = b*xlgb**3
      f(14) = b**2*xlgb
      sum=0
      do i=1,14
        sum = sum + a(i)*f(i)
      enddo
      qgdo_ms = born*sum
      return
      end
c
c
c End of wwcross.for
c
c
c
c
c Begin of wzcross.for
c
c
c----------------------------------------------------------
c d sigma(b) = f(i,j)**2 born(t,s,mw2,mz2) d phi(2)
c     f(i,j) = g/(2*sqrt(2)) * KM(i,j)
c----------------------------------------------------------
c
      function wzborn(t,s,mw2,mz2)
      implicit real *8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
c
      u = -t-s+mz2+mw2
      born = gup**2*(8*u**2*((t*u/(mw2*mz2)-1)/ 4.d0+(mz2+mw2)*s/(mw2
     1   *mz2* 2.d0))+8*(t*u-mw2*mz2))/u**2-ez*(16*gup*s*((-4*mw2*mz2/
     2   (s*u)-(mz2+mw2)/s+1)*(t*u/(mw2*mz2)-1)/ 4.d0+(mz2+mw2)*(2*mw2
     3   *mz2/u+s-mz2-mw2)/(mw2*mz2* 2.d0))/(s-mw2)-16*gdown*s*((-4*mw
     4   2*mz2/(s*t)-(mz2+mw2)/s+1)*(t*u/(mw2*mz2)-1)/ 4.d0+(mz2+mw2)*
     5   (2*mw2*mz2/t+s-mz2-mw2)/(mw2*mz2* 2.d0))/(s-mw2))
      born = 2*gdown*gup*(8*(mz2+mw2)*s-8*t*u*((t*u/(mw2*mz2)-1)/ 4.d0
     1   +(mz2+mw2)*s/(mw2*mz2* 2.d0)))/(t*u)+gdown**2*(8*t**2*((t*u/
     2   (mw2*mz2)-1)/ 4.d0+(mz2+mw2)*s/(mw2*mz2* 2.d0))+8*(t*u-mw2*
     3   mz2))/t**2+ez**2*(8*s**2*(t*u/(mw2*mz2)-1)*(((mz2+mw2)**2+8*mw2
     4   *mz2)/(s**2* 4.d0)+1/ 4.d0-(mz2+mw2)/(s* 2.d0))+8*(mz2+mw
     5   2)*s**2*(s/ 2.d0+(mw2-mz2)**2/(s* 2.d0)-mz2-mw2)/(mw2*mz2))
     6   /(s-mw2)**2+born
      wzborn = born/(xnc*s*4.d0)
c
      return
      end
c
c---------------------------------------------
c Contributo a tre corpi collineare alla sezione d'urto totale per qq->zzg
c
c d sigma^(c+) = f(i,j)**2 (alfas C_F)/(4*pi)
c *[ colp(x,q2q)*P{1/(1-x)} + coll(x,q2q)P{log(1-x)/(1-x)} ]*d Phi2^(x)
c
c e analoga per d sigma^(c-), con q2q--> q1q
c
c P{1/(1-x)} e' {1/(1-x)}_+
c P{log(1-x)/(1-x)} e' {log(1-x)/(1-x)}_+
c
      function wzcolp(x,q2q,s,mw2,mz2,xlmude)
      implicit double precision (a-z)
      character * 2 scheme
      common/scheme/scheme
      born0 = wzborn(q2q,x*s,mw2,mz2)
      wzcolp = 2*(xlmude*(1+x**2)+(1-x)**2)*born0
      if(scheme.eq.'DI') then
         dis = -1.5d0-(1+x**2)*log(x)+(1-x)*(3+2*x)
         wzcolp = wzcolp - 2*dis*born0
      elseif(scheme.eq.'MS') then
      else
         call noscheme
      endif
      return
      end

      function wzcoll(x,q2q,s,mw2,mz2)
      implicit double precision (a-z)
      character * 2 scheme
      common/scheme/scheme
      born0 = wzborn(q2q,x*s,mw2,mz2)
      wzcoll = 4*(1+x**2)*born0
      if(scheme.eq.'DI') then
         dis = 1+x**2
         wzcoll = wzcoll - 2*dis*born0
      elseif(scheme.eq.'MS') then
      else
         call noscheme
      endif
      return
      end
c
c---------------------------------------------------------------
c d sigma(sv) = 4 pi alfas C_F N f(i,j)**2 wz2b(t,s,mw2,mz2) d phi(2)
c C_F = (N_C**2 - 1)/(2 N_C)
c N = 1/(16 pi2) in n_dim=4
c---------------------------------------------------------------
c
      function wz2b(t,s,mw2,mz2)
      implicit real *8 (a-z)
      character * 2 scheme
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
      common/scheme/scheme
      common/betfac/betfac,delta
      data pi2/9.869604401089358d0/
c
      mu2=xmufct2
      nc=xnc
      sig = mz2+mw2
      del = mw2-mz2
      p = mw2*mz2
      u = -t-s+mz2+mw2
      xltw = log(-t/mw2)
      xltz = log(-t/mz2)
      xlts = log(-t/s)
      xluw = log(-u/mw2)
      xluz = log(-u/mz2)
      xlus = log(-u/s)
      betal = dsqrt(1-(dsqrt(mz2)+dsqrt(mw2))**2/s)
      xlbeta = log(betal*betfac)
      i4t = (log(-t/mz2)**2/ 2.d0+log(-t/mw2)**2/ 2.d0-2*ddilog(t/mz
     1   2)+log(-t/mw2)*log(-t/mz2)-2*log((mz2-t)/mz2)*log(-t/mz2)-2*ddi
     2   log(t/mw2)-2*log((mw2-t)/mw2)*log(-t/mw2))/(s*t)
      i4u = (log(-u/mz2)**2/ 2.d0+log(-u/mw2)**2/ 2.d0-2*ddilog(u/mz
     1   2)+log(-u/mw2)*log(-u/mz2)-2*log((mz2-u)/mz2)*log(-u/mz2)-2*ddi
     2   log(u/mw2)-2*log((mw2-u)/mw2)*log(-u/mw2))/(s*u)
      i3tw = (-log(-t/s)**2/ 2.d0+log(mw2/s)**2/ 2.d0-pi2/ 2.d0)/(
     1   mw2-t)
      i3uw = (-log(-u/s)**2/ 2.d0+log(mw2/s)**2/ 2.d0-pi2/ 2.d0)/(
     1   mw2-u)
      i3tz = (-log(-t/s)**2/ 2.d0+log(mz2/s)**2/ 2.d0-pi2/ 2.d0)/(
     1   mz2-t)
      i3uz = (-log(-u/s)**2/ 2.d0+log(mz2/s)**2/ 2.d0-pi2/ 2.d0)/(
     1   mz2-u)
      b = dsqrt(del**2/s**2-2*sig/s+1)
      trgl = (log((del/s+b+1)/ 2.d0)**2/ 2.d0-log((del/s-b+1)/ 2.d0
     1   )**2/ 2.d0+log((-del/s+b+1)/ 2.d0)**2/ 2.d0-log((-del/s-
     2   b+1)/ 2.d0)**2/ 2.d0+ddilog((del/s+b+1)/ 2.d0)-ddilog((de
     3   l/s-b+1)/ 2.d0)+ddilog((-del/s+b+1)/ 2.d0)-ddilog((-del/s-b
     4   +1)/ 2.d0)-ddilog(2*mz2/(sig-del*(del/s+b)))+ddilog(2*mw2/(si
     5   g-del*(del/s+b)))+ddilog(2*mz2/(sig-del*(del/s-b)))-ddilog(2*mw
     6   2/(sig-del*(del/s-b)))+log(mw2/s)*log((del/s-b+1)/(del/s+b+1))+
     7   log(mz2/s)*log((-del/s-b+1)/(-del/s+b+1)))/(b*s)
      xlsmu = log(s/mu2)
      brad = (8*t**2*((t*u/(mw2*mz2)-1)/ 4.d0+(mz2+mw2)*s/(mw2*mz2* 2
     1   .d0))+8*(t*u-mw2*mz2))/t**2
      brand = (8*(mz2+mw2)*s-8*t*u*((t*u/(mw2*mz2)-1)/ 4.d0+(mz2+mw2)*
     1   s/(mw2*mz2* 2.d0)))/(t*u)
      bra0 = 8*s**2*(t*u/(mw2*mz2)-1)*(((mz2+mw2)**2+8*mw2*mz2)/(s**2* 
     1   4.d0)+1/ 4.d0-(mz2+mw2)/(s* 2.d0))+8*(mz2+mw2)*s**2*(s/ 2
     2   .d0+(mw2-mz2)**2/(s* 2.d0)-mz2-mw2)/(mw2*mz2)
      bra11 = -16*s*((-4*mw2*mz2/(s*t)-(mz2+mw2)/s+1)*(t*u/(mw2*mz2)-1)/
     1    4.d0+(mz2+mw2)*(2*mw2*mz2/t+s-mz2-mw2)/(mw2*mz2* 2.d0))
      xvtad = 16*s*(t-mz2)/(t*(u+s)-p)
      xvtad = xvtad-6*(s-mz2+mw2)*(t-u)**2*(u+t)/(b**4*s**3*t)+2*(4*(-2*
     1   s+mz2-mw2)*t**2+4*(3*s*sig-s**2-2*(mz2**2-mw2**2))*t-(s-sig)*(-
     2   4*s*sig+s**2+3*(mz2**2-mw2**2)))/(b**2*s**2*t)+2*(4*t**2+(10*s-
     3   3*mz2-9*mw2)*t+12*p)/t**2
      xvtad = xltw*xvtad
      xvtad = xvtad+trgl*(-3*(t-u)**2*(u+t)**2/(b**4*s**2*t)-2*(2*(2*sig
     1   -3*s)*t**2-2*(2*s-3*sig)*(s-sig)*t-(s-2*sig)*(s-sig)**2)/(b**2*
     2   s*t)+(4*sig*t+4*s*sig-3*s**2-4*(mz2**2+mw2**2))/t)
      xvtad = pi2*(4*(t*u+2*s*sig)/(p* 3.d0)-4*(t-2*u)/(t* 3.d0))+xv
     1   tad+(4*(t*(3*u+s)-3*p)/t**2+6*(t-u)**2*(u+t)/(b**4*s**2*t)+2*(8
     2   *t**2+4*(s-3*sig)*t+4*sig**2-5*s*sig+s**2)/(b**2*s*t))*xlts-8*(
     3   t*u+2*s*sig)/p+8*i3tw*(t-mw2)*(t*u-2*p)/t**2-4*i4t*s*(t*u-2*p)/
     4   t-2*(t-u)**2/(b**2*s*t)+2*(22*t**2+(19*s-18*sig)*t+18*p)/t**2
      xvtand = -8*s*(2*t**2-(3*s+4*mz2+2*mw2)*t+2*mz2*(sig+s))/(u*(p-t*(
     1   u+s)))-8*s**2*t*(t-mz2)*(2*t-sig)/(u*(p-t*(u+s))**2)
      xvtand = xvtand+6*(s-mz2+mw2)*(s-sig)*(t-u)**2/(b**4*s**3*u)+2*(4*
     1   (-2*s+mz2-mw2)*t**2+4*(s**2+(mz2+5*mw2)*s-2*(mz2**2-mw2**2))*t+
     2   (3*s**2+8*mw2*s-3*(mz2**2-mw2**2))*(s-sig))/(b**2*s**2*u)+2*((1
     3   8*s+mz2+3*mw2)*t-12*s*sig)/(t*u)
      xvtand = xltw*xvtand
      xvtand = pi2*(4*(u+4*s)/(u* 3.d0)-4*(t*u+2*s*sig)/(p* 3.d0))+x
     1   vtand+(6*(t-u)**2*(u+t)/(b**4*s**2*u)+2*(8*t**2-4*(3*sig+s)*t-(
     2   s-sig)*(4*sig+3*s))/(b**2*s*u)-12*s*(t-sig)/(t*u))*xlts
      xvtand = xvtand+4*s**2*(2*t-sig)/(u*(p-t*(u+s)))+8*(t*u+2*s*sig)/p
     1   +trgl*(-3*(s-sig)**2*(t-u)**2/(b**4*s**2*u)-2*(2*(2*sig-3*s)*t*
     2   *2+6*(s-sig)*sig*t+(s-sig)**2*(2*sig+s))/(b**2*s*u)+3*s*(4*t-4*
     3   sig-s)/u)-2*(t-u)**2/(b**2*s*u)+2*(4*t**2+(9*s-4*sig)*t-18*s*si
     4   g)/(t*u)-16*i3tw*s*(t-mw2)*(t-sig)/(t*u)+8*i4t*s**2*(t-sig)/u
      xvta11 = pi2*(-8*(s-sig)*(t*u+2*s*sig)/(p* 3.d0)-8*(3*t**2+2*(2*
     1   s-sig)*t+2*(s*sig+p))/(t* 3.d0))+(8*s*((2*sig+3*s)*t-2*mz2*(s
     2   ig+s))/(t*(u+s)-p)+8*s**2*t*(t-mz2)*(2*t-sig)/(t*(u+s)-p)**2-8*
     3   (s-mz2+mw2)*(t-u)/(b**2*s)+8*(t**2-(2*s+mz2+3*mw2)*t+3*(s*sig+p
     4   ))/t)*xltw+(8*(t-u)/b**2-4*(3*t**2-(3*sig+s)*t+3*(s*sig+p))/t)*
     5   xlts
      xvta11 = xvta11+4*s**2*(2*t-sig)/(t*(u+s)-p)+16*(s-sig)*(t*u+2*s*s
     1   ig)/p+trgl*(4*(s-sig)*(t-u)/b**2+4*(sig-3*s)*t+4*(4*s*sig-mz2**
     2   2-mw2**2))+4*i4t*(s*t**2-s*(sig+s)*t+2*s*(s*sig+p))+4*(17*t**2+
     3   (11*s-13*sig)*t+17*(s*sig+p))/t-8*i3tw*(t-mw2)*(t**2-(sig+s)*t+
     4   2*(s*sig+p))/t
      brcd = (8*u**2*((t*u/(mw2*mz2)-1)/ 4.d0+(mz2+mw2)*s/(mw2*mz2* 2
     1   .d0))+8*(t*u-mw2*mz2))/u**2
      brcnd = (8*(mz2+mw2)*s-8*t*u*((t*u/(mw2*mz2)-1)/ 4.d0+(mz2+mw2)*
     1   s/(mw2*mz2* 2.d0)))/(t*u)
      brc11 = -16*s*((-4*mw2*mz2/(s*u)-(mz2+mw2)/s+1)*(t*u/(mw2*mz2)-1)/
     1    4.d0+(mz2+mw2)*(2*mw2*mz2/u+s-mz2-mw2)/(mw2*mz2* 2.d0))
      tmp0 = 16*s*(t-mw2)/(t*(u+s)-p)
      tmp0 = -6*(s+mz2-mw2)*(t-u)**2*(u+t)/(b**4*s**3*t)+tmp0+2*(4*(-2*s
     1   -mz2+mw2)*t**2+4*(3*s*sig-s**2-2*(mw2**2-mz2**2))*t-(s-sig)*(-4
     2   *s*sig+s**2+3*(mw2**2-mz2**2)))/(b**2*s**2*t)+2*(4*t**2+(10*s-9
     3   *mz2-3*mw2)*t+12*p)/t**2
      tmp0 = tmp0*xltz
      tmp0 = trgl*(-3*(t-u)**2*(u+t)**2/(b**4*s**2*t)-2*(2*(2*sig-3*s)*t
     1   **2-2*(2*s-3*sig)*(s-sig)*t-(s-2*sig)*(s-sig)**2)/(b**2*s*t)+(4
     2   *sig*t+4*s*sig-3*s**2-4*(mz2**2+mw2**2))/t)+tmp0
      tmp0 = pi2*(4*(t*u+2*s*sig)/(p* 3.d0)-4*(t-2*u)/(t* 3.d0))+xvt
     1   ad+(4*(t*(3*u+s)-3*p)/t**2+6*(t-u)**2*(u+t)/(b**4*s**2*t)+2*(8*
     2   t**2+4*(s-3*sig)*t+4*sig**2-5*s*sig+s**2)/(b**2*s*t))*xlts-8*(t
     3   *u+2*s*sig)/p+8*i3tz*(t-mz2)*(t*u-2*p)/t**2-4*i4t*s*(t*u-2*p)/t
     4   -2*(t-u)**2/(b**2*s*t)+tmp0+2*(22*t**2+(19*s-18*sig)*t+18*p)/t*
     5   *2
      xvtad = 2*tmp0
      xvtcd = 2*(4*(-2*s+mz2-mw2)*u**2+4*(3*s*sig-s**2-2*(mz2**2-mw2**2)
     1   )*u-(s-sig)*(-4*s*sig+s**2+3*(mz2**2-mw2**2)))/(b**2*s**2*u)
      xvtcd = xvtcd+2*(4*u**2+(10*s-3*mz2-9*mw2)*u+12*p)/u**2+16*s*(u-mz
     1   2)/((t+s)*u-p)-6*(s-mz2+mw2)*(u-t)**2*(u+t)/(b**4*s**3*u)
      xvtcd = xluw*xvtcd
      xvtcd = xvtcd+trgl*(-2*(2*(2*sig-3*s)*u**2-2*(2*s-3*sig)*(s-sig)*u
     1   -(s-2*sig)*(s-sig)**2)/(b**2*s*u)+(4*sig*u+4*s*sig-3*s**2-4*(mz
     2   2**2+mw2**2))/u-3*(u-t)**2*(u+t)**2/(b**4*s**2*u))
      xvtcd = pi2*(4*(t*u+2*s*sig)/(p* 3.d0)-4*(u-2*t)/(u* 3.d0))+xv
     1   tcd+(2*(8*u**2+4*(s-3*sig)*u+4*sig**2-5*s*sig+s**2)/(b**2*s*u)+
     2   4*((3*t+s)*u-3*p)/u**2+6*(u-t)**2*(u+t)/(b**4*s**2*u))*xlus+2*(
     3   22*u**2+(19*s-18*sig)*u+18*p)/u**2-8*(t*u+2*s*sig)/p+8*i3uw*(u-
     4   mw2)*(t*u-2*p)/u**2-4*i4u*s*(t*u-2*p)/u-2*(u-t)**2/(b**2*s*u)
      tmp0 = 2*(4*(-2*s-mz2+mw2)*u**2+4*(3*s*sig-s**2-2*(mw2**2-mz2**2))
     1   *u-(s-sig)*(-4*s*sig+s**2+3*(mw2**2-mz2**2)))/(b**2*s**2*u)
      tmp0 = 2*(4*u**2+(10*s-9*mz2-3*mw2)*u+12*p)/u**2+16*s*(u-mw2)/((t+
     1   s)*u-p)-6*(s+mz2-mw2)*(u-t)**2*(u+t)/(b**4*s**3*u)+tmp0
      tmp0 = tmp0*xluz
      tmp0 = trgl*(-2*(2*(2*sig-3*s)*u**2-2*(2*s-3*sig)*(s-sig)*u-(s-2*s
     1   ig)*(s-sig)**2)/(b**2*s*u)+(4*sig*u+4*s*sig-3*s**2-4*(mz2**2+mw
     2   2**2))/u-3*(u-t)**2*(u+t)**2/(b**4*s**2*u))+tmp0
      tmp0 = pi2*(4*(t*u+2*s*sig)/(p* 3.d0)-4*(u-2*t)/(u* 3.d0))+xvt
     1   cd+(2*(8*u**2+4*(s-3*sig)*u+4*sig**2-5*s*sig+s**2)/(b**2*s*u)+4
     2   *((3*t+s)*u-3*p)/u**2+6*(u-t)**2*(u+t)/(b**4*s**2*u))*xlus+2*(2
     3   2*u**2+(19*s-18*sig)*u+18*p)/u**2-8*(t*u+2*s*sig)/p+8*i3uz*(u-m
     4   z2)*(t*u-2*p)/u**2-4*i4u*s*(t*u-2*p)/u-2*(u-t)**2/(b**2*s*u)+tm
     5   p0
      xvtcd = 2*tmp0
      tmp0 = -8*s*(2*t**2-(3*s+2*mz2+4*mw2)*t+2*mw2*(sig+s))/(u*(p-t*(u+
     1   s)))-8*s**2*t*(t-mw2)*(2*t-sig)/(u*(p-t*(u+s))**2)
      tmp0 = 6*(s+mz2-mw2)*(s-sig)*(t-u)**2/(b**4*s**3*u)+2*(4*(-2*s-mz2
     1   +mw2)*t**2+4*(s**2+(5*mz2+mw2)*s-2*(mw2**2-mz2**2))*t+(3*s**2+8
     2   *mz2*s-3*(mw2**2-mz2**2))*(s-sig))/(b**2*s**2*u)+2*((18*s+3*mz2
     3   +mw2)*t-12*s*sig)/(t*u)+tmp0
      tmp0 = tmp0*xltz
      tmp0 = pi2*(4*(u+4*s)/(u* 3.d0)-4*(t*u+2*s*sig)/(p* 3.d0))+xvt
     1   and+(6*(t-u)**2*(u+t)/(b**4*s**2*u)+2*(8*t**2-4*(3*sig+s)*t-(s-
     2   sig)*(4*sig+3*s))/(b**2*s*u)-12*s*(t-sig)/(t*u))*xlts+tmp0
      tmp0 = 4*s**2*(2*t-sig)/(u*(p-t*(u+s)))+8*(t*u+2*s*sig)/p+trgl*(-3
     1   *(s-sig)**2*(t-u)**2/(b**4*s**2*u)-2*(2*(2*sig-3*s)*t**2+6*(s-s
     2   ig)*sig*t+(s-sig)**2*(2*sig+s))/(b**2*s*u)+3*s*(4*t-4*sig-s)/u)
     3   -2*(t-u)**2/(b**2*s*u)+2*(4*t**2+(9*s-4*sig)*t-18*s*sig)/(t*u)-
     4   16*i3tz*s*(t-mz2)*(t-sig)/(t*u)+8*i4t*s**2*(t-sig)/u+tmp0
      xvtand = 2*tmp0
      xvtcnd = 2*(4*(-2*s+mz2-mw2)*u**2+4*(s**2+(mz2+5*mw2)*s-2*(mz2**2-
     1   mw2**2))*u+(3*s**2+8*mw2*s-3*(mz2**2-mw2**2))*(s-sig))/(b**2*s*
     2   *2*t)
      xvtcnd = xvtcnd-8*s*(2*u**2-(3*s+4*mz2+2*mw2)*u+2*mz2*(sig+s))/(t*
     1   (p-(t+s)*u))-8*s**2*u*(u-mz2)*(2*u-sig)/(t*(p-(t+s)*u)**2)+2*((
     2   18*s+mz2+3*mw2)*u-12*s*sig)/(t*u)+6*(s-mz2+mw2)*(s-sig)*(u-t)**
     3   2/(b**4*s**3*t)
      xvtcnd = xluw*xvtcnd
      xvtcnd = xvtcnd+trgl*(-2*(2*(2*sig-3*s)*u**2+6*(s-sig)*sig*u+(s-si
     1   g)**2*(2*sig+s))/(b**2*s*t)+3*s*(4*u-4*sig-s)/t-3*(s-sig)**2*(u
     2   -t)**2/(b**4*s**2*t))
      xvtcnd = pi2*(4*(t+4*s)/(t* 3.d0)-4*(t*u+2*s*sig)/(p* 3.d0))+x
     1   vtcnd+(2*(8*u**2-4*(3*sig+s)*u-(s-sig)*(4*sig+3*s))/(b**2*s*t)+
     2   6*(u-t)**2*(u+t)/(b**4*s**2*t)-12*s*(u-sig)/(t*u))*xlus+2*(4*u*
     3   *2+(9*s-4*sig)*u-18*s*sig)/(t*u)+4*s**2*(2*u-sig)/(t*(p-(t+s)*u
     4   ))+8*(t*u+2*s*sig)/p-2*(u-t)**2/(b**2*s*t)-16*i3uw*s*(u-mw2)*(u
     5   -sig)/(t*u)+8*i4u*s**2*(u-sig)/t
      tmp0 = 2*(4*(-2*s-mz2+mw2)*u**2+4*(s**2+(5*mz2+mw2)*s-2*(mw2**2-mz
     1   2**2))*u+(3*s**2+8*mz2*s-3*(mw2**2-mz2**2))*(s-sig))/(b**2*s**2
     2   *t)
      tmp0 = -8*s*(2*u**2-(3*s+2*mz2+4*mw2)*u+2*mw2*(sig+s))/(t*(p-(t+s)
     1   *u))-8*s**2*u*(u-mw2)*(2*u-sig)/(t*(p-(t+s)*u)**2)+2*((18*s+3*m
     2   z2+mw2)*u-12*s*sig)/(t*u)+6*(s+mz2-mw2)*(s-sig)*(u-t)**2/(b**4*
     3   s**3*t)+tmp0
      tmp0 = tmp0*xluz
      tmp0 = trgl*(-2*(2*(2*sig-3*s)*u**2+6*(s-sig)*sig*u+(s-sig)**2*(2*
     1   sig+s))/(b**2*s*t)+3*s*(4*u-4*sig-s)/t-3*(s-sig)**2*(u-t)**2/(b
     2   **4*s**2*t))+tmp0
      tmp0 = pi2*(4*(t+4*s)/(t* 3.d0)-4*(t*u+2*s*sig)/(p* 3.d0))+xvt
     1   cnd+(2*(8*u**2-4*(3*sig+s)*u-(s-sig)*(4*sig+3*s))/(b**2*s*t)+6*
     2   (u-t)**2*(u+t)/(b**4*s**2*t)-12*s*(u-sig)/(t*u))*xlus+2*(4*u**2
     3   +(9*s-4*sig)*u-18*s*sig)/(t*u)+4*s**2*(2*u-sig)/(t*(p-(t+s)*u))
     4   +8*(t*u+2*s*sig)/p-2*(u-t)**2/(b**2*s*t)-16*i3uz*s*(u-mz2)*(u-s
     5   ig)/(t*u)+8*i4u*s**2*(u-sig)/t+tmp0
      xvtcnd = 2*tmp0
      xvta0 = 2*bra0*(4*pi2-24)/ 3.d0
      tmp0 = pi2*(-8*(s-sig)*(t*u+2*s*sig)/(p* 3.d0)-8*(3*t**2+2*(2*s-
     1   sig)*t+2*(s*sig+p))/(t* 3.d0))+(8*s*((2*sig+3*s)*t-2*mw2*(sig
     2   +s))/(t*(u+s)-p)+8*s**2*t*(t-mw2)*(2*t-sig)/(t*(u+s)-p)**2-8*(s
     3   +mz2-mw2)*(t-u)/(b**2*s)+8*(t**2-(2*s+3*mz2+mw2)*t+3*(s*sig+p))
     4   /t)*xltz+(8*(t-u)/b**2-4*(3*t**2-(3*sig+s)*t+3*(s*sig+p))/t)*xl
     5   ts
      tmp0 = xvta11+4*s**2*(2*t-sig)/(t*(u+s)-p)+16*(s-sig)*(t*u+2*s*sig
     1   )/p+trgl*(4*(s-sig)*(t-u)/b**2+4*(sig-3*s)*t+4*(4*s*sig-mz2**2-
     2   mw2**2))+tmp0+4*i4t*(s*t**2-s*(sig+s)*t+2*s*(s*sig+p))+4*(17*t*
     3   *2+(11*s-13*sig)*t+17*(s*sig+p))/t-8*i3tz*(t-mz2)*(t**2-(sig+s)
     4   *t+2*(s*sig+p))/t
      xvta11 = 2*tmp0
      xvtc11 = pi2*(-8*(3*u**2+2*(2*s-sig)*u+2*(s*sig+p))/(u* 3.d0)-8*
     1   (s-sig)*(t*u+2*s*sig)/(p* 3.d0))+(8*(u**2-(2*s+mz2+3*mw2)*u+3
     2   *(s*sig+p))/u+8*s*((2*sig+3*s)*u-2*mz2*(sig+s))/((t+s)*u-p)+8*s
     3   **2*u*(u-mz2)*(2*u-sig)/((t+s)*u-p)**2-8*(s-mz2+mw2)*(u-t)/(b**
     4   2*s))*xluw+(8*(u-t)/b**2-4*(3*u**2-(3*sig+s)*u+3*(s*sig+p))/u)*
     5   xlus
      xvtc11 = xvtc11+4*i4u*(s*u**2-s*(sig+s)*u+2*s*(s*sig+p))+4*(17*u**
     1   2+(11*s-13*sig)*u+17*(s*sig+p))/u-8*i3uw*(u-mw2)*(u**2-(sig+s)*
     2   u+2*(s*sig+p))/u+trgl*(4*(s-sig)*(u-t)/b**2+4*(sig-3*s)*u+4*(4*
     3   s*sig-mz2**2-mw2**2))+4*s**2*(2*u-sig)/((t+s)*u-p)+16*(s-sig)*(
     4   t*u+2*s*sig)/p
      tmp0 = pi2*(-8*(3*u**2+2*(2*s-sig)*u+2*(s*sig+p))/(u* 3.d0)-8*(s
     1   -sig)*(t*u+2*s*sig)/(p* 3.d0))+(8*(u**2-(2*s+3*mz2+mw2)*u+3*(
     2   s*sig+p))/u+8*s*((2*sig+3*s)*u-2*mw2*(sig+s))/((t+s)*u-p)+8*s**
     3   2*u*(u-mw2)*(2*u-sig)/((t+s)*u-p)**2-8*(s+mz2-mw2)*(u-t)/(b**2*
     4   s))*xluz+(8*(u-t)/b**2-4*(3*u**2-(3*sig+s)*u+3*(s*sig+p))/u)*xl
     5   us
      tmp0 = xvtc11+4*i4u*(s*u**2-s*(sig+s)*u+2*s*(s*sig+p))+4*(17*u**2+
     1   (11*s-13*sig)*u+17*(s*sig+p))/u-8*i3uz*(u-mz2)*(u**2-(sig+s)*u+
     2   2*(s*sig+p))/u+trgl*(4*(s-sig)*(u-t)/b**2+4*(sig-3*s)*u+4*(4*s*
     3   sig-mz2**2-mw2**2))+4*s**2*(2*u-sig)/((t+s)*u-p)+16*(s-sig)*(t*
     4   u+2*s*sig)/p+tmp0
      xvtc11 = 2*tmp0
      sfc = -4*pi2/ 3.d0+(16*xlbeta+6)*xlsmu+32*xlbeta**2
      xvsad = brad*sfc
      xvsand = brand*sfc
      xvsa0 = bra0*sfc
      xvsa11 = bra11*sfc
      xvscd = brcd*sfc
      xvscnd = brcnd*sfc
      xvsc11 = brc11*sfc
      ad2b = xvtad+xvsad
      and2b = xvtand+xvsand
      a112b = xvta11+xvsa11
      a02b = xvta0+xvsa0
      cd2b = xvtcd+xvscd
      cnd2b = xvtcnd+xvscnd
      c112b = xvtc11+xvsc11
      wz2b = -ez*(a112b*gdown-c112b*gup)/(s-mw2)+a02b*ez**2/(s-mw2)**2+c
     1   d2b*gup**2+(cnd2b+and2b)*gdown*gup+ad2b*gdown**2
      wz2b = wz2b/(nc*s* 4.d0)
      if(scheme.eq.'DI') then
      dis=-(4.5d0+pi2/3)+(-1.5d0)*2*xlbeta+2*2*xlbeta**2
      dis = -4*wzborn(t,s,mw2,mz2)*dis
      wz2b = wz2b + dis
      elseif(scheme.eq.'MS') then
      else
      call noscheme
      endif
      end
c
c----------------------------------------------------------------
c d sigma(r) = C_F alfas/(4 pi) f(i,j)^2 gw^2 wz3b(tt,uu,q1q,q2q,s,mw2,mz2)
c              P{1/(1-x)}  P{1/(1-y)+1/(1+y)}
c              beta_x/(8 pi)^2/s d cos(th1) d th2 dy dx
c Fattori definiti come sopra
c----------------------------------------------------------------
c
      function wz3b(xx,yy,tt,uu,q1q,q2q,s,mw2,mz2)
      implicit real * 8 (a-z)
      common/fixvar/sh,xmufct2,xmuren2,as,xnc
      common/weakcoup/xkm(3,3),xkm2(3,3),sw,ze2,gup,gdown,ez,gw
c
      tiny = .1d-5
      x=xx
      y=yy
      mu2=xmufct2
      nc=xnc
      if(1-x.le.tiny)then
         tmpp = 8*s*(1+x**2)*wzborn(q2q,x*s,mw2,mz2)
         tmpm = 8*s*(1+x**2)*wzborn(q1q,x*s,mw2,mz2)
         wz3b = (tmpp+tmpm)/2.d0
      elseif(1-y.le.tiny)then
         wz3b = 8*s*(1+x**2)*wzborn(q2q,x*s,mw2,mz2)
      elseif(1+y.le.tiny)then
         wz3b = 8*s*(1+x**2)*wzborn(q1q,x*s,mw2,mz2)
      else
      sig = mz2+mw2
      s2 = uu+tt+s
      q1c = -tt-s-q1q+mz2+mw2
      q2c = -uu-s-q2q+mz2+mw2
      rdd = 4*(q2q*(uu+2*s+q2q)+q1q*(s+q1q))/(mw2*mz2*tt)
      rdd = -4*(2*uu+5*s+q2q+2*q1q-mw2)/(mz2*tt)-4*(2*uu+4*s+q2q)/(mw2*t
     1   t)+16*(uu+s)/(q2q*tt)-4*s*(4*s+q2q+q1q)/(mw2*tt*uu)+16*s*(s+q2q
     2   -mz2-mw2)/(q1q*tt*uu)+4*q1q*s*(s+q1q)/(mw2*mz2*tt*uu)+16*mw2*mz
     3   2*s/(q1q*q2q*tt*uu)+4*s/(tt*uu)+16*mz2*(tt-2*mw2)/(q1q*q2q**2)+
     4   16*(-tt+2*mz2+mw2)/(q1q*q2q)+16*mw2*(s-mz2-mw2)/(q1q*q2q*tt)+16
     5   *mz2*(q1q-2*mw2)/(q2q**2*tt)+32*mw2**2*mz2/(q1q*q2q**2*tt)+16*m
     6   w2/(q1q*tt)+4/tt+rdd+8/q2q+4*q1q/(mw2*mz2)-24/q1q-4/mw2
      tmp0 = 16*mw2*(uu-2*mz2)/(q1q**2*q2q)+16*(-uu+mz2+2*mw2)/(q1q*q2q)
      rdd = 4*(q1q*(tt+2*s+q1q)+q2q*(s+q2q))/(mw2*mz2*uu)-4*(2*tt+5*s+2*
     1   q2q+q1q-mz2)/(mw2*uu)-4*(2*tt+4*s+q1q)/(mz2*uu)+16*(tt+s)/(q1q*
     2   uu)-4*s*(4*s+q2q+q1q)/(mz2*tt*uu)+4*q2q*s*(s+q2q)/(mw2*mz2*tt*u
     3   u)+16*s*(s+q1q-mz2-mw2)/(q2q*tt*uu)+16*mw2*mz2*s/(q1q*q2q*tt*uu
     4   )+4*s/(tt*uu)+16*mz2*(s-mz2-mw2)/(q1q*q2q*uu)+16*mw2*(q2q-2*mz2
     5   )/(q1q**2*uu)+32*mw2*mz2**2/(q1q**2*q2q*uu)+16*mz2/(q2q*uu)+4/u
     6   u+tmp0+rdd+4*q2q/(mw2*mz2)-24/q2q+8/q1q-4/mz2
      ruu = 4*(q2c*(uu+2*s+q2c)+q1c*(s+q1c))/(mw2*mz2*tt)
      ruu = -4*(2*uu+5*s+q2c+2*q1c-mz2)/(mw2*tt)-4*(2*uu+4*s+q2c)/(mz2*t
     1   t)+16*(uu+s)/(q2c*tt)-4*s*(4*s+q2c+q1c)/(mz2*tt*uu)+16*s*(s+q2c
     2   -mz2-mw2)/(q1c*tt*uu)+4*q1c*s*(s+q1c)/(mw2*mz2*tt*uu)+16*mw2*mz
     3   2*s/(q1c*q2c*tt*uu)+4*s/(tt*uu)+16*mw2*(tt-2*mz2)/(q1c*q2c**2)+
     4   16*(-tt+mz2+2*mw2)/(q1c*q2c)+16*mz2*(s-mz2-mw2)/(q1c*q2c*tt)+16
     5   *mw2*(q1c-2*mz2)/(q2c**2*tt)+32*mw2*mz2**2/(q1c*q2c**2*tt)+16*m
     6   z2/(q1c*tt)+4/tt+ruu+8/q2c+4*q1c/(mw2*mz2)-24/q1c-4/mz2
      tmp0 = 16*mz2*(uu-2*mw2)/(q1c**2*q2c)+16*(-uu+2*mz2+mw2)/(q1c*q2c)
      ruu = 4*(q1c*(tt+2*s+q1c)+q2c*(s+q2c))/(mw2*mz2*uu)-4*(2*tt+5*s+2*
     1   q2c+q1c-mw2)/(mz2*uu)-4*(2*tt+4*s+q1c)/(mw2*uu)+16*(tt+s)/(q1c*
     2   uu)-4*s*(4*s+q2c+q1c)/(mw2*tt*uu)+4*q2c*s*(s+q2c)/(mw2*mz2*tt*u
     3   u)+16*s*(s+q1c-mz2-mw2)/(q2c*tt*uu)+16*mw2*mz2*s/(q1c*q2c*tt*uu
     4   )+4*s/(tt*uu)+16*mw2*(s-mz2-mw2)/(q1c*q2c*uu)+16*mz2*(q2c-2*mw2
     5   )/(q1c**2*uu)+32*mw2**2*mz2/(q1c**2*q2c*uu)+16*mw2/(q2c*uu)+4/u
     6   u+tmp0+ruu+4*q2c/(mw2*mz2)-24/q2c+8/q1c-4/mw2
      rud = ((s+q1c)*(2*uu+s)+2*q2q*(s+q2c)+(q2q+q2c)*s-q1q*s+q1q*q2q+q1
     1   c*q2c)/(mw2*mz2*tt)-4*s*(uu*(uu+3*s+q1c-mw2)-(q2q-s)*(s+q2q)+(q
     2   2q-mw2)*s+q1c*(q2q-mw2)+mw2*q2q)/(mz2*q1q*q2c*tt)
      rud = 8*(s*uu*(uu+3*s+q1c)-(2*mz2*s+3*mw2*s+mw2*q1c)*(uu+s)+s**2*(
     1   s+mz2))/(q1q*q2c*q2q*tt)-4*((s+q1c-mz2)*(uu+s+q2c-2*mz2)-mz2*q1
     2   q)/(mw2*q2q*tt)+4*(2*s*uu+2*mw2*uu+5*s**2+2*q1c*s-2*mz2*s)/(q1q
     3   *q2c*tt)+4*(2*s*uu-s**2-2*q1c*s+2*mw2*s+2*mw2*q1c)/(q1q*q2c*q2q
     4   )+8*(mz2*uu-s**2+mw2*s-2*mz2*q1q-2*mz2*q1c)/(q2c*q2q*tt)+rud
      rud = 8*s*(uu-q1c+mz2)/(q1q*q2q*tt)+4*s*(-uu+s-q2q+q1q+q1c)/(mz2*q
     1   2c*tt)+2*(-uu-9*s-4*q2q-5*q2c-3*q1q-4*q1c+8*mz2)/(mw2*tt)+4*s*(
     2   -uu-q2q+q1c)/(mz2*q1q*tt)+2*(-4*uu+3*s+5*q1q+4*q1c)/(q2c*tt)+2*
     3   (s*tt+q2q*tt+s**2-q2q**2+q1c*q2q)/(mw2*mz2*uu)-8*s*(tt+s+q1c)/(
     4   mw2*q2q*uu)+2*(-tt+3*s+q2q-q1c)/(mw2*uu)-8*s**3/(q1c*q2q*tt*uu)
     5   +2*s*(2*s+q2q)/(mz2*tt*uu)+rud
      rud = -16*s**2/(q1c*tt*uu)+2*s*(2*s+q2q)/(mw2*tt*uu)-2*q2q*s*(s+q2
     1   q)/(mw2*mz2*tt*uu)-2*s/(tt*uu)-16*s**2/(q1c*q2q*uu)-8*s/(q2q*uu
     2   )-16*s/(q1c*uu)+6*s/(mz2*uu)+4*s*(tt+q1c-mw2)/(mz2*q1q*q2c)+2*(
     3   s+4*q1q+5*q1c-4*mz2)/(q2q*tt)+4*s/(q1q*tt)+4*s/(mz2*tt)+12/tt-4
     4   *s**3/(q1c*q1q*q2c*q2q)-4*s**2/(q1c*q2c*q2q)-4*s**2/(q1c*q1q*q2
     5   q)+8*s**2/(mw2*q1c*q2q)-4*s**2/(q1c*q1q*q2c)-4*(3*s+2*q1c)/(q1q
     6   *q2q)-(s+4*q2q)/(mw2*mz2)-4*(s+2*mz2)/(q2c*q2q)+4*(s+mz2)/(mw2*
     7   q2q)-4*s/(q1c*q2c)-4*s/(q1c*q1q)+12*s/(mw2*q1c)+rud-8*mw2/(q1q*
     8   q2c)+8/q2c+8/q1q
      tmp0 = 2*(s*uu+q1q*uu+s**2+q1q*q2c-q1q**2)/(mw2*mz2*tt)-8*s*(uu+s+
     1   q2c)/(mz2*q1q*tt)+4*s*(uu+q2c-mz2)/(mw2*q1c*q2q)+2*(-uu+3*s-q2c
     2   +q1q)/(mz2*tt)+((s+q2c)*(2*tt+s)+2*q1q*(s+q1c)-q2q*s+(q1q+q1c)*
     3   s+q1q*q2q+q1c*q2c)/(mw2*mz2*uu)
      tmp0 = -4*s*(tt*(tt+3*s+q2c-mz2)-(q1q-s)*(s+q1q)+(q1q-mz2)*s+(q1q-
     1   mz2)*q2c+mz2*q1q)/(mw2*q1c*q2q*uu)+8*(s*tt*(tt+3*s+q2c)-(3*mz2*
     2   s+2*mw2*s+mz2*q2c)*(tt+s)+s**2*(s+mw2))/(q1c*q1q*q2q*uu)-4*((s+
     3   q2c-mw2)*(tt+s+q1c-2*mw2)-mw2*q2q)/(mz2*q1q*uu)+tmp0
      tmp0 = 4*(2*s*tt+2*mz2*tt+5*s**2+2*q2c*s-2*mw2*s)/(q1c*q2q*uu)+8*(
     1   mw2*tt-s**2+mz2*s-2*mw2*q2q-2*mw2*q2c)/(q1c*q1q*uu)+8*s*(tt-q2c
     2   +mw2)/(q1q*q2q*uu)+4*s*(-tt+s+q2q+q2c-q1q)/(mw2*q1c*uu)+2*(-tt-
     3   9*s-3*q2q-4*q2c-4*q1q-5*q1c+8*mw2)/(mz2*uu)+4*s*(-tt+q2c-q1q)/(
     4   mw2*q2q*uu)+2*(-4*tt+3*s+5*q2q+4*q2c)/(q1c*uu)-8*s**3/(q1q*q2c*
     5   tt*uu)-16*s**2/(q2c*tt*uu)+2*s*(2*s+q1q)/(mz2*tt*uu)+2*s*(2*s+q
     6   1q)/(mw2*tt*uu)+tmp0
      rud = -2*q1q*s*(s+q1q)/(mw2*mz2*tt*uu)-2*s/(tt*uu)+2*(s+4*q2q+5*q2
     1   c-4*mw2)/(q1q*uu)+4*s/(q2q*uu)+4*s/(mw2*uu)+12/uu+4*(2*s*tt-s**
     2   2-2*q2c*s+2*mz2*s+2*mz2*q2c)/(q1c*q1q*q2q)-16*s**2/(q1q*q2c*tt)
     3   -16*s/(q2c*tt)-8*s/(q1q*tt)+6*s/(mw2*tt)+tmp0-4*s**3/(q1c*q1q*q
     4   2c*q2q)-4*s**2/(q1q*q2c*q2q)-4*s**2/(q1c*q2c*q2q)-4*s**2/(q1c*q
     5   1q*q2c)+8*s**2/(mz2*q1q*q2c)-4*(3*s+2*q2c)/(q1q*q2q)-(s+4*q1q)/
     6   (mw2*mz2)-4*(s+2*mw2)/(q1c*q1q)+4*(s+mw2)/(mz2*q1q)-4*s/(q2c*q2
     7   q)-4*s/(q1c*q2c)+12*s/(mz2*q2c)+rud-8*mz2/(q1c*q2q)+8/q2q+8/q1c
      tmp0 = 2*(s*uu+q1c*uu+s**2+q1c*q2q-q1c**2)/(mw2*mz2*tt)-8*s*(uu+s+
     1   q2q)/(mw2*q1c*tt)+4*s*(uu+q2q-mw2)/(mz2*q1q*q2c)+2*(-uu+3*s-q2q
     2   +q1c)/(mw2*tt)+((s+q2q)*(2*tt+s)+2*q1c*(s+q1q)-q2c*s+(q1q+q1c)*
     3   s+q1q*q2q+q1c*q2c)/(mw2*mz2*uu)
      tmp0 = -4*s*(tt*(tt+3*s+q2q-mw2)-(q1c-s)*(s+q1c)+(q1c-mw2)*s+(q1c-
     1   mw2)*q2q+mw2*q1c)/(mz2*q1q*q2c*uu)+8*(s*tt*(tt+3*s+q2q)-(2*mz2*
     2   s+3*mw2*s+mw2*q2q)*(tt+s)+s**2*(s+mz2))/(q1c*q1q*q2c*uu)-4*((s+
     3   q2q-mz2)*(tt+s+q1q-2*mz2)-mz2*q2c)/(mw2*q1c*uu)+tmp0
      tmp0 = 4*(2*s*tt+2*mw2*tt+5*s**2+2*q2q*s-2*mz2*s)/(q1q*q2c*uu)+8*(
     1   mz2*tt-s**2+mw2*s-2*mz2*q2q-2*mz2*q2c)/(q1c*q1q*uu)+8*s*(tt-q2q
     2   +mz2)/(q1c*q2c*uu)+4*s*(-tt+s+q2q+q2c-q1c)/(mz2*q1q*uu)+2*(-tt-
     3   9*s-4*q2q-3*q2c-5*q1q-4*q1c+8*mz2)/(mw2*uu)+4*s*(-tt+q2q-q1c)/(
     4   mz2*q2c*uu)+2*(-4*tt+3*s+4*q2q+5*q2c)/(q1q*uu)-8*s**3/(q1c*q2q*
     5   tt*uu)-16*s**2/(q2q*tt*uu)+2*s*(2*s+q1c)/(mz2*tt*uu)+2*s*(2*s+q
     6   1c)/(mw2*tt*uu)+tmp0
      rud = -2*q1c*s*(s+q1c)/(mw2*mz2*tt*uu)-2*s/(tt*uu)+2*(s+5*q2q+4*q2
     1   c-4*mz2)/(q1c*uu)+4*s/(q2c*uu)+4*s/(mz2*uu)+12/uu+4*(2*s*tt-s**
     2   2-2*q2q*s+2*mw2*s+2*mw2*q2q)/(q1c*q1q*q2c)-16*s**2/(q1c*q2q*tt)
     3   -16*s/(q2q*tt)-8*s/(q1c*tt)+6*s/(mz2*tt)+tmp0-4*s**3/(q1c*q1q*q
     4   2c*q2q)-4*s**2/(q1q*q2c*q2q)-4*s**2/(q1c*q2c*q2q)-4*s**2/(q1c*q
     5   1q*q2q)+8*s**2/(mw2*q1c*q2q)-4*(3*s+2*q2q)/(q1c*q2c)-(s+4*q1c)/
     6   (mw2*mz2)-4*(s+2*mz2)/(q1c*q1q)+4*(s+mz2)/(mw2*q1c)-4*s/(q2c*q2
     7   q)-4*s/(q1q*q2q)+12*s/(mw2*q2q)+rud-8*mw2/(q1q*q2c)+8/q2c+8/q1q
      tmp0 = ((s+q1q)*(2*uu+s)+2*q2c*(s+q2q)+(q2q+q2c)*s-q1c*s+q1q*q2q+q
     1   1c*q2c)/(mw2*mz2*tt)-4*s*(uu*(uu+3*s+q1q-mz2)-(q2c-s)*(s+q2c)+(
     2   q2c-mz2)*s+q1q*(q2c-mz2)+mz2*q2c)/(mw2*q1c*q2q*tt)
      tmp0 = 8*(s*uu*(uu+3*s+q1q)-(3*mz2*s+2*mw2*s+mz2*q1q)*(uu+s)+s**2*
     1   (s+mw2))/(q1c*q2c*q2q*tt)-4*((s+q1q-mw2)*(uu+s+q2q-2*mw2)-mw2*q
     2   1c)/(mz2*q2c*tt)+4*(2*s*uu+2*mz2*uu+5*s**2+2*q1q*s-2*mw2*s)/(q1
     3   c*q2q*tt)+4*(2*s*uu-s**2-2*q1q*s+2*mz2*s+2*mz2*q1q)/(q1c*q2c*q2
     4   q)+8*(mw2*uu-s**2+mz2*s-2*mw2*q1q-2*mw2*q1c)/(q2c*q2q*tt)+tmp0
      tmp0 = 8*s*(uu-q1q+mw2)/(q1c*q2c*tt)+4*s*(-uu+s-q2c+q1q+q1c)/(mw2*
     1   q2q*tt)+2*(-uu-9*s-5*q2q-4*q2c-4*q1q-3*q1c+8*mw2)/(mz2*tt)+4*s*
     2   (-uu-q2c+q1q)/(mw2*q1c*tt)+2*(-4*uu+3*s+4*q1q+5*q1c)/(q2q*tt)+2
     3   *(s*tt+q2c*tt+s**2-q2c**2+q1q*q2c)/(mw2*mz2*uu)-8*s*(tt+s+q1q)/
     4   (mz2*q2c*uu)+2*(-tt+3*s+q2c-q1q)/(mz2*uu)-8*s**3/(q1q*q2c*tt*uu
     5   )-16*s**2/(q1q*tt*uu)+2*s*(2*s+q2c)/(mz2*tt*uu)+tmp0
      rud = 2*s*(2*s+q2c)/(mw2*tt*uu)-2*q2c*s*(s+q2c)/(mw2*mz2*tt*uu)-2*
     1   s/(tt*uu)-16*s**2/(q1q*q2c*uu)-8*s/(q2c*uu)-16*s/(q1q*uu)+6*s/(
     2   mw2*uu)+4*s*(tt+q1q-mz2)/(mw2*q1c*q2q)+2*(s+5*q1q+4*q1c-4*mw2)/
     3   (q2c*tt)+4*s/(q1c*tt)+4*s/(mw2*tt)+12/tt+tmp0-4*s**3/(q1c*q1q*q
     4   2c*q2q)-4*s**2/(q1q*q2c*q2q)-4*s**2/(q1c*q1q*q2q)-4*s**2/(q1c*q
     5   1q*q2c)+8*s**2/(mz2*q1q*q2c)-4*(3*s+2*q1q)/(q1c*q2c)-(s+4*q2c)/
     6   (mw2*mz2)-4*(s+2*mw2)/(q2c*q2q)+4*(s+mw2)/(mz2*q2c)-4*s/(q1q*q2
     7   q)-4*s/(q1c*q1q)+12*s/(mz2*q1q)+rud-8*mz2/(q1c*q2q)+8/q2q+8/q1c
      rzd = 8*(q1q*q2q*(5*s**2/ 2.d0+3*tt**2+5*s*tt)+(tt+s)*(tt*uu**2+
     1   q1q**2*q2q)+q1q*(s*uu**2+tt**2*uu+s**2*uu-s**2*tt)+q1q**3*(uu+s
     2   )-q1q**2*s*s2)/(q1q*q2q*tt*uu)-4*sig*(tt**2*(4*uu+s-2*q2q+q1q)+
     3   (3*s+q1q)*tt*uu+(-3*q2q*s-q1q*s+(q2q+q1q)**2-2*q1q**2)*tt-q1q*s
     4   *(4*s-q2q-2*q1q))/(q1q*q2q*tt*uu)
      rzd = -4*sig**2*(tt*(5*tt/ 2.d0+uu+5*s+q2q+q1q)+s*(2*s+q1q))/(mw
     1   2*mz2*tt*uu)-16*mw2*mz2*(tt*uu/ 2.d0+q2q*tt-q1q*s)/(q1q*q2q*t
     2   t*uu)+2*s2*sig*(tt*(uu+s+2*q2q+4*q1q)+4*(tt+s)**2+2*q1q*(2*s+q1
     3   q))/(mw2*mz2*tt*uu)+sig**3*(uu+tt)/(mw2*mz2*tt*uu)+4*mw2*mz2*si
     4   g*(uu+tt)/(q1q*q2q*tt*uu)-4*q1q*s2**2*(tt+s+q1q)/(mw2*mz2*tt*uu
     5   )+4*(s2+s+q2q-q1q)*sig**2/(q1q*q2q*uu)+rzd
      tmp0 = 8*(q1q*q2q*(5*s**2/ 2.d0+3*uu**2+5*s*uu)+q2q*(tt*uu**2-s*
     1   *2*uu+s*tt**2+s**2*tt)+(uu+s)*(tt**2*uu+q1q*q2q**2)+q2q**3*(tt+
     2   s)-q2q**2*s*s2)/(q1q*q2q*tt*uu)-4*sig*((4*tt+s+q2q-2*q1q)*uu**2
     3   +(3*s+q2q)*tt*uu+(-q2q*s-3*q1q*s+(q2q+q1q)**2-2*q2q**2)*uu-q2q*
     4   s*(4*s-2*q2q-q1q))/(q1q*q2q*tt*uu)
      rzd = -4*sig**2*(uu*(5*uu/ 2.d0+tt+5*s+q2q+q1q)+s*(2*s+q2q))/(mw
     1   2*mz2*tt*uu)-16*mw2*mz2*(tt*uu/ 2.d0+q1q*uu-q2q*s)/(q1q*q2q*t
     2   t*uu)+2*s2*sig*(4*(uu+s)**2+(tt+s+4*q2q+2*q1q)*uu+2*q2q*(2*s+q2
     3   q))/(mw2*mz2*tt*uu)+sig**3*(uu+tt)/(mw2*mz2*tt*uu)+4*mw2*mz2*si
     4   g*(uu+tt)/(q1q*q2q*tt*uu)-4*q2q*s2**2*(uu+s+q2q)/(mw2*mz2*tt*uu
     5   )+4*(s2+s-q2q+q1q)*sig**2/(q1q*q2q*tt)+tmp0+rzd
      rzd = 4*mz2*(q1q*(q2q/ 2.d0+uu-tt-s+q1q)+2*uu**2-s*tt+q2q*(s-3*q
     1   2q))/(q1q*q2q*uu)-2*mz2*(tt+2*s+2*q2q)/(mw2*uu)-4*mz2**2*(-tt-2
     2   *s-q2q+q1q)/(q1q*q2q*uu)-2*(s+2*q2q)*s2/(mz2*uu)+8*mw2*mz2**2/(
     3   q1q*q2q*uu)+2*mz2**2/(mw2*uu)+rzd
      rzd = -4*mw2*(q1q*(q2q/ 2.d0+uu-tt-s+q1q)+2*uu**2-s*tt+q2q*(s-3*
     1   q2q))/(q1q*q2q*uu)+2*mw2*(tt+2*s+2*q2q)/(mz2*uu)+4*mw2**2*(-tt-
     2   2*s-q2q+q1q)/(q1q*q2q*uu)+2*(s+2*q2q)*s2/(mw2*uu)-8*mw2**2*mz2/
     3   (q1q*q2q*uu)-2*mw2**2/(mz2*uu)+rzd
      rzd = -4*mz2*(q2q*(q1q/ 2.d0-uu+tt-s+q2q)-s*uu+2*tt**2+q1q*(s-3*
     1   q1q))/(q1q*q2q*tt)+2*mz2*(uu+2*s+2*q1q)/(mw2*tt)+4*mz2**2*(-uu-
     2   2*s+q2q-q1q)/(q1q*q2q*tt)+2*(s+2*q1q)*s2/(mz2*tt)-8*mw2*mz2**2/
     3   (q1q*q2q*tt)-2*mz2**2/(mw2*tt)+rzd
      rzd = 4*mw2*(q2q*(q1q/ 2.d0-uu+tt-s+q2q)-s*uu+2*tt**2+q1q*(s-3*q
     1   1q))/(q1q*q2q*tt)-2*mw2*(uu+2*s+2*q1q)/(mz2*tt)-4*mw2**2*(-uu-2
     2   *s+q2q-q1q)/(q1q*q2q*tt)-2*(s+2*q1q)*s2/(mw2*tt)+8*mw2**2*mz2/(
     3   q1q*q2q*tt)+2*mw2**2/(mz2*tt)+rzd
      rzu = 8*(q1c*q2c*(5*s**2/ 2.d0+3*tt**2+5*s*tt)+(tt+s)*(tt*uu**2+
     1   q1c**2*q2c)+q1c*(s*uu**2+tt**2*uu+s**2*uu-s**2*tt)+q1c**3*(uu+s
     2   )-q1c**2*s*s2)/(q1c*q2c*tt*uu)-4*sig*(tt**2*(4*uu+s-2*q2c+q1c)+
     3   (3*s+q1c)*tt*uu+(-3*q2c*s-q1c*s+(q2c+q1c)**2-2*q1c**2)*tt-q1c*s
     4   *(4*s-q2c-2*q1c))/(q1c*q2c*tt*uu)
      rzu = -4*sig**2*(tt*(5*tt/ 2.d0+uu+5*s+q2c+q1c)+s*(2*s+q1c))/(mw
     1   2*mz2*tt*uu)-16*mw2*mz2*(tt*uu/ 2.d0+q2c*tt-q1c*s)/(q1c*q2c*t
     2   t*uu)+2*s2*sig*(tt*(uu+s+2*q2c+4*q1c)+4*(tt+s)**2+2*q1c*(2*s+q1
     3   c))/(mw2*mz2*tt*uu)+sig**3*(uu+tt)/(mw2*mz2*tt*uu)+4*mw2*mz2*si
     4   g*(uu+tt)/(q1c*q2c*tt*uu)-4*q1c*s2**2*(tt+s+q1c)/(mw2*mz2*tt*uu
     5   )+4*(s2+s+q2c-q1c)*sig**2/(q1c*q2c*uu)+rzu
      tmp0 = 8*(q1c*q2c*(5*s**2/ 2.d0+3*uu**2+5*s*uu)+q2c*(tt*uu**2-s*
     1   *2*uu+s*tt**2+s**2*tt)+(uu+s)*(tt**2*uu+q1c*q2c**2)+q2c**3*(tt+
     2   s)-q2c**2*s*s2)/(q1c*q2c*tt*uu)-4*sig*((4*tt+s+q2c-2*q1c)*uu**2
     3   +(3*s+q2c)*tt*uu+(-q2c*s-3*q1c*s+(q2c+q1c)**2-2*q2c**2)*uu-q2c*
     4   s*(4*s-2*q2c-q1c))/(q1c*q2c*tt*uu)
      rzu = -4*sig**2*(uu*(5*uu/ 2.d0+tt+5*s+q2c+q1c)+s*(2*s+q2c))/(mw
     1   2*mz2*tt*uu)-16*mw2*mz2*(tt*uu/ 2.d0+q1c*uu-q2c*s)/(q1c*q2c*t
     2   t*uu)+2*s2*sig*(4*(uu+s)**2+(tt+s+4*q2c+2*q1c)*uu+2*q2c*(2*s+q2
     3   c))/(mw2*mz2*tt*uu)+sig**3*(uu+tt)/(mw2*mz2*tt*uu)+4*mw2*mz2*si
     4   g*(uu+tt)/(q1c*q2c*tt*uu)-4*q2c*s2**2*(uu+s+q2c)/(mw2*mz2*tt*uu
     5   )+4*(s2+s-q2c+q1c)*sig**2/(q1c*q2c*tt)+tmp0+rzu
      rzu = 4*mw2*(q1c*(q2c/ 2.d0+uu-tt-s+q1c)+2*uu**2-s*tt+q2c*(s-3*q
     1   2c))/(q1c*q2c*uu)-2*mw2*(tt+2*s+2*q2c)/(mz2*uu)-4*mw2**2*(-tt-2
     2   *s-q2c+q1c)/(q1c*q2c*uu)-2*(s+2*q2c)*s2/(mw2*uu)+8*mw2**2*mz2/(
     3   q1c*q2c*uu)+2*mw2**2/(mz2*uu)+rzu
      rzu = -4*mz2*(q1c*(q2c/ 2.d0+uu-tt-s+q1c)+2*uu**2-s*tt+q2c*(s-3*
     1   q2c))/(q1c*q2c*uu)+2*mz2*(tt+2*s+2*q2c)/(mw2*uu)+4*mz2**2*(-tt-
     2   2*s-q2c+q1c)/(q1c*q2c*uu)+2*(s+2*q2c)*s2/(mz2*uu)-8*mw2*mz2**2/
     3   (q1c*q2c*uu)-2*mz2**2/(mw2*uu)+rzu
      rzu = -4*mw2*(q2c*(q1c/ 2.d0-uu+tt-s+q2c)-s*uu+2*tt**2+q1c*(s-3*
     1   q1c))/(q1c*q2c*tt)+2*mw2*(uu+2*s+2*q1c)/(mz2*tt)+4*mw2**2*(-uu-
     2   2*s+q2c-q1c)/(q1c*q2c*tt)+2*(s+2*q1c)*s2/(mw2*tt)-8*mw2**2*mz2/
     3   (q1c*q2c*tt)-2*mw2**2/(mz2*tt)+rzu
      rzu = 4*mz2*(q2c*(q1c/ 2.d0-uu+tt-s+q2c)-s*uu+2*tt**2+q1c*(s-3*q
     1   1c))/(q1c*q2c*tt)-2*mz2*(uu+2*s+2*q1c)/(mw2*tt)-4*mz2**2*(-uu-2
     2   *s+q2c-q1c)/(q1c*q2c*tt)-2*(s+2*q1c)*s2/(mz2*tt)+8*mw2*mz2**2/(
     3   q1c*q2c*tt)+2*mz2**2/(mw2*tt)+rzu
      rz = 2*s2*sig**2*(tt*(3*uu+9*tt+19*s+4*q2q+6*q1q)+8*s**2+6*q1q*s+2
     1   *q1q**2)/(mw2*mz2*tt*uu)
      rz = -16*sig*(2*tt*(uu/ 2.d0-tt-s+q2q+q1q)-s*(3*s/ 2.d0-2*q1q)
     1   )/(tt*uu)+8*s2*(s*(s/ 2.d0+tt)+4*q1q*(tt+s+q1q))/(tt*uu)+8*si
     2   g**2*(s/ 2.d0+2*tt)/(tt*uu)-2*sig**3*(tt*(3*uu+6*tt+11*s+2*q2
     3   q+2*q1q)+2*s*(2*s+q1q))/(mw2*mz2*tt*uu)-2*s2**2*sig*(tt*(uu+4*t
     4   t+9*s+2*q2q+6*q1q)+4*(s+q1q)**2-2*q1q*s)/(mw2*mz2*tt*uu)+4*q1q*
     5   s2**3*(tt+s+q1q)/(mw2*mz2*tt*uu)+32*mw2*mz2*s/(tt*uu)+2*sig**4/
     6   (mw2*mz2*uu)+rz
      tmp0 = 2*s2*sig**2*(uu*(9*uu+3*tt+19*s+6*q2q+4*q1q)+8*s**2+6*q2q*s
     1   +2*q2q**2)/(mw2*mz2*tt*uu)
      rz = -16*sig*(2*uu*(tt/ 2.d0-uu-s+q2q+q1q)-s*(3*s/ 2.d0-2*q2q)
     1   )/(tt*uu)+8*s2*(s*(s/ 2.d0+uu)+4*q2q*(uu+s+q2q))/(tt*uu)+8*si
     2   g**2*(s/ 2.d0+2*uu)/(tt*uu)-2*sig**3*(uu*(6*uu+3*tt+11*s+2*q2
     3   q+2*q1q)+2*s*(2*s+q2q))/(mw2*mz2*tt*uu)-2*s2**2*sig*(uu*(4*uu+t
     4   t+9*s+6*q2q+2*q1q)+4*(s+q2q)**2-2*q2q*s)/(mw2*mz2*tt*uu)+4*q2q*
     5   s2**3*(uu+s+q2q)/(mw2*mz2*tt*uu)+32*mw2*mz2*s/(tt*uu)+2*sig**4/
     6   (mw2*mz2*tt)+tmp0+rz
      rz = -2*mz2*(2*tt+11*s+18*q2q)/uu-2*mz2**2*(2*tt+3*s+2*q2q)/(mw2*u
     1   u)+2*mz2*s2*(tt+3*s+4*q2q)/(mw2*uu)-2*(s+2*q2q)*s2**2/(mw2*uu)+
     2   2*mz2**3/(mw2*uu)+20*mz2**2/uu+rz
      rz = 2*mw2*(2*tt+11*s+18*q2q)/uu+2*mw2**2*(2*tt+3*s+2*q2q)/(mz2*uu
     1   )-2*mw2*s2*(tt+3*s+4*q2q)/(mz2*uu)+2*(s+2*q2q)*s2**2/(mz2*uu)-2
     2   *mw2**3/(mz2*uu)-20*mw2**2/uu+rz
      rz = 2*mz2*(2*uu+11*s+18*q1q)/tt+2*mz2**2*(2*uu+3*s+2*q1q)/(mw2*tt
     1   )-2*mz2*s2*(uu+3*s+4*q1q)/(mw2*tt)+2*(s+2*q1q)*s2**2/(mw2*tt)-2
     2   *mz2**3/(mw2*tt)-20*mz2**2/tt+rz
      rz = -2*mw2*(2*uu+11*s+18*q1q)/tt-2*mw2**2*(2*uu+3*s+2*q1q)/(mz2*t
     1   t)+2*mw2*s2*(uu+3*s+4*q1q)/(mz2*tt)-2*(s+2*q1q)*s2**2/(mz2*tt)+
     2   2*mw2**3/(mz2*tt)+20*mw2**2/tt+rz
      wz3b = -2*ez*(gdown*rzd-gup*rzu)/(s2-mw2)+ez**2*rz/(s2-mw2)**2+gup
     1   **2*ruu+2*gdown*gup*rud+gdown**2*rdd
      wz3b = -tt*uu*wz3b/(nc*s)
      endif
      end
c
c
c-------------------------------------------------
c SEZIONI D'URTO PER q g --> WZ q
c-------------------------------------------------

c-------------------------------------------------
c Contributo a tre corpi collineare
c alla sezione d'urto totale per qg->WZq
c
c d sigma^(c-) = gw**2 (alfas T_F)/(4*pi)*colg(x,q1q)*d Phi2^(x)
c
      function wzcolg(x,q1q,s,mw2,mz2,xlmude)
      implicit double precision (a-z)
      character * 2 scheme
      common/scheme/scheme
      born0 = wzborn(q1q,x*s,mw2,mz2)
      wzcolg = 2*( (xlmude + 2*log(1-x)) * (x**2+(1-x)**2)
     #            + 2*x*(1-x) ) * born0
      if(scheme.eq.'DI') then
         dis = (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1
         wzcolg = wzcolg - 2*dis*born0
      elseif(scheme.eq.'MS') then
      else
         call noscheme
      endif
      return
      end
c
c--------------------------------------------------------------
c Contributo a tre corpi alla sezione d'urto totale per qg->WZq
c
c d sigma^(f) = gw**2 *(alfas T_F)/(4*pi)
c               * P{1/(1+y)} * wz3g
c               * beta_x/(32*pi^2*s) d cos(th1) d th2 dy dx
c
      function wz3g(xx,yy,t,u,q1q,q2q,s,mw2,mz2)
      implicit real * 8 (a-z)
c-------------------------------------------
c Otteniamo wz3g tramite crossing
c
      tiny = .1d-5
      x=xx
      y=yy
      if(x.eq.1.or.y.eq.1)then
         wz3g = 0
      elseif(1+y.le.tiny)then
         pqg = x**2+(1-x)**2
         wz3g = 4*s*pqg*wzborn(q1q,x*s,mw2,mz2)
      else
         sx = t
         tx = s
         ux = u
         q1qx = q1q
         w2   = mz2+q1q-q2q-u
         q2qx = w2
         wz3bx  = wz3b(x,y,tx,ux,q1qx,q2qx,sx,mw2,mz2)
c--------------------------------------------
c mxperu = mx * ux
c
         mxperu = wz3bx/(4*tx)
         mperu  = - mxperu*sx/s
         wz3g = - 2*s*mperu
      endif
      return
      end
c
      subroutine noscheme
      character * 2 scheme
      common/scheme/scheme
      write(6,*) ' error: unknown scheme',scheme,' use'
      write(6,*) '     character * 2 scheme'
      write(6,*) '     common/scheme/scheme'
      write(6,*) '     scheme=''MS'' for the MS_bar scheme'
      write(6,*) '     scheme=''DI'' for the deep-inelastic scheme'
      stop
      end
c
c
c End of wzcross.for
c
c
c
c
c Begin of zzcross.for
c
c
c Contributo a due corpi alla sezione d'urto totale per qq->zz
c Born term
c
c d sigma^(b) = (ga^4+gv^4+6*ga^2*gv^2)/N_C * born * d phi2
c
c dove  d phi2 e' lo spazio delle fasi a due corpi (come nell'articolo.)
c
      function zzborn(s,t,m2)
      implicit real * 8 (a-z)
      u = 2*m2-s-t
      zzborn = -2*m2/((s-2*m2)*u)+(-s**2+2*m2*s-4*m2**2)/(s*(s-2*m2)*t)
     1       -m2**2/(s*t**2)-1/s
      return
      end
c---------------------------------------------
c Contributo a due corpi alla sezione d'urto totale per qq->zz
c Virtuale + soft
c
c d sigma^(sv) = (ga^4+gv^4+6*ga^2*gv^2)*(alfas C_F)/(4*pi*N_C) * zz2b * d Phi2
c
      function zz2b(t,s,m2,mu2)
      implicit real * 8 (a-z)
      character * 2 scheme
      common/scheme/scheme
      common/betfac/betfac,delta
      pi2 = dacos(-1.d0)**2
      u = 2*m2-s-t
      xlt = log(-t/m2)
      xi4 = (-4*ddilog(t/m2)+2*log(-t/m2)**2+4*log(-m2/t)*log((m2-t)/m2)
     1   )/(s*t)
      xi3t = -(-log(-m2/t)*log(-t/s)-log(-m2/t)**2/2+pi2/2)/(m2-t)
      beta = sqrt(1-4*m2/s)
      xlbeta = log(beta*betfac)
      xlp = (beta+1)/2
      xlm = (1-beta)/2
      xi3l = 2*(log(xlm/xlp)**2/2+2*ddilog(-xlm/xlp)+pi2/6)/(beta*s)
      xls = log(s/m2)
      xlsmu = log(s/mu2)
      born0 = -((4*t**2+(4*s-8*m2)*t+4*m2**2)*u-8*m2*s*t)/(s*t**2*u)/4
      xvps = -born0*(-48*xlbeta*xlsmu-18*xlsmu-96*xlbeta**2+4*pi2)/3
c-------------------------------------------------------------------
c Ridefinizione delle funzioni di struttura.
c
      if(scheme.eq.'DI') then
         dis = -4*born0*
     #   (-(4.5d0+pi2/3)+(-1.5d0)*2*xlbeta+2*2*xlbeta**2)
         xvps = xvps + dis
      elseif(scheme.eq.'MS') then
      else
         call noscheme
      endif
      xvirt = (2*(t+s)**2*(t**2-3*s**2)/(s*t*(t-s)**2*u)-4*s*t/((t-m2)**
     1   2*(t-s))+8*s*(t-2*s)/((t-m2)*(t-s)**2)+6*m2**2/(s*t**2))*xlt
      xvirt = xvirt+(-3*(t+s)**2*(t**2+4*s*t+2*s**2)/(s*t*(2*t+s)**2*u)-
     1   (192*t**3+32*s*t**2-64*s**2*t-13*s**3)/((s-4*m2)*t*(2*t+s)**2)/
     2   8+(17*t**2-8*s*t-38*m2*t+24*m2**2)/(s*t**2)/4+3*s*(4*t+s)
     3   **2/(8*(s-4*m2)**2*t*(2*t+s)))*xls+(2*(t**2-s*t-2*m2*t+2*m2**
     4   2)/t-4*s**2/u)*xi4+(4*s*(t-s)/(t*u)-4*(t**3-s*t**2-3*m2*t**2+s*
     5   *2*t+m2*s*t+4*m2**2*t-2*m2**3)/(s*t**2))*xi3t
      xvirt = xvirt+(-(3*t**4+20*s*t**3+70*s**2*t**2+60*s**3*t+15*s**4)/
     1   (s*(2*t+s)**2*u)/2+s*(112*t**3+32*s*t**2-27*s**2*t-6*s**3)/((
     2   s-4*m2)*t*(2*t+s)**2)/16-(12*t**2+152*s*t-72*m2*t+49*s**2-72*
     3   m2*s+48*m2**2)/(s*t)/32+(-3)*s**2*(4*t+s)**2/(32*(s-4*m2)
     4   **2*t*(2*t+s)))*xi3l+pi2*(4*u/(3*s*t)+8/(3*u))+(t+s)*(t
     5   **3-28*s*t**2+21*s**2*t+18*s**3)/(s*t*(t-s)*(2*t+s)*u)+(66*t**2
     6   +s*t-140*m2*t+72*m2**2)/(s*t**2)/4-(4*t+s)**2/((s-4*m2)*t*(2*
     7   t+s))/4+4*s/((t-m2)*(t-s))
      zz2b = xvps+xvirt
      return
      end

c---------------------------------------------
c Contributo a tre corpi collineare alla sezione d'urto totale per qq->zzg
c
c d sigma^(c+) = (ga^4+gv^4+6*ga^2*gv^2)*(alfas C_F)/(4*pi*N_C)
c *[ colp(x,q2q)*P{1/(1-x)} + coll(x,q2q)P{log(1-x)/(1-x)} ]*d Phi2^(x)
c
c e analoga per d sigma^(c-), con q2q--> q1q
c
c P{1/(1-x)} e' {1/(1-x)}_+
c P{log(1-x)/(1-x)} e' {log(1-x)/(1-x)}_+
c
      function zzcolp(x,q1q,s,m2,xlmude)
      implicit double precision (a-z)
      character * 2 scheme
      common/scheme/scheme
      born0 = zzborn(x*s,q1q,m2)
      zzcolp = 2*(xlmude*(1+x**2)+(1-x)**2)*born0
      if(scheme.eq.'DI') then
         dis = -1.5d0-(1+x**2)*log(x)+(1-x)*(3+2*x)
         zzcolp = zzcolp - 2*dis*born0
      elseif(scheme.eq.'MS') then
      else
         call noscheme
      endif
      return
      end
c
      function zzcoll(x,q1q,s,m2)
      implicit double precision (a-z)
      character * 2 scheme
      common/scheme/scheme
      born0 = zzborn(x*s,q1q,m2)
      zzcoll = 4*(1+x**2)*born0
      if(scheme.eq.'DI') then
         dis = 1+x**2
         zzcoll = zzcoll - 2*dis*born0
      elseif(scheme.eq.'MS') then
      else
         call noscheme
      endif
      return
      end

c--------------------------------------------------------------
c Contributo a tre corpi alla sezione d'urto totale per qq->zzg
c
c d sigma^(f) = (ga^4+gv^4+6*ga^2*gv^2)*(alfas C_F)/(4*pi*N_C)
c               * P{1/(1-x)} * P{1/(1-y)+1/(1+y)} * ff
c               * beta_x/(8*pi)^2/s d cos(th1) d th2 dy dx
c
      function zzff(xx,yy,t,u,q1q,q2q,s,m2)
      implicit real * 8 (a-z)
c
      tiny = .1d-5
      x=xx
      y=yy
      if(1-x.le.tiny)then
         tmpp = 8*s*(1+x**2)*zzborn(x*s,q2q,m2)
         tmpm = 8*s*(1+x**2)*zzborn(x*s,q1q,m2)
         zzff = (tmpp+tmpm)/2.d0
      elseif(1-y.le.tiny)then
         zzff = 8*s*(1+x**2)*zzborn(x*s,q2q,m2)
      elseif(1+y.le.tiny)then
         zzff = 8*s*(1+x**2)*zzborn(x*s,q1q,m2)
      else
      q1c = -t-s-q1q+2*m2
      q2c = -u-s-q2q+2*m2
      r1 = t*u*((u-t)*(q1q*(u+s)/(q2q*t)-q2q*(t+s)/(q1q*u))/(q1c*q2c*s*
     1    2.d0)-(t*u**2+2*s*u**2-m2*u**2-t**2*u-2*s*t*u+2*m2*s*u+m2*t**
     2   2-2*m2*s*t-2*m2*s**2)/(q1c*q1q*q2c*s*u)+(t*u**2-m2*u**2+s*t*u-3
     3   *m2*t*u-3*m2*s*u+2*m2**2*u-m2*s*t+2*m2**2*t+4*m2**2*s)/(q1c*q1q
     4   *q2q*s*u)+(t*u**2+t**2*u-3*m2*t*u+4*m2*s*u-m2*t**2-s**2*t+4*m2*
     5   s**2)/(q1c*q2q*s*t*u)+2*(t*u**2-3*m2*t*u-m2*s*u+2*m2**2*u-m2**2
     6   *s)/(q1q*q2q*s*t*u)-2*m2*(u**2-2*m2*u+2*m2**2)/(q1q**2*q2q*s*u)
     7   +(t*u+s*u-m2*u+3*s*t-m2*t+2*s**2-2*m2*s)/(q1c*q2c*s*t)-(t*u+s*u
     8   -s*t-2*m2*t-s**2+2*m2*s)/(q1c*q1q*s*u)+q2q*(t*u-s*u-2*s*t-2*s**
     9   2)/(q1q*q2c*s*t*u)+2*(t*u-m2*u-t**2-s*t-s**2+2*m2*s)/(q1q*s*t*u
     :   )+(2*s*u+t**2+3*s*t+2*s**2)/(q1c*s*t*u)+q1q*(u+s)*(u+t)/(q1c*q2
     ;   c*s*t*u)+(t-2*m2)*(u-2*m2)/(q1c*q1q*q2c*q2q)-(q2q/q1q+q1q/q2q)/
     <   (t*u)-2*m2*(q2q-2*m2)/(q1q**2*s*u))
      r2 = t*u*((t-u)*(q2q*(t+s)/(q1q*u)-q1q*(u+s)/(q2q*t))/(q1c*q2c*s*
     1    2.d0)+(t*u**2-m2*u**2+t**2*u-3*m2*t*u-s**2*u+4*m2*s*t+4*m2*s*
     2   *2)/(q1q*q2c*s*t*u)-(-t*u**2+m2*u**2+t**2*u-2*s*t*u-2*m2*s*u+2*
     3   s*t**2-m2*t**2+2*m2*s*t-2*m2*s**2)/(q1c*q2c*q2q*s*t)+(u**2+3*s*
     4   u+2*s*t+2*s**2)/(q2c*s*t*u)+2*(-u**2+t*u-s*u-m2*t-s**2+2*m2*s)/
     5   (q2q*s*t*u)+(t**2*u+s*t*u-3*m2*t*u-m2*s*u+2*m2**2*u-m2*t**2-3*m
     6   2*s*t+2*m2**2*t+4*m2**2*s)/(q1q*q2c*q2q*s*t)+2*(t**2*u-3*m2*t*u
     7   -m2*s*t+2*m2**2*t-m2**2*s)/(q1q*q2q*s*t*u)+(t*u+3*s*u-m2*u+s*t-
     8   m2*t+2*s**2-2*m2*s)/(q1c*q2c*s*u)-(t*u-s*u-2*m2*u+s*t-s**2+2*m2
     9   *s)/(q2c*q2q*s*t)+q1q*(t*u-2*s*u-s*t-2*s**2)/(q1c*q2q*s*t*u)+q2
     :   q*(t+s)*(u+t)/(q1c*q2c*s*t*u)+(t-2*m2)*(u-2*m2)/(q1c*q1q*q2c*q2
     ;   q)-(q2q/q1q+q1q/q2q)/(t*u)-2*m2*(t**2-2*m2*t+2*m2**2)/(q1q*q2q*
     <   *2*s*t)-2*m2*(q1q-2*m2)/(q2q**2*s*t))
      zzff = 4*(r2+r1)
      endif
      return
      end

c-------------------------------------------------
c SEZIONI D'URTO PER q g --> ZZ q
c-------------------------------------------------
c Contributo a tre corpi collineare alla sezione d'urto totale per qg->zzq
c
c d sigma^(c+) = (ga^4+gv^4+6*ga^2*gv^2)*(alfas T_F)/(4*pi*N_C)
c *  colg(x,q1q) * d Phi2^(x)
c
      function zzcolg(x,q1q,s,m2,xlmude)
      implicit double precision (a-z)
      character * 2 scheme
      common/scheme/scheme
      born0 = zzborn(x*s,q1q,m2)
      zzcolg = 2*( (xlmude + 2*log(1-x)) * (x**2+(1-x)**2)
     #           + 2*x*(1-x) ) * born0
      if(scheme.eq.'DI') then
         dis = (x**2+(1-x)**2)*log((1-x)/x)+8*x*(1-x)-1
         zzcolg = zzcolg - 2*dis*born0
      elseif(scheme.eq.'MS') then
      else
         call noscheme
      endif
      return
      end

c--------------------------------------------------------------
c Contributo a tre corpi alla sezione d'urto totale per qg->zzq
c
c d sigma^(f) = (ga^4+gv^4+6*ga^2*gv^2)*(alfas T_F)/(4*pi*N_C)
c               * P{1/(1+y)} * ffg
c               * beta_x/(32*pi^2*s) d cos(th1) d th2 dy dx
c
      function zzffg(xx,yy,t,u,q1q,q2q,s,m2)
      implicit real * 8 (a-z)
c--------------------------------------------------------------
c Otteniamo ff tramite crossing
c
      tiny = .1d-5
      x=xx
      y=yy
      if(x.eq.1.or.y.eq.1)then
         zzffg = 0
      elseif(1+y.le.tiny)then
         pqg = x**2+(1-x)**2
         zzffg = 4*s*pqg*zzborn(x*s,q1q,m2)
      else
         sx = t
         tx = s
         ux = u
         q1qx = q1q
         w2   = m2+q1q-q2q-u
         q2qx = w2
         ffx  = zzff(x,y,tx,ux,q1qx,q2qx,sx,m2)
c--------------------------------------------------------------
c mxperu = mx * ux
c
         mxperu = ffx/(4*tx)
         mperu  = - mxperu*sx/s
         zzffg = - 2*s*mperu
      endif
      return
      end
c
c
c End of zzcross.for
c
c
c
c
c Begin of lepton matrix elements (from MadEvent): use for spin correlations
c
c
      function xmadevww(iflav,iborn,jproc,idr,s,tk,uk,xmom)
c Wrapper for MadEvent functions: qqbar, qg, and gq are needed, for up and
c down type quarks. Inputs are 
c   iflav = 1(up), 2(down)
c   iborn = 0(born), 1(real)
c   jproc = 2(qq), 3(qg)
c   idr   = 1(qqbar,qg), 2(qbarg), 3(qbarq,gq), 4(gqbar)
c   s     = parton cm energy squared
c   tk,uk = FNR invariants
c   xmom  = 4-momenta obtained from invar
c Output is the matrix element squared in GeV^-2, times the flux factor,
c times 4*tk*uk in the case of real matrix elements.
c MadEvent routines use the SM parameters as defined in setmepar();
c for kinematic-dependent couplings, and if simple factorized
c expressions can't be found (which appears to happen only if a Higgs 
c is involved in the reaction), the call to setmepar() must be included
c in this routine. In the present case, all the Born (real) formulae
c are proportional to e^4 (g^2 e^4), and this call can be placed somewhere
c else (and done only once).
      implicit none
      integer iflav,iborn,jproc,idr
      real*8 xmadevww,s,tk,uk,xmom(9,4),xfact,tmp(1)
      real*8 pme0(0:3,6),pme1(0:3,7)
      integer ipart,icomp
c Components: MC@NLO conventions   -> 1=px, 2=py, 3=pz, 4=E
c             MadEvent conventions -> 0=E, 1=px, 2=py, 3=pz
      integer mapcomp(0:3)
      data mapcomp/4,1,2,3/
c Subprocess: MC@NLO   -> a(1)b(2) -> c(3)W+(4)[->e+(6)n(7)]W-(5)[->e-(8)n(9)]
c             MadEvent -> a(1)b(2) -> e+(3)n(4)e-(5)n(6)c(7)
      integer mapdir(7),mapcon(7),mapref(7),mapcrf(7)
c mapping for directed events: qqbar, qg
      data mapdir/1,2,6,7,8,9,3/
c mapping for conjugated events: qbarg
      data mapcon/1,2,8,9,6,7,3/
c mapping for reflected events: qbarq, gq
      data mapref/2,1,6,7,8,9,3/
c mapping for conjugated-reflected events: gqbar
      data mapcrf/2,1,8,9,6,7,3/
c
      if(iborn.eq.0)then
        if(jproc.ne.2)then
          write(*,*)'Error #1 in xmadevww: iborn, jproc=',iborn,jproc
          stop
        endif
        xfact=1.d0
        do ipart=1,6
          do icomp=0,3
            if(idr.eq.1)then
              pme0(icomp,ipart)=xmom(mapdir(ipart),mapcomp(icomp))
            elseif(idr.eq.3)then
              pme0(icomp,ipart)=xmom(mapref(ipart),mapcomp(icomp))
            else
              write(*,*)'Error #2 in xmadevww: idr=',idr
              stop
            endif
          enddo
        enddo
      elseif(iborn.eq.1)then
        if(jproc.eq.2.and.(idr.eq.2.or.idr.eq.4))then
          write(*,*)'Error #3 in xmadevww: jproc,idr=',jproc,idr
          stop
        endif
        xfact=4*tk*uk
        do ipart=1,7
          do icomp=0,3
            if(idr.eq.1)then
              pme1(icomp,ipart)=xmom(mapdir(ipart),mapcomp(icomp))
            elseif(idr.eq.2)then
              pme1(icomp,ipart)=xmom(mapcon(ipart),mapcomp(icomp))
            elseif(idr.eq.3)then
              pme1(icomp,ipart)=xmom(mapref(ipart),mapcomp(icomp))
            elseif(idr.eq.4)then
              pme1(icomp,ipart)=xmom(mapcrf(ipart),mapcomp(icomp))
            else
              write(*,*)'Error #4 in xmadevww: idr=',idr
              stop
            endif
          enddo
        enddo
      else
        write(*,*)'xmadevww: unknown iborn value',iborn
        stop
      endif
      if(iflav.eq.1)then
        if(iborn.eq.0)then
          call uubwwborn(pme0,tmp)
        elseif(iborn.eq.1.and.jproc.eq.2)then
          call uubwwreal(pme1,tmp)
        elseif(iborn.eq.1.and.jproc.eq.3)then
          call ugwwreal(pme1,tmp)
        else
          write(*,*)'xmadevww: Error #5'
          stop
        endif
      elseif(iflav.eq.2)then
        if(iborn.eq.0)then
          call ddbwwborn(pme0,tmp)
        elseif(iborn.eq.1.and.jproc.eq.2)then
          call ddbwwreal(pme1,tmp)
        elseif(iborn.eq.1.and.jproc.eq.3)then
          call dgwwreal(pme1,tmp)
        else
          write(*,*)'xmadevww: Error #6'
          stop
        endif
      else
        write(*,*)'xmadevww: unknown iflav value',iflav
        stop
      endif
      xmadevww=xfact*tmp(1)/(2*s)
      return
      end


      subroutine setmepar(xiwmass,xiwwidth,xizmass,xizwidth,
     #                    xisin2w,xiee2,xig)
c Fills HELAS common blocks for masses and couplings. The electron charge
c squared and the masses may eventually be passed through a common block
c on a event-by-event basis. This code is mainly taken from coupsm-ORIGINAL.F 
c of the HELAS package. Here, we limit ourselves to setting the following
c parameters:
c
c       real    gw                : weak coupling constant
c       real    gwwa              : dimensionless WWA  coupling
c       real    gwwz              : dimensionless WWZ  coupling
c       complex gal(2)            : coupling with A of charged leptons
c       complex gau(2)            : coupling with A of up-type quarks
c       complex gad(2)            : coupling with A of down-type quarks
c       complex gwf(2)            : coupling with W-,W+ of fermions
c       complex gzn(2)            : coupling with Z of neutrinos
c       complex gzl(2)            : coupling with Z of charged leptons
c       complex gzu(2)            : coupling with Z of up-type quarks
c       complex gzd(2)            : coupling with Z of down-type quarks
c       complex gg(2)             : QCD gqq coupling (L,R)
c
c through the following parameters, given in input
c
c       real    zmass,wmass       : weak boson masses
c       real    zwidth,wwidth     : weak boson width
c       real    sin2w             : square of sine of the weak angle
c       real    ee2               : positron charge squared
c       real    g                 : QCD 3-,4-gluon coupling
c
      implicit none
      real * 8 xiwmass,xiwwidth,xizmass,xizwidth,xisin2w,xiee2,xig
      include "MEcoupl.inc"
      double precision zero,half,one,two,three,pi,ee2,sw,cw,ez,ey,sc2,v
      parameter (zero=0.d0)
      parameter (half=0.5d0)
      parameter (one=1.d0)
      parameter (two=2.d0)
      parameter (three=3.d0)
      parameter (pi=3.14159265358979312D0)
c
      wmass = xiwmass
      wwidth= xiwwidth
      zmass = xizmass
      zwidth= xizwidth
      sin2w = xisin2w
      ee2   = xiee2
      g     = xig
c
      amass=0.d0
      awidth=1.d-99
c
      ee=sqrt(ee2)
      alpha=ee2/(4*pi)
c
      sw  = sqrt( sin2w )
      cw  = sqrt( One - sin2w )
      ez  = ee/(sw*cw)
      ey  = ee*(sw/cw)
      sc2 = sin2w*( One - sin2w )
      v   = Two*zmass*sqrt(sc2)/ee
c
c vector boson couplings
c
      gw   = ee/sw
      gwwa = ee
      gwwz = ee*cw/sw
c
c fermion-fermion-vector couplings
c
      gal(1) = dcmplx(  ee          , Zero )
      gal(2) = dcmplx(  ee          , Zero )
      gau(1) = dcmplx( -ee*Two/Three, Zero )
      gau(2) = dcmplx( -ee*Two/Three, Zero )
      gad(1) = dcmplx(  ee/Three    , Zero )
      gad(2) = dcmplx(  ee/Three    , Zero )
c
      gwf(1) = dcmplx( -ee/sqrt(Two*sin2w), Zero )
      gwf(2) = dcmplx(  Zero              , Zero )
c
      gzn(1) = dcmplx( -ez*Half                     , Zero )
      gzn(2) = dcmplx(  Zero                        , Zero )
      gzl(1) = dcmplx( -ez*(-Half + sin2w)          , Zero )
      gzl(2) = dcmplx( -ey                          , Zero )
      gzu(1) = dcmplx( -ez*( Half - sin2w*Two/Three), Zero )
      gzu(2) = dcmplx(  ey*Two/Three                , Zero )
      gzd(1) = dcmplx( -ez*(-Half + sin2w/Three)    , Zero )
      gzd(2) = dcmplx( -ey/Three                    , Zero )
c
c QCD coupling
c
      gg(1) = dcmplx( -g, Zero )
      gg(2) = gg(1)
c
      return
      end


      subroutine switchmom(p1,p,ic,jc,nexternal)
c**************************************************************************
c     Changes stuff for crossings
c**************************************************************************
      implicit none
      integer nexternal
      integer jc(nexternal),ic(nexternal)
      real*8 p1(0:3,nexternal),p(0:3,nexternal)
      integer i,j
c-----
c Begin Code
c-----
      do i=1,nexternal
         do j=0,3
            p(j,ic(i))=p1(j,i)
         enddo
      enddo
      do i=1,nexternal
         jc(i)=1
      enddo
      jc(ic(1))=-1
      jc(ic(2))=-1
      end


C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE DDBWWBORN(P1,ANS)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : d d~ -> e+ ve e- ve~  
C  
C Crossing   1 is d d~ -> e+ ve e- ve~  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  6)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB=  64, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATDDBWWB
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    3/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   1/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,6) /-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,6) /-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,6) /-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,6) /-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,6) /-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,6) /-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,6) /-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,6) /-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,6) /-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,6) /-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,6) /-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,6) /-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,6) /-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,6) /-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,6) /-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,6) /-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,6) /-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,6) /-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,6) /-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,6) /-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,6) /-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,6) /-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,6) /-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,6) /-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,6) /-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,6) /-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,6) /-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,6) /-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,6) /-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,6) /-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,6) /-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,6) /-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,6) / 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,6) / 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,6) / 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,6) / 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,6) / 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,6) / 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,6) / 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,6) / 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,6) / 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,6) / 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,6) / 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,6) / 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,6) / 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,6) / 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,6) / 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,6) / 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,6) / 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,6) / 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,6) / 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,6) / 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,6) / 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,6) / 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,6) / 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,6) / 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,6) / 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,6) / 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,6) / 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,6) / 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,6) / 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,6) / 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,6) / 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,6) / 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,6) / 1, 2, 3, 4, 5, 6/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  36/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATDDBWWB(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATDDBWWB(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATDDBWWB
      REAL*8 FUNCTION MATDDBWWB(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : d d~ -> e+ ve e- ve~  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   3,NEIGEN=  1) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  6)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  13, NCOLOR=   1) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS),WX1(6),WX2(6)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            1/                                       
      DATA (CF(i,1  ),i=1  ,1  ) /     3/                                  
C               T[2,1]                                                     
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL OXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),ZERO ,NHEL(5   ),+1*IC(5   ),W(1,5   ))        
      CALL IXXXXX(P(0,6   ),ZERO ,NHEL(6   ),-1*IC(6   ),W(1,6   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,7   ))    
      CALL JIOXXX(W(1,6   ),W(1,5   ),GWF ,WMASS   ,WWIDTH  ,W(1,8   ))    
      CALL FVOXXX(W(1,2   ),W(1,7   ),GWF ,ZERO    ,ZERO    ,W(1,9   ))    
      CALL IOVXXX(W(1,1   ),W(1,9   ),W(1,8   ),GWF ,AMP(1   ))            
      CALL JIOXXX(W(1,1   ),W(1,2   ),GAD ,ZERO    ,AWIDTH  ,W(1,10  ))    
      CALL JVVXXX(W(1,7   ),W(1,10  ),GWWA ,WMASS   ,WWIDTH  ,W(1,         
     &     11  ))                                                          
      CALL IOVXXX(W(1,6   ),W(1,5   ),W(1,11  ),GWF ,AMP(2   ))            
      CALL JIOXXX(W(1,1   ),W(1,2   ),GZD ,ZMASS   ,ZWIDTH  ,W(1,12  ))    
      CALL JVVXXX(W(1,7   ),W(1,12  ),GWWZ ,WMASS   ,WWIDTH  ,W(1,         
     &     13  ))                                                          
      CALL IOVXXX(W(1,6   ),W(1,5   ),W(1,13  ),GWF ,AMP(3   ))            
      JAMP(   1) = -AMP(   1)-AMP(   2)-AMP(   3)
      MATDDBWWB = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATDDBWWB =MATDDBWWB+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END


C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE UUBWWBORN(P1,ANS)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve e- ve~  
C  
C Crossing   1 is u u~ -> e+ ve e- ve~  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  6)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB=  64, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATUUBWWB
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    3/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   1/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,6) /-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,6) /-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,6) /-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,6) /-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,6) /-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,6) /-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,6) /-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,6) /-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,6) /-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,6) /-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,6) /-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,6) /-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,6) /-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,6) /-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,6) /-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,6) /-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,6) /-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,6) /-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,6) /-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,6) /-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,6) /-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,6) /-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,6) /-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,6) /-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,6) /-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,6) /-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,6) /-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,6) /-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,6) /-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,6) /-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,6) /-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,6) /-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,6) / 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,6) / 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,6) / 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,6) / 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,6) / 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,6) / 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,6) / 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,6) / 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,6) / 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,6) / 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,6) / 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,6) / 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,6) / 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,6) / 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,6) / 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,6) / 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,6) / 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,6) / 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,6) / 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,6) / 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,6) / 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,6) / 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,6) / 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,6) / 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,6) / 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,6) / 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,6) / 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,6) / 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,6) / 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,6) / 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,6) / 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,6) / 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,6) / 1, 2, 3, 4, 5, 6/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  36/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATUUBWWB(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATUUBWWB(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATUUBWWB
      REAL*8 FUNCTION MATUUBWWB(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve e- ve~  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   3,NEIGEN=  1) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  6)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  13, NCOLOR=   1) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS),WX1(6),WX2(6)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            1/                                       
      DATA (CF(i,1  ),i=1  ,1  ) /     3/                                  
C               T[2,1]                                                     
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL OXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),ZERO ,NHEL(5   ),+1*IC(5   ),W(1,5   ))        
      CALL IXXXXX(P(0,6   ),ZERO ,NHEL(6   ),-1*IC(6   ),W(1,6   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,7   ))    
      CALL JIOXXX(W(1,6   ),W(1,5   ),GWF ,WMASS   ,WWIDTH  ,W(1,8   ))    
      CALL FVIXXX(W(1,1   ),W(1,7   ),GWF ,ZERO    ,ZERO    ,W(1,9   ))    
      CALL IOVXXX(W(1,9   ),W(1,2   ),W(1,8   ),GWF ,AMP(1   ))            
      CALL JIOXXX(W(1,1   ),W(1,2   ),GAU ,ZERO    ,AWIDTH  ,W(1,10  ))    
      CALL JVVXXX(W(1,7   ),W(1,10  ),GWWA ,WMASS   ,WWIDTH  ,W(1,         
     &     11  ))                                                          
      CALL IOVXXX(W(1,6   ),W(1,5   ),W(1,11  ),GWF ,AMP(2   ))            
      CALL JIOXXX(W(1,1   ),W(1,2   ),GZU ,ZMASS   ,ZWIDTH  ,W(1,12  ))    
      CALL JVVXXX(W(1,7   ),W(1,12  ),GWWZ ,WMASS   ,WWIDTH  ,W(1,         
     &     13  ))                                                          
      CALL IOVXXX(W(1,6   ),W(1,5   ),W(1,13  ),GWF ,AMP(3   ))            
      JAMP(   1) = -AMP(   1)-AMP(   2)-AMP(   3)
      MATUUBWWB = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATUUBWWB =MATUUBWWB+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END


C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE DDBWWREAL(P1,ANS)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : d d~ -> e+ ve e- ve~ g  
C  
C Crossing   1 is d d~ -> e+ ve e- ve~ g  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 128, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATDDBWWR
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    7/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   1/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,7) /-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,7) /-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,7) /-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,7) /-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,7) /-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,7) /-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,7) /-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,7) /-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,7) /-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,7) /-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,7) /-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,7) /-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,7) /-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,7) /-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,7) /-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,7) /-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,7) /-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,7) /-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,7) /-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,7) /-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,7) /-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,7) /-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,7) /-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,7) /-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,7) /-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,7) /-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,7) /-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,7) /-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,7) /-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,7) /-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,7) /-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,7) /-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,7) /-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,7) /-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,7) /-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,7) /-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,7) /-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,7) /-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,7) /-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,7) /-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,7) /-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,7) /-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,7) /-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,7) /-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,7) /-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,7) /-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,7) /-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,7) /-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,7) /-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,7) /-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,7) /-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,7) /-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,7) /-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,7) /-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,7) /-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,7) /-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,7) /-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,7) /-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,7) /-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,7) /-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,7) /-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,7) /-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,7) /-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,7) /-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,7) / 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,7) / 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,7) / 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,7) / 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,7) / 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,7) / 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,7) / 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,7) / 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,7) / 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,7) / 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,7) / 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,7) / 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,7) / 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,7) / 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,7) / 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,7) / 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,7) / 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,7) / 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,7) / 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,7) / 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,7) / 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,7) / 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,7) / 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,7) / 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,7) / 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,7) / 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,7) / 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,7) / 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,7) / 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,7) / 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,7) / 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,7) / 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,7) / 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,7) / 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,7) / 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,7) / 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,7) / 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,7) / 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,7) / 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,7) / 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,7) / 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,7) / 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,7) / 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,7) / 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,7) / 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,7) / 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,7) / 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,7) / 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,7) / 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,7) / 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,7) / 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,7) / 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,7) / 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,7) / 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,7) / 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,7) / 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,7) / 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,7) / 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,7) / 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,7) / 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,7) / 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,7) / 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,7) / 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,7) / 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,7) / 1, 2, 3, 4, 5, 6, 7/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  36/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATDDBWWR(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATDDBWWR(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATDDBWWR
      REAL*8 FUNCTION MATDDBWWR(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : d d~ -> e+ ve e- ve~ g  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   7,NEIGEN=  1) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  18, NCOLOR=   1) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS),WX1(6),WX2(6)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            1/                                       
      DATA (CF(i,1  ),i=1  ,1  ) /     4/                                  
C               T[2,1,7]                                                   
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL OXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),ZERO ,NHEL(5   ),+1*IC(5   ),W(1,5   ))        
      CALL IXXXXX(P(0,6   ),ZERO ,NHEL(6   ),-1*IC(6   ),W(1,6   ))        
      CALL VXXXXX(P(0,7   ),ZERO ,NHEL(7   ),+1*IC(7   ),W(1,7   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,8   ))    
      CALL JIOXXX(W(1,6   ),W(1,5   ),GWF ,WMASS   ,WWIDTH  ,W(1,9   ))    
      CALL FVIXXX(W(1,1   ),W(1,7   ),GG ,ZERO    ,ZERO    ,W(1,10  ))     
      CALL FVOXXX(W(1,2   ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,11  ))    
      CALL IOVXXX(W(1,10  ),W(1,11  ),W(1,9   ),GWF ,AMP(1   ))            
      CALL FVOXXX(W(1,2   ),W(1,7   ),GG ,ZERO    ,ZERO    ,W(1,12  ))     
      CALL FVOXXX(W(1,12  ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,13  ))    
      CALL IOVXXX(W(1,1   ),W(1,13  ),W(1,9   ),GWF ,AMP(2   ))            
      CALL FVIXXX(W(1,1   ),W(1,9   ),GWF ,ZERO    ,ZERO    ,W(1,14  ))    
      CALL IOVXXX(W(1,14  ),W(1,11  ),W(1,7   ),GG ,AMP(3   ))             
      CALL JIOXXX(W(1,10  ),W(1,2   ),GAD ,ZERO    ,AWIDTH  ,W(1,15  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,15  ),GWWA ,AMP(4   ))           
      CALL JIOXXX(W(1,10  ),W(1,2   ),GZD ,ZMASS   ,ZWIDTH  ,W(1,16  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,16  ),GWWZ ,AMP(5   ))           
      CALL JIOXXX(W(1,1   ),W(1,12  ),GAD ,ZERO    ,AWIDTH  ,W(1,17  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,17  ),GWWA ,AMP(6   ))           
      CALL JIOXXX(W(1,1   ),W(1,12  ),GZD ,ZMASS   ,ZWIDTH  ,W(1,18  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,18  ),GWWZ ,AMP(7   ))           
      JAMP(   1) = -AMP(   1)-AMP(   2)-AMP(   3)-AMP(   4)-AMP(   5)
     &             -AMP(   6)-AMP(   7)
      MATDDBWWR = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATDDBWWR =MATDDBWWR+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END


C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE UUBWWREAL(P1,ANS)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve e- ve~ g  
C  
C Crossing   1 is u u~ -> e+ ve e- ve~ g  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 128, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATUUBWWR
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    7/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   1/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,7) /-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,7) /-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,7) /-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,7) /-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,7) /-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,7) /-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,7) /-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,7) /-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,7) /-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,7) /-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,7) /-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,7) /-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,7) /-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,7) /-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,7) /-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,7) /-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,7) /-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,7) /-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,7) /-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,7) /-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,7) /-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,7) /-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,7) /-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,7) /-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,7) /-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,7) /-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,7) /-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,7) /-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,7) /-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,7) /-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,7) /-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,7) /-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,7) /-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,7) /-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,7) /-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,7) /-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,7) /-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,7) /-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,7) /-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,7) /-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,7) /-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,7) /-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,7) /-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,7) /-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,7) /-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,7) /-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,7) /-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,7) /-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,7) /-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,7) /-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,7) /-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,7) /-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,7) /-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,7) /-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,7) /-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,7) /-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,7) /-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,7) /-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,7) /-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,7) /-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,7) /-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,7) /-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,7) /-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,7) /-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,7) / 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,7) / 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,7) / 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,7) / 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,7) / 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,7) / 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,7) / 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,7) / 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,7) / 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,7) / 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,7) / 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,7) / 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,7) / 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,7) / 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,7) / 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,7) / 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,7) / 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,7) / 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,7) / 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,7) / 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,7) / 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,7) / 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,7) / 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,7) / 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,7) / 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,7) / 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,7) / 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,7) / 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,7) / 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,7) / 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,7) / 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,7) / 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,7) / 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,7) / 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,7) / 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,7) / 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,7) / 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,7) / 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,7) / 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,7) / 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,7) / 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,7) / 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,7) / 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,7) / 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,7) / 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,7) / 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,7) / 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,7) / 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,7) / 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,7) / 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,7) / 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,7) / 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,7) / 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,7) / 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,7) / 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,7) / 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,7) / 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,7) / 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,7) / 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,7) / 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,7) / 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,7) / 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,7) / 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,7) / 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,7) / 1, 2, 3, 4, 5, 6, 7/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  36/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATUUBWWR(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATUUBWWR(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATUUBWWR
      REAL*8 FUNCTION MATUUBWWR(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : u u~ -> e+ ve e- ve~ g  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   7,NEIGEN=  1) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  18, NCOLOR=   1) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS),WX1(6),WX2(6)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            1/                                       
      DATA (CF(i,1  ),i=1  ,1  ) /     4/                                  
C               T[2,1,7]                                                   
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL OXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),ZERO ,NHEL(5   ),+1*IC(5   ),W(1,5   ))        
      CALL IXXXXX(P(0,6   ),ZERO ,NHEL(6   ),-1*IC(6   ),W(1,6   ))        
      CALL VXXXXX(P(0,7   ),ZERO ,NHEL(7   ),+1*IC(7   ),W(1,7   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,8   ))    
      CALL JIOXXX(W(1,6   ),W(1,5   ),GWF ,WMASS   ,WWIDTH  ,W(1,9   ))    
      CALL FVIXXX(W(1,1   ),W(1,7   ),GG ,ZERO    ,ZERO    ,W(1,10  ))     
      CALL FVIXXX(W(1,10  ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,11  ))    
      CALL IOVXXX(W(1,11  ),W(1,2   ),W(1,9   ),GWF ,AMP(1   ))            
      CALL FVOXXX(W(1,2   ),W(1,7   ),GG ,ZERO    ,ZERO    ,W(1,12  ))     
      CALL FVIXXX(W(1,1   ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,13  ))    
      CALL IOVXXX(W(1,13  ),W(1,12  ),W(1,9   ),GWF ,AMP(2   ))            
      CALL FVOXXX(W(1,2   ),W(1,9   ),GWF ,ZERO    ,ZERO    ,W(1,14  ))    
      CALL IOVXXX(W(1,13  ),W(1,14  ),W(1,7   ),GG ,AMP(3   ))             
      CALL JIOXXX(W(1,10  ),W(1,2   ),GAU ,ZERO    ,AWIDTH  ,W(1,15  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,15  ),GWWA ,AMP(4   ))           
      CALL JIOXXX(W(1,10  ),W(1,2   ),GZU ,ZMASS   ,ZWIDTH  ,W(1,16  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,16  ),GWWZ ,AMP(5   ))           
      CALL JIOXXX(W(1,1   ),W(1,12  ),GAU ,ZERO    ,AWIDTH  ,W(1,17  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,17  ),GWWA ,AMP(6   ))           
      CALL JIOXXX(W(1,1   ),W(1,12  ),GZU ,ZMASS   ,ZWIDTH  ,W(1,18  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,18  ),GWWZ ,AMP(7   ))           
      JAMP(   1) = -AMP(   1)-AMP(   2)-AMP(   3)-AMP(   4)-AMP(   5)
     &             -AMP(   6)-AMP(   7)
      MATUUBWWR = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATUUBWWR =MATUUBWWR+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END


C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE DGWWREAL(P1,ANS)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : d g -> e+ ve e- ve~ d  
C  
C Crossing   1 is d g -> e+ ve e- ve~ d  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 128, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATDGWWR
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    7/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   1/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,7) /-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,7) /-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,7) /-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,7) /-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,7) /-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,7) /-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,7) /-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,7) /-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,7) /-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,7) /-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,7) /-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,7) /-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,7) /-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,7) /-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,7) /-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,7) /-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,7) /-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,7) /-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,7) /-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,7) /-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,7) /-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,7) /-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,7) /-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,7) /-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,7) /-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,7) /-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,7) /-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,7) /-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,7) /-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,7) /-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,7) /-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,7) /-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,7) /-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,7) /-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,7) /-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,7) /-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,7) /-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,7) /-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,7) /-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,7) /-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,7) /-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,7) /-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,7) /-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,7) /-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,7) /-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,7) /-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,7) /-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,7) /-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,7) /-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,7) /-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,7) /-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,7) /-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,7) /-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,7) /-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,7) /-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,7) /-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,7) /-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,7) /-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,7) /-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,7) /-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,7) /-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,7) /-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,7) /-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,7) /-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,7) / 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,7) / 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,7) / 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,7) / 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,7) / 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,7) / 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,7) / 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,7) / 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,7) / 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,7) / 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,7) / 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,7) / 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,7) / 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,7) / 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,7) / 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,7) / 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,7) / 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,7) / 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,7) / 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,7) / 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,7) / 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,7) / 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,7) / 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,7) / 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,7) / 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,7) / 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,7) / 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,7) / 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,7) / 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,7) / 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,7) / 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,7) / 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,7) / 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,7) / 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,7) / 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,7) / 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,7) / 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,7) / 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,7) / 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,7) / 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,7) / 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,7) / 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,7) / 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,7) / 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,7) / 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,7) / 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,7) / 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,7) / 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,7) / 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,7) / 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,7) / 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,7) / 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,7) / 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,7) / 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,7) / 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,7) / 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,7) / 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,7) / 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,7) / 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,7) / 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,7) / 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,7) / 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,7) / 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,7) / 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,7) / 1, 2, 3, 4, 5, 6, 7/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  96/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATDGWWR(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATDGWWR(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATDGWWR
      REAL*8 FUNCTION MATDGWWR(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : d g -> e+ ve e- ve~ d  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   7,NEIGEN=  1) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  18, NCOLOR=   1) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS),WX1(6),WX2(6)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            1/                                       
      DATA (CF(i,1  ),i=1  ,1  ) /     4/                                  
C               T[7,1,2]                                                   
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL VXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),ZERO ,NHEL(5   ),+1*IC(5   ),W(1,5   ))        
      CALL IXXXXX(P(0,6   ),ZERO ,NHEL(6   ),-1*IC(6   ),W(1,6   ))        
      CALL OXXXXX(P(0,7   ),ZERO ,NHEL(7   ),+1*IC(7   ),W(1,7   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,8   ))    
      CALL JIOXXX(W(1,6   ),W(1,5   ),GWF ,WMASS   ,WWIDTH  ,W(1,9   ))    
      CALL FVOXXX(W(1,7   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,10  ))     
      CALL FVOXXX(W(1,10  ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,11  ))    
      CALL IOVXXX(W(1,1   ),W(1,11  ),W(1,9   ),GWF ,AMP(1   ))            
      CALL FVOXXX(W(1,7   ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,12  ))    
      CALL FVOXXX(W(1,12  ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,13  ))     
      CALL IOVXXX(W(1,1   ),W(1,13  ),W(1,9   ),GWF ,AMP(2   ))            
      CALL JIOXXX(W(1,1   ),W(1,10  ),GAD ,ZERO    ,AWIDTH  ,W(1,14  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,14  ),GWWA ,AMP(3   ))           
      CALL JIOXXX(W(1,1   ),W(1,10  ),GZD ,ZMASS   ,ZWIDTH  ,W(1,15  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,15  ),GWWZ ,AMP(4   ))           
      CALL FVIXXX(W(1,1   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,16  ))     
      CALL IOVXXX(W(1,16  ),W(1,12  ),W(1,9   ),GWF ,AMP(5   ))            
      CALL JIOXXX(W(1,16  ),W(1,7   ),GAD ,ZERO    ,AWIDTH  ,W(1,17  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,17  ),GWWA ,AMP(6   ))           
      CALL JIOXXX(W(1,16  ),W(1,7   ),GZD ,ZMASS   ,ZWIDTH  ,W(1,18  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,18  ),GWWZ ,AMP(7   ))           
      JAMP(   1) = -AMP(   1)-AMP(   2)-AMP(   3)-AMP(   4)-AMP(   5)
     &             -AMP(   6)-AMP(   7)
      MATDGWWR = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATDGWWR =MATDGWWR+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END


C SF: The following routine is SMATRIX generated by MadEvent, suitably modified
      SUBROUTINE UGWWREAL(P1,ANS)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL)
C  
C FOR PROCESS : u g -> e+ ve e- ve~ u  
C  
C Crossing   1 is u g -> e+ ve e- ve~ u  
      IMPLICIT NONE
C  
C CONSTANTS
C  
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB= 128, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T, P(0:3,NEXTERNAL)
      REAL*8 MATUGWWR
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL), I
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGRAPHS
      REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER idum, ngood, igood(ncomb), jhel, j
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2

      character*79         hel_buff
C SF: comment out all common blocks
c      common/to_helicity/  hel_buff

      integer          isum_hel
      logical                    multi_channel
C SF: comment out all common blocks
c      common/to_matrix/isum_hel, multi_channel
C SF: comment out all instances of mapconfig, used by multi_channel
c      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
c      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY,IDUM /0,-1/
      DATA xtry, xrej, ngood /0,0,0/
      DATA warned, isum_hel/.false.,0/
      DATA multi_channel/.true./
      SAVE yfrac, igood, IDUM, jhel
      DATA NGRAPHS /    7/          
C SF: comment out all instances of amp2, used by multi_channel
c      DATA jamp2(0) /   1/          
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,7) /-1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,7) /-1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,7) /-1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,7) /-1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,7) /-1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,7) /-1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,7) /-1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,7) /-1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,7) /-1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,7) /-1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,7) /-1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,7) /-1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,7) /-1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,7) /-1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,7) /-1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,7) /-1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,7) /-1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,7) /-1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,7) /-1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,7) /-1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,7) /-1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,7) /-1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,7) /-1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,7) /-1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,7) /-1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,7) /-1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,7) /-1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,7) /-1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,7) /-1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,7) /-1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,7) /-1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,7) /-1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  33),IHEL=1,7) /-1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  34),IHEL=1,7) /-1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  35),IHEL=1,7) /-1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  36),IHEL=1,7) /-1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  37),IHEL=1,7) /-1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  38),IHEL=1,7) /-1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  39),IHEL=1,7) /-1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  40),IHEL=1,7) /-1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  41),IHEL=1,7) /-1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  42),IHEL=1,7) /-1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  43),IHEL=1,7) /-1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  44),IHEL=1,7) /-1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  45),IHEL=1,7) /-1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  46),IHEL=1,7) /-1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  47),IHEL=1,7) /-1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  48),IHEL=1,7) /-1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  49),IHEL=1,7) /-1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  50),IHEL=1,7) /-1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  51),IHEL=1,7) /-1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  52),IHEL=1,7) /-1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  53),IHEL=1,7) /-1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  54),IHEL=1,7) /-1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  55),IHEL=1,7) /-1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  56),IHEL=1,7) /-1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  57),IHEL=1,7) /-1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  58),IHEL=1,7) /-1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  59),IHEL=1,7) /-1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  60),IHEL=1,7) /-1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  61),IHEL=1,7) /-1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  62),IHEL=1,7) /-1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  63),IHEL=1,7) /-1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  64),IHEL=1,7) /-1, 1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  65),IHEL=1,7) / 1,-1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  66),IHEL=1,7) / 1,-1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  67),IHEL=1,7) / 1,-1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  68),IHEL=1,7) / 1,-1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  69),IHEL=1,7) / 1,-1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  70),IHEL=1,7) / 1,-1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  71),IHEL=1,7) / 1,-1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  72),IHEL=1,7) / 1,-1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  73),IHEL=1,7) / 1,-1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  74),IHEL=1,7) / 1,-1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  75),IHEL=1,7) / 1,-1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  76),IHEL=1,7) / 1,-1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  77),IHEL=1,7) / 1,-1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  78),IHEL=1,7) / 1,-1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  79),IHEL=1,7) / 1,-1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  80),IHEL=1,7) / 1,-1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  81),IHEL=1,7) / 1,-1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  82),IHEL=1,7) / 1,-1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  83),IHEL=1,7) / 1,-1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  84),IHEL=1,7) / 1,-1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  85),IHEL=1,7) / 1,-1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  86),IHEL=1,7) / 1,-1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  87),IHEL=1,7) / 1,-1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  88),IHEL=1,7) / 1,-1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  89),IHEL=1,7) / 1,-1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  90),IHEL=1,7) / 1,-1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  91),IHEL=1,7) / 1,-1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  92),IHEL=1,7) / 1,-1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  93),IHEL=1,7) / 1,-1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  94),IHEL=1,7) / 1,-1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  95),IHEL=1,7) / 1,-1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  96),IHEL=1,7) / 1,-1, 1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  97),IHEL=1,7) / 1, 1,-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  98),IHEL=1,7) / 1, 1,-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  99),IHEL=1,7) / 1, 1,-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 100),IHEL=1,7) / 1, 1,-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 101),IHEL=1,7) / 1, 1,-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 102),IHEL=1,7) / 1, 1,-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 103),IHEL=1,7) / 1, 1,-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 104),IHEL=1,7) / 1, 1,-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 105),IHEL=1,7) / 1, 1,-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 106),IHEL=1,7) / 1, 1,-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 107),IHEL=1,7) / 1, 1,-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 108),IHEL=1,7) / 1, 1,-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 109),IHEL=1,7) / 1, 1,-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 110),IHEL=1,7) / 1, 1,-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 111),IHEL=1,7) / 1, 1,-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 112),IHEL=1,7) / 1, 1,-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL, 113),IHEL=1,7) / 1, 1, 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL, 114),IHEL=1,7) / 1, 1, 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL, 115),IHEL=1,7) / 1, 1, 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL, 116),IHEL=1,7) / 1, 1, 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL, 117),IHEL=1,7) / 1, 1, 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL, 118),IHEL=1,7) / 1, 1, 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL, 119),IHEL=1,7) / 1, 1, 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL, 120),IHEL=1,7) / 1, 1, 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL, 121),IHEL=1,7) / 1, 1, 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL, 122),IHEL=1,7) / 1, 1, 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL, 123),IHEL=1,7) / 1, 1, 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL, 124),IHEL=1,7) / 1, 1, 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL, 125),IHEL=1,7) / 1, 1, 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL, 126),IHEL=1,7) / 1, 1, 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL, 127),IHEL=1,7) / 1, 1, 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL, 128),IHEL=1,7) / 1, 1, 1, 1, 1, 1, 1/
      DATA (  IC(IHEL,  1),IHEL=1,7) / 1, 2, 3, 4, 5, 6, 7/
      DATA (IDEN(IHEL),IHEL=  1,  1) /  96/
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
      CALL SWITCHMOM(P1,P,IC(1,IPROC),JC,NEXTERNAL)
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
       
C SF: comment out all instances of multi_channel
c      IF (multi_channel) THEN
c          DO IHEL=1,NGRAPHS
c              amp2(ihel)=0d0
c              jamp2(ihel)=0d0
c          ENDDO
c          DO IHEL=1,int(jamp2(0))
c              jamp2(ihel)=0d0
c          ENDDO
c      ENDIF
      ANS(IPROC) = 0D0
      write(hel_buff,'(16i5)') (0,i=1,nexternal)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
          DO IHEL=1,NCOMB
              IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
                 T=MATUGWWR(P ,NHEL(1,IHEL),JC(1))            
                 ANS(IPROC)=ANS(IPROC)+T
                  IF (T .GT. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                      GOODHEL(IHEL,IPROC)=.TRUE.
                      NGOOD = NGOOD +1
                      IGOOD(NGOOD) = IHEL
C                WRITE(*,*) ngood,IHEL,T
                  ENDIF
              ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
              IHEL = IGOOD(JHEL)
              T=MATUGWWR(P ,NHEL(1,IHEL),JC(1))            
           ANS(IPROC)=ANS(IPROC)+T*HWGT
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
          ENDIF
      ENDIF
C SF: comment out all instances of multi_channel
c      IF (MULTI_CHANNEL) THEN
c          XTOT=0D0
c          DO IHEL=1,MAPCONFIG(0)
c              XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
c          ENDDO
c          ANS(IPROC)=ANS(IPROC)*AMP2(MAPCONFIG(ICONFIG))/XTOT
c      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
C SF: the original name MATRIX has been replaced by MATUGWWR
      REAL*8 FUNCTION MATUGWWR(P,NHEL,IC)
C  
C Generated by MadGraph II Version 3.0. Updated 02/19/04                
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL)
C  
C FOR PROCESS : u g -> e+ ve e- ve~ u  
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   7,NEIGEN=  1) 
C SF: replace the following include with the explicit inclusion of
c nexternal.inc, originally included by genps.inc
c      Include "genps.inc"
      integer    nexternal
      parameter (nexternal=  7)
      INTEGER    NWAVEFUNCS     , NCOLOR
      PARAMETER (NWAVEFUNCS=  18, NCOLOR=   1) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(6,NWAVEFUNCS),WX1(6),WX2(6)
C  
C GLOBAL VARIABLES
C  
C SF: comment out all instances of amp2, used by multi_channel
c      Double Precision amp2(maxamps), jamp2(0:maxamps)
c      common/to_amps/  amp2,       jamp2
C SF: The original coupl.inc has been renamed MEcoupl.inc
      include "MEcoupl.inc"
C  
C COLOR DATA
C  
      DATA Denom(1  )/            1/                                       
      DATA (CF(i,1  ),i=1  ,1  ) /     4/                                  
C               T[7,1,2]                                                   
C ----------
C BEGIN CODE
C ----------
      CALL IXXXXX(P(0,1   ),ZERO ,NHEL(1   ),+1*IC(1   ),W(1,1   ))        
      CALL VXXXXX(P(0,2   ),ZERO ,NHEL(2   ),-1*IC(2   ),W(1,2   ))        
      CALL IXXXXX(P(0,3   ),ZERO ,NHEL(3   ),-1*IC(3   ),W(1,3   ))        
      CALL OXXXXX(P(0,4   ),ZERO ,NHEL(4   ),+1*IC(4   ),W(1,4   ))        
      CALL OXXXXX(P(0,5   ),ZERO ,NHEL(5   ),+1*IC(5   ),W(1,5   ))        
      CALL IXXXXX(P(0,6   ),ZERO ,NHEL(6   ),-1*IC(6   ),W(1,6   ))        
      CALL OXXXXX(P(0,7   ),ZERO ,NHEL(7   ),+1*IC(7   ),W(1,7   ))        
      CALL JIOXXX(W(1,3   ),W(1,4   ),GWF ,WMASS   ,WWIDTH  ,W(1,8   ))    
      CALL JIOXXX(W(1,6   ),W(1,5   ),GWF ,WMASS   ,WWIDTH  ,W(1,9   ))    
      CALL FVOXXX(W(1,7   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,10  ))     
      CALL FVIXXX(W(1,1   ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,11  ))    
      CALL IOVXXX(W(1,11  ),W(1,10  ),W(1,9   ),GWF ,AMP(1   ))            
      CALL FVOXXX(W(1,7   ),W(1,9   ),GWF ,ZERO    ,ZERO    ,W(1,12  ))    
      CALL IOVXXX(W(1,11  ),W(1,12  ),W(1,2   ),GG ,AMP(2   ))             
      CALL JIOXXX(W(1,1   ),W(1,10  ),GAU ,ZERO    ,AWIDTH  ,W(1,13  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,13  ),GWWA ,AMP(3   ))           
      CALL JIOXXX(W(1,1   ),W(1,10  ),GZU ,ZMASS   ,ZWIDTH  ,W(1,14  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,14  ),GWWZ ,AMP(4   ))           
      CALL FVIXXX(W(1,1   ),W(1,2   ),GG ,ZERO    ,ZERO    ,W(1,15  ))     
      CALL JIOXXX(W(1,15  ),W(1,7   ),GAU ,ZERO    ,AWIDTH  ,W(1,16  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,16  ),GWWA ,AMP(5   ))           
      CALL JIOXXX(W(1,15  ),W(1,7   ),GZU ,ZMASS   ,ZWIDTH  ,W(1,17  ))    
      CALL VVVXXX(W(1,9   ),W(1,8   ),W(1,17  ),GWWZ ,AMP(6   ))           
      CALL FVIXXX(W(1,15  ),W(1,8   ),GWF ,ZERO    ,ZERO    ,W(1,18  ))    
      CALL IOVXXX(W(1,18  ),W(1,7   ),W(1,9   ),GWF ,AMP(7   ))            
      JAMP(   1) = -AMP(   1)-AMP(   2)-AMP(   3)-AMP(   4)-AMP(   5)
     &             -AMP(   6)-AMP(   7)
      MATUGWWR = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATUGWWR =MATUGWWR+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
C SF: comment out all instances of amp2, used by multi_channel
c      Do I = 1, NGRAPHS
c          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
c      Enddo
c      Do I = 1, NCOLOR
c          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
c      Enddo
C      CALL GAUGECHECK(JAMP,ZTEMP,EIGEN_VEC,EIGEN_VAL,NCOLOR,NEIGEN) 
      END
c
c
c End of lepton matrix elements
c
c
