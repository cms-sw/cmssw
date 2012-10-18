c----------------------------------------------------------------------
      double precision function xDfit(zz,i1,i2,s,xp,xm,b)
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incsem'



      xDfit=0.d0
      do i=max(0,i1),i2
        call GfunPar(zz,zz,1,i,b,s,alp,bet,betp,epsp,epst,epss,gamv)
        if(i1.ge.0)then
          corp=alppar-epsp
          cort=alppar-epst
          cors=-epss
        else
          corp=alppar
          cort=alppar
          cors=0.
        endif
c        write(ifch,*)'xdfit',i,zz,b,s,alp,bet,betp,epsp,epst,epss
        xDfit=xDfit+dble(alp*xp**(bet+corp)*xm**(betp+cort)*s**cors)
      enddo
      return
      end


c----------------------------------------------------------------------
      subroutine xFitD1
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision x,y,Dsoftshval,om51p,xminr,tmp,xtmp,xDfit
      character chenergy*12

      nptg=50                   !number of point for the graphs
      iii=nint(xpar1)
      biniDf=xpar2                   !value of biniDf (impact parameter)
      y=dble(xpar3)                    !value of y (rapidity)
      xtmp=xmaxDf
      xmaxDf=dexp(-2.d0*y)
      zz=xpar7

      chenergy='E=          '
      if (engy.ge.10000.) then
        write(chenergy(4:8),'(I5)')int(engy)
        ke=10
      elseif (engy.ge.1000.) then
        write(chenergy(4:7),'(I4)')int(engy)
        ke=9
      elseif (engy.ge.100.) then
        write(chenergy(4:6),'(I3)')int(engy)
        ke=8
      elseif (engy.ge.10.) then
        write(chenergy(4:5),'(I2)')int(engy)
        ke=7
      else
        write(chenergy(4:4),'(I1)')int(engy)
        ke=6
      endif
      chenergy(ke:ke+2)='GeV'

      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

      if(iii/10.eq.1)then !...................................................

      write(ifhi,'(a)')'!----------------------------------------------'
      write(ifhi,'(a)')'!     D exact all      (blue)      '
      write(ifhi,'(a)')'!----------------------------------------------'

      write(ifhi,'(a)')'openhisto name DExact-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,a)')    'text 0.65 0.9 "exact" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        tmp=Dsoftshval(real(x)*smaxDf,x,y,biniDf,0)
        write(ifhi,*) x,tmp
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


      write(ifhi,'(a)')'!----------------------------------------------'
      write(ifhi,'(a)')'!     D exact soft      (red dot)      '
      write(ifhi,'(a)')'!----------------------------------------------'

      write(ifhi,'(a)')'openhisto name DExactSoft-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lra'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,2.d0*om51p(real(x)*smaxDf,x,y,biniDf,0)
     &       /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     D exact sea-sea      (yellow-dot)     '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name DExactSemi-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lya'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        tmp=2.d0*om51p(real(x)*smaxDf,x,y,biniDf,1)
     &       /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
        write(ifhi,*) x,tmp
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     D exact semi      (blue dot)          '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name DExactVal-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lba'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        tmp=2.d0*(om51p(real(x)*smaxDf,x,y,biniDf,2)
     &       +om51p(real(x)*smaxDf,x,y,biniDf,1)
     &       +om51p(real(x)*smaxDf,x,y,biniDf,3)
     &       +om51p(real(x)*smaxDf,x,y,biniDf,4))
     &       /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
        write(ifhi,*) x,tmp
      enddo

      write(ifhi,'(a)')    '  endarray'

      if(iii.eq.11)then
        write(ifhi,'(a)')    'closehisto plot 0-'
      else
        write(ifhi,'(a)')    'closehisto plot 0'
      endif

      endif !................................................................
      if(mod(iii,10).eq.1)then !.............................................

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     D exact all      (blue)              '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name DExact-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,a)')    'text 0.65 0.9 "exact+fit" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        tmp=Dsoftshval(real(x)*smaxDf,x,y,biniDf,0)
        write(ifhi,*) x,tmp
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     D exact all      (green)              '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name DExact-f-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lga'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,a)')    'text 0.65 0.9 "exact+fit" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        tmp=Dsoftshval(real(x)*smaxDf,x,y,biniDf,-1)
        write(ifhi,*) x,tmp
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     fit soft      (red dot)             '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lro'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      write(ifhi,'(a)')       'array 2'


      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
        tmp=xDfit(zz,0,0,smaxDf,xp,xm,biniDf)
        write(ifhi,*) x,tmp
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     fit semi      (blue dot)              '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lbo'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,,s,b)" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      write(ifhi,'(a)')       'array 2'


      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
        tmp=xDfit(zz,1,1,smaxDf,xp,xm,biniDf)
        write(ifhi,*) x,tmp
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     fit all      (red)      '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,,s,b)" '
      write(ifhi,'(3a)')          'text 0.05 0.9 "',chenergy,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.05 0.8 "b=',biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  'text 0.05 0.7 "y=',y,'"'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
        tmp=xDfit(zz,0,1,smaxDf,xp,xm,biniDf)
        write(ifhi,*) x,tmp
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      endif !...............................................................

      xmaxDf=xtmp

      end

c----------------------------------------------------------------------
      subroutine xFitD2
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

      double precision x,om51p,xDfit,z(0:200),xminr,y,xtmp,om,om5,om51
c     & ,omYuncut,omNpuncut
      character chenergy*12,chf*3,texte*15,textb*17,texty*17

      nptg=30                  !number of point for the graphs
      biniDf=xpar2                 !value of biniDf (impact parameter)
      y=dble(xpar3)                 !value of y (rapidity)
      jj1=nint(xpar4)
      jj2=nint(xpar5)
      if(jj1.ne.1.and.jj1.ne.2)jj1=3
      if(jj2.ne.1.and.jj2.ne.2)jj2=3
      zz=xpar7
      xtmp=xmaxDf
      xmaxDf=dexp(-2.d0*y)

      chenergy='E=          '
      if (engy.ge.10000.) then
        write(chenergy(4:8),'(I5)')int(engy)
        ke=10
      elseif (engy.ge.1000.) then
        write(chenergy(4:7),'(I4)')int(engy)
        ke=9
      elseif (engy.ge.100.) then
        write(chenergy(4:6),'(I3)')int(engy)
        ke=8
      elseif (engy.ge.10.) then
        write(chenergy(4:5),'(I2)')int(engy)
        ke=7
      else
        write(chenergy(4:4),'(I1)')int(engy)
        ke=6
      endif
      chenergy(ke:ke+2)='GeV'

      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

         do jj=jj1,jj2

      if(jj.eq.1)chf='  D'
      if(jj.eq.2)chf='  G'
      if(jj.eq.3)chf='FFG'
      texte='text 0.05 0.9 "'
      textb='text 0.05 0.8 "b='
      texty='text 0.05 0.7 "y='
      if(jj.eq.2)texty='text 0.15 0.7 "y='
      if(jj.eq.3)texte='text 0.05 0.3 "'
      if(jj.eq.3)textb='text 0.05 0.2 "b='
      if(jj.eq.3)texty='text 0.05 0.1 "y='

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     '//chf//' exact all      (green)         '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name '//chf//'ExaI-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lga'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a)')      'text 0 0 "yaxis '//chf//'(x+,x-,s,b)" '
      write(ifhi,'(a,a)')    'text 0.65 0.9 "exact+fit" '
      write(ifhi,'(3a)')        texte,chenergy,'"'
      write(ifhi,'(a,f5.2,a)')  textb,biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  texty,y,'"'
      write(ifhi,'(a)')       'array 2'
      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
        om=0
        do j=0,4
         om=om+om51p(engy**2*real(x),x,y,biniDf,j)
        enddo
        om=2.d0*om
        if(jj.eq.1)om=om/(x**dble(-alppar))
        if(jj.eq.3)om=om
     &               *(1-xm)**alplea(icltar)*(1-xp)**alplea(iclpro)
        write(ifhi,*) x,om
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     '//chf//' exact all +diff (blue)        '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name '//chf//'ExaD-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a)')      'text 0 0 "yaxis '//chf//'(x+,x-,s,b)" '
      write(ifhi,'(3a)')        texte,chenergy,'"'
      write(ifhi,'(a,f5.2,a)')  textb,biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  texty,y,'"'
      write(ifhi,'(a)')       'array 2'
      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
        om=0
        do j=0,4
         om=om+om51p(engy**2*real(x),x,y,biniDf,j)
        enddo
        om5=om51(x,y,biniDf,5,5)
        om=2.d0*(om+om5)
        if(jj.eq.1)om=om/(x**dble(-alppar))
        if(jj.eq.3)om=om
     &               *(1-xm)**alplea(icltar)*(1-xp)**alplea(iclpro)
        write(ifhi,*) x,om
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     '//chf//' param all      (red)          '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name '//chf//'Par-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lru'
        write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a)')      'text 0 0 "yaxis '//chf//'(x+,x-,s,b)" '
      write(ifhi,'(3a)')        texte,chenergy,'"'
      write(ifhi,'(a,f5.2,a)')  textb,biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  texty,y,'"'
      write(ifhi,'(a)')       'array 2'
      imax=idxD1
      if(iomega.eq.2)imax=1
      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
          z(i)=xDfit(zz,0,imax,engy**2,xp,xm,biniDf)
         if(jj.ge.2)z(i)=z(i)*(x**dble(-alppar))
         if(jj.eq.3)z(i)=z(i)
     &               *(1-xm)**alplea(icltar)*(1-xp)**alplea(iclpro)
         write(ifhi,*) x,z(i)
       enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')'!---------------------------------------------'
      write(ifhi,'(a)')'!     '//chf//' param all      (yellow)       '
      write(ifhi,'(a)')'!---------------------------------------------'

      write(ifhi,'(a)')'openhisto name '//chf//'Scr-'//chenergy(4:ke-2)
      write(ifhi,'(a)')       'htyp lyi'
        write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a)')      'text 0 0 "yaxis '//chf//'(x+,x-,s,b)" '
      write(ifhi,'(3a)')        texte,chenergy,'"'
      write(ifhi,'(a,f5.2,a)')  textb,biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  texty,y,'"'
      write(ifhi,'(a)')       'array 2'
      imax=idxD1
      if(iomega.eq.2)imax=1
      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
          z(i)=xDfit(zz,-1,imax,engy**2,xp,xm,biniDf)
         if(jj.ge.2)z(i)=z(i)*(x**dble(-alppar))
         if(jj.eq.3)z(i)=z(i)
     &               *(1-xm)**alplea(icltar)*(1-xp)**alplea(iclpro)
         write(ifhi,*) x,z(i)
       enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

         enddo

      xmaxDf=xtmp

      end

c----------------------------------------------------------------------
      subroutine xbExaD
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

      double precision x,y,Dsoftshval,om51p,z,xDfit!,omNpuncut


      nptg=50     !number of point for the graphs
      bmax=xpar2
      bmax=max(0.1,bmax)
             !value max of b (impact parameter)
      y=dble(xpar3)                    !value of y (rapidity)
      x=dble(xpar4)
      zz=xpar7
      if(x.eq.0.d0)x=1.d0

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DExactb-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DExactb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DExactb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DExactb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DExactb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
      write(ifhi,'(a)')       'yrange .00001 auto'
c      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=Dsoftshval(real(x)*smaxDf,x,y,b,0)
        write(ifhi,*) b,z
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name DParamb-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name DParamb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name DParamb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name DParamb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name DParamb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lrd'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      imax=idxD1
      if(iomega.eq.2)imax=1
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
          z=xDfit(zz,0,imax,smaxDf,
     &       real(dsqrt(x)*dexp(y)),real(dsqrt(x)*dexp(-y)),b)
        write(ifhi,*) b,z
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DExactSoftb-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DExactSoftb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DExactSoftb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DExactSoftb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DExactSoftb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=2.d0*om51p(real(x)*smaxDf,x,y,b,0)
     &       /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
        write(ifhi,*) b,z
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DExactSemib-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DExactSemib-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DExactSemib-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DExactSemib-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DExactSemib-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pft'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=2.d0*om51p(real(x)*smaxDf,x,y,b,1)
     &       /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
        write(ifhi,*) b,z
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DExactValb-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DExactValb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DExactValb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DExactValb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DExactValb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfs'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=2.d0*(om51p(real(x)*smaxDf,x,y,b,2)+
     &  om51p(real(x)*smaxDf,x,y,b,3)+om51p(real(x)*smaxDf,x,y,b,4))
     &           /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
        write(ifhi,*) b,z
      enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0'


      end


c----------------------------------------------------------------------
      subroutine xbnExaD
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,y,Dsoftshval,z,xDfit,Dint



      nptg=50     !number of point for the graphs
      bmax=xpar2
      bmax=max(0.1,bmax)
      y=dble(xpar2)                   !value of y (rapidity)
      x=dble(xpar3)
      zz=xpar7

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DExactbn-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DExactbn-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DExactbn-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DExactbn-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DExactbn-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      Dint=Dsoftshval(real(x)*smaxDf,x,y,0.,0)
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=Dsoftshval(real(x)*smaxDf,x,y,b,0)/Dint
        write(ifhi,*) b,z
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name DParambn-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name DParambn-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name DParambn-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name DParambn-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name DParambn-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lrd'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      imax=idxD1
      if(iomega.eq.2)imax=1


      Dint=xDfit(zz,0,imax,engy**2,
     &     real(dsqrt(x)*dexp(y)),real(dsqrt(x)*dexp(-y)),0.)

      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=xDfit(zz,0,imax,engy**2,
     &       real(dsqrt(x)*dexp(y)),real(dsqrt(x)*dexp(-y)),b)

        write(ifhi,*) b,z/Dint
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DEfitb-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DEfitb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DEfitb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DEfitb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DEfitb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      sig2=sigma2(x,2)
      if(sig2.le.0.) sig2=1.e+10
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        write(ifhi,*) b,exp(-b**2/sig2)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DEfitbnSoft-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DEfitbnSoft-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DEfitbnSoft-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DEfitbnSoft-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DEfitbnSoft-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pft'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      sig2=sigma2(x,0)
      if(sig2.le.0.) sig2=1.e+10
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        write(ifhi,*) b,exp(-b**2/sig2)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************


      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DEfitbnSh-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DEfitbnSh-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DEfitbnSh-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DEfitbnSh-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DEfitbnSh-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp poc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      sig2=sigma2(x,1)
      if(sig2.le.0.) sig2=1.e+10
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        write(ifhi,*) b,exp(-b**2/sig2)
      enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xbnParD
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,y,Dsoftshval,z,xDfit



      nptg=50     !number of point for the graphs
      bmax=xpar2                   !value max of b (impact parameter)
      bmax=max(0.1,bmax)
      y=dble(xpar3)              !value of y (rapidity)
      x=dble(xpar4)
      zz=xpar7

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DExactbn-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DExactbn-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DExactbn-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DExactbn-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DExactbn-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      Dint=Dsoftshval(real(x)*smaxDf,x,y,0.,0)
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=Dsoftshval(real(x)*smaxDf,x,y,b,0)/Dint
        write(ifhi,*) b,z
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name DParambn-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name DParambn-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name DParambn-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name DParambn-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name DParambn-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lrd'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      Dint=xDfit(zz,0,1,smaxDf,
     &     real(dsqrt(x)*dexp(y)),real(dsqrt(x)*dexp(-y)),0.)

      imax=idxD1
      if(iomega.eq.2)imax=1

      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        z=xDfit(zz,0,imax,smaxDf,
     &       real(dsqrt(x)*dexp(y)),real(dsqrt(x)*dexp(-y)),b)

        write(ifhi,*) b,z/Dint
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DEfitb-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DEfitb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DEfitb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DEfitb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DEfitb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      sig2=sigma2(x,2)
      if(sig2.le.0.) sig2=1.e+10
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        write(ifhi,*) b,exp(-b**2/sig2)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DEintb-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DEintb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DEintb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DEintb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DEintb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pft'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      sig2=sigma1i(x)
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        write(ifhi,*) b,exp(-b**2/sig2)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************


      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name DPfitb-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name DPfitb-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name DPfitb-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name DPfitb-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name DPfitb-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp poc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
c      write(ifhi,'(a)')       'yrange -.01 auto'
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      sig2=xsigmafit(x)
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        write(ifhi,*) b,exp(-b**2/sig2)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'



      end

c----------------------------------------------------------------------
      subroutine xbParD
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

      double precision x,om5s,xDfit,z(0:200),y
c     & ,omYuncut,omNpuncut,omYcut

      nptg=50                  !number of point for the graphs
      x=dble(xpar4)                 !value of biniDf (impact parameter)
      y=dble(xpar3)                 !value of y (rapidity)
      if(x.gt.dexp(-2.d0*y))x=x*dexp(-2.d0*y)
      zz=xpar7

      bmax=xpar2
      bmax=max(0.1,bmax)
      t=1.
c      iqqN=0
c      iqq=int(xpar7)
      iqq1=-1
      iqq2=-1

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name DbParam-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name DbParam-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name DbParam-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name DbParam-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name DbParam-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lbu'
      if(xpar5.eq.0.)then
        write(ifhi,'(a)')       'xmod lin ymod lin'
      else
        write(ifhi,'(a)')       'xmod lin ymod log'
      endif
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      xp=sqrt(real(x))*exp(real(y))
      xm=sqrt(real(x))*exp(-real(y))
      imax=idxD1
      if(iomega.eq.2)imax=1

      if(xpar6.eq.1.)t=real(xDfit(zz,0,imax,smaxDf,xp,xm,0.))
      if(abs(t).lt.1.d-8)t=1.
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
         z(i)=xDfit(zz,0,imax,smaxDf,xp,xm,b)/dble(t)
         write(ifhi,*) b,z(i)
       enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
        if (engy.ge.100.) then
          if (engy.ge.1000.) then
            if (engy.ge.10000.) then
           write(ifhi,'(a,I5)')  'openhisto name DbParamI-',int(engy)
            else
           write(ifhi,'(a,I4)')  'openhisto name DbParamI-',int(engy)
            endif
          else
           write(ifhi,'(a,I3)')  'openhisto name DbParamI-',int(engy)
          endif
        else
          write(ifhi,'(a,I2)')  'openhisto name DbParamI-',int(engy)
        endif
      else
        write(ifhi,'(a,I1)')  'openhisto name DbParamI-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lga'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      if(xpar6.eq.1.)t=(alpDs(idxD0,iclegy,iclpro,icltar)
     &   *(smaxDf*real(x))**betDs(idxD0,iclegy,iclpro,icltar)
     &                +alpDs(1,iclegy,iclpro,icltar)
     &   *(smaxDf*real(x))**betDs(1,iclegy,iclpro,icltar))
     &   /real(idxD0+1)
      if(abs(t).lt.1.d-8)t=1.
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        tmp=(alpDs(idxD0,iclegy,iclpro,icltar)/t
     &   *(smaxDf*real(x))**(betDs(idxD0,iclegy,iclpro,icltar)
     &           +gamDs(idxD0,iclegy,iclpro,icltar)*b**2.)
     &        *exp(-b**2./delDs(idxD0,iclegy,iclpro,icltar))
     &                 +alpDs(1,iclegy,iclpro,icltar)/t
     &   *(smaxDf*real(x))**(betDs(1,iclegy,iclpro,icltar)
     &           +gamDs(1,iclegy,iclpro,icltar)*b**2.)
     &        *exp(-b**2./delDs(1,iclegy,iclpro,icltar)))
     &   /real(idxD0+1)
        write(ifhi,*) b,tmp
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************

      if (engy.ge.10.) then
        if (engy.ge.100.) then
          if (engy.ge.1000.) then
            if (engy.ge.10000.) then
           write(ifhi,'(a,I5)')  'openhisto name DbExaI-',int(engy)
            else
           write(ifhi,'(a,I4)')  'openhisto name DbExaI-',int(engy)
            endif
          else
           write(ifhi,'(a,I3)')  'openhisto name DbExaI-',int(engy)
          endif
        else
          write(ifhi,'(a,I2)')  'openhisto name DbExaI-',int(engy)
        endif
      else
        write(ifhi,'(a,I1)')  'openhisto name DbExaI-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp poc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-bmax,bmax
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis b"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "x=',x,'"'
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "y=',y,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      if(xpar6.eq.1.)t=2*real(om5s(real(x)*smaxDf,x,0.d0,0.,iqq1,iqq2)
     &         /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar))))
      if(abs(t).lt.1.d-8)t=1.
      nptg=nptg/2
      do i=0,nptg
        b=-bmax+2.*real(i)/real(nptg)*bmax
        tmp=2*real(om5s(real(x)*smaxDf,x,y,b,iqq1,iqq2)
     &         /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar))))/t
        write(ifhi,*) b,tmp
      enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xGexaJ
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.incpar'

      parameter(nptg=50) !number of point for the graphs
      double precision x(0:nptg),omJ1(0:nptg),xtmp,w(0:nptg),corfa
      double precision z(0:nptg),om5J!,omYuncut,omYuncutJ
      double precision xminr,y,t,omJ2(0:nptg),omJ3(0:nptg),omJ4(0:nptg)
     &,omJ5(0:nptg)!,omJ6(0:nptg),omJ7(0:nptg)


      kollsave=koll        !koll modified in zzfz
      koll=1
      biniDf=xpar2                 !value of biniDf (impact parameter)
      y=dble(xpar3)
      t=1.d0
      xtmp=xmaxDf
      xmaxDf=exp(-2.d0*y)

      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

      nnnmax=1
      if(iscreen.ne.0)nnnmax=2
      iscreensave=iscreen

      do nnn=1,nnnmax

        if(nnn.eq.2)then
          iscreen=0
          corfa=0d0
          zzp=0.
          zzt=0.
        elseif(iscreen.ne.0)then
          call zzfz(zzp,zzt,kollth,biniDf)
          koll=kollth
          rs=r2had(iclpro)+r2had(icltar)+slopom*log(engy*engy)
          rpom=4.*.0389*rs
          b2x=epscrp*rpom
          zzini=epscrw*fscra(engy/egyscr)*exp(-biniDf*biniDf/2./b2x)
          zzini=min(zzini,epscrx) !saturation
          zzp=zzini+zzp
          zzt=zzini+zzt
          if(gfactor.lt.0.)then
            corfa=dble(abs(gfactor)*(zzp+zzt))
          else
            corfa=0d0
          endif
        else
          corfa=0d0
          zzp=0.
          zzt=0.
        endif


      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')  'openhisto name GIexa-',int(engy)
         else
      write(ifhi,'(a,I4)')  'openhisto name GIexa-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')  'openhisto name GIexa-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')  'openhisto name GIexa-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')  'openhisto name GIexa-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lru'
      if(xpar5.eq.0.)then
        write(ifhi,'(a)')       'xmod log ymod log'
      else
        write(ifhi,'(a)')       'xmod log ymod lin'
      endif
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      x(0)=xminr
      do i=0,nptg
        if (i.ne.0) x(i)=x(0)*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        z(i)=0.d0
        omJ1(i)=om5J(zzp,zzt,x(i),y,biniDf,0)
        omJ2(i)=om5J(zzp,zzt,x(i),y,biniDf,1)
        omJ3(i)=om5J(zzp,zzt,x(i),y,biniDf,2)
        omJ3(i)=omJ3(i)+om5J(zz,zz,x(i),y,biniDf,3)
        omJ4(i)=om5J(zzp,zzt,x(i),y,biniDf,4)
        omJ5(i)=om5J(zzp,zzt,x(i),y,biniDf,5)
        z(i)=omJ1(i)+omJ2(i)+omJ3(i)+omJ4(i)+omJ5(i)
        w(i)=om5J(zzp,zzt,x(i),y,biniDf,-1)
        if(xpar6.eq.0)t=z(i)/w(i)
        if(xpar6.eq.1)t=z(i)
        write(ifhi,*) x(i),z(i)/t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c*************************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name GIsoft-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name GIsoft-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name GIsoft-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name GIsoft-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name GIsoft-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lba'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        if(xpar6.eq.0)t=z(i)/w(i)
        if(xpar6.eq.1)t=z(i)
        gfa=1.
      if(omJ2(i).gt.0)gfa=exp(-max(0d0,corfa
     &                   *(1d0-sqrt(dble(xggfit)*omJ1(i)/omJ2(i)))))
        write(ifhi,*) x(i),omJ1(i)*gfa/t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c*************************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name GIgg-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name GIgg-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name GIgg-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name GIgg-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name GIgg-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lra'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        if(xpar6.eq.0)t=z(i)/w(i)
        if(xpar6.eq.1)t=z(i)
        gfa=1.
        if(omJ2(i).gt.0)gfa=exp(-max(0d0,corfa
     &                     *(1d0-sqrt(dble(xggfit)*omJ1(i)/omJ2(i)))))
        write(ifhi,*) x(i),(omJ2(i)+(1.-gfa)*omJ1(i))/t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c*************************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name GIgq-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name GIgq-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name GIgq-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name GIgq-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name GIgq-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lga'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        if(xpar6.eq.0)t=z(i)/w(i)
        if(xpar6.eq.1)t=z(i)
        write(ifhi,*) x(i),omJ3(i)/t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c*************************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name GIqq-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name GIqq-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name GIqq-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name GIqq-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name GIqq-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lya'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        if(xpar6.eq.0)t=z(i)/w(i)
        if(xpar6.eq.1)t=z(i)
        write(ifhi,*) x(i),omJ4(i)/t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c*************************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name GIdif-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name GIdif-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name GIdif-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name GIdif-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name GIdif-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lfa'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        if(xpar6.eq.0)t=z(i)/w(i)
        if(xpar6.eq.1)t=z(i)
        write(ifhi,*) x(i),omJ5(i)/t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c*************************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name GItot-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name GItot-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name GItot-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name GItot-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name GItot-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        if(xpar6.eq.0)t=z(i)/w(i)
        if(xpar6.eq.1)t=z(i)
        write(ifhi,*) x(i),(omJ1(i)+omJ2(i)+omJ3(i)+omJ4(i)+omJ5(i))
     &                  /t
      enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0'

        if(nnn.eq.2)iscreen=iscreensave

      enddo

      xmaxDf=xtmp
      koll=kollsave

      end



c----------------------------------------------------------------------
      subroutine xsParD
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

      double precision x,xminr,y,om5s,xtmp,z
c     & ,t,omYuncut

      nptg=50                  !number of point for the graphs
      biniDf=xpar2                 !value of biniDf (impact parameter)
      y=dble(xpar3)                 !value of y (rapidity)
      xtmp=xmaxDf
      xmaxDf=dexp(-2.d0*y)
      call Class('xsParD     ')
      iqq1=-1
      iqq2=-1
c      iqqN=0

      xminr=dble(egylow/engy)**2.d0  !value of xminr for plotting the function

c**********************************************************************

      if (engy.ge.10.) then
        if (engy.ge.100.) then
          if (engy.ge.1000.) then
            if (engy.ge.10000.) then
           write(ifhi,'(a,I5)')  'openhisto name DParamI-',int(engy)
            else
           write(ifhi,'(a,I4)')  'openhisto name DParamI-',int(engy)
            endif
          else
           write(ifhi,'(a,I3)')  'openhisto name DParamI-',int(engy)
          endif
        else
          write(ifhi,'(a,I2)')  'openhisto name DParamI-',int(engy)
        endif
      else
        write(ifhi,'(a,I1)')  'openhisto name DParamI-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lga'
      if(xpar5.eq.0.)then
        write(ifhi,'(a)')       'xmod log ymod log'
      else
        write(ifhi,'(a)')       'xmod log ymod lin'
      endif
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        egy=sqrt(real(x))*engy
        iclegy=1+int((log(egy)-log(egylow))/log(egyfac))
        write(ifhi,*) x,(alpDs(idxD0,iclegy,iclpro,icltar)
     &   *(smaxDf*real(x))**(betDs(idxD0,iclegy,iclpro,icltar)
     &          +gamDs(idxD0,iclegy,iclpro,icltar)*biniDf**2.)
     &     *exp(-biniDf**2./delDs(idxD0,iclegy,iclpro,icltar))
     &                 +alpDs(1,iclegy,iclpro,icltar)
     &   *(smaxDf*real(x))**(betDs(1,iclegy,iclpro,icltar)
     &           +gamDs(1,iclegy,iclpro,icltar)*biniDf**2.)
     &        *exp(-biniDf**2./delDs(1,iclegy,iclpro,icltar)))
     &   /real(idxD0+1)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************

      if (engy.ge.10.) then
        if (engy.ge.100.) then
          if (engy.ge.1000.) then
            if (engy.ge.10000.) then
           write(ifhi,'(a,I5)')  'openhisto name DExaI-',int(engy)
            else
           write(ifhi,'(a,I4)')  'openhisto name DExaI-',int(engy)
            endif
          else
           write(ifhi,'(a,I3)')  'openhisto name DExaI-',int(engy)
          endif
        else
          write(ifhi,'(a,I2)')  'openhisto name DExaI-',int(engy)
        endif
      else
        write(ifhi,'(a,I1)')  'openhisto name DExaI-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp poc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange .01 auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis x"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      nptg=nptg/2
      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        egy=sqrt(real(x))*engy
        iclegy=1+int((log(egy)-log(egylow))/log(egyfac))
        z=2d0*om5s(real(x)*smaxDf,1.d0,0.d0,biniDf,iqq1,iqq2)
     &              /dble(chad(iclpro)*chad(icltar))
      write(ifhi,*) x,z
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      xmaxDf=xtmp

      end

c----------------------------------------------------------------------
      subroutine xyParD
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'

      double precision x,om5s,xDfit,z(0:200),ymax,y,t
c     & ,omYuncut,omNpuncut

      nptg=50                  !number of point for the graphs
      biniDf=xpar2                 !value of biniDf (impact parameter)

      x=dble(xpar4)
      if(x.le.1.d-20)x=1.d0/dble(engy)
      ymax=-.5d0*dlog(x)
      zz=xpar7
      iqq1=-1
      iqq2=-1
c      iqqN=0


      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')       'openhisto name DyParam-',int(engy)
         else
      write(ifhi,'(a,I4)')       'openhisto name DyParam-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')       'openhisto name DyParam-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')       'openhisto name DyParam-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')       'openhisto name DyParam-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lbu'
      if(xpar5.eq.0.)then
        write(ifhi,'(a)')       'xmod lin ymod lin'
      else
        write(ifhi,'(a)')       'xmod lin ymod log'
      endif
      write(ifhi,'(a,2e11.3)')'xrange ',-ymax-1.d0,ymax+1.d0
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis y"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "x=',x,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      imax=idxD1
      if(iomega.eq.2)imax=1

      do i=0,nptg
        y=-ymax+(ymax+ymax)*(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
         z(i)=xDfit(zz,0,imax,engy**2,xp,xm,biniDf)
         write(ifhi,*) y,z(i)
       enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
        if (engy.ge.100.) then
          if (engy.ge.1000.) then
            if (engy.ge.10000.) then
           write(ifhi,'(a,I5)')  'openhisto name DyParamI-',int(engy)
            else
           write(ifhi,'(a,I4)')  'openhisto name DyParamI-',int(engy)
            endif
          else
           write(ifhi,'(a,I3)')  'openhisto name DyParamI-',int(engy)
          endif
        else
          write(ifhi,'(a,I2)')  'openhisto name DyParamI-',int(engy)
        endif
      else
        write(ifhi,'(a,I1)')  'openhisto name DyParamI-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lga'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange ',-ymax-1.d0,ymax+1.d0
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis y"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "x=',x,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        y=-ymax+(ymax+ymax)*(dble(i)/dble(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
        write(ifhi,*) y,(alpDs(idxD0,iclegy,iclpro,icltar)
     &     *smaxDf**(betDs(idxD0,iclegy,iclpro,icltar)
     &          +gamDs(idxD0,iclegy,iclpro,icltar)*biniDf**2.)
     &     *xp**(betDps(idxD0,iclegy,iclpro,icltar)
     &          +gamDs(idxD0,iclegy,iclpro,icltar)*biniDf**2.)
     &     *xm**(betDpps(idxD0,iclegy,iclpro,icltar)
     &          +gamDs(idxD0,iclegy,iclpro,icltar)*biniDf**2.)
     &     *exp(-biniDf**2./delDs(idxD0,iclegy,iclpro,icltar))
     &                  +alpDs(1,iclegy,iclpro,icltar)
     &     *smaxDf**(betDs(1,iclegy,iclpro,icltar)
     &            +gamDs(1,iclegy,iclpro,icltar)*biniDf**2.)
     &     *xp**(betDps(1,iclegy,iclpro,icltar)
     &            +gamDs(1,iclegy,iclpro,icltar)*biniDf**2.)
     &     *xm**(betDpps(1,iclegy,iclpro,icltar)
     &            +gamDs(1,iclegy,iclpro,icltar)*biniDf**2.)
     &        *exp(-biniDf**2./delDs(1,iclegy,iclpro,icltar)))
     &     /real(idxD0+1)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************

      if (engy.ge.10.) then
        if (engy.ge.100.) then
          if (engy.ge.1000.) then
            if (engy.ge.10000.) then
           write(ifhi,'(a,I5)')  'openhisto name DyExaI-',int(engy)
            else
           write(ifhi,'(a,I4)')  'openhisto name DyExaI-',int(engy)
            endif
          else
           write(ifhi,'(a,I3)')  'openhisto name DyExaI-',int(engy)
          endif
        else
          write(ifhi,'(a,I2)')  'openhisto name DyExaI-',int(engy)
        endif
      else
        write(ifhi,'(a,I1)')  'openhisto name DyExaI-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp poc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange ',-ymax-1.d0,ymax+1.d0
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')      'text 0 0 "xaxis y"'
      write(ifhi,'(a,a)')    'text 0 0 "yaxis D(x+,x-,s,b)" '
      write(ifhi,'(a,e8.2,a)')  'text 0.1 0.9 "E=',engy,' GeV"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.1 0.7 "x=',x,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      nptg=nptg/2
      do i=0,nptg
        y=-ymax+(ymax+ymax)*(dble(i)/dble(nptg))
        t=2*om5s(real(x)*smaxDf,x,y,biniDf,iqq1,iqq2)
     &         /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
        write(ifhi,*) y,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParSigma
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,w(0:200),z(0:200),xminr,t

      nptg=20                  !number of point for the graphs

      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name SigmaReel-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name SigmaReel-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name SigmaReel-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name SigmaReel-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name SigmaReel-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [s]^2!(X)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.75 "Emax=',engy,' GeV"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        z(i)=dble(sigma2(X,2))
        if(z(i).le.0.) z(i)=0.d0
        write(ifhi,'(2e14.6)') X,z(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name SigmaParam-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name SigmaParam-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name SigmaParam-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name SigmaParam-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name SigmaParam-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [s]^2!?param!(X)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.75 "Emax=',engy,' GeV"'
        endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        w(i)=dble(xsigmafit(X))
        write(ifhi,'(2e14.6)') X,w(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name SigmaInt-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name SigmaInt-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name SigmaInt-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name SigmaInt-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name SigmaInt-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lba'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [s]^2!?Int!(X)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.75 "Emax=',engy,' GeV"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=dble(sigma1i(X))
        write(ifhi,'(2e14.6)') X,t
      enddo

      write(ifhi,'(a)')    '  endarray'

      if(xpar8.eq.1)then
      write(ifhi,'(a)')    'closehisto plot 0-'


      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name SigmaReelSoft-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name SigmaReelSoft-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name SigmaReelSoft-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name SigmaReelSoft-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name SigmaReelSoft-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pft'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [s]^2!(X)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.75 "Emax=',engy,' GeV"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=dble(sigma2(X,0))
        if(t.gt.0.) write(ifhi,'(2e14.6)') X,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name SigmaReelSh-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name SigmaReelSh-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name SigmaReelSh-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name SigmaReelSh-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name SigmaReelSh-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pot'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [s]^2!(X)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.75 "Emax=',engy,' GeV"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=dble(sigma2(X,1))
        if(t.gt.0.)write(ifhi,'(2e14.6)') X,t
      enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************
      write(ifhi,'(a)')    'openhisto'
      write(ifhi,'(a)')       'htyp lya'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [s]^2!(X)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.75 "Emax=',engy,' GeV"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=1.d0/dble(-gamD(0,iclpro,icltar)*log(X*smaxDf)
     &     +1./delD(0,iclpro,icltar))
        if(t.gt.0.)write(ifhi,'(2e14.6)') X,t
      enddo

      write(ifhi,'(a)')    '  endarray'


      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************
      write(ifhi,'(a)')    'openhisto'
      write(ifhi,'(a)')       'htyp lyo'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [s]^2!(X)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.75 "Emax=',engy,' GeV"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=1.d0/dble(-gamD(1,iclpro,icltar)*log(X*smaxDf)
     *     +1./delD(1,iclpro,icltar))
        if(t.gt.0.)write(ifhi,'(2e14.6)') X,t
      enddo

      write(ifhi,'(a)')    '  endarray'

      endif


      write(ifhi,'(a)')    'closehisto plot 0'

c**********************************************************************
c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name SigmaDiff-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name SigmaDiff-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name SigmaDiff-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name SigmaDiff-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name SigmaDiff-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,1.
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [D][s]/[s]"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.05 0.9 "Emax=',engy,' GeV"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        X=xminr
        if (i.ne.0) X=X*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=0.d0
        if(w(i).gt.0.d0)t=(z(i)-w(i))/w(i)
        if(abs(t).gt.0.15d0) t=dsign(0.15d0,t)
        write(ifhi,'(2e14.6)') X,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParGauss
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,om5s,xDfit,y,enh,t!,omNpuncut

      nptg=50                  !number of point for the graphs
      x=dble(xpar4)                  !value of x (energy)
      y=dble(xpar2)                  !value of rapidity
      zz=xpar7

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name GaussExact-',int(engy)
           else
      write(ifhi,'(a,I4)')    'openhisto name GaussExact-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name GaussExact-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name GaussExact-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name GaussExact-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis b*D(x+,x-,s,b)"'
c      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.6 0.9 "E=',engy,' GeV"'
      write(ifhi,'(a,f5.2,a)')  'text 0.6 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.6 0.7 "y=',y,'"'
c      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        b=bmaxDf*(real(i)/real(nptg))
        enh=0.d0
        t=dble(b)*(2*om5s(real(x)*smaxDf,x,y,b,-1,-1)
     &                +enh)
     &         /(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
       write(ifhi,*) b,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'



c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name GaussParam-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name GaussParam-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name GaussParam-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name GaussParam-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name GaussParam-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',0.,bmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis b*D(x+,x-,s,b)"'
c      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.6 0.9 "E=',engy,' GeV"'
      write(ifhi,'(a,f5.2,a)')  'text 0.6 0.8 "x=',x,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.6 0.7 "y=',y,'"'
c      endif
      write(ifhi,'(a)')       'array 2'

      imax=idxD1
      if(iomega.eq.2)imax=1

      do i=0,nptg
        b=bmaxDf*(real(i)/real(nptg))
        xp=sqrt(real(x))*exp(real(y))
        xm=sqrt(real(x))*exp(-real(y))
        t=xDfit(zz,0,imax,engy**2,xp,xm,b)
        write(ifhi,'(2e14.6)') b,dble(b)*t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end



c----------------------------------------------------------------------
      subroutine xParOmega1
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,w(0:200),z(0:200)
      double precision yp,om1,xminr,Dsoftshval,t

      nptg=50                 !number of point for the graphs
      biniDf=xpar2               !value of biniDf (impact parameter)
      yp=xpar3                   !value of yp (rapidity)
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name Om1Exact-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name Om1Exact-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name Om1Exact-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name Om1Exact-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name Om1Exact-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [h](x,0,b)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        w(i)=Dsoftshval(real(x)*engy**2,x,0.d0,biniDf,0)
     &       *(x**dble(-alppar)*dble(chad(iclpro)*chad(icltar)))
        write(ifhi,*) x,w(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name om5param-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name om5param-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name om5param-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name om5param-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name om5param-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1!(x,0,b)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        z(i)=om1(x,yp,biniDf)
        write(ifhi,*) x,z(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'


c**********************************************************************
c**********************************************************************

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name Om1Diff-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name Om1Diff-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name Om1Diff-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name Om1Diff-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name Om1Diff-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X"'
      write(ifhi,'(a)')    'text 0 0 "yaxis ([w]?1!-[h])/[h]"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
         t=z(i)!(z(i)-w(i))
c         if(abs(w(i)).gt.0.)t=t/w(i)
c         if(abs(t).gt.0.15d0) t=dsign(0.15d0,t)
         write(ifhi,*) x,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'


      end

c----------------------------------------------------------------------
      subroutine xEpsilon(iii)
c----------------------------------------------------------------------
c iii:  modus (0,1,2)
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'

      parameter(nxeps=20,nyeps=32)
      common/cxeps1/w(0:nxeps,nyeps),y1(nyeps),y2(nyeps)
      common/cxeps2/db,b1,b2
      common/geom/rmproj,rmtarg,bmax,bkmx
      character ch*3
      common /psar34/ rrr,rrrm
      common /psar41/ rrrp,rrrmp
      external pprzz,pttzz

      b1=0.03
      b2=bkmx*1.2
      db=(b2-b1)/nyeps

        if(iii.eq.0)then

      do j=0,nxeps
       do k=1,nyeps
        w(j,k)=0
       enddo
      enddo

        elseif(iii.eq.2)then


      do nj=1,14
       y1(nj)=1e+20
       y2(nj)=1e-20
       do k=1,nyeps
        if(w(0,k).ne.0)then
         y1(nj)=min(y1(nj),w(nj,k)/w(0,k))
         y2(nj)=max(y2(nj),w(nj,k)/w(0,k))
        endif
       enddo
       if(y1(nj).ge.0)then
         y1(nj)=max(y1(nj)*.2,1e-4)
       else
         y1(nj)=y1(nj)*2.
       endif
       y2(nj)=min(y2(nj)*5,1e4)
      enddo
      y2(13)=max(y2(13),y2(14))
      y2(14)=max(y2(13),y2(14))
      y2(1)=max(y2(1),y2(2))
      y2(2)=max(y2(1),y2(2))
      y2(5)=max(y2(5),y2(6))
      y2(6)=max(y2(5),y2(6))
      y2(7)=y2(5)
      y2(8)=y2(5)
      y2(9)=max(y2(9),y2(10))
      y2(10)=max(y2(9),y2(10))
      y2(11)=y2(9)
      y2(12)=y2(9)
      do nj=1,14
       if(nj.le.9)write(ifhi,'(a,i1)')'openhisto name xEps',nj
       if(nj.gt.9)write(ifhi,'(a,i2)')'openhisto name xEps',nj
       ch='lin'
       if(nj.eq.7.or.nj.eq.11)ch='lyo'
       if(nj.eq.8.or.nj.eq.12)ch='lgo'
       write(ifhi,'(a)')     'htyp '//ch//' xmod lin'
       if(y1(nj).ge.0.)then
         write(ifhi,'(a)')     'ymod log'
       else
         write(ifhi,'(a)')     'ymod lin'
       endif
       write(ifhi,'(a,e9.2)')'xrange 0 ',b2
       if(nj.eq.1.or.nj.eq.3.or.nj.eq.5.or.nj.eq.9)then
       write(ifhi,'(a,2e9.2)')     'yrange ',min(y1(nj),y1(nj+1))
     *                                      ,max(y2(nj),y2(nj+1))
       else
       write(ifhi,'(a,2e9.2)')     'yrange ',y1(nj),y2(nj)
       endif
       write(ifhi,'(a)')     'text 0 0 "xaxis b"'
       if(nj.eq.1) write(ifhi,'(a)')'txt "yaxis [e]?GP/T!(b)"'
       if(nj.eq.1) write(ifhi,'(a)')'txt "title soft pro   soft tar"'
       if(nj.eq.3) write(ifhi,'(a)')'txt "yaxis [e]?G!(b)"'
       if(nj.eq.3) write(ifhi,'(a)')'txt "title diff"'
c       if(nj.eq.3) write(ifhi,'(a)')'txt "title soft   semi"'
       if(nj.eq.5) write(ifhi,'(a)')'txt "yaxis [b]?eff!(b)"'
       if(nj.eq.5) write(ifhi,'(a)')'txt "title soft pro   soft tar"'
       if(nj.eq.9) write(ifhi,'(a)')'txt "yaxis [b]?eff!(b)"'
       if(nj.eq.9) write(ifhi,'(a)')'txt "title semi pro   semi tar"'
       if(nj.eq.13)write(ifhi,'(a)')'txt "yaxis Z?P/T!"'
       write(ifhi,'(a)')       'array 2'
       do k=1,nyeps
        b=b1+(k-0.5)*db
        y=0
        if(w(0,k).ne.0)y=w(nj,k)/w(0,k)
        write(ifhi,'(2e11.3)')b,y
       enddo
       write(ifhi,'(a)')    '  endarray'
       if(nj.eq.2.or.nj.eq.4.or.nj.eq.8.or.nj.eq.12.or.nj.eq.14
     &    .or.nj.eq.16)then
        write(ifhi,'(a)')    'closehisto plot 0'
       else
        write(ifhi,'(a)')    'closehisto plot 0-'
       endif
      enddo
      !----15-16-17---
      write(ifhi,'(a)') 'openhisto name xEps15'
      write(ifhi,'(a)') 'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)') 'xrange 0 10'
      write(ifhi,'(a)') 'text 0 0 "xaxis b?0!"'
      write(ifhi,'(a)') 'txt "yaxis Z?P/T!(b?0!)"'
      write(ifhi,'(a)') 'array 2'
      do k=1,10
       b=(k-0.5)
       y=0
       if(w(17,k).ne.0)y=w(15,k)/w(17,k)
       write(ifhi,'(2e11.3)')b,y
      enddo
      write(ifhi,'(a)')  '  endarray'
      write(ifhi,'(a)')  'closehisto plot 0-'

      write(ifhi,'(a)') 'openhisto name xEps16'
      write(ifhi,'(a)') 'htyp lin xmod lin ymod lin'
      write(ifhi,'(a)') 'xrange 0 10'
      write(ifhi,'(a)') 'text 0 0 "xaxis b?0!"'
      write(ifhi,'(a)') 'txt "yaxis Z?P/T!(b?0!)"'
      write(ifhi,'(a)') 'array 2'
      do k=1,10
       b=(k-0.5)
       y=0
       if(w(17,k).ne.0)y=w(16,k)/w(17,k)
       write(ifhi,'(2e11.3)')b,y
      enddo
      write(ifhi,'(a)')  '  endarray'
      write(ifhi,'(a)')  'closehisto plot 0'
      !----18-19-20---
      kk=2
      do k=3,32
        if(w(18,k).ne.0)kk=k
      enddo
      xmx=(kk-1)/31.*0.1*maproj*matarg
      write(ifhi,'(a)') 'openhisto name xEps18'
      write(ifhi,'(a)') 'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,f10.2)') 'xrange 0 ',xmx
      write(ifhi,'(a)') 'text 0 0 "xaxis n?Gl!"'
      write(ifhi,'(a)') 'txt "yaxis Z?P/T!(n?Gl!)"'
      write(ifhi,'(a)') 'array 2'
      do k=1,32
       x=(k-1.)*0.1*maproj*matarg/(nyeps-1.)
       y=0
       if(w(20,k).ne.0)y=w(18,k)/w(20,k)
       write(ifhi,'(2e11.3)')x,y
      enddo
      write(ifhi,'(a)')  '  endarray'
      write(ifhi,'(a)')  'closehisto plot 0-'

      write(ifhi,'(a)') 'openhisto name xEps19'
      write(ifhi,'(a)') 'htyp lin xmod lin ymod lin'
      write(ifhi,'(a,f10.2)') 'xrange 0 ',xmx
      write(ifhi,'(a)') 'text 0 0 "xaxis n?Gl!"'
      write(ifhi,'(a)') 'txt "yaxis Z?P/T!(n?Gl!)"'
      write(ifhi,'(a)') 'array 2'
      do k=1,32
       x=(k-1.)*0.1*maproj*matarg/(nyeps-1.)
       y=0
       if(w(20,k).ne.0)y=w(19,k)/w(20,k)
       write(ifhi,'(2e11.3)')x,y
      enddo
      write(ifhi,'(a)')  '  endarray'
      write(ifhi,'(a)')  'closehisto plot 0'

        endif

      end

c----------------------------------------------------------------------
      subroutine xZnucTheo
c----------------------------------------------------------------------
c Theoretical mean Z as a function of impact parameter between proj and targ
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incems'
      include 'epos.incpar'

      common/geom/rmproj,rmtarg,bmax,bkmx
      common /psar34/ rrr,rrrm
      common /psar41/ rrrp,rrrmp
      external pprzz,pttzz


      rs=r2had(iclpro)+r2had(icltar)+slopom*log(engy**2)
      bglaub2=4.*.0389*rs
      b2x=epscrp*bglaub2
      zzini=epscrw*fscra(engy/egyscr)

      if(maproj.gt.1)then
        rrrp=radnuc(maproj)/difnuc(maproj)
        rrrmp=rrrp+log(9.)
      endif
      if(matarg.gt.1)then
        rrr=radnuc(matarg)/difnuc(matarg)
        rrrm=rrr+log(9.)
      endif

      write(ifhi,'(a)') 'openhisto name xZTheo'
      write(ifhi,'(a)') 'htyp lyi xmod lin ymod lin'
      write(ifhi,'(a)') 'xrange 0 10'
      write(ifhi,'(a)') '- text 0 0 "xaxis b?0!"'
      write(ifhi,'(a)') '+ txt "yaxis Z?P!(b?0!)"'
      write(ifhi,'(a)') '+ txt "yaxis Z?T!(b?0!)"'
      write(ifhi,'(a)') 'array -3'
      do k=1,10
       b=(k-0.5)
       call zzfz(zzp,zzt,kollth,b)
       zz=min(epscrw,zzini*exp(-b*b/2./b2x))
       write(ifhi,'(2e11.3)')b,zz+zzp,zz+zzt
      enddo
      write(ifhi,'(a)')  '  endarray'
      write(ifhi,'(a)')  'closehisto'
      write(ifhi,'(a)')  ' plot -htyp lyi xZTheo+1-',
     &                   ' plot -htyp lga xZTheo+2'

      end

c----------------------------------------------------------------------
      subroutine xParOmegaN
c----------------------------------------------------------------------
c xpar1=engy
c xpar2=b
c xpar4=xremnant
c xpar5: 0=log scale (x dep of om) 1=lin scale (b dep of om)
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      include 'epos.incsem'
      include 'epos.incems'
      double precision x,w(0:200),z(0:200),xminr,t,ghh
     *,xprem,omGam,Womint,Gammapp,WomGamint,omGamk
c     *,yp,SomY,omYuncut,y,xtmp

      nptg=30                  !number of point for the graphs
      biniDf=xpar2               !value of biniDf (impact parameter)
      xprem=dble(xpar4)            !value of x remnant
      bmax=3.
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

c**********************************************************************

      do i=0,3
        b=bmax*(real(i)/real(3))
        z(i)=1.d0
c        if(xpar5.eq.0.)z(i)=Gammapp(engy**2.,b,1)
      enddo

      write(ifhi,'(a)')    'openhisto name Womint-1'
      write(ifhi,'(a)')       'htyp lru'
      if(xpar5.eq.0.)then
        write(ifhi,'(a)')       'xmod lin ymod log'
      else
        write(ifhi,'(a)')       'xmod lin ymod lin'
      endif
      write(ifhi,'(a,2e11.3)')'xrange',0.,bmax
      write(ifhi,'(a)')    'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?int!(s,b)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      ierr=0
      ghh=0.d0
      do i=0,3
        b=bmax*(real(i)/real(3))
        w(i)=Womint(engy**2.,b)
        if(b.eq.biniDf)then
          write(*,*)'Womint(',b,',1)=',w(i)
          ghh=ghh+w(i)
        endif
        if(w(i).lt.0.d0.and.ierr.eq.0.and.xpar5.eq.0.)then
          write(*,*)'Warning Womint(1)<0 =',w(i)
          w(i)=-w(i)
          ierr=1
          elseif(w(i).lt.0.d0.and.xpar5.eq.0.)then
            w(i)=-w(i)
          elseif(w(i).ge.0.d0.and.ierr.eq.1.and.xpar5.eq.0.)then
            ierr=0
            write(*,*)'Warning Womint(1)>0 =',w(i)
        endif
        write(ifhi,*) b,w(i)/z(i)
      enddo


      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************

      write(ifhi,'(a)')    'openhisto'
      write(ifhi,'(a)')       'htyp pfc'
      if(xpar5.eq.0.)then
        write(ifhi,'(a)')       'xmod lin ymod log'
      else
        write(ifhi,'(a)')       'xmod lin ymod lin'
      endif
      write(ifhi,'(a,2e11.3)')'xrange',0.,bmax
      write(ifhi,'(a)')    'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis b"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?int!(s,b)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        b=bmax*(real(i)/real(nptg))
        z(i)=1.d0
        if(xpar5.eq.0.)then
          z(i)=z(i)+dabs(WomGamint(b))
        endif
      enddo
      ierr=0
      do i=0,nptg
        b=bmax*(real(i)/real(nptg))
        w(i)=WomGamint(b)
        if(w(i).lt.0.d0.and.ierr.eq.0.and.xpar5.eq.0.)then
          write(*,*)'Warning WomGamint(1)<0 =',w(i)
          w(i)=-w(i)
          ierr=1
          elseif(w(i).lt.0.d0.and.xpar5.eq.0.)then
            w(i)=-w(i)
          elseif(w(i).ge.0.d0.and.ierr.eq.1.and.xpar5.eq.0.)then
            ierr=0
            write(*,*)'Warning WomGamint(1)>0 =',w(i)
        endif
        write(ifhi,*) b,w(i)/z(i)
      enddo


      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'
      t=Gammapp(engy**2.,biniDf,1)
      write(*,*)'--> gamma(',biniDf,')=',ghh,t

c**********************************************************************
c**********************************************************************
      do k=1,koll
        bk(k)=biniDf
      enddo
      call GfunPark(0)
      call integom1(0)

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name xOmNG-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name xOmNG-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name xOmNG-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name xOmNG-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name xOmNG-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?NPi!(x+rem,x-rem,b)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.85 "x+?rem!=',xprem,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=dabs(omGam(xprem,x,biniDf))
        write(ifhi,*) x,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      if (engy.ge.10.) then
       if (engy.ge.100.) then
        if (engy.ge.1000.) then
         if (engy.ge.10000.) then
      write(ifhi,'(a,I5)')    'openhisto name xOmNG-',int(engy)
         else
      write(ifhi,'(a,I4)')    'openhisto name xOmNG-',int(engy)
        endif
        else
      write(ifhi,'(a,I3)')    'openhisto name xOmNG-',int(engy)
       endif
       else
      write(ifhi,'(a,I2)')    'openhisto name xOmNG-',int(engy)
      endif
      else
      write(ifhi,'(a,I1)')    'openhisto name xOmNG-',int(engy)
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?NPi!(x+rem,x-rem,b)"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.85 "x+?rem!=',xprem,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        t=dabs(omGamk(1,xprem,x))
        write(ifhi,*) x,t
      enddo

      write(ifhi,'(a)')    '  endarray'

      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParGampp
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision Gammapp,GammaGauss,sg,sgmc,Znorm,Zn,t
     *                 ,w(0:200),z(0:200)!,GammaMC

      nptg=2                  !number of point for the graphs
      biniDf=xpar2              !value of biniDf (impact parameter)


c**************************************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name Gamma-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name Gamma-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',0.,real(nptg)
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis m"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [g]?h1h2!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'
      sg=0.d0
      do i=0,nptg
        w(i)=Gammapp(engy**2.,biniDf,i)
        sg=sg+w(i)
        write(ifhi,*) i,w(i)
      write(*,*) 'G12',i,w(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name GammaMC-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name GammaMC-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',0.,real(nptg)
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis m"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [g]?h1h2!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      sgmc=0.d0
      do i=0,nptg
        z(i)=GammaGauss(engy**2.,biniDf,i)
        sgmc=sgmc+z(i)
        write(ifhi,*) i,z(i)
        write(*,*) 'G12gauss',i,z(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

c**********************************************************************
c**********************************************************************
      Zn=Znorm(engy**2,biniDf)

      write(ifhi,'(a)')    'openhisto name GammaDiff'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',0.,real(nptg)
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis m"'
      write(ifhi,'(a)')    'text 0 0 "yaxis (G-GMC)/G"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      write(ifhi,'(a,f5.2,a)')  'text 0.1 0.8 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "[S]?Guncut!=',sg,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.8 "[S]?Gcut!=',sgmc,'"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.7 "Z=',Zn,'"'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        if(w(i).ne.0d0) t=(z(i)-w(i))/w(i)
c         if(abs(t).gt.0.5d0) t=dsign(0.5d0,t)
         write(ifhi,*) i,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomInc
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,PomIncExact,PomIncUnit,xminr,xm,t
     *                 ,w(0:200),z(0:200)

      nptg=10                  !number of point for the graphs
      biniDf=xpar2              !value of biniDf (impact parameter)
      xm=dble(xpar4)              !value of biniDf (impact parameter)
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function


c********************* red = PomIncXExact *****************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncExact-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncExact-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx+/dx-"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.8 "x-=',xm,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        w(i)=PomIncExact(dsqrt(x),dsqrt(x),biniDf)
        write(ifhi,*) x,w(i)
      write(*,*) 'Xe',i,w(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'



c************************* dot = PomIncXUnit **************************
c      nptg=50     !number of point for the graphs

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncUnit-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncUnit-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx+/dx-"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.8 "x-=',xm,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        z(i)=PomIncUnit(dsqrt(x),dsqrt(x),biniDf)
        write(ifhi,*) x,z(i)
        write(*,*) 'Xu',i,z(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

c**********************************************************************
c**********************************************************************

      write(ifhi,'(a)')    'openhisto name PomIncDiff'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis ([w]?5!-G)/G"'
      if (xpar8.eq.1.) then
      write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      write(ifhi,'(a,f5.2,a)')  'text 0.5 0.8 "x-=',xm,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        t=0.d0
        if(w(i).ne.0d0) t=(z(i)-w(i))/w(i)
c         if(abs(t).gt.0.5d0) t=dsign(0.5d0,t)
         write(ifhi,*) x,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomIncX
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,PomIncXExact,PomIncXUnit,xminr,y

      nptg=20                   !number of point for the graphs
      biniDf=xpar2              !value of biniDf (impact parameter)
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function


c********************* red = PomIncXExact *****************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncXExact-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncXExact-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx(x,b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c.......x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        y=PomIncXExact(x,biniDf)
        write(ifhi,*) x,y
c      write(*,*) 'Xe',i
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'



c************************* dot = PomIncXUnit **************************
c      nptg=50     !number of point for the graphs

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncXUnit-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncXUnit-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx(x,b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c.......x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncXUnit(x,biniDf)
        write(*,*) 'Xu',i
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomIncXI
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,xminr
      double precision PomIncXIExact,PomIncXIUnit

      nptg=20                   !number of point for the graphs
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

c*********************red = PomIncXIExact *****************************

      write(ifhi,'(a)')       'openhisto name PomIncXIExact'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis d[s]?Pom!/dx(x)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c.......x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncXIExact(x)
c.......write(*,*) 'XIe',i
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c***************************dot = PomIncXIUnit ************************
c.....nptg=50     !number of point for the graphs

      write(ifhi,'(a)')       'openhisto name PomIncXIUnit'
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)')    'text 0 0 "yaxis d[s]?Pom!/dx(x)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c.......x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncXIUnit(x)
        write(*,*) 'XIu',i
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomIncP
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,PomIncPUnit,xminr
      double precision PomIncPExact

      nptg=30                  !number of point for the graphs
      biniDf=xpar1              !value of biniDf (impact parameter)
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

c*********************red = PomIncPExact *****************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncPExact-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncPExact-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx+(x+,b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncPExact(x,biniDf)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**************************dot = PomIncPUnit **************************
c.....nptg=50     !number of point for the graphs

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncPUnit-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncPUnit-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx+(x+,b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(*,*) 'Pu',i
        write(ifhi,*) x,PomIncPUnit(x,biniDf)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomIncPI
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,xminr
      double precision PomIncPIExact,PomIncPIUnit

      nptg=50                  !number of point for the graphs
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

c*********************red = PomIncPIExact *****************************
c.....nptg=100     !number of point for the graphs

      write(ifhi,'(a)')       'openhisto name PomIncPIExact'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis d[s]?Pom!/dx+(x+)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncPIExact(x)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c***************************dot = PomIncPIUnit ************************
c.....nptg=10     !number of point for the graphs

      write(ifhi,'(a)')       'openhisto name PomIncPIUnit'
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis n?Pom!/dx+(x+)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncPIUnit(x)
        write(*,*) 'PIu',i
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomIncM
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,xminr,PomIncMUnit,PomIncMExact

      nptg=50                  !number of point for the graphs
      biniDf=xpar1              !value of biniDf (impact parameter)
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

c**********************red = PomIncMExact *****************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncMExact-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncMExact-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx-(x-,b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncMExact(x,biniDf)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**************************dot = PomIncMUnit **************************
c.....nptg=100     !number of point for the graphs

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PomIncMUnit-b0.',k
      else
        write(ifhi,'(a,f3.1)')'openhisto name PomIncMUnit-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis dn?Pom!/dx-(x-,b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncMUnit(x,biniDf)
        write(*,*) 'Mu',i
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomIncMI
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'

      double precision x,xminr
      double precision PomIncMIExact,PomIncMIUnit

      nptg=30                 !number of point for the graphs
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

c*********************red = PomIncMIExact *****************************
c.....nptg=100     !number of point for the graphs

      write(ifhi,'(a)')       'openhisto name PomIncMIExact'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis d[s]?Pom!/dx-(x-)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncMIExact(x)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c***************************dot = PomIncMIUnit ************************
c.....nptg=100     !number of point for the graphs

      write(ifhi,'(a)')       'openhisto name PomIncMIUnit'
      write(ifhi,'(a)')       'htyp pfc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')    'text 0 0 "xaxis x-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis d[s]?Pom!/dx-(x-)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        write(ifhi,*) x,PomIncMIUnit(x)
        write(*,*) 'MIu',i
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParPomIncJ
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      common/geom/rmproj,rmtarg,bmax,bkmx
      parameter(nbib=48)

      double precision PomIncJExact,PomIncJUnit


      b1=0
      b2=bkmx*1.2
      db=(b2-b1)/nbib

c*************************red = PomIncJExact **************************

      write(ifhi,'(a)')       'openhisto name PomIncJExact'
      write(ifhi,'(a)')       'htyp lru xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  n?Pom!(b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.8 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        write(ifhi,*)b,PomIncJExact(b)
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c****************************dot = PomIncJUnit ***********************

      write(ifhi,'(a)')       'openhisto name PomIncJUnit'
      write(ifhi,'(a)')       'htyp pfc xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  n?Pom!(b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.8 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        write(ifhi,*)b,PomIncJUnit(b)
        write(*,*) 'Ju',k
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'


      end

c----------------------------------------------------------------------
      subroutine xParPhi1
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision x,xminr,y
      double precision PhiExpo
      double precision PhiExact

      nptg=30                  !number of point for the graphs
      biniDf=xpar2              !value of biniDf (impact parameter)
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function
      zz=xpar7

c************************* red = PhiMExact ***************************

      write(ifhi,'(a)')       'openhisto name Phi1Exact'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')      'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)') 'txt "yaxis  [F](x)/x^[a]!"'
      write(ifhi,'(a,i4,a,f4.1,a)')
     * 'txt  "title E=',nint(engy),' b=',biniDf,'"'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        y=0.d0
        if(engy**2..lt.5.e06)
     &  y=Phiexact(zz,zz,.5,dsqrt(x),dsqrt(x),engy**2,biniDf)
     &       *dsqrt(x)**dble(-alplea(iclpro))
     &       *dsqrt(x)**dble(-alplea(icltar))
        write(ifhi,*)x,y
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c********************** blue = PhiMExpo ******************************

      write(ifhi,'(a)')       'openhisto name Phi1Expo'
      write(ifhi,'(a)')       'htyp lba'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
       write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        y=Phiexpo(zz,zz,.5,dsqrt(x),dsqrt(x),engy**2,biniDf)
     &       *dsqrt(x)**dble(-alplea(iclpro))
     &       *dsqrt(x)**dble(-alplea(icltar))
        write(ifhi,*) x,y
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end


cc----------------------------------------------------------------------
c      subroutine xParPhi2
cc----------------------------------------------------------------------
c
c      include 'epos.inc'
c      include 'epos.incems'
c      include 'epos.incsem'
c      include 'epos.incpar'
c      double precision x,xminr,xm,y,u(0:100),v(0:100),w(0:100)!,z
c      double precision PhiExpo,omGam,PhiExpoK
c      double precision PhiExact
c
c      nptg=30                  !number of point for the graphs
c      biniDf=xpar2              !value of biniDf (impact parameter)
c      xminr=1.d-3 !/dble(engy**2)  !value of xminr for plotting the function
c      xm=dble(xpar4)
c
cc************************* yellow = PhiMExact ***************************
c
c      write(ifhi,'(a)')       'openhisto name Phi1Exact'
c      write(ifhi,'(a)')       'htyp lru'
c      write(ifhi,'(a)')      'xmod lin ymod lin'
c      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
c      write(ifhi,'(a)')       'yrange auto auto'
c      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
c      write(ifhi,'(a)') 'txt "yaxis  [F](x)/x^[a]!"'
c      write(ifhi,'(a,i4,a,f4.1,a)')
c     * 'txt  "title E=',nint(engy),' b=',biniDf,'"'
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c       ! x=xminr
c       ! if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c        y=0.d0
c        if(engy**2..lt.5.e06)
c     &  y=Phiexact(0.,0.,1.,dsqrt(x),dsqrt(x),engy**2,biniDf)
c    ! &       *dsqrt(x)**dble(-alplea(iclpro))
c    ! &       *dsqrt(x)**dble(-alplea(icltar))
c        write(ifhi,*)x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0-'
c
cc********************** blue = PhiMExpo ******************************
c
c      write(ifhi,'(a)')       'openhisto name Phi1Expo'
c      write(ifhi,'(a)')       'htyp lbu'
c      write(ifhi,'(a)')       'xmod lin ymod lin'
c      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
c      write(ifhi,'(a,2e11.3)')'yrange auto auto'
c       write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c       ! x=xminr
c       ! if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c        y=Phiexpo(0.,0.,1.,dsqrt(x),dsqrt(x),engy**2,biniDf)
c  !   &       *dsqrt(x)**dble(-alplea(iclpro))
c  !   &       *dsqrt(x)**dble(-alplea(icltar))
c        write(ifhi,*) x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0-'
c
cc**********************************************************************
cc**********************************************************************
c      do k=1,koll
c        bk(k)=biniDf
c      enddo
c      call GfunPark(0)
c      call integom1(0)
c
cc********************* points = PhiExpoK*********************************
c
c      if (biniDf.lt.1.) then
c        k=int(10.*biniDf)
c        write(ifhi,'(a,I1)')  'openhisto name PhiExpok-b0.',k
c      else
c        write(ifhi,'(a,f3.1)')  'openhisto name PhiExpok-b',biniDf
c      endif
c      write(ifhi,'(a)')       'htyp pfc'
c      write(ifhi,'(a)')       'xmod log ymod lin'
c      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
c      write(ifhi,'(a,2e11.3)')'yrange auto auto'
c      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
c      write(ifhi,'(a)') 'text 0 0.1 "yaxis  [F](x+,x-)/x^[a]?remn!!"'
c      if (xpar8.eq.1.) then
c        write(ifhi,'(a,e7.2,a)')'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
c        write(ifhi,'(a,f5.2,a)')'text 0.5 0.9 "b=',biniDf,' fm"'
c      endif
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c        x=xminr
c        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        y=PhiExpoK(1,dsqrt(x),dsqrt(x))
c        write(ifhi,*) x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0'
c
cc************************* red = PhiMExact*omGam ************************
c
c      write(ifhi,'(a)')       'openhisto name GPhiExact'
c      write(ifhi,'(a)')       'htyp lru'
c      if(xpar5.eq.0.)then
c        write(ifhi,'(a)')       'xmod lin ymod lin'
c      else
c        write(ifhi,'(a)')       'xmod log ymod log'
c      endif
c      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
c      write(ifhi,'(a)')       'yrange auto auto'
c      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
c      write(ifhi,'(a,a)') 'text 0 0.1 "yaxis  G(x+,x-)*[F]'
c     *,'(x+,x-)/x^[a]?remn!!"'
c      if (xpar8.eq.1.) then
c        write(ifhi,'(a,e7.2,a)')'text 0.1 0.2 "s=',engy**2,' GeV^2!"'
c        write(ifhi,'(a,f5.2,a)')'text 0.1 0.1 "b=',biniDf,' fm"'
c        write(ifhi,'(a,f5.2,a)')'text 0.1 0.3 "x-=',xm,'"'
c      endif
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c        x=xminr
c        if(xpar5.ne.0.)then
c          if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        else
c          x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c        endif
cc        z=1.d0-dsqrt(x)
c        v(i)=0.d0
c        if(engy**2..lt.5.e06)
c     *  v(i)=Phiexact(0.,0.,1.,1.d0-x,1.d0-xm,engy**2,biniDf)
cc     *  v(i)=Phiexact(0.,0.,1.,z,z,engy**2,biniDf)
c        u(i)=omGam(x,xm,biniDf)
cc        u(i)=omGam(dsqrt(x),dsqrt(x),biniDf)
c        y=u(i)*v(i)
c        if(xpar5.ne.0.)y=dabs(y)
c        write(ifhi,*)x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0-'
c
cc************************* red = PhiMExpo*omGam ************************
c
c      write(ifhi,'(a)')       'openhisto name GPhiExpo'
c      write(ifhi,'(a)')       'htyp lba'
c      if(xpar5.eq.0.)then
c        write(ifhi,'(a)')       'xmod lin ymod lin'
c      else
c        write(ifhi,'(a)')       'xmod log ymod log'
c      endif
c      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
c      write(ifhi,'(a)')       'yrange auto auto'
c      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
c      write(ifhi,'(a,a)')
c     * 'text 0 0.1 "yaxis  G(x+,x-)*[F]?'
c     * ,'(1-x+,1-x-)/x^[a]?remn!!"'
c      if (xpar8.eq.1.) then
c        write(ifhi,'(a,e7.2,a)')'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
c        write(ifhi,'(a,f5.2,a)')'text 0.1 0.8 "b=',biniDf,' fm"'
c        write(ifhi,'(a,f5.2,a)')'text 0.1 0.7 "x-=',xm,'"'
c      endif
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c        x=xminr
c        if(xpar5.ne.0.)then
c          if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        else
c          x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c        endif
cc        z=1.d0-dsqrt(x)
cc        w(i)=Phiexpo(0.,0.,1.,z,z,engy**2,biniDf)
c        w(i)=Phiexpo(0.,0.,1.,1.d0-x,1.d0-xm,engy**2,biniDf)
c        y=u(i)*w(i)
c        if(xpar5.ne.0.)y=dabs(y)
c        write(ifhi,*)x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      if(xpar5.ne.0.)then
c      write(ifhi,'(a)')    'closehisto plot 0-'
c
cc************************* green = omGam ************************
c
c      write(ifhi,'(a)')       'openhisto name GM'
c      write(ifhi,'(a)')       'htyp lgo'
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c        x=xminr
c        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        write(ifhi,*)x,dabs(u(i))
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0-'
c
cc************************* circle = PhiMExact  ************************
c
c      write(ifhi,'(a)')       'openhisto name PhiExact'
c      write(ifhi,'(a)')       'htyp poc'
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c        x=xminr
c        if(xpar5.ne.0.)then
c          if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        else
c          x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c        endif
c        y=v(i)
c        if(xpar5.ne.0.)y=dabs(y)
c        write(ifhi,*)x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0-'
c
cc************************* triangle = PhiMExpo ************************
c
c      write(ifhi,'(a)')       'openhisto name PhiExpo'
c      write(ifhi,'(a)')       'htyp pot'
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c        x=xminr
c        if(xpar5.ne.0.)then
c          if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        else
c          x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c        endif
c        y=w(i)
c        if(xpar5.ne.0.)y=dabs(y)
c        write(ifhi,*)x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      endif
c      write(ifhi,'(a)')    'closehisto plot 0'
c
c      end
c

c----------------------------------------------------------------------
      subroutine xParPhi
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision x,xminr,y,z(0:200)!,Zn,Znorm
      double precision PhiExpo,PhiExact,PhiExpoK


      nptg=10                  !number of point for the graphs
      biniDf=xpar2              !value of biniDf (impact parameter)
      xminr=max(1.d-6,1.d0/dble(engy**2))  !value of xminr for plotting the function
      zz=xpar7

c********************** full-red = PhiExact ***************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PhiExact-b0.',k
      else
        write(ifhi,'(a,f3.1)')  'openhisto name PhiExact-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis x"'
      write(ifhi,'(a)') 'text 0 0 "yaxis  [F](x)/x^[a]"'
      write(ifhi,'(a,i4,a,f4.1,a)')
     * 'txt  "title E=',nint(engy),' b=',biniDf,'"'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        !x=xminr
        !if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        y=Phiexact(zz,zz,1.,dsqrt(x),dsqrt(x),engy**2,biniDf)
     &       *dsqrt(x)**dble(-alplea(iclpro))
     &       *dsqrt(x)**dble(-alplea(icltar))
        write(ifhi,*)x,y
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c******************** blue = PhiExpo ***************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PhiExpo-b0.',k
      else
        write(ifhi,'(a,f3.1)')  'openhisto name PhiExpo-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        !x=xminr
        !if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        z(i)=Phiexpo(zz,zz,1.,dsqrt(x),dsqrt(x),engy**2,biniDf)
     &       *dsqrt(x)**dble(-alplea(iclpro))
     &       *dsqrt(x)**dble(-alplea(icltar))
        write(ifhi,*) x,z(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0-'
c
cc*********************yellow = PhiUnit*********************************
c
c      if (biniDf.lt.1.) then
c        k=int(10.*biniDf)
c        write(ifhi,'(a,I1)')  'openhisto name PhiUnit-b0.',k
c      else
c        write(ifhi,'(a,f3.1)')  'openhisto name PhiUnit-b',biniDf
c      endif
c      write(ifhi,'(a)')       'htyp lyu'
c      write(ifhi,'(a)')       'xmod lin ymod lin'
c      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
c      write(ifhi,'(a,2e11.3)')'yrange auto auto'
c      write(ifhi,'(a)')       'array 2'
c
c      Zn=Znorm(engy**2,biniDf)
c
c      do i=0,nptg
c        !x=xminr
c        !if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c        write(ifhi,*) x,z(i)/Zn
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'

      if(koll.ge.1)then

      write(ifhi,'(a)')    'closehisto plot 0-'

c**********************************************************************
c**********************************************************************
      do k=1,koll
        bk(k)=biniDf
      enddo
      call GfunPark(0)
      call integom1(0)


c*********************green = PhiExpoK*********************************

      if (biniDf.lt.1.) then
        k=int(10.*biniDf)
        write(ifhi,'(a,I1)')  'openhisto name PhiExpok-b0.',k
      else
        write(ifhi,'(a,f3.1)')  'openhisto name PhiExpok-b',biniDf
      endif
      write(ifhi,'(a)')       'htyp lga'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a,2e11.3)')'yrange auto auto'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        !x=xminr
        !if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
        x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        z(i)=PhiExpoK(1,dsqrt(x),dsqrt(x))
        write(ifhi,*) x,z(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      endif
      write(ifhi,'(a)')    'closehisto plot 0'

c
      end

c----------------------------------------------------------------------
      subroutine xParH
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      parameter(idxD2=8)
      double precision GbetUni,GbetpUni,HbetUni,HbetpUni,HalpUni
      common/DGamUni/GbetUni(  idxD0:idxD2),HbetUni(  idxD0:idxD2),
     &               GbetpUni(idxD0:idxD2),HbetpUni(idxD0:idxD2),
     &               HalpUni(idxD0:idxD2)
      double precision x,xminr,y,xm,utgam2
      double precision Hrst

      nptg=20                  !number of point for the graphs
      biniDf=xpar2              !value of biniDf (impact parameter)
      xm=dble(xpar4)            !value of xminus
c.....xminr=0.d0   !value of xminr for plotting the function
      xminr=1.d0/dble(engy**2)  !value of xminr for plotting the function

      imax=idxD1
      if(iomega.eq.2)imax=1
      do i=idxDmin,imax
        call Gfunpar(0.,0.,1,i,biniDf,smaxDf,alpx,betx,betpx,epsp,epst
     &               ,epss,gamv)
      enddo
      call GfomPar(biniDf,smaxDf)
      imax0=idxD1
      if(iomega.eq.2)imax0=1
      imax1=idxD2
      if(iomega.eq.2)imax1=imax1-1
      do i=idxDmin,imax0
        GbetUni(i)=utgam2(betUni(i,1)+1.d0)
        GbetpUni(i)=utgam2(betpUni(i,1)+1.d0)
        HbetUni(i)=utgam2(GbetUni(i))
        HbetpUni(i)=utgam2(GbetpUni(i))
        HalpUni(i)=alpUni(i,1)*dble(chad(iclpro)*chad(icltar))
      enddo
      do i=0,1
        HbetUni(imax0+1+i)=betUni(i,1)+1.d0+betfom
        HbetUni(imax0+3+i)=betUni(i,1)+1.d0
        HbetUni(imax0+5+i)=betUni(i,1)+1.d0+betfom
        HbetpUni(imax0+1+i)=betpUni(i,1)+1.d0
        HbetpUni(imax0+3+i)=betpUni(i,1)+1.d0+betfom
        HbetpUni(imax0+5+i)=betpUni(i,1)+1.d0+betfom
        GbetUni(imax0+1+i)=utgam2(HbetUni(imax0+1+i))
        GbetUni(imax0+3+i)=utgam2(HbetUni(imax0+3+i))
        GbetUni(imax0+5+i)=utgam2(HbetUni(imax0+5+i))
        GbetpUni(imax0+1+i)=utgam2(HbetpUni(imax0+1+i))
        GbetpUni(imax0+3+i)=utgam2(HbetpUni(imax0+3+i))
        GbetpUni(imax0+5+i)=utgam2(HbetpUni(imax0+5+i))
        HalpUni(imax0+1+i)=zztUni*alpUni(i,1)
        HalpUni(imax0+3+i)=zzpUni*alpUni(i,1)
        HalpUni(imax0+5+i)=zzpUni*zztUni*alpUni(i,1)
      enddo

c***********************  red = Hrst  *********************************

      write(ifhi,'(a)')       'openhisto name Hrst'
      write(ifhi,'(a)')       'htyp lru'
      write(ifhi,'(a)')       'xmod log ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')
     *     'text 0 0 "yaxis  H?2!(x+,x-)"'
      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
      write(ifhi,'(a)')
     *     'text 0 0 "yaxis  H?2!(x+,x-)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.1 0.2 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')'text 0.1 0.1 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')'text 0.1 0.3 "x-=',xm,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminr
        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
c.......x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
        y=Hrst(smaxDf,biniDf,dsqrt(x),dsqrt(x))
        write(ifhi,*)x,y
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end




cc----------------------------------------------------------------------
c      subroutine xParHPhiIntnew
cc----------------------------------------------------------------------
c
c      include 'epos.inc'
c      include 'epos.incsem'
c      include 'epos.incems'
c      include 'epos.incpar'
c      double precision x,xminr,xm,y
c      double precision PhiExact,omGam
cc      double precision PhiExpo
c
c      nptg=30                  !number of point for the graphs
c      biniDf=xpar2              !value of biniDf (impact parameter)
c      xm=dble(xpar4)            !value of xminus
cc.....xminr=0.d0   !value of xminr for plotting the function
c      xminr=1.d-3 !/dble(engy**2)  !value of xminr for plotting the function
cc************************* black = PhiExact ***************************
c
c      write(ifhi,'(a)')       'openhisto name Phi1Exact'
c      write(ifhi,'(a)')       'htyp lru'
c      if(xpar5.eq.0.)then
c        write(ifhi,'(a)')       'xmod lin ymod lin'
c      else
c        write(ifhi,'(a)')       'xmod log ymod lin'
c      endif
c      write(ifhi,'(a,2e11.3)')'xrange',xminr,xmaxDf
c      write(ifhi,'(a)')       'yrange auto auto'
c      write(ifhi,'(a)')    'text 0 0 "xaxis x+"'
c      write(ifhi,'(a)')
c     * 'text 0 0.1 "yaxis  [F]?(x+,x-)/x^[a]?remn!!"'
c      if (xpar8.eq.1.) then
c        write(ifhi,'(a,e7.2,a)')'text 0.1 0.2 "s=',engy**2,' GeV^2!"'
c        write(ifhi,'(a,f5.2,a)')'text 0.1 0.1 "b=',biniDf,' fm"'
c        write(ifhi,'(a,f5.2,a)')'text 0.1 0.3 "x-=',xm,'"'
c      endif
c      write(ifhi,'(a)')       'array 2'
c
c      do i=0,nptg
c        x=xminr
c        if (i.ne.0) x=x*(xmaxDf/xminr)**(dble(i)/dble(nptg))
cc.......x=xminr+(xmaxDf-xminr)*(dble(i)/dble(nptg))
c      y=Phiexact(0.,0.,1.,dsqrt(x),dsqrt(x),engy**2,biniDf)
c     &       *omGam(dsqrt(x),dsqrt(x),biniDf)
c        write(ifhi,*)x,y
c      enddo
c
c      write(ifhi,'(a)')    '  endarray'
c      write(ifhi,'(a)')    'closehisto plot 0'
c
c      end
c
c----------------------------------------------------------------------
      subroutine xParHPhiInt
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      double precision y,HPhiInt
      common/geom/rmproj,rmtarg,bmax,bkmx
      parameter(nbib=32)


c************************ dotted = gauss integration ******************

      b1=0
      b2=max(abs(bkmx),3.)*1.2
      db=(b2-b1)/nbib

      write(ifhi,'(a)')       'openhisto name HPhiExpoInt'
      write(ifhi,'(a)')       'htyp pfc xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  Int(H[F]?pp!)(s,b)"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
      imax=idxD1
      if(iomega.eq.2)imax=1
      do k=1,nbib
        b=b1+(k-0.5)*db
        do i=idxDmin,imax
          call Gfunpar(0.,0.,1,i,b,smaxDf,alpx,betx,betpx,epsp,epst,epss
     &                ,gamv)
          call Gfunpar(0.,0.,2,i,b,smaxDf,alpx,betx,betpx,epsp,epst,epss
     &                ,gamv)
        enddo
        y=HPhiInt(smaxDf,b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParZ
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incpar'
      double precision Znorm,y
      common/geom/rmproj,rmtarg,bmax,bkmx
      parameter(nbib=12)

      b1=0
      b2=max(abs(bkmx),3.)*1.2
      db=(b2-b1)/nbib

c************************full-red = Znorm *****************************

      write(ifhi,'(a)')       'openhisto name Znorm'
      write(ifhi,'(a)')       'htyp lru xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  Z(s,b) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.1 0.1 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
c        y=Znorm(engy**2,xpar2)
      do k=1,nbib
        b=b1+(k-0.5)*db
        y=Znorm(smaxDf,b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      write(ifhi,'(a)')       'openhisto name un'
      write(ifhi,'(a)')       'htyp lba xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        y=1
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'


      end

c----------------------------------------------------------------------
      subroutine xParPro
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision PhiExact,PhiExpo,y,om1intb,om1intbc,om1intgc
     &,om1intbi!,PhiUnit
      common/geom/rmproj,rmtarg,bmax,bkmx
      parameter(nbib=12)

      b1=0
      b2=max(abs(bkmx),3.)*1.2
      db=(b2-b1)/nbib
      zz=xpar7

c********************* full-red = 1-PhiExact **************************

      write(ifhi,'(a)')       'openhisto name 1-PhiExact'
      write(ifhi,'(a)')       'htyp lru xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  1-[F]?pp!(1,1) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        y=1.d0-Phiexact(zz,zz,1.,1.d0,1.d0,engy**2,b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c************************** blue-dashed = 1-PhiExpo *******************

      write(ifhi,'(a)')       'openhisto name 1-PhiExpo'
      write(ifhi,'(a)')       'htyp lba xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  1-[F]?pp!(1,1) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        y=1.d0-Phiexpo(zz,zz,1.,1.d0,1.d0,engy**2,b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      write(ifhi,'(a)')       'openhisto'
      write(ifhi,'(a)')       'htyp lga xmod lin ymod lin'
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        y=1
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

c****************************** red = om1intbc ********************

      write(ifhi,'(a)')       'openhisto name om1intbc'
      write(ifhi,'(a)')       'htyp lru xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  [w]?1bc!(b) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do k=1,nbib
        b=b1+(k-0.5)*db
        y=om1intbc(b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c************************* blue dashed =  om1intb ********************

      write(ifhi,'(a)')       'openhisto name om1intb'
      write(ifhi,'(a)')       'htyp lba xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  [w]?1b!(b) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do k=1,nbib
        b=b1+(k-0.5)*db
        y=om1intb(b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c****************************** green dot = om1intgc ********************

      write(ifhi,'(a)')       'openhisto name om1intgc'
      write(ifhi,'(a)')       'htyp lgo xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  [w]?1gc!(b) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do k=1,nbib
        b=b1+(k-0.5)*db
        y=om1intgc(b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

c****************************** red = om1intbi(0) ********************

      write(ifhi,'(a)')       'openhisto name om1intbc'
      write(ifhi,'(a)')       'htyp lru xmod lin ymod log'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  [w]?1bc!(b) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do k=1,nbib
        b=b1+(k-0.5)*db
        y=om1intbi(b,0)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c************************* blue dashed =  om1intbi(1) ********************

      write(ifhi,'(a)')       'openhisto name om1intb'
      write(ifhi,'(a)')       'htyp lba xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  [w]?1b!(b) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do k=1,nbib
        b=b1+(k-0.5)*db
        y=om1intbi(b,1)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c****************************** green dot = om1intbi(2) ********************

      write(ifhi,'(a)')       'openhisto name om1intgc'
      write(ifhi,'(a)')       'htyp lgo xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  [w]?1gc!(b) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'

      do k=1,nbib
        b=b1+(k-0.5)*db
        y=om1intbi(b,2)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'



      end

c----------------------------------------------------------------------
      subroutine xParPro1
c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      double precision PhiExact,PhiExpo,y
      common/geom/rmproj,rmtarg,bmax,bkmx
      parameter(nbib=12)

      b1=0
      b2=max(abs(bkmx),3.)*1.2
      db=(b2-b1)/nbib
      zz=xpar7

c********************* full-red = 1-PhiExact **************************

      write(ifhi,'(a)')       'openhisto name 1-PhiExact'
      write(ifhi,'(a)')       'htyp lru xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  1-[F]?pp!(1,1) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        y=1.d0-Phiexact(zz,zz,.5,1.d0,1.d0,engy**2,b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

c************************** blue-dashed = 1-PhiExpo *******************

      write(ifhi,'(a)')       'openhisto name 1-PhiExpo'
      write(ifhi,'(a)')       'htyp lba xmod lin ymod lin'
      write(ifhi,'(a)')    'text 0 0 "xaxis  impact parameter b (fm)"'
      write(ifhi,'(a)')    'text 0 0 "yaxis  1-[F]?pp!(1,1) "'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')'text 0.5 0.9 "s=',engy**2,' GeV^2!"'
      endif
      write(ifhi,'(a)')       'array 2'
      do k=1,nbib
        b=b1+(k-0.5)*db
        y=1.d0-Phiexpo(zz,zz,.5,1.d0,1.d0,engy**2,b)
        write(ifhi,*)b,y
      enddo
      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'




      end

c----------------------------------------------------------------------
      subroutine xParGam
c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incsem'
      include 'epos.incpar'
      dimension bet(idxD0:idxD1)
      double precision utgam2,xgammag2!,xrem
      dimension ip(idxD0:idxD1),imax(idxD0:idxD1)

      nptg=50                  !number of point for the graphs
      b=xpar2
      gamp=xpar6
      zmax=6.
c      xrem=dble(xpar4)
      if(idxD0.ne.0.or.idxD1.ne.2) stop "Check xPargam"

      do i=idxD0,idxD1
        imax(i)=4
        bet(i)=0
      enddo
      nmax=idxD1
      imax(idxD0)=int(zmax)
      imax(1)=imax(idxD0)
      imax(2)=imax(idxD0)

      do i=idxD0,nmax
        gam=gamD(i,iclpro,icltar)*b**2
        bet(i)=gam+betDp(i,iclpro,icltar)-alppar+1.
      enddo
      write(ifhi,'(a)')       'openhisto name gExact'
      write(ifhi,'(a)')       'htyp pfs'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,zmax
      write(ifhi,'(a)')       'yrange auto auto'
c      write(ifhi,'(a)')'yrange 1.e-10 2'
      write(ifhi,'(a)')    'text 0 0 "xaxis z"'
      write(ifhi,'(a)')    'text 0 0 "yaxis g(z) "'
      write(ifhi,'(a)')       'array 2'

      do ip0=0,imax(0)
        ip(0)=ip0
        do ip1=0,imax(1)
          ip(1)=ip1
          do ip2=0,imax(2)
            ip(2)=ip2
          t=0.
          do i=idxD0,nmax
            t=t+real(ip(i))*bet(i)
          enddo
          write(ifhi,'(2e14.6)')t,utgam2(dble(alplea(2))+1.D0)
     &       /utgam2(dble(alplea(2))+1.D0+dble(t))
        enddo
      enddo
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************

      write(ifhi,'(a)')       'openhisto name gExpo'
      write(ifhi,'(a)')       'htyp lbu'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,zmax
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis z"'
      write(ifhi,'(a)')    'text 0 0 "yaxis g(z)"'
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        t=zmax*(real(i)/real(nptg))
        write(ifhi,'(2e14.6)') t,dexp(-dble(t))
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


c**********************************************************************



      write(ifhi,'(a)')       'openhisto name gPower'
      write(ifhi,'(a)')       'htyp poc'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,zmax
      write(ifhi,'(a)')       'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis t"'
      write(ifhi,'(2a)')    'text 0 0 ',
     &     '"yaxis [P][G](1+[a]?L!)/[G](1+[a]?L!+[b])"'
      write(ifhi,'(a)')       'array 2'

      do ip0=0,imax(0)
        ip(0)=ip0
        do ip1=0,imax(1)
          ip(1)=ip1
          do ip2=0,imax(2)
            ip(2)=ip2
c$$$            do ip3=0,imax(3)
c$$$              ip(3)=ip3
              t=0.
              do i=idxD0,nmax
                t=t+real(ip(i))*bet(i)
              enddo
              write(ifhi,'(2e14.6)')t,xgammag2(iclpro,bet,ip,gamp)
c$$$            enddo
          enddo
        enddo
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      end

c----------------------------------------------------------------------
      subroutine xParOmega1xy
c----------------------------------------------------------------------
c xpar2=b
c xpar4=xh
c xpar3=y
c xpar7 : nucl coef
c xpar8 : 2=xp/xm instead of xh/yp
c----------------------------------------------------------------------
      include 'epos.incems'
      include 'epos.incsem'
      include 'epos.inc'
      include 'epos.incpar'
    !  double precision om1x,om1y
      double precision x,ranhis(0:51),y,ymax,xh
     &,om1xpk,om1xmk,t,om1xk,om1yk,xpr1,xmr1!,xp,xm,xmin,xmax
      common /psar7/ delx,alam3p,gam3p

      nptg=50                  !number of point for the graphs
      biniDf=xpar2              !value of biniDf (impact paramter)
      xh=0.99d0
      xpr1=1.d0
      xmr1=1.d0
      if(xpar4.lt.1..and.xpar4.gt.0.)then
        xh=dble(xpar4**2)          !value of x
        xpr1=dble(xpar4)
        xmr1=dble(xpar4)
      endif
      do i=0,51
        ranhis(i)=0.d0
      enddo
c$$$      xp=dsqrt(xh)*dble(exp(xpar3)) !y=xpar3
c$$$      xm=1.d0
c$$$      if(xp.ne.0.d0)xm=xh/xp



      if(koll.eq.0)then

      call xhistomran1(ranhis,biniDf)

      stop'om1x not defined'


      write(ifhi,'(a)')       'openhisto name Om1x'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**(dble(i)/dble(nptg))
c.......x=xminDf+(xmaxDf-xminDf)*(dble(i)/dble(nptg))
        write(ifhi,*) x,   0   !om1x(x,biniDf)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto name Om1xRan'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x! random"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,51
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**((dble(i)+.5d0)/51.d0)
c.......x=xminDf+(xmaxDf-xminDf)*((dble(i)+.5d0)/51.d0)
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      x=xh
      ymax=-.5D0*dlog(x)
      do i=0,51
        ranhis(i)=0.d0
      enddo
      call xhistomran2(ranhis,x,biniDf)

      stop'om1y not defined'

      write(ifhi,'(a)')       'openhisto name Om1y'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-ymax-1.d0,ymax+1.d0
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis Y"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1y!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        y=-ymax+(2.d0*ymax)*(dble(i)/dble(nptg))
        write(ifhi,*) y, 0  !om1y(x,y,biniDf)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto name Om1yRan'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-ymax-1.d0,ymax+1.d0
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis Y"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1y! random"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,50
        y=-ymax+(2.d0*ymax)*(dble(i)/50.d0)
        write(ifhi,*) y,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'


c**********************************************************************

      else

      do k=1,koll
        bk(k)=biniDf
      enddo
      call GfunPark(0)
      call integom1(0)

      if(xpar8.eq.2.)then

        call xhistomran8(ranhis,xpr1,xmr1)


      write(ifhi,'(a)')       'openhisto name Om1xp'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x+!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**(dble(i)/dble(nptg))
c.......x=xminDf+(xmaxDf-xminDf)*(dble(i)/dble(nptg))
        t=om1xpk(x,xpr1,1)
        write(ifhi,*) x,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto name Om1xpRan'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x+! random"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,51
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**(dble(i)/dble(nptg))
c.......x=xminDf+(xmaxDf-xminDf)*((dble(i)+.5d0)/51.d0)
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      do i=0,51
        ranhis(i)=0.d0
      enddo

      call xhistomran9(ranhis,xh,xpr1,xmr1)


      write(ifhi,'(a)')       'openhisto name Om1xm'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x-!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.8 "x+=',xh,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**(dble(i)/dble(nptg))
c.......x=xminDf+(xmaxDf-xminDf)*(dble(i)/dble(nptg))
        t=om1xmk(xh,x,xpr1,xmr1,1)
        write(ifhi,*) x,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto name Om1xmRan'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X-"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x-! random"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.8 "x+=',xh,'"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,51
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**(dble(i)/dble(nptg))
c.......x=xminDf+(xmaxDf-xminDf)*((dble(i)+.5d0)/51.d0)
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

c**********************************************************************
      else

      call xhistomran10(ranhis)


      write(ifhi,'(a)')       'openhisto name Om1xk'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**(dble(i)/dble(nptg))
c.......x=xminDf+(xmaxDf-xminDf)*(dble(i)/dble(nptg))
        t=om1xk(x,1)
        write(ifhi,*) x,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto name Om1xpRan'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod log ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',xminDf,xmaxDf
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis X+"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1x+! random"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,51
        x=xminDf
        if (i.ne.0) x=x*(xmaxDf/xminDf)**(dble(i)/dble(nptg))
c.......x=xminDf+(xmaxDf-xminDf)*((dble(i)+.5d0)/51.d0)
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'

      ymax=-.5D0*dlog(xh)
      do i=0,51
        ranhis(i)=0.d0
      enddo

      call xhistomran11(ranhis,xh)



      write(ifhi,'(a)')       'openhisto name Om1yk'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-ymax-1.d0,ymax+1.d0
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis Y"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1y!"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,nptg
        y=-ymax+(2.d0*ymax)*(dble(i)/dble(nptg))
        t=om1yk(xh,y,1)
        write(ifhi,*) y,t
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      write(ifhi,'(a)')       'openhisto name Om1yRan'
      write(ifhi,'(a)')       'htyp his'
      write(ifhi,'(a)')       'xmod lin ymod lin'
      write(ifhi,'(a,2e11.3)')'xrange',-ymax-1.d0,ymax+1.d0
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis Y"'
      write(ifhi,'(a)')    'text 0 0 "yaxis [w]?1y! random"'
      if (xpar8.eq.1.) then
        write(ifhi,'(a,e7.2,a)')  'text 0.1 0.9 "s=',engy**2,' GeV^2!"'
        write(ifhi,'(a,f5.2,a)')  'text 0.5 0.9 "b=',biniDf,' fm"'
      endif
      write(ifhi,'(a)')       'array 2'

      do i=0,50
        y=-ymax+(2.d0*ymax)*(dble(i)/50.d0)
        write(ifhi,*) y,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0'


      endif

      endif

      return
      end

c----------------------------------------------------------------------
      subroutine xRanPt
c----------------------------------------------------------------------
c xpar2=xcut
c----------------------------------------------------------------------
      include 'epos.inc'
      parameter (nptg1=501)      !number of point for the graphs
      double precision ranhis(0:nptg1)
      common /cranpt/conv

      nptg=nptg1-1
      xcut=xpar2              !value of biniDf (impact paramter)
      xfact=xpar3
      xadd=xpar4
      xmax=10.
      conv=10./float(nptg)
      if(xcut.le.0.)xcut=float(nptg)
      if(xfact.le.0.)xfact=1.
      do i=0,nptg1
        ranhis(i)=0.d0
      enddo
c$$$      xp=dsqrt(xh)*dble(exp(xpar3)) !y=xpar3
c$$$      xm=1.d0
c$$$      if(xp.ne.0.d0)xm=xh/xp

      if(xpar1.ge.1.)then

      call xranptg(ranhis,xcut,xfact,xadd)



      write(ifhi,'(a)')       'openhisto name ranpt'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,min(xmax,xfact*xcut+xadd+1.)
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis pt"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P"'
      write(ifhi,'(a)')       'array 2'


      do i=0,nptg
        x=float(i)*conv
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      endif

      if(xpar1.ge.2.)then

      call xranpte(ranhis,xcut,xfact,xadd)



      write(ifhi,'(a)')       'openhisto name ranpt'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,min(xmax,xfact*xcut+xadd+1.)
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis pt"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P"'
      write(ifhi,'(a)')       'array 2'


      do i=0,nptg
        x=float(i)*conv
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'


      endif

      if(xpar1.ge.3.)then

      call xranpts(ranhis,xcut,xfact,xadd)



      write(ifhi,'(a)')       'openhisto name ranpt'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod log'
      write(ifhi,'(a,2e11.3)')'xrange',0.,min(xmax,xfact*xcut+xadd+1.)
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis pt"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P"'
      write(ifhi,'(a)')       'array 2'


      do i=0,nptg
        x=float(i)*conv
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'
      write(ifhi,'(a)')    'closehisto plot 0-'

      endif

      call xranptc(ranhis,xcut,xfact,xadd)



      write(ifhi,'(a)')       'openhisto name ranpt'
      write(ifhi,'(a)')       'htyp lin'
      write(ifhi,'(a)')       'xmod lin ymod log'
      if(xpar1.ge.1)then
        write(ifhi,'(a,2e11.3)')'xrange',0.,min(xmax,xfact*xcut+xadd+1.)
      else
        write(ifhi,'(a,2e11.3)')'xrange',0.,1.
      endif
      write(ifhi,'(a)')'yrange auto auto'
      write(ifhi,'(a)')    'text 0 0 "xaxis pt"'
      write(ifhi,'(a)')    'text 0 0 "yaxis P"'
      write(ifhi,'(a)')       'array 2'


      do i=0,nptg
        x=float(i)*conv
        write(ifhi,*) x,ranhis(i)
      enddo

      write(ifhi,'(a)')    '  endarray'


      return
      end

c----------------------------------------------------------------------

      double precision function xgammag2(iclrem,bet,ip,gamp)

c----------------------------------------------------------------------
      include 'epos.inc'
      include 'epos.incsem'
      double precision utgam2
      dimension bet(idxD0:idxD1),ip(idxD0:idxD1)

      xgammag2=1.d0

      imax=idxD1

      do i=idxD0,imax
      if(ip(i).ne.0) xgammag2=xgammag2
     &   *(utgam2(dble(alplea(iclrem))+1.d0+dble(gamp))
     &   /(max(0.d0,dble(int(gamp+0.5))+1))
     &   /utgam2(dble(alplea(iclrem)+bet(i)+gamp)+1.D0))
     &                                          **dble(ip(i))
      enddo

      return
      end

c----------------------------------------------------------------------

      function xsigmafit(x)

c----------------------------------------------------------------------

      include 'epos.inc'
      include 'epos.incpar'
      double precision x,xDfit,sfsh,varifit,range,sig2
      double precision bf(maxdataDf),Db(maxdataDf)
      external varifit


      sig2=bmaxDf/2.
      range=sig2
      xp=real(dsqrt(x))
      xm=xp
      zz=xpar7


      sfsh=xDfit(zz,0,1,smaxDf,xp,xm,0.)
      if(dabs(sfsh).ge.1.d-5)then
      do i=0,nptf-1
        bf(i+1)=dble(-bmaxDf+real(i)*2.*bmaxDf/real(nptf-1))
        Db(i+1)=xDfit(zz,0,1,smaxDf,xp,xm,real(bf(i+1)))/sfsh
      enddo

c.....Fit of D(X,b) between -bmaxDf and bmaxDf
      call minfit(varifit,bf,Db,nptf,sig2,range)

      xsigmafit=real(sig2)
      else
      xsigmafit=0.
      endif


      return
      end


c----------------------------------------------------------------------

      subroutine xhistomran1(histo,b)

c----------------------------------------------------------------------
c.....Make Histogram of om1xr
c----------------------------------------------------------------------

      include 'epos.incpar'

      double precision histo(0:51),x,x1  !,om1xr
      integer*4 n


      n=100000
      do i=0,51
        histo(i)=0.d0
      enddo
      do 111 j=1,n
        if(mod(j,10000).eq.0)write(*,*)"x1",j,b
        x=0  !om1xr(b)
        stop'om1xr(b) not defined'
        if(x.lt.xminDf)goto 111
c.........Exponential
          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
c.........k=int(x*50.d0)
        histo(k)=histo(k)+1.d0
 111  continue
      do i=0,51

c.......Exponential

        x=xminDf
        x1=xminDf
        x=x**(1.d0-dble(i)/51.d0)
        x1=x1**(1.d0-dble(i+1)/51.d0)
        if(i.eq.51)then
          x1=1.d0
          x=0.d0
        endif
        histo(i)=histo(i)/dble(n)/(x1-x)

c.......Linear
c        histo(i)=histo(i)/dble(n)*51.d0
      enddo

      return
      end


c----------------------------------------------------------------------

      subroutine xhistomran2(histo,xh,b)

c----------------------------------------------------------------------
c.....Make Histogram of om1yr
c----------------------------------------------------------------------

      double precision histo(0:51),x,xh,dx,ymax   !,om1yr
      integer*4 n

      ymax=-.5D0*dlog(xh)
      dx=ymax/25.d0

      n=100000
      do i=0,50
        histo(i)=0.d0
      enddo
      do j=1,n
        if(mod(j,10000).eq.0)write(*,*)"y1",j,b
        x= 0 !  om1yr(xh,b)
        stop'om1yr(xh,b) not defined'
        k=int((x/ymax+1.d0)*25.d0)
c.......write(*,*)x,k
        histo(k)=histo(k)+1.d0
      enddo
      do i=0,50
        histo(i)=histo(i)/dble(n)/dx
      enddo

      return
      end


cc----------------------------------------------------------------------
c
c      subroutine xhistomran6(histo,bx,by,bmax,del)
c
cc----------------------------------------------------------------------
cc.....Make Histogram of b1 (impact parameter of vertex in Y and X)
cc----------------------------------------------------------------------
c
c      double precision histo(0:51),dx
c      integer*4 n
c
c      n=100000
c      dx=dble(bmax)/50.d0
c      do i=0,50
c        histo(i)=0.d0
c      enddo
c      do j=1,n
c        if(mod(j,10000).eq.0)write(*,*)"b1",j
c        z=rangen()
c        zp=rangen()
c        bb1x=(bx+sqrt(-del*log(z))*cos(2.*3.14*zp))/2.
c        bb1y=(by+sqrt(-del*log(z))*sin(2.*3.14*zp))/2.
c        x=sqrt((bx-bb1x)*(bx-bb1x)+(by-bb1y)*(by-bb1y))
c        k=int(x/bmax*50.)
c        if(k.le.50)then
c          histo(k)=histo(k)+1.d0
c        else
c          histo(51)=histo(51)+1.d0
c        endif
c      enddo
c      do i=0,50
c        histo(i)=histo(i)/dble(n)/dx
c      enddo
c
c      return
c      end
c

c----------------------------------------------------------------------

      subroutine xhistomran8(histo,xpr,xmr)

c----------------------------------------------------------------------
c.....Make Histogram of om1xprk
c----------------------------------------------------------------------

      include 'epos.incpar'

      double precision histo(0:51),x,x1,om1xprk,xpr,xmr
      integer*4 n


      n=100000
      do i=0,51
        histo(i)=0.d0
      enddo
      do 111 j=1,n
        if(mod(j,10000).eq.0)write(*,*)"x+",j,xmr
          x=om1xprk(1,xpr,xminDf,1)
        if(x.lt.xminDf)goto 111
c.........Exponential
          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
c.........k=int(x*50.d0)
        histo(k)=histo(k)+1.d0
 111  continue
      do i=0,51

c.......Exponential

        x=xminDf
        x1=xminDf
        x=x**(1.d0-dble(i)/51.d0)
        x1=x1**(1.d0-dble(i+1)/51.d0)
        if(i.eq.51)then
          x1=1.d0
          x=0.d0
        endif
        histo(i)=histo(i)/dble(n)/(x1-x)*xpr

c.......Linear
c        histo(i)=histo(i)/dble(n)*51.d0
      enddo

      return
      end

c----------------------------------------------------------------------

      subroutine xhistomran9(histo,xp,xpr,xmr)

c----------------------------------------------------------------------
c.....Make Histogram of om1xmrk
c----------------------------------------------------------------------

      include 'epos.incpar'

      double precision histo(0:51),x,x1,om1xmrk,xp,xpr,xmr
      integer*4 n


      n=100000
      do i=0,51
        histo(i)=0.d0
      enddo
      do 111 j=1,n
        if(mod(j,10000).eq.0)write(*,*)"x-",j
          x=om1xmrk(1,xp,xpr,xminDf,1)
        if(x.lt.xminDf)goto 111
c.........Exponential
          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
c.........k=int(x*50.d0)
        histo(k)=histo(k)+1.d0
 111  continue
      do i=0,51

c.......Exponential

        x=xminDf
        x1=xminDf
        x=x**(1.d0-dble(i)/51.d0)
        x1=x1**(1.d0-dble(i+1)/51.d0)
        if(i.eq.51)then
          x1=1.d0
          x=0.d0
        endif
        histo(i)=histo(i)/dble(n)/(x1-x)*xmr

c.......Linear
c        histo(i)=histo(i)/dble(n)*51.d0
      enddo

      return
      end

c----------------------------------------------------------------------

      subroutine xhistomran10(histo)

c----------------------------------------------------------------------
c.....Make Histogram of om1xrk
c----------------------------------------------------------------------

      include 'epos.incpar'

      double precision histo(0:51),x,x1,om1xrk
      integer*4 n


      n=100000
      do i=0,51
        histo(i)=0.d0
      enddo
      do 111 j=1,n
        if(mod(j,10000).eq.0)write(*,*)"xk",j
        x=om1xrk(1)
        if(x.lt.xminDf)goto 111
c.........Exponential
          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
c.........k=int(x*50.d0)
        histo(k)=histo(k)+1.d0
 111  continue
      do i=0,51

c.......Exponential

        x=xminDf
        x1=xminDf
        x=x**(1.d0-dble(i)/51.d0)
        x1=x1**(1.d0-dble(i+1)/51.d0)
        if(i.eq.51)then
          x1=1.d0
          x=0.d0
        endif
        histo(i)=histo(i)/dble(n)/(x1-x)

c.......Linear
c        histo(i)=histo(i)/dble(n)*51.d0
      enddo

      return
      end


c----------------------------------------------------------------------

      subroutine xhistomran11(histo,xh)

c----------------------------------------------------------------------
c.....Make Histogram of om1yrk
c----------------------------------------------------------------------

      double precision histo(0:51),x,xh,dx,om1yrk,ymax
      integer*4 n

      ymax=-.5D0*dlog(xh)
      dx=ymax/25.d0

      n=100000
      do i=0,50
        histo(i)=0.d0
      enddo
      do j=1,n
        if(mod(j,10000).eq.0)write(*,*)"yk",j
        x=om1yrk(xh)
        k=int((x/ymax+1.d0)*25.d0)
c.......write(*,*)x,k
        histo(k)=histo(k)+1.d0
      enddo
      do i=0,50
        histo(i)=histo(i)/dble(n)/dx
      enddo

      return
      end

c----------------------------------------------------------------------

      subroutine xranptg(histo,xcut,xfact,xadd)

c----------------------------------------------------------------------
c.....Make Histogram of random distribution
c----------------------------------------------------------------------

      include 'epos.incpar'

      parameter (nptg1=501)      !number of point for the graphs
      common /cranpt/conv
      double precision histo(0:nptg1)
      integer*4 n


      n=100000
      do i=0,nptg1
        histo(i)=0.d0
      enddo
      do j=1,n
        if(mod(j,10000).eq.0)write(*,*)"ptg",j
c .........exp(-x**2)
 12   x=sqrt(-log(rangen())/(3.1415927/4.)) !gauss

      if(xcut.gt.0.)then
        if(rangen().lt.x/xcut)goto 12
      endif
      x=x*xfact+xadd
c.........Exponential
c          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
        k=int(x/conv)
        k=min(k,nptg1)
        histo(k)=histo(k)+1.d0
      enddo
      do i=0,nptg1

c.......Exponential

c        x=xminDf
c        x1=xminDf
c        x=x**(1.d0-dble(i)/51.d0)
c        x1=x1**(1.d0-dble(i+1)/51.d0)
c        if(i.eq.51)then
c          x1=1.d0
c          x=0.d0
c        endif
c        histo(i)=histo(i)/dble(n)/(x1-x)

c.......Linear
        histo(i)=histo(i)/dble(n)*float(nptg1)
      enddo

      return
      end

c----------------------------------------------------------------------

      subroutine xranpte(histo,xcut,xfact,xadd)

c----------------------------------------------------------------------
c.....Make Histogram of random distribution
c----------------------------------------------------------------------

      include 'epos.incpar'

      parameter (nptg1=501)      !number of point for the graphs
      common /cranpt/conv
      double precision histo(0:nptg1)
      integer*4 n


      n=100000
      do i=0,nptg1
        histo(i)=0.d0
      enddo
      do j=1,n
        if(mod(j,10000).eq.0)write(*,*)"pte",j
c .........exp(-x)
  12  xmx=50
      r=2.
      x=0.
      do while (r.gt.1.)
  11    x=sqrt(exp(rangen()*log(1+xmx**2))-1)
        if(x.eq.0.)goto11
        r=rangen()  /  ( exp(-x)*(1+x**2) )
      enddo
      x=x/2.

      if(xcut.gt.0.)then
        if(rangen().lt.x/xcut)goto 12
      endif
      x=x*xfact+xadd
c.........Exponential
c          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
        k=int(x/conv)
        k=min(k,nptg1)
        histo(k)=histo(k)+1.d0
      enddo
      do i=0,nptg1

c.......Exponential

c        x=xminDf
c        x1=xminDf
c        x=x**(1.d0-dble(i)/51.d0)
c        x1=x1**(1.d0-dble(i+1)/51.d0)
c        if(i.eq.51)then
c          x1=1.d0
c          x=0.d0
c        endif
c        histo(i)=histo(i)/dble(n)/(x1-x)

c.......Linear
        histo(i)=histo(i)/dble(n)*float(nptg1)
      enddo

      return
      end

c----------------------------------------------------------------------

      subroutine xranpts(histo,xcut,xfact,xadd)

c----------------------------------------------------------------------
c.....Make Histogram of random distribution
c----------------------------------------------------------------------

      include 'epos.incpar'

      parameter (nptg1=501)      !number of point for the graphs
      common /cranpt/conv
      double precision histo(0:nptg1)
      integer*4 n


      n=100000
      do i=0,nptg1
        histo(i)=0.d0
      enddo
      do j=1,n
        if(mod(j,10000).eq.0)write(*,*)"pts",j
c .........exp(-sqrt(x))
 12   xmx=500
      r=2.
      x=0.
      do while (r.gt.1.)
        x=sqrt(exp(rangen()*log(1+xmx**2))-1)
        r=rangen()  /  ( exp(-sqrt(x))*(1+x**2)/5. )
      enddo
      x=x/20.

      if(xcut.gt.0.)then
        if(rangen().lt.x/xcut)goto 12
      endif
      x=x*xfact+xadd
c.........Exponential
c          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
        k=int(x/conv)
        k=min(k,nptg1)
        histo(k)=histo(k)+1.d0
      enddo
      do i=0,nptg1

c.......Exponential

c        x=xminDf
c        x1=xminDf
c        x=x**(1.d0-dble(i)/51.d0)
c        x1=x1**(1.d0-dble(i+1)/51.d0)
c        if(i.eq.51)then
c          x1=1.d0
c          x=0.d0
c        endif
c        histo(i)=histo(i)/dble(n)/(x1-x)

c.......Linear
        histo(i)=histo(i)/dble(n)*float(nptg1)
      enddo

      return
      end

c----------------------------------------------------------------------

      subroutine xranptc(histo,xcut,xfact,xadd)

c----------------------------------------------------------------------
c.....Make Histogram of random distribution
c----------------------------------------------------------------------

      include 'epos.incpar'

      parameter (nptg1=501)      !number of point for the graphs
      common /cranpt/conv
      double precision histo(0:nptg1)
      integer*4 n


      n=100000
      do i=0,nptg1
        histo(i)=0.d0
      enddo
      do j=1,n
        if(mod(j,10000).eq.0)write(*,*)"ptc",j

        x=ranptcut(xcut)*xfact+xadd
c.........Exponential
c          k=int((-dlog(x)/dlog(xminDf)+1.d0)*51.d0)
c.........Linear
        k=int(x/conv)
        k=min(k,nptg1)
        histo(k)=histo(k)+1.d0
      enddo
      do i=0,nptg1

c.......Exponential

c        x=xminDf
c        x1=xminDf
c        x=x**(1.d0-dble(i)/51.d0)
c        x1=x1**(1.d0-dble(i+1)/51.d0)
c        if(i.eq.51)then
c          x1=1.d0
c          x=0.d0
c        endif
c        histo(i)=histo(i)/dble(n)/(x1-x)

c.......Linear
        histo(i)=histo(i)/dble(n)*float(nptg1)
      enddo

      return
      end



