*----------------------------------------------------------------------------*
c----------------------------------------------------------------------------c
C..TXYGIVE
C rewritten from PYGIVE routine from PYTHIA
C...Sets values of commonblock variables.
 
      SUBROUTINE TXGIVE(CHIN)
      implicit none 
      Integer           Ipar     ! global TopRex integer parameters  
      double precision  Rpar     ! global TopRex real    parameters
      common /TXPAR/Ipar(200), Rpar(200)
      save   /TXPAR/
***
      integer CSAMODE
      double precision  MUONRW, GAMMAJRW, ZJRW, ZPRW, HLTRW, 
     &  SUSYRW, WWRW
      common /EXPAR/CSAMODE, MUONRW, GAMMAJRW, ZJRW, ZPRW, 
     &  HLTRW, SUSYRW, WWRW

      save   /EXPAR/
***
      integer IER, I, IL, IREAD
      character *(*) chin
      character *40 inam, rnam
      character *20 cvi  ! , cvj
      character *80 STRIN
****
*
      Iread = 1
*
      
      Inam = 'CSAMODE ='
        call TXRSTR2(inam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) CSAMODE 
         endif
*
      
      rnam = 'MUONRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) MUONRW 
         endif
*
*
    
      rnam = 'GAMMAJRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) GAMMAJRW 
         endif
*
      
      rnam = 'ZJRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) ZJRW 
         endif
	 
      rnam = 'ZPRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) ZPRW 
         endif
	 
	 
      rnam = 'HLTRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) HLTRW 
         endif
	 
      rnam = 'SUSYRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) SUSYRW 
         endif

      rnam = 'WWRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) WWRW 
         endif
	 
	 
***

      do I = 1,200
        call intochar(I, il, cvi)
         Inam = 'ipar('//cvi(1:il)//')'//' = '
         Rnam = 'rpar('//cvi(1:il)//')'//' = '
        call TXRSTR2(rnam, chin, strin, ier) 
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) Rpar(i) 
         endif
        call TXRSTR2(inam, chin, strin, ier) 
         if(ier.ne.1) then
           iread = 0
           read(strin(1:80),*) Ipar(i) 
         endif
      enddo
*



      return
      end

c----------------------------------------------------------------------------c

*... 11.11.2001                                                              *
*                                                                            *
      SUBROUTINE TXRSTR2(inam, aa, strout, ier) 
*     ------------------------------------                                   *
* input : INAM is character string (up to 40 symbols) with '=' as the end    *
* outout: STROUT is character string with value (values)                     *
*         IER = 0 (1) variable is readed and returned                        *
*............................................................................*
      implicit none
      integer IER, I, j1, jj 
      integer lens
      parameter (lens=40)
*
      character *40 inam, vv, vst, ww 
      character *80 strout, aa, vnu 
*
      ier = 1
       do i = 1,80
        if(aa(i:i).ne.' ') then 
         j1 = 1  ! non-blank character
         if(aa(i:i).eq.'*') goto 200   ! first non-blank item = '*' -  comment
         goto 16
        endif        
       enddo
        goto 200 !  blank string 
 16    j1 = 0
       do i = 1,80
        if(aa(i:i).ne.' ') then 
           if(aa(i:i).eq.'='.and.j1.eq.0) then
            j1 = i
            goto 17
           endif
        endif
      enddo
       goto 200
 17    jj = 0
       do i = 1,(j1-1)
       if(aa(i:i).ne.' ') then
       jj = jj + 1    
        if(ichar(aa(i:i)).ge.65.AND.ichar(aa(i:i)).LE.90) then
           vv(jj:jj) = char(ichar(aa(i:i))+32)
         else
           vv(jj:jj) = aa(i:i)
         endif
        endif
        enddo
        vst(1:jj) = vv(1:jj)
        vnu = aa((j1+1):80)
**
      j1 = 0
      do i=1,lens
        if(inam(i:i).eq.'='.and.j1.eq.0) j1 = i
      enddo
      if(j1.eq.0) return
* remove blank characters and transform capital letters to small ones
       jj = 0
       do i = 1,(j1-1)
       if(inam(i:i).ne.' ') then
       jj = jj + 1    
        if(ichar(inam(i:i)).ge.65.AND.ichar(inam(i:i)).LE.90) then
           vv(jj:jj) = char(ichar(inam(i:i))+32)
         else
           vv(jj:jj) = inam(i:i)
         endif
        endif
        enddo

        ww(1:jj) = vv(1:jj)
       j1 = jj

         vv = vst 
         IF(ww(1:j1).eq.vv(1:j1)) THEN 
             IER = 0
             strout(1:80) = vnu(1:80)
             return
         ENDIF 
 200    IER = 1
      return
      END


*----------------------------------------------------------------------------*
      subroutine intochar(IV, JL, CV)

      implicit none
      character *20 cv
      integer IV, JL, i1, i2, J 
*
      jl = 0
      if(iv.lt.0) return
      JL = 1
      I1 = IV
      I2 = i1 - 10*int(i1/10)
      CV(1:1) = char(i2 + 48)
      if(i1.le.9) return
 1000 continue
       JL = jl + 1
       i1 = i1/10
       i2 = i1 - 10*int(i1/10)
       do j=2,jl
        cv((jl+2-j):(jl+2-j)) = cv((jl+1-j):(jl+1-j))
       enddo
       CV(1:1) = char(i2 + 48)
       if(i1.le.9) return
      goto 1000
      end
      
******************************************************************************

      SUBROUTINE TXGIVE_INIT
           
      implicit none
     
      integer CSAMODE
      double precision  MUONRW, GAMMAJRW, ZJRW, ZPRW, HLTRW, 
     &  SUSYRW, WWRW
      common /EXPAR/CSAMODE, MUONRW, GAMMAJRW, ZJRW, ZPRW, 
     &  HLTRW, SUSYRW, WWRW
      save   /EXPAR/ 

      CSAMODE = 0
      MUONRW = -1        
      GAMMAJRW = -1
      ZJRW = -1
      ZPRW = -1
      HLTRW = -1
      SUSYRW = -1
      WWRW = -1
 
 
 
      END
