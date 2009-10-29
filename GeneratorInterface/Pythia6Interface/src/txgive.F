c----------------------------------------------------------------------------c
C..TXGIVE V:1.03,   06/09/2006 by Serge Slabospitsky
C rewritten from PYGIVE routine from PYTHIA
C...Sets values of commonblock variables.
 
      SUBROUTINE TXGIVE(CHIN)
      implicit none 
      Integer           Ipar     ! global TopRex integer parameters  
      double precision  Rpar     ! global TopRex real    parameters
      common /TXPAR/Ipar(512), Rpar(512)
      save   /TXPAR/
***
      integer CSAMODE
      integer pad
      double precision  MUONRW, GAMMAJRW, ZJRW, ZPRW, HLTRW, 
     &  SUSYRW, WWRW, PTPOWER
      common /EXPAR/ pad, CSAMODE, MUONRW, GAMMAJRW, ZJRW, ZPRW, 
     &  HLTRW, SUSYRW, WWRW, PTPOWER

      save   /EXPAR/
**********
      integer ndch   ! maximum number of decay channels
      parameter (ndch=50)
      integer        MID(0:ndch,4),             FID(4,ndch,5) 
      double precision   wid(0:ndch,4),brf(ndch,4),     BRS(2,4)
      common /TXdec/ MID, WID,         BRF,     FID,    brs
      save   /TXdec/
*...........
      integer  maxpro 
      parameter (maxpro=500)
      integer         txmup, txsel, txpro,         txext
      common /trxpro/ txmup, txsel, txpro(maxpro), txext(maxpro)
      save   /trxpro/
*............
      integer Irtyp 
      double precision an_FL,an_FR,an_HL,an_HR,an_XL,an_XR,an_YL,an_YR
      common /CTOPINI/Irtyp(20), an_FL(2,20), an_FR(2,20),
     &   an_HL(2,20), an_HR(2,20), an_XL(2,20), an_XR(2,20), 
     &   an_YL(2,20), an_YR(2,20) 
      save /CTOPINI/
***
      integer IER, I, J, IL, k1, k2 , Iread
      integer lenk
      parameter (lenk=512)
      character *(*) chin
      character *60 inam, rnam
      character *20 cvi  ! , cvj
      character *512 STRIN 
      character *512 strin2
      data strin2/'PROCESS : not given'/
      save strin2

*   for TAUOLA package
      integer            PJAK1, PJAK2, MDTAU
      common /ki_taumod/ PJAK1, PJAK2, MDTAU
      data pjak1/-1/, pjak2/ -1/, mdtau/-1/
      save /ki_taumod/ 
*
      character*512                            CXpar
      integer         Ixpar 
      double precision            RXpar
      common /EXGPAR/ IXpar(100), RXpar(100), CXpar(100)
      data IXpar/1, 99*0/
      save /EXGPAR/ 
      logical first
      data first /.TRUE./
      save first
****
*
* read name of the process
*
      IF(first.EQV..TRUE.) then
        first=.FALSE.
           do j = 1,100
             do i = 1,lenk
              cxpar(j)(i:i) = ' '
             enddo
            enddo
      ENDIF
***
      Iread = 1
      ier = 1
***

      Inam = 'CSAMODE ='
        call TXRSTR2(inam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) CSAMODE 
         endif
*
      
      rnam = 'MUONRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) MUONRW 
         endif
*
    
      rnam = 'GAMMAJRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) GAMMAJRW 
         endif
*
      
      rnam = 'ZJRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) ZJRW 
         endif
*	 
      rnam = 'ZPRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) ZPRW 
         endif
* 
      rnam = 'HLTRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) HLTRW 
         endif
*	 
      rnam = 'SUSYRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) SUSYRW 
         endif
*
      rnam = 'WWRW ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) WWRW 
         endif

      rnam = 'PTPOWER ='
        call TXRSTR2(rnam, chin, strin, ier) 
C	print*,'IER', ier
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) PTPOWER
         endif
	 
***

      do I = 1,512
        call intochar(I, il, cvi)
         Inam = 'ipar('//cvi(1:il)//')'//' = '
         Rnam = 'rpar('//cvi(1:il)//')'//' = '
        call TXRSTR2(rnam, chin, strin, ier) 
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) Rpar(i) 
         endif
        call TXRSTR2(inam, chin, strin, ier) 
         if(ier.ne.1) then
           iread = 0
           read(strin(1:512),*) Ipar(i) 
         endif
      enddo
*
***
*
      Inam = 'unwfile = ' ! for alpgen
        call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) then
           Iread = 0
           call strnum(chin, k1, k2)
            if(k1.ge.1.and.k2.ge.k1) then
                do i = 1,lenk
                 cxpar(1)(i:i) = ' '
                enddo
             ixpar(99) = k2 - k1 + 1
             ixpar(100) = ixpar(99) + 1
             cxpar(1)(1:ixpar(99)) = chin(k1:k2)
            endif
*        print*,'UNFILE k1 k2 cxpar ',k1,k2,cxpar(1)
         endif
***
      Inam = 'evfile = '        ! for Madgraph and CompHEP
        call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) then
            Iread = 0
          call strnum(chin, k1, k2)
            if(k1.ge.1.and.k2.ge.k1) then
                do i = 1,lenk
                 cxpar(1)(i:i) = ' '
                enddo
             ixpar(99) = k2 - k1 + 1
              cxpar(1)(1:ixpar(99)) = chin(k1:k2)
*            cxpar(1) = chin(k1:k2)
             ixpar(100) = k2 - k1 + 2
            endif
*        print*,'EVFILE k1 k2 cxpar ',k1,k2,cxpar(1)
         endif
*
      Inam = 'PROCESS = '
        call TXRSTR2(inam, chin, strin, ier) 
*        print*,'** IER ' ,ier,' process ',chin 
         if(ier.eq.0) then
            Iread = 0
           do i = 1,lenk
             cxpar(2)(i:i) = ' '
             strin2(i:i) = ' '
            enddo
           strin2 = chin
*           cxpar(2) = chin
*           print*,cxpar(2) 
          endif
           cxpar(2) = strin2
*
      do I = 1,100
       call intochar(I, il, cvi)
        Rnam = 'rxpar('//cvi(1:il)//')'//' = '
        call TXRSTR2(rnam, chin, strin, ier) 
         if(ier.eq.0) Iread = 0
         if(ier.eq.0) read(strin(1:lenk),*) RXpar(i) 
        Inam = 'ixpar('//cvi(1:il)//')'//' = '
        call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) IXpar(i) 
         if(ier.eq.0) Iread = 0
      enddo
*
      do I = 1,200
        call intochar(I, il, cvi)
         Inam = 'ipar('//cvi(1:il)//')'//' = '
         Rnam = 'rpar('//cvi(1:il)//')'//' = '
        call TXRSTR2(rnam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) Rpar(i) 
        if(ier.eq.0) Iread = 0
        call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) Ipar(i) 
        if(ier.eq.0) Iread = 0
      enddo
*
         Inam = 'txsel '//' = '
        call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) txsel 
        if(ier.eq.0) Iread = 0
      do I = 1, maxpro
        call intochar(I, il, cvi)
        Inam = 'txpro('//cvi(1:il)//')'//' = '
        call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) txpro(i) 
        if(ier.eq.0) Iread = 0
      enddo
* read decay channels
*     IT = 1 : top quark,  IT = 2 : W-boson, 
*     IT = 3 : H+- boson,  IT = 4 : Z-boson 
      do I = 1, ndch 
        call intochar(I, il, cvi)
*  top quark
         inam = 'dectop('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) mid(i, 1) 
        if(ier.eq.0) Iread = 0
*  H+- boson 
         inam = 'dechpm('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) mid(i, 3) 
         if(ier.eq.0) Iread = 0
*  W-boson 
         inam = 'decwpm('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) mid(i, 2) 
         if(ier.eq.0) Iread = 0
*  Z-boson
         inam = 'decZ0('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) mid(i, 4) 
         if(ier.eq.0) Iread = 0
      enddo 

      do I = 1,20 
        call intochar(I, il, cvi)
        inam = 'Irtyp('//cvi(1:il)//') ='
        call TXRSTR2(inam, chin, strin, ier) 
         if(ier.eq.0) read(strin(1:lenk),*) irtyp(i)
         if(ier.eq.0) Iread = 0
      enddo 
****
      do I = 1,20 
        call intochar(I, il, cvi)
*
* FL
        ier = 1
         inam = 'an_fl('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_fl(1,i), an_fl(2,i)
        if(ier.eq.0) Iread = 0
* FR
        ier = 1
         inam = 'an_fr('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_fr(1,i), an_fr(2,i)
        if(ier.eq.0) Iread = 0
*** HL
        ier = 1
         inam = 'an_hl('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_hl(1,i), an_hl(2,i)
        if(ier.eq.0) Iread = 0
* HR
        ier = 1
          inam = 'an_hr('//cvi(1:il)//') ='
          call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_hr(1,i), an_hr(2,i)
        if(ier.eq.0) Iread = 0
*** XL
        ier = 1
         inam = 'an_xl('//cvi(1:il)//') ='
          call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_xl(1,i), an_xl(2,i)
        if(ier.eq.0) Iread = 0
* XR
        ier = 1
          inam = 'an_xr('//cvi(1:il)//') ='
          call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_xr(1,i), an_xr(2,i)
        if(ier.eq.0) Iread = 0
*** YL
        ier = 1
         inam = 'an_yl('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_yl(1,i), an_yl(2,i)
        if(ier.eq.0) Iread = 0
* YR
        ier = 1
         inam = 'an_yr('//cvi(1:il)//') ='
         call TXRSTR2(inam, chin, strin, ier) 
        if(ier.eq.0) read(strin(1:lenk),*) an_yr(1,i), an_yr(2,i)
        if(ier.eq.0) Iread = 0
      enddo 
* read TAUOLA parameters
       ier = 1
       inam = 'tauola ='
       call TXRSTR2(inam, chin, strin, ier) 
       if(ier.eq.0) read(strin(1:lenk),*) pjak1, pjak2, mdtau
       if(ier.eq.0) Iread = 0
****
      IF(IXpar(1).eq.1.OR.IXpar(1).eq.2) then
        if(Iread.eq.0) return 
        k1 = 0
        k2 = 0
      do i = 1,lenk
       If(k1.eq.0.AND.
     &  (ichar(chin(i:i)).ge.33.and.
     &   ichar(chin(i:i)).le.126)) then
          k1 = i
        EndIf
         if(chin(i:i).eq.'=') then
           k2 = i 
           goto 152
         endif
      enddo
 152  continue
       if(k1.eq.0.or.k2.eq.0) then 
         print*,'TXGIVE: do not recognize input : ',chin
       else
        if(k2.eq.0) k2 = lenk
         k2 = k2 - 1
         print*,'TXGIVE: do not recognize input parameter : ',
     &    chin(k1:k2)
       endif
       if(IXpar(1).eq.2) then
         print*,'STOP '
         stop
       endif
      ENDIF
      return
      end
*----------------------------------------------------------------------------*
C..INTOCHAR, 06/09/2006 by Serge Slabospitsky
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
c----------------------------------------------------------------------------c
C..TXRSTR2, 06/09/2006 by Serge Slabospitsky
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
      parameter (lens=60)
      integer lenk
      parameter (lenk=512)
      character *512 strout, aa, vnu 
*
      character *60 inam, vv, vst, ww 
*
       do i = 1,lenk
        strout(i:i) = ' '
       enddo
      ier = 1
       do i = 1,lenk
        if(aa(i:i).ne.' ') then 
         j1 = 1  ! non-blank character
         if(aa(i:i).eq.'*') goto 200   ! first non-blank item = '*' -  comment
         goto 16
        endif        
       enddo
        goto 200 !  blank string 
 16    j1 = 0
       do i = 1,lenk
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
        vnu = aa((j1+1):lenk)
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
             strout(1:lenk) = vnu(1:lenk)
             return
         ENDIF 
 200    IER = 1
      return
      END
*----------------------------------------------------------------------------c
* STRNUM, 06/09/2006 by Serge Slabospitsky
*                                                                            *
      subroutine strnum(aa, k1, k2) 
*     ------------------------------------                                   *
* input : AA is character string (up to 40 symbols) with '=' as the end      *
* outout: k1 and k2 are the positions in AA of the first and last characters *
* in the input string after '=' sign. K1 = K2 = 0 if not found               *
*............................................................................*
      implicit none
      integer  i, k1, k2, lenk
      parameter (lenk=132)
      character *132 aa 
*
      k1 = 0
      k2 = 0
      i = 0
 51   i = i + 1
      if(i.ge.lenk) goto 59
       If(k1.eq.0) then
        if(aa(i:i).eq.'=') k1 = i 
       Elseif(k1.ge.1) then
        if(ichar(aa(i:i)).ge.33.and.
     &     ichar(aa(i:i)).le.126) then
          k1 = i
          goto 52
        endif
       EndIf
       goto 51
       if(k1.eq.0) goto 59
 52   k2 = 0
      i = k1 
 53   i = i + 1
       if(i.ge.lenk) goto 59
        if(ichar(aa(i:i)).ge.33.and.
     &     ichar(aa(i:i)).le.126) goto 53
         k2 = i-1
         if(k2.eq.0.or.k2.gt.lenk) k2 = 0
 59   return
      end
c----------------------------------------------------------------------------c
C..TXGIVE V:1.03,   01/09/2006 by Filip Moortgat 

      SUBROUTINE TXGIVE_INIT
           
      implicit none

      integer CSAMODE
      integer pad
      double precision  MUONRW, GAMMAJRW, ZJRW, ZPRW, HLTRW, 
     &  SUSYRW, WWRW, PTPOWER
      common /EXPAR/ pad, CSAMODE, MUONRW, GAMMAJRW, ZJRW, ZPRW, 
     &  HLTRW, SUSYRW, WWRW, PTPOWER
      save   /EXPAR/


      CSAMODE = 0
      pad = 0
      MUONRW = -1        
      GAMMAJRW = -1
      ZJRW = -1
      ZPRW = -1
      HLTRW = -1
      SUSYRW = -1
      WWRW = -1
      PTPOWER = 0
      
 
      END
*----------------------------------------------------------------------------c
