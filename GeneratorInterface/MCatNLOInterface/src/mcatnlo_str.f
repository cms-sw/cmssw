c Utilities working on strings
c
      function fk88strnoeq(str1,str2)
c Returns true if str1#str2, false otherwise. The comparison
c is case INSENSITIVE
      logical fk88strnoeq,flag
      character * (*) str1,str2
      character * 70 strin,tmp1,tmp2
c
      strin=str1
      call fk88low_to_upp(strin,tmp1)
      strin=str2
      call fk88low_to_upp(strin,tmp2)
      if(tmp1.eq.tmp2)then
        flag=.false.
      else
        flag=.true.
      endif
      fk88strnoeq=flag
      return
      end


      subroutine fk88low_to_upp(strin,strout)
c Converts lowercase to uppercase
      implicit real*8(a-h,o-z)
      character*70 strin,strout,tmp
      character*1 ch,ch1
c
      len=ifk88istrl(strin)
      if(len.eq.0)then
        return
      elseif(len.eq.1)then
        ch=strin
        call fk88xgetchar1(ch,ch1)
        strout=ch1
      else
        do i=1,len
          ch=strin(i:i+1)
          call fk88xgetchar1(ch,ch1)
          if(i.eq.1)then
            strout=ch1
          else
            call fk88strcat(strout,ch1,tmp)
            strout=tmp
          endif
        enddo
      endif
      return
      end


      subroutine fk88xgetchar1(ch,ch1)
c Converts lowercase to uppercase (1 character only)
      character*1 ch,ch1
c ia=ascii value of a
      parameter (ia=97)
c iz=ascii value of z
      parameter (iz=122)
c ishift=difference between the ascii value of a and A
      parameter (ishift=32)
c
      ic=ichar(ch)
      if(ic.ge.ia.and.ic.le.iz)then
        ch1=char(ic-ishift)
      else
        ch1=ch
      endif
      return
      end


      subroutine fk88strnum(string,num)
c- writes the number num on the string string starting at the blank
c- following the last non-blank character
      character * (*) string
      character * 20 tmp
      l = len(string)
      write(tmp,'(i15)')num
      j=1
      dowhile(tmp(j:j).eq.' ')
        j=j+1
      enddo
      ipos = ifk88istrl(string)
      ito = ipos+1+(15-j)
      if(ito.gt.l) then
         write(*,*)'error, string too short'
         write(*,*) string
         stop
      endif
      string(ipos+1:ito)=tmp(j:)
      end


      function ifk88istrl(string)
c returns the position of the last non-blank character in string
      character * (*) string
      i = len(string)
      dowhile(i.gt.0.and.string(i:i).eq.' ')
         i=i-1
      enddo
      ifk88istrl = i
      end


      subroutine fk88strcat(str1,str2,str)
c concatenates str1 and str2 into str. Ignores trailing blanks of str1,str2
      character *(*) str1,str2,str
      l1=ifk88istrl(str1)
      l2=ifk88istrl(str2)
      l =len(str)
      if(l.lt.l1+l2) then
          write(*,*) 'error: l1+l2>l in fk88strcat'
          write(*,*) 'l1=',l1,' str1=',str1
          write(*,*) 'l2=',l2,' str2=',str2
          write(*,*) 'l=',l
          stop
      endif
      if(l1.ne.0) str(1:l1)=str1(1:l1)
      if(l2.ne.0) str(l1+1:l1+l2)=str2(1:l2)
      if(l1+l2+1.le.l) str(l1+l2+1:l)= ' '
      end
