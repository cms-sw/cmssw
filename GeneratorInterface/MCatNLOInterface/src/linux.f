      subroutine sysdep
c use newver='NEW' for vaxes, 'UNKNOWN' in other machines,
      character * 7 newver
      common/newver/newver
      newver = 'UNKNOWN'
      end

C      subroutine delete(fname)
C      character * 80 str
C      character * (*) fname
C      l = len(fname)
C      k = 1
C      dowhile(fname(k:k).eq.' '.and.k.lt.l)
C         k = k+1
C      enddo
C      dowhile(fname(l:l).eq.' '.and.l.gt.k)
C         l = l-1
C      enddo
C      if(l-k.gt.70) then
C         write(*,*) fname
C         write(*,*) 'delete: filename > 70 chars not allowed'
C         stop
C      endif
C      if(l.eq.k) then
C         write(*,*) 'delete: void filename'
C         stop
C      endif
C      str(1:) = 'rm '
C      str(7:) = fname(k:l)
C      call system(str)
C      end

c      subroutine idate(i1,i2,i3)
c      common/slate/isl(40)
c      call datime(id,it)
c      i3=id/10000
c      i1=(id-i3*10000)/100
c      i2= id-i3*10000-i1*100
c      end

c      subroutine time(ctime)
c      character * 8 ctime,cday
c      call datimh(cday,ctime)
c      end

