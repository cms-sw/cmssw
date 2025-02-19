
      subroutine vegaslogo(vegasopen)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

#include "invegas.h"
#include "bcvegpy_set_par.inc"
      common/vegasinf/number,nitmx
	logical vegasopen

      if(vegasopen) then
           write(*,'(a,i5)') 'the used bin number in vegas:',NVEGBIN
           write(3,'(a,i5)') 'the used bin number in vegas:',NVEGBIN
	   write(*,'(a,i9,a,i3)') 'using vegas: number in each iter.='
     &	   ,number,'  iter.=',nitmx
   	   write(3,'(a,i9,a,i3)') 'using vegas: number in each iter.='
     &	   ,number,'  iter.=',nitmx
           write(*,*)
           write(3,*)
	   write(*,'(a)') 
     &	   '......................................................'
	   write(*,'(a)') 
     &	   '.............. end of initialization .................'
	   write(*,'(a)') 
     &	   '......................................................'
	   write(3,'(a)') 
     &	   '......................................................'
	   write(3,'(a)') 
     &	   '.............. end of initialization .................'
	   write(3,'(a)') 
     &	   '......................................................'
           write(*,*)
           write(3,*)
	   write(*,'(a)') '.....waiting......    vegas running......'
	   write(3,'(a)') '.....waiting......    vegas running......'
           write(*,*)
           write(3,*)
	end if

	end

c*******************************************************************

      subroutine vegasend(vegasopen,ievntdis,usegrade)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
	logical vegasopen,usegrade
        common/mixevnt/xbcsec(8),imix,imixtype

      if(vegasopen) then
	   write(*,'(a)') '...... end of vegas, grade generated ......'
	   write(3,'(a)') '...... end of vegas, grade generated ......'
	   if(ievntdis.eq.1) then
	     write(*,'(a)')'ok, to get the evnt number distributions'
	     write(3,'(a)')'ok, to get the evnt number distributions'
         else
	     write(*,'(a)')'ok, to get the differential distributions'
	     write(3,'(a)')'ok, to get the differential distributions'
	   end if
           write(*,*)
           write(3,*)
      else
	   write(*,'(a)') '......    vegas has not been used     ......'
	   write(3,'(a)') '......    vegas has not been used     ......'
           if(usegrade) then
           write(*,'(a)') 'using existed grade to generated events.....'
           write(3,'(a)') 'using existed grade to generated events.....'
	   write(*,'(a)')'(existed) grade has not been further improved'
	   write(3,'(a)')'(existed) grade has not been further improved'
           else
            write(*,'(a)') 'no grade is adopted, using trivial MC.'
            write(3,'(a)') 'no grade is adopted, using trivial MC.'
            if(imix.eq.1) then
              write(*,'(a)') 'not an appreciable way for mixed events'
              write(3,'(a)') 'not an appreciable way for mixed events'
              write(*,'(a)') 'the precision is lower than other methods'
              write(3,'(a)') 'the precision is lower than other methods'
            end if
           end if
           write(*,*)
           write(3,*)
           write(*,'(a)')
     &     '......................................................'
           write(*,'(a)')
     &     '.............. end of initialization .................'
           write(*,'(a)')
     &     '......................................................'
           write(3,'(a)')
     &     '......................................................'
           write(3,'(a)')
     &     '.............. end of initialization .................'
           write(3,'(a)')
     &     '......................................................'
           write(*,*)
           write(3,*)
	   if(ievntdis.eq.1) then
	   write(*,'(a)')'ok, to get the evnt number distributions.....'
	   write(3,'(a)')'ok, to get the evnt number distributions.....'
         else
	   write(*,'(a)')'ok, to get the differential distributions....'
	   write(3,'(a)')'ok, to get the differential distributions....'
	   end if
	end if

	end
