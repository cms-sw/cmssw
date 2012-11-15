
c*
c** initialize the grade to derive the mixed events.
c*

      subroutine initmixgrade
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

#include "invegas.h"
#include "bcvegpy_set_par.inc"

	common/mixevnt/xbcsec(8),imix,imixtype
	common/mixevnt2/xbcsum,ibclimit
	common/grade/xi(NVEGBIN,10)
      common/counter/ibcstate,nev
      common/loggrade/ievntdis,igenerate,ivegasopen,igrade
      logical usegrade,usevegas
	common/vegcross/vegsec,vegerr,iveggrade
	character*1 pfile
	logical unwght
	common/unweight/unwght
	common/xxmax/xmax(NVEGBIN,8)
       
         usegrade=.false.
         if(igrade.eq.1) usegrade=.true.
         usevegas=.false.
         if(ivegasopen.eq.1) usevegas=.true.

c*********************************************************
c*** define which state to be generated next .
c*********************************************************
           if(imixtype.eq.1.or.imixtype.eq.2) ibcstate=0
           if(imixtype.eq.3) ibcstate=2

           ranbc=xbcsum*pyr(0)
55         ibcstate=ibcstate+1
           ranbc =ranbc-xbcsec(ibcstate)
           if(ibcstate.lt.ibclimit .and. ranbc.gt.1.0d-16) go to 55

c----------------------
           if(ibcstate.eq.1 .or. ibcstate.eq.2) then
               call paraswave(ibcstate)
           end if
           if(ibcstate.gt.2 .and. ibcstate.lt.7) then
               call parapwave
           end if
           if(ibcstate.eq.7 .or. ibcstate.eq.8) then
               call paraswave(ibcstate)
           end if
	   ndim=7
c------------------------------------------------------------
c...the following way to generate the mixed events are not
c...appreciable, list here only for completeness of the program.
c...rough idea to achieve such case: adding/reading the total cross-
c...sections for all the states to be mixed, and then select a 
c...particular state to be generate with
c...its corresponding cross-section, and then define a uniform
c...grade for every state.
c------------------------------------------------------------
	   if((.not. usevegas) .and. (.not. usegrade)) then
c...initialize the grade.
             rc=1.0d0/NVEGBIN
             do 77 j=1,ndim
               xi(NVEGBIN,j)=1.0d0
               dr=0.0d0
             do 77 i=1,NVEGBIN-1
               dr=dr+rc
               xi(i,j)=dr
77           continue
             if(ibcstate.eq.7) ibcstate=1
             if(ibcstate.eq.8) ibcstate=2
             return
	   end if

c**********************************************************
c**********************************************************
c*** since not using vegas and not using grade has been 
c*** returned, so the following can be introduced without
c*** if conditions.
c**********************************************************
c**********************************************************
	     if(ibcstate.eq.1) then
	         do i=1,NVEGBIN
	             read(36,*) (xi(i,j),j=1,7)
	          end do
                  do
	           read(36,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	          do i=1,NVEGBIN
	           read(36,*) (xmax(i,j),j=1,7)
	          end do
                   do
	           read(36,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(36,*) vegsec,vegerr,crossmax
	          rewind(36)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	      end if

c***************************************
	      if(ibcstate.eq.2) then
	           do i=1,NVEGBIN
	             read(37,*) (xi(i,j),j=1,7)
	           end do
	         do
	           read(37,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	         do i=1,NVEGBIN
	           read(37,*) (xmax(i,j),j=1,7)
	          end do
                  do
	           read(37,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(37,*) vegsec,vegerr,crossmax
	          rewind(37)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	       end if

c***************************************
	      if(ibcstate.eq.3) then
	         do i=1,NVEGBIN
	             read(38,*) (xi(i,j),j=1,7)
	           end do
                  do
	           read(38,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	         do i=1,NVEGBIN
	           read(38,*) (xmax(i,j),j=1,7)
	          end do
                 do
	           read(38,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(38,*) vegsec,vegerr,crossmax
	          rewind(38)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	      end if

c***************************************
	       if(ibcstate.eq.4) then
	          do i=1,NVEGBIN
	             read(39,*) (xi(i,j),j=1,7)
	           end do
                  do
	           read(39,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	          do i=1,NVEGBIN
	           read(39,*) (xmax(i,j),j=1,7)
	          end do
                  do
	           read(39,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(39,*) vegsec,vegerr,crossmax
	          rewind(39)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	        end if

c***************************************
	       if(ibcstate.eq.5) then
	           do i=1,NVEGBIN
	             read(46,*) (xi(i,j),j=1,7)
	           end do
                   do
	           read(46,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	          do i=1,NVEGBIN
	           read(46,*) (xmax(i,j),j=1,7)
	          end do
                  do
	           read(46,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(46,*) vegsec,vegerr,crossmax
	          rewind(46)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	       end if

c***************************************
	       if(ibcstate.eq.6) then
	          do i=1,NVEGBIN
	             read(47,*) (xi(i,j),j=1,7)
	           end do
                  do
	           read(47,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	          do i=1,NVEGBIN
	           read(47,*) (xmax(i,j),j=1,7)
	          end do
                  do
	           read(47,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(47,*) vegsec,vegerr,crossmax
	          rewind(47)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	        end if

c***************************************
	      if(ibcstate.eq.7) then
                      ibcstate=1   !1s0 and color-octet
	          do i=1,NVEGBIN
	            read(48,*) (xi(i,j),j=1,7)
	           end do
                  do
	           read(48,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	          do i=1,NVEGBIN
	           read(48,*) (xmax(i,j),j=1,7)
	          end do
                  do
	           read(48,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(48,*) vegsec,vegerr,crossmax
	          rewind(48)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	       end if

c***************************************
	       if(ibcstate.eq.8) then
                      ibcstate=2   !3s1 and color-octet
	           do i=1,NVEGBIN
	               read(49,*) (xi(i,j),j=1,7)
	            end do
                  do
	           read(49,*) pfile
	           if(pfile.eq.'&') exit
	          enddo
	         do i=1,NVEGBIN
	           read(49,*) (xmax(i,j),j=1,7)
	          end do
                  do
	           read(49,*) pfile
	           if(pfile.eq.'+') exit
	          enddo
	          read(49,*) vegsec,vegerr,crossmax
	          rewind(49)
	          if(unwght) call gen(totfun,ndim,xmax,0)
	        end if
	 end

