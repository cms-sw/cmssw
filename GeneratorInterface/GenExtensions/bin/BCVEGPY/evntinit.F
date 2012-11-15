
c*******************************************************
c*******************************************************
      subroutine evntinit
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

#include "invegas.h"
#include "bcvegpy_set_par.inc"

c...three pythia functions return integers, so need declaring.
c...external statement.
      external pydata,totfun

c****************************************************
c*************** define most of the common block used
c****************************************************

c...pythia common block.
      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      parameter (maxnup=500)
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/

      parameter (maxpup=100)
      integer pdfgup,pdfsup,lprup
      common/heprup/idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &idwtup,nprup,xsecup(maxpup),xerrup(maxpup),xmaxup(maxpup),
     &lprup(maxpup)
      save /heprup/

c...user process event common block.
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
	double complex colmat,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	common/counter/ibcstate,nev
	common/grade/xi(NVEGBIN,10)
	common/vegcross/vegsec,vegerr,iveggrade
	common/colflow/amp2cf(10),smatval

c...transform of running information.
	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/confine/ptcut,etacut
	common/histcol/inx

c...generate---switch for full events. vegasopen---switch for using
c...vegas.
	logical generate,vegasopen,usegrade
	common/genefull/generate

c...parameters transformtion.
      common/funtrans/nq2,npdfu
	common/usertran/ishower,idpp

c...transform of the vegas information
      common/vegasinf/number,nitmx

c...to get the subprocess cross-section.
      common/subopen/subfactor,subenergy,isubonly

c...to get the distribution of an extra factor z=(2(k1+k2).p_bc)/shat.
      common/extraz/zfactor,zmin,zmax
	common/outpdf/ioutpdf,ipdfnum

c...used in grv98l
	common/intinip/iinip
	common/intinif/iinif

c...transform some variables
      common/loggrade/ievntdis,igenerate,ivegasopen,igrade

c...for transform the subprocess information, i.e.,  whether using
c...the subprocess q\bar{q}->bc+b+\bar{c} to generate events.
      common/qqbar/iqqbar,iqcode

	dimension nfin(20),alamin(20)
      common/coloct/ioctet
	common/octmatrix/coeoct
c...transform the first derivative of the wavefunction at the zero or the
c...wavefunction at the zero.
	common/wavezero/fbc

c...data:lambda and n_f values for parton distributions. this is obtained
c...from pythia and used for vegas running.
      data alamin/0.177d0,0.239d0,0.247d0,0.2322d0,0.248d0,0.248d0,
     &0.192d0,0.326d0,2*0.2d0,0.2d0,0.2d0,0.29d0,0.2d0,0.4d0,5*0.2d0/,
     &nfin/20*4/

c...xbcsec(8) records the total differential cross-sections for different
c...states: 1---singlet 1s0; 2---singlet 3s1; 7---octet 1s0; 8---octet 3s1;
c...3---singlet 1p1; 4---singlet 3p0; 5---singlet 3p1; 6---singlet 3p2.
	common/mixevnt/xbcsec(8),imix,imixtype
	common/mixevnt2/xbcsum,ibclimit
	character*1 pfile
	logical unwght
	common/unweight/unwght
	common/xxmax/xmax(NVEGBIN,8)


c************************************************
c... open recording files.--record mainly the 
c... vegas running information
c************************************************
	call upopenfile(imix,imixtype,ibcstate,ioctet)

c************************************************

	generate =.false.
	if(igenerate.eq.1) generate =.true.

	vegasopen=.false.
	if(ivegasopen.eq.1) vegasopen=.true.

	usegrade=.false.
	if(igrade.eq.1) usegrade=.true.

	if(isubonly.eq.1) subfactor=subenergy/ecm

c--------------------------------------------
c...reference light-like momentum, which can be choosing arbitrarily.
c...this can be used to check tht rightness of the program.
c...--used only for s-wave gluon fusion mechanism.
	pmomup(1,8)=3.0d0
	pmomup(2,8)=4.0d0
	pmomup(3,8)=5.0d0
	pmomup(4,8)=dsqrt(pmomup(1,8)**2+pmomup(2,8)**2+pmomup(3,8)**2)
	pmomup(5,8)=0.0d0

c--------------------------------------------
c...output program logo.
      call uplogo

c--------------------------------------------
c...setting the alphas.
c   mstu(111)=0---fixed; d=1---fir order; =2---sec order
	mstp(2)  =mstu(111)

c---------------------------------------------------------
c...choose lambda value to be used in case of using vegas.
	if(vegasopen) then
	   if(ioutpdf.eq.0) then
c...this part is from the inner part of pythia.
	      if(mstp(51).ge.1 .and. mstp(51).le.20) then
              paru(112)=alamin(mstp(51))
              mstu(112)=nfin(mstp(51))
            end if
	   else
	      if(ipdfnum.eq.100) then
			 paru(112)=0.1750d0    !grv98l
		     mstu(112)=4
	      end if
		  if(ipdfnum.eq.200) then
			 paru(112)=0.220d0     !msrt2001l
	         mstu(112)=4
	      end if
		  if(ipdfnum.eq.300) then
			 paru(112)=0.215d0     !cteq6l1
	         mstu(112)=4
	      end if
	   end if
	end if

c---------------------------------
c...logo for vegas.
	call vegaslogo(vegasopen)

c******************************************************************
c...using vegas to get the importance function and then the 
c...sampling stratage: xi(), which will be stored in grade files.
c******************************************************************
	ndim=7
	if(vegasopen) then
	 if(isubonly.eq.0) then
      if(imix.eq.0) then
c--------------------------------------
c... using existed grade to generate more precise grade.
c... this way is only used for imix=0 (preferable); 
c... for imix=1, the way is the same,
c... but is tedious to write all of them here.
c--------------------------------------
		 if(iveggrade.eq.1) then
	       write(*,'(a)')'existed grade to gene. more precise grade.'
		   write(3,'(a)')'existed grade to gene. more precise grade.'
		   do i=1,NVEGBIN
	          read(11,*) (xi(i,j),j=1,7)
	       end do
	       open(unit=29,file='newgrade.grid',status='unknown')
          do
	      read(11,*) pfile
	      if(pfile.eq.'&') exit
	     enddo
		 do i=1,NVEGBIN
	        read(11,*) (xmax(i,j),j=1,ndim)
	    end do
        do
	      read(11,*) pfile
	      if(pfile.eq.'+') exit
	     enddo
	      read(11,*) vegsec,vegerr,crossmax
	      rewind(11)
	     end if

	     if(iveggrade.eq.1) then
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght) call gen(totfun,ndim,xmax,0)
	   else 
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
        if(unwght) call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght) call gen(totfun,ndim,xmax,0)
	   end if

		 do i=1,NVEGBIN
	       if(iveggrade.eq.1) then
			  write(29,*) (xi(i,j),j=1,7)
	       else
	          write(11,*) (xi(i,j),j=1,7)
		   end if
		 end do

	     write(29,*) '&'
       if(iveggrade.eq.1) then
	   else
	     write(11,*) '&'
	   end if

		do i=1,NVEGBIN
	     if(iveggrade.eq.1) then
			write(29,*) (xmax(i,j),j=1,ndim)
        else
	      write(11,*) (xmax(i,j),j=1,ndim)          
		  end if
		end do

	   if(iveggrade.eq.1) then
	     write(29,*) '+'
	     write(29,*) vegsec,vegerr,crossmax
	   else
	     write(11,*) '+'
	     write(11,*) vegsec,vegerr,crossmax
	   end if

		end if!mix=0 end

c******************************************************
	  if(imix.eq.1) then  !mix=1 begin
	   if(imixtype.eq.1.or.imixtype.eq.2) then
		 ibcstate=1                         !1s0 and singlet
       write(*,'(a)') '...generating cross-sec and grade (1s0)...'
       write(3,'(a)') '...generating cross-sec and grade (1s0)...'
		 call paraswave(ibcstate)
       call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
       if(unwght)call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
       if(unwght)call gen(totfun,ndim,xmax,0)
       xbcsec(1)=vegsec
		 do i=1,NVEGBIN
	       write(36,*) (xi(i,j),j=1,7)
		 end do
       write(36,*) '&'
       do i=1,NVEGBIN
		  write(36,*) (xmax(i,j),j=1,ndim)
		 end do
	    write(36,*) '+'
       write(36,*) vegsec,vegerr,crossmax
		 rewind(36)
	   end if

c********************************
	 if(imixtype.eq.1.or.imixtype.eq.2) then
	    ibcstate=2                         !3s1 and singlet
        write(*,'(a)') '...generating cross-sec and grade (3s1)...'
        write(3,'(a)') '...generating cross-sec and grade (3s1)...'
        call paraswave(ibcstate)
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
        if(unwght)call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght)call gen(totfun,ndim,xmax,0)
        xbcsec(2)=vegsec
	     do i=1,NVEGBIN
	       write(37,*) (xi(i,j),j=1,ndim)
	     end do
        write(37,*) '&'
        do i=1,NVEGBIN
	 	   write(37,*) (xmax(i,j),j=1,ndim)
	     end do
        write(37,*) '+'
        write(37,*) vegsec,vegerr,crossmax
        rewind(37)
	  end if

c********************************
	 if(imixtype.eq.1.or.imixtype.eq.3) then
	    ibcstate=3
        write(*,'(a)') '...generating cross-sec and grade (1p1)...'
        write(3,'(a)') '...generating cross-sec and grade (1p1)...'
        call parapwave
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
        if(unwght) call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght) call gen(totfun,ndim,xmax,0)
        xbcsec(3)=vegsec
	     do i=1,NVEGBIN
	       write(38,*) (xi(i,j),j=1,ndim)
	     end do
        write(38,*) '&'
        do i=1,NVEGBIN
	 	   write(38,*) (xmax(i,j),j=1,ndim)
	     end do
        write(38,*) '+'
        write(38,*) vegsec,vegerr,crossmax
	  rewind(38)
	 end if

c********************************
	 if(imixtype.eq.1.or.imixtype.eq.3) then
	    ibcstate=4
        write(*,'(a)') '...generating cross-sec and grade (3p0)...'
        write(3,'(a)') '...generating cross-sec and grade (3p0)...'
        call parapwave
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
        if(unwght)call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght)call gen(totfun,ndim,xmax,0)
        xbcsec(4)=vegsec
	     do i=1,NVEGBIN
	     end do
        write(39,*) '&'
        do i=1,NVEGBIN
	 	   write(39,*) (xmax(i,j),j=1,ndim)
	     end do
        write(39,*) '+'
        write(39,*) vegsec,vegerr,crossmax
	  rewind(39)
	 end if

c********************************
	 if(imixtype.eq.1.or.imixtype.eq.3) then
	  ibcstate=5
         write(*,'(a)') '...generating cross-sec and grade (3p1)...'
         write(3,'(a)') '...generating cross-sec and grade (3p1)...'
         call parapwave
         call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
         if(unwght)call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
         if(unwght)call gen(totfun,ndim,xmax,0)
         xbcsec(5)=vegsec
	      do i=1,NVEGBIN
	        write(46,*) (xi(i,j),j=1,ndim)
	      end do
         write(46,*) '&'
         do i=1,NVEGBIN
	  	   write(46,*) (xmax(i,j),j=1,ndim)
	      end do
         write(46,*) '+'
         write(46,*) vegsec,vegerr,crossmax
	   rewind(46)
	  end if

c********************************
	  if(imixtype.eq.1.or.imixtype.eq.3) then
	     ibcstate=6
         write(*,'(a)') '...generating cross-sec and grade (3p2)...'
         write(3,'(a)') '...generating cross-sec and grade (3p2)...'
         call parapwave
         call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
         if(unwght)call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
         if(unwght)call gen(totfun,ndim,xmax,0)
         xbcsec(6)=vegsec
	      do i=1,NVEGBIN
	        write(47,*) (xi(i,j),j=1,ndim)
	      end do
         write(47,*) '&'
         do i=1,NVEGBIN
	  	   write(47,*) (xmax(i,j),j=1,ndim)
	      end do
         write(47,*) '+'
         write(47,*) vegsec,vegerr,crossmax
	   rewind(47)
	  end if

c********************************
	  if(imixtype.eq.1.or.imixtype.eq.3) then
	     ibcstate=7                         !1s0 and octet
         write(*,'(a)') '.generating cross-sec and grade (1s0)_8...'
         write(3,'(a)') '.generating cross-sec and grade (1s0)_8...'
         call paraswave(ibcstate)
         ibcstate=1    !return to original definition
         call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
         if(unwght)call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
         if(unwght)call gen(totfun,ndim,xmax,0)
         xbcsec(7)=vegsec
	      do i=1,NVEGBIN
	        write(48,*) (xi(i,j),j=1,ndim)
	      end do
         write(48,*) '&'
         do i=1,NVEGBIN
	  	   write(48,*) (xmax(i,j),j=1,ndim)
	      end do
	      write(48,*) '+'
         write(48,*) vegsec,vegerr,crossmax
	   rewind(48)
	  end if

c********************************
	 if(imixtype.eq.1.or.imixtype.eq.3) then
	    ibcstate=8                         !3s1 and octet
        write(*,'(a)') '.generating cross-sec and grade (3s1)_8...'
        write(3,'(a)') '.generating cross-sec and grade (3s1)_8...'
        call paraswave(ibcstate)
        ibcstate=2                   !return to original definition
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
        if(unwght)call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght)call gen(totfun,ndim,xmax,0)
        xbcsec(8)=vegsec
	     do i=1,NVEGBIN
	       write(49,*) (xi(i,j),j=1,ndim)
	     end do
        write(49,*) '&'
        do i=1,NVEGBIN
	 	   write(49,*) (xmax(i,j),j=1,ndim)
	     end do
        write(49,*) '+'
        write(49,*) vegsec,vegerr,crossmax
	  rewind(49)
	 end if
	   end if!mix=1 end
	 else !isubonly=0 else
		write(*,'(a,a,1x,g12.5)')'getting subprocess info....',
     &		 'at c.m. energy(gev)',subenergy
		write(3,'(a,a,1x,g12.5)')'getting subprocess info....',
     &		 'at c.m. energy(gev)',subenergy
	   	write(*,'(a)') 'the info. of hadron collider is no use!!!'
		write(3,'(a)') 'the info. of hadron collider is no use!!!'

          if(iveggrade.eq.1) then
	       write(*,'(a)')'using existed grade to gene. precise grade.'
		   write(3,'(a)')'using existed grade to gene. precise grade.'
		   do i=1,NVEGBIN
	          read(11,*) (xi(i,j),j=1,ndim)
	       end do
         do
	       read(11,*) pfile
	       if(pfile.eq.'&') exit
	      enddo
		   do i=1,NVEGBIN
	        read(11,*) (xmax(i,j),j=1,ndim)
	      end do
         do
	       read(11,*) pfile
	       if(pfile.eq.'+') exit
	      enddo
	      read(11,*) vegsec,vegerr,crossmax
	      rewind(11)
	       open(unit=29,file='newgrade.grid',status='unknown')
	    end if	   
		ndim=5
       if(iveggrade.eq.1) then
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght) call gen(totfun,ndim,xmax,0)
	   else 
        call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,0)
        if(unwght) call vegas(totfun,ndim,number,nitmx,2,vegsec,xmax,1)
        if(unwght) call gen(totfun,ndim,xmax,0)
	   end if

		 do i=1,NVEGBIN
	       if(iveggrade.eq.1) then
			  write(29,*) (xi(i,j),j=1,7)
	       else
	          write(11,*) (xi(i,j),j=1,7)
		   end if
		 end do

       if(iveggrade.eq.1) then
	     write(29,*) '&'
	   else
	     write(11,*) '&'
	   end if

		do i=1,NVEGBIN
	     if(iveggrade.eq.1) then
			write(29,*) (xmax(i,j),j=1,ndim)
        else
	      write(11,*) (xmax(i,j),j=1,ndim)          
		  end if
		end do

	   if(iveggrade.eq.1) then
	     write(29,*) '+'
	     write(29,*) vegsec,vegerr,crossmax
	   else
	     write(11,*) '+'
	     write(11,*) vegsec,vegerr,crossmax
	   end if
	 end if!isubonly=0 end
	  call vegasend(vegasopen,ievntdis,usegrade)
	else !vegasopen else
	  call vegasend(vegasopen,ievntdis,usegrade)
	  if(isubonly.eq.0) then
	     ndim=7
	  else
	     ndim=5
	     write(*,'(a)')'getting the info. of the subporcess....'
		 write(3,'(a)')'getting the info. of the subporcess....'
	  end if

c...initialize the grade.
	  rc=1.0d0/NVEGBIN
        do 77 j=1,ndim
           xi(NVEGBIN,j)=1.0d0
           dr=0.0d0
        do 77 i=1,NVEGBIN-1
           dr=dr+rc
           xi(i,j)=dr
77      continue

c*****************************************************************
c...using the existed grade. one thing should be care here is that
c...the existed grade should be formed under the same parameters.
      if(imix.eq.0) then
	    if(usegrade) then
	     write(*,'(a)') 'using the existed vegas grade.'
	     write(3,'(a)') 'using the existed vegas grade.'
	     do i=1,NVEGBIN
	        read(11,*) (xi(i,j),j=1,7)
	      end do
	      do
	       read(11,*) pfile
	       if(pfile.eq.'&') exit
	      enddo
	      do i=1,NVEGBIN
	        read(11,*) (xmax(i,j),j=1,ndim)
	      end do
	      do
	       read(11,*) pfile
	       if(pfile.eq.'+') exit
	      enddo
	      read(11,*) vegsec,vegerr,crossmax
	      rewind(11)
	      if(unwght) call gen(totfun,ndim,xmax,0)
	    end if
	  end if !mix=0 end
	end if !vegasopen end

c********************************************
c***  find the total-corssection for each state 
c***  from the previous generated grade, used
c***  when vegas not used.
c********************************************
      if(imix.eq.1 .and. (.not.vegasopen)) then
	 if(imixtype.eq.1) then
	  do i=1,4
	   do
	     read(36+i-1,*) pfile
	     if(pfile.eq.'+') exit
	   enddo
	   read(36+i-1,*) xbcsec(i)
	   rewind(36+i-1)
	  enddo
	  do i=1,4
	   do
	     read(46+i-1,*) pfile
	     if(pfile.eq.'+') exit
	   enddo
	   read(46+i-1,*) xbcsec(4+i)
	   rewind(46+i-1)
	  enddo
	 end if
c---------------------------------------
	 if(imixtype.eq.2) then
	  do i=1,2
	   do
	     read(36+i-1,*) pfile
	     if(pfile.eq.'+') exit
	   enddo
	   read(36+i-1,*) xbcsec(i)
	   rewind(36+i-1)
	  enddo
	 end if
c----------------------------------------
	 if(imixtype.eq.3) then
	  do i=3,4
	   do
	     read(36+i-1,*) pfile
	     if(pfile.eq.'+') exit
	   enddo
	   read(36+i-1,*) xbcsec(i)
	   rewind(36+i-1)
	  enddo
	  do i=1,4
	   do
	     read(46+i-1,*) pfile
	     if(pfile.eq.'+') exit
	   enddo
	   read(46+i-1,*) xbcsec(4+i)
	   rewind(46+i-1)
	  enddo
	 end if
	end if

c*******************************************
	
	if(imix.eq.1) then
	 if(imixtype.eq.1) then 
	  xbcsum=0.0d0
	  do i=1,8
	   xbcsum=xbcsum+xbcsec(i)
	  end do
	  ibclimit=8
	 end if

	 if(imixtype.eq.2) then 
	  xbcsum=0.0d0
	  do i=1,2
	   xbcsum=xbcsum+xbcsec(i)
	  end do
	  ibclimit=2
	 end if

	 if(imixtype.eq.3) then 
	  xbcsum=0.0d0
	  do i=3,8
	   xbcsum=xbcsum+xbcsec(i)
	  end do
	  ibclimit=8
	 end if
	end if

	crossmax=0.0d0
c...if to get mixed results, then one should redefine grade
c...(the sampling importance function), the old grade files
c...must be exist and be generated under the same condition.
      if(imix.eq.1 .and. isubonly.eq.0) call initmixgrade

c***************************************************
c***    some switches for event generation     *****
c***************************************************
c...switch off aspects: initial and final state
c...showers, multiple interactions, hadronization.
      if(ishower.eq.0) then
	mstp(61) =0
        mstp(71) =0
        mstp(81) =0
        mstp(111)=0
	end if

c...expanded event listing (required for histogramming).
      mstp(125)=2

c...pyinit--->to initialize the generation procedure.
      call pyinit('user',' ',' ',0d0)

c...set bc+ stable.
	   mdcy(pycomp(541),1)=0
c***************************************************

      return
	end

