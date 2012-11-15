      subroutine uplogo
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

	common/mixevnt/xbcsec(8),imix,imixtype
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
	double complex colmat,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	common/counter/ibcstate,nev
      common/subopen/subfactor,subenergy,isubonly
	common/confine/ptcut,etacut
      common/funtrans/nq2,npdfu
	common/usertran/ishower,idpp
      common/qqbar/iqqbar,iqcode
      common/coloct/ioctet
	common/octmatrix/coeoct
	common/outpdf/ioutpdf,ipdfnum
	common/wavezero/fbc
      character*8 alporder
      logical unwght
	common/unweight/unwght

	
	write(*,'(a)') 
     &	'......................................................'
	write(*,'(a)') 
     &	'......................................................'
	write(*,'(a)')
     &	'...........    BCVEGPY2.1 LINUX version ..............'
	write(*,'(a)') 
     &	'......................................................'
      write(3,'(a)') 
     &	'......................................................'
      write(3,'(a)') 
     &	'......................................................'
	write(3,'(a)') 
     &	'...........    BCVEGPY2.1 LINUX version ..............'
      write(3,'(a)') 
     &	'......................................................'
	write(*,'(a)') 
     &	'.... copyright (c)  c.h. chang  and x.g. wu  ..........'
      write(3,'(a)') 
     &	'.... copyright (c)  c.h. chang  and x.g. wu  ..........'
	write(*,'(a)') 
     &	'.... any question email to: wuxg@itp.ac.cn  ..........'
      write(3,'(a)') 
     &	'.... any question email to: wuxg@itp.ac.cn  ..........'
	write(*,'(a)') 
     &	'......................................................'
	write(*,'(a)') 
     &	'......................................................'
      write(3,'(a)') 
     &	'......................................................'
      write(3,'(a)') 
     &	'......................................................'
       write(*,*)
       write(3,*)
      write(*,'(a)')
     &  '******************************************************'
      write(3,'(a)')
     &  '******************************************************'
      write(*,'(a)')
     &  '**********        running information       **********'
      write(3,'(a)')
     &  '**********        running information       **********'
      write(*,'(a)')
     &  '******************************************************'
      write(3,'(a)')
     &  '******************************************************'
       write(*,*)
       write(3,*)

c*****************************************************************

	write(*,'(a)') 
     &	'......................................................'
	write(*,'(a)')
     &	'.............. initial parameters ....................'
	write(*,'(a)') 
     &	'......................................................'
      write(3,'(a)') 
     &	'......................................................'
	write(3,'(a)') 
     &	'.............. initial parameters ....................'
      write(3,'(a)') 
     &	'......................................................'

      if(imix.eq.1) go to 111
      
      if(ibcstate.eq.3) then
        write(*,*)
	  write(*,'(a)') 'get the result for bc in 1^p_1.'
	  write(3,*)
	  write(3,'(a)') 'get the result for bc in 1^p_1.'
	end if

      if(ibcstate.eq.4) then
        write(*,*)
	  write(*,'(a)') 'get the result for bc in 3^p_0.'
	  write(3,*)
	  write(3,'(a)') 'get the result for bc in 3^p_0.'
	end if

      if(ibcstate.eq.5) then
        write(*,*)
	  write(*,'(a)') 'get the result for bc in 3^p_1.'
	  write(3,*)
	  write(3,'(a)') 'get the result for bc in 3^p_1.'
	end if

      if(ibcstate.eq.6) then
        write(*,*)
	  write(*,'(a)') 'get the result for bc in 3^p_2.'
	  write(3,*)
	  write(3,'(a)') 'get the result for bc in 3^p_2.'
	end if

	if (ibcstate.eq.1) then
	  write(*,*)
	  write(*,'(a)') 'get the result for bc in 1^s_0.'
	  write(3,*)
	  write(3,'(a)') 'get the result for bc in 1^s_0.'
	end if

	if (ibcstate.eq.2) then
	  write(*,*)
	  write(*,'(a)') 'get the result for bc in 3^s_1.'
	  write(3,*)
	  write(3,'(a)') 'get the result for bc in 3^s_1.'
	end if

      if(ioctet.eq.0) then
	  write(*,'(a)') 'bc in color-singlet state.'
	  write(3,'(a)') 'bc in color-singlet state.'
	end if

	if(ioctet.eq.1) then 
	  write(*,'(a)') 'bc in color-octet state.'
	  write(3,'(a)') 'bc in color-octet state.'
	end if

111   continue

c*************************************************

	if(imix.eq.1) then
         write(*,*)
         write(3,*)
	 write(*,'(a)') 'get the mixing results for gluon-gluon fusion.'
	 write(3,'(a)') 'get the mixing results for gluon-gluon fusion.'
	 if(imixtype.eq.1) then
	  write(*,'(a)') 
     &	'states to be mixed: 1s0,3s1,1p1,3p0,3p1,3p2,(1s0)_8,(3s1)_8'
	  write(3,'(a)')
     &	'states to be mixed: 1s0,3s1,1p1,3p0,3p1,3p2,(1s0)_8,(3s1)_8'
	 end if
	 if(imixtype.eq.2) then
	  write(*,'(a)') 'states to be mixed: color-singlet 1s0,3s1'
	  write(3,'(a)') 'states to be mixed: color-singlet 1s0,3s1'
	 end if
	 if(imixtype.eq.3) then
	  write(*,'(a)') 
     &	  'states to be mixed: 1p1,3p0,3p1,3p2,(1s0)_8,(3s1)_8'
	  write(3,'(a)') 
     &	  'states to be mixed: 1p1,3p0,3p1,3p2,(1s0)_8,(3s1)_8'
	 end if
	end if

c*************************************************

	if(npdfu.eq.1) then
	  write(3,41)'generate evnts', nev,'teva energy(gev)',ecm
	  write(*,41)'generate evnts', nev,'teva energy(gev)',ecm
	else    !npdfu=2
	  write(3,41) 'generate evnts', nev,'lhc energy(gev)',ecm
	  write(*,41) 'generate evnts', nev,'lhc energy(gev)',ecm
	end if

      write(*,'(a)') '*************************************************'
      write(3,'(a)') '*************************************************'

      if(iqqbar.eq.0) then
	 write(*,'(a)')'*   using subprocess: g+g->bc+b+\bar{c}       *'
	 write(3,'(a)')'*   using subprocess: g+g->bc+b+\bar{c}       *'
	else
	 if(iqcode.eq.1) then
	  if(ioutpdf.eq.1) then
c...for outer pdf, code for u is 1 , d is 2 and s is 3.
	   write(*,'(a)')'*   using subprocess: u+~u->bc+b+\bar{c}   *'
	   write(3,'(a)')'*   using subprocess: u+~u->bc+b+\bar{c}   *'
	  end if
c...in pythia, code for d is 1, u is 2 and s is 3.
	  if(ioutpdf.eq.0) then
	   write(*,'(a)')'*   using subprocess: d+~d->bc+b+\bar{c}   *'
	   write(3,'(a)')'*   using subprocess: d+~d->bc+b+\bar{c}   *'
	  end if
       end if
	 if(iqcode.eq.2) then
c...for outer pdf, code for u is 1, d is 2 and s is 3.
	  if(ioutpdf.eq.1) then
	   write(*,'(a)')'*   using subprocess: d+~d->bc+b+\bar{c}   *'
	   write(3,'(a)')'*   using subprocess: d+~d->bc+b+\bar{c}   *'
        end if
c...in pythia, code for d is 1, u is 2 and s is 3.
	  if(ioutpdf.eq.0) then
	   write(*,'(a)')'*   using subprocess: u+~u->bc+b+\bar{c}   *'
	   write(3,'(a)')'*   using subprocess: u+~u->bc+b+\bar{c}   *'
        end if
	 end if
	 if(iqcode.eq.3) then
	  write(*,'(a)')'*   using subprocess: s+~s->bc+b+\bar{c}   *'
	  write(3,'(a)')'*   using subprocess: s+~s->bc+b+\bar{c}   *'
       end if
	end if

      write(*,'(a)') '*************************************************'
      write(3,'(a)') '*************************************************'
	
	if(imix.eq.0) then
         if(ibcstate.lt.3) then
	   write(3,32) pmbc,pmb,pmc,fbc
	   write(*,32) pmbc,pmb,pmc,fbc
         else
           write(3,33) pmbc,pmb,pmc,fbc
           write(*,33) pmbc,pmb,pmc,fbc
         end if
	else
	 write(3,31) pmbc,pmb,pmc
	 write(*,31) pmbc,pmb,pmc
         write(*,'(a)') "R(0)=1.241 and R'(0)=0.44833"
         write(3,'(a)') "R(0)=1.241 and R'(0)=0.44833"
	end if

	if(isubonly.eq.0) then
	 if(mstu(111).eq.0) alporder='const'
	 if(mstu(111).eq.1) alporder='lo'
	 if(mstu(111).eq.2) alporder='nlo'
	 write(*,'(a,i2,3x,a,a)')'q2 type=',nq2, 'alph order=',alporder
       write(3,'(a,i2,3x,a,a)')'q2 type=',nq2, 'alph order=',alporder
	 if(ioutpdf.eq.0) then
	  write(*,'(a)') 'the innter pdfs which are from pythia'
	  write(3,'(a)') 'the innter pdfs which are from pythia'
c...you may add some more from pythia manual book.
	  if(mstp(51).eq.1) then
	    write(*,'(a)')'pdf: cteq 3l'
	    write(3,'(a)')'pdf: cteq 3l'
	  end if
	  if(mstp(51).eq.2) then
	    write(*,'(a)')'pdf: cteq 3m'
	    write(3,'(a)')'pdf: cteq 3m'
	  end if
	  if(mstp(51).eq.7) then
	    write(*,'(a)')'pdf: cteq 5l'
	    write(3,'(a)')'pdf: cteq 5l'
	  end if
	  if(mstp(51).eq.8) then
	    write(*,'(a)')'pdf: cteq 5m1'
	    write(3,'(a)')'pdf: cteq 5m1'
	  end if
	  if(mstp(51).eq.13) then
	    write(*,'(a)')'pdf: ehlq set 2'
	    write(3,'(a)')'pdf: ehlq set 2'
	  end if
	 end if
	 if(ioutpdf.eq.1) then
	 write(*,'(a)') 'the new version of outer pdfs which are from www'
	 write(3,'(a)') 'the new version of outer pdfs which are from www'
	   if(ipdfnum.eq.100) then
	     write(*,'(a)')'pdf: grv98lo; alph in lo'
		 write(3,'(a)')'pdf: grv98lo; alph in lo'
	   end if
	   if(ipdfnum.eq.200) then
	     write(*,'(a)')'pdf: mstr2001lo; alph in lo'
		 write(3,'(a)')'pdf: mstr2001lo; alph in lo'
	   end if
	   if(ipdfnum.eq.300) then
	     write(*,'(a)')'pdf: cteq6l1; alph in lo'
		 write(3,'(a)')'pdf: cteq6l1; alph in lo'
	   end if
	 end if
	end if

	if(isubonly.eq.1) then
	  write(*,'(a)') 'alpha_s take constant value: 0.20'
	end if

      write(*,'(a,i3)') 'using pythia idwtup=',idpp
	write(3,'(a,i3)') 'using pythia idwtup=',idpp
	if(unwght) then
	write(*,'(a)') 'generate unweight events unwght= true'
	write(3,'(a)') 'generate unweight events unwght= true'
	else 
	write(*,'(a)') 'generate unweight events unwght= false'
	write(3,'(a)') 'generate unweight events unwght= false'
	end if

	write(*,'(a,g10.4,a)') 'ptcut=',ptcut,'gev'
	write(3,'(a,g10.4,a)') 'ptcut=',ptcut,'gev'

	if(etacut.gt.1.0d+5) then
	   write(*,'(a)') 'no rapidity cut'
	   write(3,'(a)') 'no rapidity cut'
	else
	   write(*,'(a,g10.4)') 'rapidity cut=',etacut
	   write(3,'(a,g10.4)') 'rapidity cut=',etacut
	end if

       write(*,'(a)') '------------------------------------'
       write(3,'(a)') '------------------------------------'

31    format('m_{bc}=',g11.5,'m_{b}=',g11.5,'m_{c}=',g11.5)
32    format('m_{bc}=',g11.5,'m_{b}=',g11.5,'m_{c}=',g11.5,
     &       'R(0)=',g11.5)
33    format('m_{bc}=',g11.5,'m_{b}=',g11.5,'m_{c}=',g11.5,
     &       "R'(0)=",g11.5)

41    format(a,1x,i9,';',3x,a,g11.3,1x)

	end
