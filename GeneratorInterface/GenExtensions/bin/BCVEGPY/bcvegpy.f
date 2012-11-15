cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                            main idea                                 c
c  using vegas together with pythia to get the more precise results.   c
c  by the several first runs of in the vegas, we may get the optimized c
c  density distribution function which make the distribution of cross- c
c  section not fluctuate too much. After the running of vegas, it will c
c  call the pythia subroutines to generate events. in this way the     c
c  MC efficiency is greatly improved.                                  c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c    A pure linux version BCVEGPY2.1;    using GNU C compiler make     c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c!!!       to save all the datas, you have to create a directory    !!!c
c!!!              named ( data ) at the present directory.   !!!    !!!c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c main program for seting the external process gg->bc+b+\bar{c}        c
c into pythia, whose version is pythia6208.if higher version of pythia c
c exists, your may directly replay the old one with the new one, but   c
c remember to commoment out two subroutines upinit and upevnt there.   c
c we also can provide the quark-anti-quark annihilation mechanism      c
c which via the subprocess q+\bar{q}->Bc+b+\bar{c} with Bc in s-wave   c
c states. Such mechanism has a quite small contribution around 1% of   c
c that of gluon-gluon fusion.                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c  bcvegpy1.0   finished  in 10, auguest 2003                          c
c               improved  in 1, november 2003                          c
c copyright (c) c.h. chang, chafik driouich, paula eerola and x.g. wu  c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c the original version of bcvegpy is in reference: hep-ph/0309120      c
c or, in computer physics communication 159, 192-224(2004).            c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c second version bcvegpy2.0, in which (p-wave) generation based on     c
c the gluon-gluon fusion subprocess is given. the gauge invariance     c
c for all the p-wave states have been checked exactly. because we      c
c have implicitly used the propertities of poralization vector to      c
c simplify the amplitude, (such (p.\epsilon(p)=0)) the gauge check can c
c not be done by using the present amplitudes. the interesting reader  c
c may ask the authors for the (full amplitudes) that keep the gauge    c
c invariance exactly.         reference: hep-ph/0504017                c
c      computer physics communication 174, 41,251(2006)                c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c    bcvegpy2.0         finished in 24, febrary 2005                   c
c   copyright (c)       c.h chang, j.x. wang and x.g. wu               c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c problems or suggestions email to:         wuxg@itp.ac.cn             c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c this is the linux version for BCVEGPY2.1, with better modularity and c
c code reusability, i.e. less cross-talk among different modules.      c
c this version combines all the feedback suggestions from the users.   c
c thanks are given for: y.n. gao, j.b. he and z.w. yang                c
c                        finished in 6, April 2006                     c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c...main program.
c      program bcvegpy
      subroutine bcvegpy
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

c...three pythia functions return integers, so need declaring.
      external pydata

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
	common/counter/ibcstate,nev
	common/vegcross/vegsec,vegerr,iveggrade
	common/confine/ptcut,etacut
	logical generate
	common/genefull/generate
      common/totcross/appcross
      common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
      common/loggrade/ievntdis,igenerate,ivegasopen,igrade
	common/mixevnt/xbcsec(8),imix,imixtype
      	logical unwght
	common/unweight/unwght

	character*8 begin_time,end_time,blank
c....Temporaty file for initialization/event output
        MSTP(161) = 77
        OPEN (77, FILE='BCVEGPY.init',STATUS='unknown')
        MSTP(162) = 78
        OPEN (78, FILE='BCVEGPY.evnt',STATUS='unknown')
c....Final Les Houches Event file, obtained by combaining above two
        MSTP(163) = 79
        OPEN (79, FILE='BCVEGPY.lhe',STATUS='unknown')
c        MSTP(164) = 1 !save tmp file bcvegpy.evnt and bcvegpy.init

c*********************************************
c... setting initial parameters in parameter.F. 
c... User may change the values to his/her favorate one.
c*********************************************
      call setparameter

c... initialization, including: 
c... if ivegasopen=1, then generate vegas-grade; 
c... open files to record data (intermediate results for vegas or
c... pythia running information); 
c... if ivegasopen=0 and igrade=1, then initialize the grade; 
c... setting some initial values for pythia running.
	call evntinit

C...Fills the HEPRUP commonblock with info on incoming beams and allowed
C...processes, and optionally stores that information on file.
        call bcvegpy_PYUPIN
c...
c*****************************************************************
c...using pybook to record the different distributions or 
c...event-distributions.
c*****************************************************************
c*****************************************************************
c...any subroutines about pybook can be conveniently commented out
c...and then one can directly use his/her own way to record the data.
c*****************************************************************

c...pybook init.
	call pybookinit

c...approximate total cross-section.
	appcross =0.0d0
c
	blank  ='    '
	ncount =0
c	call time(begin_time)

c*******************************************************
c...there list some typical ways for recording the data.
c...users may use one convenient way/or their own way.
c******************************************************


	do iev=1,nev
c		call pyevnt  !don't use this subroutine
        if (unwght) then
         call upevnt
	      call  bcvegpy_write_lhe !if you want generate weighted LHE file need comment "if (unwght)"
      else 
	      call pyevnt !old mode generate weighted event for theoritical study
      end if

c
		if (idwtup.eq.1.and.iev.ne.1.and.generate) then
	        call pylist(7)
c	        call time(end_time)
	        print *, iev,blank,end_time
	    end if

		if(msti(51).eq.1) go to 400

c*********************************************************
	    do i=1,10
             if(idup(i).eq.541) then
c...pt of the Bc. 
             pt =sqrt(pup(1,i)*pup(1,i)+pup(2,i)*pup(2,i))  !sqrt(Px^2+Py^2)
c...true rapidity.
	          eta=0.5*log((pup(4,i)+pup(3,i))/(pup(4,i)-pup(3,i))) !0.5*log((E+pz)/(E-pz))
c...pseudo-rapidity
	          pseta=-log(tan( atan2(pt,pup(3,i))/2 ))

c...these two constrain (and other) may be added here to partly compensate
c...some numerical problems.
			  if(pt.lt.ptcut) xwgtup=0.0d0
	          if(abs(eta) .gt. etacut) xwgtup=0.0d0
c...rapidity of the hard-interaction subsystem(ln(x1/x2)/2.0)
			  y=pari(37)
c...the mass of the complete three- final state(\sqrt(shat))
			  st=pari(19)

c**********************************************************************
c...   users may use his own way to record the data      ..............
c**********************************************************************
c...to fill the histogram. we list three methods to get the differential
c...distributions. 1) idwtup=3; 2) (idwtup=1 and generate.eq.fause); 3)
c...(idwtup=1 and generate.eq.true). where method 1) and 2) are the 
c...quickest, while the third method is slow.

c...we also list three ways to get the event number distributions:
c...1) (idwtup=1 and generte.eq.true) and ievntdis.eq.1;
c...2)  idwtup=3 and ievntdis.eq.0; 3) (idwtup=1 and generate.eq.fause)
c...and ievntdis.eq.0; the method 2) and 3) are the same, both needs
c...a proper normalization for numbers and at the same time
c...recording every event with its corresponding weight so as to get
c...final right event number distributions. the method 1) is general
c...one used by experimental, which will spend a long time. so for 
c...theoretical studies we suggest using method 2) or 3).
c**********************************************************************
	call uppyfill(idwtup,generate,xwgtup,pt,eta,st,y,pseta)

			  isub=msti(1)
	          ncount=ncount+1
                  if(ncount.le.10) then
                  write(*,'(a,i5)') 'following event is subprocess',isub
                  call pylist(7)
		        call pylist(1)
                end if
                call pyedit(2)
	       end if
	    end do
	end do


c***************************************************************
c...close grade files
      call upclosegradefile(iveggrade,imix,imixtype)
c***************************************************************

c***************************************************************
c...can be commentted out by user, if using his/her own way to
c...record the data.
c--------------------------------------------------------------- 
c...pyfact all the pybooks.
		call uppyfact(idwtup,generate,ievntdis)
c...open files to record the obtained pybook data for distributions.
		call updatafile
c...dump the data into the corresponding files.
	   call uppydump
c...close pybook files.
	 call upclosepyfile
c***************************************************************

c	call time(end_time)
	write(3,'(a,d19.6,a)') "maximum diff. cross-sec=",crossmax,"pb"
	write(3,'(i9,3x,a,3x,a)') nev,begin_time,end_time

c...when the number of sampling points are high enough, it is
c...just the real value. for (idwtup.eq.1.and.ievntdis.eq.1), 
c...becaue of small event number the value of appcros is not
c...accurate.
	if(idwtup.ne.1.and.ievntdis.eq.1) then
	  write(3,'(a,d16.6,1x,a)') "!approxi. total cross-sec=",
     &	appcross,'nb'
      end if

c...store the approximate total cross-section.
c...appcross=\sum(xwgtup*wgt)/nev. when the number of sampling points 
c...are high enough, it will be the true value obtained from pythia.
		pari(1) =appcross*1.0d-3
		write(3,5111) 'cross-sec(pythia)=',pari(1),'mb'
		write(3,5112) "cross-sec(vegas)=", xsecup(1),'+-',xerrup(1),'nb'
		write(*,*) "cross-sec (vegas)=",xsecup(1),"+-",xerrup(1),"nb"

 5111 format(a,3x,d16.6,x,a)
 5112 format(a,3x,d16.6,x,a,x,d16.6,a)

c*****************************
c...histograms.
       call pyhist
c*****************************
400	continue
c...type cross-section table.
c      call pystat(1)
c
c      write(*,'(a)') 'the cross-section in PYSTAT table is nonsense'
c      write(*,'(a)') 'see the true value in obtained cross-section file'
c      write(*,*)
c      write(3,*)
c      write(3,'(a)') 'note: cross-section in PYSTAT table is nonsense,'
c      write(3,'(a)') 'especially for the mixed events.'
c      write(3,*)
       write(*,*)
       write(*,'(a)') '     program finished !'
       write(*,*)

c***************************

c***************************
c...close intemediate file.
      close(3)
      call pylhef

      end     !end of main program


