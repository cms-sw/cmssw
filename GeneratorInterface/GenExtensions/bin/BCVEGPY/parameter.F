cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...this subroutine is used to set the necessary parameters for      c
c...the initialization for hadronic production of bc meson.          c
c...to use the program youd need to make a directory: (data) to      c
c...save all the obtained data-files.                                c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   to have a better understanding of setting the parameters         c
c   you may see the README file to get more detailed information.    c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c copyright (c) z.x zhang, chafik driouich, paula eerola and x.g. wu c
c reference: hep-ph/0309120                                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine setparameter
c...preamble: declarations.
        implicit double precision(a-h, o-z)
	implicit integer(i-n)

#include "bcvegpy_set_par.inc"

c...user process event common block.
      common/pypars/mstp(200),parp(200),msti(200),pari(200)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydatr/mrpy(6),rrpy(100)

c...transform of running information.
	common/confine/ptcut,etacut
c...parameters transformtion.
	double complex colmat,bundamp
      common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/funtrans/nq2,npdfu
	common/rconst/pi
	common/usertran/ishower,idpp
	common/vegcross/vegsec,vegerr,iveggrade
c...to transform the subprocess cross-section.
      common/subopen/subfactor,subenergy,isubonly
c...transform of pdf information
	common/outpdf/ioutpdf,ipdfnum
c...transform the subprocess information
      common/qqbar/iqqbar,iqcode
c...transform of the vegas information
      common/vegasinf/number,nitmx
c...transform the events number and bc state.
	common/counter/ibcstate,nev
c...transform some variables
      common/loggrade/ievntdis,igenerate,ivegasopen,igrade
c...ioctet--whether getting the color-octet component contributions. 
c...here only for gg->(c\bar{b})+b+~c, (c\bar{b}) in color-octet 
c...s-waee states. coeoct--coefficient for color-octet
c...matrixment: relation between the color-octet matrixment and the
c...color singlet matrixment.
      common/coloct/ioctet
	common/octmatrix/coeoct
c...transform the first derivative of the wavefunction at the zero or the
c...wavefunction at the zero.
	common/wavezero/fbc
c...xbcsec(8) records the total differential cross-sections for different
c...states: 1---singlet 1s0; 2---singlet 3s1; 7---octet 1s0; 8---octet 3s1;
c...3---singlet 1p1; 4---singlet 3p0; 5---singlet 3p1; 6---singlet 3p2.
	common/mixevnt/xbcsec(8),imix,imixtype
	logical wronginput,unwght
	common/unweight/unwght
c
c... Read parameter setting from file bcvegpy_set_par.nam
c	
        Call Read_parameter_settings
c	
c... change the intitial state of the random number
c        mrpy(1) = 19780503		! default value
	 mrpy(1) = ibcrandom
	 write( 6, * ) ' '
	 write( 6, * ) ' Change default value of random, mrpy(1), to ',
     +	 mrpy(1)	  
	 write( 6, * ) ' '
c... end random change	 	

	pi = dacos(-1.0d0)

c*************************************************
c** some commen parameters that are used for both
c** imix=0 or imix=1 are listed here for convenience.
c*************************************************

c...setting the basic parameters. note pmbc should be equal to
c...(pmb+pmc) to ensure the gauge invariance of the amplitude.
c      pmb  =4.9d0    !b quark mass (gev)
c      pmc  =1.386d0    !~c quark mass (gev)
      pmb  = pmassb    !b quark mass (gev)
      pmc  = pmassc    !~c quark mass (gev)
      pmbc = pmb+pmc   !~bc mass (gev)

c------------------------------------------
c...kinematical variables
      ptcut =0.0d+0   !bc p_t cut (gev)
      etacut=1.0d+9   !bc rapidity (y) cut

c---------------------------------------
c...npdfu=2:lhc=1.40d+4GeV; npdfu=1:tevatron=1.96d+3GeV.
c---------------------------------------

c...hadron information
      nq2   =3      !types of q^2 (seven typical energy scales are
c                     ! designed in program)
c      npdfu =2      !types of incoming beams. =1, tevatron; =2, lhc.
      npdfu = naccel
      if(npdfu.eq.1) ecm=ENERGYOFTEVA  !defined in run.F
      if(npdfu.eq.2) ecm=ENERGYOFLHC   !defined in run.F
      mstp(51) =7   !types of parton distribution functions. used for
c                     !pythia inner pdf and under the condition ioutpdf=0.
      idpp     =3      !=idwtup: types of pythia generates events
      mstu(111)=1      !setting the value of alphas. pythia parameter
      paru(111)=0.2d0  !constant value of alphas, used when mstu(111)=0.
      ievntdis =0      !switch on/off to get the event number
c                        !distributions for p_t, rap, shat, pseta, y, z.

c------------------------------------------
c...event information
      unwght =.true. !generate the unweighted events
      igenerate =0 !whether generating full events must be with idwtup=1
      ishower=0   !switch off aspects: initial and final state showers,
c                    !multiple interactions and hadronization.

c...using pythia pdf or not.
      ioutpdf  =1
c...=100, grv98lo; =200, mrst2001l; =300, cteq6l1
      ipdfnum  =300
      if(ioutpdf.eq.1 .and. ipdfnum.eq.300) call setctq6(4)   !cteq6l1

c**************************************************
c**************************************************
c...whether to get the mixed events for bc meson. 
c...here, we only consider the mixed results for
c...the dominant gluon-gluon fusion mechanism.
c--------------------------------------------------
c...if(imix==1), then we can generate the mixed events 
c...up to the eight states (1s0,3s1,1p1,3p0,3p1,3p2,1s0_8,3s1_8), 
c...which are produced through the gluon-gluon fusion mechanism.
c...if(imix==0), then one can generate every states separately.
c**************************************************
c	imix=1
        imix = i_mix

c--------------------------------------------------
c***** set the parameters for mixing results
c--------------------------------------------------

	if (imix.eq.1) then
c*************************************************
c...setting fixed values for mixing processes. 
c...imixtype: set how many states need to be mixed:
c...imixtype=1: all the eight states need to be mixed.
c...imixtype=2: the mixed events for two color-singlet s-wave states.
c...imixtype=3: the mixed events for the four color-singlet 
c...            p-wave plus two color-octet s-wave states,
c************************************************
        imixtype = mix_type

c*************************************************
c...some of the parameters are fixed only for simplicity.
c...here, to make our program short, we only provide
c...three possible way to get the mixed results are programmed:
c... 1) ivegasopen=1, no matter what value of igrade.
c       iveggrade=0, not improve the existed grade.
c... 2) ivegasopen=1, no matter what value of igrade.
c       iveggrade=1, to improve the existed grade.
c... 3) ivegasopen=0 and igrade=1.
c... 4) ivegasopen=0 and igrade=0.
c*************************************************
        mixmethod = igentype

        if(mixmethod.eq.1) then
	     ivegasopen=1
           igrade    =0   !or 1, no use.
c***********************
c...improve vegas grade
         iveggrade=0    ! not improve the grade
        end if
        if(mixmethod.eq.2) then
           ivegasopen=1
           igrade    =0
           iveggrade=1    ! use existed grade for more precise
c                         ! grade, generated under same condition
        end if
        if(mixmethod.eq.3) then
           ivegasopen=0
           igrade    =1
           iveggrade=0    ! not improve the grade
        end if
       if(mixmethod.eq.4) then
           ivegasopen=0
           igrade    =0
           iveggrade=0    ! not improve the grade
        end if

c----------------------------
c... the following value should be increased
c... to achieve the users precision goal.
c----------------------------
	  number=NVEGCALL
	  nitmx =NVEGITMX  
	  nev   =NUMOFEVENTS
        igrade   =1   ! no use when ivegasopen=1

c---------------------------------------------------
c-- fixed parameter in the case of imix=1
        isubonly =0
	  subenergy=100.0d0  ! no use for present purpose
        iqqbar   =0        ! should be fixed to 0.
        iqcode   =2        ! no use for present purpose

	  return
	end if
c--------------------------------------------------
c***** end of the initialization for mixing results
c--------------------------------------------------

c******************************************************
c******************************************************
c***** The following is for generating single state
c******************************************************
c******************************************************

c*****************************************
c*** this is the previous way for easy running. with ireaddata=1, 
c*** it will read the all the necessary parameter values from the 
c*** datafile (totput.dat).
c***
c*** since now we take ( makefile ) to do the compiling,
c*** such method is not very useful and now is commented out.
c***
c***	ireaddata=0
c***	if(ireaddata.eq.1) goto 1001
c*****************************************
c*****************************************
c...note: the following initial value is only for checking that
c...the program is ok. to do numerical calculations/event generation, 
c...one needs to be careful about all these values, especailly
c...those parameters that can improve the precision.
c*****************************************

c****************************************
c...to choose what bc state we need to generate.
c...=1, color-singlet 1s0 ;=2, color-singlet 3s1 ;=3, 1p1 ;=4, 3p0 ;
c...=5, 3p1 ;=6, 3p2 ;=7, color-octet 1s0 ;=8, color-octet 3s1.
	ibcstate  =4

c*****************************************
c...for s-wave, r(0)=1.241  --->  \psi(0)=(0.35).
      ioctet=0
      if(ibcstate.eq.7 .or. ibcstate.eq.8) then
        ioctet=1
        coeoct=0.2d0
        if(ibcstate.eq.7) ibcstate=1
        if(ibcstate.eq.8) ibcstate=2
      end if

c******************************************
	if(ibcstate.eq.1 .or. ibcstate.eq.2) then 
	  fbc =1.241
	  fbcc=dsqrt(3.0d0*fbc**2/pi/pmbc)
	else
c...for p-wave, r'(0)=0.44833  --->  \psi'(0)=(0.219); r'(0)**2=(0.201)
	  fbc =0.44833
c...the value of p-wave matrix element <0|p|0>.
	  fbcc=fbc**2*9.0d0/(2.0d0*pi)
	end if



	number=NVEGCALL !total number of calling fxn in each iteration in vegas
	nitmx =NVEGITMX     !maximum iteration used in vegas
	nev   =NUMOFEVENTS   !number of events to be generated

c****************************************
c...improve vegas grade
c      iveggrade=0  ! switch on/off to use the existed grade before 
c                  ! running vegas, the initial condition for the running
c                  ! and that for getting the existed grade should be the
c                  ! same. this is used to get a more precise sampling
c                  ! grade. used together with ivegasopen=1.

c***************************************
c...subprocess information
	isubonly =0    !switch on whether only to get the information, 
c                      !for the subprocess including cross-section, pt, 
c                      !rapidity distributions and so on.
c...c.m. energy(gev) of the subprocess gg->bc+b+c~
      if(isubonly.eq.1) subenergy=100.0d0 

c***************************************
c...mechanism through subprocess q~q->bc
      iqqbar   =0      !switch on/off whether using the subprocess 
c	                 q+\bar{q}-> to generate the bc events.
c...all the quarks are massless.
c...=1, u; =2, d; =3, s
      if(iqqbar.eq.1) iqcode=2
       
c  1001  continue

c****************************************
c****************************************
c****************************************
c**** comment out the method of reading data from outer 
c**** data file.
c**** this way is the old way to save compile time and is
c**** not necessary for the present linux version, since 
c**** now we use ( make ) to compile the program.
c****************************************
c...open input data file.
c	if(ireaddata.eq.1) then
c	open(unit=1,file='totput.dat',status='old',access='sequential',
c     &    form='formatted')
c...read the (zero point wave function) -----> initial for fbcc.
c       read(1,*) pmbc,pmb,pmc,fbcc
c*** here the input ibcstate should be 1,2,3,4,5,6.
c       read(1,*) ptcut,etacut,ecm,ibcstate,igenerate,ivegasopen
c	read(1,*) number,nitmx
c       read(1,*) nq2,npdfu,nev
c       read(1,*) ishower,mstp51,idpp,mstu111,paru111
c	read(1,*) isubonly,subenergy,igrade
c       read(1,*) inumevnt,iveggrade,iqqbar,iqcode
c       read(1,*) ioutpdf,ipdfnum,ioctet,coeoct
c
c****************************************
c         if(ibcstate.eq.7 .or. ibcstate.eq.8) then
c          if(ibcstate.eq.7) ibcstate=1
c          if(ibcstate.eq.8) ibcstate=2
c          ioctet=1
c          coeoct=0.2d0
c         end if

c****************************************
c	 if(ibcstate.eq.1 .or. ibcstate.eq.2) then
c...decay constant f_{bc} of s-wave.
c	   fbc=fbcc
c	   fbcc=dsqrt(fbcc**2*3.0d0/(pmbc*pi))
c	 else
c...the value of p-wave matrix element <0|p|0>.
c	   fbc=fbcc
c	   fbcc=fbcc**2*9.0d0/(2.0d0*pi)
c	 end if
c
c	 mstp(51)=mstp51
c	 mstu(111)=mstu111
c	 paru(111)=paru111
c
c	 close(1)
c	end if
c***************************************
c***************************************
c***************************************

c----------------------------------------------
c...error message.
      wronginput=.false.
	call uperror(wronginput)
	if(wronginput) stop '-----input error! stop the program !!!'

	call parameters()
      call dparameters()
      call coupling()
	
	return
	end
c***************************************
c***************************************
	Subroutine Read_parameter_settings
c
c... Get parameters from namelist
        implicit double precision(a-h, o-z)
	implicit integer(i-n)
	
	Namelist / bcvegpy_set_par / ENERGYOFTEVA, ENERGYOFLHC,
     +	 pmassb,  pmassc, naccel, i_mix, mix_type, igentype, 
     +	 NUMOFEVENTS, NVEGCALL, NVEGITMX, ibcrandom
#include "bcvegpy_set_par.inc"
c
c-------------------------------------------------------------------------------
c
	open( unit=1, file='bcvegpy_set_par.nam',Status='Old',Err=99)
        read( 1, nml=bcvegpy_set_par, err=90)
	write( 6, * ) ' '
	write( 6, * ) ' Contents of namelist *bcvegpy_set_par*: '
        write( 6, nml=bcvegpy_set_par)
	write( 6, * ) ' '
        Close( 1 ) 
        Return
c
  90  	Write( 6, * ) ' !!!!! Unable to read namelist bcvegpy_set_par '  
        Call Exit 
  99	Write( 6, * ) ' !!!!! Unable to open bcvegpy_set_par.nam'
        Call Exit
 	End
