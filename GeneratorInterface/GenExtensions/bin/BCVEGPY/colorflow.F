
c**********************************************************
c***** setting the final state particles' information
c***** mainly setting the color flow of the process
c**********************************************************
      subroutine bcpythia(ipartic)
 
c...double precision and integer declarations.
      implicit double precision(a-h, o-z)
      implicit integer(i-n)

c...pythia common block.
      parameter (maxnup=500)
	common/pypars/mstp(200),parp(200),msti(200),pari(200)
      common/hepeup/nup,idprup,xwgtup,scalup,aqedup,aqcdup,idup(maxnup),
     &istup(maxnup),mothup(2,maxnup),icolup(2,maxnup),pup(5,maxnup),
     &vtimup(maxnup),spinup(maxnup)
      save /hepeup/

c...the user's own transfer of information.
 	common/colflow/amp2cf(10),smatval
	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
c...color-octet.
      common/coloct/ioctet

c...define number of partons - two incoming and three outgoing.
      nup=5

      if(ipartic.eq.21) then
c...set up flavour and history of g + g -> b_c + b + cbar. 
       idup(1)= 21	   ! gluon 1
       idup(2)= 21	   ! gluon 2
      else
c...set up flavour and history of q + \bar{q} -> b_c + b + cbar. 
	 if(ipartic.eq.1) then
	   idup(1)= 1	   ! u 
         idup(2)= -1	   ! \bar{u}
	 end if
	 if(ipartic.eq.2) then
	   idup(1)= 2	   ! d 
         idup(2)= -2	   ! \bar{d}
	 end if
	 if(ipartic.eq.3) then
	   idup(1)= 3	   ! s 
         idup(2)= -3	   ! \bar{s}
	 end if
	end if

      idup(3)= 541   ! b_c+
      idup(4)= 5	   ! b-quark
      idup(5)=-4	   ! c-antiquark

c...status codes.

      istup(1)=-1
      istup(2)=-1
      istup(3)= 1
      istup(4)= 1
      istup(5)= 1

c...mother codes.
      
      mothup(1,1)=0	! gluon 1 or q <-- proton 1
      mothup(2,1)=0      
      mothup(1,2)=0	! gluon 2 or \bar{q} <-- proton 2
      mothup(2,2)=0      
      mothup(1,3)=1	! bc+
      mothup(2,3)=2      
      mothup(1,4)=1	! b
      mothup(2,4)=2      
      mothup(1,5)=1	!cbar
      mothup(2,5)=2

c...seclect one of the three possible types of color flow at random, 
c...according to its contribution to the square of amplitude.
      ranmat =smatval*pyr(0)
	icolnum=0

      if(ipartic.eq.21) then
	  iuplimit=3
	else
	  iuplimit=2
	end if

	if(ioctet.eq.1) then
	  iuplimit=10
	end if

110   icolnum=icolnum+1
      ranmat =ranmat-amp2cf(icolnum)
      if(icolnum.lt.iuplimit .and. ranmat.gt.1.0d-16) go to 110

      if(ipartic.eq.21) then
c----------------
	if(ioctet.eq.0) then
c...three color flow for color-singlet in the gluon-gluon fusion.
	 if(icolnum.eq.1) then
        icolup(1,1)=502
        icolup(2,1)=501
	  icolup(1,2)=503
	  icolup(2,2)=502
        icolup(1,3)=0
	  icolup(2,3)=0
        icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.2) then
        icolup(1,1)=503
        icolup(2,1)=502
	  icolup(1,2)=502
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0
        icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501 
       end if

	 if(icolnum.eq.3) then
        icolup(1,1)=501
        icolup(2,1)=502
	  icolup(1,2)=502
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0
        icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=503
       end if
	end if

c------the following is for color-octet and can be found in ref.
c... hep-ph/0504017. however, in pythia, at the present, there is
c... no color-octet meson, so practically, one might think the
c... color-octet meson will emit a soft gluon with 100% and then 
c... changes to a color singlet state. under such prescription,
c... one may observe that the ten color flow are correspond to
c... three independent flow.
      if(ioctet.eq.1) then
c...ten color flow for color-octet in the gluon-gluon fusion.
	 if(icolnum.eq.1) then
        icolup(1,1)=503
        icolup(2,1)=502
	  icolup(1,2)=502
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.2) then
        icolup(1,1)=502
        icolup(2,1)=501
	  icolup(1,2)=503
	  icolup(2,2)=502
        icolup(1,3)=0
	  icolup(2,3)=0        
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.3) then
        icolup(1,1)=503
        icolup(2,1)=502
	  icolup(1,2)=502
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.4) then
        icolup(1,1)=502
        icolup(2,1)=501
	  icolup(1,2)=503
	  icolup(2,2)=502
        icolup(1,3)=0
	  icolup(2,3)=0        
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.5) then
        icolup(1,1)=503
        icolup(2,1)=502
	  icolup(1,2)=502
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0        
        icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.6) then
        icolup(1,1)=502
        icolup(2,1)=501
	  icolup(1,2)=503
	  icolup(2,2)=502
        icolup(1,3)=0
	  icolup(2,3)=0        
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.7) then
        icolup(1,1)=503
        icolup(2,1)=502
	  icolup(1,2)=502
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0        
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.8) then
        icolup(1,1)=502
        icolup(2,1)=501
	  icolup(1,2)=503
	  icolup(2,2)=502
        icolup(1,3)=0
	  icolup(2,3)=0        
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.9) then
        icolup(1,1)=503
        icolup(2,1)=502
	  icolup(1,2)=502
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0        
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if

	 if(icolnum.eq.10) then
        icolup(1,1)=502
        icolup(2,1)=501
	  icolup(1,2)=503
	  icolup(2,2)=502
        icolup(1,3)=0
	  icolup(2,3)=0        
	  icolup(1,4)=503
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=501
       end if
	end if

c--------the following is the theoretical color-flow from 
c--------hep-ph/0504017.
c      if(ioctet.eq.1) then
c...three color flow for color-octet in the gluon-gluon fusion.
c	 if(icolnum.eq.1) then
c        icolup(1,1)=503
c        icolup(2,1)=502
c	  icolup(1,2)=502
c	  icolup(2,2)=501
c        icolup(1,3)=504   ! bc in color-octet state.
c	  icolup(2,3)=503
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c
c	 if(icolnum.eq.2) then
c        icolup(1,1)=502
c        icolup(2,1)=501
c	  icolup(1,2)=503
c	  icolup(2,2)=502
c        icolup(1,3)=504
c	  icolup(2,3)=503
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c
c	 if(icolnum.eq.3) then
c        icolup(1,1)=503
c        icolup(2,1)=502
c	  icolup(1,2)=501
c	  icolup(2,2)=504
c        icolup(1,3)=502
c	  icolup(2,3)=503
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c
c	 if(icolnum.eq.4) then
c        icolup(1,1)=501
c        icolup(2,1)=504
c	  icolup(1,2)=503
c	  icolup(2,2)=502
c        icolup(1,3)=502
c	  icolup(2,3)=503
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c
c	 if(icolnum.eq.5) then
c        icolup(1,1)=503
c        icolup(2,1)=502
c	  icolup(1,2)=502
c	  icolup(2,2)=501
c        icolup(1,3)=501
c	  icolup(2,3)=503
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=504
c       end if

c	 if(icolnum.eq.6) then
c        icolup(1,1)=502
c        icolup(2,1)=501
c	  icolup(1,2)=503
c	  icolup(2,2)=502
c        icolup(1,3)=501
c	  icolup(2,3)=503
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=504
c       end if
c
c	 if(icolnum.eq.7) then
c        icolup(1,1)=504
c        icolup(2,1)=503
c	  icolup(1,2)=503
c	  icolup(2,2)=502
c        icolup(1,3)=502
c	  icolup(2,3)=501
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c
c	 if(icolnum.eq.8) then
c        icolup(1,1)=503
c        icolup(2,1)=502
c	  icolup(1,2)=504
c	  icolup(2,2)=503
c        icolup(1,3)=502
c	  icolup(2,3)=501
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c
c	 if(icolnum.eq.9) then
c        icolup(1,1)=502
c        icolup(2,1)=501
c	  icolup(1,2)=504
c	  icolup(2,2)=503
c        icolup(1,3)=503
c	  icolup(2,3)=502
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c
c	 if(icolnum.eq.10) then
c        icolup(1,1)=504
c        icolup(2,1)=503
c	  icolup(1,2)=502
c	  icolup(2,2)=501
c        icolup(1,3)=503
c	  icolup(2,3)=502
c	  icolup(1,4)=504
c        icolup(2,4)=0
c        icolup(1,5)=0
c        icolup(2,5)=501
c       end if
c	end if
c-------
	else
c...two color flow for quark-antiquark annihilation.
	 if(icolnum.eq.1) then
        icolup(1,1)=501
        icolup(2,1)=0
	  icolup(1,2)=0
	  icolup(2,2)=502
        icolup(1,3)=0
	  icolup(2,3)=0
        icolup(1,4)=501
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=502
	 end if

	 if(icolnum.eq.2) then
        icolup(1,1)=501
        icolup(2,1)=0
	  icolup(1,2)=0
	  icolup(2,2)=501
        icolup(1,3)=0
	  icolup(2,3)=0
        icolup(1,4)=502
        icolup(2,4)=0
        icolup(1,5)=0
        icolup(2,5)=502
	 end if
      end if

      return
      end
