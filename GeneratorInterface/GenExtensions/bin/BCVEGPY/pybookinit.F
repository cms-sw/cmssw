c*******************************************************
c*******************************************************
c***              initialize the pybook
c*******************************************************
c*******************************************************
      subroutine pybookinit
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/histcol/inx
      common/subopen/subfactor,subenergy,isubonly
      common/extraz/zfactor,zmin,zmax

c************************************************
c...typical histogram range of the kinematic variables.
      call uphistrange(isubonly)

c...book histograms.
      if(isubonly.eq.1) inx=50
      if(isubonly.eq.0) inx=100

	call pybook(1,'pt spectrum',inx,ptmin,ptmax)
      call pybook(2,'rap spectrum',inx,etamin,etamax)
	call pybook(3,'cm energy spectrum',inx,smin,smax)
	call pybook(4,'y spectrum',inx,ymin,ymax)
	call pybook(5,'pseudo rap spectrum',inx,psetamin,psetamax)
	call pybook(6,'z spectrum',inx,zmin,zmax)

      call pybook(11,'rap spectrum (pt=5.00)',inx,etamin,etamax)
      call pybook(12,'rap spectrum (pt=20.0)',inx,etamin,etamax)
      call pybook(13,'rap spectrum (pt=35.0)',inx,etamin,etamax)
      call pybook(14,'rap spectrum (pt=50.0)',inx,etamin,etamax)
      call pybook(15,'rap spectrum (pt=100.)',inx,etamin,etamax)

	call pybook(21,'pt spectrum (|y|=0.5)',inx,ptmin,ptmax)
	call pybook(22,'pt spectrum (|y|=1.0)',inx,ptmin,ptmax)
	call pybook(23,'pt spectrum (|y|=1.5)',inx,ptmin,ptmax)
	call pybook(24,'pt spectrum (|y|=2.0)',inx,ptmin,ptmax)
	call pybook(25,'pt spectrum (|y|=2.5)',inx,ptmin,ptmax)

	call pybook(31,'cm energy spectrum (|y|=0.5)',inx,smin,smax)
	call pybook(32,'cm energy spectrum (|y|=1.0)',inx,smin,smax)
	call pybook(33,'cm energy spectrum (|y|=1.5)',inx,smin,smax)
	call pybook(34,'cm energy spectrum (|y|=2.0)',inx,smin,smax)
	call pybook(35,'cm energy spectrum (|y|=2.5)',inx,smin,smax)

	call pybook(41,'cm energy spectrum (pt=5.00)',inx,smin,smax)
	call pybook(42,'cm energy spectrum (pt=20.0)',inx,smin,smax)
	call pybook(43,'cm energy spectrum (pt=35.0)',inx,smin,smax)
	call pybook(44,'cm energy spectrum (pt=50.0)',inx,smin,smax)
	call pybook(45,'cm energy spectrum (pt=100.)',inx,smin,smax)

	call pybook(52,'pt spectrum (0.5<|y|<1.0)',inx,ptmin,ptmax)
	call pybook(53,'pt spectrum (1.0<|y|<1.5)',inx,ptmin,ptmax)
	call pybook(54,'pt spectrum (1.5<|y|<2.0)',inx,ptmin,ptmax)
	call pybook(55,'pt spectrum (2.0<|y|<2.5)',inx,ptmin,ptmax)

      return
	end
	
