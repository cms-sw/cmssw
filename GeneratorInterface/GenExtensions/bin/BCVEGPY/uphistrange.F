      subroutine uphistrange(isubonly)
	implicit integer (i-n)
	implicit double precision (a-h,o-z)
	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
c...to get the distribution of an extra factor z=(2(k1+k2).p_bc)/shat.
      common/extraz/zfactor,zmin,zmax

c...typical histogram range of the kinematic variables.
c...you may short the range to get a more precise result.
	if(isubonly.eq.1) then
	  ptmin   = 0.0d0
	  ptmax   = 5.0d+1
	  etamin  =-5.0d+0
	  etamax  = 5.0d+0
	  smin    = 0.0d0
	  smax    = 5.0d+1
	  ymin    =-5.0d+0
	  ymax    = 5.0d+0
	  psetamin=-5.0d+0
	  psetamax= 5.0d+0
	  zmin    = 0.0d0
	  zmax    = 1.0d0
	else
	  ptmin   = 0.0d0
	  ptmax   = 7.0d+1
	  etamin  =-5.0d+0
	  etamax  = 5.0d+0
	  smin    = 0.0d0
	  smax    = 7.0d+1
	  ymin    =-5.0d+0
	  ymax    = 5.0d+0
	  psetamin=-5.0d+0
	  psetamax= 5.0d+0
	  zmin    = 0.0d0
	  zmax    = 1.0d0
	end if

	end
