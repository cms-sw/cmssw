      subroutine uppyfact(idwtup,generate,ievntdis)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

      common/pypars/mstp(200),parp(200),msti(200),pari(200)
	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/histcol/inx
      common/extraz/zfactor,zmin,zmax
      common/totcross/appcross
      logical unwght
      common/unweight/unwght
	common/mixevnt2/xbcsum,ibclimit

	logical generate

c...if not to generate the normalized event number distribution, then 
c...gets the value of the differential cross-sec of different 
c...parameters (the unit will be changed from mb to nb).
	if(idwtup.eq.1.and.generate) then
c...the unit of pari(2) is mb.
	  if(ievntdis.eq.0) then
	   call pyfact(1,1.0d+6*pari(2)*inx/(ptmax-ptmin))
         call pyfact(2,1.0d+6*pari(2)*inx/(etamax-etamin))
	   call pyfact(3,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(4,1.0d+6*pari(2)*inx/(ymax-ymin))
	   call pyfact(5,1.0d+6*pari(2)*inx/(psetamax-psetamin))
	   call pyfact(6,1.0d+6*pari(2)*inx/(zmax-zmin))
	   call pyfact(11,1.0d+6*pari(2)*inx/(etamax-etamin))
         call pyfact(12,1.0d+6*pari(2)*inx/(etamax-etamin))
	   call pyfact(13,1.0d+6*pari(2)*inx/(etamax-etamin))
	   call pyfact(14,1.0d+6*pari(2)*inx/(etamax-etamin))
	   call pyfact(15,1.0d+6*pari(2)*inx/(etamax-etamin))
	   call pyfact(21,1.0d+6*pari(2)*inx/(ptmax-ptmin))
         call pyfact(22,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	   call pyfact(23,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	   call pyfact(24,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	   call pyfact(25,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	   call pyfact(31,1.0d+6*pari(2)*inx/(smax-smin))
         call pyfact(32,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(33,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(34,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(35,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(41,1.0d+6*pari(2)*inx/(smax-smin))
         call pyfact(42,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(43,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(44,1.0d+6*pari(2)*inx/(smax-smin))
	   call pyfact(45,1.0d+6*pari(2)*inx/(smax-smin))
         call pyfact(52,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	   call pyfact(53,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	   call pyfact(54,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	   call pyfact(55,1.0d+6*pari(2)*inx/(ptmax-ptmin))
	  end if
	else
	  if(ievntdis.eq.0) then
	    if(unwght) then
         call pyfact(1,1.0d-3*xbcsum)
         call pyfact(2,1.0d-3*xbcsum)
         call pyfact(3,1.0d-3*xbcsum)
         call pyfact(4,1.0d-3*xbcsum)
         call pyfact(5,1.0d-3*xbcsum)
         call pyfact(6,1.0d-3*xbcsum)
         call pyfact(11,1.0d-3*xbcsum)
         call pyfact(12,1.0d-3*xbcsum)
         call pyfact(13,1.0d-3*xbcsum)
         call pyfact(14,1.0d-3*xbcsum)
         call pyfact(15,1.0d-3*xbcsum)
         call pyfact(21,1.0d-3*xbcsum)
         call pyfact(22,1.0d-3*xbcsum)
         call pyfact(23,1.0d-3*xbcsum)
         call pyfact(24,1.0d-3*xbcsum)
         call pyfact(25,1.0d-3*xbcsum)
         call pyfact(31,1.0d-3*xbcsum)
         call pyfact(32,1.0d-3*xbcsum)
         call pyfact(33,1.0d-3*xbcsum)
         call pyfact(34,1.0d-3*xbcsum)
         call pyfact(35,1.0d-3*xbcsum)
         call pyfact(41,1.0d-3*xbcsum)
         call pyfact(42,1.0d-3*xbcsum)
         call pyfact(43,1.0d-3*xbcsum)
         call pyfact(44,1.0d-3*xbcsum)
         call pyfact(45,1.0d-3*xbcsum)
         call pyfact(52,1.0d-3*xbcsum)
         call pyfact(53,1.0d-3*xbcsum)
         call pyfact(54,1.0d-3*xbcsum)
         call pyfact(55,1.0d-3*xbcsum)
	   else
         call pyfact(1,1.0d-3)
         call pyfact(2,1.0d-3)
         call pyfact(3,1.0d-3)
         call pyfact(4,1.0d-3)
         call pyfact(5,1.0d-3)
         call pyfact(6,1.0d-3)
         call pyfact(11,1.0d-3)
         call pyfact(12,1.0d-3)
         call pyfact(13,1.0d-3)
         call pyfact(14,1.0d-3)
         call pyfact(15,1.0d-3)
         call pyfact(21,1.0d-3)
         call pyfact(22,1.0d-3)
         call pyfact(23,1.0d-3)
         call pyfact(24,1.0d-3)
         call pyfact(25,1.0d-3)
         call pyfact(31,1.0d-3)
         call pyfact(32,1.0d-3)
         call pyfact(33,1.0d-3)
         call pyfact(34,1.0d-3)
         call pyfact(35,1.0d-3)
         call pyfact(41,1.0d-3)
         call pyfact(42,1.0d-3)
         call pyfact(43,1.0d-3)
         call pyfact(44,1.0d-3)
         call pyfact(45,1.0d-3)
         call pyfact(52,1.0d-3)
         call pyfact(53,1.0d-3)
         call pyfact(54,1.0d-3)
         call pyfact(55,1.0d-3)
      end if
	  end if
	  if(ievntdis.eq.1) then
	   call pyfact(1,1.0d0/(1.0d+3*appcross)*nev)
         call pyfact(2,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(3,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(4,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(5,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(6,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(11,1.0d0/(1.0d+3*appcross)*nev)
         call pyfact(12,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(13,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(14,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(15,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(21,1.0d0/(1.0d+3*appcross)*nev)
         call pyfact(22,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(23,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(24,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(25,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(31,1.0d0/(1.0d+3*appcross)*nev)
         call pyfact(32,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(33,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(34,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(35,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(41,1.0d0/(1.0d+3*appcross)*nev)
         call pyfact(42,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(43,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(44,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(45,1.0d0/(1.0d+3*appcross)*nev)
         call pyfact(52,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(53,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(54,1.0d0/(1.0d+3*appcross)*nev)
	   call pyfact(55,1.0d0/(1.0d+3*appcross)*nev)
	  end if
	end if

	end
