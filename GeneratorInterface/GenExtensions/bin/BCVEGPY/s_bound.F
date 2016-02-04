
c********************************************************************

c...helicity amplitude of the bound state part: \bar{b}+c->bc.
      subroutine bundhelicity(ibc)
      implicit double precision (a-h,o-z)
	implicit integer (i-n)
	double complex colmat,bundamp,polsppup,inpup
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/pol/polar(4,3)
c...for color-octet production.
      common/coloct/ioctet
	common/octmatrix/coeoct

c...wavezero is the value of the wave function at origin.
      wavezero=dsqrt(pmbc*fbcc**2/12.0d0)

c...color-octet matrixment.
	if(ioctet.eq.1) then
	  wavezero=coeoct*wavezero
	end if

c...bundamp(1)->++, bundamp(2)->--, bundamp(3)->+-, bundamp(4)->-+.
c...b_c in 1s0 state and not get the combined results for bc and bc*
      if(ibc.eq.1) then
	  bundamp(1)=dcmplx(-wavezero*dsqrt(pmbc)/(2*dsqrt(pmb*pmc)))
	  bundamp(2)=dcmplx(+wavezero*dsqrt(pmbc)/(2*dsqrt(pmb*pmc)))
	  bundamp(3)=dcmplx(0.0d0)
	  bundamp(4)=dcmplx(0.0d0)
	end if

c...the expression of polarization vectors depend on the gauge choice.
c...one way of constructing polarization vector of 3s1 state: all
c...satisfies: polar(i)**2=-1, polar(i).polar(j)=0 (i.ne.j),
c...polar(i).pmomup(3)=0; in addition, we choose: polar(1,1)=0.0d0
c...polar(2,1)=0.0d0, and polar(1,3)=0.0d0.
      if(ibc.eq.2) then
	 polar(1,1)=0.0d0
	 polar(2,1)=0.0d0
	 polar(3,1)=pmomup(4,3)/dsqrt(pmomup(4,3)**2-pmomup(3,3)**2)
	 polar(4,1)=pmomup(3,3)/dsqrt(pmomup(4,3)**2-pmomup(3,3)**2)

       if(pmomup(3,3).lt.0.0d0) then
	  polar(1,2)=-dsqrt(pmomup(4,3)**2-pmomup(2,3)**2-pmomup(3,3)**2)
     &	   /pmomup(5,3)
	  polar(2,2)=-pmomup(1,3)*pmomup(2,3)/dsqrt(pmomup(4,3)**2
     &	  -pmomup(2,3)**2-pmomup(3,3)**2)/pmomup(5,3)
	  polar(3,2)=-pmomup(1,3)*pmomup(3,3)/dsqrt(pmomup(4,3)**2
     &	  -pmomup(2,3)**2-pmomup(3,3)**2)/pmomup(5,3)
	  polar(4,2)=-pmomup(1,3)*pmomup(4,3)/dsqrt(pmomup(4,3)**2
     &	  -pmomup(2,3)**2-pmomup(3,3)**2)/pmomup(5,3)
	 else
	  polar(1,2)=dsqrt(pmomup(4,3)**2-pmomup(2,3)**2-pmomup(3,3)**2)
     &	   /pmomup(5,3)
	  polar(2,2)=pmomup(1,3)*pmomup(2,3)/dsqrt(pmomup(4,3)**2
     &	  -pmomup(2,3)**2-pmomup(3,3)**2)/pmomup(5,3)
	  polar(3,2)=pmomup(1,3)*pmomup(3,3)/dsqrt(pmomup(4,3)**2
     &	  -pmomup(2,3)**2-pmomup(3,3)**2)/pmomup(5,3)
	  polar(4,2)=pmomup(1,3)*pmomup(4,3)/dsqrt(pmomup(4,3)**2
     &	  -pmomup(2,3)**2-pmomup(3,3)**2)/pmomup(5,3)
	 end if

	 polar(1,3)=0.0d0
	 polar(2,3)=dsqrt(pmomup(4,3)**2-pmomup(3,3)**2)
     &   /dsqrt(pmomup(4,3)**2-pmomup(2,3)**2-pmomup(3,3)**2)
	 polar(3,3)=pmomup(2,3)*pmomup(3,3)/dsqrt(pmomup(4,3)**2-
     &	  pmomup(3,3)**2)/dsqrt(pmomup(4,3)**2-pmomup(2,3)**2
     &	  -pmomup(3,3)**2)
	 polar(4,3)=pmomup(2,3)*pmomup(4,3)/dsqrt(pmomup(4,3)**2-
     &	  pmomup(3,3)**2)/dsqrt(pmomup(4,3)**2-pmomup(2,3)**2
     &	  -pmomup(3,3)**2)

c...four bound state matrix elements of 3s1.
	 bundamp(1)=dcmplx(0.0d0)
	 do i=1,3
	  bundamp(1)=bundamp(1)+wavezero*dsqrt(pmbc)/(2*dsqrt(pmb*pmc))
     &	*pmbc*(polar(4,i)*pmomup(4,8)-polar(1,i)*pmomup(1,8)-
     &    polar(2,i)*pmomup(2,8)-polar(3,i)*pmomup(3,8))/dotup(3,8)
	 end do
	 bundamp(2)=dconjg(bundamp(1))
       bundamp(3)=dcmplx(0.0d0)
	 do i=1,3
        bundamp(3)=bundamp(3)+wavezero*dsqrt(pmbc)/(2*dsqrt(pmb*pmc))
     &	*polsppup(i)/(2*dotup(3,8))
       end do
   	 bundamp(4)=-dconjg(bundamp(3))
	
	end if
      
	return
	end
	
