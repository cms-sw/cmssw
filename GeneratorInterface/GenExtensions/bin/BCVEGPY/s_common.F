c********************************************************************
c...dotup()-------dot product of two momentum.                       c
c...polsppup()----spinnor product involving bc* polarization vector. c
c...inpup()-------inner product of two massless partical.            c
c...sppup()-------spinor product of <p1+|\slash(p3)|p2+>.            c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c...       bc in color-singlet and color-octet states.               c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c...dotproduct of two momenta.
      double precision function dotup(ip,jp)
	implicit double precision (a-h,o-z)
	implicit integer (i-n)

	double complex colmat,bundamp
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      
      dotup=pmomup(4,ip)*pmomup(4,jp)-pmomup(1,ip)*pmomup(1,jp)-
     &	pmomup(2,ip)*pmomup(2,jp)-pmomup(3,ip)*pmomup(3,jp)
	
      return
      end

c******************************************************************

c...inner product involving bc polarization vector. ip stands one
c...of the three polarization vectors.
      double complex function polsppup(ip)
      implicit double precision (a-h,o-z)
	implicit integer (i-n)
	double complex colmat,bundamp,piv,pjv,ephase,pkv
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
      common/pol/polar(4,3)    
     
	pia=polar(4,ip)+polar(3,ip)
	pip=polar(4,ip)-polar(3,ip)
	piv=dcmplx(polar(1,ip),polar(2,ip))

	pja=pmomup(4,8)+pmomup(3,8)
	pjp=pmomup(4,8)-pmomup(3,8)
	pjv=dcmplx(pmomup(1,8),pmomup(2,8))

	pka=pmomzero(4,3)+pmomzero(3,3)
	pkp=pmomzero(4,3)-pmomzero(3,3)
	pkv=dcmplx(pmomzero(1,3),pmomzero(2,3))

      if (pja .gt. 1.0d-16 .and. pjp .gt. 1.0d-16) then
	   ephase=dconjg(pjv)/dsqrt(pja*pjp)
	else
	   ephase=dcmplx(1.0d0)
	end if

	polsppup=pka*pip*dconjg(pjv)-dconjg(pkv)*pja*pip-pka*piv*pjp
     &	*ephase**2+piv*dconjg(pkv)*dconjg(pjv)-pkv*dconjg(piv)*
     &    dconjg(pjv)+pja*dconjg(piv)*pkp+pia*pkv*pjp*ephase**2-
     &    pia*pkp*dconjg(pjv)

      return
	end

c*****************************************************************

c...inner scalar product of two momenta.
      double complex function inpup(ip,jp)
      implicit double precision (a-h,o-z)
	implicit integer (i-n)

	double complex colmat,piv,pjv,bundamp
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
            
c-ap initialize to avoid compiler warming:
        inpup = 0.
c-ap end init

	pia=pmomzero(4,ip)+pmomzero(3,ip)
	pip=pmomzero(4,ip)-pmomzero(3,ip)
	piv=dcmplx(pmomzero(1,ip),pmomzero(2,ip))

	pja=pmomzero(4,jp)+pmomzero(3,jp)
	pjp=pmomzero(4,jp)-pmomzero(3,jp)
	pjv=dcmplx(pmomzero(1,jp),pmomzero(2,jp))

      if (pia .gt. 1.0d-16 .and. pja .gt. 1.0d-16) then
	   inpup=piv*dsqrt(pja/pia)-pjv*dsqrt(pia/pja)
	end if
	if (pia .lt. 1.0d-16 .and. pja .gt. 1.0d-16) then
	   inpup=dsqrt(pip*pja)
      end if
      if (pia .gt. 1.0d-16 .and. pja .lt. 1.0d-16) then
	   inpup=-dsqrt(pia*pjp)
      end if
	if (pia .lt. 1.0d-16 .and. pja .lt. 1.0d-16) then
	   inpup=0.0d0
      end if

      return
      end

c******************************************************************

c...spinor product <p1+|\slash(p3)|p2+>, all momenta are light-like.
      double complex function sppup(ip,kp,jp)
	implicit double precision (a-h,o-z)
	implicit integer(i-n)
	double complex colmat,inpup,bundamp
	common/upcom/ecm,pmbc,pmb,pmc,fbcc,pmomup(5,8),
     & 	colmat(10,64),bundamp(4),pmomzero(5,8)
	      
      sppup=dconjg(inpup(kp,ip))*inpup(kp,jp)
      
      return
      end
      
