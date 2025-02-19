	subroutine uppydump
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

	dimension ihi(10)
	
	ihi(1)=1
	call pydump(3,4,1,ihi)
      ihi(1)=2
	call pydump(3,5,1,ihi)
	ihi(1)=3
	call pydump(3,16,1,ihi)
	ihi(1)=4
	call pydump(3,7,1,ihi)
	ihi(1)=5
	call pydump(3,8,1,ihi)
	ihi(1)=6
	call pydump(3,9,1,ihi)

	ihi(1)=11
	call pydump(3,21,1,ihi)
	ihi(1)=12
	call pydump(3,22,1,ihi)
	ihi(1)=13
	call pydump(3,23,1,ihi)
	ihi(1)=14
	call pydump(3,24,1,ihi)
	ihi(1)=15
	call pydump(3,25,1,ihi)

	ihi(1)=21
	call pydump(3,31,1,ihi)
	ihi(1)=22
	call pydump(3,32,1,ihi)
	ihi(1)=23
	call pydump(3,33,1,ihi)
	ihi(1)=24
	call pydump(3,34,1,ihi)
	ihi(1)=25
	call pydump(3,35,1,ihi)

	ihi(1)=31
	call pydump(3,41,1,ihi)
	ihi(1)=32
	call pydump(3,42,1,ihi)
	ihi(1)=33
	call pydump(3,43,1,ihi)
	ihi(1)=34
	call pydump(3,44,1,ihi)
	ihi(1)=35
	call pydump(3,45,1,ihi)

	ihi(1)=41
	call pydump(3,51,1,ihi)
	ihi(1)=42
	call pydump(3,52,1,ihi)
	ihi(1)=43
	call pydump(3,53,1,ihi)
	ihi(1)=44
	call pydump(3,54,1,ihi)
	ihi(1)=45
	call pydump(3,55,1,ihi)

	ihi(1)=52
	call pydump(3,62,1,ihi)
	ihi(1)=53
	call pydump(3,63,1,ihi)
	ihi(1)=54
	call pydump(3,64,1,ihi)
	ihi(1)=55
	call pydump(3,65,1,ihi)

	end
	
