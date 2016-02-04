	subroutine upclosegradefile(iveggrade,imix,imixtype)
      implicit double precision(a-h, o-z)
	implicit integer(i-n) 

	close(11)
	if(iveggrade.eq.1) close(29)

      if(imix.eq.1) then
	 if(imixtype.eq.1) then
	  close(36)
	  close(37)
	  close(38)
	  close(39)
        close(46)
	  close(47)
	  close(48)
	  close(49)
	 end if
	 if(imixtype.eq.2) then
	  close(36)
	  close(37)
	 end if
	 if(imixtype.eq.3) then
	  close(38)
	  close(39)
        close(46)
	  close(47)
	  close(48)
	  close(49)
	 end if
	end if

	end
	
