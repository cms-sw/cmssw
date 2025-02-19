
      subroutine uperror(wronginput)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)
      common/counter/ibcstate,nev
      common/qqbar/iqqbar,iqcode
      common/coloct/ioctet
	logical wronginput

c...error message.
      if((ibcstate.gt.8).or.(ibcstate.lt.1)) then
      write(*,*) 'only eight s and p-wave states have been programmed!'
	 wronginput=.true.
	endif

	if((ibcstate.gt.2).and.(iqqbar.eq.1)) then
      write(*,*) 'p-wave states for q-\bar{q} has not been programmed!'
	 wronginput=.true.
	endif

	if((iqqbar.eq.1) .and. (ioctet.eq.1)) then
	 write(*,'(a)')
     &  'the color-octet contribution from quark-antiquark annihilation'
       write(*,'(a)')
     &  'is small and has not been programmed for bc production !!!!!!!'
	 wronginput=.true.
	end if

	if(((ibcstate.eq.3).or.(ibcstate.eq.4).or.(ibcstate.eq.5).or.
     &	(ibcstate.eq.6)) .and. (ioctet.eq.1)) then
	 write(*,'(a)')
     &  'the color-octet contribution from p-wave states'
       write(*,'(a)')
     &  'is small and has not been programmed for bc production !!!!!!!'
	 wronginput=.true.
	end if

	end
