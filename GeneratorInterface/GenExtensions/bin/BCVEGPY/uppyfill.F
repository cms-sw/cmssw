
c...pyfill all the data.
      subroutine uppyfill(idwtup,generate,xwgtup,pt,eta,st,y,pseta)
      implicit double precision(a-h, o-z)
	implicit integer(i-n)

	common/ptpass/ptmin,ptmax,crossmax,etamin,etamax,
     &	smin,smax,ymin,ymax,psetamin,psetamax
	common/histcol/inx
	common/counter/ibcstate,nev
      common/extraz/zfactor,zmin,zmax
      logical generate

c      write (91,'(d20.10)') pseta
	
		 if(idwtup.eq.1.and.generate) then
			    call pyfill(1,pt,1.0d0)
	            call pyfill(2,eta,1.0d0)
	            call pyfill(3,st,1.0d0)
	            call pyfill(4,y,1.0d0)
			    call pyfill(5,pseta,1.0d0)
				call pyfill(6,zfactor,1.0d0)
c...added output files. record differential distribution of rapidity
c...with certain pt-cut.
                  if(pt .gt. 5.0d+0) then
				    call pyfill(11,eta,1.0d0)
					call pyfill(41,st,1.0d0)
	            end if
				if(pt .gt. 2.0d+1) then
				    call pyfill(12,eta,1.0d0)
					call pyfill(42,st,1.0d0)
	            end if
				if(pt .gt. 3.5d+1) then
				    call pyfill(13,eta,1.0d0)
					call pyfill(43,st,1.0d0)
				end if
				if(pt .gt. 5.0d+1) then
				    call pyfill(14,eta,1.0d0)
					call pyfill(44,st,1.0d0)
				end if
				if(pt .gt. 1.0d+2) then
				    call pyfill(15,eta,1.0d0)
					call pyfill(45,st,1.0d0)
	            end if
c...added output files. record differential distribution of pt
c...and shat with certain rapidity-cut.
                  if(dabs(eta) .le. 5.0d-1) then
				    call pyfill(21,pt,1.0d0)
	                call pyfill(31,st,1.0d0)
	            end if
	            if(dabs(eta) .le. 1.0d+0) then
				    call pyfill(22,pt,1.0d0)
	                call pyfill(32,st,1.0d0)
	            end if
	            if(dabs(eta) .le. 1.5d+0) then
				    call pyfill(23,pt,1.0d0)
	                call pyfill(33,st,1.0d0)
	            end if
				if(dabs(eta) .le. 2.0d+0) then
				    call pyfill(24,pt,1.0d0)
	                call pyfill(34,st,1.0d0)
	            end if
				if(dabs(eta) .le. 2.5d+0) then
				    call pyfill(25,pt,1.0d0)
	                call pyfill(35,st,1.0d0)
	            end if
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
				if(dabs(eta).ge.5.0d-1 .and. dabs(eta).le.1.0d+0) then
				    call pyfill(52,pt,1.0d0)
	            end if
				if(dabs(eta).ge.1.0d+0 .and. dabs(eta).le.1.5d+0) then
				    call pyfill(53,pt,1.0d0)
	            end if
				if(dabs(eta).ge.1.5d+0 .and. dabs(eta).le.2.0d+0) then
				    call pyfill(54,pt,1.0d0)
	            end if
				if(dabs(eta).ge.2.0d+0 .and. dabs(eta).le.2.5d+0) then
				    call pyfill(55,pt,1.0d0)
	            end if
	      else
				call pyfill(1,pt,xwgtup/nev*inx/(ptmax-ptmin))
	            call pyfill(2,eta,xwgtup/nev*inx/(etamax-etamin))
	            call pyfill(3,st,xwgtup/nev*inx/(smax-smin))
	            call pyfill(4,y,xwgtup/nev*inx/(ymax-ymin))
			    call pyfill(5,pseta,xwgtup/nev*inx/(psetamax-psetamin))
				call pyfill(6,zfactor,xwgtup/nev*inx/(zmax-zmin))
c...added output files. record differential distribution of rapidity
c...with certain pt-cut.
                  if(pt .ge. 5.0d+0) then
      			    call pyfill(11,eta,xwgtup/nev*inx/(etamax-etamin))
					call pyfill(41,st,xwgtup/nev*inx/(smax-smin))
	            end if
				if(pt .ge. 2.0d+1) then
          		    call pyfill(12,eta,xwgtup/nev*inx/(etamax-etamin))
					call pyfill(42,st,xwgtup/nev*inx/(smax-smin))
	            end if
				if(pt .ge. 3.5d+1) then
      			    call pyfill(13,eta,xwgtup/nev*inx/(etamax-etamin))
				    call pyfill(43,st,xwgtup/nev*inx/(smax-smin))
				end if
				if(pt .ge. 5.0d+1) then
      			    call pyfill(14,eta,xwgtup/nev*inx/(etamax-etamin))
				    call pyfill(44,st,xwgtup/nev*inx/(smax-smin))
				end if
				if(pt .ge. 1.0d+2) then
      			    call pyfill(15,eta,xwgtup/nev*inx/(etamax-etamin))
	                call pyfill(45,st,xwgtup/nev*inx/(smax-smin))
	            end if
c...added output files. record differential distribution of pt
c...and shat with certain rapidity-cut.
                  if(dabs(eta) .le. 5.0d-1) then
				    call pyfill(21,pt,xwgtup/nev*inx/(ptmax-ptmin))
	                call pyfill(31,st,xwgtup/nev*inx/(smax-smin))
	            end if
	            if(dabs(eta) .le. 1.0d+0) then
				    call pyfill(22,pt,xwgtup/nev*inx/(ptmax-ptmin))
	                call pyfill(32,st,xwgtup/nev*inx/(smax-smin))
	            end if
	            if(dabs(eta) .le. 1.5d+0) then
				    call pyfill(23,pt,xwgtup/nev*inx/(ptmax-ptmin))
	                call pyfill(33,st,xwgtup/nev*inx/(smax-smin))
	            end if
				if(dabs(eta) .le. 2.0d+0) then
				    call pyfill(24,pt,xwgtup/nev*inx/(ptmax-ptmin))
	                call pyfill(34,st,xwgtup/nev*inx/(smax-smin))
	            end if
				if(dabs(eta) .le. 2.5d+0) then
				    call pyfill(25,pt,xwgtup/nev*inx/(ptmax-ptmin))
	                call pyfill(35,st,xwgtup/nev*inx/(smax-smin))
	            end if
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
				if(dabs(eta).ge.5.0d-1 .and. dabs(eta).le.1.0d+0) then
				    call pyfill(52,pt,xwgtup/nev*inx/(ptmax-ptmin))
	            end if
				if(dabs(eta).ge.1.0d+0 .and. dabs(eta).le.1.5d+0) then
				    call pyfill(53,pt,xwgtup/nev*inx/(ptmax-ptmin))
	            end if
				if(dabs(eta).ge.1.5d+0 .and. dabs(eta).le.2.0d+0) then
				    call pyfill(54,pt,xwgtup/nev*inx/(ptmax-ptmin))
	            end if
				if(dabs(eta).ge.2.0d+0 .and. dabs(eta).le.2.5d+0) then
				    call pyfill(55,pt,xwgtup/nev*inx/(ptmax-ptmin))
	            end if
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	      end if

      end
